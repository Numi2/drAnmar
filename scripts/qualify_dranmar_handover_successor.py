#!/usr/bin/env python3
"""Fail-closed qualification for one standalone learned handover actor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


QUALIFICATION_SEEDS = (17, 2361, 4099)
# Frozen before candidate-2 evaluation and disjoint from demonstrations,
# follow-up mining, both smoke cohorts, and the sealed qualification seeds.
DEVELOPMENT_SEEDS = (16000057, 16065541, 16131091)
HARD_ZERO_TERMS = (
    "excessive_object_force",
    "protected_surface_force",
)
STATISTICAL_NONINCREASE_ENDPOINTS = {
    "drop": (
        "needle_dropped_after_pickup",
        "object_dropping",
    ),
    "retention": ("receiver_retention_lost",),
    "premature_release": ("premature_giver_release",),
}
STATISTICAL_ALPHA = 0.05
PER_SEED_SUCCESS_MARGIN = 0.02


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_path(value: str) -> tuple[int, Path]:
    encoded_seed, separator, encoded_path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError(
            "expected SEED=/path/to/evidence.json"
        )
    try:
        seed = int(encoded_seed)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "qualification seed must be an integer"
        ) from error
    path = Path(encoded_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(
            f"qualification evidence does not exist: {path}"
        )
    return seed, path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qualify one standalone recurrent handover successor"
    )
    parser.add_argument(
        "--incumbent",
        action="append",
        type=_seed_path,
        required=True,
        help="one fresh incumbent result per seed as SEED=PATH",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        type=_seed_path,
        required=True,
        help="two frozen learned-policy results per seed as SEED=PATH",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--jit", required=True)
    parser.add_argument(
        "--profile",
        choices=("development", "qualification"),
        default="qualification",
    )
    parser.add_argument("--output", required=True)
    return parser


def _load(seed: int, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("kind") != "held_out_play":
        raise ValueError(f"not held-out play evidence: {path}")
    if int(payload.get("seed", -1)) != seed:
        raise ValueError(f"seed label does not match evidence: {path}")
    payload["_source"] = {
        "path": str(path),
        "sha256": _sha256(path),
    }
    return payload


def _indices(payload: dict[str, Any], name: str) -> set[int]:
    outcomes = payload.get("environment_outcomes")
    if not isinstance(outcomes, dict) or name not in outcomes:
        raise ValueError(
            "qualification evidence lacks paired environment outcomes: "
            f"{payload['_source']['path']}"
        )
    return {int(value) for value in outcomes[name]}


def _counts(payload: dict[str, Any]) -> dict[str, int]:
    return {
        "episodes": int(payload["num_envs"]),
        "success": len(_indices(payload, "successful_indices")),
        "lift": len(_indices(payload, "lifted_10mm_indices")),
        "acquisition": len(
            _indices(payload, "receiver_acquired_indices")
        ),
    }


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _gate(identifier: str, passed: bool, **evidence: Any) -> dict[str, Any]:
    return {
        "id": identifier,
        "passed": bool(passed),
        **evidence,
    }


def _endpoint_count(
    payload: dict[str, Any],
    terms: tuple[str, ...],
) -> int:
    termination_counts = payload["termination_term_counts"]
    observed = sum(
        int(termination_counts.get(term, 0))
        for term in terms
    )
    episodes = int(payload["num_envs"])
    if not 0 <= observed <= episodes:
        raise ValueError(
            "qualification endpoint count exceeds its episode count"
        )
    return observed


def _cmh_one_sided_increase_p_value(
    strata: list[tuple[int, int, int, int]],
) -> float:
    """Test whether candidate event odds increased, stratified by seed."""

    numerator = 0.0
    variance = 0.0
    for (
        candidate_events,
        candidate_episodes,
        incumbent_events,
        incumbent_episodes,
    ) in strata:
        if (
            candidate_episodes <= 0
            or incumbent_episodes <= 0
            or not 0 <= candidate_events <= candidate_episodes
            or not 0 <= incumbent_events <= incumbent_episodes
        ):
            raise ValueError("invalid adverse-event comparison stratum")
        total_events = candidate_events + incumbent_events
        total_episodes = candidate_episodes + incumbent_episodes
        expected_candidate = (
            total_events * candidate_episodes / total_episodes
        )
        numerator += candidate_events - expected_candidate
        if total_episodes > 1:
            variance += (
                candidate_episodes
                * incumbent_episodes
                * total_events
                * (total_episodes - total_events)
                / (
                    total_episodes
                    * total_episodes
                    * (total_episodes - 1)
                )
            )
    if variance <= 0.0:
        return 1.0
    continuity_corrected = max(0.0, numerator - 0.5)
    z_score = continuity_corrected / math.sqrt(variance)
    return 0.5 * math.erfc(z_score / math.sqrt(2.0))


def _holm_adjusted_p_values(
    p_values: dict[str, float],
) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running_maximum = 0.0
    hypothesis_count = len(ordered)
    for rank, identifier in enumerate(ordered):
        raw = p_values[identifier]
        running_maximum = max(
            running_maximum,
            (hypothesis_count - rank) * raw,
        )
        adjusted[identifier] = min(1.0, running_maximum)
    return adjusted


def qualify(
    incumbents: dict[int, dict[str, Any]],
    candidates: dict[int, list[dict[str, Any]]],
    *,
    checkpoint_sha256: str,
    jit_sha256: str,
    seeds: tuple[int, ...] = QUALIFICATION_SEEDS,
    expected_num_envs: int = 1200,
    profile: str = "qualification",
) -> dict[str, Any]:
    gates: list[dict[str, Any]] = []
    aggregate = defaultdict(int)
    aggregate_incumbent = defaultdict(int)
    per_seed: dict[str, dict[str, Any]] = {}
    candidate_hashes: set[str] = set()
    controller_revisions: set[str] = set()
    asset_revisions: set[str] = set()
    statistical_strata: dict[
        str,
        list[tuple[int, int, int, int]],
    ] = defaultdict(list)

    for seed in seeds:
        incumbent = incumbents[seed]
        incumbent_counts = _counts(incumbent)
        for name, value in incumbent_counts.items():
            aggregate_incumbent[name] += value
        seed_counts = defaultdict(int)
        for role, payload in (
            ("incumbent", incumbent),
            *(
                (f"candidate_{index + 1}", run)
                for index, run in enumerate(candidates[seed])
            ),
        ):
            shape_passed = (
                int(payload.get("num_envs", -1)) == expected_num_envs
                and int(payload.get("completed_episodes", -1))
                == expected_num_envs
                and int(payload.get("frames_per_env", -1)) == 2000
                and float(payload.get("episode_length_s", -1.0)) == 40.0
                and float(
                    payload.get(
                        "reset_rotation_randomization_deg",
                        0.0,
                    )
                )
                == 0.0
                and bool(
                    payload.get(
                        "first_terminal_outcome_per_environment"
                    )
                )
            )
            gates.append(
                _gate(
                    f"seed_{seed}_{role}_qualification_shape",
                    shape_passed,
                )
            )

        for run_index, candidate in enumerate(
            candidates[seed],
            start=1,
        ):
            run_counts = _counts(candidate)
            for name, value in run_counts.items():
                seed_counts[name] += value
                aggregate[name] += value
            successor = candidate.get("handover_successor")
            standalone = (
                isinstance(successor, dict)
                and successor.get("enabled") is True
                and successor.get("runtime_heuristic_stack") is False
                and candidate.get("pickup_recovery") == {"enabled": False}
                and candidate.get("receiver_recovery") == {"enabled": False}
                and candidate.get("checkpoint") is None
            )
            gates.append(
                _gate(
                    f"seed_{seed}_run_{run_index}_standalone_actor",
                    standalone,
                )
            )
            if isinstance(successor, dict):
                candidate_hashes.add(str(successor.get("sha256", "")))
            source = candidate.get("runtime", {}).get("source", {})
            controller_revisions.add(
                str(source.get("dranmar_revision", ""))
            )
            asset_revisions.add(str(source.get("asset_revision", "")))

            candidate_terms = candidate["termination_term_counts"]
            for term in HARD_ZERO_TERMS:
                observed = int(candidate_terms.get(term, 0))
                gates.append(
                    _gate(
                        f"seed_{seed}_run_{run_index}_{term}_zero",
                        observed == 0,
                        observed=observed,
                    )
                )

        metrics = {
            **seed_counts,
            "success_rate": _rate(
                seed_counts["success"],
                seed_counts["episodes"],
            ),
            "lift_rate": _rate(
                seed_counts["lift"],
                seed_counts["episodes"],
            ),
            "receiver_acquisition_given_lift": _rate(
                seed_counts["acquisition"],
                seed_counts["lift"],
            ),
            "retention_given_acquisition": _rate(
                seed_counts["success"],
                seed_counts["acquisition"],
            ),
        }
        incumbent_success_rate = _rate(
            incumbent_counts["success"],
            incumbent_counts["episodes"],
        )
        metrics["incumbent_success_rate"] = incumbent_success_rate
        metrics["success_rate_difference_vs_incumbent"] = (
            metrics["success_rate"] - incumbent_success_rate
        )
        per_seed[str(seed)] = metrics
        gates.append(
            _gate(
                f"seed_{seed}_success_at_least_78_percent",
                metrics["success_rate"] >= 0.78,
                observed=metrics["success_rate"],
            )
        )
        if profile == "qualification":
            gates.append(
                _gate(
                    (
                        f"seed_{seed}_success_noninferior_to_"
                        "incumbent_with_2pp_margin"
                    ),
                    metrics["success_rate_difference_vs_incumbent"]
                    >= -PER_SEED_SUCCESS_MARGIN - 1.0e-12,
                    candidate=metrics["success_rate"],
                    incumbent=incumbent_success_rate,
                    difference=(
                        metrics[
                            "success_rate_difference_vs_incumbent"
                        ]
                    ),
                    margin=PER_SEED_SUCCESS_MARGIN,
                )
            )
            candidate_episodes = seed_counts["episodes"]
            for endpoint, terms in (
                STATISTICAL_NONINCREASE_ENDPOINTS.items()
            ):
                candidate_events = sum(
                    _endpoint_count(run, terms)
                    for run in candidates[seed]
                )
                incumbent_events = _endpoint_count(
                    incumbent,
                    terms,
                )
                statistical_strata[endpoint].append(
                    (
                        candidate_events,
                        candidate_episodes,
                        incumbent_events,
                        incumbent_counts["episodes"],
                    )
                )
        gates.append(
            _gate(
                f"seed_{seed}_lift_at_least_92_percent",
                metrics["lift_rate"] >= 0.92,
                observed=metrics["lift_rate"],
            )
        )

    aggregate_incumbent_metrics = {
        **aggregate_incumbent,
        "success_rate": _rate(
            aggregate_incumbent["success"],
            aggregate_incumbent["episodes"],
        ),
    }
    aggregate_metrics = {
        **aggregate,
        "success_rate": _rate(
            aggregate["success"],
            aggregate["episodes"],
        ),
        "lift_rate": _rate(
            aggregate["lift"],
            aggregate["episodes"],
        ),
        "receiver_acquisition_given_lift": _rate(
            aggregate["acquisition"],
            aggregate["lift"],
        ),
        "retention_given_acquisition": _rate(
            aggregate["success"],
            aggregate["acquisition"],
        ),
    }
    aggregate_metrics["incumbent_success_rate"] = (
        aggregate_incumbent_metrics["success_rate"]
    )
    aggregate_metrics["success_rate_difference_vs_incumbent"] = (
        aggregate_metrics["success_rate"]
        - aggregate_incumbent_metrics["success_rate"]
    )
    for identifier, observed, threshold in (
        (
            "aggregate_success_at_least_80_percent",
            aggregate_metrics["success_rate"],
            0.80,
        ),
        (
            "aggregate_lift_at_least_94_percent",
            aggregate_metrics["lift_rate"],
            0.94,
        ),
        (
            "aggregate_receiver_acquisition_given_lift_at_least_88_percent",
            aggregate_metrics["receiver_acquisition_given_lift"],
            0.88,
        ),
        (
            "aggregate_retention_given_acquisition_at_least_97_percent",
            aggregate_metrics["retention_given_acquisition"],
            0.97,
        ),
    ):
        gates.append(
            _gate(
                identifier,
                observed >= threshold,
                observed=observed,
                threshold=threshold,
            )
        )
    if profile == "qualification":
        gates.append(
            _gate(
                "aggregate_success_strictly_exceeds_incumbent",
                (
                    aggregate_metrics[
                        "success_rate_difference_vs_incumbent"
                    ]
                    > 0.0
                ),
                candidate=aggregate_metrics["success_rate"],
                incumbent=aggregate_incumbent_metrics["success_rate"],
                difference=(
                    aggregate_metrics[
                        "success_rate_difference_vs_incumbent"
                    ]
                ),
            )
        )
        raw_p_values = {
            endpoint: _cmh_one_sided_increase_p_value(strata)
            for endpoint, strata in statistical_strata.items()
        }
        adjusted_p_values = _holm_adjusted_p_values(raw_p_values)
        for endpoint in STATISTICAL_NONINCREASE_ENDPOINTS:
            strata = statistical_strata[endpoint]
            candidate_events = sum(value[0] for value in strata)
            candidate_episodes = sum(value[1] for value in strata)
            incumbent_events = sum(value[2] for value in strata)
            incumbent_episodes = sum(value[3] for value in strata)
            gates.append(
                _gate(
                    (
                        f"aggregate_{endpoint}_rate_has_no_"
                        "statistically_supported_increase"
                    ),
                    adjusted_p_values[endpoint] >= STATISTICAL_ALPHA,
                    candidate_events=candidate_events,
                    candidate_episodes=candidate_episodes,
                    candidate_rate=_rate(
                        candidate_events,
                        candidate_episodes,
                    ),
                    incumbent_events=incumbent_events,
                    incumbent_episodes=incumbent_episodes,
                    incumbent_rate=_rate(
                        incumbent_events,
                        incumbent_episodes,
                    ),
                    raw_one_sided_p_value=raw_p_values[endpoint],
                    holm_adjusted_p_value=adjusted_p_values[endpoint],
                    alpha=STATISTICAL_ALPHA,
                    method=(
                        "one_sided_cmh_candidate_increase_"
                        "stratified_by_seed_with_holm_correction"
                    ),
                )
            )

    gates.extend(
        (
            _gate(
                "single_frozen_successor_checkpoint",
                candidate_hashes == {checkpoint_sha256},
                expected=checkpoint_sha256,
                observed=sorted(candidate_hashes),
            ),
            _gate(
                "single_source_and_asset_lock",
                len(controller_revisions) == 1
                and "" not in controller_revisions
                and len(asset_revisions) == 1
                and "" not in asset_revisions,
                controller_revisions=sorted(controller_revisions),
                asset_revisions=sorted(asset_revisions),
            ),
            _gate(
                "standalone_jit_frozen",
                len(jit_sha256) == 64,
                jit_sha256=jit_sha256,
            ),
        )
    )
    return {
        "schema_version": (
            "dranmar-handover-successor-qualification-1.0"
        ),
        "profile": profile,
        "qualified": all(gate["passed"] for gate in gates),
        "checkpoint_sha256": checkpoint_sha256,
        "jit_sha256": jit_sha256,
        "aggregate": aggregate_metrics,
        "aggregate_incumbent": aggregate_incumbent_metrics,
        "per_seed": per_seed,
        "gates": gates,
        "inputs": {
            "incumbents": {
                str(seed): incumbents[seed]["_source"]
                for seed in seeds
            },
            "candidates": {
                str(seed): [
                    run["_source"] for run in candidates[seed]
                ]
                for seed in seeds
            },
        },
    }


def main(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    jit = Path(args.jit).expanduser().resolve()
    if not checkpoint.is_file() or not jit.is_file():
        raise ValueError("checkpoint and JIT artifacts must exist")
    seeds = (
        DEVELOPMENT_SEEDS
        if args.profile == "development"
        else QUALIFICATION_SEEDS
    )
    expected_num_envs = 600 if args.profile == "development" else 1200
    expected_candidate_runs = 1 if args.profile == "development" else 2
    incumbent_paths: dict[int, list[Path]] = defaultdict(list)
    candidate_paths: dict[int, list[Path]] = defaultdict(list)
    for seed, path in args.incumbent:
        incumbent_paths[seed].append(path)
    for seed, path in args.candidate:
        candidate_paths[seed].append(path)
    if set(incumbent_paths) != set(seeds):
        raise ValueError(
            f"incumbent evidence must cover {list(seeds)}"
        )
    if set(candidate_paths) != set(seeds):
        raise ValueError(
            f"candidate evidence must cover {list(seeds)}"
        )
    for seed in seeds:
        if len(incumbent_paths[seed]) != 1:
            raise ValueError(
                f"seed {seed} requires exactly one incumbent run"
            )
        if len(candidate_paths[seed]) != expected_candidate_runs:
            raise ValueError(
                f"seed {seed} requires exactly "
                f"{expected_candidate_runs} candidate runs"
            )
    incumbents = {
        seed: _load(seed, incumbent_paths[seed][0])
        for seed in seeds
    }
    candidates = {
        seed: [_load(seed, path) for path in candidate_paths[seed]]
        for seed in seeds
    }
    report = qualify(
        incumbents,
        candidates,
        checkpoint_sha256=_sha256(checkpoint),
        jit_sha256=_sha256(jit),
        seeds=seeds,
        expected_num_envs=expected_num_envs,
        profile=args.profile,
    )
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise ValueError(
            f"refusing to overwrite qualification report: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
