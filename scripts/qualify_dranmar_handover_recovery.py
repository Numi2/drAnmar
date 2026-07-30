#!/usr/bin/env python3
"""Fail-closed promotion gate for the frozen 40-second recovery composite."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


QUALIFICATION_SEEDS = (17, 2361, 4099)
IMMUTABLE_BASE_SHA256 = (
    "f33e41883f80f4dd791d0033568a4241bf366adcf2eb739c20c9ffd9ab568aad"
)
ZERO_TERMS = ("excessive_object_force", "object_dropping")
NONINCREASE_TERMS = (
    "needle_dropped_after_pickup",
    "receiver_retention_lost",
    "protected_surface_force",
    "premature_giver_release",
)
ALLOWED_CORRECTION_CAPS = {
    (0.0025, 2.0),
    (0.004, 4.0),
    (0.005, 5.0),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_path(value: str) -> tuple[int, Path]:
    seed_text, separator, path_text = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("expected SEED=/path/to/evidence.json")
    try:
        seed = int(seed_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("seed must be an integer") from error
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"evidence does not exist: {path}")
    return seed, path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate all frozen handover-recovery promotion gates"
    )
    parser.add_argument(
        "--incumbent",
        action="append",
        type=_seed_path,
        required=True,
        help="one paired 40-second incumbent result as SEED=PATH",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        type=_seed_path,
        required=True,
        help="two frozen candidate results per seed as SEED=PATH",
    )
    parser.add_argument("--output", required=True)
    return parser


def _load(seed: int, path: Path) -> dict[str, Any]:
    evidence = json.loads(path.read_text())
    if evidence.get("kind") != "held_out_play":
        raise ValueError(f"not held-out play evidence: {path}")
    if int(evidence.get("seed", -1)) != seed:
        raise ValueError(f"seed label does not match evidence: {path}")
    evidence["_source"] = {
        "path": str(path),
        "sha256": _sha256(path),
    }
    return evidence


def _indices(evidence: dict[str, Any], name: str) -> set[int]:
    outcomes = evidence.get("environment_outcomes")
    if not isinstance(outcomes, dict) or name not in outcomes:
        raise ValueError(
            f"evidence lacks paired environment outcomes: "
            f"{evidence['_source']['path']}"
        )
    return {int(value) for value in outcomes[name]}


def _retry_counts(evidence: dict[str, Any], stage: str) -> list[int]:
    payload = evidence.get(stage)
    if not isinstance(payload, dict) or not payload.get("enabled"):
        raise ValueError(
            f"candidate lacks enabled {stage}: "
            f"{evidence['_source']['path']}"
        )
    counts = payload.get("retry_count_by_environment")
    if not isinstance(counts, list):
        raise ValueError(
            f"candidate lacks paired {stage} retry counts: "
            f"{evidence['_source']['path']}"
        )
    return [int(value) for value in counts]


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _run_counts(evidence: dict[str, Any]) -> dict[str, int]:
    return {
        "episodes": int(evidence["num_envs"]),
        "success": len(_indices(evidence, "successful_indices")),
        "lift": len(_indices(evidence, "lifted_10mm_indices")),
        "acquisition": len(
            _indices(evidence, "receiver_acquired_indices")
        ),
    }


def _gate(identifier: str, passed: bool, **evidence: Any) -> dict[str, Any]:
    return {"id": identifier, "passed": bool(passed), **evidence}


def qualify(
    incumbents: dict[int, dict[str, Any]],
    candidates: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    gates: list[dict[str, Any]] = []
    all_candidate_runs = [
        run for seed in QUALIFICATION_SEEDS for run in candidates[seed]
    ]

    for seed in QUALIFICATION_SEEDS:
        incumbent = incumbents[seed]
        runs = candidates[seed]
        for role, evidence in (
            ("incumbent", incumbent),
            *((f"candidate_{index + 1}", run) for index, run in enumerate(runs)),
        ):
            shape_passed = (
                int(evidence.get("num_envs", -1)) == 1200
                and int(evidence.get("completed_episodes", -1)) == 1200
                and int(evidence.get("frames_per_env", -1)) == 2000
                and float(evidence.get("episode_length_s", -1.0)) == 40.0
                and float(
                    evidence.get("reset_rotation_randomization_deg", 0.0)
                )
                == 0.0
                and bool(evidence.get("first_terminal_outcome_per_environment"))
            )
            gates.append(
                _gate(
                    f"seed_{seed}_{role}_qualification_shape",
                    shape_passed,
                    num_envs=evidence.get("num_envs"),
                    completed_episodes=evidence.get("completed_episodes"),
                    frames_per_env=evidence.get("frames_per_env"),
                    episode_length_s=evidence.get("episode_length_s"),
                )
            )

        incumbent_successes = _indices(
            incumbent,
            "successful_indices",
        )
        for run_index, candidate in enumerate(runs, start=1):
            run_counts = _run_counts(candidate)
            gates.append(
                _gate(
                    f"seed_{seed}_run_{run_index}_success_at_least_78_percent",
                    _rate(
                        run_counts["success"],
                        run_counts["episodes"],
                    )
                    >= 0.78,
                    observed=_rate(
                        run_counts["success"],
                        run_counts["episodes"],
                    ),
                )
            )
            pickup_retries = _retry_counts(candidate, "pickup_recovery")
            receiver_retries = _retry_counts(
                candidate,
                "receiver_recovery",
            )
            candidate_successes = _indices(
                candidate,
                "successful_indices",
            )
            unchanged_successes = {
                index
                for index in candidate_successes
                if pickup_retries[index] == 0
                and receiver_retries[index] == 0
            }
            missing = sorted(incumbent_successes - unchanged_successes)
            gates.append(
                _gate(
                    (
                        f"seed_{seed}_run_{run_index}_"
                        "cached_first_attempt_successes"
                    ),
                    not missing,
                    incumbent_successes=len(incumbent_successes),
                    preserved=len(incumbent_successes) - len(missing),
                    missing_environment_indices=missing,
                )
            )
            candidate_terms = candidate["termination_term_counts"]
            incumbent_terms = incumbent["termination_term_counts"]
            for term in ZERO_TERMS:
                count = int(candidate_terms.get(term, 0))
                gates.append(
                    _gate(
                        f"seed_{seed}_run_{run_index}_{term}_zero",
                        count == 0,
                        candidate=count,
                    )
                )
            for term in NONINCREASE_TERMS:
                candidate_count = int(candidate_terms.get(term, 0))
                incumbent_count = int(incumbent_terms.get(term, 0))
                gates.append(
                    _gate(
                        (
                            f"seed_{seed}_run_{run_index}_{term}_"
                            "nonincrease"
                        ),
                        candidate_count <= incumbent_count,
                        candidate=candidate_count,
                        incumbent=incumbent_count,
                    )
                )
    per_seed: dict[str, dict[str, Any]] = {}
    aggregate = defaultdict(int)
    for seed in QUALIFICATION_SEEDS:
        counts = defaultdict(int)
        for run in candidates[seed]:
            for name, value in _run_counts(run).items():
                counts[name] += value
                aggregate[name] += value
        metrics = {
            **counts,
            "success_rate": _rate(counts["success"], counts["episodes"]),
            "lift_rate": _rate(counts["lift"], counts["episodes"]),
            "receiver_acquisition_given_lift": _rate(
                counts["acquisition"],
                counts["lift"],
            ),
            "retention_given_acquisition": _rate(
                counts["success"],
                counts["acquisition"],
            ),
        }
        per_seed[str(seed)] = metrics
        gates.extend(
            (
                _gate(
                    f"seed_{seed}_success_at_least_78_percent",
                    metrics["success_rate"] >= 0.78,
                    observed=metrics["success_rate"],
                ),
                _gate(
                    f"seed_{seed}_lift_at_least_92_percent",
                    metrics["lift_rate"] >= 0.92,
                    observed=metrics["lift_rate"],
                ),
            )
        )

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
    for identifier, observed, threshold in (
        ("aggregate_success_at_least_80_percent", aggregate_metrics["success_rate"], 0.80),
        ("aggregate_lift_at_least_94_percent", aggregate_metrics["lift_rate"], 0.94),
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

    base_hashes = {
        run["checkpoint"]["sha256"]
        for run in all_candidate_runs
    }
    pickup_hashes = {
        run["pickup_recovery"]["head_checkpoint"]["sha256"]
        for run in all_candidate_runs
    }
    receiver_hashes = {
        run["receiver_recovery"]["head_checkpoint"]["sha256"]
        for run in all_candidate_runs
    }
    controller_revisions = {
        run["runtime"]["source"]["dranmar_revision"]
        for run in all_candidate_runs
    }
    asset_revisions = {
        run["runtime"]["source"]["asset_revision"]
        for run in all_candidate_runs
    }
    pickup_caps = {
        (
            float(run["pickup_recovery"]["position_cap_m"]),
            float(run["pickup_recovery"]["orientation_cap_deg"]),
        )
        for run in all_candidate_runs
    }
    receiver_caps = {
        (
            float(run["receiver_recovery"]["position_cap_m"]),
            float(run["receiver_recovery"]["orientation_cap_deg"]),
        )
        for run in all_candidate_runs
    }
    immutable_bundle = {
        "base_checkpoint_sha256": sorted(base_hashes),
        "pickup_head_sha256": sorted(pickup_hashes),
        "receiver_head_sha256": sorted(receiver_hashes),
        "controller_revisions": sorted(controller_revisions),
        "asset_revisions": sorted(asset_revisions),
        "pickup_correction_caps": sorted(pickup_caps),
        "receiver_correction_caps": sorted(receiver_caps),
    }
    gates.append(
        _gate(
            "single_frozen_candidate_bundle",
            all(
                len(values) == 1
                for values in (
                    base_hashes,
                    pickup_hashes,
                    receiver_hashes,
                    controller_revisions,
                    asset_revisions,
                    pickup_caps,
                    receiver_caps,
                )
            ),
            bundle=immutable_bundle,
        )
    )
    gates.append(
        _gate(
            "correction_caps_are_approved_and_frozen",
            len(pickup_caps) == 1
            and len(receiver_caps) == 1
            and pickup_caps <= ALLOWED_CORRECTION_CAPS
            and receiver_caps <= ALLOWED_CORRECTION_CAPS,
            approved=sorted(ALLOWED_CORRECTION_CAPS),
            pickup=sorted(pickup_caps),
            receiver=sorted(receiver_caps),
        )
    )
    gates.append(
        _gate(
            "immutable_base_checkpoint_matches_lock",
            base_hashes == {IMMUTABLE_BASE_SHA256},
            expected=IMMUTABLE_BASE_SHA256,
            observed=sorted(base_hashes),
        )
    )

    return {
        "schema_version": "dranmar-handover-recovery-qualification-1.0",
        "qualified": all(gate["passed"] for gate in gates),
        "aggregate": aggregate_metrics,
        "per_seed": per_seed,
        "immutable_bundle": immutable_bundle,
        "gates": gates,
        "inputs": {
            "incumbents": {
                str(seed): incumbents[seed]["_source"]
                for seed in QUALIFICATION_SEEDS
            },
            "candidates": {
                str(seed): [
                    run["_source"] for run in candidates[seed]
                ]
                for seed in QUALIFICATION_SEEDS
            },
        },
    }


def main(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    incumbent_paths: dict[int, list[Path]] = defaultdict(list)
    candidate_paths: dict[int, list[Path]] = defaultdict(list)
    for seed, path in args.incumbent:
        incumbent_paths[seed].append(path)
    for seed, path in args.candidate:
        candidate_paths[seed].append(path)
    if set(incumbent_paths) != set(QUALIFICATION_SEEDS):
        raise ValueError("incumbent evidence must cover seeds 17, 2361, 4099")
    if set(candidate_paths) != set(QUALIFICATION_SEEDS):
        raise ValueError("candidate evidence must cover seeds 17, 2361, 4099")
    for seed in QUALIFICATION_SEEDS:
        if len(incumbent_paths[seed]) != 1:
            raise ValueError(f"seed {seed} requires exactly one incumbent")
        if len(candidate_paths[seed]) != 2:
            raise ValueError(
                f"seed {seed} requires exactly two candidate qualifications"
            )
    incumbents = {
        seed: _load(seed, incumbent_paths[seed][0])
        for seed in QUALIFICATION_SEEDS
    }
    candidates = {
        seed: [_load(seed, path) for path in candidate_paths[seed]]
        for seed in QUALIFICATION_SEEDS
    }
    report = qualify(incumbents, candidates)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
