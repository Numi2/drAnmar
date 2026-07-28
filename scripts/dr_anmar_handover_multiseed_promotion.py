#!/usr/bin/env python3
"""Promote one handover candidate only across complete held-out populations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


HARD_FAILURES = (
    "object_dropping",
    "needle_dropped_after_pickup",
    "excessive_object_force",
    "premature_giver_release",
)
PROTECTED_SURFACE_FAILURE = "protected_surface_force"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires a positive population")
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            rate * (1.0 - rate) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - margin, center + margin


def _load(path: Path) -> dict[str, Any]:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "task",
        "kind",
        "seed",
        "num_envs",
        "frames_per_env",
        "completed_episodes",
        "successful_episodes",
        "failure_distribution",
        "checkpoint",
        "policy_runtime_contract_sha256",
        "environment_runtime_contract_sha256",
        "first_terminal_outcome_per_environment",
    }
    missing = sorted(required - evidence.keys())
    if missing:
        raise ValueError(f"{path}: missing {', '.join(missing)}")
    if evidence["kind"] != "held_out_play":
        raise ValueError(f"{path}: evidence is not held-out play")
    if not evidence["first_terminal_outcome_per_environment"]:
        raise ValueError(f"{path}: first-terminal accounting is required")
    if evidence["completed_episodes"] != evidence["num_envs"]:
        raise ValueError(f"{path}: incomplete environment population")
    if not evidence["checkpoint"]:
        raise ValueError(f"{path}: checkpoint identity is required")
    evidence["_path"] = path
    return evidence


def _by_seed(paths: list[Path]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for path in paths:
        evidence = _load(path)
        seed = int(evidence["seed"])
        if seed in result:
            raise ValueError(f"duplicate evidence for seed {seed}")
        result[seed] = evidence
    return result


def evaluate_multiseed(
    baseline_paths: list[Path],
    candidate_paths: list[Path],
    *,
    required_seeds: set[int],
    minimum_success_rate: float,
    maximum_seed_regression: float,
    minimum_aggregate_improvement: float = 0.005,
    minimum_seed_count: int = 3,
    maximum_protected_surface_rate_increase: float = 0.001,
    require_paired_population: bool = True,
) -> dict[str, Any]:
    if len(required_seeds) < minimum_seed_count:
        raise ValueError(
            "promotion requires at least "
            f"{minimum_seed_count} pre-registered seeds"
        )
    baseline = _by_seed(baseline_paths)
    candidate = _by_seed(candidate_paths)
    if set(baseline) != required_seeds:
        raise ValueError("baseline does not cover the exact required seeds")
    if set(candidate) != required_seeds:
        raise ValueError("candidate does not cover the exact required seeds")

    checkpoint_hashes = {
        item["checkpoint"]["sha256"] for item in candidate.values()
    }
    if len(checkpoint_hashes) != 1:
        raise ValueError("candidate evidence does not use one frozen checkpoint")
    baseline_runtime_contract_hashes = {
        item["policy_runtime_contract_sha256"] for item in baseline.values()
    }
    candidate_runtime_contract_hashes = {
        item["policy_runtime_contract_sha256"] for item in candidate.values()
    }
    if len(baseline_runtime_contract_hashes) != 1:
        raise ValueError(
            "baseline evidence does not use one frozen runtime contract"
        )
    if len(candidate_runtime_contract_hashes) != 1:
        raise ValueError(
            "candidate evidence does not use one frozen runtime contract"
        )
    baseline_environment_contract_hashes = {
        item["environment_runtime_contract_sha256"]
        for item in baseline.values()
    }
    candidate_environment_contract_hashes = {
        item["environment_runtime_contract_sha256"]
        for item in candidate.values()
    }
    if len(baseline_environment_contract_hashes) != 1:
        raise ValueError(
            "baseline evidence does not use one frozen environment contract"
        )
    if len(candidate_environment_contract_hashes) != 1:
        raise ValueError(
            "candidate evidence does not use one frozen environment contract"
        )
    if (
        baseline_environment_contract_hashes
        != candidate_environment_contract_hashes
    ):
        raise ValueError(
            "baseline and candidate environment contracts differ"
        )

    per_seed = []
    total_successes = 0
    total_episodes = 0
    total_baseline_successes = 0
    hard_failure_totals = {name: 0 for name in HARD_FAILURES}
    baseline_protected_surface_failures = 0
    candidate_protected_surface_failures = 0
    paired_population_gate = True
    for seed in sorted(required_seeds):
        baseline_item = baseline[seed]
        candidate_item = candidate[seed]
        for field in ("task", "num_envs", "frames_per_env"):
            if baseline_item[field] != candidate_item[field]:
                raise ValueError(
                    f"seed {seed}: baseline and candidate {field} differ"
                )
        baseline_population = baseline_item.get(
            "initial_state_population_sha256"
        )
        candidate_population = candidate_item.get(
            "initial_state_population_sha256"
        )
        if require_paired_population and (
            not baseline_population or not candidate_population
        ):
            raise ValueError(
                f"seed {seed}: paired initial-state population hash is required"
            )
        population_matches = (
            bool(baseline_population)
            and baseline_population == candidate_population
        )
        paired_population_gate &= (
            population_matches if require_paired_population else True
        )
        baseline_rate = (
            baseline_item["successful_episodes"]
            / baseline_item["completed_episodes"]
        )
        candidate_rate = (
            candidate_item["successful_episodes"]
            / candidate_item["completed_episodes"]
        )
        seed_regression = baseline_rate - candidate_rate
        seed_hard = {
            name: int(
                candidate_item["failure_distribution"].get(name, 0)
            )
            for name in HARD_FAILURES
        }
        for name, count in seed_hard.items():
            hard_failure_totals[name] += count
        baseline_protected = int(
            baseline_item["failure_distribution"].get(
                PROTECTED_SURFACE_FAILURE,
                0,
            )
        )
        candidate_protected = int(
            candidate_item["failure_distribution"].get(
                PROTECTED_SURFACE_FAILURE,
                0,
            )
        )
        baseline_protected_surface_failures += baseline_protected
        candidate_protected_surface_failures += candidate_protected
        total_successes += int(candidate_item["successful_episodes"])
        total_episodes += int(candidate_item["completed_episodes"])
        total_baseline_successes += int(
            baseline_item["successful_episodes"]
        )
        per_seed.append(
            {
                "seed": seed,
                "baseline_success_rate": baseline_rate,
                "candidate_success_rate": candidate_rate,
                "candidate_minus_baseline": (
                    candidate_rate - baseline_rate
                ),
                "seed_regression_gate_passed": (
                    seed_regression <= maximum_seed_regression
                ),
                "candidate_hard_failures": seed_hard,
                "baseline_protected_surface_failures": baseline_protected,
                "candidate_protected_surface_failures": candidate_protected,
                "initial_state_population_sha256": baseline_population,
                "paired_population_gate_passed": population_matches,
                "baseline_evidence_path": str(
                    baseline_item["_path"].resolve()
                ),
                "baseline_evidence_sha256": _sha256(
                    baseline_item["_path"]
                ),
                "candidate_evidence_path": str(
                    candidate_item["_path"].resolve()
                ),
                "candidate_evidence_sha256": _sha256(
                    candidate_item["_path"]
                ),
            }
        )

    aggregate_rate = total_successes / total_episodes
    baseline_rate = total_baseline_successes / total_episodes
    lower, upper = _wilson_interval(total_successes, total_episodes)
    success_gate = (
        aggregate_rate >= minimum_success_rate
        and lower >= minimum_success_rate
    )
    aggregate_improvement = aggregate_rate - baseline_rate
    aggregate_improvement_gate = (
        aggregate_improvement >= minimum_aggregate_improvement
    )
    per_seed_gate = all(
        item["seed_regression_gate_passed"] for item in per_seed
    )
    catastrophic_safety_gate = not any(hard_failure_totals.values())
    baseline_protected_rate = (
        baseline_protected_surface_failures / total_episodes
    )
    candidate_protected_rate = (
        candidate_protected_surface_failures / total_episodes
    )
    protected_surface_noninferiority_gate = (
        candidate_protected_rate - baseline_protected_rate
        <= maximum_protected_surface_rate_increase
    )
    safety_gate = (
        catastrophic_safety_gate
        and protected_surface_noninferiority_gate
    )
    promotable = (
        success_gate
        and aggregate_improvement_gate
        and per_seed_gate
        and safety_gate
        and paired_population_gate
    )
    return {
        "schema_version": "dranmar-handover-multiseed-promotion-1.3",
        "decision": (
            "candidate_promoted"
            if promotable
            else "baseline_retained"
        ),
        "required_seeds": sorted(required_seeds),
        "candidate_checkpoint_sha256": checkpoint_hashes.pop(),
        "baseline_policy_runtime_contract_sha256": (
            baseline_runtime_contract_hashes.pop()
        ),
        "candidate_policy_runtime_contract_sha256": (
            candidate_runtime_contract_hashes.pop()
        ),
        "environment_runtime_contract_sha256": (
            baseline_environment_contract_hashes.pop()
        ),
        "aggregate": {
            "baseline_successes": total_baseline_successes,
            "candidate_successes": total_successes,
            "episodes": total_episodes,
            "baseline_success_rate": baseline_rate,
            "candidate_success_rate": aggregate_rate,
            "candidate_minus_baseline": aggregate_improvement,
            "candidate_success_wilson_95": [lower, upper],
            "hard_failures": hard_failure_totals,
            "baseline_protected_surface_failures": (
                baseline_protected_surface_failures
            ),
            "candidate_protected_surface_failures": (
                candidate_protected_surface_failures
            ),
            "baseline_protected_surface_failure_rate": (
                baseline_protected_rate
            ),
            "candidate_protected_surface_failure_rate": (
                candidate_protected_rate
            ),
        },
        "gates": {
            "minimum_success_rate": minimum_success_rate,
            "minimum_aggregate_improvement": (
                minimum_aggregate_improvement
            ),
            "aggregate_improvement_gate_passed": (
                aggregate_improvement_gate
            ),
            "minimum_seed_count": minimum_seed_count,
            "maximum_seed_regression": maximum_seed_regression,
            "success_gate_passed": success_gate,
            "per_seed_regression_gate_passed": per_seed_gate,
            "require_paired_population": require_paired_population,
            "paired_population_gate_passed": paired_population_gate,
            "zero_catastrophic_failure_gate_passed": (
                catastrophic_safety_gate
            ),
            "maximum_protected_surface_rate_increase": (
                maximum_protected_surface_rate_increase
            ),
            "protected_surface_noninferiority_gate_passed": (
                protected_surface_noninferiority_gate
            ),
            "safety_gate_passed": safety_gate,
        },
        "per_seed": per_seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--required-seeds",
        default="2361,4099,7919",
    )
    parser.add_argument("--minimum-success-rate", type=float, default=0.60)
    parser.add_argument(
        "--minimum-aggregate-improvement",
        type=float,
        default=0.005,
    )
    parser.add_argument("--minimum-seed-count", type=int, default=3)
    parser.add_argument(
        "--maximum-seed-regression",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--maximum-protected-surface-rate-increase",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--allow-unpaired-populations",
        action="store_true",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.minimum_aggregate_improvement < 0.0:
        parser.error("minimum aggregate improvement must be non-negative")
    if args.minimum_seed_count <= 0:
        parser.error("minimum seed count must be positive")
    required_seeds = {
        int(value.strip())
        for value in args.required_seeds.split(",")
        if value.strip()
    }
    result = evaluate_multiseed(
        args.baseline,
        args.candidate,
        required_seeds=required_seeds,
        minimum_success_rate=args.minimum_success_rate,
        maximum_seed_regression=args.maximum_seed_regression,
        minimum_aggregate_improvement=(
            args.minimum_aggregate_improvement
        ),
        minimum_seed_count=args.minimum_seed_count,
        maximum_protected_surface_rate_increase=(
            args.maximum_protected_surface_rate_increase
        ),
        require_paired_population=not args.allow_unpaired_populations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[DrAnmar] Multiseed decision: {result['decision']}")
    print(f"[DrAnmar] Evidence: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
