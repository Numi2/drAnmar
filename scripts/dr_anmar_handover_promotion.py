#!/usr/bin/env python3
"""Select a handover checkpoint only when deterministic physics evidence improves."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SAFETY_FAILURES = (
    "excessive_object_force",
    "needle_dropped_after_pickup",
    "object_dropping",
    "pickup_attempts_exhausted",
    "premature_giver_release",
    "protected_surface_force",
    "receiver_retention_lost",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    """Record repository-relative evidence paths when invoked from the repo."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _load_play_evidence(path: Path) -> dict[str, Any]:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "task",
        "seed",
        "num_envs",
        "frames_per_env",
        "completed_episodes",
        "success_rate",
        "failure_distribution",
        "checkpoint",
    }
    missing = sorted(required - evidence.keys())
    if missing:
        raise ValueError(f"{path}: missing fields: {', '.join(missing)}")
    if evidence.get("kind") not in {"play", "held_out_play"}:
        raise ValueError(f"{path}: promotion requires deterministic play evidence")
    if evidence["completed_episodes"] <= 0:
        raise ValueError(f"{path}: no completed episodes")
    return evidence


def _failure_rates(evidence: dict[str, Any]) -> dict[str, float]:
    completed = float(evidence["completed_episodes"])
    failures = evidence["failure_distribution"]
    return {
        name: float(failures.get(name, 0)) / completed
        for name in SAFETY_FAILURES
    }


def _compatible(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    return all(
        baseline[field] == candidate[field]
        for field in ("task", "seed", "num_envs", "frames_per_env")
    )


def evaluate_candidate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    minimum_success_improvement: float,
    maximum_safety_rate_increase: float,
) -> dict[str, Any]:
    if not _compatible(baseline, candidate):
        raise ValueError(
            "candidate evidence must match baseline task, seed, "
            "environment count, and frames"
        )
    baseline_rate = float(baseline["success_rate"])
    candidate_rate = float(candidate["success_rate"])
    baseline_failures = _failure_rates(baseline)
    candidate_failures = _failure_rates(candidate)
    safety_regressions = {
        name: {
            "baseline_rate": baseline_failures[name],
            "candidate_rate": candidate_failures[name],
            "increase": candidate_failures[name] - baseline_failures[name],
        }
        for name in SAFETY_FAILURES
        if (
            candidate_failures[name] - baseline_failures[name]
            > maximum_safety_rate_increase
        )
    }
    success_improvement = candidate_rate - baseline_rate
    success_gate = success_improvement >= minimum_success_improvement
    return {
        "checkpoint": candidate["checkpoint"],
        "success_rate": candidate_rate,
        "success_improvement": success_improvement,
        "success_gate_passed": success_gate,
        "safety_gate_passed": not safety_regressions,
        "safety_regressions": safety_regressions,
        "promotable": success_gate and not safety_regressions,
    }


def select_checkpoint(
    baseline_path: Path,
    candidate_paths: list[Path],
    *,
    minimum_success_improvement: float,
    maximum_safety_rate_increase: float,
) -> dict[str, Any]:
    baseline = _load_play_evidence(baseline_path)
    comparisons = []
    for candidate_path in candidate_paths:
        candidate = _load_play_evidence(candidate_path)
        comparison = evaluate_candidate(
            baseline,
            candidate,
            minimum_success_improvement=minimum_success_improvement,
            maximum_safety_rate_increase=maximum_safety_rate_increase,
        )
        comparison["evidence_path"] = _portable_path(candidate_path)
        comparison["evidence_sha256"] = _sha256(candidate_path)
        comparisons.append(comparison)

    promotable = [item for item in comparisons if item["promotable"]]
    selected = (
        max(promotable, key=lambda item: item["success_rate"])
        if promotable
        else {
            "checkpoint": baseline["checkpoint"],
            "success_rate": baseline["success_rate"],
            "evidence_path": _portable_path(baseline_path),
            "evidence_sha256": _sha256(baseline_path),
        }
    )
    return {
        "schema_version": "dranmar-handover-promotion-1.0",
        "decision": (
            "candidate_promoted" if promotable else "baseline_retained"
        ),
        "qualification_contract": {
            "deterministic_play_only": True,
            "matched_task_seed_num_envs_and_frames": True,
            "minimum_success_improvement": minimum_success_improvement,
            "maximum_safety_failure_rate_increase": (
                maximum_safety_rate_increase
            ),
            "safety_failure_terms": list(SAFETY_FAILURES),
        },
        "baseline": {
            "checkpoint": baseline["checkpoint"],
            "success_rate": baseline["success_rate"],
            "failure_rates": _failure_rates(baseline),
            "evidence_path": _portable_path(baseline_path),
            "evidence_sha256": _sha256(baseline_path),
        },
        "candidates": comparisons,
        "selected": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-success-improvement", type=float, default=0.0)
    parser.add_argument(
        "--maximum-safety-rate-increase",
        type=float,
        default=0.0,
    )
    args = parser.parse_args()
    if args.minimum_success_improvement < 0.0:
        parser.error("minimum success improvement must be non-negative")
    if args.maximum_safety_rate_increase < 0.0:
        parser.error("maximum safety rate increase must be non-negative")
    result = select_checkpoint(
        args.baseline.resolve(),
        [path.resolve() for path in args.candidate],
        minimum_success_improvement=args.minimum_success_improvement,
        maximum_safety_rate_increase=args.maximum_safety_rate_increase,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[DrAnmar] Promotion decision: {result['decision']}")
    print(f"[DrAnmar] Evidence: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
