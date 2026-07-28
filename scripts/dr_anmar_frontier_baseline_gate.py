#!/usr/bin/env python3
"""Qualify the zero-residual nominal handover before frontier PPO is allowed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class BaselineGateError(ValueError):
    """Baseline evidence is incomplete or does not match the contract."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BaselineGateError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise BaselineGateError(f"JSON root must be an object: {path}")
    return value


def _require_rate(value: Any, field: str) -> float:
    if not isinstance(value, (float, int)):
        raise BaselineGateError(f"{field} must be numeric")
    rate = float(value)
    if not 0.0 <= rate <= 1.0:
        raise BaselineGateError(f"{field} must be in [0, 1]")
    return rate


def analyze_nominal_baseline(
    config: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Return a fail-closed report for a no-learning frontier baseline."""
    baseline = config["nominal_baseline_qualification"]
    expected_task = str(config["qualification"]["task"])
    if evidence.get("kind") != "held_out_play":
        raise BaselineGateError("baseline evidence must be held_out_play")
    if evidence.get("task") != expected_task:
        raise BaselineGateError(
            f"baseline task must be {expected_task!r}"
        )
    requested = int(evidence["requested_num_envs"])
    actual = int(evidence["num_envs"])
    if requested != int(baseline["num_envs"]) or actual != requested:
        raise BaselineGateError(
            "baseline must run the exact configured environment population"
        )
    if int(evidence["frames_per_env"]) < int(baseline["frames_per_env"]):
        raise BaselineGateError("baseline did not run the full horizon")
    bundle = evidence.get("policy_bundle", {})
    if (
        not bundle.get("bound")
        or bundle.get("adaptation_mode") != "frontier_hardening"
        or bundle.get("controller_profile", {}).get("name")
        != "frontier-hardening-v24"
    ):
        raise BaselineGateError(
            "baseline requires a bound frontier-hardening-v24 policy bundle"
        )
    checkpoint = evidence.get("checkpoint", {})
    if checkpoint.get("sha256") != bundle.get("checkpoint_sha256"):
        raise BaselineGateError("checkpoint and policy bundle hashes differ")

    diagnostics = evidence.get(
        "first_episode_handover_diagnostics",
        {},
    )
    roles = diagnostics.get("initial_giver_role_population", {})
    robot_1_population = int(roles.get("robot_1", -1))
    robot_2_population = int(roles.get("robot_2", -1))
    if robot_1_population + robot_2_population != requested:
        raise BaselineGateError(
            "initial giver-role populations do not cover the baseline"
        )
    role_imbalance = abs(robot_1_population - robot_2_population)
    yaw_buckets = diagnostics.get("initial_yaw_bucket_statistics", {})
    expected_bucket_count = int(baseline["yaw_bucket_count"])
    if len(yaw_buckets) != expected_bucket_count:
        raise BaselineGateError(
            f"expected {expected_bucket_count} initial-yaw buckets"
        )

    total_contact = 0.0
    total_lift = 0.0
    total_success = 0.0
    total_safety = 0.0
    bucket_population = 0
    completed_outcomes = 0
    minimum_bucket_contact = 1.0
    minimum_bucket_lift = 1.0
    minimum_bucket_samples = requested
    bucket_results: dict[str, dict[str, Any]] = {}
    for label, stats in yaw_buckets.items():
        count = int(stats["count"])
        completed = int(stats["completed"])
        if count < 0 or completed < 0 or completed > count:
            raise BaselineGateError(
                f"{label} has invalid count/completed accounting"
            )
        contact_rate = _require_rate(
            stats["giver_bilateral_contact_rate"],
            f"{label}.giver_bilateral_contact_rate",
        )
        lift_rate = _require_rate(
            stats["reached_10mm_lift_rate"],
            f"{label}.reached_10mm_lift_rate",
        )
        success_rate = _require_rate(
            stats["retained_handover_success_rate"],
            f"{label}.retained_handover_success_rate",
        )
        safety_rate = _require_rate(
            stats["safety_terminal_rate"],
            f"{label}.safety_terminal_rate",
        )
        bucket_population += count
        completed_outcomes += completed
        minimum_bucket_samples = min(minimum_bucket_samples, count)
        minimum_bucket_contact = min(
            minimum_bucket_contact,
            contact_rate,
        )
        minimum_bucket_lift = min(minimum_bucket_lift, lift_rate)
        total_contact += count * contact_rate
        total_lift += count * lift_rate
        total_success += count * success_rate
        total_safety += count * safety_rate
        bucket_results[label] = {
            "count": count,
            "completed": completed,
            "giver_bilateral_contact_rate": contact_rate,
            "reached_10mm_lift_rate": lift_rate,
            "retained_handover_success_rate": success_rate,
            "safety_terminal_rate": safety_rate,
        }
    if bucket_population != requested:
        raise BaselineGateError(
            "initial-yaw bucket populations do not cover the baseline"
        )
    overall_contact = total_contact / requested
    overall_lift = total_lift / requested
    first_episode_success_rate = total_success / requested
    first_episode_safety_rate = total_safety / requested
    completed_outcome_fraction = completed_outcomes / requested

    outcomes_by_role = diagnostics.get("outcomes_by_giver_role", {})
    role_lift_rates = {}
    role_outcome_population = 0
    expected_role_populations = {
        "robot_1": robot_1_population,
        "robot_2": robot_2_population,
    }
    for role in ("robot_1", "robot_2"):
        stats = outcomes_by_role.get(role, {})
        count = int(stats.get("count", 0))
        if count <= 0:
            raise BaselineGateError(f"{role} has no baseline episodes")
        if count != expected_role_populations[role]:
            raise BaselineGateError(
                f"{role} outcome count differs from its initial population"
            )
        reached_lift = int(stats.get("reached_10mm_lift", -1))
        if not 0 <= reached_lift <= count:
            raise BaselineGateError(
                f"{role} has invalid 10 mm lift accounting"
            )
        role_outcome_population += count
        role_lift_rates[role] = reached_lift / count
    if role_outcome_population != requested:
        raise BaselineGateError(
            "role outcome populations do not cover the baseline"
        )
    role_pickup_gap = abs(
        role_lift_rates["robot_1"] - role_lift_rates["robot_2"]
    )

    completed = int(evidence["completed_episodes"])
    termination_counts = evidence.get("termination_term_counts", {})
    safety_names = (
        "object_dropping",
        "excessive_object_force",
        "protected_surface_force",
    )
    if any(name not in termination_counts for name in safety_names):
        raise BaselineGateError(
            "baseline evidence is missing safety termination counts"
        )
    safety_counts = {
        name: int(termination_counts[name]) for name in safety_names
    }
    if any(value < 0 for value in safety_counts.values()):
        raise BaselineGateError(
            "safety termination counts cannot be negative"
        )
    safety_terminals = sum(safety_counts.values())
    global_safety_rate = (
        safety_terminals / completed if completed else 1.0
    )
    safety_rate = max(first_episode_safety_rate, global_safety_rate)
    residual_norm = float(
        diagnostics.get("frontier_hardening_residual", {}).get(
            "maximum_normalized_l2",
            float("inf"),
        )
    )

    checks = {
        "exact_initial_role_balance": role_imbalance
        <= int(baseline["maximum_initial_role_imbalance"]),
        "yaw_bucket_coverage": minimum_bucket_samples
        >= int(baseline["minimum_samples_per_yaw_bucket"]),
        "completed_outcome_coverage": completed_outcome_fraction
        >= float(baseline["minimum_completed_outcome_fraction"]),
        "overall_giver_contact": overall_contact
        >= float(
            baseline[
                "minimum_overall_giver_bilateral_contact_rate"
            ]
        ),
        "worst_yaw_bucket_giver_contact": minimum_bucket_contact
        >= float(
            baseline[
                "minimum_yaw_bucket_giver_bilateral_contact_rate"
            ]
        ),
        "overall_10mm_lift": overall_lift
        >= float(baseline["minimum_overall_10mm_lift_rate"]),
        "worst_yaw_bucket_10mm_lift": minimum_bucket_lift
        >= float(baseline["minimum_yaw_bucket_10mm_lift_rate"]),
        "retained_handover_success": first_episode_success_rate
        >= float(
            baseline["minimum_retained_handover_success_rate"]
        ),
        "giver_role_pickup_parity": role_pickup_gap
        <= float(baseline["maximum_giver_role_pickup_rate_gap"]),
        "zero_safety_terminals": safety_rate
        <= float(baseline["maximum_safety_terminal_rate"]),
        "zero_frontier_residual": 0.0
        <= residual_norm
        <= float(baseline["maximum_frontier_residual_norm"]),
    }
    passed = all(checks.values())
    return {
        "schema_version": "dranmar-frontier-baseline-gate-1.0",
        "experiment_id": config["experiment_id"],
        "task": expected_task,
        "passed": passed,
        "policy_updates_allowed": passed,
        "checks": checks,
        "measurements": {
            "requested_num_envs": requested,
            "completed_episodes": completed,
            "completed_first_episode_outcomes": completed_outcomes,
            "completed_outcome_fraction": completed_outcome_fraction,
            "initial_role_imbalance": role_imbalance,
            "minimum_yaw_bucket_samples": minimum_bucket_samples,
            "overall_giver_bilateral_contact_rate": overall_contact,
            "minimum_yaw_bucket_giver_bilateral_contact_rate": (
                minimum_bucket_contact
            ),
            "overall_10mm_lift_rate": overall_lift,
            "minimum_yaw_bucket_10mm_lift_rate": minimum_bucket_lift,
            "retained_handover_success_rate": (
                first_episode_success_rate
            ),
            "giver_role_10mm_lift_rates": role_lift_rates,
            "giver_role_pickup_rate_gap": role_pickup_gap,
            "safety_terminal_rate": safety_rate,
            "maximum_frontier_residual_norm": residual_norm,
            "yaw_buckets": bucket_results,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Qualify the zero-residual handover controller before PPO"
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = analyze_nominal_baseline(
            _read_json(args.config.resolve()),
            _read_json(args.evidence.resolve()),
        )
    except (KeyError, TypeError, ValueError, BaselineGateError) as error:
        print(f"error: {error}")
        return 2
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
