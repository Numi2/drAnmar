#!/usr/bin/env python3
"""Gate a uniform receiver-trajectory intervention across deterministic runs.

Each runtime seed is represented by three separate Isaac processes:

* an all-no-op control;
* an independently repeated all-no-op replay;
* one uniform receiver-translation treatment.

The repeated no-op must reproduce both the pre-branch state and terminal
outcomes exactly.  The treatment must reproduce the pre-branch state exactly,
must alter only the analytically expected receiver translation channels, and
must improve paired outcomes without increasing receiver safety failures.
Risk scores remain retrospective diagnostics and never select an action.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from evaluate_dranmar_receiver_trajectory_replay import (
    IDENTITY_FIELDS,
    _exact_one_sided_sign_probability,
    _load_dataset,
    _maximum_delta,
    _risk_quartile_effects,
    _same_outcomes,
)


REPORT_SCHEMA_VERSION = (
    "dranmar-uniform-receiver-trajectory-evaluation-1.0"
)
CROSS_SEED_FIELDS = tuple(
    field for field in IDENTITY_FIELDS if field != "runtime_seed"
)
PAIR_CONTRACT_FIELDS = (
    "method",
    "pairing_key",
    "start_distance_m",
    "end_distance_m",
    "barrier_release_frame",
    "barrier_contract",
    "profile",
    "modified_action_channels",
    "risk_role",
)


def _is_scale(value: float, expected: float) -> bool:
    return math.isclose(
        value,
        expected,
        rel_tol=0.0,
        abs_tol=1.0e-7,
    )


def _require_uniform_role(
    dataset: dict[str, Any],
    *,
    role: str,
    intervention_scale: float | None = None,
) -> None:
    if dataset["replicas"] != 1 or len(dataset["scales"]) != 1:
        raise ValueError(f"{role} must contain one uniform scale")
    scale = dataset["scales"][0]
    if role in {"control", "noop"}:
        if not _is_scale(scale, 1.0):
            raise ValueError(f"{role} must use the no-op scale 1.0")
    elif (
        intervention_scale is None
        or not 0.0 < intervention_scale < 1.0
        or not _is_scale(scale, intervention_scale)
    ):
        raise ValueError(
            "candidate must use the requested uniform intervention scale"
        )
    if dataset["contract"].get(
        "grouped_initial_condition_allocation"
    ):
        raise ValueError(
            f"{role} must use independent initial-condition allocation"
        )


def _require_pair_identity(
    reference: dict[str, Any],
    other: dict[str, Any],
    *,
    label: str,
) -> None:
    for field in IDENTITY_FIELDS:
        if (
            reference["payload"].get(field)
            != other["payload"].get(field)
        ):
            raise ValueError(
                f"{label} differs on identity field {field}"
            )
    for field in PAIR_CONTRACT_FIELDS:
        if (
            reference["contract"].get(field)
            != other["contract"].get(field)
        ):
            raise ValueError(
                f"{label} differs on replay-contract field {field}"
            )
    if reference["termination_names"] != other["termination_names"]:
        raise ValueError(f"{label} differs on termination names")


def _expected_scaled_action(
    candidate: dict[str, Any],
    mask: torch.Tensor,
) -> torch.Tensor:
    base_action = candidate["base_action"][mask].clone()
    context = candidate["context"][mask]
    distance = candidate["distance"][mask]
    start = float(candidate["contract"]["start_distance_m"])
    end = float(candidate["contract"]["end_distance_m"])
    progress = ((start - distance) / (start - end)).clamp(0.0, 1.0)
    minimum_jerk = progress.square() * (3.0 - 2.0 * progress)
    scale = candidate["scales"][0]
    multiplier = 1.0 - (1.0 - scale) * minimum_jerk
    receiver_is_robot_1 = ~(context[:, 82] > 0.5)
    base_action[receiver_is_robot_1, :3] *= multiplier[
        receiver_is_robot_1
    ].unsqueeze(-1)
    base_action[~receiver_is_robot_1, 7:10] *= multiplier[
        ~receiver_is_robot_1
    ].unsqueeze(-1)
    return base_action.clamp(-1.0, 1.0)


def _prebranch_comparison(
    reference: dict[str, Any],
    other: dict[str, Any],
    *,
    context_atol: float,
    action_atol: float,
) -> dict[str, Any]:
    activation_mask_exact = torch.equal(
        reference["activation_seen"],
        other["activation_seen"],
    )
    activation_frame_exact = torch.equal(
        reference["activation_frame"],
        other["activation_frame"],
    )
    barrier_hold_exact = torch.equal(
        reference["barrier_hold_frames"],
        other["barrier_hold_frames"],
    )
    resolved_exact = torch.equal(
        reference["resolved"],
        other["resolved"],
    )
    all_resolved = bool(
        reference["resolved"].all().item()
        and other["resolved"].all().item()
    )
    active = reference["activation_seen"] & other["activation_seen"]
    context_delta = _maximum_delta(
        reference["context"][active],
        other["context"][active],
    )
    action_delta = _maximum_delta(
        reference["base_action"][active],
        other["base_action"][active],
    )
    correction_delta = _maximum_delta(
        reference["correction"][active],
        other["correction"][active],
    )
    distance_delta = _maximum_delta(
        reference["distance"][active],
        other["distance"][active],
    )
    passed = (
        all_resolved
        and resolved_exact
        and activation_mask_exact
        and activation_frame_exact
        and barrier_hold_exact
        and context_delta <= context_atol
        and action_delta <= action_atol
        and correction_delta <= context_atol
        and distance_delta <= context_atol
    )
    return {
        "passed": passed,
        "all_first_episodes_resolved": all_resolved,
        "first_episode_resolution_exact": resolved_exact,
        "activation_mask_exact": activation_mask_exact,
        "activation_frame_exact": activation_frame_exact,
        "barrier_hold_frames_exact": barrier_hold_exact,
        "activated_environments": int(active.sum().item()),
        "maximum_context_delta": context_delta,
        "maximum_base_action_delta": action_delta,
        "maximum_candidate_correction_delta": correction_delta,
        "maximum_activation_distance_delta": distance_delta,
        "context_atol": context_atol,
        "action_atol": action_atol,
    }


def _evaluate_seed(
    control: dict[str, Any],
    noop: dict[str, Any],
    candidate: dict[str, Any],
    *,
    context_atol: float,
    action_atol: float,
) -> dict[str, Any]:
    _require_uniform_role(control, role="control")
    _require_uniform_role(noop, role="noop")
    candidate_scale = candidate["scales"][0]
    _require_uniform_role(
        candidate,
        role="candidate",
        intervention_scale=candidate_scale,
    )
    _require_pair_identity(control, noop, label="no-op replay")
    _require_pair_identity(control, candidate, label="candidate replay")

    control_noop_prebranch = _prebranch_comparison(
        control,
        noop,
        context_atol=context_atol,
        action_atol=action_atol,
    )
    control_candidate_prebranch = _prebranch_comparison(
        control,
        candidate,
        context_atol=context_atol,
        action_atol=action_atol,
    )
    active = (
        control["activation_seen"]
        & noop["activation_seen"]
        & candidate["activation_seen"]
    )
    control_action_delta = _maximum_delta(
        control["scaled_action"][active],
        control["base_action"][active],
    )
    noop_action_delta = _maximum_delta(
        noop["scaled_action"][active],
        noop["base_action"][active],
    )
    expected_candidate_action = _expected_scaled_action(
        candidate,
        active,
    )
    candidate_action_delta = _maximum_delta(
        candidate["scaled_action"][active],
        expected_candidate_action,
    )
    control_quiescent = bool(
        torch.equal(
            control["modified_frames"],
            torch.zeros_like(control["modified_frames"]),
        )
        and torch.equal(
            noop["modified_frames"],
            torch.zeros_like(noop["modified_frames"]),
        )
    )
    candidate_modified = bool(
        int(active.sum().item()) > 0
        and torch.equal(
            candidate["modified_frames"][active],
            candidate["active_frames"][active],
        )
        and bool(
            torch.all(candidate["modified_frames"][active] > 0).item()
        )
        and bool(
            torch.any(candidate["minimum_multiplier"][active] < 1.0)
            .item()
        )
    )
    intervention_integrity = (
        control_action_delta <= action_atol
        and noop_action_delta <= action_atol
        and candidate_action_delta <= action_atol
        and control_quiescent
        and candidate_modified
    )
    noop_outcome_parity = _same_outcomes(
        control,
        noop,
        torch.ones_like(active, dtype=torch.bool),
    )

    control_success = control["success"][active]
    candidate_success = candidate["success"][active]
    control_safety = control["safety"][active]
    candidate_safety = candidate["safety"][active]
    wins = int((candidate_success & ~control_success).sum().item())
    losses = int((~candidate_success & control_success).sum().item())
    samples = int(active.sum().item())
    safety_delta = int(
        candidate_safety.sum().item() - control_safety.sum().item()
    )
    return {
        "runtime_seed": candidate["payload"]["runtime_seed"],
        "seed": candidate["payload"].get("seed"),
        "seed_stream_offset": candidate["payload"].get(
            "seed_stream_offset"
        ),
        "scale": candidate_scale,
        "control_path": str(control["path"]),
        "control_sha256": control["sha256"],
        "noop_path": str(noop["path"]),
        "noop_sha256": noop["sha256"],
        "candidate_path": str(candidate["path"]),
        "candidate_sha256": candidate["sha256"],
        "control_noop_prebranch": control_noop_prebranch,
        "control_candidate_prebranch": control_candidate_prebranch,
        "noop_terminal_outcome_exact": noop_outcome_parity,
        "intervention_integrity": {
            "passed": intervention_integrity,
            "control_scaled_action_delta": control_action_delta,
            "noop_scaled_action_delta": noop_action_delta,
            "candidate_expected_action_delta": candidate_action_delta,
            "control_and_noop_modified_frames_zero": (
                control_quiescent
            ),
            "candidate_modified_every_active_frame": (
                candidate_modified
            ),
        },
        "samples": samples,
        "control_successes": int(control_success.sum().item()),
        "candidate_successes": int(candidate_success.sum().item()),
        "wins": wins,
        "losses": losses,
        "paired_net_successes": wins - losses,
        "paired_success_uplift": (
            (wins - losses) / samples if samples else 0.0
        ),
        "control_safety_failures": int(control_safety.sum().item()),
        "candidate_safety_failures": int(
            candidate_safety.sum().item()
        ),
        "safety_delta": safety_delta,
        "risk_quartiles_exploratory": _risk_quartile_effects(
            control,
            candidate,
            active,
        ),
    }


def _gate(
    per_seed: list[dict[str, Any]],
    *,
    minimum_seeds: int,
    significance_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    replay_parity = all(
        item["control_noop_prebranch"]["passed"]
        and item["control_candidate_prebranch"]["passed"]
        and item["noop_terminal_outcome_exact"]
        for item in per_seed
    )
    intervention_integrity = all(
        item["intervention_integrity"]["passed"]
        for item in per_seed
    )
    wins = sum(item["wins"] for item in per_seed)
    losses = sum(item["losses"] for item in per_seed)
    samples = sum(item["samples"] for item in per_seed)
    safety_delta = sum(item["safety_delta"] for item in per_seed)
    probability = _exact_one_sided_sign_probability(wins, losses)
    aggregate = {
        "seeds": len(per_seed),
        "samples": samples,
        "wins": wins,
        "losses": losses,
        "paired_net_successes": wins - losses,
        "paired_success_uplift": (
            (wins - losses) / samples if samples else 0.0
        ),
        "safety_delta": safety_delta,
        "one_sided_exact_sign_probability": probability,
    }
    seed_consistent = all(
        item["paired_net_successes"] >= 0
        and item["safety_delta"] <= 0
        for item in per_seed
    )
    first_seed_promising = (
        per_seed[0]["paired_net_successes"] > 0
        and per_seed[0]["safety_delta"] <= 0
    )
    if not replay_parity:
        decision = "replay_invalid"
        next_action = "stop_and_fix_replay"
    elif not intervention_integrity:
        decision = "intervention_delivery_invalid"
        next_action = "stop_and_fix_intervention_instrumentation"
    elif not seed_consistent or not first_seed_promising:
        decision = "uniform_intervention_rejected"
        next_action = "do_not_train_from_this_intervention"
    elif len(per_seed) < minimum_seeds:
        decision = "replication_required"
        next_action = "run_remaining_prespecified_runtime_seeds"
    elif (
        wins > losses
        and safety_delta <= 0
        and probability <= significance_threshold
    ):
        decision = "uniform_intervention_positive"
        next_action = "train_advantage_weighted_behavior_clone"
    else:
        decision = "uniform_intervention_rejected"
        next_action = "do_not_train_from_this_intervention"
    return aggregate, {
        "passed": decision == "uniform_intervention_positive",
        "decision": decision,
        "minimum_seeds": minimum_seeds,
        "seed_consistent_nonnegative_effect": seed_consistent,
        "first_seed_strictly_positive": first_seed_promising,
        "replay_parity_passed": replay_parity,
        "intervention_integrity_passed": intervention_integrity,
        "significance_threshold": significance_threshold,
        "next_action": next_action,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--control-dataset",
        action="append",
        required=True,
    )
    parser.add_argument(
        "--noop-dataset",
        action="append",
        required=True,
    )
    parser.add_argument(
        "--candidate-dataset",
        action="append",
        required=True,
    )
    parser.add_argument("--context-atol", type=float, default=0.0)
    parser.add_argument("--action-atol", type=float, default=0.0)
    parser.add_argument("--minimum-seeds", type=int, default=3)
    parser.add_argument(
        "--paired-significance-threshold",
        type=float,
        default=0.05,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    dataset_counts = {
        len(args.control_dataset),
        len(args.noop_dataset),
        len(args.candidate_dataset),
    }
    if (
        args.context_atol < 0.0
        or args.action_atol < 0.0
        or args.minimum_seeds < 1
        or not 0.0 < args.paired_significance_threshold <= 0.05
        or len(dataset_counts) != 1
    ):
        print("error: invalid uniform evaluator arguments")
        return 2
    try:
        controls = [
            _load_dataset(Path(value).expanduser().resolve())
            for value in args.control_dataset
        ]
        noops = [
            _load_dataset(Path(value).expanduser().resolve())
            for value in args.noop_dataset
        ]
        candidates = [
            _load_dataset(Path(value).expanduser().resolve())
            for value in args.candidate_dataset
        ]
        per_seed = [
            _evaluate_seed(
                control,
                noop,
                candidate,
                context_atol=args.context_atol,
                action_atol=args.action_atol,
            )
            for control, noop, candidate in zip(
                controls,
                noops,
                candidates,
                strict=True,
            )
        ]
        reference = candidates[0]
        for candidate in candidates[1:]:
            for field in CROSS_SEED_FIELDS:
                if (
                    candidate["payload"].get(field)
                    != reference["payload"].get(field)
                ):
                    raise ValueError(
                        f"candidate datasets differ across seeds on {field}"
                    )
            if candidate["scales"] != reference["scales"]:
                raise ValueError(
                    "candidate intervention scale differs across seeds"
                )
        runtime_seeds = [
            item["payload"]["runtime_seed"] for item in candidates
        ]
        if len(set(runtime_seeds)) != len(runtime_seeds):
            raise ValueError("runtime seeds must be unique")
        aggregate, gate = _gate(
            per_seed,
            minimum_seeds=args.minimum_seeds,
            significance_threshold=(
                args.paired_significance_threshold
            ),
        )
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"error: {error}")
        return 2

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_contract": {
            field: reference["payload"].get(field)
            for field in CROSS_SEED_FIELDS
        },
        "intervention": {
            "scale": reference["scales"][0],
            "profile": reference["contract"].get("profile"),
            "start_distance_m": reference["contract"].get(
                "start_distance_m"
            ),
            "end_distance_m": reference["contract"].get(
                "end_distance_m"
            ),
            "barrier_release_frame": reference["contract"].get(
                "barrier_release_frame"
            ),
            "modified_action_channels": reference["contract"].get(
                "modified_action_channels"
            ),
        },
        "causal_design": (
            "Separate deterministic Isaac processes for control, repeated "
            "no-op, and one uniform treatment at each runtime seed."
        ),
        "risk_usage": (
            "Control-run post-acquisition scores are exploratory "
            "heterogeneity labels only. They did not select actions and "
            "are not part of the causal gate."
        ),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "gate": gate,
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(f"[DrAnmar] Uniform trajectory evaluation: {output_path}")
    print(
        "[DrAnmar] Decision: "
        f"{gate['decision']} (passed={gate['passed']})"
    )
    return (
        2
        if gate["decision"]
        in {"replay_invalid", "intervention_delivery_invalid"}
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
