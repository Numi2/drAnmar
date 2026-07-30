#!/usr/bin/env python3
"""Gate paired receiver final-approach trajectory replays.

The evaluator treats simultaneous grouped replicas as an intent-to-treat
counterfactual experiment.  It first proves that every replica is identical
at the branch boundary.  An all-no-op run must additionally produce exactly
identical outcomes inside every group.  Intervention candidates are eligible
for learning only after a positive, seed-consistent, safety-neutral paired
effect with a one-sided exact sign-test probability at or below 0.05.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


SCHEMA_VERSION = "dranmar-receiver-approach-trajectory-replay-1.0"
REPORT_SCHEMA_VERSION = (
    "dranmar-receiver-approach-trajectory-replay-evaluation-1.0"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_one_sided_sign_probability(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0 or wins <= losses:
        return 1.0
    numerator = sum(
        math.comb(discordant, value)
        for value in range(wins, discordant + 1)
    )
    return numerator / float(2**discordant)


def _maximum_replica_delta(
    grouped: torch.Tensor,
    group_mask: torch.Tensor,
) -> float:
    selected = grouped[group_mask]
    if selected.numel() == 0:
        return 0.0
    reference = selected[:, :1]
    return float((selected - reference).abs().max().item())


def _require_tensor(
    payload: dict[str, Any],
    name: str,
    *,
    length: int,
) -> torch.Tensor:
    value = payload.get(name)
    if not isinstance(value, torch.Tensor) or value.shape[0] != length:
        raise ValueError(f"{name} must be a tensor with {length} rows")
    return value.cpu()


def _termination_counts(
    names: list[str],
    flags: torch.Tensor,
) -> dict[str, int]:
    return {
        name: int(flags[:, index].sum().item())
        for index, name in enumerate(names)
    }


def _risk_quartile_effects(
    record: dict[str, Any],
    candidate_index: int,
) -> list[dict[str, Any]]:
    risk_observed = record["risk_observed"][:, 0]
    if int(risk_observed.sum().item()) < 8:
        return []
    risk = record["risk"][:, 0]
    selected_risk = risk[risk_observed]
    boundaries = torch.quantile(
        selected_risk,
        torch.tensor((0.25, 0.5, 0.75)),
    )
    bucket = torch.bucketize(risk, boundaries)
    output = []
    for quartile in range(4):
        selected = risk_observed & (bucket == quartile)
        control = record["success"][selected, 0]
        candidate = record["success"][selected, candidate_index]
        output.append(
            {
                "quartile": quartile + 1,
                "groups": int(selected.sum().item()),
                "control_successes": int(control.sum().item()),
                "candidate_successes": int(candidate.sum().item()),
                "paired_net_successes": int(
                    (candidate & ~control).sum().item()
                    - (~candidate & control).sum().item()
                ),
            }
        )
    return output


def _evaluate_dataset(
    path: Path,
    *,
    context_atol: float,
    action_atol: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("dataset root must be a dictionary")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema: {payload.get('schema_version')!r}"
        )
    replay_contract = payload.get("replay_contract")
    if not isinstance(replay_contract, dict):
        raise ValueError("replay_contract is missing")
    scales = replay_contract.get("candidate_scales")
    replicas = replay_contract.get("group_replicas")
    num_envs = payload.get("num_envs")
    if (
        not isinstance(scales, list)
        or not isinstance(replicas, int)
        or not isinstance(num_envs, int)
        or replicas != len(scales)
        or replicas < 2
        or num_envs <= 0
        or num_envs % replicas
        or not math.isclose(
            float(scales[0]),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        or replay_contract.get("method")
        != "simultaneous_grouped_clones"
        or replay_contract.get("modified_action_channels")
        != "receiver_translation_xyz_only"
        or replay_contract.get("risk_role")
        != "postbranch_retrospective_stratification_only"
    ):
        raise ValueError("replay contract drifted")
    group_count = num_envs // replicas
    environment_index = _require_tensor(
        payload,
        "environment_index",
        length=num_envs,
    ).long()
    group_index = _require_tensor(
        payload,
        "group_index",
        length=num_envs,
    ).long()
    candidate_index = _require_tensor(
        payload,
        "candidate_index",
        length=num_envs,
    ).long()
    assigned_scale = _require_tensor(
        payload,
        "assigned_scale",
        length=num_envs,
    ).float()
    expected_environment = torch.arange(num_envs)
    expected_candidate = expected_environment % replicas
    expected_group = expected_environment // replicas
    expected_scale = torch.tensor(scales).float()[expected_candidate]
    if (
        not torch.equal(environment_index, expected_environment)
        or not torch.equal(group_index, expected_group)
        or not torch.equal(candidate_index, expected_candidate)
        or not torch.allclose(
            assigned_scale,
            expected_scale,
            rtol=0.0,
            atol=1.0e-7,
        )
    ):
        raise ValueError("group or candidate assignment drifted")
    resolved = _require_tensor(
        payload,
        "first_episode_resolved",
        length=num_envs,
    ).bool()
    activation_seen = _require_tensor(
        payload,
        "activation_seen",
        length=num_envs,
    ).bool()
    activation_frame = _require_tensor(
        payload,
        "activation_frame",
        length=num_envs,
    ).long()
    context = _require_tensor(
        payload,
        "activation_context",
        length=num_envs,
    ).float()
    base_action = _require_tensor(
        payload,
        "base_action_at_activation",
        length=num_envs,
    ).float()
    scaled_action = _require_tensor(
        payload,
        "scaled_action_at_activation",
        length=num_envs,
    ).float()
    correction = _require_tensor(
        payload,
        "receiver_candidate_correction",
        length=num_envs,
    ).float()
    full_success = _require_tensor(
        payload,
        "full_success",
        length=num_envs,
    ).bool()
    maximum_phase = _require_tensor(
        payload,
        "maximum_phase",
        length=num_envs,
    ).long()
    terminal_flags = _require_tensor(
        payload,
        "termination_flags",
        length=num_envs,
    ).bool()
    safety = _require_tensor(
        payload,
        "receiver_safety_failure",
        length=num_envs,
    ).bool()
    risk_observed = _require_tensor(
        payload,
        "postbranch_preprobe_risk_observed",
        length=num_envs,
    ).bool()
    risk = _require_tensor(
        payload,
        "postbranch_predicted_preprobe_risk",
        length=num_envs,
    ).float()
    active_frames = _require_tensor(
        payload,
        "active_frames",
        length=num_envs,
    ).long()
    modified_frames = _require_tensor(
        payload,
        "modified_frames",
        length=num_envs,
    ).long()
    minimum_multiplier = _require_tensor(
        payload,
        "minimum_multiplier",
        length=num_envs,
    ).float()
    termination_names = payload.get("termination_names")
    if (
        not isinstance(termination_names, list)
        or terminal_flags.ndim != 2
        or terminal_flags.shape[1] != len(termination_names)
        or context.shape != (num_envs, 98)
        or base_action.shape != (num_envs, 14)
        or scaled_action.shape != (num_envs, 14)
        or correction.shape != (num_envs, 6)
    ):
        raise ValueError("trajectory tensor shape contract drifted")

    grouped_seen = activation_seen.reshape(group_count, replicas)
    activation_consistent = bool(
        torch.all(grouped_seen == grouped_seen[:, :1]).item()
    )
    active_groups = grouped_seen.all(dim=1)
    inactive_groups = ~grouped_seen.any(dim=1)
    grouped_frame = activation_frame.reshape(group_count, replicas)
    frame_consistent = bool(
        torch.all(
            grouped_frame[active_groups]
            == grouped_frame[active_groups, :1]
        ).item()
    )
    context_delta = _maximum_replica_delta(
        context.reshape(group_count, replicas, 98),
        active_groups,
    )
    action_delta = _maximum_replica_delta(
        base_action.reshape(group_count, replicas, 14),
        active_groups,
    )
    correction_delta = _maximum_replica_delta(
        correction.reshape(group_count, replicas, 6),
        active_groups,
    )
    grouped_success = full_success.reshape(group_count, replicas)
    grouped_phase = maximum_phase.reshape(group_count, replicas)
    grouped_terminal = terminal_flags.reshape(
        group_count,
        replicas,
        len(termination_names),
    )
    grouped_safety = safety.reshape(group_count, replicas)
    inactive_outcome_parity = bool(
        torch.all(
            grouped_success[inactive_groups]
            == grouped_success[inactive_groups, :1]
        ).item()
        and torch.all(
            grouped_phase[inactive_groups]
            == grouped_phase[inactive_groups, :1]
        ).item()
        and torch.all(
            grouped_terminal[inactive_groups]
            == grouped_terminal[inactive_groups, :1]
        ).item()
        and torch.all(
            grouped_safety[inactive_groups]
            == grouped_safety[inactive_groups, :1]
        ).item()
    )
    control_environments = candidate_index == 0
    control_noop_action_delta = (
        float(
            (
                scaled_action[
                    control_environments & activation_seen
                ]
                - base_action[control_environments & activation_seen]
            )
            .abs()
            .max()
            .item()
        )
        if bool((control_environments & activation_seen).any())
        else 0.0
    )
    prebranch_parity_passed = (
        bool(resolved.all().item())
        and activation_consistent
        and frame_consistent
        and context_delta <= context_atol
        and action_delta <= action_atol
        and correction_delta <= context_atol
        and inactive_outcome_parity
        and control_noop_action_delta <= action_atol
    )
    all_noop = all(
        math.isclose(
            float(scale),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        for scale in scales
    )
    noop_outcome_parity = None
    if all_noop:
        noop_outcome_parity = bool(
            torch.all(
                grouped_success == grouped_success[:, :1]
            ).item()
            and torch.all(
                grouped_phase == grouped_phase[:, :1]
            ).item()
            and torch.all(
                grouped_terminal == grouped_terminal[:, :1]
            ).item()
            and torch.all(
                grouped_safety == grouped_safety[:, :1]
            ).item()
            and torch.allclose(
                scaled_action[activation_seen],
                base_action[activation_seen],
                rtol=0.0,
                atol=action_atol,
            )
            and int(modified_frames.sum().item()) == 0
            and torch.allclose(
                minimum_multiplier,
                torch.ones_like(minimum_multiplier),
                rtol=0.0,
                atol=action_atol,
            )
        )
    record = {
        "path": path,
        "payload": payload,
        "scales": [float(value) for value in scales],
        "replicas": replicas,
        "groups": group_count,
        "success": grouped_success,
        "phase": grouped_phase,
        "terminal": grouped_terminal,
        "safety": grouped_safety,
        "risk_observed": risk_observed.reshape(group_count, replicas),
        "risk": risk.reshape(group_count, replicas),
        "active_groups": active_groups,
    }
    summary = {
        "path": str(path),
        "sha256": _sha256(path),
        "dataset_id": payload.get("dataset_id"),
        "source_revision": payload.get("source_revision"),
        "seed": payload.get("seed"),
        "seed_stream_offset": payload.get("seed_stream_offset"),
        "groups": group_count,
        "replicas": replicas,
        "candidate_scales": record["scales"],
        "resolved_environments": int(resolved.sum().item()),
        "active_groups": int(active_groups.sum().item()),
        "inactive_groups": int(inactive_groups.sum().item()),
        "prebranch_parity": {
            "passed": prebranch_parity_passed,
            "activation_consistent": activation_consistent,
            "activation_frame_consistent": frame_consistent,
            "maximum_context_delta": context_delta,
            "maximum_base_action_delta": action_delta,
            "maximum_candidate_correction_delta": correction_delta,
            "inactive_group_outcome_parity": inactive_outcome_parity,
            "control_noop_action_delta": control_noop_action_delta,
            "context_atol": context_atol,
            "action_atol": action_atol,
        },
        "all_noop": all_noop,
        "noop_outcome_parity": noop_outcome_parity,
        "candidate_outcomes": [
            {
                "candidate_index": index,
                "scale": record["scales"][index],
                "successes": int(grouped_success[:, index].sum().item()),
                "safety_failures": int(
                    grouped_safety[:, index].sum().item()
                ),
                "termination_counts": _termination_counts(
                    termination_names,
                    grouped_terminal[:, index],
                ),
                "active_frame_mean": float(
                    active_frames.reshape(
                        group_count,
                        replicas,
                    )[:, index]
                    .float()
                    .mean()
                    .item()
                ),
                "modified_frame_mean": float(
                    modified_frames.reshape(
                        group_count,
                        replicas,
                    )[:, index]
                    .float()
                    .mean()
                    .item()
                ),
            }
            for index in range(replicas)
        ],
    }
    return record, summary


def _candidate_seed_result(
    record: dict[str, Any],
    candidate_index: int,
) -> dict[str, Any]:
    control = record["success"][:, 0]
    candidate = record["success"][:, candidate_index]
    control_safety = record["safety"][:, 0]
    candidate_safety = record["safety"][:, candidate_index]
    wins = int((candidate & ~control).sum().item())
    losses = int((~candidate & control).sum().item())
    return {
        "seed": record["payload"].get("seed"),
        "seed_stream_offset": record["payload"].get(
            "seed_stream_offset"
        ),
        "groups": record["groups"],
        "control_successes": int(control.sum().item()),
        "candidate_successes": int(candidate.sum().item()),
        "wins": wins,
        "losses": losses,
        "paired_net_successes": wins - losses,
        "control_safety_failures": int(control_safety.sum().item()),
        "candidate_safety_failures": int(candidate_safety.sum().item()),
        "safety_delta": int(
            candidate_safety.sum().item()
            - control_safety.sum().item()
        ),
        "risk_quartiles_exploratory": _risk_quartile_effects(
            record,
            candidate_index,
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--context-atol", type=float, default=1.0e-6)
    parser.add_argument("--action-atol", type=float, default=1.0e-7)
    parser.add_argument(
        "--paired-significance-threshold",
        type=float,
        default=0.05,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if (
        args.context_atol < 0.0
        or args.action_atol < 0.0
        or not 0.0 < args.paired_significance_threshold <= 0.05
    ):
        raise SystemExit("invalid evaluator tolerance or significance gate")
    dataset_paths = [
        Path(value).expanduser().resolve()
        for value in args.dataset
    ]
    records = []
    summaries = []
    try:
        for path in dataset_paths:
            if not path.is_file():
                raise ValueError(f"dataset not found: {path}")
            record, summary = _evaluate_dataset(
                path,
                context_atol=args.context_atol,
                action_atol=args.action_atol,
            )
            records.append(record)
            summaries.append(summary)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"error: {error}")
        return 2
    reference = records[0]
    identity_fields = (
        "base_checkpoint_sha256",
        "receiver_candidate_checkpoint_sha256",
        "preprobe_risk_checkpoint_sha256",
        "task",
    )
    for record in records[1:]:
        if record["scales"] != reference["scales"]:
            print("error: candidate scale portfolios differ across datasets")
            return 2
        for field in identity_fields:
            if (
                record["payload"].get(field)
                != reference["payload"].get(field)
            ):
                print(f"error: {field} differs across datasets")
                return 2
    parity_passed = all(
        summary["prebranch_parity"]["passed"]
        for summary in summaries
    )
    all_noop = all(summary["all_noop"] for summary in summaries)
    candidate_results = []
    winning_candidate = None
    if all_noop:
        noop_parity_passed = parity_passed and all(
            summary["noop_outcome_parity"]
            for summary in summaries
        )
        decision = (
            "parity_qualified"
            if noop_parity_passed
            else "replay_invalid"
        )
        gate = {
            "passed": noop_parity_passed,
            "decision": decision,
            "next_action": (
                "run_bounded_trajectory_screen"
                if noop_parity_passed
                else "stop_and_fix_replay"
            ),
        }
    else:
        if any(summary["all_noop"] for summary in summaries):
            print("error: no-op and intervention datasets cannot be mixed")
            return 2
        for candidate_index in range(1, reference["replicas"]):
            per_seed = [
                _candidate_seed_result(record, candidate_index)
                for record in records
            ]
            wins = sum(item["wins"] for item in per_seed)
            losses = sum(item["losses"] for item in per_seed)
            groups = sum(item["groups"] for item in per_seed)
            safety_delta = sum(item["safety_delta"] for item in per_seed)
            probability = _exact_one_sided_sign_probability(
                wins,
                losses,
            )
            eligible = (
                parity_passed
                and wins > losses
                and all(
                    item["paired_net_successes"] >= 0
                    for item in per_seed
                )
                and all(item["safety_delta"] <= 0 for item in per_seed)
                and safety_delta <= 0
                and probability
                <= args.paired_significance_threshold
            )
            candidate_results.append(
                {
                    "candidate_index": candidate_index,
                    "scale": reference["scales"][candidate_index],
                    "groups": groups,
                    "wins": wins,
                    "losses": losses,
                    "paired_net_successes": wins - losses,
                    "intent_to_treat_uplift": (
                        (wins - losses) / groups
                    ),
                    "safety_delta": safety_delta,
                    "one_sided_exact_sign_probability": probability,
                    "per_seed": per_seed,
                    "eligible_for_learning": eligible,
                }
            )
        eligible_candidates = [
            item
            for item in candidate_results
            if item["eligible_for_learning"]
        ]
        if eligible_candidates:
            winning_candidate = sorted(
                eligible_candidates,
                key=lambda item: (
                    -item["paired_net_successes"],
                    item["safety_delta"],
                    -item["scale"],
                ),
            )[0]
        gate = {
            "passed": winning_candidate is not None,
            "decision": (
                "paired_intervention_family_positive"
                if winning_candidate is not None
                else (
                    "replay_invalid"
                    if not parity_passed
                    else "paired_intervention_family_rejected"
                )
            ),
            "winning_candidate": winning_candidate,
            "next_action": (
                "replicate_winner_then_advantage_weighted_bc"
                if winning_candidate is not None
                else (
                    "stop_and_fix_replay"
                    if not parity_passed
                    else "do_not_train_from_this_intervention_family"
                )
            ),
        }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "datasets": summaries,
        "source_contract": {
            field: reference["payload"].get(field)
            for field in identity_fields
        },
        "risk_usage": (
            "Control-branch post-acquisition scores are exploratory "
            "heterogeneity labels only. They did not select actions and "
            "are not part of the causal gate."
        ),
        "parity_passed": parity_passed,
        "candidate_results": candidate_results,
        "gate": gate,
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(f"[DrAnmar] Receiver trajectory evaluation: {output_path}")
    print(
        "[DrAnmar] Decision: "
        f"{gate['decision']} (passed={gate['passed']})"
    )
    return 2 if gate["decision"] == "replay_invalid" else 0


if __name__ == "__main__":
    raise SystemExit(main())
