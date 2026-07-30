#!/usr/bin/env python3
"""Gate common-random-number receiver trajectory replays.

Each candidate run is paired with an all-no-op control run at the same source
revision, runtime seed, environment count, and environment index.  This
replays the complete PhysX evolution, including solver history that IsaacLab
scene snapshots do not serialize.  A candidate family is learnable only when
the pre-branch tensors replay exactly, the scale-1 lane has exact terminal
parity, and an intervention has a positive, seed-consistent, safety-neutral
paired effect with one-sided exact sign probability at or below 0.05.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


SCHEMA_VERSION = "dranmar-receiver-approach-trajectory-replay-1.3"
REPORT_SCHEMA_VERSION = (
    "dranmar-receiver-approach-trajectory-paired-evaluation-1.0"
)
IDENTITY_FIELDS = (
    "source_revision",
    "task",
    "runtime_seed",
    "num_envs",
    "num_frames",
    "base_checkpoint_sha256",
    "receiver_candidate_checkpoint_sha256",
    "preprobe_risk_checkpoint_sha256",
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


def _maximum_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0:
        return 0.0
    return float((left - right).abs().max().item())


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


def _load_dataset(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: dataset root must be a dictionary")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unsupported schema {payload.get('schema_version')!r}"
        )
    replay_contract = payload.get("replay_contract")
    if (
        not isinstance(replay_contract, dict)
        or replay_contract.get("method")
        != "same_environment_index_common_random_numbers"
        or replay_contract.get("modified_action_channels")
        != "receiver_translation_xyz_only"
        or replay_contract.get("risk_role")
        != "postbranch_retrospective_stratification_only"
        or replay_contract.get("barrier_contract")
        != (
            "hold_ready_receiver_translation_then_release_"
            "all_qualified_environments_simultaneously"
        )
        or not isinstance(
            replay_contract.get("barrier_release_frame"),
            int,
        )
        or replay_contract["barrier_release_frame"] <= 0
        or replay_contract.get("pairing_key")
        != [
            "source_revision",
            "task",
            "runtime_seed",
            "num_envs",
            "environment_index",
        ]
    ):
        raise ValueError(f"{path}: replay contract drifted")
    scales = replay_contract.get("candidate_scales")
    replicas = replay_contract.get("candidate_lanes")
    num_envs = payload.get("num_envs")
    if (
        not isinstance(scales, list)
        or not isinstance(replicas, int)
        or replicas != len(scales)
        or replicas < 1
        or not isinstance(num_envs, int)
        or num_envs <= 0
        or num_envs % replicas
        or not all(
            isinstance(scale, (int, float))
            and math.isfinite(float(scale))
            and 0.0 < float(scale) <= 1.0
            for scale in scales
        )
        or (
            replicas > 1
            and not math.isclose(
                float(scales[0]),
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        )
    ):
        raise ValueError(f"{path}: candidate lane contract drifted")
    grouped_allocation = replay_contract.get(
        "grouped_initial_condition_allocation"
    )
    if grouped_allocation is not (replicas > 1):
        raise ValueError(
            f"{path}: grouped initial-condition contract drifted"
        )
    environment_index = _require_tensor(
        payload,
        "environment_index",
        length=num_envs,
    ).long()
    candidate_index = _require_tensor(
        payload,
        "candidate_index",
        length=num_envs,
    ).long()
    group_index = _require_tensor(
        payload,
        "group_index",
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
        or not torch.equal(candidate_index, expected_candidate)
        or not torch.equal(group_index, expected_group)
        or not torch.allclose(
            assigned_scale,
            expected_scale,
            rtol=0.0,
            atol=1.0e-7,
        )
    ):
        raise ValueError(f"{path}: environment pairing assignment drifted")
    termination_names = payload.get("termination_names")
    if not isinstance(termination_names, list):
        raise ValueError(f"{path}: termination names are missing")
    tensors = {
        "resolved": _require_tensor(
            payload,
            "first_episode_resolved",
            length=num_envs,
        ).bool(),
        "activation_seen": _require_tensor(
            payload,
            "activation_seen",
            length=num_envs,
        ).bool(),
        "activation_frame": _require_tensor(
            payload,
            "activation_frame",
            length=num_envs,
        ).long(),
        "context": _require_tensor(
            payload,
            "activation_context",
            length=num_envs,
        ).float(),
        "base_action": _require_tensor(
            payload,
            "base_action_at_activation",
            length=num_envs,
        ).float(),
        "scaled_action": _require_tensor(
            payload,
            "scaled_action_at_activation",
            length=num_envs,
        ).float(),
        "distance": _require_tensor(
            payload,
            "distance_at_activation_m",
            length=num_envs,
        ).float(),
        "correction": _require_tensor(
            payload,
            "receiver_candidate_correction",
            length=num_envs,
        ).float(),
        "success": _require_tensor(
            payload,
            "full_success",
            length=num_envs,
        ).bool(),
        "phase": _require_tensor(
            payload,
            "maximum_phase",
            length=num_envs,
        ).long(),
        "terminal": _require_tensor(
            payload,
            "termination_flags",
            length=num_envs,
        ).bool(),
        "safety": _require_tensor(
            payload,
            "receiver_safety_failure",
            length=num_envs,
        ).bool(),
        "risk_observed": _require_tensor(
            payload,
            "postbranch_preprobe_risk_observed",
            length=num_envs,
        ).bool(),
        "risk": _require_tensor(
            payload,
            "postbranch_predicted_preprobe_risk",
            length=num_envs,
        ).float(),
        "active_frames": _require_tensor(
            payload,
            "active_frames",
            length=num_envs,
        ).long(),
        "modified_frames": _require_tensor(
            payload,
            "modified_frames",
            length=num_envs,
        ).long(),
        "minimum_multiplier": _require_tensor(
            payload,
            "minimum_multiplier",
            length=num_envs,
        ).float(),
        "barrier_hold_frames": _require_tensor(
            payload,
            "barrier_hold_frames",
            length=num_envs,
        ).long(),
    }
    if (
        tensors["context"].shape != (num_envs, 98)
        or tensors["base_action"].shape != (num_envs, 14)
        or tensors["scaled_action"].shape != (num_envs, 14)
        or tensors["correction"].shape != (num_envs, 6)
        or tensors["terminal"].shape
        != (num_envs, len(termination_names))
    ):
        raise ValueError(f"{path}: trajectory tensor shape drifted")
    return {
        "path": path,
        "sha256": _sha256(path),
        "payload": payload,
        "contract": replay_contract,
        "scales": [float(value) for value in scales],
        "replicas": replicas,
        "candidate_index": candidate_index,
        "termination_names": termination_names,
        **tensors,
    }


def _same_outcomes(
    control: dict[str, Any],
    candidate: dict[str, Any],
    mask: torch.Tensor,
) -> bool:
    return bool(
        torch.equal(
            control["success"][mask],
            candidate["success"][mask],
        )
        and torch.equal(
            control["phase"][mask],
            candidate["phase"][mask],
        )
        and torch.equal(
            control["terminal"][mask],
            candidate["terminal"][mask],
        )
        and torch.equal(
            control["safety"][mask],
            candidate["safety"][mask],
        )
    )


def _risk_quartile_effects(
    control: dict[str, Any],
    candidate: dict[str, Any],
    mask: torch.Tensor,
) -> list[dict[str, Any]]:
    observed = mask & control["risk_observed"]
    if int(observed.sum().item()) < 8:
        return []
    risk = control["risk"]
    boundaries = torch.quantile(
        risk[observed],
        torch.tensor((0.25, 0.5, 0.75)),
    )
    bucket = torch.bucketize(risk, boundaries)
    output = []
    for quartile in range(4):
        selected = observed & (bucket == quartile)
        control_success = control["success"][selected]
        candidate_success = candidate["success"][selected]
        wins = int((candidate_success & ~control_success).sum().item())
        losses = int((~candidate_success & control_success).sum().item())
        output.append(
            {
                "quartile": quartile + 1,
                "samples": int(selected.sum().item()),
                "control_successes": int(control_success.sum().item()),
                "candidate_successes": int(
                    candidate_success.sum().item()
                ),
                "paired_net_successes": wins - losses,
            }
        )
    return output


def _candidate_result(
    control: dict[str, Any],
    candidate: dict[str, Any],
    lane: int,
) -> dict[str, Any]:
    mask = (
        (candidate["candidate_index"] == lane)
        & control["activation_seen"]
        & candidate["activation_seen"]
    )
    control_success = control["success"][mask]
    candidate_success = candidate["success"][mask]
    control_safety = control["safety"][mask]
    candidate_safety = candidate["safety"][mask]
    wins = int((candidate_success & ~control_success).sum().item())
    losses = int((~candidate_success & control_success).sum().item())
    return {
        "seed": candidate["payload"].get("seed"),
        "seed_stream_offset": candidate["payload"].get(
            "seed_stream_offset"
        ),
        "runtime_seed": candidate["payload"].get("runtime_seed"),
        "samples": int(mask.sum().item()),
        "control_successes": int(control_success.sum().item()),
        "candidate_successes": int(candidate_success.sum().item()),
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
            control,
            candidate,
            mask,
        ),
    }


def _evaluate_pair(
    control: dict[str, Any],
    candidate: dict[str, Any],
    *,
    context_atol: float,
    action_atol: float,
) -> dict[str, Any]:
    for field in IDENTITY_FIELDS:
        if (
            control["payload"].get(field)
            != candidate["payload"].get(field)
        ):
            raise ValueError(
                f"paired datasets differ on identity field {field}"
            )
    if control["replicas"] != candidate["replicas"]:
        raise ValueError("paired datasets use different candidate lanes")
    for field in (
        "start_distance_m",
        "end_distance_m",
        "barrier_release_frame",
        "barrier_contract",
        "profile",
    ):
        if control["contract"].get(field) != candidate["contract"].get(
            field
        ):
            raise ValueError(
                f"paired replay contracts differ on {field}"
            )
    if not all(
        math.isclose(
            scale,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        for scale in control["scales"]
    ):
        raise ValueError("control dataset must use all no-op scales")
    if not torch.equal(
        control["candidate_index"],
        candidate["candidate_index"],
    ):
        raise ValueError("paired candidate lane assignment differs")
    active = control["activation_seen"] & candidate["activation_seen"]
    activation_parity = torch.equal(
        control["activation_seen"],
        candidate["activation_seen"],
    )
    activation_frame_parity = torch.equal(
        control["activation_frame"],
        candidate["activation_frame"],
    )
    barrier_hold_parity = torch.equal(
        control["barrier_hold_frames"],
        candidate["barrier_hold_frames"],
    )
    context_delta = _maximum_delta(
        control["context"][active],
        candidate["context"][active],
    )
    action_delta = _maximum_delta(
        control["base_action"][active],
        candidate["base_action"][active],
    )
    correction_delta = _maximum_delta(
        control["correction"][active],
        candidate["correction"][active],
    )
    distance_delta = _maximum_delta(
        control["distance"][active],
        candidate["distance"][active],
    )
    candidate_noop_mask = (
        torch.tensor(candidate["scales"])[
            candidate["candidate_index"]
        ]
        == 1.0
    )
    candidate_noop_action_delta = _maximum_delta(
        candidate["scaled_action"][
            active & candidate_noop_mask
        ],
        candidate["base_action"][active & candidate_noop_mask],
    )
    resolved = bool(
        control["resolved"].all().item()
        and candidate["resolved"].all().item()
    )
    prebranch_parity = (
        resolved
        and activation_parity
        and activation_frame_parity
        and barrier_hold_parity
        and context_delta <= context_atol
        and action_delta <= action_atol
        and correction_delta <= context_atol
        and distance_delta <= context_atol
        and candidate_noop_action_delta <= action_atol
    )
    all_candidate_noop = all(
        math.isclose(
            scale,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        for scale in candidate["scales"]
    )
    outcome_parity_mask = (
        torch.ones_like(candidate["candidate_index"], dtype=torch.bool)
        if all_candidate_noop
        else candidate_noop_mask
    )
    noop_outcome_parity = _same_outcomes(
        control,
        candidate,
        outcome_parity_mask,
    )
    return {
        "control_path": str(control["path"]),
        "control_sha256": control["sha256"],
        "candidate_path": str(candidate["path"]),
        "candidate_sha256": candidate["sha256"],
        "source_revision": candidate["payload"].get("source_revision"),
        "runtime_seed": candidate["payload"].get("runtime_seed"),
        "candidate_scales": candidate["scales"],
        "environments": candidate["payload"]["num_envs"],
        "activated_environments": int(active.sum().item()),
        "barrier_release_frame": candidate["contract"][
            "barrier_release_frame"
        ],
        "all_candidate_noop": all_candidate_noop,
        "prebranch_parity": {
            "passed": prebranch_parity,
            "all_first_episodes_resolved": resolved,
            "activation_mask_exact": activation_parity,
            "activation_frame_exact": activation_frame_parity,
            "barrier_hold_frames_exact": barrier_hold_parity,
            "maximum_context_delta": context_delta,
            "maximum_base_action_delta": action_delta,
            "maximum_candidate_correction_delta": correction_delta,
            "maximum_activation_distance_delta": distance_delta,
            "candidate_noop_action_delta": (
                candidate_noop_action_delta
            ),
            "context_atol": context_atol,
            "action_atol": action_atol,
        },
        "noop_lane_outcome_parity": noop_outcome_parity,
        "candidate_results": [
            _candidate_result(control, candidate, lane)
            for lane in range(1, candidate["replicas"])
        ],
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
        "--candidate-dataset",
        action="append",
        required=True,
    )
    parser.add_argument("--context-atol", type=float, default=0.0)
    parser.add_argument("--action-atol", type=float, default=0.0)
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
        or len(args.control_dataset) != len(args.candidate_dataset)
    ):
        print("error: invalid paired evaluator arguments")
        return 2
    try:
        controls = [
            _load_dataset(Path(value).expanduser().resolve())
            for value in args.control_dataset
        ]
        candidates = [
            _load_dataset(Path(value).expanduser().resolve())
            for value in args.candidate_dataset
        ]
        pairs = [
            _evaluate_pair(
                control,
                candidate,
                context_atol=args.context_atol,
                action_atol=args.action_atol,
            )
            for control, candidate in zip(
                controls,
                candidates,
                strict=True,
            )
        ]
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"error: {error}")
        return 2
    reference_scales = candidates[0]["scales"]
    for candidate in candidates[1:]:
        if candidate["scales"] != reference_scales:
            print("error: candidate scale portfolios differ across seeds")
            return 2
        for field in (
            "source_revision",
            "task",
            "num_envs",
            "num_frames",
            "base_checkpoint_sha256",
            "receiver_candidate_checkpoint_sha256",
            "preprobe_risk_checkpoint_sha256",
        ):
            if (
                candidate["payload"].get(field)
                != candidates[0]["payload"].get(field)
            ):
                print(f"error: candidate datasets differ on {field}")
                return 2
    replay_parity = all(
        pair["prebranch_parity"]["passed"]
        and pair["noop_lane_outcome_parity"]
        for pair in pairs
    )
    all_noop = all(pair["all_candidate_noop"] for pair in pairs)
    candidate_results = []
    winning_candidate = None
    if all_noop:
        gate = {
            "passed": replay_parity,
            "decision": (
                "parity_qualified"
                if replay_parity
                else "replay_invalid"
            ),
            "next_action": (
                "run_bounded_trajectory_screen"
                if replay_parity
                else "stop_and_fix_replay"
            ),
        }
    else:
        if any(pair["all_candidate_noop"] for pair in pairs):
            print("error: no-op and intervention pairs cannot be mixed")
            return 2
        for lane in range(1, candidates[0]["replicas"]):
            per_seed = [
                pair["candidate_results"][lane - 1]
                for pair in pairs
            ]
            wins = sum(item["wins"] for item in per_seed)
            losses = sum(item["losses"] for item in per_seed)
            samples = sum(item["samples"] for item in per_seed)
            safety_delta = sum(item["safety_delta"] for item in per_seed)
            probability = _exact_one_sided_sign_probability(
                wins,
                losses,
            )
            eligible = (
                replay_parity
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
                    "candidate_index": lane,
                    "scale": reference_scales[lane],
                    "samples": samples,
                    "wins": wins,
                    "losses": losses,
                    "paired_net_successes": wins - losses,
                    "intent_to_treat_uplift": (
                        (wins - losses) / samples
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
                    if not replay_parity
                    else "paired_intervention_family_rejected"
                )
            ),
            "winning_candidate": winning_candidate,
            "next_action": (
                "replicate_winner_then_advantage_weighted_bc"
                if winning_candidate is not None
                else (
                    "stop_and_fix_replay"
                    if not replay_parity
                    else "do_not_train_from_this_intervention_family"
                )
            ),
        }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "pairs": pairs,
        "source_contract": {
            field: candidates[0]["payload"].get(field)
            for field in (
                "source_revision",
                "task",
                "num_envs",
                "num_frames",
                "base_checkpoint_sha256",
                "receiver_candidate_checkpoint_sha256",
                "preprobe_risk_checkpoint_sha256",
            )
        },
        "risk_usage": (
            "Control-run post-acquisition scores are exploratory "
            "heterogeneity labels only. They did not select actions and "
            "are not part of the causal gate."
        ),
        "replay_parity_passed": replay_parity,
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
