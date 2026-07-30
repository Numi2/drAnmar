#!/usr/bin/env python3
"""Admit isolated handover demonstrations and train the full-action successor."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional


TRACE_SCHEMA = "dranmar-handover-teacher-trace-1.0"
DATASET_SCHEMA = "dranmar-handover-successor-dataset-1.0"
RECEIPT_SCHEMA = "dranmar-handover-teacher-receipt-1.0"
ACTION_SCHEDULE_SCHEMA = "dranmar-handover-teacher-action-schedule-1.0"
BASELINE_LABEL_SOURCE = "frozen_baseline_success_distillation"
TEACHER_LABEL_SOURCE = "independent_teacher_rescue"
ALLOWED_LABEL_SOURCES = {
    BASELINE_LABEL_SOURCE,
    TEACHER_LABEL_SOURCE,
}
QUALIFICATION_SEEDS = {17, 2361, 4099}
ALLOWED_TEACHERS = {
    "constrained_trajectory_optimizer",
    "clinician_teleoperation",
}
SAFETY_TERMS = (
    "excessive_object_force",
    "needle_dropped_after_pickup",
    "object_dropping",
    "premature_giver_release",
    "protected_surface_force",
    "receiver_retention_lost",
)
OBSERVATION_DIM = 98
ACTION_DIM = 14
PHASE_SLICE = slice(77, 82)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(payload: dict[str, Any], output: Path) -> None:
    if output.exists():
        raise ValueError(f"refusing to overwrite immutable artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json_save(payload: dict[str, Any], output: Path) -> None:
    if output.exists():
        raise ValueError(f"refusing to overwrite immutable artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def create_optimizer_receipt(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.seed) in QUALIFICATION_SEEDS:
        raise ValueError("qualification seed cannot produce a teacher receipt")
    values = [
        float(value.strip())
        for value in args.receiver_correction.split(",")
    ]
    if len(values) != 6:
        raise ValueError("receiver correction requires dx,dy,dz,rx,ry,rz")
    position_cap = float(args.position_cap_m)
    orientation_cap = float(args.orientation_cap_deg)
    if position_cap <= 0.0 or orientation_cap <= 0.0:
        raise ValueError("optimizer correction caps must be positive")
    if any(abs(value) > position_cap + 1.0e-12 for value in values[:3]):
        raise ValueError("optimizer receiver position exceeds its cap")
    if any(abs(value) > orientation_cap + 1.0e-12 for value in values[3:]):
        raise ValueError("optimizer receiver orientation exceeds its cap")

    proposal_paths = [
        Path(value).expanduser().resolve() for value in args.proposal_source
    ]
    for path in proposal_paths:
        if not path.is_file():
            raise ValueError(f"optimizer proposal source does not exist: {path}")
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "task": args.task,
        "seed": int(args.seed),
        "pair_id": args.pair_id,
        "teacher_kind": "constrained_trajectory_optimizer",
        "optimizer": {
            "algorithm": "risk_guided_bounded_receiver_correction_search",
            "objective": "terminal_success_without_any_safety_event",
            "constraints": {
                "single_environment_final_replay": True,
                "position_cap_m": position_cap,
                "orientation_cap_deg": orientation_cap,
                "environment_action_bound": 1.0,
                "proposal_sources_are_not_training_labels": True,
            },
            "selected_parameters": {
                "receiver_recovery_fixed_correction": values,
            },
            "proposal_sources": [
                {
                    "path": str(path),
                    "sha256": _sha256(path),
                }
                for path in proposal_paths
            ],
        },
    }
    output = Path(args.output).expanduser().resolve()
    _atomic_json_save(receipt, output)
    return {
        "schema_version": RECEIPT_SCHEMA,
        "output": str(output),
        "sha256": _sha256(output),
        "pair_id": args.pair_id,
        "seed": int(args.seed),
        "receiver_recovery_fixed_correction": values,
    }


def _load_trace(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"teacher trace is not a mapping: {path}")
    if payload.get("schema_version") != TRACE_SCHEMA:
        raise ValueError(f"unsupported teacher trace schema: {path}")
    return payload


def _validate_trace(path: Path, trace: dict[str, Any]) -> None:
    if trace.get("task") != "DrAnmar-Handover-Needle-Dual-PSM-IK-Rel-v0":
        raise ValueError(f"trace is not the qualified needle handover task: {path}")
    if int(trace.get("num_envs", -1)) != 1:
        raise ValueError(f"teacher traces must use exactly one environment: {path}")
    if int(trace.get("seed", -1)) in QUALIFICATION_SEEDS:
        raise ValueError(f"qualification seed cannot become training data: {path}")
    if not trace.get("pair_id"):
        raise ValueError(f"trace lacks a stable pair id: {path}")
    if not trace.get("terminal", {}).get("complete"):
        raise ValueError(f"trace does not contain one complete episode: {path}")

    observations = trace.get("observations")
    actions = trace.get("actions")
    rewards = trace.get("rewards")
    safety_events = trace.get("safety_events")
    phases = trace.get("phases")
    tensors = (observations, actions, rewards, safety_events, phases)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise ValueError(f"trace tensors are incomplete: {path}")
    frame_count = int(observations.shape[0])
    if tuple(observations.shape) != (frame_count, OBSERVATION_DIM):
        raise ValueError(f"trace observation contract drifted: {path}")
    if tuple(actions.shape) != (frame_count, ACTION_DIM):
        raise ValueError(f"trace action contract drifted: {path}")
    if tuple(rewards.shape) != (frame_count,):
        raise ValueError(f"trace reward contract drifted: {path}")
    if tuple(phases.shape) != (frame_count,):
        raise ValueError(f"trace phase contract drifted: {path}")
    if tuple(safety_events.shape) != (frame_count, len(SAFETY_TERMS)):
        raise ValueError(f"trace safety-event contract drifted: {path}")
    if list(trace.get("safety_term_names", ())) != list(SAFETY_TERMS):
        raise ValueError(f"trace safety terms drifted: {path}")
    if int(trace["terminal"].get("frame_count", -1)) != frame_count:
        raise ValueError(f"terminal frame count does not match trace tensors: {path}")
    if frame_count < 2:
        raise ValueError(f"teacher trace is too short: {path}")
    if not torch.isfinite(observations).all():
        raise ValueError(f"teacher observations contain non-finite values: {path}")
    if not torch.isfinite(actions).all():
        raise ValueError(f"teacher actions contain non-finite values: {path}")
    if not torch.isfinite(rewards).all():
        raise ValueError(f"teacher rewards contain non-finite values: {path}")
    if bool((actions.abs() > 1.000001).any()):
        raise ValueError(f"teacher actions exceed the environment contract: {path}")
    observed_phases = torch.argmax(observations[:, PHASE_SLICE], dim=-1)
    if not torch.equal(phases.long(), observed_phases.long()):
        raise ValueError(f"trace phase labels disagree with observations: {path}")

    runtime = trace.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError(f"trace lacks runtime provenance: {path}")
    source = runtime.get("source")
    if (
        not isinstance(source, dict)
        or not source.get("dranmar_revision")
        or not source.get("asset_revision")
        or not source.get("asset_root")
    ):
        raise ValueError(f"trace lacks source revisions: {path}")
    policy = trace.get("policy")
    if not isinstance(policy, dict) or not policy.get("base_checkpoint_sha256"):
        raise ValueError(f"trace lacks policy provenance: {path}")


def propose_retention_schedule(args: argparse.Namespace) -> dict[str, Any]:
    """Extend the last proven centering command across the custody transition."""

    control_path = Path(args.control).expanduser().resolve()
    control = _load_trace(control_path)
    _validate_trace(control_path, control)
    if control.get("role") != "control":
        raise ValueError("retention schedule requires a frozen control trace")
    if control["terminal"]["outcome"] != "receiver_retention_lost":
        raise ValueError("retention schedule requires a receiver-retention failure")
    if args.duration <= 0 or args.duration > 64:
        raise ValueError("retention extension duration must be in [1, 64]")
    if args.lookback <= 0 or args.lookback > 128:
        raise ValueError("retention action lookback must be in [1, 128]")

    phase_three = torch.nonzero(
        control["phases"].long() == 3,
        as_tuple=False,
    ).flatten()
    if phase_three.numel() == 0:
        raise ValueError("control never reached receiver custody")
    branch_frame = int(phase_three[0].item())
    observation = control["observations"][branch_frame]
    giver_is_robot_1 = bool(observation[82].item() > 0.5)
    receiver_action_start = 7 if giver_is_robot_1 else 0
    receiver_contact_start = 68 if giver_is_robot_1 else 66

    search_start = max(0, branch_frame - int(args.lookback))
    recent_translation = control["actions"][
        search_start:branch_frame,
        receiver_action_start : receiver_action_start + 3,
    ]
    active_rows = torch.nonzero(
        recent_translation.abs().amax(dim=-1) > 1.0e-7,
        as_tuple=False,
    ).flatten()
    if active_rows.numel() == 0:
        raise ValueError("no causal pre-custody centering action was observed")
    source_frame = search_start + int(active_rows[-1].item())
    source_translation = control["actions"][
        source_frame,
        receiver_action_start : receiver_action_start + 3,
    ]
    axis = int(torch.argmax(source_translation.abs()).item())
    value = float(source_translation[axis].item())
    if not 0.0 < abs(value) <= 0.05:
        raise ValueError("selected centering action is outside the teacher cap")

    schedule_path = Path(args.output_schedule).expanduser().resolve()
    receipt_path = Path(args.output_receipt).expanduser().resolve()
    if schedule_path.exists() or receipt_path.exists():
        raise ValueError("refusing to overwrite immutable teacher artifacts")
    schedule = {
        "schema_version": ACTION_SCHEDULE_SCHEMA,
        "task": control["task"],
        "seed": int(control["seed"]),
        "pair_id": control["pair_id"],
        "source_control": {
            "path": str(control_path),
            "sha256": _sha256(control_path),
        },
        "branch_frame": branch_frame,
        "segments": [
            {
                "start_frame_inclusive": branch_frame,
                "stop_frame_exclusive": branch_frame + int(args.duration),
                "action_indices": [receiver_action_start + axis],
                "values": [value],
                "mode": "replace",
            }
        ],
        "derivation": {
            "kind": "extend_last_observed_receiver_centering_action",
            "lookback_frames": int(args.lookback),
            "source_frame": source_frame,
            "receiver_action_axis": axis,
            "receiver_contact_at_branch_n": (
                observation[
                    receiver_contact_start : receiver_contact_start + 2
                ]
                / 0.2
            ).tolist(),
        },
    }
    _atomic_json_save(schedule, schedule_path)
    schedule_sha256 = _sha256(schedule_path)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "task": control["task"],
        "seed": int(control["seed"]),
        "pair_id": control["pair_id"],
        "teacher_kind": "constrained_trajectory_optimizer",
        "optimizer": {
            "algorithm": "deterministic_contact_retention_action_extension",
            "objective": "terminal_success_without_any_safety_event",
            "constraints": {
                "single_environment_final_replay": True,
                "environment_action_bound": 1.0,
                "teacher_axis_action_cap": 0.05,
                "maximum_extension_frames": 64,
                "proposal_sources_are_not_training_labels": True,
            },
            "selected_parameters": {
                "action_schedule_sha256": schedule_sha256,
            },
            "proposal_sources": [
                {
                    "path": str(control_path),
                    "sha256": _sha256(control_path),
                }
            ],
        },
    }
    _atomic_json_save(receipt, receipt_path)
    return {
        "schema_version": ACTION_SCHEDULE_SCHEMA,
        "schedule": str(schedule_path),
        "schedule_sha256": schedule_sha256,
        "receipt": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
        "pair_id": control["pair_id"],
        "seed": int(control["seed"]),
        "branch_frame": branch_frame,
        "source_frame": source_frame,
        "action_index": receiver_action_start + axis,
        "action_value": value,
        "duration": int(args.duration),
    }


def _same_contract(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_source = left["runtime"]["source"]
    right_source = right["runtime"]["source"]
    return all(
        (
            left["task"] == right["task"],
            int(left["seed"]) == int(right["seed"]),
            left["pair_id"] == right["pair_id"],
            left_source["dranmar_revision"] == right_source["dranmar_revision"],
            left_source["asset_revision"] == right_source["asset_revision"],
            left["policy"]["base_checkpoint_sha256"] == right["policy"]["base_checkpoint_sha256"],
            float(left.get("reset_rotation_randomization_deg", 0.0))
            == float(right.get("reset_rotation_randomization_deg", 0.0)),
        )
    )


def _exact_control_replay(control_a: dict[str, Any], control_b: dict[str, Any]) -> bool:
    return all(
        (
            control_a["policy"] == control_b["policy"],
            control_a["terminal"] == control_b["terminal"],
            torch.equal(control_a["observations"], control_b["observations"]),
            torch.equal(control_a["actions"], control_b["actions"]),
            torch.equal(control_a["rewards"], control_b["rewards"]),
            torch.equal(control_a["phases"], control_b["phases"]),
            torch.equal(control_a["safety_events"], control_b["safety_events"]),
        )
    )


def _episode_payload(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "observations": trace["observations"].float().clone(),
        "actions": trace["actions"].float().clone(),
        "rewards": trace["rewards"].float().clone(),
        "phases": trace["phases"].long().clone(),
        "safety_events": trace["safety_events"].bool().clone(),
        "frame_count": int(trace["observations"].shape[0]),
    }


def admit_baseline_pair(
    control_a_path: Path,
    control_b_path: Path,
) -> dict[str, Any]:
    """Admit one reproducible safe incumbent success for policy distillation."""

    control_a = _load_trace(control_a_path)
    control_b = _load_trace(control_b_path)
    _validate_trace(control_a_path, control_a)
    _validate_trace(control_b_path, control_b)
    if control_a.get("role") != "control" or control_b.get("role") != "control":
        raise ValueError("baseline demonstrations require two control traces")
    if (
        control_a.get("teacher_kind") != "frozen_baseline"
        or control_b.get("teacher_kind") != "frozen_baseline"
    ):
        raise ValueError("baseline demonstrations must use the frozen incumbent")

    contract_match = _same_contract(control_a, control_b)
    control_replay = _exact_control_replay(control_a, control_b)
    if not contract_match:
        raise ValueError("baseline traces do not share one source and policy lock")
    if not control_replay:
        raise ValueError("baseline replay is not exact; this episode is inadmissible")
    if control_a["terminal"]["outcome"] != "success":
        raise ValueError("baseline distillation admits successful episodes only")
    if bool(control_a["safety_events"].any()):
        raise ValueError("baseline distillation rejects episodes with safety events")
    phases = set(control_a["phases"].long().unique().tolist())
    if not set(range(4)).issubset(phases):
        raise ValueError(
            "baseline success lacks an action-bearing handover phase"
        )
    if (
        control_a.get("teacher_receipt")
        or control_a.get("teacher_action_schedule")
        or control_a["policy"].get("successor_checkpoint_sha256")
        or control_a["policy"].get("teacher_action_schedule_sha256")
    ):
        raise ValueError("baseline demonstration contains a teacher or successor override")

    gates = {
        "single_environment_only": True,
        "development_seed_only": True,
        "source_and_checkpoint_match": contract_match,
        "exact_replay": control_replay,
        "safe_terminal_success": True,
        "complete_action_phase_coverage": True,
        "frozen_incumbent_only": True,
    }
    return {
        "schema_version": DATASET_SCHEMA,
        "accepted": True,
        "label_source": BASELINE_LABEL_SOURCE,
        "gates": gates,
        "pair_id": control_a["pair_id"],
        "task": control_a["task"],
        "seed": int(control_a["seed"]),
        "teacher_kind": "frozen_baseline",
        "teacher_receipt": None,
        "branch_frame": None,
        "control_outcome": "success",
        "teacher_outcome": None,
        "source": copy.deepcopy(control_a["runtime"]["source"]),
        "base_checkpoint_sha256": control_a["policy"][
            "base_checkpoint_sha256"
        ],
        "trace_sources": {
            "control_a": {
                "path": str(control_a_path),
                "sha256": _sha256(control_a_path),
            },
            "control_b": {
                "path": str(control_b_path),
                "sha256": _sha256(control_b_path),
            },
        },
        "episode": _episode_payload(control_a),
    }


def accept_teacher_pair(
    control_a_path: Path,
    control_b_path: Path,
    teacher_path: Path,
) -> dict[str, Any]:
    """Fail closed unless an isolated teacher unambiguously rescues one episode."""

    traces = {
        "control_a": _load_trace(control_a_path),
        "control_b": _load_trace(control_b_path),
        "teacher": _load_trace(teacher_path),
    }
    for role, path in (
        ("control_a", control_a_path),
        ("control_b", control_b_path),
        ("teacher", teacher_path),
    ):
        _validate_trace(path, traces[role])

    control_a = traces["control_a"]
    control_b = traces["control_b"]
    teacher = traces["teacher"]
    if control_a.get("role") != "control" or control_b.get("role") != "control":
        raise ValueError("both no-op replays must be labeled as controls")
    if (
        control_a.get("teacher_kind") != "frozen_baseline"
        or control_b.get("teacher_kind") != "frozen_baseline"
    ):
        raise ValueError("control traces must identify the frozen baseline")
    if teacher.get("role") != "teacher":
        raise ValueError("teacher trace must be labeled as the teacher")
    if teacher.get("teacher_kind") not in ALLOWED_TEACHERS:
        raise ValueError("teacher must be an optimizer or clinician trajectory")
    receipt = teacher.get("teacher_receipt")
    if not isinstance(receipt, dict) or not receipt.get("sha256"):
        raise ValueError("teacher trace lacks its immutable optimizer or teleoperation receipt")
    selected_schedule = None
    schedule_metadata = None
    schedule_branch_frame = None
    schedule_segment_start = None
    if teacher.get("teacher_kind") == "constrained_trajectory_optimizer":
        selected_parameters = receipt.get("selected_parameters", {})
        selected_schedule = selected_parameters.get("action_schedule_sha256")
        schedule_metadata = teacher.get("teacher_action_schedule")
        recorded_schedule = teacher["policy"].get(
            "teacher_action_schedule_sha256"
        )
        selected = selected_parameters.get(
            "receiver_recovery_fixed_correction"
        )
        recorded = teacher["policy"].get("receiver_recovery_fixed_correction")
        if selected_schedule:
            schedule_segments = (
                schedule_metadata.get("segments")
                if isinstance(schedule_metadata, dict)
                else None
            )
            if (
                not isinstance(schedule_metadata, dict)
                or not isinstance(schedule_segments, list)
                or not schedule_segments
                or not isinstance(schedule_segments[0], dict)
                or selected_schedule != schedule_metadata.get("sha256")
                or selected_schedule != recorded_schedule
            ):
                raise ValueError(
                    "optimizer receipt does not match the recorded action schedule"
                )
            schedule_branch_frame = int(
                schedule_metadata.get("branch_frame", -1)
            )
            schedule_segment_start = int(
                schedule_segments[0].get("start_frame_inclusive", -1)
            )
        elif not isinstance(selected, list) or len(selected) != 6 or not recorded:
            raise ValueError("optimizer receipt and trace lack one selected correction")
        else:
            recorded_values = [
                float(value.strip()) for value in recorded.split(",")
            ]
            if len(recorded_values) != 6 or any(
                abs(float(expected) - actual) > 1.0e-12
                for expected, actual in zip(
                    selected,
                    recorded_values,
                    strict=True,
                )
            ):
                raise ValueError(
                    "optimizer receipt does not match the recorded policy"
                )

    contract_match = _same_contract(control_a, control_b) and _same_contract(control_a, teacher)
    control_replay = _exact_control_replay(control_a, control_b)
    if not contract_match:
        raise ValueError("paired traces do not share one seed, source, task, and base checkpoint")
    if not control_replay:
        raise ValueError("no-op replay is not exact; this seed cannot produce labels")

    control_actions = control_a["actions"]
    teacher_actions = teacher["actions"]
    shared_frames = min(control_actions.shape[0], teacher_actions.shape[0])
    action_difference = (
        teacher_actions[:shared_frames] - control_actions[:shared_frames]
    ).abs().amax(dim=-1)
    divergent = torch.nonzero(action_difference > 1.0e-7, as_tuple=False).flatten()
    if divergent.numel() == 0:
        raise ValueError("teacher never branches from the frozen control")
    branch_frame = int(divergent[0].item())
    if selected_schedule and (
        schedule_branch_frame != branch_frame
        or schedule_segment_start != branch_frame
    ):
        raise ValueError(
            "recorded action schedule does not match the observed branch"
        )
    prebranch_observation_parity = torch.equal(
        teacher["observations"][: branch_frame + 1],
        control_a["observations"][: branch_frame + 1],
    )
    prebranch_action_parity = torch.equal(
        teacher_actions[:branch_frame],
        control_actions[:branch_frame],
    )
    if not prebranch_observation_parity or not prebranch_action_parity:
        raise ValueError("teacher and control diverge before the declared action branch")

    control_outcome = control_a["terminal"]["outcome"]
    teacher_outcome = teacher["terminal"]["outcome"]
    teacher_win = control_outcome != "success" and teacher_outcome == "success"
    teacher_safe = not bool(teacher["safety_events"].any())
    if not teacher_win:
        raise ValueError("teacher must turn a reproducible control failure into success")
    if not teacher_safe:
        raise ValueError("teacher introduced a safety event")

    gates = {
        "single_environment_only": True,
        "development_seed_only": True,
        "source_and_checkpoint_match": contract_match,
        "exact_noop_replay": control_replay,
        "prebranch_observation_parity": prebranch_observation_parity,
        "prebranch_action_parity": prebranch_action_parity,
        "teacher_branches": True,
        "teacher_wins": teacher_win,
        "teacher_adds_no_safety_event": teacher_safe,
        "accepted_teacher_type": True,
        "immutable_teacher_receipt": True,
    }
    if not all(gates.values()):
        raise AssertionError("teacher acceptance gates unexpectedly failed open")

    return {
        "schema_version": DATASET_SCHEMA,
        "accepted": True,
        "label_source": TEACHER_LABEL_SOURCE,
        "gates": gates,
        "pair_id": teacher["pair_id"],
        "task": teacher["task"],
        "seed": int(teacher["seed"]),
        "teacher_kind": teacher["teacher_kind"],
        "teacher_receipt": copy.deepcopy(receipt),
        "branch_frame": branch_frame,
        "control_outcome": control_outcome,
        "teacher_outcome": teacher_outcome,
        "source": copy.deepcopy(teacher["runtime"]["source"]),
        "base_checkpoint_sha256": teacher["policy"]["base_checkpoint_sha256"],
        "trace_sources": {
            "control_a": {
                "path": str(control_a_path),
                "sha256": _sha256(control_a_path),
            },
            "control_b": {
                "path": str(control_b_path),
                "sha256": _sha256(control_b_path),
            },
            "teacher": {
                "path": str(teacher_path),
                "sha256": _sha256(teacher_path),
            },
        },
        "episode": _episode_payload(teacher),
    }


def _load_accepted_dataset(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != DATASET_SCHEMA:
        raise ValueError(f"unsupported accepted successor dataset: {path}")
    if not payload.get("accepted") or not all(payload.get("gates", {}).values()):
        raise ValueError(f"successor dataset did not pass every gate: {path}")
    if payload.get("label_source") not in ALLOWED_LABEL_SOURCES:
        raise ValueError(f"successor dataset has an unsupported label source: {path}")
    episode = payload.get("episode")
    if not isinstance(episode, dict):
        raise ValueError(f"accepted dataset lacks an episode: {path}")
    observations = episode.get("observations")
    actions = episode.get("actions")
    phases = episode.get("phases")
    if not all(isinstance(value, torch.Tensor) for value in (observations, actions, phases)):
        raise ValueError(f"accepted dataset tensors are incomplete: {path}")
    if observations.ndim != 2 or observations.shape[-1] != OBSERVATION_DIM:
        raise ValueError(f"accepted observation contract drifted: {path}")
    if actions.shape != (observations.shape[0], ACTION_DIM):
        raise ValueError(f"accepted action contract drifted: {path}")
    if phases.shape != (observations.shape[0],):
        raise ValueError(f"accepted phase contract drifted: {path}")
    if not torch.isfinite(observations).all() or not torch.isfinite(actions).all():
        raise ValueError(f"accepted successor data contains non-finite values: {path}")
    if bool((actions.abs() > 1.000001).any()):
        raise ValueError(f"accepted actions exceed the environment contract: {path}")
    return payload


def _stable_validation_pair_ids(pair_ids: list[str], fraction: float) -> set[str]:
    ranked = sorted(
        pair_ids,
        key=lambda value: hashlib.sha256(value.encode()).digest(),
    )
    count = max(2, round(len(ranked) * fraction))
    count = min(count, len(ranked) - 2)
    return set(ranked[:count])


def train_successor(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    policy_path = (
        repo_root
        / "source/extensions/orbit.surgical.tasks/orbit/surgical/tasks/"
        "surgical/handover/successor_policy.py"
    )
    specification = importlib.util.spec_from_file_location(
        "dranmar_handover_successor_policy",
        policy_path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("unable to load the handover successor policy")
    policy_module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(policy_module)
    HANDOVER_ACTION_DIM = policy_module.HANDOVER_ACTION_DIM
    HANDOVER_OBSERVATION_DIM = policy_module.HANDOVER_OBSERVATION_DIM
    HANDOVER_PHASE_SLICE = policy_module.HANDOVER_PHASE_SLICE
    SUCCESSOR_CHECKPOINT_SCHEMA = policy_module.SUCCESSOR_CHECKPOINT_SCHEMA
    PhaseConditionedHandoverPolicy = (
        policy_module.PhaseConditionedHandoverPolicy
    )

    if HANDOVER_OBSERVATION_DIM != OBSERVATION_DIM or HANDOVER_ACTION_DIM != ACTION_DIM:
        raise ValueError("trainer and successor policy contracts disagree")
    if (HANDOVER_PHASE_SLICE.start, HANDOVER_PHASE_SLICE.stop) != (
        PHASE_SLICE.start,
        PHASE_SLICE.stop,
    ):
        raise ValueError("trainer and successor phase contracts disagree")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    if not 0.1 <= args.validation_fraction <= 0.4:
        raise ValueError("validation fraction must be in [0.1, 0.4]")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("optimizer parameters are invalid")
    if args.patience <= 0:
        raise ValueError("early-stopping patience must be positive")

    paths = [Path(value).expanduser().resolve() for value in args.dataset]
    datasets = [_load_accepted_dataset(path) for path in paths]
    pair_ids = [str(payload["pair_id"]) for payload in datasets]
    if len(datasets) < 8 or len(set(pair_ids)) != len(pair_ids):
        raise ValueError(
            "training requires at least eight distinct accepted successor pairs"
        )
    if len({int(payload["seed"]) for payload in datasets}) < 4:
        raise ValueError("training requires at least four distinct development seeds")
    baseline_pair_count = sum(
        payload["label_source"] == BASELINE_LABEL_SOURCE
        for payload in datasets
    )
    teacher_pair_count = sum(
        payload["label_source"] == TEACHER_LABEL_SOURCE
        for payload in datasets
    )
    tasks = {payload["task"] for payload in datasets}
    base_hashes = {payload["base_checkpoint_sha256"] for payload in datasets}
    source_revisions = {payload["source"]["dranmar_revision"] for payload in datasets}
    asset_revisions = {payload["source"]["asset_revision"] for payload in datasets}
    if len(tasks) != 1:
        raise ValueError("accepted demonstrations do not share one task")
    if len(base_hashes) != 1:
        raise ValueError("accepted demonstrations do not share one baseline")
    if len(source_revisions) != 1 or len(asset_revisions) != 1:
        raise ValueError("accepted demonstrations do not share one source lock")

    validation_ids = _stable_validation_pair_ids(pair_ids, args.validation_fraction)
    train_payloads = [
        payload for payload in datasets if payload["pair_id"] not in validation_ids
    ]
    validation_payloads = [
        payload for payload in datasets if payload["pair_id"] in validation_ids
    ]
    if len(train_payloads) < 2 or len(validation_payloads) < 2:
        raise ValueError("episode-level split is too small")

    train_observations = torch.cat(
        [payload["episode"]["observations"].float() for payload in train_payloads]
    )
    train_actions = torch.cat(
        [payload["episode"]["actions"].float() for payload in train_payloads]
    )
    train_phases = torch.cat(
        [payload["episode"]["phases"].long() for payload in train_payloads]
    )
    validation_observations = torch.cat(
        [payload["episode"]["observations"].float() for payload in validation_payloads]
    )
    validation_actions = torch.cat(
        [payload["episode"]["actions"].float() for payload in validation_payloads]
    )
    validation_phases = torch.cat(
        [payload["episode"]["phases"].long() for payload in validation_payloads]
    )
    phase_counts = torch.bincount(train_phases, minlength=5)
    missing_action_phases = torch.nonzero(
        phase_counts[:4] == 0,
        as_tuple=False,
    ).flatten()
    if missing_action_phases.numel():
        raise ValueError(
            "accepted demonstration set lacks action-bearing phases: "
            f"{missing_action_phases.tolist()}"
        )

    observation_mean = train_observations.mean(dim=0)
    observation_std = train_observations.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    action_loss_scale = torch.quantile(
        train_actions.abs(),
        0.90,
        dim=0,
    ).clamp_min(0.01)
    hidden_dims = tuple(int(value) for value in args.hidden_dims.split(",") if value)
    if not hidden_dims:
        raise ValueError("at least one shared hidden dimension is required")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    model = PhaseConditionedHandoverPolicy(
        observation_mean,
        observation_std,
        hidden_dims=hidden_dims,
        head_dim=args.head_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_observations = train_observations.to(device)
    train_actions = train_actions.to(device)
    train_phases = train_phases.to(device)
    validation_observations = validation_observations.to(device)
    validation_actions = validation_actions.to(device)
    validation_phases = validation_phases.to(device)
    action_loss_scale_device = action_loss_scale.to(device)
    phase_counts_device = phase_counts.to(device=device, dtype=torch.float32)
    active_phase = phase_counts_device > 0
    phase_weight = torch.zeros_like(phase_counts_device)
    phase_weight[active_phase] = (
        phase_counts_device.sum()
        / (
            active_phase.sum()
            * phase_counts_device[active_phase]
        )
    )
    best_validation_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(args.epochs):
        model.train()
        permutation = torch.randperm(train_observations.shape[0], device=device)
        loss_sum = 0.0
        sample_count = 0
        for start in range(0, permutation.numel(), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            predicted = model(train_observations[indices])
            per_frame = functional.smooth_l1_loss(
                predicted / action_loss_scale_device,
                train_actions[indices] / action_loss_scale_device,
                reduction="none",
                beta=0.1,
            ).mean(dim=-1)
            loss = (
                per_frame * phase_weight[train_phases[indices]]
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item()) * indices.numel()
            sample_count += indices.numel()

        model.eval()
        with torch.inference_mode():
            validation_prediction = model(validation_observations)
            validation_per_frame = functional.smooth_l1_loss(
                validation_prediction / action_loss_scale_device,
                validation_actions / action_loss_scale_device,
                reduction="none",
                beta=0.1,
            ).mean(dim=-1)
            validation_loss = float(
                (
                    validation_per_frame
                    * phase_weight[validation_phases]
                )
                .mean()
                .item()
            )
            validation_mae = float(
                (validation_prediction - validation_actions).abs().mean().item()
            )
        train_loss = loss_sum / sample_count
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_mae": validation_mae,
            }
        )
        if validation_loss < best_validation_loss - 1.0e-7:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is None:
        raise AssertionError("successor training did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.inference_mode():
        final_prediction = model(validation_observations)
        final_validation_mae = float(
            (final_prediction - validation_actions).abs().mean().item()
        )
        final_max_abs_error = float(
            (final_prediction - validation_actions).abs().max().item()
        )
        final_phase_mae = {
            str(phase): (
                float(
                    (
                        final_prediction[validation_phases == phase]
                        - validation_actions[validation_phases == phase]
                    )
                    .abs()
                    .mean()
                    .item()
                )
                if bool((validation_phases == phase).any())
                else None
            )
            for phase in range(5)
        }

    output = Path(args.output).expanduser().resolve()
    if baseline_pair_count and teacher_pair_count:
        capability_scope = "incumbent_distillation_plus_teacher_rescues"
    elif teacher_pair_count:
        capability_scope = "teacher_rescue_only"
    else:
        capability_scope = "incumbent_distillation_only"
    checkpoint = {
        "schema_version": SUCCESSOR_CHECKPOINT_SCHEMA,
        "deployment_status": "candidate_only",
        "training_gate_passed": True,
        "capability_scope": capability_scope,
        "improvement_labels_present": teacher_pair_count > 0,
        "task": next(iter(tasks)),
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "phase_slice": [PHASE_SLICE.start, PHASE_SLICE.stop],
        "architecture": {
            "kind": "shared_encoder_with_five_phase_heads",
            "hidden_dims": list(hidden_dims),
            "head_dim": int(args.head_dim),
            "full_action_policy": True,
            "runtime_heuristic_stack": False,
            "terminal_phase_action": "zero_initialized_no_training_required",
        },
        "observation_mean": observation_mean.cpu(),
        "observation_std": observation_std.cpu(),
        "action_loss_scale": action_loss_scale.cpu(),
        "model": best_state,
        "base_checkpoint_sha256": next(iter(base_hashes)),
        "source": {
            "dranmar_revision": next(iter(source_revisions)),
            "asset_revision": next(iter(asset_revisions)),
        },
        "training": {
            "seed": int(args.seed),
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "loss": (
                "phase_balanced_p90_action_scaled_smooth_l1_beta_0.1"
            ),
            "phase_counts": phase_counts.tolist(),
            "train_frames": int(train_observations.shape[0]),
            "validation_frames": int(validation_observations.shape[0]),
            "train_pair_ids": [payload["pair_id"] for payload in train_payloads],
            "validation_pair_ids": sorted(validation_ids),
            "best_validation_smooth_l1": best_validation_loss,
            "validation_action_mae": final_validation_mae,
            "validation_action_max_abs_error": final_max_abs_error,
            "validation_phase_mae": final_phase_mae,
            "baseline_distillation_pairs": baseline_pair_count,
            "teacher_rescue_pairs": teacher_pair_count,
            "history_tail": history[-10:],
        },
        "successor_datasets": [
            {
                "path": str(path),
                "sha256": _sha256(path),
                "pair_id": payload["pair_id"],
                "seed": int(payload["seed"]),
                "label_source": payload["label_source"],
                "teacher_kind": payload["teacher_kind"],
            }
            for path, payload in zip(paths, datasets, strict=True)
        ],
    }
    _atomic_torch_save(checkpoint, output)
    return {
        "schema_version": SUCCESSOR_CHECKPOINT_SCHEMA,
        "output": str(output),
        "sha256": _sha256(output),
        "deployment_status": "candidate_only",
        "accepted_successor_pairs": len(datasets),
        "baseline_distillation_pairs": baseline_pair_count,
        "teacher_rescue_pairs": teacher_pair_count,
        "capability_scope": capability_scope,
        "train_frames": int(train_observations.shape[0]),
        "validation_frames": int(validation_observations.shape[0]),
        "best_epoch": best_epoch,
        "validation_action_mae": final_validation_mae,
        "validation_action_max_abs_error": final_max_abs_error,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the compact learned handover successor"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    receipt = subparsers.add_parser(
        "receipt",
        help="lock one risk-guided bounded optimizer proposal before replay",
    )
    receipt.add_argument("--pair_id", required=True)
    receipt.add_argument(
        "--task",
        default="DrAnmar-Handover-Needle-Dual-PSM-IK-Rel-v0",
    )
    receipt.add_argument("--seed", type=int, required=True)
    receipt.add_argument("--receiver_correction", required=True)
    receipt.add_argument("--position_cap_m", type=float, default=0.0025)
    receipt.add_argument("--orientation_cap_deg", type=float, default=2.0)
    receipt.add_argument("--proposal_source", action="append", required=True)
    receipt.add_argument("--output", required=True)

    retention = subparsers.add_parser(
        "propose-retention",
        help="derive a bounded post-acquisition action schedule from a control",
    )
    retention.add_argument("--control", required=True)
    retention.add_argument("--duration", type=int, default=12)
    retention.add_argument("--lookback", type=int, default=16)
    retention.add_argument("--output_schedule", required=True)
    retention.add_argument("--output_receipt", required=True)

    accept = subparsers.add_parser(
        "accept",
        help="accept one isolated control/control/teacher episode triplet",
    )
    accept.add_argument("--control_a", required=True)
    accept.add_argument("--control_b", required=True)
    accept.add_argument("--teacher", required=True)
    accept.add_argument("--output", required=True)

    distill = subparsers.add_parser(
        "admit-baseline",
        help="admit one exact safe incumbent success for distillation",
    )
    distill.add_argument("--control_a", required=True)
    distill.add_argument("--control_b", required=True)
    distill.add_argument("--output", required=True)

    train = subparsers.add_parser(
        "train",
        help="train one full-action policy from accepted demonstrations",
    )
    train.add_argument("--dataset", action="append", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--epochs", type=int, default=300)
    train.add_argument("--batch_size", type=int, default=512)
    train.add_argument("--learning_rate", type=float, default=3.0e-4)
    train.add_argument("--weight_decay", type=float, default=1.0e-5)
    train.add_argument("--validation_fraction", type=float, default=0.2)
    train.add_argument("--hidden_dims", default="256,256")
    train.add_argument("--head_dim", type=int, default=128)
    train.add_argument("--patience", type=int, default=30)
    train.add_argument("--seed", type=int, default=104729)
    train.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def main(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    if args.command == "receipt":
        result = create_optimizer_receipt(args)
    elif args.command == "propose-retention":
        result = propose_retention_schedule(args)
    elif args.command == "accept":
        output = Path(args.output).expanduser().resolve()
        payload = accept_teacher_pair(
            Path(args.control_a).expanduser().resolve(),
            Path(args.control_b).expanduser().resolve(),
            Path(args.teacher).expanduser().resolve(),
        )
        _atomic_torch_save(payload, output)
        result = {
            "schema_version": DATASET_SCHEMA,
            "accepted": True,
            "output": str(output),
            "sha256": _sha256(output),
            "pair_id": payload["pair_id"],
            "seed": payload["seed"],
            "branch_frame": payload["branch_frame"],
            "teacher_kind": payload["teacher_kind"],
            "frames": payload["episode"]["frame_count"],
        }
    elif args.command == "admit-baseline":
        output = Path(args.output).expanduser().resolve()
        payload = admit_baseline_pair(
            Path(args.control_a).expanduser().resolve(),
            Path(args.control_b).expanduser().resolve(),
        )
        _atomic_torch_save(payload, output)
        result = {
            "schema_version": DATASET_SCHEMA,
            "accepted": True,
            "output": str(output),
            "sha256": _sha256(output),
            "pair_id": payload["pair_id"],
            "seed": payload["seed"],
            "label_source": payload["label_source"],
            "frames": payload["episode"]["frame_count"],
        }
    else:
        result = train_successor(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
