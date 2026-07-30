#!/usr/bin/env python3
"""Admit isolated handover demonstrations and train the full-action successor."""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional


TRACE_SCHEMA = "dranmar-handover-teacher-trace-1.0"
DAGGER_TRACE_SCHEMA = "dranmar-handover-dagger-trace-1.0"
DATASET_SCHEMA_V1 = "dranmar-handover-successor-dataset-1.0"
DATASET_SCHEMA_V2 = "dranmar-handover-successor-dataset-2.0"
DATASET_SCHEMA = DATASET_SCHEMA_V1
SCAFFOLD_COMPATIBILITY_SCHEMA = (
    "dranmar-handover-scaffold-compatibility-1.0"
)
RECEIPT_SCHEMA = "dranmar-handover-teacher-receipt-1.0"
ACTION_SCHEDULE_SCHEMA = "dranmar-handover-teacher-action-schedule-1.0"
SUCCESSOR_CHECKPOINT_SCHEMA_V3 = "dranmar-handover-successor-policy-3.0"
LEGACY_SUCCESSOR_CHECKPOINT_SCHEMA_V2 = (
    "dranmar-handover-successor-policy-2.0"
)
BASELINE_LABEL_SOURCE = "frozen_baseline_success_distillation"
DAGGER_LABEL_SOURCE = "frozen_baseline_dagger"
TEACHER_LABEL_SOURCE = "independent_teacher_rescue"
ALLOWED_LABEL_SOURCES = {
    BASELINE_LABEL_SOURCE,
    DAGGER_LABEL_SOURCE,
    TEACHER_LABEL_SOURCE,
}
QUALIFICATION_SEEDS = {17, 2361, 4099}
ALLOWED_TEACHERS = {
    "constrained_trajectory_optimizer",
    "clinician_teleoperation",
}
ALLOWED_CONTROL_POLICIES = {
    "frozen_baseline",
    "successor_candidate",
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
HANDOVER_CONTRACT_ROOT = (
    "source/extensions/orbit.surgical.tasks/orbit/surgical/tasks/"
    "surgical/handover"
)
HANDOVER_POLICY_ONLY_FILES = {
    f"{HANDOVER_CONTRACT_ROOT}/recovery_policy.py",
    f"{HANDOVER_CONTRACT_ROOT}/residual_model.py",
    f"{HANDOVER_CONTRACT_ROOT}/successor_policy.py",
}
HANDOVER_SUCCESSOR_POLICY_FILE = (
    f"{HANDOVER_CONTRACT_ROOT}/successor_policy.py"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_revision(repo_root: Path, revision: str) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _revision_blob(revision: str, repository_path: str) -> str:
    """Resolve one source blob used to interpret frozen policy weights."""

    repo_root = Path(__file__).resolve().parents[1]
    resolved = _resolve_revision(repo_root, revision)
    completed = subprocess.run(
        ["git", "rev-parse", f"{resolved}:{repository_path}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


@functools.lru_cache(maxsize=16)
def _handover_contract_manifest(revision: str) -> dict[str, Any]:
    """Hash the frozen task contract while excluding policy implementations."""

    repo_root = Path(__file__).resolve().parents[1]
    resolved = _resolve_revision(repo_root, revision)
    completed = subprocess.run(
        [
            "git",
            "ls-tree",
            "-r",
            resolved,
            "--",
            HANDOVER_CONTRACT_ROOT,
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    entries: list[dict[str, str]] = []
    digest = hashlib.sha256()
    for line in completed.stdout.splitlines():
        metadata, path = line.split("\t", 1)
        _, object_type, object_sha256 = metadata.split()
        if object_type != "blob" or path in HANDOVER_POLICY_ONLY_FILES:
            continue
        entries.append({"path": path, "git_blob": object_sha256})
        digest.update(path.encode())
        digest.update(b"\0")
        digest.update(object_sha256.encode())
        digest.update(b"\n")
    if not entries:
        raise ValueError("handover task contract manifest is empty")
    return {
        "revision": resolved,
        "sha256": digest.hexdigest(),
        "entries": entries,
    }


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


def bootstrap_failure_mining_seed(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Re-freeze the preserved recurrent seed for current-source mining."""

    repo_root = Path(__file__).resolve().parents[1]
    runtime_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source = Path(args.checkpoint).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"failure-mining seed does not exist: {source}")
    source_sha256 = _sha256(source)
    if source_sha256 != args.expected_sha256:
        raise ValueError("failure-mining seed hash mismatch")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("failure-mining seed must contain a mapping")
    if (
        payload.get("schema_version")
        != LEGACY_SUCCESSOR_CHECKPOINT_SCHEMA_V2
    ):
        raise ValueError("failure-mining seed must use the preserved v2 schema")
    if payload.get("deployment_status") != "candidate_only":
        raise ValueError("failure-mining seed must remain candidate-only")
    if not payload.get("training_gate_passed"):
        raise ValueError("failure-mining seed lacks its data gate")
    architecture = payload.get("architecture", {})
    if (
        not isinstance(architecture, dict)
        or architecture.get("runtime_heuristic_stack") is not False
        or architecture.get("full_action_policy") is not True
        or architecture.get("recurrent_state") != "gru_reset_per_episode"
    ):
        raise ValueError("failure-mining seed architecture is not standalone")
    old_source = payload.get("source", {})
    if (
        not isinstance(old_source, dict)
        or not old_source.get("dranmar_revision")
        or not old_source.get("asset_revision")
    ):
        raise ValueError("failure-mining seed lacks source provenance")

    successor = copy.deepcopy(payload)
    successor["schema_version"] = SUCCESSOR_CHECKPOINT_SCHEMA_V3
    successor["deployment_status"] = "candidate_only"
    successor["promotion_eligible"] = False
    successor["source"] = {
        "dranmar_revision": runtime_revision,
        "weight_training_dranmar_revision": old_source[
            "dranmar_revision"
        ],
        "asset_revision": old_source["asset_revision"],
    }
    successor["failure_mining_seed"] = {
        "path": str(source),
        "sha256": source_sha256,
        "schema_version": payload["schema_version"],
        "purpose": "initialization_and_failure_mining_only",
        "promotion_candidate": False,
    }
    output = Path(args.output).expanduser().resolve()
    _atomic_torch_save(successor, output)
    return {
        "schema_version": SUCCESSOR_CHECKPOINT_SCHEMA_V3,
        "deployment_status": "candidate_only",
        "promotion_eligible": False,
        "output": str(output),
        "sha256": _sha256(output),
        "source_checkpoint_sha256": source_sha256,
        "runtime_revision": runtime_revision,
    }


def rebind_successor_candidate(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Bind unchanged candidate weights to a compatible current source."""

    repo_root = Path(__file__).resolve().parents[1]
    runtime_revision = _resolve_revision(repo_root, "HEAD")
    source = Path(args.checkpoint).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"successor candidate does not exist: {source}")
    source_sha256 = _sha256(source)
    if source_sha256 != args.expected_sha256:
        raise ValueError("successor candidate hash mismatch")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version")
        not in {
            SUCCESSOR_CHECKPOINT_SCHEMA_V3,
            LEGACY_SUCCESSOR_CHECKPOINT_SCHEMA_V2,
        }
        or payload.get("deployment_status") != "candidate_only"
        or not payload.get("training_gate_passed")
    ):
        raise ValueError("source is not a preserved successor candidate")
    architecture = payload.get("architecture", {})
    if (
        not isinstance(architecture, dict)
        or architecture.get("runtime_heuristic_stack") is not False
        or architecture.get("full_action_policy") is not True
        or architecture.get("recurrent_state") != "gru_reset_per_episode"
    ):
        raise ValueError("successor candidate architecture is not standalone")
    old_source = payload.get("source", {})
    old_revision = old_source.get("dranmar_revision")
    if not old_revision or not old_source.get("asset_revision"):
        raise ValueError("successor candidate lacks source provenance")
    old_contract = _handover_contract_manifest(str(old_revision))
    runtime_contract = _handover_contract_manifest(runtime_revision)
    if old_contract["sha256"] != runtime_contract["sha256"]:
        raise ValueError(
            "handover environment contract changed; candidate replay "
            "and retraining are required"
        )
    old_policy_blob = _revision_blob(
        str(old_revision),
        HANDOVER_SUCCESSOR_POLICY_FILE,
    )
    runtime_policy_blob = _revision_blob(
        runtime_revision,
        HANDOVER_SUCCESSOR_POLICY_FILE,
    )
    if old_policy_blob != runtime_policy_blob:
        raise ValueError(
            "successor policy implementation changed; candidate replay "
            "and retraining are required"
        )

    successor = copy.deepcopy(payload)
    successor["schema_version"] = SUCCESSOR_CHECKPOINT_SCHEMA_V3
    successor["deployment_status"] = "candidate_only"
    successor["promotion_eligible"] = False
    successor["source"] = {
        **copy.deepcopy(old_source),
        "dranmar_revision": runtime_revision,
        "weight_training_dranmar_revision": old_source.get(
            "weight_training_dranmar_revision",
            old_revision,
        ),
    }
    successor["source_rebind"] = {
        "schema_version": "dranmar-handover-source-rebind-1.0",
        "parent_checkpoint": {
            "path": str(source),
            "sha256": source_sha256,
            "schema_version": payload["schema_version"],
        },
        "parent_dranmar_revision": old_revision,
        "runtime_dranmar_revision": runtime_revision,
        "handover_environment_contract_sha256": old_contract["sha256"],
        "successor_policy_git_blob": old_policy_blob,
        "weights_unchanged": True,
        "promotion_candidate": False,
    }
    output = Path(args.output).expanduser().resolve()
    _atomic_torch_save(successor, output)
    return {
        "schema_version": SUCCESSOR_CHECKPOINT_SCHEMA_V3,
        "deployment_status": "candidate_only",
        "promotion_eligible": False,
        "output": str(output),
        "sha256": _sha256(output),
        "parent_checkpoint_sha256": source_sha256,
        "runtime_revision": runtime_revision,
        "handover_environment_contract_sha256": old_contract["sha256"],
        "successor_policy_git_blob": old_policy_blob,
        "weights_unchanged": True,
    }


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


def _load_dagger_trace(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"DAgger trace is not a mapping: {path}")
    if payload.get("schema_version") != DAGGER_TRACE_SCHEMA:
        raise ValueError(f"unsupported DAgger trace schema: {path}")
    return payload


def _validate_dagger_trace(
    path: Path,
    trace: dict[str, Any],
) -> None:
    oracle_actions = trace.get("oracle_actions")
    proxy = dict(trace)
    proxy["schema_version"] = TRACE_SCHEMA
    proxy["actions"] = oracle_actions
    _validate_trace(path, proxy)

    frame_count = int(trace["observations"].shape[0])
    student_actions = trace.get("student_actions")
    executed_actions = trace.get("executed_actions")
    if not all(
        isinstance(value, torch.Tensor)
        for value in (student_actions, oracle_actions, executed_actions)
    ):
        raise ValueError(f"DAgger action tensors are incomplete: {path}")
    for label, actions in (
        ("student", student_actions),
        ("oracle", oracle_actions),
        ("executed", executed_actions),
    ):
        if tuple(actions.shape) != (frame_count, ACTION_DIM):
            raise ValueError(
                f"DAgger {label} action contract drifted: {path}"
            )
        if not torch.isfinite(actions).all():
            raise ValueError(
                f"DAgger {label} actions contain non-finite values: {path}"
            )
        if bool((actions.abs() > 1.000001).any()):
            raise ValueError(
                f"DAgger {label} actions exceed the environment contract: "
                f"{path}"
            )

    oracle_beta = float(trace.get("oracle_beta", -1.0))
    if not 0.5 <= oracle_beta < 1.0:
        raise ValueError(f"DAgger oracle beta is outside [0.5, 1.0): {path}")
    expected_executed = (
        oracle_beta * oracle_actions
        + (1.0 - oracle_beta) * student_actions
    )
    if not torch.allclose(
        executed_actions,
        expected_executed,
        rtol=1.0e-6,
        atol=1.0e-7,
    ):
        raise ValueError(
            f"DAgger executed actions do not match the recorded mixture: {path}"
        )

    policy = trace["policy"]
    if (
        policy.get("oracle_kind") != "frozen_promoted_composite"
        or not policy.get("successor_checkpoint_sha256")
        or policy.get("mixture")
        != (
            "oracle_beta_times_oracle_plus_"
            "one_minus_beta_times_student"
        )
    ):
        raise ValueError(f"DAgger policy provenance is incomplete: {path}")
    oracle_configuration = policy.get("oracle_configuration")
    if (
        not isinstance(oracle_configuration, dict)
        or oracle_configuration.get("base_checkpoint_sha256")
        != policy["base_checkpoint_sha256"]
        or oracle_configuration.get("successor_checkpoint_sha256")
        is not None
    ):
        raise ValueError(
            f"DAgger oracle configuration is not independently locked: {path}"
        )


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
            left["policy"].get("successor_checkpoint_sha256")
            == right["policy"].get("successor_checkpoint_sha256"),
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


def _exact_dagger_replay(
    trace_a: dict[str, Any],
    trace_b: dict[str, Any],
) -> bool:
    return all(
        (
            trace_a["policy"] == trace_b["policy"],
            trace_a["terminal"] == trace_b["terminal"],
            float(trace_a["oracle_beta"])
            == float(trace_b["oracle_beta"]),
            torch.equal(
                trace_a["observations"],
                trace_b["observations"],
            ),
            torch.equal(
                trace_a["student_actions"],
                trace_b["student_actions"],
            ),
            torch.equal(
                trace_a["oracle_actions"],
                trace_b["oracle_actions"],
            ),
            torch.equal(
                trace_a["executed_actions"],
                trace_b["executed_actions"],
            ),
            torch.equal(trace_a["rewards"], trace_b["rewards"]),
            torch.equal(trace_a["phases"], trace_b["phases"]),
            torch.equal(
                trace_a["safety_events"],
                trace_b["safety_events"],
            ),
        )
    )


def admit_dagger_pair(
    trace_a_path: Path,
    trace_b_path: Path,
) -> dict[str, Any]:
    """Admit one exact safe on-policy trajectory with frozen-oracle labels."""

    trace_a = _load_dagger_trace(trace_a_path)
    trace_b = _load_dagger_trace(trace_b_path)
    _validate_dagger_trace(trace_a_path, trace_a)
    _validate_dagger_trace(trace_b_path, trace_b)

    contract_match = _same_contract(trace_a, trace_b)
    exact_replay = _exact_dagger_replay(trace_a, trace_b)
    if not contract_match:
        raise ValueError(
            "DAgger traces do not share one source, seed, and baseline lock"
        )
    if not exact_replay:
        raise ValueError(
            "DAgger replay is not exact; this trajectory is inadmissible"
        )
    if trace_a["terminal"]["outcome"] != "success":
        raise ValueError("DAgger admission requires a terminal success")
    if bool(trace_a["safety_events"].any()):
        raise ValueError("DAgger admission rejects trajectories with safety events")
    phases = set(trace_a["phases"].long().unique().tolist())
    if not set(range(4)).issubset(phases):
        raise ValueError(
            "DAgger success lacks an action-bearing handover phase"
        )

    gates = {
        "single_environment_only": True,
        "development_seed_only": True,
        "source_baseline_and_student_match": contract_match,
        "exact_replay": exact_replay,
        "safe_terminal_success": True,
        "complete_action_phase_coverage": True,
        "frozen_promoted_oracle_only": True,
        "on_policy_mixture_recorded": True,
    }
    return {
        "schema_version": DATASET_SCHEMA,
        "accepted": True,
        "label_source": DAGGER_LABEL_SOURCE,
        "gates": gates,
        "pair_id": trace_a["pair_id"],
        "task": trace_a["task"],
        "seed": int(trace_a["seed"]),
        "teacher_kind": "frozen_promoted_composite_oracle",
        "teacher_receipt": None,
        "branch_frame": None,
        "control_outcome": None,
        "teacher_outcome": "success",
        "source": copy.deepcopy(trace_a["runtime"]["source"]),
        "base_checkpoint_sha256": trace_a["policy"][
            "base_checkpoint_sha256"
        ],
        "collection": {
            "oracle_beta": float(trace_a["oracle_beta"]),
            "successor_checkpoint_sha256": trace_a["policy"][
                "successor_checkpoint_sha256"
            ],
            "oracle_kind": trace_a["policy"]["oracle_kind"],
        },
        "trace_sources": {
            "dagger_a": {
                "path": str(trace_a_path),
                "sha256": _sha256(trace_a_path),
            },
            "dagger_b": {
                "path": str(trace_b_path),
                "sha256": _sha256(trace_b_path),
            },
        },
        "episode": {
            "observations": trace_a["observations"].float().clone(),
            "actions": trace_a["oracle_actions"].float().clone(),
            "student_actions": (
                trace_a["student_actions"].float().clone()
            ),
            "executed_actions": (
                trace_a["executed_actions"].float().clone()
            ),
            "rewards": trace_a["rewards"].float().clone(),
            "phases": trace_a["phases"].long().clone(),
            "safety_events": trace_a["safety_events"].bool().clone(),
            "frame_count": int(trace_a["observations"].shape[0]),
        },
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
    control_policy_kind = control_a.get("teacher_kind")
    if (
        control_policy_kind not in ALLOWED_CONTROL_POLICIES
        or control_b.get("teacher_kind") != control_policy_kind
    ):
        raise ValueError(
            "control traces must identify one frozen baseline or successor"
        )
    successor_checkpoint_sha256 = control_a["policy"].get(
        "successor_checkpoint_sha256"
    )
    if control_policy_kind == "successor_candidate":
        if not successor_checkpoint_sha256:
            raise ValueError(
                "successor controls lack the candidate checkpoint hash"
            )
    elif successor_checkpoint_sha256 is not None:
        raise ValueError(
            "frozen-baseline controls cannot contain a successor checkpoint"
        )
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
    schedule_uses_nominal = False
    rescue_stratum = None
    if teacher.get("teacher_kind") == "constrained_trajectory_optimizer":
        rescue_stratum = receipt.get("stratum")
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
            schedule_uses_nominal = bool(
                schedule_metadata.get("nominal_trace")
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
        or (
            not schedule_uses_nominal
            and schedule_segment_start != branch_frame
        )
        or (
            schedule_uses_nominal
            and schedule_segment_start < branch_frame
        )
    ):
        raise ValueError(
            "recorded action schedule does not match the observed branch"
        )
    if selected_schedule and rescue_stratum not in {
        "phase0",
        "custody",
    }:
        raise ValueError(
            "optimizer receipt lacks a recognized rescue stratum"
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
        "rescue_stratum": rescue_stratum,
        "control_policy_kind": control_policy_kind,
        "teacher_receipt": copy.deepcopy(receipt),
        "branch_frame": branch_frame,
        "control_outcome": control_outcome,
        "teacher_outcome": teacher_outcome,
        "source": copy.deepcopy(teacher["runtime"]["source"]),
        "base_checkpoint_sha256": teacher["policy"]["base_checkpoint_sha256"],
        "successor_checkpoint_sha256": (
            teacher["policy"].get("successor_checkpoint_sha256")
        ),
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


def _equal_nested_payload(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(
                _equal_nested_payload(left[key], right[key])
                for key in left
            )
        )
    if isinstance(left, (list, tuple)) or isinstance(
        right,
        (list, tuple),
    ):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(
                _equal_nested_payload(left_item, right_item)
                for left_item, right_item in zip(
                    left,
                    right,
                    strict=True,
                )
            )
        )
    return left == right


def migrate_scaffold_dataset(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Bind an accepted scaffold to an unchanged later task contract."""

    source_path = Path(args.dataset).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    payload = _load_accepted_dataset(source_path)
    if payload["schema_version"] != DATASET_SCHEMA_V1:
        raise ValueError("only preserved v1 scaffolds may be migrated")
    if payload["label_source"] not in {
        BASELINE_LABEL_SOURCE,
        DAGGER_LABEL_SOURCE,
    }:
        raise ValueError("independent rescue labels may not be migrated")
    original_revision = payload["source"]["dranmar_revision"]
    repo_root = Path(__file__).resolve().parents[1]
    target_revision = _resolve_revision(
        repo_root,
        args.target_revision,
    )
    original_contract = _handover_contract_manifest(
        original_revision
    )
    target_contract = _handover_contract_manifest(target_revision)
    if original_contract["sha256"] != target_contract["sha256"]:
        raise ValueError(
            "handover environment contract changed; scaffold recapture "
            "is required"
        )

    migrated = copy.deepcopy(payload)
    migrated["schema_version"] = DATASET_SCHEMA_V2
    migrated["source"] = copy.deepcopy(payload["source"])
    migrated["source"]["collection_dranmar_revision"] = (
        original_revision
    )
    migrated["source"]["dranmar_revision"] = target_revision
    migrated["compatibility"] = {
        "schema_version": SCAFFOLD_COMPATIBILITY_SCHEMA,
        "original_dataset": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "schema_version": DATASET_SCHEMA_V1,
        },
        "collection_dranmar_revision": original_revision,
        "training_contract_dranmar_revision": target_revision,
        "handover_contract_sha256": original_contract["sha256"],
        "contract_entries": original_contract["entries"],
        "migration_runtime_revision": _resolve_revision(
            repo_root,
            "HEAD",
        ),
    }
    migrated["gates"] = copy.deepcopy(payload["gates"])
    migrated["gates"].update(
        {
            "original_dataset_hash_verified": True,
            "handover_environment_contract_unchanged": True,
            "original_collection_revision_preserved": True,
        }
    )
    _atomic_torch_save(migrated, output)
    return {
        "schema_version": DATASET_SCHEMA_V2,
        "output": str(output),
        "sha256": _sha256(output),
        "pair_id": migrated["pair_id"],
        "label_source": migrated["label_source"],
        "collection_dranmar_revision": original_revision,
        "training_contract_dranmar_revision": target_revision,
        "handover_contract_sha256": original_contract["sha256"],
    }


def _validate_migrated_scaffold(
    path: Path,
    payload: dict[str, Any],
) -> None:
    compatibility = payload.get("compatibility")
    if (
        payload.get("label_source")
        not in {BASELINE_LABEL_SOURCE, DAGGER_LABEL_SOURCE}
        or not isinstance(compatibility, dict)
        or compatibility.get("schema_version")
        != SCAFFOLD_COMPATIBILITY_SCHEMA
    ):
        raise ValueError(f"invalid migrated scaffold contract: {path}")
    source = payload.get("source", {})
    collection_revision = compatibility.get(
        "collection_dranmar_revision"
    )
    training_revision = compatibility.get(
        "training_contract_dranmar_revision"
    )
    if (
        source.get("collection_dranmar_revision")
        != collection_revision
        or source.get("dranmar_revision") != training_revision
    ):
        raise ValueError(
            f"migrated scaffold source revisions drifted: {path}"
        )
    original = compatibility.get("original_dataset", {})
    original_path = Path(str(original.get("path", "")))
    if (
        not original_path.is_file()
        or original.get("schema_version") != DATASET_SCHEMA_V1
        or _sha256(original_path) != original.get("sha256")
    ):
        raise ValueError(
            f"migrated scaffold original artifact mismatch: {path}"
        )
    original_payload = torch.load(
        original_path,
        map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(original_payload, dict)
        or original_payload.get("schema_version")
        != DATASET_SCHEMA_V1
        or original_payload.get("source", {}).get(
            "dranmar_revision"
        )
        != collection_revision
    ):
        raise ValueError(
            f"migrated scaffold original provenance drifted: {path}"
        )
    for key in (
        "accepted",
        "label_source",
        "pair_id",
        "task",
        "seed",
        "teacher_kind",
        "base_checkpoint_sha256",
        "episode",
    ):
        if not _equal_nested_payload(
            payload.get(key),
            original_payload.get(key),
        ):
            raise ValueError(
                f"migrated scaffold content differs from original: {path}"
            )
    original_contract = _handover_contract_manifest(
        str(collection_revision)
    )
    training_contract = _handover_contract_manifest(
        str(training_revision)
    )
    expected_contract = compatibility.get(
        "handover_contract_sha256"
    )
    if (
        original_contract["sha256"] != expected_contract
        or training_contract["sha256"] != expected_contract
        or compatibility.get("contract_entries")
        != original_contract["entries"]
    ):
        raise ValueError(
            f"migrated scaffold task contract mismatch: {path}"
        )


def _load_accepted_dataset(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version")
        not in {DATASET_SCHEMA_V1, DATASET_SCHEMA_V2}
    ):
        raise ValueError(f"unsupported accepted successor dataset: {path}")
    if payload["schema_version"] == DATASET_SCHEMA_V2:
        _validate_migrated_scaffold(path, payload)
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


def _teacher_training_start_frame(
    payload: dict[str, Any],
    frame_count: int,
) -> int | None:
    """Locate the observed branch where the accepted teacher takes over.

    The optimizer receipt records where its residual schedule begins, but a
    nominal-guided safe teacher can already differ from the failed candidate
    before that residual window.  Weight the complete observed
    branch-to-terminal interval so those earlier corrective actions are not
    diluted by easy frames.
    """

    if payload.get("label_source") != TEACHER_LABEL_SOURCE:
        return None
    branch_frame = int(payload.get("branch_frame", -1))
    if not 0 <= branch_frame < frame_count:
        return None
    if payload.get("teacher_kind") == "constrained_trajectory_optimizer":
        receipt = payload.get("teacher_receipt")
        optimizer_start = (
            receipt.get("optimization_start_frame")
            if isinstance(receipt, dict)
            else None
        )
        if optimizer_start is not None:
            if (
                not isinstance(optimizer_start, int)
                or not branch_frame <= optimizer_start < frame_count
            ):
                raise ValueError(
                    "optimizer training start is outside its accepted episode"
                )
    return branch_frame


def _validation_checkpoint_improved(
    validation_loss: float,
    validation_gripper_errors: int,
    validation_saturation_errors: int,
    *,
    best_validation_loss: float,
    best_validation_gripper_errors: int,
    best_validation_saturation_errors: int,
) -> bool:
    """Improve imitation without regressing either discrete action contract."""

    tolerance = 1.0e-7
    if (
        validation_gripper_errors > best_validation_gripper_errors
        or validation_saturation_errors > best_validation_saturation_errors
    ):
        return False
    if validation_loss < best_validation_loss - tolerance:
        return True
    if abs(validation_loss - best_validation_loss) > tolerance:
        return False
    return (
        validation_gripper_errors,
        validation_saturation_errors,
    ) < (
        best_validation_gripper_errors,
        best_validation_saturation_errors,
    )


def _payload_label_signatures(
    payload: dict[str, Any],
    continuous_indices: tuple[int, ...],
    gripper_indices: tuple[int, ...],
) -> set[tuple[str, int, int, int]]:
    actions = payload["episode"]["actions"].float()
    phases = payload["episode"]["phases"].long()
    continuous_actions = actions[:, continuous_indices]
    saturation_classes = torch.ones(
        continuous_actions.shape,
        dtype=torch.long,
    )
    saturation_classes[continuous_actions <= -0.999] = 0
    saturation_classes[continuous_actions >= 0.999] = 2
    gripper_classes = (
        actions[:, gripper_indices] > 0.0
    ).long()
    signatures: set[tuple[str, int, int, int]] = set()
    for phase in range(4):
        phase_mask = phases == phase
        for action_offset in range(len(continuous_indices)):
            for label in torch.unique(
                saturation_classes[phase_mask, action_offset]
            ).tolist():
                signatures.add(
                    ("saturation", phase, action_offset, int(label))
                )
        for gripper in range(2):
            for label in torch.unique(
                gripper_classes[phase_mask, gripper]
            ).tolist():
                signatures.add(
                    ("gripper", phase, gripper, int(label))
                )
    return signatures


def _stable_validation_seeds(
    datasets: list[dict[str, Any]],
    fraction: float,
    *,
    continuous_indices: tuple[int, ...],
    gripper_indices: tuple[int, ...],
) -> set[int]:
    seeds = [int(payload["seed"]) for payload in datasets]
    ranked = sorted(
        set(seeds),
        key=lambda value: hashlib.sha256(str(value).encode()).digest(),
    )
    if len(ranked) < 4:
        raise ValueError("seed-grouped validation requires four seeds")
    count = max(1, round(len(ranked) * fraction))
    count = min(count, len(ranked) - 2)
    signature_seeds: dict[
        tuple[str, int, int, int],
        set[int],
    ] = {}
    for payload in datasets:
        seed = int(payload["seed"])
        for signature in _payload_label_signatures(
            payload,
            continuous_indices,
            gripper_indices,
        ):
            signature_seeds.setdefault(signature, set()).add(seed)

    selected: set[int] = set()
    for seed in ranked:
        candidate = selected | {seed}
        strands_validation_label = any(
            supporting_seeds & candidate
            and not supporting_seeds - candidate
            for supporting_seeds in signature_seeds.values()
        )
        if strands_validation_label:
            continue
        selected = candidate
        if len(selected) == count:
            return selected
    raise ValueError(
        "cannot build a seed-grouped validation split without "
        "stranding an action class"
    )


def _source_balanced_epoch_indices(
    label_sources: list[str],
    *,
    generator: torch.Generator,
) -> list[int]:
    """Sample episodes without letting long incumbent traces swamp rescues."""

    groups = {
        source: [
            index
            for index, observed in enumerate(label_sources)
            if observed == source
        ]
        for source in ALLOWED_LABEL_SOURCES
    }
    active = {source: values for source, values in groups.items() if values}
    if len(active) < 2:
        return torch.randperm(
            len(label_sources),
            generator=generator,
        ).tolist()

    epoch_size = max(len(label_sources), 8)
    if set(active) == {
        BASELINE_LABEL_SOURCE,
        DAGGER_LABEL_SOURCE,
        TEACHER_LABEL_SOURCE,
    }:
        requested = {
            TEACHER_LABEL_SOURCE: (epoch_size + 1) // 2,
            BASELINE_LABEL_SOURCE: epoch_size // 4,
        }
        requested[DAGGER_LABEL_SOURCE] = (
            epoch_size - sum(requested.values())
        )
    else:
        base = epoch_size // len(active)
        requested = {source: base for source in active}
        for source in sorted(active)[: epoch_size - base * len(active)]:
            requested[source] += 1

    sampled: list[int] = []
    for source, count in requested.items():
        candidates = active[source]
        selections = torch.randint(
            len(candidates),
            (count,),
            generator=generator,
        ).tolist()
        sampled.extend(candidates[index] for index in selections)
    order = torch.randperm(
        len(sampled),
        generator=generator,
    ).tolist()
    return [sampled[index] for index in order]


def _training_example_ids(
    paths: list[Path],
    datasets: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Give repeated rescue campaigns distinct, provenance-bound identities."""

    if len(paths) != len(datasets):
        raise ValueError("dataset paths and payloads do not align")
    artifact_sha256 = [_sha256(path) for path in paths]
    if len(set(artifact_sha256)) != len(artifact_sha256):
        raise ValueError("training data repeats an accepted artifact")

    pair_groups: dict[str, list[int]] = {}
    for index, payload in enumerate(datasets):
        pair_groups.setdefault(str(payload["pair_id"]), []).append(index)
    example_ids = [str(payload["pair_id"]) for payload in datasets]
    for pair_id, indices in pair_groups.items():
        if len(indices) == 1:
            continue
        collided = [datasets[index] for index in indices]
        if (
            any(
                payload["label_source"] != TEACHER_LABEL_SOURCE
                or payload.get("control_policy_kind")
                != "successor_candidate"
                for payload in collided
            )
            or len({int(payload["seed"]) for payload in collided}) != 1
            or len({str(payload["task"]) for payload in collided}) != 1
        ):
            raise ValueError(
                "only repeated successor-candidate rescue campaigns may "
                "share a pair id"
            )
        candidate_hashes = [
            str(payload.get("successor_checkpoint_sha256") or "")
            for payload in collided
        ]
        if (
            "" in candidate_hashes
            or len(set(candidate_hashes)) != len(candidate_hashes)
        ):
            raise ValueError(
                "colliding rescue pairs must come from distinct frozen "
                "candidate checkpoints"
            )
        for index, candidate_hash in zip(
            indices,
            candidate_hashes,
            strict=True,
        ):
            example_ids[index] = (
                f"{pair_id}@successor-{candidate_hash[:16]}"
            )
    if len(set(example_ids)) != len(example_ids):
        raise AssertionError("training example identity is not unique")
    return example_ids, artifact_sha256


def train_successor(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    runtime_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
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
    HANDOVER_CONTINUOUS_INDICES = (
        policy_module.HANDOVER_CONTINUOUS_INDICES
    )
    HANDOVER_GRIPPER_INDICES = policy_module.HANDOVER_GRIPPER_INDICES
    HANDOVER_OBSERVATION_DIM = policy_module.HANDOVER_OBSERVATION_DIM
    HANDOVER_PHASE_SLICE = policy_module.HANDOVER_PHASE_SLICE
    HANDOVER_SATURATION_CLASS_COUNT = (
        policy_module.HANDOVER_SATURATION_CLASS_COUNT
    )
    HANDOVER_SATURATION_LOGIT_MARGIN = (
        policy_module.HANDOVER_SATURATION_LOGIT_MARGIN
    )
    SUCCESSOR_CHECKPOINT_SCHEMA = policy_module.SUCCESSOR_CHECKPOINT_SCHEMA
    PhaseConditionedHandoverPolicy = (
        policy_module.PhaseConditionedHandoverPolicy
    )
    continuous_action_indices = list(HANDOVER_CONTINUOUS_INDICES)

    if HANDOVER_OBSERVATION_DIM != OBSERVATION_DIM or HANDOVER_ACTION_DIM != ACTION_DIM:
        raise ValueError("trainer and successor policy contracts disagree")
    if (HANDOVER_PHASE_SLICE.start, HANDOVER_PHASE_SLICE.stop) != (
        PHASE_SLICE.start,
        PHASE_SLICE.stop,
    ):
        raise ValueError("trainer and successor phase contracts disagree")
    if HANDOVER_SATURATION_CLASS_COUNT != 3:
        raise ValueError("trainer and successor saturation contracts disagree")
    if (
        args.epochs <= 0
        or args.episode_batch_size <= 0
    ):
        raise ValueError("epochs and episode batch size must be positive")
    if not 0.1 <= args.validation_fraction <= 0.4:
        raise ValueError("validation fraction must be in [0.1, 0.4]")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("optimizer parameters are invalid")
    if args.patience <= 0:
        raise ValueError("early-stopping patience must be positive")
    if args.memory_dim <= 0:
        raise ValueError("memory dimension must be positive")

    paths = [Path(value).expanduser().resolve() for value in args.dataset]
    datasets = [_load_accepted_dataset(path) for path in paths]
    if len(datasets) < 8:
        raise ValueError(
            "training requires at least eight accepted successor examples"
        )
    example_ids, dataset_artifact_sha256 = _training_example_ids(
        paths,
        datasets,
    )
    example_id_by_payload = {
        id(payload): example_id
        for payload, example_id in zip(
            datasets,
            example_ids,
            strict=True,
        )
    }
    artifact_sha256_by_payload = {
        id(payload): artifact_sha256
        for payload, artifact_sha256 in zip(
            datasets,
            dataset_artifact_sha256,
            strict=True,
        )
    }
    if len({int(payload["seed"]) for payload in datasets}) < 4:
        raise ValueError("training requires at least four distinct development seeds")
    baseline_pair_count = sum(
        payload["label_source"] == BASELINE_LABEL_SOURCE
        for payload in datasets
    )
    dagger_pair_count = sum(
        payload["label_source"] == DAGGER_LABEL_SOURCE
        for payload in datasets
    )
    teacher_pair_count = sum(
        payload["label_source"] == TEACHER_LABEL_SOURCE
        for payload in datasets
    )
    if getattr(args, "completion_gate", False):
        teacher_payloads = [
            payload
            for payload in datasets
            if payload["label_source"] == TEACHER_LABEL_SOURCE
        ]
        teacher_seed_counts: dict[int, int] = {}
        phase_zero_rescues = 0
        custody_rescues = 0
        for payload in teacher_payloads:
            seed = int(payload["seed"])
            teacher_seed_counts[seed] = (
                teacher_seed_counts.get(seed, 0) + 1
            )
            branch_frame = int(payload.get("branch_frame", -1))
            phases = payload["episode"]["phases"].long()
            if not 0 <= branch_frame < phases.shape[0]:
                raise ValueError(
                    "teacher rescue lacks a valid branch frame"
                )
            branch_phase = int(phases[branch_frame].item())
            rescue_stratum = payload.get("rescue_stratum")
            if rescue_stratum == "phase0" or (
                rescue_stratum is None and branch_phase == 0
            ):
                phase_zero_rescues += 1
            legacy_custody_rescue = (
                rescue_stratum is None
                and (
                    branch_phase >= 2
                    or payload.get("control_outcome")
                    in {
                        "needle_dropped_after_pickup",
                        "premature_giver_release",
                        "receiver_retention_lost",
                    }
                )
            )
            if (
                rescue_stratum == "custody"
                or legacy_custody_rescue
            ):
                custody_rescues += 1
        if any(count > 2 for count in teacher_seed_counts.values()):
            raise ValueError(
                "completion data admits at most two teacher rescues per seed"
            )
        if (
            teacher_pair_count < 64
            or len(teacher_seed_counts) < 32
            or phase_zero_rescues < 32
            or custody_rescues < 32
        ):
            raise ValueError(
                "completion requires 64 teacher rescues across 32 seeds, "
                "including 32 phase-zero and 32 custody rescues"
            )
    tasks = {payload["task"] for payload in datasets}
    base_hashes = {payload["base_checkpoint_sha256"] for payload in datasets}
    source_revisions = {payload["source"]["dranmar_revision"] for payload in datasets}
    collection_source_revisions = {
        payload["source"].get(
            "collection_dranmar_revision",
            payload["source"]["dranmar_revision"],
        )
        for payload in datasets
    }
    asset_revisions = {payload["source"]["asset_revision"] for payload in datasets}
    if len(tasks) != 1:
        raise ValueError("accepted demonstrations do not share one task")
    if len(base_hashes) != 1:
        raise ValueError("accepted demonstrations do not share one baseline")
    if len(asset_revisions) != 1:
        raise ValueError("accepted demonstrations do not share one asset lock")
    source_contracts = {
        revision: _handover_contract_manifest(revision)["sha256"]
        for revision in source_revisions
    }
    environment_contracts = set(source_contracts.values())
    if len(environment_contracts) != 1:
        raise ValueError(
            "accepted demonstrations span incompatible environment contracts"
        )
    environment_contract_sha256 = next(iter(environment_contracts))

    validation_seeds = _stable_validation_seeds(
        datasets,
        args.validation_fraction,
        continuous_indices=HANDOVER_CONTINUOUS_INDICES,
        gripper_indices=HANDOVER_GRIPPER_INDICES,
    )
    train_payloads = [
        payload
        for payload in datasets
        if int(payload["seed"]) not in validation_seeds
    ]
    validation_payloads = [
        payload
        for payload in datasets
        if int(payload["seed"]) in validation_seeds
    ]
    if len(train_payloads) < 2 or len(validation_payloads) < 2:
        raise ValueError("episode-level split is too small")

    train_episode_observations = [
        payload["episode"]["observations"].float()
        for payload in train_payloads
    ]
    train_episode_actions = [
        payload["episode"]["actions"].float()
        for payload in train_payloads
    ]
    train_episode_phases = [
        payload["episode"]["phases"].long()
        for payload in train_payloads
    ]
    train_episode_weights = []
    teacher_training_start_frames: dict[str, int] = {}
    for payload in train_payloads:
        frame_count = int(payload["episode"]["phases"].shape[0])
        weights = torch.ones(frame_count)
        if payload["label_source"] == TEACHER_LABEL_SOURCE:
            teacher_receipt = payload.get("teacher_receipt")
            optimizer_start = (
                teacher_receipt.get("optimization_start_frame")
                if isinstance(teacher_receipt, dict)
                else None
            )
            if (
                getattr(args, "completion_gate", False)
                and payload.get("teacher_kind")
                == "constrained_trajectory_optimizer"
                and not isinstance(optimizer_start, int)
            ):
                raise ValueError(
                    "completion optimizer rescue lacks its information "
                    "start frame"
                )
            training_start_frame = _teacher_training_start_frame(
                payload,
                frame_count,
            )
            if (
                getattr(args, "completion_gate", False)
                and training_start_frame is None
            ):
                raise ValueError(
                    "teacher rescue lacks a valid training start frame"
                )
            if training_start_frame is not None:
                weights[training_start_frame:] = 2.0
                teacher_training_start_frames[
                    example_id_by_payload[id(payload)]
                ] = training_start_frame
        train_episode_weights.append(weights)
    train_label_sources = [
        str(payload["label_source"]) for payload in train_payloads
    ]
    validation_episode_observations = [
        payload["episode"]["observations"].float()
        for payload in validation_payloads
    ]
    validation_episode_actions = [
        payload["episode"]["actions"].float()
        for payload in validation_payloads
    ]
    validation_episode_phases = [
        payload["episode"]["phases"].long()
        for payload in validation_payloads
    ]
    validation_episode_weights = []
    for payload in validation_payloads:
        frame_count = int(payload["episode"]["phases"].shape[0])
        weights = torch.ones(frame_count)
        if payload["label_source"] == TEACHER_LABEL_SOURCE:
            training_start_frame = _teacher_training_start_frame(
                payload,
                frame_count,
            )
            if (
                getattr(args, "completion_gate", False)
                and training_start_frame is None
            ):
                raise ValueError(
                    "validation teacher rescue lacks a valid training "
                    "start frame"
                )
            if training_start_frame is not None:
                weights[training_start_frame:] = 2.0
        validation_episode_weights.append(weights)
    train_observations = torch.cat(
        train_episode_observations
    )
    train_actions = torch.cat(
        train_episode_actions
    )
    train_phases = torch.cat(
        train_episode_phases
    )
    validation_observations = torch.cat(
        validation_episode_observations
    )
    validation_actions = torch.cat(
        validation_episode_actions
    )
    validation_phases = torch.cat(
        validation_episode_phases
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
    train_continuous_actions = train_actions[
        :, HANDOVER_CONTINUOUS_INDICES
    ]
    train_saturation_class = torch.ones(
        train_continuous_actions.shape,
        dtype=torch.long,
    )
    train_saturation_class[
        train_continuous_actions <= -0.999
    ] = 0
    train_saturation_class[
        train_continuous_actions >= 0.999
    ] = 2
    action_loss_scale = torch.ones(ACTION_DIM)
    for offset, action_index in enumerate(
        HANDOVER_CONTINUOUS_INDICES
    ):
        precision_values = train_actions[
            train_saturation_class[:, offset] == 1,
            action_index,
        ].abs()
        if precision_values.numel():
            action_loss_scale[action_index] = torch.quantile(
                precision_values,
                0.90,
            ).clamp_min(0.01)
    saturation_class_counts = torch.zeros(
        5,
        len(HANDOVER_CONTINUOUS_INDICES),
        HANDOVER_SATURATION_CLASS_COUNT,
        dtype=torch.long,
    )
    for phase in range(5):
        phase_mask = train_phases == phase
        for action_offset in range(
            len(HANDOVER_CONTINUOUS_INDICES)
        ):
            saturation_class_counts[phase, action_offset] = (
                torch.bincount(
                    train_saturation_class[
                        phase_mask,
                        action_offset,
                    ],
                    minlength=HANDOVER_SATURATION_CLASS_COUNT,
                )
            )
    saturation_class_weight = (
        saturation_class_counts > 0
    ).float()
    gripper_class_counts = torch.zeros(5, 2, 2, dtype=torch.long)
    train_gripper_class = (
        train_actions[:, HANDOVER_GRIPPER_INDICES] > 0.0
    ).long()
    for phase in range(5):
        phase_mask = train_phases == phase
        for gripper in range(2):
            gripper_class_counts[phase, gripper] = torch.bincount(
                train_gripper_class[phase_mask, gripper],
                minlength=2,
            )
    gripper_class_weight = torch.zeros_like(
        gripper_class_counts,
        dtype=torch.float32,
    )
    for phase in range(5):
        for gripper in range(2):
            counts = gripper_class_counts[phase, gripper]
            present = counts > 0
            if bool(present.any()):
                gripper_class_weight[phase, gripper, present] = (
                    counts.sum()
                    / (present.sum() * counts[present])
                )
    validation_gripper_class = (
        validation_actions[:, HANDOVER_GRIPPER_INDICES] > 0.0
    ).long()
    validation_continuous_actions = validation_actions[
        :, HANDOVER_CONTINUOUS_INDICES
    ]
    validation_saturation_class = torch.ones(
        validation_continuous_actions.shape,
        dtype=torch.long,
    )
    validation_saturation_class[
        validation_continuous_actions <= -0.999
    ] = 0
    validation_saturation_class[
        validation_continuous_actions >= 0.999
    ] = 2
    for phase in range(4):
        phase_mask = validation_phases == phase
        for action_offset in range(
            len(HANDOVER_CONTINUOUS_INDICES)
        ):
            validation_classes = torch.bincount(
                validation_saturation_class[
                    phase_mask,
                    action_offset,
                ],
                minlength=HANDOVER_SATURATION_CLASS_COUNT,
            )
            if bool(
                (
                    (validation_classes > 0)
                    & (
                        saturation_class_counts[
                            phase,
                            action_offset,
                        ]
                        == 0
                    )
                ).any()
            ):
                raise ValueError(
                    "training split lacks a validation saturation class"
                )
        for gripper in range(2):
            validation_classes = torch.bincount(
                validation_gripper_class[phase_mask, gripper],
                minlength=2,
            )
            if bool(
                (
                    (validation_classes > 0)
                    & (gripper_class_counts[phase, gripper] == 0)
                ).any()
            ):
                raise ValueError(
                    "training split lacks a validation gripper class"
                )
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
        memory_dim=args.memory_dim,
        head_dim=args.head_dim,
    ).to(device)
    initial_checkpoint_path = None
    initial_checkpoint_payload = None
    if getattr(args, "initial_checkpoint", None):
        initial_checkpoint_path = (
            Path(args.initial_checkpoint).expanduser().resolve()
        )
        if not initial_checkpoint_path.is_file():
            raise ValueError(
                "initial successor checkpoint does not exist: "
                f"{initial_checkpoint_path}"
            )
        initial_checkpoint_payload = torch.load(
            initial_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if (
            not isinstance(initial_checkpoint_payload, dict)
            or initial_checkpoint_payload.get("schema_version")
            not in {
                SUCCESSOR_CHECKPOINT_SCHEMA_V3,
                LEGACY_SUCCESSOR_CHECKPOINT_SCHEMA_V2,
            }
            or initial_checkpoint_payload.get("deployment_status")
            != "candidate_only"
        ):
            raise ValueError(
                "initial successor checkpoint is not a preserved candidate"
            )
        initial_architecture = initial_checkpoint_payload.get(
            "architecture",
            {},
        )
        if (
            initial_checkpoint_payload.get("task") not in tasks
            or initial_checkpoint_payload.get("base_checkpoint_sha256")
            not in base_hashes
            or initial_architecture.get("hidden_dims")
            != list(hidden_dims)
            or int(initial_architecture.get("memory_dim", -1))
            != args.memory_dim
            or int(initial_architecture.get("head_dim", -1))
            != args.head_dim
            or initial_architecture.get("runtime_heuristic_stack")
            is not False
        ):
            raise ValueError(
                "initial successor checkpoint contract does not match training"
            )
        initial_source = initial_checkpoint_payload.get("source", {})
        initial_revision = initial_source.get("dranmar_revision")
        try:
            initial_contract_sha256 = _handover_contract_manifest(
                str(initial_revision)
            )["sha256"]
        except (subprocess.CalledProcessError, ValueError) as error:
            raise ValueError(
                "initial successor checkpoint source revision is unavailable"
            ) from error
        if (
            initial_source.get("asset_revision") not in asset_revisions
            or initial_contract_sha256 != environment_contract_sha256
        ):
            raise ValueError(
                "initial successor checkpoint provenance does not match data"
            )
        model.load_state_dict(
            initial_checkpoint_payload["model"],
            strict=True,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_observations = train_observations.to(device)
    train_actions = train_actions.to(device)
    train_phases = train_phases.to(device)
    train_episode_observations = [
        value.to(device) for value in train_episode_observations
    ]
    train_episode_actions = [
        value.to(device) for value in train_episode_actions
    ]
    train_episode_phases = [
        value.to(device) for value in train_episode_phases
    ]
    train_episode_weights = [
        value.to(device) for value in train_episode_weights
    ]
    validation_observations = validation_observations.to(device)
    validation_actions = validation_actions.to(device)
    validation_phases = validation_phases.to(device)
    validation_episode_observations = [
        value.to(device)
        for value in validation_episode_observations
    ]
    validation_episode_actions = [
        value.to(device) for value in validation_episode_actions
    ]
    validation_episode_phases = [
        value.to(device) for value in validation_episode_phases
    ]
    validation_episode_weights = [
        value.to(device) for value in validation_episode_weights
    ]
    action_loss_scale_device = action_loss_scale.to(device)
    saturation_class_weight_device = (
        saturation_class_weight.to(device)
    )
    continuous_index = torch.arange(
        len(HANDOVER_CONTINUOUS_INDICES),
        device=device,
    )
    gripper_class_weight_device = gripper_class_weight.to(device)
    gripper_index = torch.arange(2, device=device)
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
    discrete_transition_weight = 8.0
    gripper_logit_margin = 1.5
    gripper_loss_weight = 4.0

    def padded_episode_batch(
        observations: list[torch.Tensor],
        actions: list[torch.Tensor],
        phases: list[torch.Tensor],
        weights: list[torch.Tensor],
        indices: list[int],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        selected_observations = [
            observations[index] for index in indices
        ]
        selected_actions = [actions[index] for index in indices]
        selected_phases = [phases[index] for index in indices]
        selected_weights = [weights[index] for index in indices]
        lengths = torch.tensor(
            [value.shape[0] for value in selected_observations],
            device=device,
        )
        padded_observations = torch.nn.utils.rnn.pad_sequence(
            selected_observations,
            batch_first=True,
        )
        padded_actions = torch.nn.utils.rnn.pad_sequence(
            selected_actions,
            batch_first=True,
        )
        padded_phases = torch.nn.utils.rnn.pad_sequence(
            selected_phases,
            batch_first=True,
            padding_value=4,
        )
        padded_weights = torch.nn.utils.rnn.pad_sequence(
            selected_weights,
            batch_first=True,
        )
        valid = (
            torch.arange(
                padded_observations.shape[1],
                device=device,
            ).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        return (
            padded_observations,
            padded_actions,
            padded_phases,
            padded_weights,
            valid,
        )

    def hybrid_per_frame_loss(
        predicted: torch.Tensor,
        gripper_logits: torch.Tensor,
        saturation_logits: torch.Tensor,
        target: torch.Tensor,
        phases: torch.Tensor,
        gripper_transition_weights: torch.Tensor,
        saturation_transition_weights: torch.Tensor,
    ) -> torch.Tensor:
        continuous_target = target[
            :, continuous_action_indices
        ]
        saturation_target = torch.ones(
            continuous_target.shape,
            dtype=torch.long,
            device=device,
        )
        saturation_target[continuous_target <= -0.999] = 0
        saturation_target[continuous_target >= 0.999] = 2
        precision_mask = (saturation_target == 1).float()
        continuous_per_action = (
            (
                predicted[:, continuous_action_indices]
                - continuous_target
            )
            / action_loss_scale_device[continuous_action_indices]
        ).square()
        masked_continuous_loss = (
            continuous_per_action * precision_mask
        )
        continuous_loss = (
            masked_continuous_loss.sum(dim=-1)
            / precision_mask.sum(dim=-1).clamp_min(1.0)
            + 0.25 * masked_continuous_loss.amax(dim=-1)
        )
        saturation_per_action = functional.cross_entropy(
            saturation_logits.flatten(0, 1),
            saturation_target.flatten(),
            reduction="none",
        ).view(target.shape[0], -1)
        saturation_sample_weight = (
            saturation_class_weight_device[
                phases.unsqueeze(-1),
                continuous_index,
                saturation_target,
            ]
        )
        saturation_loss = (
            saturation_per_action
            * saturation_sample_weight
            * saturation_transition_weights
        ).mean(dim=-1)
        gripper_targets = (
            target[:, HANDOVER_GRIPPER_INDICES] > 0.0
        ).float()
        signed_gripper_target = 2.0 * gripper_targets - 1.0
        gripper_per_action = functional.softplus(
            gripper_logit_margin
            - signed_gripper_target * gripper_logits
        )
        gripper_sample_weight = gripper_class_weight_device[
            phases.unsqueeze(-1),
            gripper_index,
            gripper_targets.long(),
        ]
        gripper_loss = (
            gripper_per_action
            * gripper_sample_weight
            * gripper_transition_weights
        ).mean(dim=-1)
        return (
            continuous_loss
            + saturation_loss
            + gripper_loss_weight * gripper_loss
        )

    def discrete_transition_weights(
        action_sequences: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        continuous_actions = action_sequences[
            :, :, continuous_action_indices
        ]
        saturation_target = torch.ones(
            continuous_actions.shape,
            dtype=torch.long,
            device=device,
        )
        saturation_target[continuous_actions <= -0.999] = 0
        saturation_target[continuous_actions >= 0.999] = 2
        saturation_transition = torch.zeros_like(
            saturation_target,
            dtype=torch.bool,
        )
        saturation_transition[:, 1:] = (
            saturation_target[:, 1:]
            != saturation_target[:, :-1]
        )
        gripper_target = (
            action_sequences[:, :, HANDOVER_GRIPPER_INDICES] > 0.0
        )
        gripper_transition = torch.zeros_like(
            gripper_target,
            dtype=torch.bool,
        )
        gripper_transition[:, 1:] = (
            gripper_target[:, 1:]
            != gripper_target[:, :-1]
        )
        return (
            torch.where(
                gripper_transition,
                discrete_transition_weight,
                1.0,
            ),
            torch.where(
                saturation_transition,
                discrete_transition_weight,
                1.0,
            ),
        )

    validation_sequence_batch = padded_episode_batch(
        validation_episode_observations,
        validation_episode_actions,
        validation_episode_phases,
        validation_episode_weights,
        list(range(len(validation_episode_observations))),
    )

    def evaluate_validation() -> tuple[float, float, int, int]:
        model.eval()
        with torch.inference_mode():
            (
                validation_observation_sequences,
                validation_action_sequences,
                validation_phase_sequences,
                validation_frame_weight_sequences,
                validation_valid_sequences,
            ) = validation_sequence_batch
            (
                validation_prediction_soft_sequences,
                validation_gripper_logit_sequences,
                validation_saturation_logit_sequences,
            ) = model.training_sequence_outputs(
                validation_observation_sequences
            )
            validation_prediction_soft = (
                validation_prediction_soft_sequences[
                    validation_valid_sequences
                ]
            )
            validation_gripper_logits = (
                validation_gripper_logit_sequences[
                    validation_valid_sequences
                ]
            )
            validation_saturation_logits = (
                validation_saturation_logit_sequences[
                    validation_valid_sequences
                ]
            )
            validation_target = validation_action_sequences[
                validation_valid_sequences
            ]
            validation_phase = validation_phase_sequences[
                validation_valid_sequences
            ]
            (
                validation_gripper_transition_sequences,
                validation_saturation_transition_sequences,
            ) = discrete_transition_weights(
                validation_action_sequences
            )
            validation_per_frame = hybrid_per_frame_loss(
                validation_prediction_soft,
                validation_gripper_logits,
                validation_saturation_logits,
                validation_target,
                validation_phase,
                validation_gripper_transition_sequences[
                    validation_valid_sequences
                ],
                validation_saturation_transition_sequences[
                    validation_valid_sequences
                ],
            )
            validation_loss = float(
                (
                    validation_per_frame
                    * phase_weight[validation_phase]
                    * validation_frame_weight_sequences[
                        validation_valid_sequences
                    ]
                )
                .mean()
                .item()
            )
            validation_prediction = (
                model.training_sequence_actions(
                    validation_observation_sequences
                )[validation_valid_sequences]
            )
            validation_mae = float(
                (
                    validation_prediction - validation_target
                ).abs().mean().item()
            )
            validation_gripper_error_count = int(
                (
                    (validation_gripper_logits >= 0.0)
                    != (
                        validation_target[
                            :, HANDOVER_GRIPPER_INDICES
                        ]
                        > 0.0
                    )
                )
                .sum()
                .item()
            )
            validation_continuous_target = validation_target[
                :, continuous_action_indices
            ]
            validation_saturation_target = torch.ones(
                validation_continuous_target.shape,
                dtype=torch.long,
                device=device,
            )
            validation_saturation_target[
                validation_continuous_target <= -0.999
            ] = 0
            validation_saturation_target[
                validation_continuous_target >= 0.999
            ] = 2
            validation_saturation_prediction = torch.ones_like(
                validation_saturation_target
            )
            validation_precision_logit = (
                validation_saturation_logits[:, :, 1]
            )
            validation_saturation_prediction[
                validation_saturation_logits[:, :, 0]
                >= (
                    validation_precision_logit
                    + HANDOVER_SATURATION_LOGIT_MARGIN
                )
            ] = 0
            validation_saturation_prediction[
                validation_saturation_logits[:, :, 2]
                >= (
                    validation_precision_logit
                    + HANDOVER_SATURATION_LOGIT_MARGIN
                )
            ] = 2
            validation_saturation_error_count = int(
                (
                    validation_saturation_prediction
                    != validation_saturation_target
                )
                .sum()
                .item()
            )
        return (
            validation_loss,
            validation_mae,
            validation_gripper_error_count,
            validation_saturation_error_count,
        )

    best_validation_loss = float("inf")
    best_validation_gripper_errors = sys.maxsize
    best_validation_saturation_errors = sys.maxsize
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    episode_generator = torch.Generator().manual_seed(args.seed)
    if initial_checkpoint_payload is not None:
        (
            initial_validation_loss,
            initial_validation_mae,
            initial_validation_gripper_errors,
            initial_validation_saturation_errors,
        ) = evaluate_validation()
        best_validation_gripper_errors = (
            initial_validation_gripper_errors
        )
        best_validation_saturation_errors = (
            initial_validation_saturation_errors
        )
        best_validation_loss = initial_validation_loss
        best_epoch = 0
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        history.append(
            {
                "epoch": 0,
                "train_loss": float("nan"),
                "validation_loss": initial_validation_loss,
                "validation_mae": initial_validation_mae,
                "validation_gripper_errors": (
                    initial_validation_gripper_errors
                ),
                "validation_saturation_errors": (
                    initial_validation_saturation_errors
                ),
            }
        )

    for epoch in range(args.epochs):
        model.train()
        episode_order = _source_balanced_epoch_indices(
            train_label_sources,
            generator=episode_generator,
        )
        loss_sum = 0.0
        sample_count = 0
        for start in range(
            0,
            len(episode_order),
            args.episode_batch_size,
        ):
            episode_indices = episode_order[
                start : start + args.episode_batch_size
            ]
            (
                observation_sequences,
                action_sequences,
                phase_sequences,
                frame_weight_sequences,
                valid_sequences,
            ) = padded_episode_batch(
                train_episode_observations,
                train_episode_actions,
                train_episode_phases,
                train_episode_weights,
                episode_indices,
            )
            (
                predicted_sequences,
                gripper_logit_sequences,
                saturation_logit_sequences,
            ) = model.training_sequence_outputs(
                observation_sequences
            )
            predicted = predicted_sequences[valid_sequences]
            gripper_logits = gripper_logit_sequences[
                valid_sequences
            ]
            saturation_logits = saturation_logit_sequences[
                valid_sequences
            ]
            target = action_sequences[valid_sequences]
            phases = phase_sequences[valid_sequences]
            (
                gripper_transition_sequences,
                saturation_transition_sequences,
            ) = discrete_transition_weights(action_sequences)
            per_frame = hybrid_per_frame_loss(
                predicted,
                gripper_logits,
                saturation_logits,
                target,
                phases,
                gripper_transition_sequences[valid_sequences],
                saturation_transition_sequences[valid_sequences],
            )
            frame_weights = frame_weight_sequences[valid_sequences]
            weighted_loss = (
                per_frame
                * phase_weight[phases]
                * frame_weights
            )
            loss = weighted_loss.sum() / frame_weights.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item()) * target.shape[0]
            sample_count += target.shape[0]

        (
            validation_loss,
            validation_mae,
            validation_gripper_errors,
            validation_saturation_errors,
        ) = evaluate_validation()
        train_loss = loss_sum / sample_count
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_mae": validation_mae,
                "validation_gripper_errors": (
                    validation_gripper_errors
                ),
                "validation_saturation_errors": (
                    validation_saturation_errors
                ),
            }
        )
        validation_improved = _validation_checkpoint_improved(
            validation_loss,
            validation_gripper_errors,
            validation_saturation_errors,
            best_validation_loss=best_validation_loss,
            best_validation_gripper_errors=(
                best_validation_gripper_errors
            ),
            best_validation_saturation_errors=(
                best_validation_saturation_errors
            ),
        )
        if validation_improved:
            best_validation_loss = validation_loss
            best_validation_gripper_errors = (
                validation_gripper_errors
            )
            best_validation_saturation_errors = (
                validation_saturation_errors
            )
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
    (
        final_validation_loss,
        _selected_validation_mae,
        final_validation_gripper_errors,
        final_validation_saturation_errors,
    ) = evaluate_validation()
    with torch.inference_mode():
        (
            final_validation_observations,
            final_validation_actions,
            final_validation_phases,
            _final_validation_weights,
            final_validation_valid,
        ) = validation_sequence_batch
        final_prediction = model.training_sequence_actions(
            final_validation_observations
        )[final_validation_valid]
        final_target = final_validation_actions[
            final_validation_valid
        ]
        final_phase = final_validation_phases[
            final_validation_valid
        ]
        final_validation_mae = float(
            (final_prediction - final_target).abs().mean().item()
        )
        final_max_abs_error = float(
            (final_prediction - final_target).abs().max().item()
        )
        final_phase_mae = {
            str(phase): (
                float(
                    (
                        final_prediction[final_phase == phase]
                        - final_target[final_phase == phase]
                    )
                    .abs()
                    .mean()
                    .item()
                )
                if bool((final_phase == phase).any())
                else None
            )
            for phase in range(5)
        }

    output = Path(args.output).expanduser().resolve()
    if teacher_pair_count and dagger_pair_count:
        capability_scope = (
            "incumbent_on_policy_distillation_plus_teacher_rescues"
        )
    elif baseline_pair_count and teacher_pair_count:
        capability_scope = "incumbent_distillation_plus_teacher_rescues"
    elif teacher_pair_count:
        capability_scope = "teacher_rescue_only"
    elif dagger_pair_count:
        capability_scope = "incumbent_on_policy_distillation"
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
            "kind": (
                "shared_encoder_gru_with_five_hybrid_phase_heads"
            ),
            "hidden_dims": list(hidden_dims),
            "memory_dim": int(args.memory_dim),
            "head_dim": int(args.head_dim),
            "full_action_policy": True,
            "binary_gripper_indices": list(HANDOVER_GRIPPER_INDICES),
            "continuous_action_indices": list(
                HANDOVER_CONTINUOUS_INDICES
            ),
            "saturation_classes": [
                "negative_limit",
                "precision",
                "positive_limit",
            ],
            "saturation_logit_margin": (
                HANDOVER_SATURATION_LOGIT_MARGIN
            ),
            "recurrent_state": "gru_reset_per_episode",
            "runtime_heuristic_stack": False,
            "terminal_phase_action": "zero_initialized_no_training_required",
        },
        "observation_mean": best_state["observation_mean"].clone(),
        "observation_std": best_state["observation_std"].clone(),
        "action_loss_scale": action_loss_scale.cpu(),
        "model": best_state,
        "base_checkpoint_sha256": next(iter(base_hashes)),
        "source": {
            "dranmar_revision": runtime_revision,
            "training_data_dranmar_revision": (
                next(iter(source_revisions))
                if len(source_revisions) == 1
                else None
            ),
            "training_data_dranmar_revisions": sorted(
                source_revisions
            ),
            "training_data_collection_revisions": sorted(
                collection_source_revisions
            ),
            "training_data_environment_contract_sha256": (
                environment_contract_sha256
            ),
            "asset_revision": next(iter(asset_revisions)),
        },
        "training": {
            "seed": int(args.seed),
            "initial_checkpoint": (
                {
                    "path": str(initial_checkpoint_path),
                    "sha256": _sha256(initial_checkpoint_path),
                    "schema_version": initial_checkpoint_payload[
                        "schema_version"
                    ],
                }
                if initial_checkpoint_path is not None
                and initial_checkpoint_payload is not None
                else None
            ),
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "episode_batch_size": int(args.episode_batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "loss": (
                "recurrent_phase_balanced_tail_aware_precision_mse_plus_"
                "conservative_saturation_cross_entropy_plus_"
                "phase_class_balanced_margin_gripper_classification"
            ),
            "phase_counts": phase_counts.tolist(),
            "saturation_class_counts": (
                saturation_class_counts.tolist()
            ),
            "gripper_class_counts": gripper_class_counts.tolist(),
            "discrete_transition_weight": (
                discrete_transition_weight
            ),
            "gripper_logit_margin": gripper_logit_margin,
            "gripper_loss_weight": gripper_loss_weight,
            "teacher_information_to_terminal_weight": 2.0,
            "teacher_information_start": (
                "observed_teacher_branch_frame"
            ),
            "teacher_training_start_frames": (
                teacher_training_start_frames
            ),
            "episode_source_sampling": {
                TEACHER_LABEL_SOURCE: 0.5,
                BASELINE_LABEL_SOURCE: 0.25,
                DAGGER_LABEL_SOURCE: 0.25,
            },
            "training_example_identity": (
                "pair_id_else_pair_id_at_successor_checkpoint_prefix_for_"
                "repeated_candidate_rescue_campaigns"
            ),
            "accepted_datasets": [
                {
                    "path": str(path),
                    "sha256": artifact_sha256_by_payload[id(payload)],
                    "example_id": example_id_by_payload[id(payload)],
                    "pair_id": str(payload["pair_id"]),
                    "seed": int(payload["seed"]),
                    "label_source": str(payload["label_source"]),
                }
                for path, payload in zip(
                    paths,
                    datasets,
                    strict=True,
                )
            ],
            "train_frames": int(train_observations.shape[0]),
            "validation_frames": int(validation_observations.shape[0]),
            "train_example_ids": [
                example_id_by_payload[id(payload)]
                for payload in train_payloads
            ],
            "validation_example_ids": [
                example_id_by_payload[id(payload)]
                for payload in validation_payloads
            ],
            "train_pair_ids": [
                payload["pair_id"] for payload in train_payloads
            ],
            "validation_pair_ids": [
                payload["pair_id"] for payload in validation_payloads
            ],
            "validation_seeds": sorted(validation_seeds),
            "best_validation_smooth_l1": best_validation_loss,
            "best_validation_hybrid_loss": final_validation_loss,
            "checkpoint_selection": (
                "rescue_weighted_hybrid_loss_with_"
                "nonincreasing_discrete_errors"
            ),
            "validation_gripper_error_count": (
                final_validation_gripper_errors
            ),
            "validation_saturation_error_count": (
                final_validation_saturation_errors
            ),
            "validation_action_mae": final_validation_mae,
            "validation_action_max_abs_error": final_max_abs_error,
            "validation_phase_mae": final_phase_mae,
            "baseline_distillation_pairs": baseline_pair_count,
            "dagger_pairs": dagger_pair_count,
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
                "schema_version": payload["schema_version"],
                "teacher_training_start_frame": (
                    _teacher_training_start_frame(
                        payload,
                        int(payload["episode"]["phases"].shape[0]),
                    )
                ),
                "collection_dranmar_revision": payload[
                    "source"
                ].get(
                    "collection_dranmar_revision",
                    payload["source"]["dranmar_revision"],
                ),
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
        "dagger_pairs": dagger_pair_count,
        "teacher_rescue_pairs": teacher_pair_count,
        "capability_scope": capability_scope,
        "train_frames": int(train_observations.shape[0]),
        "validation_frames": int(validation_observations.shape[0]),
        "best_epoch": best_epoch,
        "validation_gripper_error_count": (
            final_validation_gripper_errors
        ),
        "validation_saturation_error_count": (
            final_validation_saturation_errors
        ),
        "validation_action_mae": final_validation_mae,
        "validation_action_max_abs_error": final_max_abs_error,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the compact learned handover successor"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser(
        "bootstrap-seed",
        help="re-freeze the preserved recurrent seed for failure mining",
    )
    bootstrap.add_argument("--checkpoint", required=True)
    bootstrap.add_argument("--expected_sha256", required=True)
    bootstrap.add_argument("--output", required=True)

    rebind = subparsers.add_parser(
        "rebind-candidate",
        help="bind unchanged candidate weights to a compatible source",
    )
    rebind.add_argument("--checkpoint", required=True)
    rebind.add_argument("--expected_sha256", required=True)
    rebind.add_argument("--output", required=True)

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

    dagger = subparsers.add_parser(
        "admit-dagger",
        help="admit two exact safe on-policy frozen-oracle replays",
    )
    dagger.add_argument("--trace_a", required=True)
    dagger.add_argument("--trace_b", required=True)
    dagger.add_argument("--output", required=True)

    migrate_scaffold = subparsers.add_parser(
        "migrate-scaffold",
        help=(
            "bind a preserved baseline or DAgger scaffold to an "
            "unchanged later task contract"
        ),
    )
    migrate_scaffold.add_argument("--dataset", required=True)
    migrate_scaffold.add_argument("--target_revision", required=True)
    migrate_scaffold.add_argument("--output", required=True)

    train = subparsers.add_parser(
        "train",
        help="train one full-action policy from accepted demonstrations",
    )
    train.add_argument("--dataset", action="append", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--epochs", type=int, default=300)
    train.add_argument("--episode_batch_size", type=int, default=4)
    train.add_argument("--learning_rate", type=float, default=3.0e-4)
    train.add_argument("--weight_decay", type=float, default=1.0e-5)
    train.add_argument("--validation_fraction", type=float, default=0.2)
    train.add_argument("--hidden_dims", default="256,256")
    train.add_argument("--memory_dim", type=int, default=128)
    train.add_argument("--head_dim", type=int, default=128)
    train.add_argument("--patience", type=int, default=30)
    train.add_argument("--seed", type=int, default=104729)
    train.add_argument(
        "--initial_checkpoint",
        help="preserved recurrent candidate used only as initialization",
    )
    train.add_argument(
        "--completion_gate",
        action="store_true",
        help="require the full independent-rescue dataset contract",
    )
    train.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def main(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    if args.command == "bootstrap-seed":
        result = bootstrap_failure_mining_seed(args)
    elif args.command == "rebind-candidate":
        result = rebind_successor_candidate(args)
    elif args.command == "receipt":
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
    elif args.command == "admit-dagger":
        output = Path(args.output).expanduser().resolve()
        payload = admit_dagger_pair(
            Path(args.trace_a).expanduser().resolve(),
            Path(args.trace_b).expanduser().resolve(),
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
            "oracle_beta": payload["collection"]["oracle_beta"],
            "frames": payload["episode"]["frame_count"],
        }
    elif args.command == "migrate-scaffold":
        result = migrate_scaffold_dataset(args)
    else:
        result = train_successor(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
