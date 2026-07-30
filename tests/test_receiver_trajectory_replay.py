from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_dranmar_receiver_trajectory_replay.py"
SPEC = importlib.util.spec_from_file_location(
    "evaluate_dranmar_receiver_trajectory_replay",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _noop_dataset(path: Path) -> None:
    groups = 8
    replicas = 4
    count = groups * replicas
    group_index = torch.arange(count) // replicas
    context = (
        torch.arange(count * 98, dtype=torch.float32)
        .reshape(count, 98)
        * 1.0e-5
    )
    action = torch.sin(
        torch.arange(count * 14, dtype=torch.float32).reshape(
            count,
            14,
        )
    )
    correction = (
        torch.arange(count * 6, dtype=torch.float32).reshape(count, 6)
        * 1.0e-6
    )
    success = torch.arange(count) % 3 == 0
    phase = torch.where(
        success,
        torch.full((count,), 4),
        torch.full((count,), 3),
    )
    terminal = torch.stack(
        (success, ~success),
        dim=-1,
    )
    torch.save(
        {
            "schema_version": MODULE.SCHEMA_VERSION,
            "dataset_id": "test-noop",
            "task": "DrAnmar-Handover-Test-v0",
            "seed": 17,
            "seed_stream_offset": 0,
            "runtime_seed": 17,
            "num_envs": count,
            "num_frames": 2000,
            "source_revision": "a" * 40,
            "base_checkpoint_sha256": "b" * 64,
            "receiver_candidate_checkpoint_sha256": "c" * 64,
            "preprobe_risk_checkpoint_sha256": "d" * 64,
            "replay_contract": {
                "method": (
                    "same_environment_index_common_random_numbers"
                ),
                "pairing_key": [
                    "source_revision",
                    "task",
                    "runtime_seed",
                    "num_envs",
                    "environment_index",
                ],
                "candidate_lanes": replicas,
                "candidate_scales": [1.0] * replicas,
                "modified_action_channels": (
                    "receiver_translation_xyz_only"
                ),
                "risk_role": (
                    "postbranch_retrospective_stratification_only"
                ),
            },
            "environment_index": torch.arange(count),
            "group_index": group_index,
            "candidate_index": torch.arange(count) % replicas,
            "assigned_scale": torch.ones(count),
            "first_episode_resolved": torch.ones(
                count,
                dtype=torch.bool,
            ),
            "activation_seen": torch.ones(count, dtype=torch.bool),
            "activation_frame": torch.full((count,), 100),
            "activation_context": context,
            "base_action_at_activation": action,
            "scaled_action_at_activation": action,
            "distance_at_activation_m": torch.full((count,), 0.0039),
            "active_frames": torch.full((count,), 3),
            "modified_frames": torch.zeros(count, dtype=torch.long),
            "minimum_multiplier": torch.ones(count),
            "receiver_candidate_correction": correction,
            "full_success": success,
            "maximum_phase": phase,
            "termination_names": ["success", "time_out"],
            "termination_flags": terminal,
            "receiver_safety_failure": torch.zeros(
                count,
                dtype=torch.bool,
            ),
            "postbranch_preprobe_risk_observed": torch.ones(
                count,
                dtype=torch.bool,
            ),
            "postbranch_predicted_preprobe_risk": torch.linspace(
                0.0,
                1.0,
                count,
            ),
        },
        path,
    )


def test_noop_pair_proves_same_environment_replay_parity(
    tmp_path: Path,
) -> None:
    control_path = tmp_path / "control.pt"
    candidate_path = tmp_path / "candidate.pt"
    _noop_dataset(control_path)
    _noop_dataset(candidate_path)
    control = MODULE._load_dataset(control_path)
    candidate = MODULE._load_dataset(candidate_path)

    summary = MODULE._evaluate_pair(
        control,
        candidate,
        context_atol=0.0,
        action_atol=0.0,
    )

    assert summary["prebranch_parity"]["passed"]
    assert summary["noop_lane_outcome_parity"] is True
    assert summary["activated_environments"] == 32


def test_cross_run_context_drift_invalidates_replay(
    tmp_path: Path,
) -> None:
    control_path = tmp_path / "control.pt"
    candidate_path = tmp_path / "drift.pt"
    _noop_dataset(control_path)
    _noop_dataset(candidate_path)
    payload = torch.load(
        candidate_path,
        map_location="cpu",
        weights_only=False,
    )
    payload["activation_context"][1, 0] += 1.0e-4
    torch.save(payload, candidate_path)
    control = MODULE._load_dataset(control_path)
    candidate = MODULE._load_dataset(candidate_path)

    summary = MODULE._evaluate_pair(
        control,
        candidate,
        context_atol=1.0e-6,
        action_atol=1.0e-7,
    )

    assert not summary["prebranch_parity"]["passed"]
    assert (
        summary["prebranch_parity"]["maximum_context_delta"]
        > 1.0e-6
    )


def test_exact_paired_gate_is_one_sided() -> None:
    assert MODULE._exact_one_sided_sign_probability(10, 2) < 0.05
    assert MODULE._exact_one_sided_sign_probability(6, 6) == 1.0
    assert MODULE._exact_one_sided_sign_probability(2, 10) == 1.0


def test_launcher_exposes_source_locked_trajectory_replay() -> None:
    launcher = (ROOT / "dr_anmar_learning.sh").read_text()
    benchmark = (
        ROOT / "scripts/dr_anmar_learning_benchmark.py"
    ).read_text()
    policy = (
        ROOT
        / "source/extensions/orbit.surgical.tasks/orbit/surgical/tasks"
        / "surgical/handover/recovery_policy.py"
    ).read_text()

    assert "receiver-trajectory-handover)" in launcher
    assert "DR_ANMAR_PROMOTED_ALLOW_APPROACH_TRAJECTORY=1" in launcher
    assert "HandoverReceiverApproachTrajectoryPolicy" in policy
    assert "_RECEIVER_OWNER_CORRECTED_APPROACH" in policy
    assert "minimum_jerk" in policy
    assert (
        "dranmar-receiver-approach-trajectory-replay-1.1"
        in benchmark
    )
    assert (
        '"same_environment_index_common_random_numbers"'
        in benchmark
    )
