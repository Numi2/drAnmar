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
    context_by_group = torch.randn(groups, 98)
    action_by_group = torch.randn(groups, 14).clamp(-1.0, 1.0)
    correction_by_group = torch.randn(groups, 6) * 0.001
    success_by_group = torch.arange(groups) % 2 == 0
    phase_by_group = torch.where(
        success_by_group,
        torch.full((groups,), 4),
        torch.full((groups,), 3),
    )
    terminal_by_group = torch.stack(
        (success_by_group, ~success_by_group),
        dim=-1,
    )
    torch.save(
        {
            "schema_version": MODULE.SCHEMA_VERSION,
            "dataset_id": "test-noop",
            "task": "DrAnmar-Handover-Test-v0",
            "seed": 17,
            "seed_stream_offset": 0,
            "num_envs": count,
            "num_frames": 2000,
            "source_revision": "a" * 40,
            "base_checkpoint_sha256": "b" * 64,
            "receiver_candidate_checkpoint_sha256": "c" * 64,
            "preprobe_risk_checkpoint_sha256": "d" * 64,
            "replay_contract": {
                "method": "simultaneous_grouped_clones",
                "group_replicas": replicas,
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
            "activation_context": context_by_group.repeat_interleave(
                replicas,
                dim=0,
            ),
            "base_action_at_activation": (
                action_by_group.repeat_interleave(replicas, dim=0)
            ),
            "scaled_action_at_activation": (
                action_by_group.repeat_interleave(replicas, dim=0)
            ),
            "distance_at_activation_m": torch.full((count,), 0.0039),
            "active_frames": torch.full((count,), 3),
            "modified_frames": torch.zeros(count, dtype=torch.long),
            "minimum_multiplier": torch.ones(count),
            "receiver_candidate_correction": (
                correction_by_group.repeat_interleave(replicas, dim=0)
            ),
            "full_success": success_by_group.repeat_interleave(replicas),
            "maximum_phase": phase_by_group.repeat_interleave(replicas),
            "termination_names": ["success", "time_out"],
            "termination_flags": terminal_by_group.repeat_interleave(
                replicas,
                dim=0,
            ),
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


def test_noop_dataset_proves_grouped_replay_parity(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "noop.pt"
    _noop_dataset(dataset)

    _, summary = MODULE._evaluate_dataset(
        dataset,
        context_atol=1.0e-6,
        action_atol=1.0e-7,
    )

    assert summary["prebranch_parity"]["passed"]
    assert summary["noop_outcome_parity"] is True
    assert summary["active_groups"] == 8


def test_context_drift_invalidates_replay(tmp_path: Path) -> None:
    dataset = tmp_path / "drift.pt"
    _noop_dataset(dataset)
    payload = torch.load(dataset, map_location="cpu", weights_only=False)
    payload["activation_context"][1, 0] += 1.0e-4
    torch.save(payload, dataset)

    _, summary = MODULE._evaluate_dataset(
        dataset,
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
        "dranmar-receiver-approach-trajectory-replay-1.0"
        in benchmark
    )
    assert '"method": "simultaneous_grouped_clones"' in benchmark
