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

UNIFORM_SCRIPT = (
    ROOT / "scripts/evaluate_dranmar_uniform_receiver_trajectory.py"
)
UNIFORM_SPEC = importlib.util.spec_from_file_location(
    "evaluate_dranmar_uniform_receiver_trajectory",
    UNIFORM_SCRIPT,
)
assert UNIFORM_SPEC is not None and UNIFORM_SPEC.loader is not None
UNIFORM_MODULE = importlib.util.module_from_spec(UNIFORM_SPEC)
sys.modules[UNIFORM_SPEC.name] = UNIFORM_MODULE
UNIFORM_SPEC.loader.exec_module(UNIFORM_MODULE)


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
                "grouped_initial_condition_allocation": True,
                "candidate_scales": [1.0] * replicas,
                "start_distance_m": 0.004,
                "end_distance_m": 0.001,
                "barrier_release_frame": 1200,
                "barrier_contract": (
                    "hold_ready_receiver_translation_then_release_"
                    "all_qualified_environments_simultaneously"
                ),
                "profile": "minimum_jerk_translation_scale",
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
            "barrier_hold_frames": torch.full(
                (count,),
                100,
                dtype=torch.long,
            ),
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


def _uniform_dataset(path: Path, *, scale: float) -> None:
    _noop_dataset(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    count = payload["num_envs"]
    payload["dataset_id"] = f"test-uniform-{scale}"
    payload["replay_contract"]["candidate_lanes"] = 1
    payload["replay_contract"][
        "grouped_initial_condition_allocation"
    ] = False
    payload["replay_contract"]["candidate_scales"] = [scale]
    payload["group_index"] = torch.arange(count)
    payload["candidate_index"] = torch.zeros(count, dtype=torch.long)
    payload["assigned_scale"] = torch.full((count,), scale)
    if scale < 1.0:
        base_action = payload["base_action_at_activation"]
        scaled_action = base_action.clone()
        distance = payload["distance_at_activation_m"]
        progress = ((0.004 - distance) / 0.003).clamp(0.0, 1.0)
        minimum_jerk = progress.square() * (3.0 - 2.0 * progress)
        multiplier = 1.0 - (1.0 - scale) * minimum_jerk
        context = payload["activation_context"]
        receiver_is_robot_1 = ~(context[:, 82] > 0.5)
        scaled_action[receiver_is_robot_1, :3] *= multiplier[
            receiver_is_robot_1
        ].unsqueeze(-1)
        scaled_action[~receiver_is_robot_1, 7:10] *= multiplier[
            ~receiver_is_robot_1
        ].unsqueeze(-1)
        payload["scaled_action_at_activation"] = scaled_action.clamp(
            -1.0,
            1.0,
        )
        payload["modified_frames"] = payload["active_frames"].clone()
        payload["minimum_multiplier"] = torch.full((count,), scale)
    torch.save(payload, path)


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


def test_uniform_triplet_requires_replication_before_learning(
    tmp_path: Path,
) -> None:
    control_path = tmp_path / "control.pt"
    noop_path = tmp_path / "noop.pt"
    candidate_path = tmp_path / "candidate.pt"
    _uniform_dataset(control_path, scale=1.0)
    _uniform_dataset(noop_path, scale=1.0)
    _uniform_dataset(candidate_path, scale=0.6)
    payload = torch.load(
        candidate_path,
        map_location="cpu",
        weights_only=False,
    )
    control_success = payload["full_success"].clone()
    candidate_success = control_success.clone()
    losses = torch.nonzero(control_success).flatten()[:2]
    wins = torch.nonzero(~control_success).flatten()[:10]
    candidate_success[losses] = False
    candidate_success[wins] = True
    payload["full_success"] = candidate_success
    payload["maximum_phase"] = torch.where(
        candidate_success,
        torch.full_like(payload["maximum_phase"], 4),
        torch.full_like(payload["maximum_phase"], 3),
    )
    payload["termination_flags"] = torch.stack(
        (candidate_success, ~candidate_success),
        dim=-1,
    )
    torch.save(payload, candidate_path)

    result = UNIFORM_MODULE._evaluate_seed(
        MODULE._load_dataset(control_path),
        MODULE._load_dataset(noop_path),
        MODULE._load_dataset(candidate_path),
        context_atol=0.0,
        action_atol=0.0,
    )
    aggregate, gate = UNIFORM_MODULE._gate(
        [result],
        minimum_seeds=3,
        significance_threshold=0.05,
    )

    assert result["control_noop_prebranch"]["passed"]
    assert result["control_candidate_prebranch"]["passed"]
    assert result["noop_terminal_outcome_exact"]
    assert result["inactive_candidate_outcome_exact"]
    assert result["intervention_integrity"]["passed"]
    assert (
        result["intervention_integrity"][
            "candidate_expected_action_atol"
        ]
        == UNIFORM_MODULE.FLOAT32_ACTION_RECONSTRUCTION_ATOL
    )
    assert aggregate["paired_net_successes"] == 8
    assert (
        aggregate["one_sided_exact_sign_probability"] < 0.05
    )
    assert gate["decision"] == "replication_required"
    assert gate["passed"] is False
    assert gate["postbranch_isolation_passed"]


def test_uniform_triplet_rejects_nonpositive_first_seed(
    tmp_path: Path,
) -> None:
    control_path = tmp_path / "control.pt"
    noop_path = tmp_path / "noop.pt"
    candidate_path = tmp_path / "candidate.pt"
    _uniform_dataset(control_path, scale=1.0)
    _uniform_dataset(noop_path, scale=1.0)
    _uniform_dataset(candidate_path, scale=0.6)

    result = UNIFORM_MODULE._evaluate_seed(
        MODULE._load_dataset(control_path),
        MODULE._load_dataset(noop_path),
        MODULE._load_dataset(candidate_path),
        context_atol=0.0,
        action_atol=0.0,
    )
    _, gate = UNIFORM_MODULE._gate(
        [result],
        minimum_seeds=3,
        significance_threshold=0.05,
    )

    assert result["paired_net_successes"] == 0
    assert gate["decision"] == "uniform_intervention_rejected"


def test_uniform_gate_blocks_postbranch_cross_environment_drift() -> None:
    result = {
        "control_noop_prebranch": {"passed": True},
        "control_candidate_prebranch": {"passed": True},
        "noop_terminal_outcome_exact": True,
        "inactive_candidate_outcome_exact": False,
        "intervention_integrity": {"passed": True},
        "wins": 10,
        "losses": 2,
        "samples": 32,
        "paired_net_successes": 8,
        "safety_delta": 0,
    }

    _, gate = UNIFORM_MODULE._gate(
        [result],
        minimum_seeds=1,
        significance_threshold=0.05,
    )

    assert gate["decision"] == "postbranch_isolation_invalid"
    assert gate["passed"] is False
    assert gate["postbranch_isolation_passed"] is False


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
        "dranmar-receiver-approach-trajectory-replay-1.3"
        in benchmark
    )
    assert (
        '"same_environment_index_common_random_numbers"'
        in benchmark
    )
    assert UNIFORM_SCRIPT.is_file()
