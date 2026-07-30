from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


SUCCESSOR_TOOL = _load_module(
    "dr_anmar_handover_successor_tool",
    ROOT / "scripts/dr_anmar_handover_successor.py",
)
SUCCESSOR_POLICY = _load_module(
    "dr_anmar_handover_successor_policy_test",
    ROOT
    / "source/extensions/orbit.surgical.tasks/orbit/surgical/tasks/"
    "surgical/handover/successor_policy.py",
)
SUCCESSOR_PPO = _load_module(
    "dr_anmar_handover_successor_ppo_test",
    ROOT / "scripts/dr_anmar_handover_ppo.py",
)


def _trace(
    *,
    role: str,
    teacher_kind: str,
    outcome: str,
    pair_id: str = "pair-104729-0",
    seed: int = 104729,
    teacher_branch: int | None = None,
    unsafe: bool = False,
) -> dict:
    frames = 10
    observations = torch.zeros(frames, 98)
    phases = torch.arange(frames) % 5
    observations[torch.arange(frames), 77 + phases] = 1.0
    observations[:, 82] = 1.0
    actions = torch.zeros(frames, 14)
    if teacher_branch is not None:
        actions[teacher_branch:, 7] = 0.2
    safety = torch.zeros(
        frames,
        len(SUCCESSOR_TOOL.SAFETY_TERMS),
        dtype=torch.bool,
    )
    if unsafe:
        safety[-1, 0] = True
    return {
        "schema_version": SUCCESSOR_TOOL.TRACE_SCHEMA,
        "role": role,
        "teacher_kind": teacher_kind,
        "teacher_receipt": (
            {
                "path": "/immutable/receipt.json",
                "sha256": "a" * 64,
                "schema_version": "dranmar-handover-teacher-receipt-1.0",
                "selected_parameters": (
                    {
                        "receiver_recovery_fixed_correction": [
                            0.001,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ]
                    }
                    if teacher_kind == "constrained_trajectory_optimizer"
                    else None
                ),
            }
            if role == "teacher"
            else None
        ),
        "pair_id": pair_id,
        "task": "DrAnmar-Handover-Needle-Dual-PSM-IK-Rel-v0",
        "seed": seed,
        "num_envs": 1,
        "reset_rotation_randomization_deg": 0.0,
        "observations": observations,
        "actions": actions,
        "rewards": torch.zeros(frames),
        "phases": phases,
        "safety_term_names": list(SUCCESSOR_TOOL.SAFETY_TERMS),
        "safety_events": safety,
        "terminal": {
            "complete": True,
            "outcome": outcome,
            "frame_count": frames,
        },
        "policy": {
            "base_checkpoint_sha256": "b" * 64,
            "successor_checkpoint_sha256": None,
            "receiver_recovery_fixed_correction": (
                "0.001,0,0,0,0,0"
                if teacher_kind == "constrained_trajectory_optimizer"
                else None
            ),
        },
        "runtime": {
            "source": {
                "dranmar_revision": "c" * 40,
                "asset_revision": "d" * 40,
                "asset_root": "/immutable/orbit.surgical.assets",
            }
        },
    }


def _dagger_trace(
    *,
    pair_id: str = "dagger-104729-round-1",
    seed: int = 104729,
    outcome: str = "success",
    unsafe: bool = False,
) -> dict:
    trace = _trace(
        role="control",
        teacher_kind="frozen_baseline",
        outcome=outcome,
        pair_id=pair_id,
        seed=seed,
        unsafe=unsafe,
    )
    oracle_beta = 0.9
    oracle_actions = trace.pop("actions")
    student_actions = torch.full_like(oracle_actions, 0.2)
    trace["schema_version"] = SUCCESSOR_TOOL.DAGGER_TRACE_SCHEMA
    trace["oracle_beta"] = oracle_beta
    trace["student_actions"] = student_actions
    trace["oracle_actions"] = oracle_actions
    trace["executed_actions"] = (
        oracle_beta * oracle_actions
        + (1.0 - oracle_beta) * student_actions
    )
    trace["policy"] = {
        "base_checkpoint_sha256": "b" * 64,
        "successor_checkpoint_sha256": "e" * 64,
        "oracle_kind": "frozen_promoted_composite",
        "oracle_configuration": {
            "base_checkpoint_sha256": "b" * 64,
            "residual_scale": 0.03,
        },
        "mixture": (
            "oracle_beta_times_oracle_plus_"
            "one_minus_beta_times_student"
        ),
    }
    return trace


def _save(path: Path, payload: dict) -> Path:
    torch.save(payload, path)
    return path


def test_optimizer_receipt_locks_bounded_proposal(tmp_path: Path) -> None:
    proposal = tmp_path / "proposal.pt"
    proposal.write_bytes(b"proposal-only")
    output = tmp_path / "receipt.json"
    result = SUCCESSOR_TOOL.create_optimizer_receipt(
        argparse.Namespace(
            pair_id="pair-104729-0",
            task="DrAnmar-Handover-Needle-Dual-PSM-IK-Rel-v0",
            seed=104729,
            receiver_correction="0.001,0,0,0,0,0",
            position_cap_m=0.0025,
            orientation_cap_deg=2.0,
            proposal_source=[str(proposal)],
            output=str(output),
        )
    )
    assert output.is_file()
    assert result["receiver_recovery_fixed_correction"][0] == 0.001
    with pytest.raises(ValueError, match="overwrite"):
        SUCCESSOR_TOOL.create_optimizer_receipt(
            argparse.Namespace(
                pair_id="pair-104729-0",
                task="DrAnmar-Handover-Needle-Dual-PSM-IK-Rel-v0",
                seed=104729,
                receiver_correction="0.001,0,0,0,0,0",
                position_cap_m=0.0025,
                orientation_cap_deg=2.0,
                proposal_source=[str(proposal)],
                output=str(output),
            )
        )


def test_retention_schedule_extends_last_causal_centering_action(
    tmp_path: Path,
) -> None:
    control = _trace(
        role="control",
        teacher_kind="frozen_baseline",
        outcome="receiver_retention_lost",
    )
    phases = torch.tensor([0, 0, 1, 1, 2, 2, 2, 2, 3, 3])
    control["phases"] = phases
    control["observations"][:, 77:82] = 0.0
    control["observations"][
        torch.arange(phases.numel()),
        77 + phases,
    ] = 1.0
    control["observations"][:, 68:70] = torch.tensor([0.02, 0.002])
    control["actions"][7, 9] = -0.0025
    control_path = _save(tmp_path / "control.pt", control)
    schedule_path = tmp_path / "schedule.json"
    receipt_path = tmp_path / "schedule-receipt.json"

    result = SUCCESSOR_TOOL.propose_retention_schedule(
        argparse.Namespace(
            control=str(control_path),
            duration=12,
            lookback=16,
            output_schedule=str(schedule_path),
            output_receipt=str(receipt_path),
        )
    )

    schedule = json.loads(schedule_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    segment = schedule["segments"][0]
    assert result["branch_frame"] == 8
    assert result["source_frame"] == 7
    assert segment["action_indices"] == [9]
    assert segment["values"] == pytest.approx([-0.0025])
    assert segment["stop_frame_exclusive"] == 20
    assert (
        receipt["optimizer"]["selected_parameters"][
            "action_schedule_sha256"
        ]
        == result["schedule_sha256"]
    )


def test_exact_safe_baseline_success_is_admitted_for_distillation(
    tmp_path: Path,
) -> None:
    control = _trace(
        role="control",
        teacher_kind="frozen_baseline",
        outcome="success",
    )
    phases = torch.tensor([0, 0, 1, 1, 2, 2, 2, 3, 3, 3])
    control["phases"] = phases
    control["observations"][:, 77:82] = 0.0
    control["observations"][
        torch.arange(phases.numel()),
        77 + phases,
    ] = 1.0
    accepted = SUCCESSOR_TOOL.admit_baseline_pair(
        _save(tmp_path / "control-a.pt", control),
        _save(tmp_path / "control-b.pt", control),
    )

    assert accepted["accepted"] is True
    assert all(accepted["gates"].values())
    assert (
        accepted["label_source"]
        == SUCCESSOR_TOOL.BASELINE_LABEL_SOURCE
    )
    assert accepted["teacher_receipt"] is None
    assert torch.equal(
        accepted["episode"]["actions"],
        control["actions"],
    )

    failed = _trace(
        role="control",
        teacher_kind="frozen_baseline",
        outcome="receiver_retention_lost",
    )
    with pytest.raises(ValueError, match="successful episodes only"):
        SUCCESSOR_TOOL.admit_baseline_pair(
            _save(tmp_path / "failed-a.pt", failed),
            _save(tmp_path / "failed-b.pt", failed),
        )


def test_exact_safe_dagger_replay_is_admitted_with_oracle_labels(
    tmp_path: Path,
) -> None:
    trace = _dagger_trace()
    accepted = SUCCESSOR_TOOL.admit_dagger_pair(
        _save(tmp_path / "dagger-a.pt", trace),
        _save(tmp_path / "dagger-b.pt", trace),
    )

    assert accepted["accepted"] is True
    assert all(accepted["gates"].values())
    assert accepted["label_source"] == SUCCESSOR_TOOL.DAGGER_LABEL_SOURCE
    assert accepted["collection"]["oracle_beta"] == pytest.approx(0.9)
    assert torch.equal(
        accepted["episode"]["actions"],
        trace["oracle_actions"],
    )
    assert torch.equal(
        accepted["episode"]["executed_actions"],
        trace["executed_actions"],
    )


def test_scaffold_migration_preserves_original_and_unchanged_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _trace(
        role="control",
        teacher_kind="frozen_baseline",
        outcome="success",
    )
    accepted = SUCCESSOR_TOOL.admit_baseline_pair(
        _save(tmp_path / "control-a.pt", control),
        _save(tmp_path / "control-b.pt", control),
    )
    original = _save(tmp_path / "accepted-v1.pt", accepted)
    target_revision = "e" * 40
    manifest = {
        "revision": "resolved",
        "sha256": "f" * 64,
        "entries": [{"path": "contract.py", "git_blob": "a" * 40}],
    }
    monkeypatch.setattr(
        SUCCESSOR_TOOL,
        "_resolve_revision",
        lambda _repo, revision: revision,
    )
    monkeypatch.setattr(
        SUCCESSOR_TOOL,
        "_handover_contract_manifest",
        lambda _revision: manifest,
    )
    migrated_path = tmp_path / "accepted-v2.pt"

    result = SUCCESSOR_TOOL.migrate_scaffold_dataset(
        argparse.Namespace(
            dataset=str(original),
            target_revision=target_revision,
            output=str(migrated_path),
        )
    )
    migrated = SUCCESSOR_TOOL._load_accepted_dataset(migrated_path)

    assert result["schema_version"] == SUCCESSOR_TOOL.DATASET_SCHEMA_V2
    assert migrated["source"]["dranmar_revision"] == target_revision
    assert migrated["source"]["collection_dranmar_revision"] == "c" * 40
    assert torch.equal(
        migrated["episode"]["actions"],
        accepted["episode"]["actions"],
    )

    accepted["episode"]["actions"][0, 0] = 0.5
    torch.save(accepted, original)
    with pytest.raises(ValueError, match="original artifact mismatch"):
        SUCCESSOR_TOOL._load_accepted_dataset(migrated_path)


def test_scaffold_migration_rejects_environment_contract_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _trace(
        role="control",
        teacher_kind="frozen_baseline",
        outcome="success",
    )
    accepted = SUCCESSOR_TOOL.admit_baseline_pair(
        _save(tmp_path / "control-a.pt", control),
        _save(tmp_path / "control-b.pt", control),
    )
    original = _save(tmp_path / "accepted-v1.pt", accepted)
    monkeypatch.setattr(
        SUCCESSOR_TOOL,
        "_resolve_revision",
        lambda _repo, revision: revision,
    )
    monkeypatch.setattr(
        SUCCESSOR_TOOL,
        "_handover_contract_manifest",
        lambda revision: {
            "revision": revision,
            "sha256": hashlib.sha256(revision.encode()).hexdigest(),
            "entries": [],
        },
    )

    with pytest.raises(ValueError, match="recapture is required"):
        SUCCESSOR_TOOL.migrate_scaffold_dataset(
            argparse.Namespace(
                dataset=str(original),
                target_revision="e" * 40,
                output=str(tmp_path / "accepted-v2.pt"),
            )
        )


def test_dagger_admission_rejects_nonexact_or_unbound_mixtures(
    tmp_path: Path,
) -> None:
    trace_a = _dagger_trace()
    trace_b = _dagger_trace()
    trace_b["observations"][2, 0] = 1.0e-6
    with pytest.raises(ValueError, match="replay is not exact"):
        SUCCESSOR_TOOL.admit_dagger_pair(
            _save(tmp_path / "dagger-a.pt", trace_a),
            _save(tmp_path / "dagger-b.pt", trace_b),
        )

    broken = _dagger_trace(pair_id="dagger-broken")
    broken["executed_actions"][0, 0] = 0.5
    with pytest.raises(ValueError, match="recorded mixture"):
        SUCCESSOR_TOOL.admit_dagger_pair(
            _save(tmp_path / "broken-a.pt", broken),
            _save(tmp_path / "broken-b.pt", broken),
        )


def test_successor_uses_hard_saturation_modes() -> None:
    model = SUCCESSOR_POLICY.PhaseConditionedHandoverPolicy(
        torch.zeros(98),
        torch.ones(98),
        hidden_dims=(16,),
        head_dim=8,
    )
    output_layer = model.phase_heads[0][-1]
    with torch.no_grad():
        saturation_bias = output_layer.bias[
            SUCCESSOR_POLICY.HANDOVER_ACTION_DIM:
        ].view(
            len(SUCCESSOR_POLICY.HANDOVER_CONTINUOUS_INDICES),
            SUCCESSOR_POLICY.HANDOVER_SATURATION_CLASS_COUNT,
        )
        saturation_bias[0] = torch.tensor([-5.0, -5.0, 5.0])
        saturation_bias[1] = torch.tensor([5.0, -5.0, -5.0])
    observation = torch.zeros(1, 98)
    observation[:, 77] = 1.0
    action = model(observation)
    assert action[0, 0].item() == 1.0
    assert action[0, 1].item() == -1.0


def test_successor_runtime_memory_matches_offline_sequence() -> None:
    torch.manual_seed(7)
    model = SUCCESSOR_POLICY.PhaseConditionedHandoverPolicy(
        torch.zeros(98),
        torch.ones(98),
        hidden_dims=(16,),
        memory_dim=8,
        head_dim=8,
    )
    sequence = torch.randn(1, 6, 98)
    sequence[:, :, 77:82] = 0.0
    sequence[:, :, 77] = 1.0
    with torch.inference_mode():
        offline = model.training_sequence_actions(sequence)
        model.reset()
        runtime = torch.stack(
            [model(sequence[:, frame]) for frame in range(6)],
            dim=1,
        )
        model.reset(torch.ones(1, dtype=torch.bool))
        replay = torch.stack(
            [model(sequence[:, frame]) for frame in range(6)],
            dim=1,
        )
    assert torch.allclose(runtime, offline)
    assert torch.allclose(replay, offline)


def test_successor_explicit_step_jit_matches_eager(tmp_path: Path) -> None:
    torch.manual_seed(11)
    model = SUCCESSOR_POLICY.PhaseConditionedHandoverPolicy(
        torch.zeros(98),
        torch.ones(98),
        hidden_dims=(16,),
        memory_dim=8,
        head_dim=8,
    )
    checkpoint = tmp_path / "successor.pt"
    torch.save(
        {
            "schema_version": (
                SUCCESSOR_POLICY.SUCCESSOR_CHECKPOINT_SCHEMA
            ),
            "deployment_status": "candidate_only",
            "training_gate_passed": True,
            "observation_dim": 98,
            "action_dim": 14,
            "phase_slice": [77, 82],
            "observation_mean": torch.zeros(98),
            "observation_std": torch.ones(98),
            "architecture": {
                "hidden_dims": [16],
                "memory_dim": 8,
                "head_dim": 8,
                "binary_gripper_indices": [6, 13],
                "continuous_action_indices": list(
                    SUCCESSOR_POLICY.HANDOVER_CONTINUOUS_INDICES
                ),
                "saturation_classes": [
                    "negative_limit",
                    "precision",
                    "positive_limit",
                ],
                "saturation_logit_margin": 1.5,
                "recurrent_state": "gru_reset_per_episode",
            },
            "model": model.state_dict(),
        },
        checkpoint,
    )
    output = tmp_path / "successor.jit"
    SUCCESSOR_POLICY.export_handover_successor_checkpoint(
        str(checkpoint),
        str(output),
    )
    scripted = torch.jit.load(str(output))
    observation = torch.randn(3, 98)
    observation[:, 77:82] = 0.0
    observation[:, 77] = 1.0
    hidden = model.initial_hidden(3)
    with torch.inference_mode():
        eager_action, eager_hidden = model.step(observation, hidden)
        scripted_action, scripted_hidden = scripted(
            observation,
            hidden,
        )
    assert torch.equal(scripted_action, eager_action)
    assert torch.equal(scripted_hidden, eager_hidden)


def test_failure_mining_seed_bootstrap_is_versioned_and_candidate_only(
    tmp_path: Path,
) -> None:
    model = SUCCESSOR_POLICY.PhaseConditionedHandoverPolicy(
        torch.zeros(98),
        torch.ones(98),
        hidden_dims=(16,),
        memory_dim=8,
        head_dim=8,
    )
    source = tmp_path / "preserved-v2.pt"
    torch.save(
        {
            "schema_version": (
                SUCCESSOR_TOOL.LEGACY_SUCCESSOR_CHECKPOINT_SCHEMA_V2
            ),
            "deployment_status": "candidate_only",
            "training_gate_passed": True,
            "observation_dim": 98,
            "action_dim": 14,
            "phase_slice": [77, 82],
            "observation_mean": torch.zeros(98),
            "observation_std": torch.ones(98),
            "architecture": {
                "hidden_dims": [16],
                "memory_dim": 8,
                "head_dim": 8,
                "full_action_policy": True,
                "runtime_heuristic_stack": False,
                "recurrent_state": "gru_reset_per_episode",
            },
            "model": model.state_dict(),
            "source": {
                "dranmar_revision": "c" * 40,
                "asset_revision": "d" * 40,
            },
        },
        source,
    )
    output = tmp_path / "failure-mining-v3.pt"
    result = SUCCESSOR_TOOL.bootstrap_failure_mining_seed(
        argparse.Namespace(
            checkpoint=str(source),
            expected_sha256=SUCCESSOR_TOOL._sha256(source),
            output=str(output),
        )
    )

    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert result["promotion_eligible"] is False
    assert payload["schema_version"] == (
        SUCCESSOR_TOOL.SUCCESSOR_CHECKPOINT_SCHEMA_V3
    )
    assert payload["promotion_eligible"] is False
    assert payload["failure_mining_seed"]["sha256"] == (
        SUCCESSOR_TOOL._sha256(source)
    )
    assert payload["source"]["weight_training_dranmar_revision"] == (
        "c" * 40
    )


def test_candidate_rebind_preserves_weights_and_requires_same_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = SUCCESSOR_POLICY.PhaseConditionedHandoverPolicy(
        torch.zeros(98),
        torch.ones(98),
        hidden_dims=(16,),
        memory_dim=8,
        head_dim=8,
    )
    source = tmp_path / "candidate.pt"
    torch.save(
        {
            "schema_version": (
                SUCCESSOR_TOOL.SUCCESSOR_CHECKPOINT_SCHEMA_V3
            ),
            "deployment_status": "candidate_only",
            "training_gate_passed": True,
            "observation_dim": 98,
            "action_dim": 14,
            "phase_slice": [77, 82],
            "observation_mean": torch.zeros(98),
            "observation_std": torch.ones(98),
            "architecture": {
                "hidden_dims": [16],
                "memory_dim": 8,
                "head_dim": 8,
                "full_action_policy": True,
                "runtime_heuristic_stack": False,
                "recurrent_state": "gru_reset_per_episode",
            },
            "model": model.state_dict(),
            "source": {
                "dranmar_revision": "c" * 40,
                "asset_revision": "d" * 40,
            },
        },
        source,
    )
    monkeypatch.setattr(
        SUCCESSOR_TOOL,
        "_resolve_revision",
        lambda repo_root, revision: "e" * 40,
    )
    monkeypatch.setattr(
        SUCCESSOR_TOOL,
        "_handover_contract_manifest",
        lambda revision: {
            "revision": revision,
            "sha256": "f" * 64,
            "entries": [],
        },
    )
    monkeypatch.setattr(
        SUCCESSOR_TOOL,
        "_revision_blob",
        lambda revision, repository_path: "1" * 40,
    )
    output = tmp_path / "rebound.pt"

    result = SUCCESSOR_TOOL.rebind_successor_candidate(
        argparse.Namespace(
            checkpoint=str(source),
            expected_sha256=SUCCESSOR_TOOL._sha256(source),
            output=str(output),
        )
    )

    rebound = torch.load(output, map_location="cpu", weights_only=False)
    assert result["weights_unchanged"] is True
    assert rebound["source"]["dranmar_revision"] == "e" * 40
    assert rebound["source"]["weight_training_dranmar_revision"] == (
        "c" * 40
    )
    assert all(
        torch.equal(value, rebound["model"][name])
        for name, value in model.state_dict().items()
    )
    assert rebound["source_rebind"]["successor_policy_git_blob"] == (
        "1" * 40
    )

    monkeypatch.setattr(
        SUCCESSOR_TOOL,
        "_handover_contract_manifest",
        lambda revision: {
            "revision": revision,
            "sha256": (
                "f" * 64 if revision == "c" * 40 else "0" * 64
            ),
            "entries": [],
        },
    )
    with pytest.raises(ValueError, match="environment contract changed"):
        SUCCESSOR_TOOL.rebind_successor_candidate(
            argparse.Namespace(
                checkpoint=str(source),
                expected_sha256=SUCCESSOR_TOOL._sha256(source),
                output=str(tmp_path / "rejected.pt"),
            )
        )

    monkeypatch.setattr(
        SUCCESSOR_TOOL,
        "_handover_contract_manifest",
        lambda revision: {
            "revision": revision,
            "sha256": "f" * 64,
            "entries": [],
        },
    )
    monkeypatch.setattr(
        SUCCESSOR_TOOL,
        "_revision_blob",
        lambda revision, repository_path: (
            "1" * 40 if revision == "c" * 40 else "2" * 40
        ),
    )
    with pytest.raises(
        ValueError,
        match="policy implementation changed",
    ):
        SUCCESSOR_TOOL.rebind_successor_candidate(
            argparse.Namespace(
                checkpoint=str(source),
                expected_sha256=SUCCESSOR_TOOL._sha256(source),
                output=str(tmp_path / "policy-rejected.pt"),
            )
        )


def test_isolated_teacher_acceptance_requires_exact_replay_and_safe_win(
    tmp_path: Path,
) -> None:
    control = _trace(
        role="control",
        teacher_kind="frozen_baseline",
        outcome="time_out",
    )
    teacher = _trace(
        role="teacher",
        teacher_kind="constrained_trajectory_optimizer",
        outcome="success",
        teacher_branch=5,
    )
    accepted = SUCCESSOR_TOOL.accept_teacher_pair(
        _save(tmp_path / "control-a.pt", control),
        _save(tmp_path / "control-b.pt", control),
        _save(tmp_path / "teacher.pt", teacher),
    )

    assert accepted["accepted"] is True
    assert all(accepted["gates"].values())
    assert accepted["label_source"] == SUCCESSOR_TOOL.TEACHER_LABEL_SOURCE
    assert accepted["branch_frame"] == 5
    assert torch.equal(
        accepted["episode"]["actions"],
        teacher["actions"],
    )


def test_successor_teacher_rescue_binds_exact_candidate_hash(
    tmp_path: Path,
) -> None:
    control = _trace(
        role="control",
        teacher_kind="successor_candidate",
        outcome="protected_surface_force",
    )
    teacher = _trace(
        role="teacher",
        teacher_kind="constrained_trajectory_optimizer",
        outcome="success",
        teacher_branch=5,
    )
    candidate_sha256 = "f" * 64
    control["policy"]["successor_checkpoint_sha256"] = candidate_sha256
    teacher["policy"]["successor_checkpoint_sha256"] = candidate_sha256
    accepted = SUCCESSOR_TOOL.accept_teacher_pair(
        _save(tmp_path / "candidate-control-a.pt", control),
        _save(tmp_path / "candidate-control-b.pt", control),
        _save(tmp_path / "candidate-teacher.pt", teacher),
    )
    assert accepted["control_policy_kind"] == "successor_candidate"
    assert (
        accepted["successor_checkpoint_sha256"]
        == candidate_sha256
    )

    teacher["policy"]["successor_checkpoint_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="paired traces do not share"):
        SUCCESSOR_TOOL.accept_teacher_pair(
            tmp_path / "candidate-control-a.pt",
            tmp_path / "candidate-control-b.pt",
            _save(tmp_path / "wrong-candidate-teacher.pt", teacher),
        )


def test_episode_sampler_reserves_half_of_mixed_batches_for_rescues() -> None:
    labels = (
        [SUCCESSOR_TOOL.BASELINE_LABEL_SOURCE] * 4
        + [SUCCESSOR_TOOL.DAGGER_LABEL_SOURCE] * 4
        + [SUCCESSOR_TOOL.TEACHER_LABEL_SOURCE] * 4
    )
    order = SUCCESSOR_TOOL._source_balanced_epoch_indices(
        labels,
        generator=torch.Generator().manual_seed(7),
    )
    sampled = [labels[index] for index in order]
    assert sampled.count(SUCCESSOR_TOOL.TEACHER_LABEL_SOURCE) == 6
    assert sampled.count(SUCCESSOR_TOOL.BASELINE_LABEL_SOURCE) == 3
    assert sampled.count(SUCCESSOR_TOOL.DAGGER_LABEL_SOURCE) == 3


def test_ppo_manifest_retains_base_and_all_followup_rescues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets = []
    payload_by_path = {}
    labels = (
        [SUCCESSOR_TOOL.TEACHER_LABEL_SOURCE] * 92
        + [SUCCESSOR_TOOL.BASELINE_LABEL_SOURCE] * 8
        + [SUCCESSOR_TOOL.DAGGER_LABEL_SOURCE] * 8
    )
    for index, label in enumerate(labels):
        path = tmp_path / f"accepted-{index:03d}.pt"
        path.write_bytes(f"accepted-{index}".encode())
        sha256 = SUCCESSOR_PPO._sha256(path)
        datasets.append(
            {
                "path": str(path),
                "sha256": sha256,
                "label_source": label,
            }
        )
        payload_by_path[path.resolve()] = {
            "pair_id": f"pair-{index}",
            "label_source": label,
            "seed": 100_000 + index,
            "task": "DrAnmar-Handover-Needle-Dual-PSM-IK-Rel-v0",
        }

    base_manifest = tmp_path / "base-manifest.json"
    base_manifest.write_text(
        json.dumps(
            {
                "schema_version": (
                    "dranmar-handover-training-buffer-2.0"
                ),
                "preservation": {
                    "all_accepted_data_retained": True,
                },
                "datasets": datasets[:96],
            }
        )
    )
    successor_manifest = tmp_path / "successor-manifest.json"
    successor_manifest.write_text(
        json.dumps(
            {
                "schema_version": (
                    "dranmar-handover-successor-dataset-manifest-2.0"
                ),
                "base_manifest": {
                    "path": str(base_manifest),
                    "sha256": SUCCESSOR_PPO._sha256(base_manifest),
                    "dataset_count": 96,
                },
                "dataset_count": 108,
                "new_dataset_count": 12,
                "qualification_seed_exclusion_verified": True,
                "datasets": datasets,
            }
        )
    )
    monkeypatch.setattr(
        SUCCESSOR_TOOL,
        "_load_accepted_dataset",
        lambda path: payload_by_path[path.resolve()],
    )

    payloads, normalized = SUCCESSOR_PPO._load_demonstrations(
        successor_manifest,
        SUCCESSOR_TOOL,
    )

    assert len(payloads) == 108
    assert normalized["preservation"]["teacher_rescue_count"] == 92
    assert normalized["preservation"]["total_episode_count"] == 108

    dropped = json.loads(successor_manifest.read_text())
    dropped["datasets"] = dropped["datasets"][1:]
    dropped["dataset_count"] = 107
    dropped["new_dataset_count"] = 11
    dropped_manifest = tmp_path / "dropped-manifest.json"
    dropped_manifest.write_text(json.dumps(dropped))
    with pytest.raises(ValueError, match="dropped accepted base data"):
        SUCCESSOR_PPO._load_demonstrations(
            dropped_manifest,
            SUCCESSOR_TOOL,
        )


def test_optimizer_rescue_weighting_starts_at_observed_branch() -> None:
    payload = {
        "label_source": SUCCESSOR_TOOL.TEACHER_LABEL_SOURCE,
        "teacher_kind": "constrained_trajectory_optimizer",
        "branch_frame": 0,
        "teacher_receipt": {"optimization_start_frame": 45},
    }
    assert (
        SUCCESSOR_TOOL._teacher_training_start_frame(payload, 100)
        == 0
    )

    payload["teacher_receipt"]["optimization_start_frame"] = 100
    with pytest.raises(ValueError, match="outside"):
        SUCCESSOR_TOOL._teacher_training_start_frame(payload, 100)


def test_isolated_schedule_teacher_is_bound_to_receipt(
    tmp_path: Path,
) -> None:
    control = _trace(
        role="control",
        teacher_kind="frozen_baseline",
        outcome="receiver_retention_lost",
    )
    teacher = _trace(
        role="teacher",
        teacher_kind="constrained_trajectory_optimizer",
        outcome="success",
        teacher_branch=5,
    )
    schedule_hash = "e" * 64
    teacher["teacher_receipt"]["selected_parameters"] = {
        "action_schedule_sha256": schedule_hash,
    }
    teacher["teacher_receipt"]["stratum"] = "custody"
    teacher["policy"]["receiver_recovery_fixed_correction"] = None
    teacher["policy"]["teacher_action_schedule_sha256"] = schedule_hash
    teacher["teacher_action_schedule"] = {
        "sha256": schedule_hash,
        "branch_frame": 5,
        "segments": [{"start_frame_inclusive": 5}],
    }

    accepted = SUCCESSOR_TOOL.accept_teacher_pair(
        _save(tmp_path / "control-a.pt", control),
        _save(tmp_path / "control-b.pt", control),
        _save(tmp_path / "teacher.pt", teacher),
    )
    assert accepted["accepted"] is True

    teacher["teacher_action_schedule"]["branch_frame"] = 6
    with pytest.raises(ValueError, match="observed branch"):
        SUCCESSOR_TOOL.accept_teacher_pair(
            tmp_path / "control-a.pt",
            tmp_path / "control-b.pt",
            _save(tmp_path / "teacher-wrong-branch.pt", teacher),
        )


def test_teacher_acceptance_rejects_vectorized_or_nondeterministic_data(
    tmp_path: Path,
) -> None:
    control_a = _trace(
        role="control",
        teacher_kind="frozen_baseline",
        outcome="time_out",
    )
    control_b = _trace(
        role="control",
        teacher_kind="frozen_baseline",
        outcome="time_out",
    )
    teacher = _trace(
        role="teacher",
        teacher_kind="constrained_trajectory_optimizer",
        outcome="success",
        teacher_branch=5,
    )
    control_b["observations"][2, 0] = 1.0e-6
    with pytest.raises(ValueError, match="no-op replay is not exact"):
        SUCCESSOR_TOOL.accept_teacher_pair(
            _save(tmp_path / "control-a.pt", control_a),
            _save(tmp_path / "control-b.pt", control_b),
            _save(tmp_path / "teacher.pt", teacher),
        )

    vectorized = _trace(
        role="control",
        teacher_kind="frozen_baseline",
        outcome="time_out",
    )
    vectorized["num_envs"] = 8
    with pytest.raises(ValueError, match="exactly one environment"):
        SUCCESSOR_TOOL.accept_teacher_pair(
            _save(tmp_path / "vectorized.pt", vectorized),
            tmp_path / "control-b.pt",
            tmp_path / "teacher.pt",
        )


def test_teacher_acceptance_rejects_safety_events(tmp_path: Path) -> None:
    control = _trace(
        role="control",
        teacher_kind="frozen_baseline",
        outcome="time_out",
    )
    teacher = _trace(
        role="teacher",
        teacher_kind="clinician_teleoperation",
        outcome="success",
        teacher_branch=5,
        unsafe=True,
    )
    with pytest.raises(ValueError, match="safety event"):
        SUCCESSOR_TOOL.accept_teacher_pair(
            _save(tmp_path / "control-a.pt", control),
            _save(tmp_path / "control-b.pt", control),
            _save(tmp_path / "teacher.pt", teacher),
        )


def test_repeated_rescue_campaigns_keep_both_provenance_bound_examples(
    tmp_path: Path,
) -> None:
    paths = [
        tmp_path / "candidate-a.pt",
        tmp_path / "candidate-b.pt",
    ]
    paths[0].write_bytes(b"first accepted rescue")
    paths[1].write_bytes(b"second accepted rescue")
    datasets = [
        {
            "pair_id": "rescue-527699-phase0",
            "label_source": SUCCESSOR_TOOL.TEACHER_LABEL_SOURCE,
            "control_policy_kind": "successor_candidate",
            "successor_checkpoint_sha256": "a" * 64,
            "seed": 527699,
            "task": "DrAnmar-Handover-Needle-Dual-PSM-IK-Rel-v0",
        },
        {
            "pair_id": "rescue-527699-phase0",
            "label_source": SUCCESSOR_TOOL.TEACHER_LABEL_SOURCE,
            "control_policy_kind": "successor_candidate",
            "successor_checkpoint_sha256": "b" * 64,
            "seed": 527699,
            "task": "DrAnmar-Handover-Needle-Dual-PSM-IK-Rel-v0",
        },
    ]

    example_ids, artifact_hashes = (
        SUCCESSOR_TOOL._training_example_ids(paths, datasets)
    )

    assert example_ids == [
        "rescue-527699-phase0@successor-aaaaaaaaaaaaaaaa",
        "rescue-527699-phase0@successor-bbbbbbbbbbbbbbbb",
    ]
    assert len(set(artifact_hashes)) == 2

    datasets[1]["successor_checkpoint_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="distinct frozen candidate"):
        SUCCESSOR_TOOL._training_example_ids(paths, datasets)


def test_hybrid_ppo_recurrent_log_probability_replay() -> None:
    policy = SUCCESSOR_POLICY.PhaseConditionedHandoverPolicy(
        torch.zeros(98),
        torch.ones(98),
        hidden_dims=(32, 32),
        memory_dim=16,
        head_dim=16,
    )
    actor_critic = SUCCESSOR_PPO.RecurrentHybridActorCritic(policy)
    observations = torch.zeros(3, 2, 98)
    observations[:, :, 77] = 1.0
    initial_hidden = actor_critic.initial_hidden(
        2,
        device=torch.device("cpu"),
    )
    dones = torch.zeros(3, 2, dtype=torch.bool)
    dones[0, 0] = True
    hidden = initial_hidden
    sampled = []
    for step, observation in enumerate(observations):
        if step:
            hidden = hidden * (
                ~dones[step - 1]
            ).to(hidden.dtype).view(1, -1, 1)
        result = actor_critic.act(observation, hidden)
        sampled.append(result)
        hidden = result["hidden"]

    replayed = actor_critic.evaluate_sequence(
        observations,
        initial_hidden,
        dones,
        torch.stack([item["motion_mode"] for item in sampled]),
        torch.stack(
            [item["precision_pre_tanh"] for item in sampled]
        ),
        torch.stack([item["gripper_bits"] for item in sampled]),
    )

    assert torch.allclose(
        replayed["log_probability"],
        torch.stack(
            [item["log_probability"] for item in sampled]
        ),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    assert torch.isfinite(replayed["entropy"]).all()
    assert replayed["value"].shape == (3, 2)
    assert replayed["safety_value"].shape == (3, 2)


def test_hybrid_ppo_exploration_matches_rescue_calibration() -> None:
    policy = SUCCESSOR_POLICY.PhaseConditionedHandoverPolicy(
        torch.zeros(98),
        torch.ones(98),
        hidden_dims=(16,),
        memory_dim=8,
        head_dim=8,
    )
    actor_critic = SUCCESSOR_PPO.RecurrentHybridActorCritic(policy)
    motion_logits = torch.tensor(
        [[[1.0, 0.0, -1.0]] * 12],
        dtype=torch.float32,
    )
    precision_mean = torch.zeros(1, 12)
    gripper_logits = torch.tensor([[1.0, -1.0]])
    phase = torch.zeros(1, dtype=torch.long)

    motion, precision, gripper = actor_critic._distribution(
        motion_logits,
        precision_mean,
        gripper_logits,
        phase,
    )

    torch.testing.assert_close(
        precision.scale,
        torch.full_like(
            precision.scale,
            SUCCESSOR_PPO.PRECISION_EXPLORATION_STD,
        ),
    )
    torch.testing.assert_close(
        motion.probs,
        torch.softmax(
            motion_logits
            / SUCCESSOR_PPO.MOTION_EXPLORATION_TEMPERATURE,
            dim=-1,
        ),
    )
    torch.testing.assert_close(
        gripper.probs,
        torch.sigmoid(
            gripper_logits
            / SUCCESSOR_PPO.GRIPPER_EXPLORATION_TEMPERATURE
        ),
    )


def test_hybrid_ppo_deterministic_actor_matches_exported_policy_step() -> None:
    policy = SUCCESSOR_POLICY.PhaseConditionedHandoverPolicy(
        torch.zeros(98),
        torch.ones(98),
        hidden_dims=(16,),
        memory_dim=8,
        head_dim=8,
    )
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        # Both saturation limits clear the exact 1.5 margin.  The deployed
        # actor resolves this edge case to the positive limit.
        output = policy.phase_heads[0][-1]
        output.bias[14] = 1.5
        output.bias[16] = 1.5
    actor_critic = SUCCESSOR_PPO.RecurrentHybridActorCritic(policy)
    observation = torch.zeros(2, 98)
    observation[:, 77] = 1.0
    hidden = policy.initial_hidden(2, device=torch.device("cpu"))

    expected_action, expected_hidden = policy.step(
        observation,
        hidden,
    )
    observed = actor_critic.act(
        observation,
        hidden,
        deterministic=True,
    )

    assert torch.equal(observed["action"], expected_action)
    assert torch.equal(observed["hidden"], expected_hidden)
    assert torch.equal(
        observed["motion_mode"][:, 0],
        torch.full((2,), 2, dtype=torch.long),
    )


def test_hybrid_ppo_imitation_minibatches_are_exactly_source_balanced() -> None:
    sources = (
        [SUCCESSOR_TOOL.TEACHER_LABEL_SOURCE] * 80
        + [SUCCESSOR_TOOL.BASELINE_LABEL_SOURCE] * 8
        + [SUCCESSOR_TOOL.DAGGER_LABEL_SOURCE] * 8
    )
    indices = SUCCESSOR_PPO._source_balanced_batch_indices(
        sources,
        teacher_source=SUCCESSOR_TOOL.TEACHER_LABEL_SOURCE,
        baseline_source=SUCCESSOR_TOOL.BASELINE_LABEL_SOURCE,
        dagger_source=SUCCESSOR_TOOL.DAGGER_LABEL_SOURCE,
        generator=torch.Generator().manual_seed(104729),
    )
    observed = [sources[index] for index in indices]

    assert len(indices) == 4
    assert observed.count(SUCCESSOR_TOOL.TEACHER_LABEL_SOURCE) == 2
    assert observed.count(SUCCESSOR_TOOL.BASELINE_LABEL_SOURCE) == 1
    assert observed.count(SUCCESSOR_TOOL.DAGGER_LABEL_SOURCE) == 1


def test_hybrid_ppo_zero_cost_dual_starts_active_and_never_decays() -> None:
    costs = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    dones = torch.tensor(
        [[True, False, False], [True, True, False]]
    )

    updated, rate = SUCCESSOR_PPO._updated_safety_multiplier(
        1.0,
        costs,
        dones,
        learning_rate=0.05,
    )
    unchanged, zero_rate = SUCCESSOR_PPO._updated_safety_multiplier(
        updated,
        torch.zeros_like(costs),
        torch.zeros_like(dones),
        learning_rate=0.05,
    )

    assert rate == pytest.approx(2.0 / 3.0)
    assert updated == pytest.approx(1.0 + 0.05 * 2.0 / 3.0)
    assert zero_rate == 0.0
    assert unchanged == updated


def test_hybrid_ppo_backtracks_adam_step_into_empirical_kl_region() -> None:
    module = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        module.weight.zero_()
    optimizer = torch.optim.Adam(module.parameters(), lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    module.weight.grad = torch.ones_like(module.weight)

    accepted, scale, observed_kl, attempts = (
        SUCCESSOR_PPO._backtracked_optimizer_step(
            module,
            optimizer,
            nominal_learning_rate=0.1,
            target_kl=0.001,
            measure_kl=lambda: float(module.weight.square().item()),
        )
    )

    assert accepted
    assert attempts > 0
    assert 0.0 < scale < 1.0
    assert observed_kl <= 0.0015
    assert module.weight.abs().item() > 0.0
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_hybrid_ppo_requires_25_iteration_nonqualification_chunks() -> None:
    args = SUCCESSOR_PPO._parser().parse_args(
        [
            "--checkpoint",
            "candidate.pt",
            "--dataset_manifest",
            "manifest.json",
            "--output_dir",
            "output",
        ]
    )
    SUCCESSOR_PPO._validate_arguments(args)

    args.iterations = 1
    with pytest.raises(ValueError, match="25-iteration"):
        SUCCESSOR_PPO._validate_arguments(args)
    args.iterations = 25
    args.seed = 17
    with pytest.raises(ValueError, match="qualification seed"):
        SUCCESSOR_PPO._validate_arguments(args)


def test_rescue_weighted_validation_rejects_discrete_regression() -> None:
    assert not SUCCESSOR_TOOL._validation_checkpoint_improved(
        0.19,
        12,
        9,
        best_validation_loss=0.20,
        best_validation_gripper_errors=10,
        best_validation_saturation_errors=8,
    )
    assert SUCCESSOR_TOOL._validation_checkpoint_improved(
        0.19,
        10,
        8,
        best_validation_loss=0.20,
        best_validation_gripper_errors=10,
        best_validation_saturation_errors=8,
    )
    assert not SUCCESSOR_TOOL._validation_checkpoint_improved(
        0.21,
        8,
        7,
        best_validation_loss=0.20,
        best_validation_gripper_errors=10,
        best_validation_saturation_errors=8,
    )
    assert not SUCCESSOR_TOOL._validation_checkpoint_improved(
        0.20,
        9,
        20,
        best_validation_loss=0.20,
        best_validation_gripper_errors=10,
        best_validation_saturation_errors=8,
    )
    assert SUCCESSOR_TOOL._validation_checkpoint_improved(
        0.20,
        9,
        8,
        best_validation_loss=0.20,
        best_validation_gripper_errors=10,
        best_validation_saturation_errors=8,
    )


def test_seed_grouped_split_keeps_unique_action_class_in_training() -> None:
    datasets = []
    seeds = (104729, 130363, 196613, 262147)
    ranked = sorted(
        seeds,
        key=lambda value: hashlib.sha256(str(value).encode()).digest(),
    )
    rare_seed = ranked[0]
    for seed in seeds:
        actions = torch.zeros(4, 14)
        if seed == rare_seed:
            actions[0, 0] = 1.0
        datasets.append(
            {
                "seed": seed,
                "episode": {
                    "actions": actions,
                    "phases": torch.arange(4),
                },
            }
        )

    validation_seeds = SUCCESSOR_TOOL._stable_validation_seeds(
        datasets,
        0.25,
        continuous_indices=tuple(
            SUCCESSOR_POLICY.HANDOVER_CONTINUOUS_INDICES
        ),
        gripper_indices=tuple(
            SUCCESSOR_POLICY.HANDOVER_GRIPPER_INDICES
        ),
    )

    assert len(validation_seeds) == 1
    assert rare_seed not in validation_seeds


def test_full_action_successor_trains_with_episode_level_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        SUCCESSOR_TOOL,
        "_handover_contract_manifest",
        lambda revision: {
            "revision": revision,
            "sha256": "f" * 64,
            "entries": [],
        },
    )
    dataset_paths = []
    for index in range(8):
        seed = (104729, 130363, 196613, 262147)[index % 4]
        trace = _trace(
            role="teacher",
            teacher_kind="constrained_trajectory_optimizer",
            outcome="success",
            pair_id=f"pair-{seed}-{index}",
            seed=seed,
            teacher_branch=5,
        )
        phases = torch.tensor([0, 0, 1, 1, 2, 2, 2, 3, 3, 3])
        trace["phases"] = phases
        trace["observations"][:, 77:82] = 0.0
        trace["observations"][
            torch.arange(phases.numel()),
            77 + phases,
        ] = 1.0
        dataset = {
            "schema_version": SUCCESSOR_TOOL.DATASET_SCHEMA,
            "accepted": True,
            "label_source": (
                SUCCESSOR_TOOL.BASELINE_LABEL_SOURCE
                if index < 3
                else (
                    SUCCESSOR_TOOL.DAGGER_LABEL_SOURCE
                    if index < 6
                    else SUCCESSOR_TOOL.TEACHER_LABEL_SOURCE
                )
            ),
            "gates": {"isolated": True, "safe": True, "teacher_wins": True},
            "pair_id": trace["pair_id"],
            "task": trace["task"],
            "seed": seed,
            "teacher_kind": trace["teacher_kind"],
            "source": {
                **trace["runtime"]["source"],
                "dranmar_revision": (
                    "c" * 40 if index < 4 else "e" * 40
                ),
            },
            "base_checkpoint_sha256": trace["policy"][
                "base_checkpoint_sha256"
            ],
            "episode": {
                "observations": trace["observations"],
                "actions": trace["actions"],
                "phases": trace["phases"],
                "frame_count": trace["observations"].shape[0],
            },
        }
        if index >= 6:
            dataset["branch_frame"] = 0
            dataset["teacher_receipt"] = {
                "optimization_start_frame": 5,
            }
        path = tmp_path / f"accepted-{index}.pt"
        torch.save(dataset, path)
        dataset_paths.append(str(path))

    initial_model = SUCCESSOR_POLICY.PhaseConditionedHandoverPolicy(
        torch.zeros(98),
        torch.ones(98),
        hidden_dims=(32, 32),
        memory_dim=16,
        head_dim=16,
    )
    initial_checkpoint = tmp_path / "initial-successor.pt"
    torch.save(
        {
            "schema_version": (
                SUCCESSOR_POLICY.SUCCESSOR_CHECKPOINT_SCHEMA
            ),
            "deployment_status": "candidate_only",
            "training_gate_passed": True,
            "task": trace["task"],
            "observation_dim": 98,
            "action_dim": 14,
            "phase_slice": [77, 82],
            "observation_mean": torch.zeros(98),
            "observation_std": torch.ones(98),
            "base_checkpoint_sha256": "b" * 64,
            "source": {
                "dranmar_revision": "c" * 40,
                "weight_training_dranmar_revision": "e" * 40,
                "asset_revision": "d" * 40,
            },
            "architecture": {
                "hidden_dims": [32, 32],
                "memory_dim": 16,
                "head_dim": 16,
                "runtime_heuristic_stack": False,
            },
            "model": initial_model.state_dict(),
        },
        initial_checkpoint,
    )
    output = tmp_path / "successor.pt"
    result = SUCCESSOR_TOOL.train_successor(
        argparse.Namespace(
            dataset=dataset_paths,
            output=str(output),
            epochs=2,
            episode_batch_size=4,
            learning_rate=3.0e-4,
            weight_decay=1.0e-5,
            validation_fraction=0.25,
            hidden_dims="32,32",
            memory_dim=16,
            head_dim=16,
            patience=2,
            seed=104729,
            initial_checkpoint=str(initial_checkpoint),
            device="cpu",
        )
    )
    assert result["deployment_status"] == "candidate_only"
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert payload["architecture"]["full_action_policy"] is True
    assert payload["architecture"]["runtime_heuristic_stack"] is False
    assert payload["architecture"]["recurrent_state"] == (
        "gru_reset_per_episode"
    )
    assert payload["architecture"]["saturation_logit_margin"] == (
        SUCCESSOR_POLICY.HANDOVER_SATURATION_LOGIT_MARGIN
    )
    assert payload["capability_scope"] == (
        "incumbent_on_policy_distillation_plus_teacher_rescues"
    )
    assert result["baseline_distillation_pairs"] == 3
    assert result["dagger_pairs"] == 3
    assert result["teacher_rescue_pairs"] == 2
    validation_seeds = set(payload["training"]["validation_seeds"])
    assert validation_seeds
    assert all(
        int(pair_id.split("-")[1]) not in validation_seeds
        for pair_id in payload["training"]["train_pair_ids"]
    )
    assert all(
        int(pair_id.split("-")[1]) in validation_seeds
        for pair_id in payload["training"]["validation_pair_ids"]
    )
    assert payload["training"]["episode_source_sampling"] == {
        SUCCESSOR_TOOL.TEACHER_LABEL_SOURCE: 0.5,
        SUCCESSOR_TOOL.BASELINE_LABEL_SOURCE: 0.25,
        SUCCESSOR_TOOL.DAGGER_LABEL_SOURCE: 0.25,
    }
    assert payload["training"]["discrete_transition_weight"] == 8.0
    assert payload["training"]["gripper_logit_margin"] == 1.5
    assert payload["training"]["gripper_loss_weight"] == 4.0
    assert payload["training"]["checkpoint_selection"] == (
        "rescue_weighted_hybrid_loss_with_"
        "nonincreasing_discrete_errors"
    )
    teacher_starts = payload["training"][
        "teacher_training_start_frames"
    ]
    assert teacher_starts
    assert set(teacher_starts.values()) == {0}
    assert payload["training"]["initial_checkpoint"]["sha256"] == (
        SUCCESSOR_TOOL._sha256(initial_checkpoint)
    )
    assert payload["source"]["training_data_dranmar_revision"] is None
    assert payload["source"]["training_data_dranmar_revisions"] == [
        "c" * 40,
        "e" * 40,
    ]
    assert (
        payload["source"][
            "training_data_environment_contract_sha256"
        ]
        == "f" * 64
    )
    assert set(payload["training"]["train_pair_ids"]).isdisjoint(
        payload["training"]["validation_pair_ids"]
    )

    model, loaded = SUCCESSOR_POLICY.load_handover_successor_checkpoint(
        str(output),
        device="cpu",
    )
    action = model(torch.zeros(2, 98))
    assert loaded["training_gate_passed"] is True
    assert action.shape == (2, 14)
    assert bool((action.abs() <= 1.0).all())
    assert bool(
        (
            action[:, SUCCESSOR_POLICY.HANDOVER_GRIPPER_INDICES].abs()
            == 1.0
        ).all()
    )
    terminal_observation = torch.zeros(1, 98)
    terminal_observation[:, 81] = 1.0
    assert torch.equal(
        model(terminal_observation),
        torch.zeros(1, 14),
    )
