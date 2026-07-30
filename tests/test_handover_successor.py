from __future__ import annotations

import argparse
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
            }
        },
    }


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
    assert accepted["branch_frame"] == 5
    assert torch.equal(
        accepted["episode"]["actions"],
        teacher["actions"],
    )


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


def test_full_action_successor_trains_with_episode_level_split(
    tmp_path: Path,
) -> None:
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
        dataset = {
            "schema_version": SUCCESSOR_TOOL.DATASET_SCHEMA,
            "accepted": True,
            "gates": {"isolated": True, "safe": True, "teacher_wins": True},
            "pair_id": trace["pair_id"],
            "task": trace["task"],
            "seed": seed,
            "teacher_kind": trace["teacher_kind"],
            "source": trace["runtime"]["source"],
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
        path = tmp_path / f"accepted-{index}.pt"
        torch.save(dataset, path)
        dataset_paths.append(str(path))

    output = tmp_path / "successor.pt"
    result = SUCCESSOR_TOOL.train_successor(
        argparse.Namespace(
            dataset=dataset_paths,
            output=str(output),
            epochs=2,
            batch_size=16,
            learning_rate=3.0e-4,
            weight_decay=1.0e-5,
            validation_fraction=0.25,
            hidden_dims="32,32",
            head_dim=16,
            patience=2,
            seed=104729,
            device="cpu",
        )
    )
    assert result["deployment_status"] == "candidate_only"
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert payload["architecture"]["full_action_policy"] is True
    assert payload["architecture"]["runtime_heuristic_stack"] is False
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
