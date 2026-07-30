from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "train_dranmar_active_custody_controller.py"
)
SPEC = importlib.util.spec_from_file_location(
    "train_dranmar_active_custody_controller",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _payload() -> dict[str, object]:
    count = 6
    action_id = torch.tensor((-1, 0, 1, -1, 0, 1))
    direction = torch.tensor((-1.0, 1.0, -1.0, 1.0, -1.0, 1.0))
    action_limit = 0.0025
    action = torch.zeros(count, 7)
    action[:, 2] = action_id.float() * direction * action_limit
    action[:, 6] = -1.0
    observation = torch.zeros(count, 98)
    observation[:, 82] = 1.0
    return {
        "schema_version": MODULE.DATASET_SCHEMA_VERSION,
        "seed": 104729,
        "seed_stream_offset": 6000013,
        "runtime_seed": 6104742,
        "num_envs": count,
        "base_checkpoint_sha256": "a" * 64,
        "receiver_candidate_checkpoint_sha256": "b" * 64,
        "observation_dimension": 98,
        "probe_frames": 1,
        "probe_intervention": "giver_gripper_open_pulse",
        "environment_index": torch.arange(count),
        "pre_observation": observation,
        "post_observation": observation.clone(),
        "receiver_correction": torch.zeros(count, 6),
        "retry_count": torch.zeros(count, dtype=torch.long),
        "probe_survived": torch.ones(count, dtype=torch.bool),
        "eventual_full_success": torch.tensor((True, False, True, True, True, False)),
        "termination_names": [
            "time_out",
            "object_dropping",
            "needle_dropped_after_pickup",
            "premature_giver_release",
            "receiver_retention_lost",
            "success",
            "excessive_object_force",
            "protected_surface_force",
        ],
        "eventual_termination_flags": torch.zeros(
            count,
            8,
            dtype=torch.bool,
        ),
        "randomization": "seeded_hash_uniform_three_arm",
        "randomization_seed": 6209471,
        "intervention_frames": 1,
        "intervention_action_limit": action_limit,
        "intervention_action_semantics": {
            str(key): value for key, value in MODULE._ACTION_SEMANTICS.items()
        },
        "assigned_action_id": action_id,
        "assigned_action_probability": torch.full((count,), 1.0 / 3.0),
        "applied_receiver_action": action,
        "force_centering_direction": direction,
    }


def test_intervention_loader_enforces_logged_propensity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "interventions.pt"
    payload = _payload()
    payload["assigned_action_probability"][0] = 0.5
    torch.save(payload, path)

    with pytest.raises(ValueError, match="behavior propensity"):
        MODULE._load_dataset(path)


def test_intervention_loader_accepts_bounded_three_arm_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "interventions.pt"
    torch.save(_payload(), path)

    loaded = MODULE._load_dataset(path)

    assert loaded["schema_version"] == MODULE.DATASET_SCHEMA_VERSION
    assert MODULE._causal_features(loaded).shape == (6, 171)


def test_doubly_robust_value_matches_influence_formula() -> None:
    probability = torch.tensor(
        (
            (0.2, 0.5, 0.7),
            (0.3, 0.6, 0.8),
            (0.4, 0.7, 0.9),
        )
    )
    target_action = torch.tensor((2, 2, 2))
    logged_action = torch.tensor((0, 1, 2))
    propensity = torch.full((3,), 1.0 / 3.0)
    outcome = torch.tensor((False, True, False))

    value = MODULE._dr_value(
        probability,
        target_action,
        logged_action,
        propensity,
        outcome,
    )

    expected = torch.tensor((0.7, 0.8, 0.9 + 3.0 * (0.0 - 0.9)))
    assert torch.allclose(value, expected)


def test_activation_threshold_caps_training_distribution() -> None:
    probability = torch.tensor(
        [[0.1, 0.5, value] for value in (0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58)]
    )

    threshold = MODULE._activation_threshold(
        probability,
        minimum_advantage=0.02,
        activation_cap=0.25,
    )
    actions = MODULE._controller_actions(
        probability,
        advantage_threshold=threshold,
    )

    assert threshold == pytest.approx(0.07)
    assert int((actions != MODULE.NO_OP_INDEX).sum()) == 2
