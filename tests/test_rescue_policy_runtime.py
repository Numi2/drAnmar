from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dr_anmar_rescue_dataset import (  # noqa: E402
    OBSERVATION_KEYS,
    rescue_policy_observation_shapes,
)
from dr_anmar_rescue_policy import (  # noqa: E402
    RescueOutcomeMonitor,
    RescuePolicyRuntime,
    infer_rescue_phase_code,
)


class FakePolicy:
    def __init__(self, action: np.ndarray) -> None:
        self.action = action
        self.started = 0

    def start_episode(self) -> None:
        self.started += 1

    def __call__(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        assert tuple(observation) == OBSERVATION_KEYS
        return self.action.copy()


def _observation() -> dict[str, np.ndarray]:
    return {
        name: np.ones(2, dtype=np.float32)
        for name in OBSERVATION_KEYS
    }


def test_policy_runtime_owns_shape_finiteness_and_action_bounds() -> None:
    policy = FakePolicy(
        np.asarray([2.0, -2.0, 0.25], dtype=np.float32)
    )
    runtime = RescuePolicyRuntime(
        policy,
        checkpoint_path=Path("rescue.pth"),
        checkpoint_digest="abc",
        action_dim=3,
        device="cpu",
    )
    runtime.reset()
    action = runtime.act(_observation())

    np.testing.assert_array_equal(
        action,
        np.asarray([1.0, -1.0, 0.25], dtype=np.float32),
    )
    assert policy.started == 1
    assert runtime.steps == 1
    assert runtime.clipped_action_values == 2


def test_policy_runtime_rejects_non_finite_action() -> None:
    runtime = RescuePolicyRuntime(
        FakePolicy(np.asarray([np.nan], dtype=np.float32)),
        checkpoint_path=Path("rescue.pth"),
        checkpoint_digest="abc",
        action_dim=1,
        device="cpu",
    )
    with pytest.raises(ValueError, match="NaN or infinity"):
        runtime.act(_observation())


def test_checkpoint_contract_rejects_training_serving_shape_drift() -> None:
    shapes = {
        name: list(shape)
        for name, shape in rescue_policy_observation_shapes(2).items()
    }
    checkpoint = {
        "shape_metadata": {
            "ac_dim": 14,
            "all_obs_keys": list(OBSERVATION_KEYS),
            "all_shapes": shapes,
        }
    }
    RescuePolicyRuntime.validate_checkpoint_contract(
        checkpoint,
        action_dim=14,
    )
    checkpoint["shape_metadata"]["all_shapes"]["rescue_contact"] = [99]
    with pytest.raises(ValueError, match="rescue_contact"):
        RescuePolicyRuntime.validate_checkpoint_contract(
            checkpoint,
            action_dim=14,
        )


def test_outcome_monitor_requires_contact_hold_then_release() -> None:
    monitor = RescueOutcomeMonitor(required_effective_steps=3)
    effective = {
        "sensor_authority_available": True,
        "measured_contact": {
            "left_normal_force_n": 1.0,
            "right_normal_force_n": 1.0,
        },
        "vessel": {
            "transient_compression_fraction": 0.4,
            "overload_damage_fraction": 0.0,
            "distal_perfusion_fraction": 0.7,
        },
        "release_observed": False,
    }
    for _ in range(3):
        state = monitor.update(effective)
    assert state["effective_hold_observed"] is True
    assert state["success"] is False

    released = {
        **effective,
        "measured_contact": {
            "left_normal_force_n": 0.0,
            "right_normal_force_n": 0.0,
        },
        "vessel": {
            **effective["vessel"],
            "transient_compression_fraction": 0.0,
        },
        "release_observed": True,
    }
    state = monitor.update(released)
    assert state["success"] is True
    assert state["status"] == "completed"


def test_outcome_monitor_does_not_accept_telemetry_only_release() -> None:
    monitor = RescueOutcomeMonitor(required_effective_steps=1)
    state = monitor.update(
        {
            "sensor_authority_available": False,
            "measured_contact": {},
            "vessel": {
                "transient_compression_fraction": 0.0,
                "overload_damage_fraction": 0.0,
                "distal_perfusion_fraction": 1.0,
            },
            "release_observed": True,
        }
    )
    assert state["success"] is False


def test_outcome_monitor_rejects_recovered_end_state_after_patient_harm() -> None:
    monitor = RescueOutcomeMonitor(required_effective_steps=1)
    harmful_hold = {
        "sensor_authority_available": True,
        "measured_contact": {
            "left_normal_force_n": 2.0,
            "right_normal_force_n": 2.0,
        },
        "vessel": {
            "transient_compression_fraction": 0.5,
            "overload_damage_fraction": 0.2,
            "distal_perfusion_fraction": 0.1,
        },
        "vital_signs": {
            "mean_arterial_pressure_mmhg": 40.0,
            "spo2_fraction": 0.7,
            "global_perfusion_fraction": 0.3,
        },
    }
    monitor.update(harmful_hold)
    monitor.effective_hold_observed = True
    recovered_release = {
        **harmful_hold,
        "measured_contact": {},
        "vessel": {
            "transient_compression_fraction": 0.0,
            "overload_damage_fraction": 0.0,
            "distal_perfusion_fraction": 1.0,
        },
        "vital_signs": {
            "mean_arterial_pressure_mmhg": 100.0,
            "spo2_fraction": 1.0,
            "global_perfusion_fraction": 1.0,
        },
        "release_observed": True,
    }
    state = monitor.update(recovered_release)
    assert state["release_observed"] is True
    assert state["success"] is False


def test_policy_phase_is_driven_by_physical_patient_evidence() -> None:
    assert infer_rescue_phase_code(
        {},
        nearest_target_distance_m=0.03,
        effective_hold_observed=False,
        success=False,
    ) == 1
    contact = {
        "sensor_authority_available": True,
        "measured_contact": {
            "left_normal_force_n": 1.0,
            "right_normal_force_n": 1.0,
        },
        "vessel": {"transient_compression_fraction": 0.3},
    }
    assert infer_rescue_phase_code(
        contact,
        nearest_target_distance_m=0.0,
        effective_hold_observed=False,
        success=False,
    ) == 5
    assert infer_rescue_phase_code(
        contact,
        nearest_target_distance_m=0.0,
        effective_hold_observed=True,
        success=False,
    ) == 6
