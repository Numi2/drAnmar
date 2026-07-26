from __future__ import annotations

import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dr_anmar_expert import ExpertDemonstrationController  # noqa: E402
from dr_anmar_rescue_dataset import (  # noqa: E402
    CONTACT_FEATURES,
    FLUID_BALANCE_FEATURES,
    OBSERVATION_KEYS,
    SCHEMA,
    VESSEL_FEATURES,
    VITAL_SIGN_FEATURES,
    build_rescue_policy_observation,
    merge_rescue_training_hdf5,
    write_rescue_training_hdf5,
)


def rescue_evidence(
    *,
    compression: float = 0.4,
    released: bool = False,
) -> dict:
    return {
        "measured_contact": {
            "left_normal_force_n": 1.8,
            "right_normal_force_n": 1.7,
        },
        "vessel": {
            "transient_compression_fraction": compression,
            "distal_perfusion_fraction": 0.65,
            "overload_damage_fraction": 0.0,
            "residual_flow_ml_s": 1.0,
        },
        "release_observed": released,
    }


def test_rescue_expert_requires_observed_patient_effect() -> None:
    target = np.asarray((0.0, 0.0, 0.055), dtype=np.float32)
    expert = ExpertDemonstrationController(
        procedure_id="dr-anmar-autonomous-rescue-or",
        guide_kind="autonomous_rescue_or",
        action_dim=14,
        arms=2,
        has_grippers=True,
        waypoints=np.zeros((0, 3), dtype=np.float32),
    )
    expert.start()
    expert.phase_index = 5
    expert.phase_started_at -= 2.0
    tools = {0: target.copy(), 1: target + (0.05, 0.0, 0.0)}

    for _ in range(25):
        command = expert.step(
            tools,
            target,
            [True, True],
            task_evidence={},
        )
    assert expert.phase == "manipulate"
    assert command.completed is False

    for _ in range(20):
        command = expert.step(
            tools,
            target,
            [True, True],
            task_evidence=rescue_evidence(),
        )
    assert expert.phase == "verify"
    assert command.phase_changed is True
    assert expert.rescue_peak_compression_fraction == 0.4


def test_live_rescue_observation_matches_training_contract() -> None:
    observation = build_rescue_policy_observation(
        joint_positions=(
            np.arange(7, dtype=np.float32),
            np.arange(7, dtype=np.float32) + 10.0,
        ),
        joint_velocities=(
            np.arange(7, dtype=np.float32) + 20.0,
            np.arange(7, dtype=np.float32) + 30.0,
        ),
        tool_positions_w=np.asarray(
            ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6)),
            dtype=np.float32,
        ),
        target_position_w=np.asarray(
            (0.5, 0.7, 0.9),
            dtype=np.float32,
        ),
        rescue_contact={
            "left_normal_force_n": 1.2,
            "right_normal_force_n": 1.1,
        },
        rescue_vessel={"distal_perfusion_fraction": 0.8},
        rescue_vital_signs={"heart_rate_bpm": 90.0},
        rescue_fluid_balance={"intravascular_volume_ml": 4200.0},
        procedure_phase=5,
        sensor_authority=True,
        tool_position_valid=np.asarray((1.0, 0.0)),
        selected_arm=2,
    )

    assert tuple(observation) == OBSERVATION_KEYS
    assert all(value.dtype == np.float32 for value in observation.values())
    assert all(value.ndim == 1 for value in observation.values())
    np.testing.assert_allclose(
        observation["target_relative_tool_positions"],
        np.asarray(
            (0.4, 0.5, 0.6, 0.1, 0.2, 0.3),
            dtype=np.float32,
        ),
        atol=1.0e-7,
    )
    np.testing.assert_array_equal(
        observation["selected_arm"],
        np.asarray((0.0, 1.0)),
    )


def _write_capture(path: Path, frames: int = 5) -> None:
    with h5py.File(path, "w") as output:
        output.create_dataset(
            "time_s",
            data=np.arange(frames, dtype=np.float64) * 0.02,
        )
        actions = np.repeat(
            np.arange(frames, dtype=np.float32)[:, None],
            14,
            axis=1,
        )
        output.create_dataset("cartesian_actions", data=actions)
        output.create_dataset(
            "environment_reward",
            data=np.arange(frames, dtype=np.float32),
        )
        output.create_dataset(
            "environment_terminated",
            data=np.zeros(frames, dtype=np.bool_),
        )
        output.create_dataset(
            "environment_truncated",
            data=np.zeros(frames, dtype=np.bool_),
        )
        output.create_dataset(
            "environment_success",
            data=np.zeros(frames, dtype=np.float32),
        )
        for robot_index, name in enumerate(("psm_1", "psm_2")):
            values = (
                np.arange(frames, dtype=np.float32)[:, None]
                + robot_index * 100.0
            )
            output.create_dataset(
                f"{name}_joint_positions",
                data=np.repeat(values, 7, axis=1),
            )
            output.create_dataset(
                f"{name}_joint_velocities",
                data=np.repeat(values + 0.5, 7, axis=1),
            )
        output.create_dataset(
            "rescue_tool_positions_w",
            data=np.zeros((frames, 2, 3), dtype=np.float32),
        )
        output.create_dataset(
            "rescue_target_position_w",
            data=np.repeat(
                np.asarray([[0.0, 0.0, 0.055]], dtype=np.float32),
                frames,
                axis=0,
            ),
        )
        output.create_dataset(
            "rescue_measured_contact",
            data=np.zeros(
                (frames, len(CONTACT_FEATURES)),
                dtype=np.float32,
            ),
        )
        output.create_dataset(
            "rescue_vessel_state",
            data=np.zeros(
                (frames, len(VESSEL_FEATURES)),
                dtype=np.float32,
            ),
        )
        output.create_dataset(
            "rescue_vital_signs",
            data=np.zeros(
                (frames, len(VITAL_SIGN_FEATURES)),
                dtype=np.float32,
            ),
        )
        output.create_dataset(
            "rescue_fluid_balance",
            data=np.zeros(
                (frames, len(FLUID_BALANCE_FEATURES)),
                dtype=np.float32,
            ),
        )
        output.create_dataset(
            "procedure_phase_code",
            data=np.arange(frames, dtype=np.int16),
        )
        output.create_dataset(
            "rescue_sensor_authority_available",
            data=np.ones(frames, dtype=np.float32),
        )
        output.create_dataset(
            "rescue_tool_position_valid",
            data=np.ones((frames, 2), dtype=np.float32),
        )
        output.create_dataset(
            "rescue_selected_arm_one_hot",
            data=np.repeat(
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                frames,
                axis=0,
            ),
        )
        output.create_dataset(
            "endoscope_time_s",
            data=np.asarray((0.01, 0.05), dtype=np.float64),
        )
        output.create_dataset(
            "endoscope_rgb",
            data=np.asarray(
                (
                    np.full((2, 3, 3), 17, dtype=np.uint8),
                    np.full((2, 3, 3), 29, dtype=np.uint8),
                )
            ),
        )
        output.create_dataset(
            "endoscope_sensor_dropout_active",
            data=np.asarray((False, True), dtype=np.bool_),
        )


def _export_capture(capture: Path, destination: Path, seed: int) -> None:
    write_rescue_training_hdf5(
        capture,
        destination,
        task="Isaac-Handover-Needle-Dual-PSM-IK-Rel-v0",
        procedure_id="dr-anmar-autonomous-rescue-or",
        scenario_id="baseline",
        scenario_seed=seed,
        robot_names=("psm_1", "psm_2"),
        action_contract={"id": "dr-anmar-cartesian-ik-relative-v1"},
        source_revision="test",
        reference_eligible=True,
        expert_status="completed",
    )


def test_rescue_export_uses_next_command_for_current_observation(
    tmp_path: Path,
) -> None:
    capture = tmp_path / "capture.hdf5"
    dataset = tmp_path / "episode.hdf5"
    _write_capture(capture)
    _export_capture(capture, dataset, 1)

    with h5py.File(dataset, "r") as data:
        episode = data["data/demo_0"]
        assert data.attrs["schema"] == SCHEMA
        assert episode.attrs["post_action_frame_offset"] == 1
        np.testing.assert_array_equal(
            episode["obs/joint_pos"][:, 0],
            np.asarray((0.0, 1.0, 2.0, 3.0)),
        )
        np.testing.assert_array_equal(
            episode["actions"][:, 0],
            np.asarray((1.0, 2.0, 3.0, 4.0)),
        )
        np.testing.assert_array_equal(
            episode["next_obs/joint_pos"][:, 0],
            np.asarray((1.0, 2.0, 3.0, 4.0)),
        )
        assert episode["dones"][-1] == 1
        assert episode["obs/room_valid"][0, 0] == 0.0
        assert np.all(episode["obs/room"][0] == 0)
        assert episode["next_obs/room_valid"][0, 0] == 1.0
        assert np.all(episode["next_obs/room"][0] == 17)
        assert np.all(episode["obs/room_age_s"][:] >= 0.0)
        env_args = json.loads(data["data"].attrs["env_args"])
        assert (
            env_args["env_kwargs"]["procedure_id"]
            == "dr-anmar-autonomous-rescue-or"
        )


def test_rescue_merge_splits_only_complete_episodes(
    tmp_path: Path,
) -> None:
    sources = []
    for index in range(3):
        capture = tmp_path / f"capture_{index}.hdf5"
        dataset = tmp_path / f"episode_{index}.hdf5"
        _write_capture(capture)
        _export_capture(capture, dataset, index)
        sources.append(dataset)
    merged = tmp_path / "merged.hdf5"
    merge_rescue_training_hdf5(
        sources,
        merged,
        validation_fraction=0.34,
        seed=9,
    )
    with h5py.File(merged, "r") as data:
        train = {
            item.decode() if isinstance(item, bytes) else str(item)
            for item in data["mask/train"][:]
        }
        valid = {
            item.decode() if isinstance(item, bytes) else str(item)
            for item in data["mask/valid"][:]
        }
        assert train.isdisjoint(valid)
        assert train | valid == set(data["data"].keys())
        assert len(valid) == 1


def test_rescue_merge_rejects_failed_expert_by_default(
    tmp_path: Path,
) -> None:
    capture = tmp_path / "capture.hdf5"
    dataset = tmp_path / "failed_episode.hdf5"
    _write_capture(capture)
    _export_capture(capture, dataset, 1)
    with h5py.File(dataset, "r+") as data:
        data["data/demo_0"].attrs["reference_eligible"] = False
    with pytest.raises(ValueError, match="not an eligible expert"):
        merge_rescue_training_hdf5(
            [dataset],
            tmp_path / "merged.hdf5",
        )
