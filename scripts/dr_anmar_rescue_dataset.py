# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Transition-aligned imitation datasets for Autonomous Rescue OR.

The workstation records each control frame after applying its command.  A
training transition therefore uses state[i] -> action[i + 1] -> state[i + 1].
Keeping that offset explicit prevents behavior cloning from seeing the result
of an action in the observation used to predict the same action.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np


SCHEMA = "dr.anmar.autonomous-rescue-imitation.v1"
OUTCOME_AUTHORITY = "prim_bound_post_physics_scene_evidence"

CONTACT_FEATURES = (
    "left_normal_force_n",
    "right_normal_force_n",
    "jaw_separation_m",
    "tool_speed_m_s",
    "target_distance_m",
)
VESSEL_FEATURES = (
    "residual_flow_ml_s",
    "distal_perfusion_fraction",
    "modeled_cumulative_leak_volume_ml",
    "transient_compression_fraction",
    "retained_clip_fraction",
    "patch_seal_fraction",
    "overload_damage_fraction",
    "pressure_challenge_active",
    "measured_upstream_pressure_mmhg",
    "hemostasis_verified",
)
VITAL_SIGN_FEATURES = (
    "heart_rate_bpm",
    "respiratory_rate_bpm",
    "systolic_pressure_mmhg",
    "diastolic_pressure_mmhg",
    "mean_arterial_pressure_mmhg",
    "spo2_fraction",
    "etco2_mmhg",
    "core_temperature_c",
    "cardiac_output_l_min",
    "shock_index",
    "lactate_mmol_l",
    "cumulative_blood_loss_ml",
    "active_blood_loss_ml_min",
    "urine_output_ml_h",
    "bile_leak_ml_h",
    "global_perfusion_fraction",
)
FLUID_BALANCE_FEATURES = (
    "baseline_blood_volume_ml",
    "intravascular_volume_ml",
    "interstitial_volume_ml",
    "plasma_excess_ml",
    "interstitial_excess_ml",
    "hemoglobin_mass_g",
    "hemoglobin_g_dl",
    "crystalloid_input_ml",
    "colloid_input_ml",
    "transfused_red_cell_ml",
    "cumulative_blood_loss_ml",
    "urine_output_ml",
    "bile_output_ml",
    "suction_output_ml",
    "irrigation_input_ml",
    "irrigation_recovered_ml",
)

RECORDED_RESCUE_FIELDS = {
    "rescue_measured_contact": CONTACT_FEATURES,
    "rescue_vessel_state": VESSEL_FEATURES,
    "rescue_vital_signs": VITAL_SIGN_FEATURES,
    "rescue_fluid_balance": FLUID_BALANCE_FEATURES,
}

OBSERVATION_KEYS = (
    "joint_pos",
    "joint_vel",
    "tool_positions_w",
    "target_relative_tool_positions",
    "rescue_contact",
    "rescue_vessel",
    "rescue_vital_signs",
    "rescue_fluid_balance",
    "procedure_phase",
    "sensor_authority",
    "tool_position_valid",
    "selected_arm",
)


def rescue_policy_observation_shapes(
    arms: int,
) -> dict[str, tuple[int, ...]]:
    if arms < 1:
        raise ValueError("rescue policy requires at least one PSM")
    return {
        "joint_pos": (arms * 7,),
        "joint_vel": (arms * 7,),
        "tool_positions_w": (arms * 3,),
        "target_relative_tool_positions": (arms * 3,),
        "rescue_contact": (len(CONTACT_FEATURES),),
        "rescue_vessel": (len(VESSEL_FEATURES),),
        "rescue_vital_signs": (len(VITAL_SIGN_FEATURES),),
        "rescue_fluid_balance": (len(FLUID_BALANCE_FEATURES),),
        "procedure_phase": (1,),
        "sensor_authority": (1,),
        "tool_position_valid": (arms,),
        "selected_arm": (arms,),
    }


def rescue_vector(
    values: Mapping[str, Any],
    features: Sequence[str],
) -> np.ndarray:
    """Return one finite fixed-width vector in the declared feature order."""

    result = []
    for name in features:
        value = values.get(name, 0.0)
        if isinstance(value, (bool, np.bool_)):
            result.append(float(value))
        elif value is None:
            result.append(0.0)
        else:
            result.append(float(value))
    array = np.asarray(result, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("rescue telemetry contains NaN or infinity")
    return array


def _read_required(source: h5py.File, name: str, frames: int) -> np.ndarray:
    if name not in source:
        raise ValueError(f"rescue recording is missing {name}")
    values = np.asarray(source[name])
    if values.ndim == 0 or values.shape[0] != frames:
        raise ValueError(
            f"rescue recording field {name} is not frame-aligned: "
            f"{values.shape} versus ({frames}, ...)"
        )
    return values


def _as_feature_matrix(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        values = values[:, None]
    return values.reshape(values.shape[0], -1).astype(np.float32, copy=False)


def build_rescue_policy_observations(
    *,
    joint_positions: Sequence[np.ndarray],
    joint_velocities: Sequence[np.ndarray],
    tool_positions_w: np.ndarray,
    target_position_w: np.ndarray,
    rescue_contact: np.ndarray,
    rescue_vessel: np.ndarray,
    rescue_vital_signs: np.ndarray,
    rescue_fluid_balance: np.ndarray,
    procedure_phase: np.ndarray,
    sensor_authority: np.ndarray,
    tool_position_valid: np.ndarray,
    selected_arm: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build the canonical batched observation consumed by rescue policies.

    This function is the single training-serving contract. The dataset
    exporter and the live workstation both call it so a policy cannot receive
    a subtly different feature ordering or target-relative convention at
    rollout time.
    """

    tools = np.asarray(tool_positions_w, dtype=np.float32)
    target = np.asarray(target_position_w, dtype=np.float32)
    if tools.ndim != 3 or tools.shape[2] != 3:
        raise ValueError(
            "rescue tool positions must have shape (frames, arms, 3)"
        )
    frames = int(tools.shape[0])
    if target.shape != (frames, 3):
        raise ValueError(
            "rescue target positions must have shape (frames, 3)"
        )
    arms = int(tools.shape[1])
    if (
        not joint_positions
        or len(joint_positions) != len(joint_velocities)
        or len(joint_positions) != arms
    ):
        raise ValueError(
            "rescue observations require matching joint state for every arm"
        )

    def matrix(name: str, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim == 0 or array.shape[0] != frames:
            raise ValueError(
                f"rescue observation {name} is not frame-aligned"
            )
        result = _as_feature_matrix(array)
        if not np.isfinite(result).all():
            raise ValueError(
                f"rescue observation {name} contains NaN or infinity"
            )
        return result

    position_matrices = [
        matrix(f"joint_pos[{index}]", values)
        for index, values in enumerate(joint_positions)
    ]
    velocity_matrices = [
        matrix(f"joint_vel[{index}]", values)
        for index, values in enumerate(joint_velocities)
    ]
    relative = target[:, None, :] - tools
    observations = {
        "joint_pos": np.concatenate(position_matrices, axis=1),
        "joint_vel": np.concatenate(velocity_matrices, axis=1),
        "tool_positions_w": matrix("tool_positions_w", tools),
        "target_relative_tool_positions": matrix(
            "target_relative_tool_positions",
            relative,
        ),
        "rescue_contact": matrix("rescue_contact", rescue_contact),
        "rescue_vessel": matrix("rescue_vessel", rescue_vessel),
        "rescue_vital_signs": matrix(
            "rescue_vital_signs",
            rescue_vital_signs,
        ),
        "rescue_fluid_balance": matrix(
            "rescue_fluid_balance",
            rescue_fluid_balance,
        ),
        "procedure_phase": matrix("procedure_phase", procedure_phase),
        "sensor_authority": matrix("sensor_authority", sensor_authority),
        "tool_position_valid": matrix(
            "tool_position_valid",
            tool_position_valid,
        ),
        "selected_arm": matrix("selected_arm", selected_arm),
    }
    if tuple(observations) != OBSERVATION_KEYS:
        raise RuntimeError("canonical rescue observation order drifted")
    expected_shapes = rescue_policy_observation_shapes(arms)
    for name, shape in expected_shapes.items():
        if observations[name].shape != (frames, *shape):
            raise ValueError(
                f"rescue observation {name} has shape "
                f"{observations[name].shape}; "
                f"expected {(frames, *shape)}"
            )
    return observations


def build_rescue_policy_observation(
    *,
    joint_positions: Sequence[np.ndarray],
    joint_velocities: Sequence[np.ndarray],
    tool_positions_w: np.ndarray,
    target_position_w: np.ndarray,
    rescue_contact: Mapping[str, Any],
    rescue_vessel: Mapping[str, Any],
    rescue_vital_signs: Mapping[str, Any],
    rescue_fluid_balance: Mapping[str, Any],
    procedure_phase: int,
    sensor_authority: bool,
    tool_position_valid: np.ndarray,
    selected_arm: int,
) -> dict[str, np.ndarray]:
    """Build one unbatched live policy observation.

    ``selected_arm`` uses the runtime's one-based PSM numbering. Zero means
    that no contact-authoritative arm is currently selected.
    """

    arms = int(np.asarray(tool_positions_w).shape[0])
    if not 0 <= selected_arm <= arms:
        raise ValueError(
            f"selected rescue arm {selected_arm} is outside 0..{arms}"
        )
    selected_arm_one_hot = np.zeros(arms, dtype=np.float32)
    if selected_arm:
        selected_arm_one_hot[selected_arm - 1] = 1.0
    batched = build_rescue_policy_observations(
        joint_positions=[
            np.asarray(values)[None, ...]
            for values in joint_positions
        ],
        joint_velocities=[
            np.asarray(values)[None, ...]
            for values in joint_velocities
        ],
        tool_positions_w=np.asarray(
            tool_positions_w,
            dtype=np.float32,
        )[None, ...],
        target_position_w=np.asarray(
            target_position_w,
            dtype=np.float32,
        )[None, ...],
        rescue_contact=rescue_vector(
            rescue_contact,
            CONTACT_FEATURES,
        )[None, ...],
        rescue_vessel=rescue_vector(
            rescue_vessel,
            VESSEL_FEATURES,
        )[None, ...],
        rescue_vital_signs=rescue_vector(
            rescue_vital_signs,
            VITAL_SIGN_FEATURES,
        )[None, ...],
        rescue_fluid_balance=rescue_vector(
            rescue_fluid_balance,
            FLUID_BALANCE_FEATURES,
        )[None, ...],
        procedure_phase=np.asarray([procedure_phase], dtype=np.float32),
        sensor_authority=np.asarray(
            [sensor_authority],
            dtype=np.float32,
        ),
        tool_position_valid=np.asarray(
            tool_position_valid,
            dtype=np.float32,
        )[None, ...],
        selected_arm=selected_arm_one_hot[None, ...],
    )
    return {
        name: np.ascontiguousarray(values[0], dtype=np.float32)
        for name, values in batched.items()
    }


def _write_array(
    group: h5py.Group,
    name: str,
    values: np.ndarray,
) -> h5py.Dataset:
    values = np.asarray(values)
    rows = int(values.shape[0])
    chunks = (min(128, max(1, rows)), *values.shape[1:])
    return group.create_dataset(
        name,
        data=values,
        chunks=chunks,
        compression="lzf",
        shuffle=values.dtype.itemsize > 1,
    )


def _transition_observations(
    source: h5py.File,
    robot_names: Sequence[str],
    frames: int,
) -> dict[str, np.ndarray]:
    joint_positions = [
        _as_feature_matrix(
            _read_required(source, f"{name}_joint_positions", frames)
        )
        for name in robot_names
    ]
    joint_velocities = [
        _as_feature_matrix(
            _read_required(source, f"{name}_joint_velocities", frames)
        )
        for name in robot_names
    ]
    return build_rescue_policy_observations(
        joint_positions=joint_positions,
        joint_velocities=joint_velocities,
        tool_positions_w=_read_required(
            source,
            "rescue_tool_positions_w",
            frames,
        ),
        target_position_w=_read_required(
            source,
            "rescue_target_position_w",
            frames,
        ),
        rescue_contact=_read_required(
            source,
            "rescue_measured_contact",
            frames,
        ),
        rescue_vessel=_read_required(
            source,
            "rescue_vessel_state",
            frames,
        ),
        rescue_vital_signs=_read_required(
            source,
            "rescue_vital_signs",
            frames,
        ),
        rescue_fluid_balance=_read_required(
            source,
            "rescue_fluid_balance",
            frames,
        ),
        procedure_phase=_read_required(
            source,
            "procedure_phase_code",
            frames,
        ),
        sensor_authority=_read_required(
            source,
            "rescue_sensor_authority_available",
            frames,
        ),
        tool_position_valid=_read_required(
            source,
            "rescue_tool_position_valid",
            frames,
        ),
        selected_arm=_read_required(
            source,
            "rescue_selected_arm_one_hot",
            frames,
        ),
    )


def _write_optional_aligned_vision(
    source: h5py.File,
    obs_group: h5py.Group,
    next_obs_group: h5py.Group,
    control_time_s: np.ndarray,
) -> tuple[str, ...]:
    required = {"endoscope_rgb", "endoscope_time_s"}
    if not required.issubset(source.keys()):
        return ()
    vision_time_s = np.asarray(
        source["endoscope_time_s"],
        dtype=np.float64,
    ).reshape(-1)
    if not len(vision_time_s):
        return ()
    if (
        not np.isfinite(vision_time_s).all()
        or np.any(np.diff(vision_time_s) < 0.0)
    ):
        raise ValueError(
            "endoscope_time_s is non-finite or non-monotonic"
        )
    raw_indices = (
        np.searchsorted(
            vision_time_s,
            control_time_s,
            side="right",
        )
        - 1
    )
    has_causal_frame = raw_indices >= 0
    indices = np.clip(raw_indices, 0, len(vision_time_s) - 1)
    dropout = (
        np.asarray(
            source["endoscope_sensor_dropout_active"],
            dtype=np.bool_,
        ).reshape(-1)
        if "endoscope_sensor_dropout_active" in source
        else np.zeros(len(vision_time_s), dtype=np.bool_)
    )
    if len(dropout) != len(vision_time_s):
        raise ValueError(
            "endoscope dropout state is not vision-frame aligned"
        )
    selected_dropout = dropout[indices]
    valid = (has_causal_frame & ~selected_dropout).astype(np.float32)
    age_s = np.where(
        has_causal_frame,
        control_time_s - vision_time_s[indices],
        0.0,
    ).astype(np.float32)
    if np.any(age_s < -1.0e-9):
        raise ValueError("endoscope alignment selected a future frame")

    rgb = source["endoscope_rgb"]
    transition_count = len(control_time_s) - 1
    for group, selected, selected_valid in (
        (obs_group, indices[:-1], valid[:-1].astype(np.bool_)),
        (next_obs_group, indices[1:], valid[1:].astype(np.bool_)),
    ):
        room = group.create_dataset(
            "room",
            shape=(transition_count, *rgb.shape[1:]),
            dtype=rgb.dtype,
            chunks=(
                min(8, max(1, transition_count)),
                *rgb.shape[1:],
            ),
            compression="lzf",
        )
        for start in range(0, transition_count, 8):
            batch = selected[start : start + 8]
            batch_values = np.stack(
                [rgb[int(index)] for index in batch]
            )
            invalid = ~selected_valid[start : start + len(batch)]
            if np.any(invalid):
                batch_values[invalid] = 0
            room[start : start + len(batch)] = batch_values
    _write_array(obs_group, "room_valid", valid[:-1, None])
    _write_array(next_obs_group, "room_valid", valid[1:, None])
    _write_array(obs_group, "room_age_s", age_s[:-1, None])
    _write_array(next_obs_group, "room_age_s", age_s[1:, None])
    _write_array(
        obs_group,
        "room_sensor_dropout",
        selected_dropout[:-1, None].astype(np.float32),
    )
    _write_array(
        next_obs_group,
        "room_sensor_dropout",
        selected_dropout[1:, None].astype(np.float32),
    )
    return (
        "room",
        "room_valid",
        "room_age_s",
        "room_sensor_dropout",
    )


def write_rescue_training_hdf5(
    spool_path: Path,
    destination: Path,
    *,
    task: str,
    procedure_id: str,
    scenario_id: str,
    scenario_seed: int,
    robot_names: Sequence[str],
    action_contract: Mapping[str, Any],
    source_revision: str | None,
    reference_eligible: bool,
    expert_status: str,
) -> Path:
    """Write one strict, Robomimic-compatible rescue demonstration.

    Control frames contain post-action state.  Frame zero has no recorded
    predecessor and is used only as the first observation; transition actions,
    rewards, and next observations begin at frame one.
    """

    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with h5py.File(spool_path, "r") as source:
            if "time_s" not in source:
                raise ValueError("rescue recording is missing time_s")
            frames = int(source["time_s"].shape[0])
            if frames < 2:
                raise ValueError(
                    "rescue imitation export requires at least two control frames"
                )
            actions = _read_required(
                source,
                "cartesian_actions",
                frames,
            ).astype(np.float32, copy=False)
            if actions.ndim != 2 or not np.isfinite(actions).all():
                raise ValueError(
                    "cartesian rescue actions must be a finite matrix"
                )
            observations = _transition_observations(
                source,
                robot_names,
                frames,
            )
            rewards = _read_required(
                source,
                "environment_reward",
                frames,
            ).astype(np.float32, copy=False)
            terminated = _read_required(
                source,
                "environment_terminated",
                frames,
            ).astype(np.bool_, copy=False)
            truncated = _read_required(
                source,
                "environment_truncated",
                frames,
            ).astype(np.bool_, copy=False)
            success = _read_required(
                source,
                "environment_success",
                frames,
            ).astype(np.float32, copy=False)
            times = np.asarray(source["time_s"], dtype=np.float64)
            if not np.isfinite(times).all() or np.any(np.diff(times) < 0.0):
                raise ValueError("rescue control time is non-finite or non-monotonic")

            transition_count = frames - 1
            with h5py.File(temporary, "w", libver="latest") as output:
                data = output.create_group("data")
                episode = data.create_group("demo_0")
                obs_group = episode.create_group("obs")
                next_obs_group = episode.create_group("next_obs")
                for name in OBSERVATION_KEYS:
                    values = observations[name]
                    _write_array(obs_group, name, values[:-1])
                    _write_array(next_obs_group, name, values[1:])
                optional_observation_keys = (
                    _write_optional_aligned_vision(
                        source,
                        obs_group,
                        next_obs_group,
                        times,
                    )
                )

                # Action[i + 1] produced state[i + 1], so pair it with state[i].
                _write_array(episode, "actions", actions[1:])
                _write_array(
                    episode,
                    "rewards",
                    rewards.reshape(frames, -1)[1:, 0],
                )
                dones = (
                    terminated.reshape(frames, -1)[1:, 0]
                    | truncated.reshape(frames, -1)[1:, 0]
                ).astype(np.uint8)
                dones[-1] = 1
                _write_array(episode, "dones", dones)
                _write_array(
                    episode,
                    "success",
                    success.reshape(frames, -1)[1:, 0],
                )
                _write_array(episode, "time_s", times[1:])

                episode.attrs.update(
                    {
                        "num_samples": transition_count,
                        "success": bool(np.any(success[1:] > 0.5)),
                        "scenario_id": scenario_id,
                        "scenario_seed": int(scenario_seed),
                        "procedure_id": procedure_id,
                        "source_control_frames": frames,
                        "post_action_frame_offset": 1,
                        "outcome_authority": OUTCOME_AUTHORITY,
                        "policy_can_write_patient_outcome": False,
                        "reference_eligible": bool(
                            reference_eligible
                        ),
                        "expert_status": str(expert_status),
                    }
                )
                env_args = {
                    "env_name": task,
                    "type": 2,
                    "env_kwargs": {
                        "procedure_id": procedure_id,
                        "scenario_id": scenario_id,
                        "scenario_seed": int(scenario_seed),
                    },
                }
                data.attrs.update(
                    {
                        "total": transition_count,
                        "num_samples": transition_count,
                        "env_args": json.dumps(env_args, sort_keys=True),
                    }
                )
                output.attrs.update(
                    {
                        "schema": SCHEMA,
                        "outcome_authority": OUTCOME_AUTHORITY,
                        "action_contract": json.dumps(
                            dict(action_contract),
                            sort_keys=True,
                        ),
                        "action_semantics": (
                            "recorded Cartesian relative-IK intent; outcomes "
                            "remain environment-owned"
                        ),
                        "transition_alignment": (
                            "obs=post_action_frame[i], "
                            "action=command[i+1], "
                            "next_obs=post_action_frame[i+1]"
                        ),
                        "observation_keys": json.dumps(OBSERVATION_KEYS),
                        "optional_observation_keys": json.dumps(
                            optional_observation_keys
                        ),
                        "vision_alignment": (
                            "latest endoscope frame at or before control "
                            "time; validity, age, and dropout are explicit"
                            if optional_observation_keys
                            else "not recorded"
                        ),
                        "contact_features": json.dumps(CONTACT_FEATURES),
                        "vessel_features": json.dumps(VESSEL_FEATURES),
                        "vital_sign_features": json.dumps(
                            VITAL_SIGN_FEATURES
                        ),
                        "fluid_balance_features": json.dumps(
                            FLUID_BALANCE_FEATURES
                        ),
                        "source_revision": source_revision or "",
                        "research_only": True,
                    }
                )
                output.flush()
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def merge_rescue_training_hdf5(
    sources: Iterable[Path],
    destination: Path,
    *,
    validation_fraction: float = 0.15,
    seed: int = 7777,
    require_reference_eligible: bool = True,
) -> Path:
    """Merge complete episodes and create leakage-free train/valid masks."""

    source_paths = sorted({Path(path).resolve() for path in sources})
    if not source_paths:
        raise ValueError("at least one rescue dataset is required")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    total = 0
    episode_names = []
    reference_attrs: dict[str, Any] | None = None
    env_args: str | None = None
    try:
        with h5py.File(temporary, "w", libver="latest") as output:
            output_data = output.create_group("data")
            for index, path in enumerate(source_paths):
                with h5py.File(path, "r") as source:
                    if source.attrs.get("schema") != SCHEMA:
                        raise ValueError(
                            f"{path} is not a {SCHEMA} dataset"
                        )
                    if "data/demo_0" not in source:
                        raise ValueError(f"{path} has no data/demo_0 episode")
                    source_episode = source["data/demo_0"]
                    if (
                        require_reference_eligible
                        and not bool(
                            source_episode.attrs.get(
                                "reference_eligible",
                                False,
                            )
                        )
                    ):
                        raise ValueError(
                            f"{path} is not an eligible expert reference; "
                            "keep failed or interrupted runs out of behavior "
                            "cloning datasets"
                        )
                    attrs = {
                        key: source.attrs[key]
                        for key in (
                            "schema",
                            "outcome_authority",
                            "action_contract",
                            "observation_keys",
                            "contact_features",
                            "vessel_features",
                            "vital_sign_features",
                            "fluid_balance_features",
                        )
                    }
                    if reference_attrs is None:
                        reference_attrs = attrs
                        env_args = str(source["data"].attrs["env_args"])
                    elif attrs != reference_attrs:
                        raise ValueError(
                            f"{path} uses a different rescue dataset contract"
                        )
                    name = f"demo_{index}"
                    source.copy("data/demo_0", output_data, name)
                    episode_names.append(name)
                    total += int(output_data[name].attrs["num_samples"])

            output_data.attrs.update(
                {
                    "total": total,
                    "num_samples": total,
                    "env_args": env_args or "{}",
                }
            )
            for key, value in (reference_attrs or {}).items():
                output.attrs[key] = value
            output.attrs.update(
                {
                    "schema": SCHEMA,
                    "source_episode_count": len(episode_names),
                    "split_unit": "complete_episode",
                    "split_seed": int(seed),
                }
            )
            order = np.random.default_rng(seed).permutation(
                len(episode_names)
            )
            validation_count = (
                max(1, int(round(len(episode_names) * validation_fraction)))
                if validation_fraction > 0.0 and len(episode_names) > 1
                else 0
            )
            valid_indices = set(order[:validation_count].tolist())
            train_names = [
                name
                for index, name in enumerate(episode_names)
                if index not in valid_indices
            ]
            valid_names = [
                name
                for index, name in enumerate(episode_names)
                if index in valid_indices
            ]
            mask = output.create_group("mask")
            string_dtype = h5py.string_dtype(encoding="utf-8")
            mask.create_dataset(
                "train",
                data=np.asarray(train_names, dtype=string_dtype),
            )
            mask.create_dataset(
                "valid",
                data=np.asarray(valid_names, dtype=string_dtype),
            )
            output.flush()
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination
