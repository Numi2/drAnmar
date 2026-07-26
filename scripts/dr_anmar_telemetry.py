# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Contracts and offline validation for Dr.Anmar demonstration telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


DEMONSTRATION_SCHEMA = "dr.anmar.demonstration.v3"
CONTROL_STREAM = "control"
VISION_STREAM = "endoscope"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _units_and_frame(key: str) -> tuple[str, str]:
    if key in {"time_s", "endoscope_time_s"}:
        return "s", "monotonic_recording_clock"
    if key == "actions":
        return "action_contract_units", "controller"
    if key == "cartesian_actions":
        return "normalized_translation_rotation_gripper", "controller"
    if key == "resolved_joint_targets":
        return "per_joint_coordinate_units", "articulation"
    if key.endswith("_joint_positions") or key.endswith("_joint_velocities"):
        suffix = "per_joint_coordinate_units" if key.endswith("_joint_positions") else "per_joint_coordinate_units_per_s"
        return suffix, "articulation"
    if key.endswith("_joint_torque"):
        return "per_joint_effort_units", "articulation"
    if key.endswith("_body_positions_w") or key.endswith("_position_w") or key.endswith("_centroid_w"):
        return "m", "world"
    if key.endswith("_body_quaternions_w") or key.endswith("_quaternion_w"):
        return "unit_quaternion_wxyz", "world"
    if key.endswith("_root_pose_w") or key.endswith("_pose_w"):
        return "m_then_unit_quaternion_wxyz", "world"
    if key.endswith("_root_velocity_w"):
        return "m_per_s_then_rad_per_s", "world"
    if key.endswith("_force_n") or key.endswith("_contact_force"):
        return "N", "world"
    if key.endswith("_torque_nm"):
        return "N_m", "world"
    if key.endswith("_stress_pa"):
        return "Pa", "simulation_mesh"
    if key.endswith("_depth_m"):
        return "m", "camera_optical"
    if key.endswith("_point_cloud_camera_m"):
        return "m", "camera_optical"
    if key.endswith("_camera_position_w"):
        return "m", "world"
    if key.endswith("_camera_quaternion_w"):
        return "unit_quaternion_wxyz", "world"
    if key.endswith("_rgb"):
        return "uint8_srgb", "camera_pixel"
    if key.endswith("_semantic_id"):
        return "uint32_semantic_id", "camera_pixel"
    if key.endswith("_m") or "_distance_m" in key or "_displacement_m" in key:
        return "m", "declared_by_field"
    if key.endswith("_s"):
        return "s", "monotonic_recording_clock"
    if "fraction" in key or key.endswith("_valid") or key.endswith("_active"):
        return "unitless", "declared_by_field"
    return "dimensionless_or_categorical", "field_semantics"


def build_array_contract(
    arrays: dict[str, dict[str, Any]],
    *,
    control_frames: int,
    vision_frames: int,
) -> dict[str, dict[str, Any]]:
    """Describe every retained array without guessing authority at read time."""

    contract: dict[str, dict[str, Any]] = {}
    for key, descriptor in sorted(arrays.items()):
        shape = [int(value) for value in descriptor["shape"]]
        first_dimension = shape[0] if shape else 0
        declared_stream = str(descriptor.get("stream", ""))
        if declared_stream == "vision":
            stream = VISION_STREAM
            expected_frames = vision_frames
        else:
            stream = CONTROL_STREAM
            expected_frames = control_frames
        if first_dimension != expected_frames:
            raise ValueError(
                f"{key} has {first_dimension} rows but the {stream} stream has "
                f"{expected_frames} frames"
            )
        units, coordinate_frame = _units_and_frame(key)
        contract[key] = {
            "shape": shape,
            "dtype": str(descriptor["dtype"]),
            "stream": stream,
            "units": units,
            "coordinate_frame": coordinate_frame,
            "authority": (
                "native_simulator_post_step"
                if stream == CONTROL_STREAM
                else "native_sensor_capture"
            ),
        }
    return contract


def _require(condition: bool, message: str, issues: list[str]) -> None:
    if not condition:
        issues.append(message)


def _finite_or_declared_nan(key: str, values: np.ndarray) -> bool:
    if np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.bool_):
        return True
    if key in {
        "tool_to_object_distance_m",
        "camera_valid_depth_fraction",
        "camera_semantic_foreground_fraction",
        "camera_mean_luminance",
        "dr_anmar_suture_minimum_break_force_n",
        "dr_anmar_suture_maximum_observed_strain",
    }:
        return not np.isinf(values).any()
    return bool(np.isfinite(values).all())


def _validate_quaternions(key: str, values: np.ndarray, issues: list[str]) -> None:
    if key.endswith("_root_pose_w") or key.endswith("_pose_w"):
        quaternions = values[..., -4:]
    elif key.endswith("_quaternion_w") or key.endswith("_quaternions_w"):
        quaternions = values
    else:
        return
    if quaternions.size == 0 or quaternions.shape[-1] != 4:
        issues.append(f"{key} does not contain four-component quaternions")
        return
    norms = np.linalg.norm(quaternions.astype(np.float64), axis=-1)
    if not np.isfinite(norms).all() or np.any(np.abs(norms - 1.0) > 1.0e-3):
        issues.append(f"{key} contains non-unit quaternions")


def validate_demonstration(manifest_path: Path) -> dict[str, Any]:
    """Validate one manifest and its content-addressed NPZ without pickle."""

    manifest_path = manifest_path.resolve()
    issues: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"passed": False, "issues": [f"manifest is unreadable: {exc}"]}

    _require(manifest.get("schema") == DEMONSTRATION_SCHEMA, "demonstration schema is not current", issues)
    data_name = str(manifest.get("data_file", ""))
    data_path = (manifest_path.parent / data_name).resolve()
    _require(
        bool(data_name) and data_path.parent == manifest_path.parent,
        "data_file must be a sibling filename",
        issues,
    )
    if not data_path.is_file():
        issues.append("data_file does not exist")
        return {"passed": False, "issues": issues}
    _require(
        sha256_file(data_path) == manifest.get("data_sha256"),
        "data SHA-256 does not match the manifest",
        issues,
    )

    declared_arrays = manifest.get("arrays", {})
    array_contract = manifest.get("array_contract", {})
    _require(isinstance(declared_arrays, dict), "arrays must be an object", issues)
    _require(isinstance(array_contract, dict), "array_contract must be an object", issues)
    required = {
        "time_s",
        "actions",
        "environment_reward",
        "environment_terminated",
        "environment_truncated",
        "environment_success",
        "anatomy_showcase_position_w",
        "anatomy_showcase_quaternion_w",
    }
    try:
        with np.load(data_path, allow_pickle=False) as data:
            keys = set(data.files)
            _require(required.issubset(keys), f"required arrays missing: {sorted(required - keys)}", issues)
            _require(keys == set(declared_arrays), "manifest array inventory does not match NPZ members", issues)
            _require(keys == set(array_contract), "array contract inventory does not match NPZ members", issues)
            control_frames = int(manifest.get("frames", -1))
            vision_frames = int(manifest.get("vision_frames", -1))
            for key in sorted(keys):
                values = np.asarray(data[key])
                shape = list(values.shape)
                descriptor = array_contract.get(key, {})
                _require(shape == declared_arrays.get(key), f"{key} shape differs from arrays manifest", issues)
                _require(shape == descriptor.get("shape"), f"{key} shape differs from array contract", issues)
                _require(str(values.dtype) == descriptor.get("dtype"), f"{key} dtype differs from array contract", issues)
                expected_rows = (
                    vision_frames
                    if descriptor.get("stream") == VISION_STREAM
                    else control_frames
                )
                _require(bool(shape) and shape[0] == expected_rows, f"{key} is not frame-aligned", issues)
                _require(
                    descriptor.get("units") not in {None, "", "declared_by_field"},
                    f"{key} has no explicit units",
                    issues,
                )
                _require(_finite_or_declared_nan(key, values), f"{key} contains invalid infinity or NaN", issues)
                _validate_quaternions(key, values, issues)

            control_time = np.asarray(data["time_s"], dtype=np.float64).reshape(-1)
            _require(
                bool(
                    len(control_time)
                    and np.isfinite(control_time).all()
                    and np.all(np.diff(control_time) > 0)
                ),
                "control timestamps must be finite and strictly increasing",
                issues,
            )
            if "endoscope_time_s" in keys:
                vision_time = np.asarray(data["endoscope_time_s"], dtype=np.float64).reshape(-1)
                _require(
                    bool(
                        len(vision_time)
                        and np.isfinite(vision_time).all()
                        and np.all(np.diff(vision_time) > 0)
                    ),
                    "vision timestamps must be finite and strictly increasing",
                    issues,
                )
                if len(control_time) and len(vision_time):
                    _require(
                        vision_time[0] >= control_time[0] - 0.25
                        and vision_time[-1] <= control_time[-1] + 0.25,
                        "vision timestamps fall outside the control clock interval",
                        issues,
                    )
            if "endoscope_depth_m" in keys:
                depth = np.asarray(data["endoscope_depth_m"])
                _require(
                    bool(np.isfinite(depth).all() and np.all(depth >= 0.0)),
                    "depth must be finite non-negative metres",
                    issues,
                )
            if "endoscope_rgb" in keys:
                rgb = np.asarray(data["endoscope_rgb"])
                _require(rgb.dtype == np.uint8 and rgb.ndim == 4 and rgb.shape[-1] in {3, 4}, "RGB must be uint8 HxWx3/4", issues)
    except Exception as exc:
        issues.append(f"data file is unreadable: {exc}")

    joint_names = manifest.get("robot_joint_names", {})
    joint_units = manifest.get("robot_joint_units", {})
    _require(isinstance(joint_names, dict) and bool(joint_names), "robot joint ordering is missing", issues)
    _require(set(joint_names) == set(joint_units), "robot joint unit inventory is inconsistent", issues)
    for robot_name, names in joint_names.items() if isinstance(joint_names, dict) else []:
        units = joint_units.get(robot_name, {})
        _require(
            len(names)
            == len(units.get("coordinate", []))
            == len(units.get("velocity", []))
            == len(units.get("effort", [])),
            f"{robot_name} joint names and units are not aligned",
            issues,
        )
        for suffix in (
            "joint_positions",
            "joint_velocities",
            "applied_joint_torque",
            "computed_joint_torque",
        ):
            key = f"{robot_name}_{suffix}"
            shape = declared_arrays.get(key)
            if shape is not None:
                _require(
                    bool(shape) and int(shape[-1]) == len(names),
                    f"{key} does not match the declared joint ordering",
                    issues,
                )
    intrinsics = np.asarray(manifest.get("modalities", {}).get("camera_intrinsics", []), dtype=np.float64)
    if int(manifest.get("vision_frames", 0)):
        _require(
            bool(
                intrinsics.shape == (3, 3)
                and np.isfinite(intrinsics).all()
                and intrinsics[0, 0] > 0
                and intrinsics[1, 1] > 0
            ),
            "camera intrinsics are missing or invalid",
            issues,
        )

    return {
        "schema": "dr.anmar.demonstration-validation.v1",
        "manifest": manifest_path.name,
        "data_file": data_path.name,
        "frames": int(manifest.get("frames", 0)),
        "vision_frames": int(manifest.get("vision_frames", 0)),
        "issues": issues,
        "passed": not issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    arguments = parser.parse_args()
    result = validate_demonstration(arguments.manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
