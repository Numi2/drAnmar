from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dr_anmar_telemetry import (  # noqa: E402
    DEMONSTRATION_SCHEMA,
    build_array_contract,
    validate_demonstration,
)


def _write_fixture(tmp_path: Path, *, invalid_quaternion: bool = False) -> Path:
    control_frames = 3
    vision_frames = 2
    quaternion = [2.0, 0.0, 0.0, 0.0] if invalid_quaternion else [1.0, 0.0, 0.0, 0.0]
    arrays = {
        "time_s": np.asarray([0.0, 0.02, 0.04], dtype=np.float64),
        "actions": np.zeros((control_frames, 7), dtype=np.float32),
        "environment_reward": np.zeros(control_frames, dtype=np.float32),
        "environment_terminated": np.zeros(control_frames, dtype=np.bool_),
        "environment_truncated": np.zeros(control_frames, dtype=np.bool_),
        "environment_success": np.zeros(control_frames, dtype=np.float32),
        "anatomy_showcase_position_w": np.zeros((control_frames, 3), dtype=np.float32),
        "anatomy_showcase_quaternion_w": np.tile(
            np.asarray(quaternion, dtype=np.float32),
            (control_frames, 1),
        ),
        "psm_joint_positions": np.zeros((control_frames, 2), dtype=np.float32),
        "psm_joint_velocities": np.zeros((control_frames, 2), dtype=np.float32),
        "psm_applied_joint_torque": np.zeros((control_frames, 2), dtype=np.float32),
        "endoscope_time_s": np.asarray([0.01, 0.03], dtype=np.float64),
        "endoscope_rgb": np.zeros((vision_frames, 4, 6, 3), dtype=np.uint8),
        "endoscope_depth_m": np.ones((vision_frames, 4, 6), dtype=np.float32),
        "endoscope_camera_position_w": np.zeros((vision_frames, 3), dtype=np.float32),
        "endoscope_camera_quaternion_w": np.tile(
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            (vision_frames, 1),
        ),
    }
    data_path = tmp_path / "episode.npz"
    np.savez_compressed(data_path, **arrays)
    descriptors = {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "stream": "vision" if key.startswith("endoscope_") else "control",
        }
        for key, value in arrays.items()
    }
    manifest = {
        "schema": DEMONSTRATION_SCHEMA,
        "simulation_only": True,
        "data_file": data_path.name,
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "frames": control_frames,
        "vision_frames": vision_frames,
        "arrays": {key: list(value.shape) for key, value in arrays.items()},
        "array_contract": build_array_contract(
            descriptors,
            control_frames=control_frames,
            vision_frames=vision_frames,
        ),
        "robot_joint_names": {"psm": ["yaw", "insertion"]},
        "robot_joint_units": {
            "psm": {
                "coordinate": ["rad", "m"],
                "velocity": ["rad/s", "m/s"],
                "effort": ["N*m", "N"],
            }
        },
        "modalities": {
            "camera_intrinsics": [
                [100.0, 0.0, 3.0],
                [0.0, 100.0, 2.0],
                [0.0, 0.0, 1.0],
            ]
        },
    }
    manifest_path = tmp_path / "episode.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def test_demonstration_contract_validates_hash_alignment_units_and_frames(tmp_path: Path) -> None:
    result = validate_demonstration(_write_fixture(tmp_path))
    assert result["passed"], result["issues"]


def test_demonstration_contract_rejects_non_unit_world_quaternion(tmp_path: Path) -> None:
    result = validate_demonstration(
        _write_fixture(tmp_path, invalid_quaternion=True)
    )
    assert not result["passed"]
    assert any("non-unit quaternions" in issue for issue in result["issues"])


def test_demonstration_contract_rejects_content_hash_drift(tmp_path: Path) -> None:
    manifest_path = _write_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["data_sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    result = validate_demonstration(manifest_path)
    assert not result["passed"]
    assert "data SHA-256 does not match the manifest" in result["issues"]
