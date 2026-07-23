# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Apply Dr.Anmar's one NVIDIA/ORBIT PSM foundation profile.

The values live in ``orbit.surgical.assets/config/psm_foundation.json``.
There are deliberately no per-room, command-line, or environment overrides.
This module only configures NVIDIA/Isaac Lab objects; PhysX remains responsible
for contact and object retention.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PsmGripperProfile:
    id: str
    open_rad: float
    close_rad: float
    effort_limit_nm: float
    velocity_limit_rad_s: float
    stiffness: float
    damping: float
    camera_update_period_s: float
    camera_width_px: int
    camera_height_px: int
    camera_focal_length_mm: float
    camera_backoff_m: float
    camera_lateral_offset_m: float
    camera_lookahead_m: float


def _load_canonical_profile() -> PsmGripperProfile:
    profile_path = (
        Path(__file__).resolve().parents[1]
        / "source/extensions/orbit.surgical.assets/config/psm_foundation.json"
    )
    with profile_path.open(encoding="utf-8") as profile_file:
        raw = json.load(profile_file)
    gripper = raw["gripper"]
    cameras = raw["tool_cameras"]
    return PsmGripperProfile(
        id=str(raw["profile_id"]),
        open_rad=float(gripper["open_rad"]),
        close_rad=float(gripper["close_rad"]),
        effort_limit_nm=float(gripper["effort_limit_nm"]),
        velocity_limit_rad_s=float(gripper["velocity_limit_rad_s"]),
        stiffness=float(gripper["stiffness"]),
        damping=float(gripper["damping"]),
        camera_update_period_s=float(cameras["update_period_s"]),
        camera_width_px=int(cameras["width_px"]),
        camera_height_px=int(cameras["height_px"]),
        camera_focal_length_mm=float(cameras["focal_length_mm"]),
        camera_backoff_m=float(cameras["backoff_m"]),
        camera_lateral_offset_m=float(cameras["lateral_offset_m"]),
        camera_lookahead_m=float(cameras["lookahead_m"]),
    )


CANONICAL_PSM_GRIPPER_PROFILE = _load_canonical_profile()
PSM_GRIPPER_ACTION_TERMS = (
    "gripper_action",
    "gripper_1_action",
    "gripper_2_action",
    "robot_1_gripper_action",
    "robot_2_gripper_action",
    "finger_joint_pos",
    "finger_joint_pos_2",
)


def psm_gripper_command_expr(aperture_rad: float) -> dict[str, float]:
    """Build the symmetric physical targets used by NVIDIA's binary action."""

    value = float(aperture_rad)
    return {
        "psm_tool_gripper1_joint": -value,
        "psm_tool_gripper2_joint": value,
    }


def apply_psm_gripper_action_profile(actions: Any) -> list[str]:
    """Apply identical open/close commands to every recognized PSM action term."""

    profile = CANONICAL_PSM_GRIPPER_PROFILE
    applied: list[str] = []
    for term_name in PSM_GRIPPER_ACTION_TERMS:
        term = getattr(actions, term_name, None)
        if term is None or not hasattr(term, "close_command_expr"):
            continue
        term.open_command_expr = psm_gripper_command_expr(profile.open_rad)
        term.close_command_expr = psm_gripper_command_expr(profile.close_rad)
        applied.append(term_name)
    return applied


def apply_psm_gripper_articulation_profile(robot_cfg: Any) -> bool:
    """Apply the same jaw actuator and reset posture to one PSM articulation."""

    actuators = getattr(robot_cfg, "actuators", None)
    init_state = getattr(robot_cfg, "init_state", None)
    joint_pos = getattr(init_state, "joint_pos", None)
    if not isinstance(actuators, dict) or "psm_tool" not in actuators or not isinstance(joint_pos, dict):
        return False

    profile = CANONICAL_PSM_GRIPPER_PROFILE
    joint_pos.update(psm_gripper_command_expr(profile.open_rad))
    actuator = actuators["psm_tool"]
    actuator.effort_limit_sim = profile.effort_limit_nm
    actuator.velocity_limit_sim = profile.velocity_limit_rad_s
    actuator.stiffness = profile.stiffness
    actuator.damping = profile.damping
    return True


def psm_gripper_profile_manifest(
    *,
    action_terms: list[str] | None = None,
    articulations: list[str] | None = None,
) -> dict[str, Any]:
    """Return an API-safe record of the physical configuration in force."""

    manifest = asdict(CANONICAL_PSM_GRIPPER_PROFILE)
    manifest["action_terms"] = list(action_terms or [])
    manifest["articulations"] = list(articulations or [])
    return manifest
