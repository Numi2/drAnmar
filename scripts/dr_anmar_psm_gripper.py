# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Apply Dr.Anmar's one NVIDIA/ORBIT PSM foundation profile.

The values live in ``orbit.surgical.assets/config/psm_foundation.json``.
There are deliberately no per-room, command-line, or environment overrides.
This module only configures NVIDIA/Isaac Lab objects; PhysX remains responsible
for contact and object retention.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, replace
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

NVIDIA_ORBIT_GRIPPER_TERM_MAP = (
    ("gripper_action", "robot_1_gripper_action", "robot"),
    ("gripper_1_action", "robot_1_gripper_action", "robot_1"),
    ("gripper_2_action", "robot_2_gripper_action", "robot_2"),
    ("robot_1_gripper_action", "robot_1_gripper_action", "robot_1"),
    ("robot_2_gripper_action", "robot_2_gripper_action", "robot_2"),
)


def resolve_psm_gripper_profile(
    *,
    open_rad: float | None = None,
    close_rad: float | None = None,
) -> PsmGripperProfile:
    """Resolve one safe runtime profile without changing NVIDIA's action layout."""

    profile = replace(
        CANONICAL_PSM_GRIPPER_PROFILE,
        open_rad=(
            CANONICAL_PSM_GRIPPER_PROFILE.open_rad
            if open_rad is None
            else float(open_rad)
        ),
        close_rad=(
            CANONICAL_PSM_GRIPPER_PROFILE.close_rad
            if close_rad is None
            else float(close_rad)
        ),
    )
    if not 0.10 <= profile.open_rad <= 0.60:
        raise ValueError("PSM open target must be between 0.10 and 0.60 radians")
    if not 0.00 <= profile.close_rad <= 0.15:
        raise ValueError("PSM closed target must be between 0.00 and 0.15 radians")
    if profile.close_rad >= profile.open_rad:
        raise ValueError("PSM closed target must be smaller than the open target")
    return profile


def complete_psm_actions_from_nvidia_orbit(
    actions: Any,
    scene: Any,
    reference_actions: Any,
) -> list[str]:
    """Fill optional PSM jaw slots from ORBIT's working handover config.

    Reach environments intentionally leave their gripper slots empty.  Rather
    than recreating those actions per room, copy the corresponding native
    action term from the released ORBIT needle-handover configuration and only
    retarget its articulation name.  Existing task-owned terms are preserved.
    """

    active: list[str] = []
    for target_name, source_name, asset_name in NVIDIA_ORBIT_GRIPPER_TERM_MAP:
        if not hasattr(actions, target_name) or getattr(scene, asset_name, None) is None:
            continue
        term = getattr(actions, target_name)
        if term is None:
            source = getattr(reference_actions, source_name, None)
            if source is None:
                raise RuntimeError(
                    f"NVIDIA/ORBIT PSM reference is missing {source_name}"
                )
            term = copy.deepcopy(source)
            term.asset_name = asset_name
            setattr(actions, target_name, term)
        active.append(target_name)
    return active


def psm_articulation_names(scene: Any) -> list[str]:
    """Return PSM articulation slots without changing their native config."""

    names: list[str] = []
    for name in ("robot", "robot_1", "robot_2"):
        robot_cfg = getattr(scene, name, None)
        actuators = getattr(robot_cfg, "actuators", None)
        if isinstance(actuators, dict) and "psm" in actuators and "psm_tool" in actuators:
            names.append(name)
    return names


def psm_gripper_command_expr(aperture_rad: float) -> dict[str, float]:
    """Build the symmetric physical targets used by NVIDIA's binary action."""

    value = float(aperture_rad)
    return {
        "psm_tool_gripper1_joint": -value,
        "psm_tool_gripper2_joint": value,
    }


def apply_psm_gripper_action_profile(
    actions: Any,
    profile: PsmGripperProfile | None = None,
) -> list[str]:
    """Apply identical open/close commands to every recognized PSM action term."""

    profile = profile or CANONICAL_PSM_GRIPPER_PROFILE
    applied: list[str] = []
    for term_name in PSM_GRIPPER_ACTION_TERMS:
        term = getattr(actions, term_name, None)
        if term is None or not hasattr(term, "close_command_expr"):
            continue
        term.open_command_expr = psm_gripper_command_expr(profile.open_rad)
        term.close_command_expr = psm_gripper_command_expr(profile.close_rad)
        applied.append(term_name)
    return applied


def apply_psm_gripper_articulation_profile(
    robot_cfg: Any,
    profile: PsmGripperProfile | None = None,
) -> bool:
    """Apply the same jaw actuator and reset posture to one PSM articulation."""

    actuators = getattr(robot_cfg, "actuators", None)
    init_state = getattr(robot_cfg, "init_state", None)
    joint_pos = getattr(init_state, "joint_pos", None)
    if not isinstance(actuators, dict) or "psm_tool" not in actuators or not isinstance(joint_pos, dict):
        return False

    profile = profile or CANONICAL_PSM_GRIPPER_PROFILE
    joint_pos.update(psm_gripper_command_expr(profile.open_rad))
    actuator = actuators["psm_tool"]
    actuator.effort_limit_sim = profile.effort_limit_nm
    actuator.velocity_limit_sim = profile.velocity_limit_rad_s
    actuator.stiffness = profile.stiffness
    actuator.damping = profile.damping
    return True


def psm_gripper_profile_manifest(
    *,
    profile: PsmGripperProfile | None = None,
    action_terms: list[str] | None = None,
    articulations: list[str] | None = None,
) -> dict[str, Any]:
    """Return an API-safe record of the physical configuration in force."""

    manifest = asdict(profile or CANONICAL_PSM_GRIPPER_PROFILE)
    manifest["action_terms"] = list(action_terms or [])
    manifest["articulations"] = list(articulations or [])
    return manifest
