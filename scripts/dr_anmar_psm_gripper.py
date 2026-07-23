# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""One physical gripper profile for every Dr.Anmar NVIDIA/ORBIT PSM path.

The values match the working ORBIT-Surgical needle lift and handover rooms.
This module only configures NVIDIA/Isaac Lab articulation and binary-action
objects; PhysX remains responsible for contact and object retention.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PsmGripperProfile:
    id: str = "nvidia-orbit-needle-v1"
    open_rad: float = 0.5
    close_rad: float = 0.02
    effort_limit_nm: float = 0.1
    velocity_limit_rad_s: float = 0.2
    stiffness: float = 500.0
    damping: float = 0.1


CANONICAL_PSM_GRIPPER_PROFILE = PsmGripperProfile()
PSM_GRIPPER_ACTION_TERMS = (
    "gripper_action",
    "gripper_1_action",
    "gripper_2_action",
    "robot_1_gripper_action",
    "robot_2_gripper_action",
    "finger_joint_pos",
    "finger_joint_pos_2",
)


def psm_gripper_close_rad_from_environment() -> float:
    """Return one global close target, rejecting unsafe per-room drift."""

    raw_value = os.environ.get(
        "DR_ANMAR_PSM_GRIPPER_CLOSE_RAD",
        str(CANONICAL_PSM_GRIPPER_PROFILE.close_rad),
    )
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = CANONICAL_PSM_GRIPPER_PROFILE.close_rad
    if not 0.0 <= value < CANONICAL_PSM_GRIPPER_PROFILE.open_rad:
        raise ValueError(
            "DR_ANMAR_PSM_GRIPPER_CLOSE_RAD must be in "
            f"[0.0, {CANONICAL_PSM_GRIPPER_PROFILE.open_rad}), got {value}"
        )
    return value


def psm_gripper_command_expr(aperture_rad: float) -> dict[str, float]:
    """Build the symmetric physical targets used by NVIDIA's binary action."""

    value = float(aperture_rad)
    return {
        "psm_tool_gripper1_joint": -value,
        "psm_tool_gripper2_joint": value,
    }


def apply_psm_gripper_action_profile(actions: Any, close_rad: float) -> list[str]:
    """Apply identical open/close commands to every recognized PSM action term."""

    applied: list[str] = []
    for term_name in PSM_GRIPPER_ACTION_TERMS:
        term = getattr(actions, term_name, None)
        if term is None or not hasattr(term, "close_command_expr"):
            continue
        term.open_command_expr = psm_gripper_command_expr(
            CANONICAL_PSM_GRIPPER_PROFILE.open_rad
        )
        term.close_command_expr = psm_gripper_command_expr(close_rad)
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
    close_rad: float,
    *,
    action_terms: list[str] | None = None,
    articulations: list[str] | None = None,
) -> dict[str, Any]:
    """Return an API-safe record of the physical configuration in force."""

    manifest = asdict(CANONICAL_PSM_GRIPPER_PROFILE)
    manifest["close_rad"] = float(close_rad)
    manifest["action_terms"] = list(action_terms or [])
    manifest["articulations"] = list(articulations or [])
    return manifest
