# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Reset the physical PSM/needle custody and entry evidence state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from orbit.surgical.assets.psm import PSM_GRIPPER_PROFILE
from orbit.surgical.tasks.surgical import mdp_common

from ..contract import PunctureGateState

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


NEEDLE_MID_GRASP_POSITION_M = (0.00613661575091, 0.00337363305778, 0.0)
# This Isaac Lab build exposes tensors in XYZW, while OpenUSD authors quatf in
# WXYZ.  Keep the calibrated jaw-seat rotation in tensor convention and
# convert only at the USD boundary below.
NEEDLE_MID_GRASP_QUAT_XYZW = (0.50904141575, 0.860742027004, 0.0, 0.0)
# Authored entry-policy presentation frame.
NEEDLE_POLICY_GRASP_QUAT_XYZW = (0.50904141575, 0.860742027004, 0.0, 0.0)
PSM_TOOL_TIP_TO_JAW_COLLISION_M = (0.0, 0.0, 0.0)
PENETRATION_GRIPPER_CLOSE_RAD = float(PSM_GRIPPER_PROFILE["close_rad"])
PENETRATION_GRIPPER_OPEN_RAD = float(PSM_GRIPPER_PROFILE["open_rad"])
def seat_pregrasped_needle(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Place the needle from current link kinematics."""

    robot: Articulation = env.scene["robot"]
    needle: RigidObject = env.scene["needle"]
    body_ids, _ = robot.find_bodies("psm_tool_tip_link")
    ee_pos = mdp_common.as_torch(robot.data.body_pos_w)[env_ids, body_ids[0], :]
    ee_quat = mdp_common.as_torch(robot.data.body_quat_w)[env_ids, body_ids[0], :]
    local_quat = torch.tensor(
        NEEDLE_MID_GRASP_QUAT_XYZW, device=env.device, dtype=ee_quat.dtype
    ).repeat(len(env_ids), 1)
    needle_quat = quat_mul(ee_quat, quat_conjugate(local_quat))
    jaw_offset = torch.tensor(
        PSM_TOOL_TIP_TO_JAW_COLLISION_M, device=env.device, dtype=ee_pos.dtype
    ).repeat(len(env_ids), 1)
    # The authored preferred closure-needle grasp is centered on this PSM
    # asset's tool-tip frame; the older pickup mesh used a different offset.
    grasp_target = ee_pos + quat_apply(ee_quat, jaw_offset)
    local_position = torch.tensor(
        NEEDLE_MID_GRASP_POSITION_M, device=env.device, dtype=ee_pos.dtype
    ).repeat(len(env_ids), 1)
    needle_pos = grasp_target - quat_apply(needle_quat, local_position)
    needle.write_root_pose_to_sim_index(
        root_pose=torch.cat((needle_pos, needle_quat), dim=-1), env_ids=env_ids
    )
    needle.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids
    )


def attach_pregrasped_needle(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Close the jaws for the authored pregrasp presentation."""

    robot: Articulation = env.scene["robot"]
    joint_ids, _ = robot.find_joints("psm_tool_gripper.*_joint")
    close = PENETRATION_GRIPPER_CLOSE_RAD
    jaw_position = torch.tensor((-close, close), device=env.device).repeat(len(env_ids), 1)
    robot.set_joint_position_target_index(
        target=jaw_position, joint_ids=joint_ids, env_ids=env_ids
    )


def reset_pregrasped_needle(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> None:
    """Open the jaws; authored pregrasp seating follows fresh kinematics."""

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    robot: Articulation = env.scene["robot"]
    joint_ids, _ = robot.find_joints("psm_tool_gripper.*_joint")
    opened = PENETRATION_GRIPPER_OPEN_RAD
    jaw_position = torch.tensor((-opened, opened), device=env.device).repeat(len(env_ids), 1)
    jaw_velocity = torch.zeros_like(jaw_position)
    robot.write_joint_state_to_sim(
        jaw_position, jaw_velocity, joint_ids=joint_ids, env_ids=env_ids
    )
    robot.set_joint_position_target_index(
        target=jaw_position, joint_ids=joint_ids, env_ids=env_ids
    )


def reset_penetration_evidence(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    *,
    fixed_domain: bool = False,
) -> None:
    """Clear monotonic evidence and sample the documented tissue ranges."""

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    state = getattr(env, "_dr_anmar_penetration_state", None)
    if state is None:
        state = {
            "gates": [PunctureGateState() for _ in range(env.num_envs)],
            "puncture_force_n": torch.full((env.num_envs,), 2.0, device=env.device),
            "shaft_drag_n_m": torch.full((env.num_envs,), 25.0, device=env.device),
            "wetness": torch.full((env.num_envs,), 0.45, device=env.device),
            "material_scale": torch.ones(env.num_envs, device=env.device),
            "settle_control_steps": torch.full(
                (env.num_envs,), 2, dtype=torch.long, device=env.device
            ),
            "grasp_loss_steps": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "bilateral_seen": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "grasp_attach_stage": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "grasp_local_position_m": torch.tensor(
                NEEDLE_MID_GRASP_POSITION_M, device=env.device
            ).repeat(env.num_envs, 1),
            "grasp_local_quaternion_xyzw": torch.tensor(
                NEEDLE_MID_GRASP_QUAT_XYZW, device=env.device
            ).repeat(env.num_envs, 1),
            "last_update_step": -1,
        }
        env._dr_anmar_penetration_state = state
    count = len(env_ids)
    if fixed_domain:
        state["puncture_force_n"][env_ids] = 2.0
        state["shaft_drag_n_m"][env_ids] = 25.0
        state["wetness"][env_ids] = 0.45
    else:
        state["puncture_force_n"][env_ids] = 0.35 + 2.85 * torch.rand(count, device=env.device)
        state["shaft_drag_n_m"][env_ids] = 8.0 + 47.0 * torch.rand(count, device=env.device)
        state["wetness"][env_ids] = 0.1 + 0.8 * torch.rand(count, device=env.device)
    state["material_scale"][env_ids] = 1.0
    # The first observation is evaluated immediately after reset.  Two cached
    # evaluations admit one complete 20 ms interval (ten 2 ms PhysX steps)
    # after the authored pregrasp pose is consumed and before policy authority.
    state["settle_control_steps"][env_ids] = 2
    state["grasp_loss_steps"][env_ids] = 0
    state["bilateral_seen"][env_ids] = False
    state["grasp_attach_stage"][env_ids] = 0
    state["grasp_local_position_m"][env_ids] = torch.tensor(
        NEEDLE_MID_GRASP_POSITION_M, device=env.device
    )
    state["grasp_local_quaternion_xyzw"][env_ids] = torch.tensor(
        NEEDLE_MID_GRASP_QUAT_XYZW, device=env.device
    )
    for index in env_ids.detach().cpu().tolist():
        state["gates"][index] = PunctureGateState()
    state.pop("measurement", None)
    state.pop("wrench", None)
    state.pop("force_derivative", None)
    state.pop("force_integral", None)
    state.pop("previous_entry_error", None)
    state["last_update_step"] = -1
