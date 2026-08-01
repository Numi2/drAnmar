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


NEEDLE_MID_GRASP_POSITION_M = (0.0070028174960433945, 0.0, 0.0)
NEEDLE_MID_GRASP_QUAT_XYZW = (0.0, 0.7071067811865475, 0.7071067811865475, 0.0)
PSM_TOOL_TIP_TO_JAW_COLLISION_M = (0.0, 0.0, 0.0)


def reset_pregrasped_needle(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> None:
    """Seat the authored driver frame in the tool tip and close both jaws."""

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    robot: Articulation = env.scene["robot"]
    needle: RigidObject = env.scene["needle"]
    body_ids, _ = robot.find_bodies("psm_tool_tip_link")
    ee_pos = mdp_common.as_torch(robot.data.body_pos_w)[env_ids, body_ids[0], :]
    ee_quat = mdp_common.as_torch(robot.data.body_quat_w)[env_ids, body_ids[0], :]
    local_quat = torch.tensor(
        NEEDLE_MID_GRASP_QUAT_XYZW, device=env.device, dtype=ee_quat.dtype
    ).repeat(len(env_ids), 1)
    needle_quat = quat_mul(ee_quat, quat_conjugate(local_quat))
    jaw_collision_offset = torch.tensor(
        PSM_TOOL_TIP_TO_JAW_COLLISION_M, device=env.device, dtype=ee_pos.dtype
    ).repeat(len(env_ids), 1)
    grasp_target = ee_pos + quat_apply(ee_quat, jaw_collision_offset)
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

    joint_ids, _ = robot.find_joints("psm_tool_gripper.*_joint")
    close = float(PSM_GRIPPER_PROFILE["close_rad"])
    jaw_position = torch.tensor((-close, close), device=env.device).repeat(len(env_ids), 1)
    jaw_velocity = torch.zeros_like(jaw_position)
    robot.write_joint_state_to_sim(
        jaw_position, jaw_velocity, joint_ids=joint_ids, env_ids=env_ids
    )
    robot.set_joint_position_target_index(
        target=jaw_position, joint_ids=joint_ids, env_ids=env_ids
    )


def reset_penetration_evidence(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> None:
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
            "last_update_step": -1,
        }
        env._dr_anmar_penetration_state = state
    count = len(env_ids)
    state["puncture_force_n"][env_ids] = 0.35 + 2.85 * torch.rand(count, device=env.device)
    state["shaft_drag_n_m"][env_ids] = 8.0 + 47.0 * torch.rand(count, device=env.device)
    state["wetness"][env_ids] = 0.1 + 0.8 * torch.rand(count, device=env.device)
    state["material_scale"][env_ids] = 1.0
    # The first state observation is evaluated immediately after reset.  Two
    # cached control evaluations therefore admit exactly one 20 ms interval,
    # i.e. ten 2 ms PhysX settle steps, before custody is authoritative.
    state["settle_control_steps"][env_ids] = 2
    for index in env_ids.detach().cpu().tolist():
        state["gates"][index] = PunctureGateState()
    state.pop("measurement", None)
    state.pop("wrench", None)
    state.pop("force_derivative", None)
    state.pop("force_integral", None)
    state.pop("previous_entry_error", None)
    state["last_update_step"] = -1
