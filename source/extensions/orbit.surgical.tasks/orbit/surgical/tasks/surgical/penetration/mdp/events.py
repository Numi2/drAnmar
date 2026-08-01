# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Reset the physical PSM/needle custody and entry evidence state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from pxr import Gf, Sdf, UsdPhysics

from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from orbit.surgical.assets.psm import PSM_GRIPPER_PROFILE
from orbit.surgical.tasks.surgical import mdp_common

from ..contract import PunctureGateState

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


NEEDLE_MID_GRASP_POSITION_M = (0.00613661575091, 0.00337363305778, 0.0)
# OpenUSD authors quatf as (real, i, j, k).  The Isaac tensors used here are
# XYZW, so the preferred needle-driver frame (0, .509, .861, 0) becomes:
NEEDLE_MID_GRASP_QUAT_XYZW = (0.50904141575, 0.860742027004, 0.0, 0.0)
PSM_TOOL_TIP_TO_JAW_COLLISION_M = (0.0, 0.0, 0.0014)
PENETRATION_GRIPPER_CLOSE_RAD = 0.005
PENETRATION_GRIPPER_OPEN_RAD = float(PSM_GRIPPER_PROFILE["open_rad"])
GRASP_BREAK_FORCE_N = 5.0
GRASP_BREAK_TORQUE_NM = 0.1


def _grasp_joint(
    env: ManagerBasedRLEnv,
    env_index: int,
    *,
    local_pos1: tuple[float, float, float] = NEEDLE_MID_GRASP_POSITION_M,
    local_quat1: tuple[float, float, float, float] = NEEDLE_MID_GRASP_QUAT_XYZW,
) -> UsdPhysics.FixedJoint:
    """Clamp the authored jaw/needle frames with a breakable PhysX joint."""

    namespace = f"/World/envs/env_{env_index}"
    joint = UsdPhysics.FixedJoint.Define(
        env.scene.stage, f"{namespace}/NeedleDriverGraspJoint"
    )
    joint.CreateBody0Rel().SetTargets(
        [Sdf.Path(f"{namespace}/Robot/psm_tool_tip_link")]
    )
    joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{namespace}/Needle")])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*PSM_TOOL_TIP_TO_JAW_COLLISION_M))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*local_pos1))
    x, y, z, w = local_quat1
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(w, x, y, z))
    joint.CreateBreakForceAttr().Set(GRASP_BREAK_FORCE_N)
    joint.CreateBreakTorqueAttr().Set(GRASP_BREAK_TORQUE_NM)
    joint.CreateCollisionEnabledAttr().Set(False)
    return joint


def _set_grasp_joint_enabled(
    env: ManagerBasedRLEnv, env_index: int, *, enabled: bool
) -> None:
    """Enable an aligned grasp joint or disable the prior episode's joint."""

    path = f"/World/envs/env_{env_index}/NeedleDriverGraspJoint"
    prim = env.scene.stage.GetPrimAtPath(path)
    if not enabled and not prim.IsValid():
        return
    joint = UsdPhysics.FixedJoint(prim) if prim.IsValid() else _grasp_joint(env, env_index)
    joint.CreateJointEnabledAttr().Set(enabled)


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
    grasp_target = ee_pos
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
    """Attach the aligned breakable grasp, then close the physical jaws."""

    robot: Articulation = env.scene["robot"]
    for env_index in env_ids.detach().cpu().tolist():
        _set_grasp_joint_enabled(env, env_index, enabled=True)
    joint_ids, _ = robot.find_joints("psm_tool_gripper.*_joint")
    close = PENETRATION_GRIPPER_CLOSE_RAD
    jaw_position = torch.tensor((-close, close), device=env.device).repeat(len(env_ids), 1)
    robot.set_joint_position_target_index(
        target=jaw_position, joint_ids=joint_ids, env_ids=env_ids
    )


def reset_pregrasped_needle(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> None:
    """Disable the old grasp and close jaws; seating follows fresh kinematics."""

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    robot: Articulation = env.scene["robot"]
    for env_index in env_ids.detach().cpu().tolist():
        _set_grasp_joint_enabled(env, env_index, enabled=False)

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
                (env.num_envs,), 31, dtype=torch.long, device=env.device
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
            "last_update_step": -1,
        }
        env._dr_anmar_penetration_state = state
    count = len(env_ids)
    state["puncture_force_n"][env_ids] = 0.35 + 2.85 * torch.rand(count, device=env.device)
    state["shaft_drag_n_m"][env_ids] = 8.0 + 47.0 * torch.rand(count, device=env.device)
    state["wetness"][env_ids] = 0.1 + 0.8 * torch.rand(count, device=env.device)
    state["material_scale"][env_ids] = 1.0
    # The first observation is evaluated immediately after reset.  Thirty
    # control intervals let the foundation-profile jaws close at their native
    # velocity, followed by more than ten complete physics intervals of seat.
    state["settle_control_steps"][env_ids] = 31
    state["grasp_loss_steps"][env_ids] = 0
    state["bilateral_seen"][env_ids] = False
    state["grasp_attach_stage"][env_ids] = 0
    for index in env_ids.detach().cpu().tolist():
        state["gates"][index] = PunctureGateState()
    state.pop("measurement", None)
    state.pop("wrench", None)
    state.pop("force_derivative", None)
    state.pop("force_integral", None)
    state.pop("previous_entry_error", None)
    state["last_update_step"] = -1
