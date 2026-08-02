# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Reset the physical PSM/needle custody and entry evidence state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from orbit.surgical.assets.psm import PSM_GRIPPER_PROFILE
from orbit.surgical.tasks.surgical import mdp_common

from ..contract import PunctureGateState
from ..through_contract import ThroughPunctureGateState
from ..pullout_contract import PulloutGateState

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Use the authored preferred driver grasp on the needle body. The needle's
# sharp direction is local -X (toward the endpoint at -pi/2); treating +X as
# sharp put this valid grasp ahead of the tip and led to an artificial mirrored
# grasp outside the rendered semicircle.
NEEDLE_MID_GRASP_POSITION_M = (0.009204923626365, 0.00506044958667, 0.0)
# This Isaac Lab build exposes tensors in XYZW, while OpenUSD authors quatf in
# WXYZ.  Keep the calibrated jaw-seat rotation in tensor convention and
# convert only at the USD boundary below.
NEEDLE_MID_GRASP_QUAT_XYZW = (
    -0.2588191330432892,
    0.9659258127212524,
    0.0,
    0.0,
)
# Authored entry-policy presentation frame.
NEEDLE_POLICY_GRASP_QUAT_XYZW = NEEDLE_MID_GRASP_QUAT_XYZW
PSM_TOOL_TIP_TO_JAW_COLLISION_M = (0.0, 0.0, 0.0)
PENETRATION_GRIPPER_CLOSE_RAD = float(PSM_GRIPPER_PROFILE["close_rad"])
PENETRATION_GRIPPER_OPEN_RAD = float(PSM_GRIPPER_PROFILE["open_rad"])
TISSUE_OUTER_ANCHOR_WIDTH_M = 0.004


def reset_and_anchor_tissue_fem(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor | None
) -> None:
    """Reset both FEM flaps and pin only their remote outer boundary bands."""

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    for asset_name, outer_side in (
        ("tissue_left", "minimum"),
        ("tissue_right", "maximum"),
    ):
        tissue: DeformableObject = env.scene[asset_name]
        default_state = mdp_common.as_torch(tissue.data.default_nodal_state_w)[
            env_ids
        ].clone()
        default_state[..., 3:] = 0.0
        tissue.write_nodal_state_to_sim(default_state, env_ids=env_ids)

        targets = mdp_common.as_torch(tissue.data.nodal_kinematic_target)[
            env_ids
        ].clone()
        positions = default_state[..., :3]
        coordinates = positions[..., 0]
        if outer_side == "minimum":
            boundary = coordinates.amin(dim=1, keepdim=True) + TISSUE_OUTER_ANCHOR_WIDTH_M
            anchor_mask = coordinates <= boundary
        else:
            boundary = coordinates.amax(dim=1, keepdim=True) - TISSUE_OUTER_ANCHOR_WIDTH_M
            anchor_mask = coordinates >= boundary
        targets[..., :3] = positions
        # PhysX uses zero for constrained nodes and one for freely simulated
        # nodes. Wound-edge nodes remain free to indent, stretch, and rebound.
        targets[..., 3] = torch.where(
            anchor_mask,
            torch.zeros_like(targets[..., 3]),
            torch.ones_like(targets[..., 3]),
        )
        writer = getattr(tissue, "write_nodal_kinematic_target_to_sim_index", None)
        if writer is not None:
            writer(targets, env_ids=env_ids)
        else:
            tissue.write_nodal_kinematic_target_to_sim(targets, env_ids=env_ids)
        tissue.write_data_to_sim()


def configure_tissue_collision_filter(env: ManagerBasedRLEnv) -> None:
    """Keep FEM tissue in its sole PhysX collision group.

    The authored entry needle has no rigid collision shapes: the environment's
    force-gated tract backend owns needle resistance. Creating a second USD
    collision group for a volume deformable is unsupported in Isaac 6 and can
    suppress its intended PSM contact, so no needle/tissue filter is needed.
    """

    if getattr(env, "_dr_anmar_tissue_collision_filter_configured", False):
        return
    env._dr_anmar_tissue_collision_filter_configured = True


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

    configure_tissue_collision_filter(env)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    through_puncture = bool(getattr(env.cfg, "through_puncture", False))
    pullout = bool(getattr(env.cfg, "pullout", False))
    gate_factory = (
        PulloutGateState
        if pullout
        else ThroughPunctureGateState
        if through_puncture
        else PunctureGateState
    )
    state = getattr(env, "_dr_anmar_penetration_state", None)
    if state is None:
        state = {
            "gates": [gate_factory() for _ in range(env.num_envs)],
            "through_puncture": through_puncture,
            "pullout": pullout,
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
            "custody_owner": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "receiver_contact_history": torch.zeros(
                (env.num_envs, 5), dtype=torch.bool, device=env.device
            ),
            "receiver_grasp_local_position_m": torch.zeros(
                (env.num_envs, 3), device=env.device
            ),
            "receiver_grasp_local_quaternion_xyzw": torch.zeros(
                (env.num_envs, 4), device=env.device
            ),
            "giver_regrasp_stage": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "giver_regrasp_stage_steps": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "giver_regrasp_contact_history": torch.zeros(
                (env.num_envs, 5), dtype=torch.bool, device=env.device
            ),
            "tract_support_pose": torch.zeros((env.num_envs, 7), device=env.device),
            "giver_regrasp_retreat_target_w": torch.zeros(
                (env.num_envs, 3), device=env.device
            ),
            "drive_start_tangent_w": torch.zeros((env.num_envs, 3), device=env.device),
            "drive_previous_tangent_w": torch.zeros(
                (env.num_envs, 3), device=env.device
            ),
            "drive_rotation_rad": torch.zeros(env.num_envs, device=env.device),
            "drive_start_tool_quat_w": torch.zeros((env.num_envs, 4), device=env.device),
            "drive_start_tangent_valid": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "receiver_curve_previous_tangent_w": torch.zeros(
                (env.num_envs, 3), device=env.device
            ),
            "receiver_curve_rotation_rad": torch.zeros(
                env.num_envs, device=env.device
            ),
            "receiver_curve_center_w": torch.zeros(
                (env.num_envs, 3), device=env.device
            ),
            "receiver_curve_tracking_active": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
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
    state["custody_owner"][env_ids] = 0
    state["receiver_contact_history"][env_ids] = False
    state["receiver_grasp_local_position_m"][env_ids] = 0.0
    state["receiver_grasp_local_quaternion_xyzw"][env_ids] = 0.0
    state["giver_regrasp_stage"][env_ids] = 0
    state["giver_regrasp_stage_steps"][env_ids] = 0
    state["giver_regrasp_contact_history"][env_ids] = False
    state["tract_support_pose"][env_ids] = 0.0
    state["giver_regrasp_retreat_target_w"][env_ids] = 0.0
    state["drive_start_tangent_w"][env_ids] = 0.0
    state["drive_previous_tangent_w"][env_ids] = 0.0
    state["drive_rotation_rad"][env_ids] = 0.0
    state["drive_start_tool_quat_w"][env_ids] = 0.0
    state["drive_start_tangent_valid"][env_ids] = False
    state["receiver_curve_previous_tangent_w"][env_ids] = 0.0
    state["receiver_curve_rotation_rad"][env_ids] = 0.0
    state["receiver_curve_center_w"][env_ids] = 0.0
    state["receiver_curve_tracking_active"][env_ids] = False
    state["grasp_local_position_m"][env_ids] = torch.tensor(
        NEEDLE_MID_GRASP_POSITION_M, device=env.device
    )
    state["grasp_local_quaternion_xyzw"][env_ids] = torch.tensor(
        NEEDLE_MID_GRASP_QUAT_XYZW, device=env.device
    )
    if bool(state.get("through_puncture", False)) != through_puncture:
        raise RuntimeError("penetration task mode cannot change without recreating the environment")
    for index in env_ids.detach().cpu().tolist():
        state["gates"][index] = gate_factory()
    state.pop("measurement", None)
    state.pop("wrench", None)
    state.pop("force_derivative", None)
    state.pop("force_integral", None)
    state.pop("previous_entry_error", None)
    state.pop("previous_exposed_fraction", None)
    state.pop("previous_tip_pos", None)
    state.pop("rewarded_phase", None)
    state.pop("previous_receiver_distance", None)
    state.pop("previous_embedded_arc_length", None)
    state["last_update_step"] = -1
