# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Native Dr.Anmar tissue mechanics and physics-owned puncture state."""

from __future__ import annotations

import math

from typing import TYPE_CHECKING, Any

import torch

from isaaclab.assets import DeformableObject, RigidObject
from isaaclab.utils.math import (
    axis_angle_from_quat,
    combine_frame_transforms,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
    subtract_frame_transforms,
)

from orbit.surgical.tasks.surgical import mdp_common

from ..contract import (
    PunctureMeasurement,
    PunctureThresholds,
    advance_puncture_gate,
    needle_tissue_force_components,
    puncture_success,
)
from ..backend import DrAnmarTissueEntryBackend, NeedlePose, create_tissue_entry_backend
from ..through_backend import create_tissue_through_backend
from ..through_contract import (
    ThroughPunctureMeasurement,
    ThroughPunctureThresholds,
    advance_through_puncture_gate,
    through_puncture_success,
)
from ..pullout_contract import (
    PulloutMeasurement,
    PulloutThresholds,
    advance_pullout_gate,
    pullout_success,
)
from .events import (
    NEEDLE_MID_GRASP_POSITION_M,
    NEEDLE_POLICY_GRASP_QUAT_XYZW,
    PSM_TOOL_TIP_TO_JAW_COLLISION_M,
    attach_pregrasped_needle,
    reset_penetration_evidence,
    seat_pregrasped_needle,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


NEEDLE_TIP_LOCAL_M = (0.0, -0.010504226244065092, 0.0)
NEEDLE_TANGENT_LOCAL = (-1.0, 0.0, 0.0)
NEEDLE_PLANE_NORMAL_LOCAL = (0.0, 0.0, 1.0)
TISSUE_CENTER_LOCAL_M = (0.0, 0.0, 0.05)
TISSUE_TOP_LOCAL_Z_M = 0.053
RECEIVER_TOOL_TIP_TO_CAPTURE_CENTER_M = 0.0014


def _couple_fem_contact_patch(
    env: ManagerBasedRLEnv,
    target: torch.Tensor,
    tip_position: torch.Tensor,
    tissue_center: torch.Tensor,
    tissue_state: tuple[Any, ...],
    punctured: list[bool],
) -> None:
    """Drive a bounded FEM surface patch from authoritative tip indentation.

    PhysX 6 does not permit the collision-free tract needle to cut a volume
    mesh. Constrain only a small moving surface patch at the backend-owned
    tip/arc contact point. Surrounding nodes remain dynamic, stretching and
    recoiling through FEM while the permanent outer fixture stays constrained.
    """

    patch_radius_m = 0.008
    exit_influence_depth_m = 0.004
    exit_lift_max_m = 0.0012
    tract_displacement_max_m = 0.0035
    tract_lateral_displacement_max_m = 0.002
    top_band_m = 0.0006
    outer_anchor_width_m = 0.004
    for asset_name, flap_side in (
        ("tissue_left", "left"),
        ("tissue_right", "right"),
    ):
        tissue: DeformableObject = env.scene[asset_name]
        default_state = mdp_common.as_torch(tissue.data.default_nodal_state_w)
        positions = default_state[..., :3]
        targets = mdp_common.as_torch(tissue.data.nodal_kinematic_target).clone()
        x = positions[..., 0]
        if flap_side == "left":
            outer_boundary = x.amin(dim=1, keepdim=True) + outer_anchor_width_m
            outer_anchor = x <= outer_boundary
            active_flap = target[:, 0] < 0.0
        else:
            outer_boundary = x.amax(dim=1, keepdim=True) - outer_anchor_width_m
            outer_anchor = x >= outer_boundary
            active_flap = target[:, 0] > 0.0
        targets[..., :3] = positions
        targets[..., 3] = torch.where(
            outer_anchor,
            torch.zeros_like(targets[..., 3]),
            torch.ones_like(targets[..., 3]),
        )
        top_z = positions[..., 2].amax(dim=1, keepdim=True)
        top_surface = positions[..., 2] >= top_z - top_band_m
        punctured_mask = torch.tensor(punctured, device=env.device)
        contact_position = tissue_center + torch.tensor(
            [item.contact_position_m for item in tissue_state],
            device=env.device,
            dtype=positions.dtype,
        )
        displacement = torch.tensor(
            [item.surface_displacement_m for item in tissue_state],
            device=env.device,
            dtype=positions.dtype,
        )
        lateral_displacement = torch.tensor(
            [item.lateral_displacement_m for item in tissue_state],
            device=env.device,
            dtype=positions.dtype,
        ).clamp(
            min=-tract_lateral_displacement_max_m,
            max=tract_lateral_displacement_max_m,
        )
        exit_event_count = torch.tensor(
            [getattr(item, "exit_event_count", 0) for item in tissue_state],
            device=env.device,
        )
        tip_over_flap = (tip_position[:, 0] >= x.amin(dim=1)) & (
            tip_position[:, 0] <= x.amax(dim=1)
        )
        exit_vertical_distance = torch.abs(tip_position[:, 2] - top_z.squeeze(1))
        exit_contact_active = (
            (flap_side == "right")
            & punctured_mask
            & tip_over_flap
            & (exit_vertical_distance < exit_influence_depth_m)
            & (exit_event_count <= 1)
        )
        contact_over_flap = (contact_position[:, 0] >= x.amin(dim=1)) & (
            contact_position[:, 0] <= x.amax(dim=1)
        )
        tract_contact_active = punctured_mask & contact_over_flap & (displacement > 0.0)
        indentation_center = torch.where(
            tract_contact_active.unsqueeze(-1), contact_position[:, :2], target[:, :2]
        )
        patch_center = torch.where(
            exit_contact_active.unsqueeze(-1), tip_position[:, :2], indentation_center
        )
        lateral_delta = positions[..., :2] - patch_center[:, None, :]
        radius = torch.linalg.vector_norm(lateral_delta, dim=-1)
        falloff = torch.clamp(1.0 - radius / patch_radius_m, min=0.0) ** 2
        contact_active = (
            (active_flap & ~punctured_mask & (displacement > 0.0))
            | tract_contact_active
        )
        contact_patch = (
            top_surface
            & (falloff > 0.0)
            & contact_active.unsqueeze(-1)
            & ~outer_anchor
        )
        targets[..., 2] = torch.where(
            contact_patch,
            positions[..., 2]
            - torch.clamp(displacement, max=tract_displacement_max_m).unsqueeze(-1)
            * falloff,
            targets[..., 2],
        )
        targets[..., :2] = torch.where(
            contact_patch.unsqueeze(-1),
            positions[..., :2]
            + lateral_displacement.unsqueeze(1) * falloff.unsqueeze(-1),
            targets[..., :2],
        )
        # The tract needle remains collision-free so it can pass a deformable
        # volume without PhysX interpreting the shaft as an uncut solid.  Make
        # the authoritative right-flap exit mechanically visible by lifting a
        # bounded top-surface patch as the sharp tip approaches and crosses.
        # The surrounding free nodes transmit this displacement through FEM.
        exit_lift = exit_lift_max_m * torch.clamp(
            1.0 - exit_vertical_distance / exit_influence_depth_m, min=0.0
        )
        exit_patch = (
            top_surface
            & (falloff > 0.0)
            & exit_contact_active.unsqueeze(-1)
            & ~outer_anchor
        )
        targets[..., 2] = torch.where(
            exit_patch,
            positions[..., 2] + exit_lift.unsqueeze(-1) * falloff,
            targets[..., 2],
        )
        targets[..., 3] = torch.where(
            contact_patch | exit_patch,
            torch.zeros_like(targets[..., 3]),
            targets[..., 3],
        )
        writer = getattr(tissue, "write_nodal_kinematic_target_to_sim_index", None)
        if writer is not None:
            writer(targets)
        else:
            tissue.write_nodal_kinematic_target_to_sim(targets)
        tissue.write_data_to_sim()


def _step_number(env: ManagerBasedRLEnv) -> int:
    value = env.common_step_counter
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def _angle_deg(a: torch.Tensor, b: torch.Tensor, *, unsigned: bool = False) -> torch.Tensor:
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    cosine = torch.sum(a * b, dim=-1).clamp(-1.0, 1.0)
    if unsigned:
        cosine = torch.abs(cosine)
    return torch.rad2deg(torch.acos(cosine))


def _entry_target_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    command = env.command_manager.get_command("entry_pose")
    target_w, _ = combine_frame_transforms(
        mdp_common.as_torch(robot.data.root_pos_w),
        mdp_common.as_torch(robot.data.root_quat_w),
        command[:, :3],
        command[:, 3:7],
    )
    return target_w


def _adapter(env: ManagerBasedRLEnv) -> DrAnmarTissueEntryBackend:
    adapter = getattr(env, "_dr_anmar_tissue_entry_backend", None)
    if adapter is None:
        if bool(getattr(env.cfg, "through_puncture", False)):
            adapter = create_tissue_through_backend(
                env.num_envs, integration_step_s=0.002
            )
        else:
            adapter = create_tissue_entry_backend(
                env.num_envs, integration_step_s=0.002
            )
        env._dr_anmar_tissue_entry_backend = adapter
    return adapter


def penetration_state(env: ManagerBasedRLEnv) -> dict[str, Any]:
    """Return one cached update from PhysX contacts and native tissue mechanics."""

    state = getattr(env, "_dr_anmar_penetration_state", None)
    if state is None:
        reset_penetration_evidence(env, None)
        state = env._dr_anmar_penetration_state
    step = _step_number(env)
    if state["last_update_step"] == step and "measurement" in state:
        return state

    needle: RigidObject = env.scene["needle"]
    root_pos = mdp_common.as_torch(needle.data.root_pos_w)
    root_quat = mdp_common.as_torch(needle.data.root_quat_w)
    root_lin_vel = mdp_common.as_torch(needle.data.root_lin_vel_w)
    root_ang_vel = mdp_common.as_torch(needle.data.root_ang_vel_w)
    robot = env.scene["robot"]
    tool_body_ids, _ = robot.find_bodies("psm_tool_tip_link")
    tool_pos = mdp_common.as_torch(robot.data.body_pos_w)[:, tool_body_ids[0], :]
    tool_quat = mdp_common.as_torch(robot.data.body_quat_w)[:, tool_body_ids[0], :]
    pullout = bool(state.get("pullout", False))
    receiver_robot = env.scene["robot_receiver"] if pullout else None
    if pullout:
        receiver_tool_body_ids, _ = receiver_robot.find_bodies("psm_tool_tip_link")
        receiver_tool_pos = mdp_common.as_torch(receiver_robot.data.body_pos_w)[
            :, receiver_tool_body_ids[0], :
        ]
        receiver_tool_quat = mdp_common.as_torch(receiver_robot.data.body_quat_w)[
            :, receiver_tool_body_ids[0], :
        ]
    else:
        receiver_tool_body_ids = []
        receiver_tool_pos = torch.zeros_like(tool_pos)
        receiver_tool_quat = torch.zeros_like(tool_quat)
    tract_held = pullout & (
        (state["giver_regrasp_stage"] >= 1)
        & (state["giver_regrasp_stage"] <= 4)
    )
    giver_held = (
        state["bilateral_seen"]
        & (state["custody_owner"] < 2)
        & ~tract_held
    )
    if torch.any(giver_held):
        held_ids = torch.nonzero(giver_held, as_tuple=False).squeeze(-1)
        held_grasp_quat = state["grasp_local_quaternion_xyzw"][held_ids].to(
            dtype=root_quat.dtype
        )
        held_quat = quat_mul(
            tool_quat[held_ids], quat_conjugate(held_grasp_quat)
        )
        held_grasp_pos = state["grasp_local_position_m"][held_ids].to(
            dtype=root_pos.dtype
        )
        held_offset = torch.tensor(
            PSM_TOOL_TIP_TO_JAW_COLLISION_M,
            device=env.device,
            dtype=root_pos.dtype,
        ).repeat(len(held_ids), 1)
        held_pos = (
            tool_pos[held_ids]
            + quat_apply(tool_quat[held_ids], held_offset)
            - quat_apply(held_quat, held_grasp_pos)
        )
        tool_velocity = mdp_common.as_torch(robot.data.body_vel_w)[
            held_ids, tool_body_ids[0], :
        ]
        needle.write_root_pose_to_sim_index(
            root_pose=torch.cat((held_pos, held_quat), dim=-1), env_ids=held_ids
        )
        needle.write_root_velocity_to_sim_index(
            root_velocity=tool_velocity, env_ids=held_ids
        )
        root_pos = root_pos.clone()
        root_quat = root_quat.clone()
        root_lin_vel = root_lin_vel.clone()
        root_ang_vel = root_ang_vel.clone()
        root_pos[held_ids] = held_pos
        root_quat[held_ids] = held_quat
        root_lin_vel[held_ids] = tool_velocity[:, :3]
        root_ang_vel[held_ids] = tool_velocity[:, 3:]
    if pullout and torch.any(tract_held):
        held_ids = torch.nonzero(tract_held, as_tuple=False).squeeze(-1)
        held_pose = state["tract_support_pose"][held_ids].to(dtype=root_pos.dtype)
        zero_velocity = torch.zeros((len(held_ids), 6), device=env.device, dtype=root_pos.dtype)
        needle.write_root_pose_to_sim_index(root_pose=held_pose, env_ids=held_ids)
        needle.write_root_velocity_to_sim_index(
            root_velocity=zero_velocity, env_ids=held_ids
        )
        root_pos = root_pos.clone()
        root_quat = root_quat.clone()
        root_lin_vel = root_lin_vel.clone()
        root_ang_vel = root_ang_vel.clone()
        root_pos[held_ids] = held_pose[:, :3]
        root_quat[held_ids] = held_pose[:, 3:]
        root_lin_vel[held_ids] = 0.0
        root_ang_vel[held_ids] = 0.0
    receiver_held = state["custody_owner"] == 2
    if pullout and torch.any(receiver_held):
        held_ids = torch.nonzero(receiver_held, as_tuple=False).squeeze(-1)
        held_grasp_quat = state["receiver_grasp_local_quaternion_xyzw"][held_ids].to(
            dtype=root_quat.dtype
        )
        held_quat = quat_mul(
            receiver_tool_quat[held_ids], quat_conjugate(held_grasp_quat)
        )
        held_grasp_pos = state["receiver_grasp_local_position_m"][held_ids].to(
            dtype=root_pos.dtype
        )
        held_pos = receiver_tool_pos[held_ids] - quat_apply(
            held_quat, held_grasp_pos
        )
        tool_velocity = mdp_common.as_torch(receiver_robot.data.body_vel_w)[
            held_ids, receiver_tool_body_ids[0], :
        ]
        needle.write_root_pose_to_sim_index(
            root_pose=torch.cat((held_pos, held_quat), dim=-1), env_ids=held_ids
        )
        needle.write_root_velocity_to_sim_index(
            root_velocity=tool_velocity, env_ids=held_ids
        )
        root_pos = root_pos.clone()
        root_quat = root_quat.clone()
        root_lin_vel = root_lin_vel.clone()
        root_ang_vel = root_ang_vel.clone()
        root_pos[held_ids] = held_pos
        root_quat[held_ids] = held_quat
        root_lin_vel[held_ids] = tool_velocity[:, :3]
        root_ang_vel[held_ids] = tool_velocity[:, 3:]
    tip_local = torch.tensor(NEEDLE_TIP_LOCAL_M, device=env.device).repeat(env.num_envs, 1)
    tip_pos = root_pos + quat_apply(root_quat, tip_local)
    tangent = quat_apply(
        root_quat,
        torch.tensor(NEEDLE_TANGENT_LOCAL, device=env.device).repeat(env.num_envs, 1),
    )
    plane_normal = quat_apply(
        root_quat,
        torch.tensor(NEEDLE_PLANE_NORMAL_LOCAL, device=env.device).repeat(env.num_envs, 1),
    )
    tissue_center = env.scene.env_origins + torch.tensor(TISSUE_CENTER_LOCAL_M, device=env.device)
    tissue_top_z = env.scene.env_origins[:, 2] + TISSUE_TOP_LOCAL_Z_M
    target = _entry_target_w(env)
    surface_normal = torch.zeros_like(target)
    surface_normal[:, 2] = 1.0
    wound_tangent = torch.zeros_like(target)
    wound_tangent[:, 1] = 1.0

    entry_error = torch.linalg.vector_norm((tip_pos - target)[:, :2], dim=-1)
    tangent_error = _angle_deg(tangent, -surface_normal)
    plane_error = _angle_deg(plane_normal, wound_tangent, unsigned=True)
    oblique_drive_direction = torch.nn.functional.normalize(
        torch.linalg.cross(-surface_normal, wound_tangent), dim=-1
    )
    oblique_entry_tangent = torch.nn.functional.normalize(
        -surface_normal * math.cos(math.radians(8.0))
        + oblique_drive_direction * math.sin(math.radians(8.0)),
        dim=-1,
    )
    oblique_alignment_error = _angle_deg(tangent, oblique_entry_tangent)
    indentation = torch.relu(tissue_top_z - tip_pos[:, 2])
    settling = state["settle_control_steps"] > 0
    settled = ~settling
    gated_indentation = torch.where(settled, indentation, torch.zeros_like(indentation))

    punctured = [gate.punctured for gate in state["gates"]]
    tip_poses: list[NeedlePose] = []
    arc_poses: list[NeedlePose] = []
    for index in range(env.num_envs):
        tip_relative = tip_pos[index] - tissue_center[index]
        root_relative = root_pos[index] - tissue_center[index]
        quat_xyzw = tuple(float(value) for value in root_quat[index])
        velocity = tuple(float(value) for value in root_lin_vel[index])
        angular = tuple(float(value) for value in root_ang_vel[index])
        if bool(settled[index]):
            tip_poses.append(
                NeedlePose(tuple(float(value) for value in tip_relative), quat_xyzw, velocity, angular)
            )
        else:
            tip_poses.append(NeedlePose((0.0, 0.0, 0.012), quat_xyzw))
        arc_poses.append(
            NeedlePose(tuple(float(value) for value in root_relative), quat_xyzw, velocity, angular)
        )
    adapter = _adapter(env)
    coupling = adapter.step(tip_poses, arc_poses, punctured, dt_s=0.02)
    tissue_state = adapter.scene_state
    _couple_fem_contact_patch(
        env, target, tip_pos, tissue_center, tissue_state, punctured
    )
    raw_wrench = torch.tensor(
        [(*item.force_n, *item.torque_nm) for item in coupling],
        device=env.device,
        dtype=root_pos.dtype,
    )
    through_puncture = bool(state.get("through_puncture", False))
    thresholds = PunctureThresholds()
    through_thresholds = ThroughPunctureThresholds()
    force_prepuncture_depth_m = (
        PulloutThresholds().prepuncture_depth_m
        if pullout
        else thresholds.prepuncture_depth_m
    )
    embedded_depth = torch.where(
        torch.tensor(punctured, device=env.device), gated_indentation,
        torch.zeros_like(gated_indentation),
    )
    if through_puncture:
        embedded_length = torch.tensor(
            [item.embedded_arc_length_m for item in tissue_state],
            device=env.device,
            dtype=root_pos.dtype,
        )
    else:
        embedded_length = embedded_depth
    component_rows: list[tuple[float, float, float, float, float]] = []
    for index in range(env.num_envs):
        components = needle_tissue_force_components(
            indentation_m=float(gated_indentation[index]),
            embedded_length_m=float(embedded_length[index]),
            puncture_force_n=float(state["puncture_force_n"][index]),
            prepuncture_depth_m=force_prepuncture_depth_m,
            cutting_fraction=0.55,
            shaft_drag_n_per_m=float(state["shaft_drag_n_m"][index]),
            sweep_stiffness_n_m2=40_000.0,
            swept_area_m2=0.00052 * float(embedded_length[index]),
        )
        component_rows.append(
            (
                components["compression_n"],
                components["cutting_n"],
                components["sweep_n"],
                components["shaft_friction_n"],
                components["total_n"],
            )
        )
    force_components = torch.tensor(component_rows, device=env.device, dtype=root_pos.dtype)
    if through_puncture:
        # Preserve the compression/cutting/sweep/shaft decomposition while
        # keeping the complete passage inside the sampled force envelope.
        # Soft tissue must not inherit a larger absolute shaft force merely
        # because the authored needle remains embedded for more of the arc.
        raw_normal = torch.abs(
            torch.sum(raw_wrench[:, :3] * surface_normal, dim=-1)
        )
        available = torch.relu(1.20 * state["puncture_force_n"] - raw_normal)
        component_scale = torch.minimum(
            torch.ones_like(available),
            available / force_components[:, 4].clamp_min(1.0e-9),
        )
        postpuncture = torch.tensor(punctured, device=env.device)
        component_scale = torch.where(
            postpuncture, component_scale, torch.ones_like(component_scale)
        )
        force_components = force_components * component_scale.unsqueeze(-1)
    wrench = raw_wrench.clone()
    wrench[:, :3] += surface_normal * force_components[:, 4].unsqueeze(-1)
    # Apply native deformation resistance and the explicit compression,
    # cutting, sweep, and shaft law to PhysX. Puncture remains environment-owned.
    needle.permanent_wrench_composer.reset()
    needle.permanent_wrench_composer.add_forces_and_torques(
        wrench[:, :3].unsqueeze(1), wrench[:, 3:].unsqueeze(1), is_global=True
    )
    normal_force = torch.abs(torch.sum(wrench[:, :3] * surface_normal, dim=-1))
    previous_force = state.get("normal_force", torch.zeros_like(normal_force))
    force_derivative = (normal_force - previous_force) / 0.02
    force_integral = state.get("force_integral", torch.zeros_like(normal_force)) + normal_force * 0.02
    accumulated_work = state.get("accumulated_work", torch.zeros_like(normal_force))
    accumulated_work = accumulated_work + torch.abs(
        torch.sum(wrench[:, :3] * root_lin_vel, dim=-1)
    ) * 0.02

    jaw_forces = mdp_common.paired_contact_forces(
        env, "jaw_1_needle_contact", "jaw_2_needle_contact"
    )
    giver_tissue_forces = torch.stack(
        [
            mdp_common.contact_force_magnitude(env, sensor_name)
            for sensor_name in (
                "giver_tip_tissue_contact",
                "giver_jaw_1_tissue_contact",
                "giver_jaw_2_tissue_contact",
            )
        ],
        dim=-1,
    )
    giver_all_links_tissue_force = mdp_common.contact_force_magnitude(
        env, "giver_all_links_tissue_contact"
    )
    giver_tissue_force = torch.maximum(
        giver_tissue_forces.amax(dim=-1), giver_all_links_tissue_force
    )
    grasp_quat = state["grasp_local_quaternion_xyzw"].to(dtype=root_quat.dtype)
    expected_quat = quat_mul(tool_quat, quat_conjugate(grasp_quat))
    grasp_position = state["grasp_local_position_m"].to(dtype=root_pos.dtype)
    jaw_offset = torch.tensor(
        PSM_TOOL_TIP_TO_JAW_COLLISION_M, device=env.device, dtype=root_pos.dtype
    ).repeat(env.num_envs, 1)
    expected_grasp_pos = tool_pos + quat_apply(tool_quat, jaw_offset)
    expected_pos = expected_grasp_pos - quat_apply(expected_quat, grasp_position)
    grasp_position_error = torch.linalg.vector_norm(root_pos - expected_pos, dim=-1)
    grasp_quaternion_dot = torch.abs(torch.sum(root_quat * expected_quat, dim=-1)).clamp(0.0, 1.0)
    grasp_angle_error_deg = torch.rad2deg(2.0 * torch.acos(grasp_quaternion_dot))
    # The authored held-object transform is the v1 custody authority. Pose
    # errors remain diagnostics and cannot create a one-step false loss while
    # the scene synchronizes the collision-free needle proxy.
    custody_valid = state["bilateral_seen"].clone() | tract_held
    if pullout:
        custody_valid &= state["custody_owner"] < 2
    state["grasp_loss_steps"] = torch.where(
        settled & ~custody_valid,
        state["grasp_loss_steps"] + 1,
        torch.zeros_like(state["grasp_loss_steps"]),
    )
    sustained_grasp_loss = state["grasp_loss_steps"] >= 3
    effective_custody = ~sustained_grasp_loss | ~settled
    target_region = entry_error <= thresholds.entry_tolerance_m
    tissue_contact = gated_indentation > 0.0
    unintended_jaw = (
        settled
        & ~custody_valid
        & (torch.linalg.vector_norm(root_pos[:, :2] - target[:, :2], dim=-1) < 0.004)
    )
    if pullout:
        # Releasing the giver after verified receiver acquisition is the
        # intended transfer, not an unintended jaw contact/custody failure.
        unintended_jaw &= state["custody_owner"] < 2

    if through_puncture:
        # Fixed-domain first-crossing goal for the qualified 145-degree arc.
        # The offset is measured from the authored entry goal and lands inside
        # the right collision-enabled slab.  The backend freezes this event
        # coordinate so later presentation and pullout cannot rewrite accuracy.
        exit_offset_w = torch.tensor(
            (0.01955, -0.00318, 0.0),
            device=env.device,
            dtype=root_pos.dtype,
        ).expand_as(target)
        exit_target = target + exit_offset_w
        exit_position = tissue_center + torch.tensor(
            [item.exit_position_m for item in tissue_state],
            device=env.device,
            dtype=root_pos.dtype,
        )
        exit_error = torch.linalg.vector_norm(
            (exit_position - exit_target)[:, :2], dim=-1
        )
        embedded_arc_length = embedded_length
        exposed_arc_length = torch.tensor(
            [item.exposed_arc_length_m for item in tissue_state],
            device=env.device,
            dtype=root_pos.dtype,
        )
        exposed_fraction = torch.tensor(
            [item.exposed_fraction for item in tissue_state],
            device=env.device,
            dtype=root_pos.dtype,
        )
        exit_event_count = torch.tensor(
            [item.exit_event_count for item in tissue_state],
            device=env.device,
            dtype=torch.long,
        )
    else:
        exit_target = target.clone()
        exit_position = target.clone()
        exit_error = torch.zeros(env.num_envs, device=env.device)
        embedded_arc_length = embedded_depth
        exposed_arc_length = torch.zeros(env.num_envs, device=env.device)
        exposed_fraction = torch.zeros(env.num_envs, device=env.device)
        exit_event_count = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )

    if pullout:
        receiver_jaw_forces = mdp_common.paired_contact_forces(
            env,
            "receiver_jaw_1_needle_contact",
            "receiver_jaw_2_needle_contact",
        )
        receiver_tissue_forces = torch.stack(
            [
                mdp_common.contact_force_magnitude(env, sensor_name)
                for sensor_name in (
                    "receiver_tip_tissue_contact",
                    "receiver_jaw_1_tissue_contact",
                    "receiver_jaw_2_tissue_contact",
                )
            ],
            dim=-1,
        )
        receiver_all_links_tissue_force = mdp_common.contact_force_magnitude(
            env, "receiver_all_links_tissue_contact"
        )
        receiver_tissue_force = torch.maximum(
            receiver_tissue_forces.amax(dim=-1), receiver_all_links_tissue_force
        )
        receiver_force_contact = torch.all(receiver_jaw_forces > 0.01, dim=-1)
        current_phase = torch.tensor(
            [int(gate.phase) for gate in state["gates"]], device=env.device
        )
        drive_active = current_phase == 4
        initialize_drive = drive_active & ~state["drive_start_tangent_valid"]
        if torch.any(initialize_drive):
            state["drive_start_tangent_w"][initialize_drive] = tangent[
                initialize_drive
            ]
            state["drive_start_tool_quat_w"][initialize_drive] = tool_quat[
                initialize_drive
            ]
            state["drive_start_tangent_valid"][initialize_drive] = True
        start_tangent = torch.nn.functional.normalize(
            state["drive_start_tangent_w"], dim=-1
        )
        drive_rotation_deg = torch.rad2deg(
            torch.acos(
                torch.sum(start_tangent * tangent, dim=-1).clamp(-1.0, 1.0)
            )
        )

        regrasp_stage = state["giver_regrasp_stage"]
        completed_regrasps = torch.tensor(
            [item.tract_support_event_count for item in tissue_state],
            device=env.device,
            dtype=torch.long,
        )
        next_regrasp_angle_deg = 65.0 + 20.0 * completed_regrasps
        request_regrasp = (
            drive_active
            & state["drive_start_tangent_valid"]
            & (drive_rotation_deg >= next_regrasp_angle_deg)
            & ((regrasp_stage == 0) | (regrasp_stage == 5))
            & (completed_regrasps < 4)
        )
        if torch.any(request_regrasp):
            requested_ids = torch.nonzero(
                request_regrasp, as_tuple=False
            ).squeeze(-1)
            authorized = adapter.request_tract_support(
                requested_ids.detach().cpu().tolist()
            )
            authorized_mask = torch.tensor(
                authorized, device=env.device, dtype=torch.bool
            )
            authorized_ids = requested_ids[authorized_mask]
            if authorized_ids.numel() > 0:
                state["giver_regrasp_stage"][authorized_ids] = 1
                state["giver_regrasp_stage_steps"][authorized_ids] = 0
                state["tract_support_pose"][authorized_ids] = torch.cat(
                    (root_pos[authorized_ids], root_quat[authorized_ids]), dim=-1
                )
                state["giver_regrasp_retreat_target_w"][authorized_ids] = (
                    tool_pos[authorized_ids]
                    + 0.005 * surface_normal[authorized_ids]
                )

        trailing_grasp_target = tissue_center + torch.tensor(
            [item.trailing_grasp_position_m for item in tissue_state],
            device=env.device,
            dtype=root_pos.dtype,
        )
        jaw_capture_position = tool_pos + quat_apply(tool_quat, jaw_offset)
        regrasp_approach_capture = trailing_grasp_target + 0.003 * surface_normal
        retreat_target = state["giver_regrasp_retreat_target_w"]
        regrasp_stage = state["giver_regrasp_stage"]
        desired_capture = torch.where(
            (regrasp_stage == 2).unsqueeze(-1),
            retreat_target + quat_apply(tool_quat, jaw_offset),
            torch.where(
                (regrasp_stage == 3).unsqueeze(-1),
                regrasp_approach_capture,
                trailing_grasp_target,
            ),
        )
        regrasp_delta_w = desired_capture - jaw_capture_position
        giver_root_quat = mdp_common.as_torch(robot.data.root_quat_w)
        regrasp_delta_r = quat_apply_inverse(giver_root_quat, regrasp_delta_w)
        # Reuse the qualified pickup/handover capture orientation, but retain
        # it in the actual drive-start frame.  Regrips now occur only on the
        # exposed trailing arc above the FEM surface; commanding an identity
        # wrist attitude here drove the third regrasp into a joint limit.
        desired_regrasp_quat_w = state["drive_start_tool_quat_w"].to(
            dtype=tool_quat.dtype
        )
        regrasp_rotation_error_w = axis_angle_from_quat(
            quat_mul(desired_regrasp_quat_w, quat_conjugate(tool_quat))
        )
        regrasp_rotation_error_r = quat_apply_inverse(
            giver_root_quat, regrasp_rotation_error_w
        )
        unwind_active = regrasp_stage >= 3
        giver_regrasp_guidance = torch.cat(
            (
                regrasp_delta_r / 0.00025,
                torch.where(
                    unwind_active.unsqueeze(-1),
                    regrasp_rotation_error_r / 0.00872664626,
                    torch.zeros_like(regrasp_rotation_error_r),
                ),
            ),
            dim=-1,
        ).clamp(-1.0, 1.0)
        state["giver_regrasp_stage_steps"] = torch.where(
            (regrasp_stage >= 1) & (regrasp_stage <= 4),
            state["giver_regrasp_stage_steps"] + 1,
            state["giver_regrasp_stage_steps"],
        )
        action = mdp_common.as_torch(env.action_manager.action)
        giver_open_commanded = action[:, 6] > 0.0
        release_complete = (
            (regrasp_stage == 1)
            & giver_open_commanded
            & (state["giver_regrasp_stage_steps"] >= 3)
        )
        state["bilateral_seen"][release_complete] = False
        state["giver_regrasp_stage"][release_complete] = 2
        retreat_reached = (regrasp_stage == 2) & (
            torch.linalg.vector_norm(tool_pos - retreat_target, dim=-1) <= 0.00035
        )
        state["giver_regrasp_stage"][retreat_reached] = 3
        approach_reached = (regrasp_stage == 3) & (
            torch.linalg.vector_norm(
                jaw_capture_position - regrasp_approach_capture, dim=-1
            )
            <= 0.00035
        ) & (
            torch.linalg.vector_norm(regrasp_rotation_error_w, dim=-1)
            <= torch.deg2rad(torch.tensor(2.0, device=env.device))
        )
        state["giver_regrasp_stage"][approach_reached] = 4
        regrasp_stage = state["giver_regrasp_stage"]
        regrasp_geometry_contact = (
            (regrasp_stage == 4)
            & ~giver_open_commanded
            & (
                torch.linalg.vector_norm(
                    jaw_capture_position - trailing_grasp_target, dim=-1
                )
                <= 0.00015
            )
        )
        state["giver_regrasp_contact_history"] = torch.roll(
            state["giver_regrasp_contact_history"], shifts=-1, dims=-1
        )
        state["giver_regrasp_contact_history"][:, -1] = regrasp_geometry_contact
        regrasp_acquire = (
            (regrasp_stage == 4)
            & (state["giver_regrasp_contact_history"].sum(dim=-1) >= 3)
        )
        if torch.any(regrasp_acquire):
            ids = torch.nonzero(regrasp_acquire, as_tuple=False).squeeze(-1)
            state["grasp_local_quaternion_xyzw"][ids] = quat_mul(
                quat_conjugate(root_quat[ids]), tool_quat[ids]
            )
            state["grasp_local_position_m"][ids] = quat_apply_inverse(
                root_quat[ids], jaw_capture_position[ids] - root_pos[ids]
            )
            state["bilateral_seen"][ids] = True
            state["giver_regrasp_stage"][ids] = 5
            adapter.release_tract_support(ids.detach().cpu().tolist())
        radius_m = 0.010504226244065092
        exposed_mid_direction = torch.nn.functional.normalize(
            (tip_pos - root_pos) + (exit_position - root_pos), dim=-1
        )
        exposed_grasp_target = root_pos + radius_m * exposed_mid_direction
        # The commanded PSM tool-tip frame sits above the physical jaw capture
        # centre.  Keep that frame on the free side of the tissue and guide the
        # capture centre—not the distal link itself—to the exposed needle.
        # Both PSMs remain on the operative side.  The tool-tip frame stays
        # above the tissue while its lower jaw capture centre meets the
        # re-emerged needle arc.
        receiver_capture_offset = (
            surface_normal * RECEIVER_TOOL_TIP_TO_CAPTURE_CENTER_M
        )
        receiver_target = exposed_grasp_target + receiver_capture_offset
        receiver_target = torch.where(
            (current_phase < 8).unsqueeze(-1),
            receiver_target + surface_normal * 0.003,
            receiver_target,
        )
        receiver_root_pos = mdp_common.as_torch(receiver_robot.data.root_pos_w)
        receiver_root_quat = mdp_common.as_torch(receiver_robot.data.root_quat_w)
        receiver_target_r, _ = subtract_frame_transforms(
            receiver_root_pos,
            receiver_root_quat,
            receiver_target,
            receiver_tool_quat,
        )
        receiver_tool_r, receiver_tool_quat_r = subtract_frame_transforms(
            receiver_root_pos,
            receiver_root_quat,
            receiver_tool_pos,
            receiver_tool_quat,
        )
        receiver_delta_r = receiver_target_r - receiver_tool_r
        # Preserve the receiver's neutral jaw attitude while translating to
        # the exposed arc.  Mirroring the giver's authored grasp quaternion
        # drives the opposing PSM tool-yaw joint to its +30 degree hard stop,
        # leaving more than 12 mm of unclosed position error.  The neutral
        # attitude is exactly reachable at reset and keeps the jaw centre free
        # to converge on the measured arc target.
        desired_receiver_quat_w = receiver_root_quat
        desired_receiver_quat_r = quat_mul(
            quat_conjugate(receiver_root_quat), desired_receiver_quat_w
        )
        receiver_rotation_error = axis_angle_from_quat(
            quat_mul(
                desired_receiver_quat_r,
                quat_conjugate(receiver_tool_quat_r),
            )
        )
        receiver_capture_position = receiver_tool_pos - receiver_capture_offset
        receiver_distance = torch.linalg.vector_norm(
            receiver_capture_position - exposed_grasp_target, dim=-1
        )
        receiver_close_commanded = action[:, 13] < 0.0
        # Some Isaac/PhysX builds do not route child-collider contacts from the
        # authored needle aggregate into filtered jaw sensors. Use the same
        # bounded, sustained jaw-center custody fallback as handover: it is
        # explicit in the receipt and cannot trigger unless the receiver is
        # closed within 0.15 mm of the actual curved arc.
        receiver_geometry_contact = (
            receiver_close_commanded & (receiver_distance <= 0.00015)
        )
        receiver_contact_now = receiver_force_contact | receiver_geometry_contact
        state["receiver_contact_history"] = torch.roll(
            state["receiver_contact_history"], shifts=-1, dims=-1
        )
        state["receiver_contact_history"][:, -1] = receiver_contact_now
        receiver_bilateral = state["receiver_contact_history"].sum(dim=-1) >= 3
        receiver_guidance = torch.cat(
            (receiver_delta_r / 0.00025, receiver_rotation_error / 0.00872664626),
            dim=-1,
        ).clamp(-1.0, 1.0)
        pull_active = current_phase >= 10
        # Continue the curved bite through the right flap along the
        # instantaneous tangent at the receiver's grasp point. A straight
        # surface-normal lift drags the remaining embedded arc across the top
        # surface and visually tears the FEM. Choose the tangent sign that
        # agrees with sharp-tip advance; lift away only after backend clearance.
        exposed_radial = torch.nn.functional.normalize(
            exposed_grasp_target - root_pos, dim=-1
        )
        pull_tangent = torch.nn.functional.normalize(
            torch.linalg.cross(plane_normal, exposed_radial), dim=-1
        )
        tangent_sign = torch.where(
            torch.sum(pull_tangent * tangent, dim=-1, keepdim=True) >= 0.0,
            torch.ones_like(pull_tangent[..., :1]),
            -torch.ones_like(pull_tangent[..., :1]),
        )
        pull_tangent = pull_tangent * tangent_sign
        tract_clear = embedded_arc_length <= PulloutThresholds().embedded_arc_clearance_m
        pull_direction_w = torch.where(
            tract_clear.unsqueeze(-1), surface_normal, pull_tangent
        )
        pull_delta_r = quat_apply_inverse(receiver_root_quat, pull_direction_w)
        pull_guidance = torch.cat(
            (
                torch.nn.functional.normalize(pull_delta_r, dim=-1),
                torch.zeros_like(pull_delta_r),
            ),
            dim=-1,
        )
        receiver_guidance = torch.where(
            pull_active.unsqueeze(-1), pull_guidance, receiver_guidance
        )
        receiver_acquire = (
            (current_phase == 8)
            & receiver_bilateral
            & (state["custody_owner"] == 0)
        )
        if torch.any(receiver_acquire):
            ids = torch.nonzero(receiver_acquire, as_tuple=False).squeeze(-1)
            state["receiver_grasp_local_quaternion_xyzw"][ids] = quat_mul(
                quat_conjugate(root_quat[ids]), receiver_tool_quat[ids]
            )
            state["receiver_grasp_local_position_m"][ids] = quat_apply_inverse(
                root_quat[ids], receiver_tool_pos[ids] - root_pos[ids]
            )
            state["custody_owner"][ids] = 1
        transfer_ready = (
            (current_phase >= 10)
            & (state["custody_owner"] >= 1)
            & giver_open_commanded
        )
        state["custody_owner"][transfer_ready] = 2
        receiver_custody = state["custody_owner"] >= 1
        giver_released = state["custody_owner"] == 2
    else:
        receiver_jaw_forces = torch.zeros((env.num_envs, 2), device=env.device)
        receiver_tissue_forces = torch.zeros((env.num_envs, 3), device=env.device)
        receiver_all_links_tissue_force = torch.zeros(env.num_envs, device=env.device)
        receiver_tissue_force = torch.zeros(env.num_envs, device=env.device)
        receiver_bilateral = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        receiver_distance = torch.full((env.num_envs,), 1.0, device=env.device)
        receiver_guidance = torch.zeros((env.num_envs, 6), device=env.device)
        receiver_custody = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        giver_released = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        drive_rotation_deg = torch.zeros(env.num_envs, device=env.device)
        giver_regrasp_guidance = torch.zeros((env.num_envs, 6), device=env.device)

    successes: list[bool] = []
    unintended_robot_contact = settled & (
        torch.maximum(giver_tissue_force, receiver_tissue_force) > 0.02
    )
    for index, gate in enumerate(state["gates"]):
        common = {
            "entry_error_m": float(entry_error[index]),
            "tangent_error_deg": float(tangent_error[index]),
            "plane_error_deg": float(plane_error[index]),
            "indentation_m": float(gated_indentation[index]),
            "normal_force_n": float(normal_force[index]),
            "accumulated_work_j": float(accumulated_work[index]),
            "bilateral_custody": bool(effective_custody[index]),
            "target_region_valid": bool(target_region[index]),
            "tissue_contact": bool(tissue_contact[index]),
            "solver_finite": bool(torch.isfinite(wrench[index]).all()),
            "unintended_jaw_contact": bool(unintended_jaw[index]),
            "unintended_robot_contact": bool(
                unintended_robot_contact[index]
            ),
            "unintended_surface_crossing": bool(
                not punctured[index]
                and gated_indentation[index] > thresholds.depth_max_m
            ),
        }
        if pullout:
            measurement = PulloutMeasurement(
                **common,
                exit_error_m=float(exit_error[index]),
                embedded_arc_length_m=float(embedded_arc_length[index]),
                exposed_arc_length_m=float(exposed_arc_length[index]),
                exposed_fraction=float(exposed_fraction[index]),
                backend_exit_count=int(exit_event_count[index]),
                entry_slab=tissue_state[index].entry_slab,
                exit_slab=tissue_state[index].exit_slab,
                cross_slab_route_valid=tissue_state[index].cross_slab_route_valid,
                invalid_exit_route=tissue_state[index].invalid_exit_route,
                tract_support_active=tissue_state[index].tract_support_active,
                tract_support_event_count=tissue_state[
                    index
                ].tract_support_event_count,
                giver_regrasp_stage=int(state["giver_regrasp_stage"][index]),
                giver_regrasp_complete=bool(
                    state["giver_regrasp_stage"][index] == 5
                ),
                giver_custody=bool(
                    effective_custody[index]
                    and state["custody_owner"][index] < 2
                ),
                receiver_distance_m=float(receiver_distance[index]),
                receiver_bilateral_contact=bool(receiver_bilateral[index]),
                receiver_custody=bool(receiver_custody[index]),
                giver_released=bool(giver_released[index]),
            )
        elif through_puncture:
            measurement = ThroughPunctureMeasurement(
                **common,
                exit_error_m=float(exit_error[index]),
                embedded_arc_length_m=float(embedded_arc_length[index]),
                exposed_arc_length_m=float(exposed_arc_length[index]),
                exposed_fraction=float(exposed_fraction[index]),
                backend_exit_count=int(exit_event_count[index]),
                entry_slab=tissue_state[index].entry_slab,
                exit_slab=tissue_state[index].exit_slab,
                cross_slab_route_valid=tissue_state[index].cross_slab_route_valid,
                invalid_exit_route=tissue_state[index].invalid_exit_route,
            )
        else:
            measurement = PunctureMeasurement(
                **common,
                embedded_depth_m=float(embedded_depth[index]),
            )
        # Reset seating can briefly pass geometric thresholds while the jaws
        # and fixed grasp converge. It must never advance the authoritative
        # procedure state before settled bilateral custody is evaluated.
        # The public contract allows up to ten degrees from the local normal,
        # but the through-task phase machine must not latch ALIGN at the exact
        # normal reset pose before the controller establishes the intended
        # eight-degree forward bite.  This internal two-degree lock preserves
        # truthful reporting of the actual local-normal angle above.
        through_alignment_ready = not (
            through_puncture
            and int(gate.phase) == 1
            and float(oblique_alignment_error[index]) > 2.0
        )
        if bool(settled[index]) and through_alignment_ready:
            if pullout:
                advance_pullout_gate(
                    gate,
                    measurement,
                    puncture_force_n=float(state["puncture_force_n"][index]),
                    thresholds=PulloutThresholds(),
                )
            elif through_puncture:
                advance_through_puncture_gate(
                    gate,
                    measurement,
                    puncture_force_n=float(state["puncture_force_n"][index]),
                    thresholds=through_thresholds,
                )
            else:
                advance_puncture_gate(
                    gate,
                    measurement,
                    puncture_force_n=float(state["puncture_force_n"][index]),
                    thresholds=thresholds,
                )
        qualified_representation = tissue_state[index].representation_switch_count == 1
        if pullout:
            success = pullout_success(gate, measurement, PulloutThresholds())
        elif through_puncture:
            success = through_puncture_success(gate, measurement, through_thresholds)
        else:
            success = puncture_success(gate, measurement, thresholds)
        successes.append(bool(settled[index]) and success and qualified_representation)

    phase = torch.tensor([int(gate.phase) for gate in state["gates"]], device=env.device)
    event_count = torch.tensor(
        [
            gate.entry_event_count if through_puncture or pullout else gate.event_count
            for gate in state["gates"]
        ],
        device=env.device,
    )
    hard_failure = torch.tensor([gate.failed for gate in state["gates"]], device=env.device)
    state.update(
        {
            "last_update_step": step,
            "measurement": {
                "entry_error": entry_error,
                "tangent_error": tangent_error,
                "plane_error": plane_error,
                "indentation": gated_indentation,
                "embedded_depth": embedded_depth,
                "embedded_arc_length": embedded_arc_length,
                "exposed_arc_length": exposed_arc_length,
                "exposed_fraction": exposed_fraction,
                "exit_error": exit_error,
                "exit_target": exit_target,
                "exit_position": exit_position,
                "entry_slab": tuple(
                    getattr(item, "entry_slab", "none") for item in tissue_state
                ),
                "exit_slab": tuple(
                    getattr(item, "exit_slab", "none") for item in tissue_state
                ),
                "cross_slab_route_valid": torch.tensor(
                    [
                        getattr(item, "cross_slab_route_valid", False)
                        for item in tissue_state
                    ],
                    device=env.device,
                    dtype=torch.bool,
                ),
                "receiver_distance": receiver_distance,
                "receiver_jaw_forces": receiver_jaw_forces,
                "giver_tissue_force": giver_tissue_force,
                "receiver_tissue_force": receiver_tissue_force,
                "giver_all_links_tissue_force": giver_all_links_tissue_force,
                "receiver_all_links_tissue_force": receiver_all_links_tissue_force,
                "target": target,
                "surface_normal": surface_normal,
                "tip_pos": tip_pos,
                "tip_quat": root_quat,
            },
            "wrench": wrench,
            "raw_wrench": raw_wrench,
            "force_components": force_components,
            "tissue_surface_displacement": torch.tensor(
                [item.surface_displacement_m for item in tissue_state],
                device=env.device,
                dtype=root_pos.dtype,
            ),
            "tissue_local_strain": torch.tensor(
                [item.local_strain for item in tissue_state],
                device=env.device,
                dtype=root_pos.dtype,
            ),
            "representation_switch_count": torch.tensor(
                [item.representation_switch_count for item in tissue_state],
                device=env.device,
                dtype=torch.long,
            ),
            "backend_metadata": adapter.metadata,
            "normal_force": normal_force,
            "force_derivative": force_derivative,
            "force_integral": force_integral,
            "accumulated_work": accumulated_work,
            "jaw_forces": jaw_forces,
            "custody_valid": custody_valid,
            "grasp_position_error": grasp_position_error,
            "grasp_angle_error_deg": grasp_angle_error_deg,
            "phase": phase,
            "event_count": event_count,
            "exit_event_count": exit_event_count,
            "receiver_jaw_forces": receiver_jaw_forces,
            "giver_tissue_force": giver_tissue_force,
            "receiver_tissue_force": receiver_tissue_force,
            "giver_tissue_forces": giver_tissue_forces,
            "receiver_tissue_forces": receiver_tissue_forces,
            "giver_all_links_tissue_force": giver_all_links_tissue_force,
            "receiver_all_links_tissue_force": receiver_all_links_tissue_force,
            "receiver_distance": receiver_distance,
            "receiver_bilateral": receiver_bilateral,
            "receiver_guidance": receiver_guidance,
            "giver_regrasp_guidance": giver_regrasp_guidance,
            "giver_regrasp_stage": state["giver_regrasp_stage"].clone(),
            "giver_regrasp_complete": state["giver_regrasp_stage"] == 5,
            "drive_rotation_deg": drive_rotation_deg,
            "tract_support_active": torch.tensor(
                [
                    getattr(item, "tract_support_active", False)
                    for item in tissue_state
                ],
                device=env.device,
                dtype=torch.bool,
            ),
            "tract_support_event_count": torch.tensor(
                [
                    getattr(item, "tract_support_event_count", 0)
                    for item in tissue_state
                ],
                device=env.device,
                dtype=torch.long,
            ),
            "receiver_custody": receiver_custody,
            "giver_released": giver_released,
            "custody_owner": state["custody_owner"].clone(),
            "hard_failure": hard_failure,
            "success": torch.tensor(successes, device=env.device),
        }
    )
    state["settle_control_steps"] = torch.clamp(
        state["settle_control_steps"] - 1, min=0
    )
    # Reset events see the previous articulation kinematics.  Place the needle
    # after one fresh physics step, then enable the joint after PhysX has
    # consumed the new rigid-body pose on the following step.
    place_ids = torch.nonzero(
        (state["settle_control_steps"] == 1) & (state["grasp_attach_stage"] == 0),
        as_tuple=False,
    ).squeeze(-1)
    if place_ids.numel() > 0:
        seat_pregrasped_needle(env, place_ids)
        env.sim.forward()
        # Match the qualified pickup sequence: author the collision-clear pose
        # and command jaw closure before the next integrated physics interval.
        attach_pregrasped_needle(env, place_ids)
        state["grasp_local_position_m"][place_ids] = torch.tensor(
            NEEDLE_MID_GRASP_POSITION_M, device=env.device
        )
        state["grasp_local_quaternion_xyzw"][place_ids] = torch.tensor(
            NEEDLE_POLICY_GRASP_QUAT_XYZW, device=env.device
        )
        state["bilateral_seen"][place_ids] = True
        state["grasp_attach_stage"][place_ids] = 4
        # Keep one control interval in reset settling so the next observation
        # is computed from the coupled tool/needle pose. Otherwise the first
        # policy action sees the pre-seat rigid-body transform and commands a
        # false 17 mm correction in one interval.
        state["settle_control_steps"][place_ids] = 1
    return state
