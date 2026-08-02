# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Native Dr.Anmar tissue mechanics and physics-owned puncture state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from isaaclab.assets import RigidObject
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


NEEDLE_TIP_LOCAL_M = (0.0, -0.0070028174960433945, 0.0)
NEEDLE_TANGENT_LOCAL = (1.0, 0.0, 0.0)
NEEDLE_PLANE_NORMAL_LOCAL = (0.0, 0.0, 1.0)
TISSUE_CENTER_LOCAL_M = (0.0, 0.0, 0.05)
TISSUE_TOP_LOCAL_Z_M = 0.053
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
    giver_held = state["bilateral_seen"] & (state["custody_owner"] < 2)
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
    giver_tissue_force = giver_tissue_forces.amax(dim=-1)
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
    custody_valid = state["bilateral_seen"].clone()
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
        curvature_radius_m = 0.0070028174960433945
        tissue_thickness_m = 0.006
        exit_angle = torch.asin(
            torch.tensor(tissue_thickness_m / curvature_radius_m, device=env.device)
        )
        exit_chord_m = curvature_radius_m * (1.0 - torch.cos(exit_angle))
        drive_direction = torch.linalg.cross(wound_tangent, -surface_normal)
        exit_target = (
            target
            + drive_direction * exit_chord_m
            - surface_normal * tissue_thickness_m
        )
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
        receiver_tissue_force = receiver_tissue_forces.amax(dim=-1)
        receiver_force_contact = torch.all(receiver_jaw_forces > 0.01, dim=-1)
        current_phase = torch.tensor(
            [int(gate.phase) for gate in state["gates"]], device=env.device
        )
        radius_m = 0.0070028174960433945
        exposed_mid_direction = torch.nn.functional.normalize(
            (tip_pos - root_pos) + (exit_position - root_pos), dim=-1
        )
        exposed_grasp_target = root_pos + radius_m * exposed_mid_direction
        receiver_target = torch.where(
            (current_phase < 8).unsqueeze(-1),
            exposed_grasp_target - surface_normal * 0.003,
            exposed_grasp_target,
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
        desired_receiver_quat_w = quat_mul(
            root_quat,
            torch.tensor(
                NEEDLE_POLICY_GRASP_QUAT_XYZW,
                device=env.device,
                dtype=root_quat.dtype,
            ).repeat(env.num_envs, 1),
        )
        desired_receiver_quat_r = quat_mul(
            quat_conjugate(receiver_root_quat), desired_receiver_quat_w
        )
        receiver_rotation_error = axis_angle_from_quat(
            quat_mul(
                desired_receiver_quat_r,
                quat_conjugate(receiver_tool_quat_r),
            )
        )
        receiver_distance = torch.linalg.vector_norm(
            receiver_tool_pos - exposed_grasp_target, dim=-1
        )
        action = mdp_common.as_torch(env.action_manager.action)
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
        exit_delta_w = exit_target - exit_position
        exit_delta_r = quat_apply_inverse(receiver_root_quat, exit_delta_w)
        pull_axis_r = quat_apply_inverse(receiver_root_quat, plane_normal)
        pull_guidance = torch.cat(
            (
                (exit_delta_r / 0.00025).clamp(-1.0, 1.0),
                torch.nn.functional.normalize(pull_axis_r, dim=-1),
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
        giver_open_commanded = action[:, 6] > 0.0
        transfer_ready = (
            (current_phase == 9)
            & (state["custody_owner"] >= 1)
            & giver_open_commanded
        )
        state["custody_owner"][transfer_ready] = 2
        receiver_custody = state["custody_owner"] >= 1
        giver_released = state["custody_owner"] == 2
    else:
        receiver_jaw_forces = torch.zeros((env.num_envs, 2), device=env.device)
        receiver_tissue_forces = torch.zeros((env.num_envs, 3), device=env.device)
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
            )
        else:
            measurement = PunctureMeasurement(
                **common,
                embedded_depth_m=float(embedded_depth[index]),
            )
        # Reset seating can briefly pass geometric thresholds while the jaws
        # and fixed grasp converge. It must never advance the authoritative
        # procedure state before settled bilateral custody is evaluated.
        if bool(settled[index]):
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
                "receiver_distance": receiver_distance,
                "receiver_jaw_forces": receiver_jaw_forces,
                "giver_tissue_force": giver_tissue_force,
                "receiver_tissue_force": receiver_tissue_force,
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
            "receiver_distance": receiver_distance,
            "receiver_bilateral": receiver_bilateral,
            "receiver_guidance": receiver_guidance,
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
