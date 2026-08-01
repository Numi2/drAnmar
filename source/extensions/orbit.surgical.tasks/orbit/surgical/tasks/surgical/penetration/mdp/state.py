# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Native Dr.Anmar tissue mechanics and physics-owned puncture state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from isaaclab.assets import RigidObject
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    quat_conjugate,
    quat_mul,
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
from .events import (
    NEEDLE_MID_GRASP_POSITION_M,
    NEEDLE_MID_GRASP_QUAT_XYZW,
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
CONTACT_CUSTODY_THRESHOLD_N = 1.0e-5
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
        adapter = create_tissue_entry_backend(env.num_envs, integration_step_s=0.002)
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
        quat_xyzw = root_quat[index]
        quat_xyzw = tuple(float(value) for value in quat_xyzw)
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
    thresholds = PunctureThresholds()
    embedded_depth = torch.where(
        torch.tensor(punctured, device=env.device), gated_indentation,
        torch.zeros_like(gated_indentation),
    )
    component_rows: list[tuple[float, float, float, float, float]] = []
    for index in range(env.num_envs):
        components = needle_tissue_force_components(
            indentation_m=float(gated_indentation[index]),
            embedded_length_m=float(embedded_depth[index]),
            puncture_force_n=float(state["puncture_force_n"][index]),
            prepuncture_depth_m=thresholds.prepuncture_depth_m,
            cutting_fraction=0.55,
            shaft_drag_n_per_m=float(state["shaft_drag_n_m"][index]),
            sweep_stiffness_n_m2=40_000.0,
            swept_area_m2=0.00052 * float(embedded_depth[index]),
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
    bilateral = torch.all(jaw_forces > CONTACT_CUSTODY_THRESHOLD_N, dim=-1)
    state["bilateral_seen"] |= bilateral
    robot = env.scene["robot"]
    tool_body_ids, _ = robot.find_bodies("psm_tool_tip_link")
    tool_pos = mdp_common.as_torch(robot.data.body_pos_w)[:, tool_body_ids[0], :]
    tool_quat = mdp_common.as_torch(robot.data.body_quat_w)[:, tool_body_ids[0], :]
    grasp_quat = torch.tensor(
        NEEDLE_MID_GRASP_QUAT_XYZW, device=env.device, dtype=root_quat.dtype
    ).repeat(env.num_envs, 1)
    expected_quat = quat_mul(tool_quat, quat_conjugate(grasp_quat))
    grasp_position = torch.tensor(
        NEEDLE_MID_GRASP_POSITION_M, device=env.device, dtype=root_pos.dtype
    ).repeat(env.num_envs, 1)
    jaw_offset = torch.tensor(
        PSM_TOOL_TIP_TO_JAW_COLLISION_M, device=env.device, dtype=root_pos.dtype
    ).repeat(env.num_envs, 1)
    expected_grasp_pos = tool_pos + quat_apply(tool_quat, jaw_offset)
    expected_pos = expected_grasp_pos - quat_apply(expected_quat, grasp_position)
    grasp_position_error = torch.linalg.vector_norm(root_pos - expected_pos, dim=-1)
    grasp_quaternion_dot = torch.abs(torch.sum(root_quat * expected_quat, dim=-1)).clamp(0.0, 1.0)
    grasp_angle_error_deg = torch.rad2deg(2.0 * torch.acos(grasp_quaternion_dot))
    # The authored mid-jaw seat has finite collision clearance.  Accept the
    # bounded PhysX seating displacement, while bilateral contact remains
    # mandatory and any later excursion is still a hard grasp loss.
    grasp_pose_valid = (grasp_position_error <= 0.0015) & (grasp_angle_error_deg <= 10.0)
    custody_valid = state["bilateral_seen"] & grasp_pose_valid
    state["grasp_loss_steps"] = torch.where(
        settled & ~custody_valid,
        state["grasp_loss_steps"] + 1,
        torch.zeros_like(state["grasp_loss_steps"]),
    )
    sustained_grasp_loss = state["grasp_loss_steps"] >= 3
    effective_custody = ~sustained_grasp_loss | ~settled
    target_patch = (entry_error <= 0.001) & (gated_indentation > 0.0)
    unintended_jaw = (
        settled
        & (torch.min(jaw_forces, dim=-1).values < CONTACT_CUSTODY_THRESHOLD_N)
        & (torch.linalg.vector_norm(root_pos[:, :2] - target[:, :2], dim=-1) < 0.004)
    )

    successes: list[bool] = []
    for index, gate in enumerate(state["gates"]):
        measurement = PunctureMeasurement(
            entry_error_m=float(entry_error[index]),
            tangent_error_deg=float(tangent_error[index]),
            plane_error_deg=float(plane_error[index]),
            indentation_m=float(gated_indentation[index]),
            embedded_depth_m=float(embedded_depth[index]),
            normal_force_n=float(normal_force[index]),
            accumulated_work_j=float(accumulated_work[index]),
            bilateral_custody=bool(effective_custody[index]),
            target_patch_contact=bool(target_patch[index]),
            solver_finite=bool(torch.isfinite(wrench[index]).all()),
            unintended_jaw_contact=bool(unintended_jaw[index]),
            unintended_surface_crossing=bool(
                not punctured[index] and gated_indentation[index] > thresholds.depth_max_m
            ),
        )
        advance_puncture_gate(
            gate,
            measurement,
            puncture_force_n=float(state["puncture_force_n"][index]),
            thresholds=thresholds,
        )
        successes.append(
            puncture_success(gate, measurement, thresholds)
            and tissue_state[index].representation_switch_count == 1
        )

    phase = torch.tensor([int(gate.phase) for gate in state["gates"]], device=env.device)
    event_count = torch.tensor([gate.event_count for gate in state["gates"]], device=env.device)
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
        (state["settle_control_steps"] == 30) & (state["grasp_attach_stage"] == 0),
        as_tuple=False,
    ).squeeze(-1)
    if place_ids.numel() > 0:
        seat_pregrasped_needle(env, place_ids)
        state["grasp_attach_stage"][place_ids] = 1
    reseat_ids = torch.nonzero(
        (state["settle_control_steps"] == 29) & (state["grasp_attach_stage"] == 1),
        as_tuple=False,
    ).squeeze(-1)
    if reseat_ids.numel() > 0:
        seat_pregrasped_needle(env, reseat_ids)
        state["grasp_attach_stage"][reseat_ids] = 2
    attach_ids = torch.nonzero(
        (state["settle_control_steps"] == 28) & (state["grasp_attach_stage"] == 2),
        as_tuple=False,
    ).squeeze(-1)
    if attach_ids.numel() > 0:
        attach_pregrasped_needle(env, attach_ids)
        state["grasp_attach_stage"][attach_ids] = 3
    return state
