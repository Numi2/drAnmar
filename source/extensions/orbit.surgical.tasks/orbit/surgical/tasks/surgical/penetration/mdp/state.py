# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""CRESSim/PhysX co-simulation and physics-owned puncture state."""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.assets import RigidObject
from isaaclab.utils.math import combine_frame_transforms, quat_apply

from orbit.surgical.tasks.surgical import mdp_common

from ..contract import (
    PunctureMeasurement,
    PunctureThresholds,
    advance_puncture_gate,
    needle_tissue_force_components,
    puncture_success,
)
from ..backend import DrAnmarTissueEntryBackend, create_tissue_entry_backend
from ..cressim import NeedlePose
from .events import reset_penetration_evidence

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
    adapter = getattr(env, "_dr_anmar_cressim_adapter", None)
    if adapter is None:
        adapter = create_tissue_entry_backend(env.num_envs, integration_step_s=0.002)
        env._dr_anmar_cressim_adapter = adapter
        env._dr_anmar_cressim_finalizer = weakref.finalize(env, adapter.close)
    return adapter


def penetration_state(env: ManagerBasedRLEnv) -> dict[str, Any]:
    """Return one cached update built from native contacts and CRESSim wrench."""

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
    coupling = _adapter(env).step(tip_poses, arc_poses, punctured, dt_s=0.02)
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
    # Apply the combined MPM momentum balance and explicit
    # compression/cutting/sweep/shaft contact law to PhysX.  Puncture remains
    # environment-owned and requires the force-gated backend receipt.
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
    effective_custody = bilateral | ~settled
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
        successes.append(puncture_success(gate, measurement, thresholds))

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
            "normal_force": normal_force,
            "force_derivative": force_derivative,
            "force_integral": force_integral,
            "accumulated_work": accumulated_work,
            "jaw_forces": jaw_forces,
            "phase": phase,
            "event_count": event_count,
            "hard_failure": hard_failure,
            "success": torch.tensor(successes, device=env.device),
        }
    )
    state["settle_control_steps"] = torch.clamp(
        state["settle_control_steps"] - 1, min=0
    )
    return state
