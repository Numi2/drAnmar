# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as functional

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from orbit.surgical.tasks.surgical import mdp_common

from .state import penetration_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def end_effector_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    return mdp_common.end_effector_pose_in_robot_root_frame(
        env, SceneEntityCfg("ee_frame"), SceneEntityCfg("robot")
    )


def needle_tip_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = penetration_state(env)["measurement"]
    robot = env.scene["robot"]
    pos, quat = subtract_frame_transforms(
        mdp_common.as_torch(robot.data.root_pos_w),
        mdp_common.as_torch(robot.data.root_quat_w),
        state["tip_pos"],
        state["tip_quat"],
    )
    return torch.cat((pos, quat), dim=-1)


def needle_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    return mdp_common.object_velocity_in_robot_root_frame(
        env, SceneEntityCfg("robot"), SceneEntityCfg("needle")
    )


def entry_surface_normal(env: ManagerBasedRLEnv) -> torch.Tensor:
    normal_w = penetration_state(env)["measurement"]["surface_normal"]
    robot_quat = mdp_common.as_torch(env.scene["robot"].data.root_quat_w)
    from isaaclab.utils.math import quat_apply_inverse

    return quat_apply_inverse(robot_quat, normal_w)


def indentation_and_depth(env: ManagerBasedRLEnv) -> torch.Tensor:
    measurement = penetration_state(env)["measurement"]
    return torch.stack((measurement["indentation"], measurement["embedded_depth"]), dim=-1)


def jaw_contacts(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = penetration_state(env)
    settled = (state["settle_control_steps"] == 0).unsqueeze(-1)
    custody = state["custody_valid"].unsqueeze(-1)
    observed = torch.maximum(
        state["jaw_forces"] * 0.2,
        torch.full_like(state["jaw_forces"], 0.05),
    )
    return torch.where(settled & custody, observed, torch.zeros_like(observed))


def normalized_tissue_wrench(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = penetration_state(env)
    threshold = state["puncture_force_n"].clamp_min(1.0e-6).unsqueeze(-1)
    force = state["wrench"][:, :3] / threshold
    torque = state["wrench"][:, 3:] / (threshold * 0.0105042262)
    return torch.cat((force, torque), dim=-1).clamp(-5.0, 5.0)


def force_history(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = penetration_state(env)
    threshold = state["puncture_force_n"].clamp_min(1.0e-6)
    derivative = 0.02 * state["force_derivative"] / threshold
    integral = state["force_integral"] / threshold
    return torch.stack((derivative, integral), dim=-1).clamp(-5.0, 5.0)


def penetration_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    return functional.one_hot(penetration_state(env)["phase"], num_classes=5).float()


def through_puncture_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    return functional.one_hot(
        penetration_state(env)["phase"].clamp(max=6), num_classes=7
    ).float()


def through_puncture_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    measurement = penetration_state(env)["measurement"]
    return torch.stack(
        (
            measurement["embedded_arc_length"],
            measurement["exposed_fraction"],
            measurement["exit_error"],
        ),
        dim=-1,
    )


def through_exit_delta(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = penetration_state(env)["measurement"]
    delta_w = state["exit_target"] - state["exit_position"]
    robot_quat = mdp_common.as_torch(env.scene["robot"].data.root_quat_w)
    from isaaclab.utils.math import quat_apply_inverse

    return quat_apply_inverse(robot_quat, delta_w)


def through_drive_rotation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return unwrapped needle-drive rotation, normalized by one revolution."""

    return (
        penetration_state(env)["drive_rotation_deg"].unsqueeze(-1) / 360.0
    ).clamp(0.0, 1.0)


def privileged_puncture_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = penetration_state(env)
    scalar_state = torch.stack(
        (
            state["event_count"].float(),
            state["puncture_force_n"],
            state["shaft_drag_n_m"],
            state["wetness"],
            state["material_scale"],
            state["measurement"]["indentation"],
            state["measurement"]["embedded_depth"],
        ),
        dim=-1,
    )
    local_strain = state["tissue_local_strain"].unsqueeze(-1)
    exact_surface_displacement = state["tissue_surface_displacement"].unsqueeze(-1)
    return torch.cat(
        (scalar_state, state["force_components"], local_strain, exact_surface_displacement),
        dim=-1,
    )


def privileged_through_puncture_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = penetration_state(env)
    return torch.cat(
        (
            privileged_puncture_state(env),
            state["measurement"]["embedded_arc_length"].unsqueeze(-1),
            state["measurement"]["exposed_arc_length"].unsqueeze(-1),
            state["measurement"]["exposed_fraction"].unsqueeze(-1),
            state["measurement"]["exit_error"].unsqueeze(-1),
            state["exit_event_count"].float().unsqueeze(-1),
        ),
        dim=-1,
    )


def pullout_receiver_ee_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    return mdp_common.end_effector_pose_in_robot_root_frame(
        env, SceneEntityCfg("receiver_frame"), SceneEntityCfg("robot_receiver")
    )


def pullout_receiver_contacts(env: ManagerBasedRLEnv) -> torch.Tensor:
    return penetration_state(env)["receiver_jaw_forces"].clamp(0.0, 5.0)


def pullout_receiver_guidance(env: ManagerBasedRLEnv) -> torch.Tensor:
    return penetration_state(env)["receiver_guidance"]


def pullout_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    return functional.one_hot(penetration_state(env)["phase"], num_classes=12).float()


def pullout_custody(env: ManagerBasedRLEnv) -> torch.Tensor:
    return functional.one_hot(
        penetration_state(env)["custody_owner"], num_classes=3
    ).float()


def pullout_giver_regrasp_guidance(env: ManagerBasedRLEnv) -> torch.Tensor:
    return penetration_state(env)["giver_regrasp_guidance"]


def pullout_giver_regrasp_stage(env: ManagerBasedRLEnv) -> torch.Tensor:
    return functional.one_hot(
        penetration_state(env)["giver_regrasp_stage"], num_classes=6
    ).float()


def privileged_pullout_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = penetration_state(env)
    return torch.cat(
        (
            privileged_through_puncture_state(env),
            state["receiver_distance"].unsqueeze(-1),
            state["receiver_custody"].float().unsqueeze(-1),
            state["giver_released"].float().unsqueeze(-1),
            state["custody_owner"].float().unsqueeze(-1),
            state["drive_rotation_deg"].unsqueeze(-1) / 180.0,
            state["tract_support_active"].float().unsqueeze(-1),
            state["tract_support_event_count"].float().unsqueeze(-1),
            state["giver_regrasp_complete"].float().unsqueeze(-1),
        ),
        dim=-1,
    )
