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
    return penetration_state(env)["jaw_forces"] * 0.2


def normalized_tissue_wrench(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = penetration_state(env)
    threshold = state["puncture_force_n"].clamp_min(1.0e-6).unsqueeze(-1)
    force = state["wrench"][:, :3] / threshold
    torque = state["wrench"][:, 3:] / (threshold * 0.0070028175)
    return torch.cat((force, torque), dim=-1).clamp(-5.0, 5.0)


def force_history(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = penetration_state(env)
    threshold = state["puncture_force_n"].clamp_min(1.0e-6)
    derivative = 0.02 * state["force_derivative"] / threshold
    integral = state["force_integral"] / threshold
    return torch.stack((derivative, integral), dim=-1).clamp(-5.0, 5.0)


def penetration_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    return functional.one_hot(penetration_state(env)["phase"], num_classes=5).float()


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
    local_strain = (state["measurement"]["indentation"] / 0.006).unsqueeze(-1)
    exact_surface_displacement = state["measurement"]["indentation"].unsqueeze(-1)
    return torch.cat(
        (scalar_state, state["force_components"], local_strain, exact_surface_displacement),
        dim=-1,
    )
