# Copyright (c) 2024-2026, The ORBIT-Surgical and Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from orbit.surgical.tasks.surgical import mdp_common

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Legacy position-only goal predicate retained for downstream configs."""
    pos_error, _ = mdp_common.object_goal_errors(env, command_name, robot_cfg, object_cfg)
    return pos_error < threshold


def successful_lift(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    position_threshold: float = 0.015,
    orientation_threshold: float = 0.35,
    contact_threshold: float = 0.01,
    maximum_linear_speed: float = 0.08,
    maximum_angular_speed: float = 1.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    sensor_1_name: str = "jaw_1_object_contact",
    sensor_2_name: str = "jaw_2_object_contact",
) -> torch.Tensor:
    """Terminate only after a grasped, aligned, stable placement at the goal."""
    pos_error, rot_error = mdp_common.object_goal_errors(env, command_name, robot_cfg, object_cfg)
    motion = mdp_common.object_motion(env, object_cfg)
    grasped = mdp_common.bilateral_contact(env, sensor_1_name, sensor_2_name, contact_threshold)
    return (
        grasped
        & (pos_error < position_threshold)
        & (rot_error < orientation_threshold)
        & (motion[:, 0] < maximum_linear_speed)
        & (motion[:, 1] < maximum_angular_speed)
    )


def excessive_contact_force(
    env: ManagerBasedRLEnv, sensor_names: tuple[str, ...], hard_limit: float
) -> torch.Tensor:
    """Terminate when native PhysX contact exceeds a configured hard envelope."""
    return mdp_common.maximum_contact_force(env, sensor_names) > hard_limit


def excessive_non_object_contact_force(
    env: ManagerBasedRLEnv, sensor_names: tuple[str, ...], hard_limit: float
) -> torch.Tensor:
    """Terminate on excessive contact with a surface other than the target object."""
    return mdp_common.maximum_non_object_contact_force(env, sensor_names) > hard_limit
