# Copyright (c) 2024-2026, The ORBIT-Surgical and Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from orbit.surgical.tasks.surgical import mdp_common

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object position in the robot root frame."""
    return mdp_common.object_pose_in_robot_root_frame(env, robot_cfg, object_cfg)[:, :3]


def object_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object position and orientation in the robot root frame."""
    return mdp_common.object_pose_in_robot_root_frame(env, robot_cfg, object_cfg)


def object_velocity_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object linear and angular velocity in the robot root frame."""
    return mdp_common.object_velocity_in_robot_root_frame(env, robot_cfg, object_cfg)


def end_effector_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Measured end-effector pose in the robot root frame."""
    return mdp_common.end_effector_pose_in_robot_root_frame(env, frame_cfg, robot_cfg)


def jaw_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_1_name: str = "jaw_1_object_contact",
    sensor_2_name: str = "jaw_2_object_contact",
    scale: float = 1.0,
) -> torch.Tensor:
    """Measured filtered contact force for each opposing jaw."""
    return mdp_common.paired_contact_forces(env, sensor_1_name, sensor_2_name, scale)
