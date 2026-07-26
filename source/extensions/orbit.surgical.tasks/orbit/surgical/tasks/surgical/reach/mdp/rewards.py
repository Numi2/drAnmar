# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# Copyright (c) 2026, Dr.Anmar Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dense, bounded learning signals for Dr.Anmar single-arm pose reaching."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply_inverse,
    quat_box_minus,
    quat_error_magnitude,
)

from ...mdp_common import as_torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def pose_command_errors(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return root-frame position vector, position norm, and orientation error."""
    robot: Articulation = env.scene[robot_cfg.name]
    frame: FrameTransformer = env.scene[frame_cfg.name]
    command = env.command_manager.get_command(command_name)

    root_pos_w = as_torch(robot.data.root_pos_w)
    root_quat_w = as_torch(robot.data.root_quat_w)
    desired_pos_w, desired_quat_w = combine_frame_transforms(
        root_pos_w, root_quat_w, command[:, :3], command[:, 3:7]
    )
    current_pos_w = as_torch(frame.data.target_pos_w)[:, 0, :]
    current_quat_w = as_torch(frame.data.target_quat_w)[:, 0, :]
    position_error_w = desired_pos_w - current_pos_w
    position_error_b = quat_apply_inverse(root_quat_w, position_error_w)
    position_error = torch.linalg.vector_norm(position_error_w, dim=-1)
    orientation_error = quat_error_magnitude(current_quat_w, desired_quat_w)
    return position_error_b, position_error, orientation_error


def pose_command_error_vector(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Direct target-relative tool position for sample-efficient learning."""
    return pose_command_errors(env, command_name, robot_cfg, frame_cfg)[0]


def pose_command_orientation_error_vector(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Direct target-relative axis-angle orientation error in the robot root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    frame: FrameTransformer = env.scene[frame_cfg.name]
    command = env.command_manager.get_command(command_name)

    root_pos_w = as_torch(robot.data.root_pos_w)
    root_quat_w = as_torch(robot.data.root_quat_w)
    _, desired_quat_w = combine_frame_transforms(
        root_pos_w, root_quat_w, command[:, :3], command[:, 3:7]
    )
    current_quat_w = as_torch(frame.data.target_quat_w)[:, 0, :]
    orientation_error_w = quat_box_minus(desired_quat_w, current_quat_w)
    return quat_apply_inverse(root_quat_w, orientation_error_w)


def position_command_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Position distance retained as a compatible diagnostic term."""
    del asset_cfg
    return pose_command_errors(env, command_name, robot_cfg, frame_cfg)[1]


def orientation_command_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Shortest-path orientation error retained as a compatible diagnostic."""
    del asset_cfg
    return pose_command_errors(env, command_name, robot_cfg, frame_cfg)[2]


def position_command_tanh(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Bounded coarse-to-fine position tracking reward."""
    del asset_cfg
    distance = pose_command_errors(env, command_name, robot_cfg, frame_cfg)[1]
    return 1.0 - torch.tanh(distance / std)


def orientation_command_tanh(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Bounded orientation tracking reward."""
    del asset_cfg
    angle = pose_command_errors(env, command_name, robot_cfg, frame_cfg)[2]
    return 1.0 - torch.tanh(angle / std)


def successful_reach(
    env: ManagerBasedRLEnv,
    command_name: str,
    position_threshold: float,
    orientation_threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Whether the measured tool pose is inside the declared success envelope."""
    _, position_error, orientation_error = pose_command_errors(env, command_name, robot_cfg, frame_cfg)
    return (position_error < position_threshold) & (orientation_error < orientation_threshold)


def success_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    position_threshold: float,
    orientation_threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Sparse confirmation bonus layered over dense pose shaping."""
    return successful_reach(
        env,
        command_name,
        position_threshold,
        orientation_threshold,
        robot_cfg,
        frame_cfg,
    ).float()
