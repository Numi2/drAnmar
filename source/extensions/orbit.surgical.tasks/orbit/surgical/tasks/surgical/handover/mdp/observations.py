# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as functional

from isaaclab.managers import SceneEntityCfg

from orbit.surgical.tasks.surgical import mdp_common

from .state import handover_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    return mdp_common.object_pose_in_robot_root_frame(env, robot_cfg, object_cfg)


def object_velocity_in_robot_root_frame(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    return mdp_common.object_velocity_in_robot_root_frame(env, robot_cfg, object_cfg)


def end_effector_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv, frame_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg
) -> torch.Tensor:
    return mdp_common.end_effector_pose_in_robot_root_frame(env, frame_cfg, robot_cfg)


def jaw_contact_forces(
    env: ManagerBasedRLEnv, sensor_1_name: str, sensor_2_name: str, scale: float = 1.0
) -> torch.Tensor:
    return mdp_common.paired_contact_forces(env, sensor_1_name, sensor_2_name, scale)


def handover_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One-hot physical phase: approach, grasp, present, dual grasp, recovery."""
    return functional.one_hot(handover_state(env)["phase"], num_classes=5).float()
