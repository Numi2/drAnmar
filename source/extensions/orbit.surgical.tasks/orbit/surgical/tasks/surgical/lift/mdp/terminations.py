# Copyright (c) 2024-2026, The ORBIT-Surgical and Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from orbit.surgical.tasks.surgical import mdp_common
from .rewards import successful_lift  # noqa: F401

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
