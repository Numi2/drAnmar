# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# Copyright (c) 2026, Dr.Anmar Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared pose-reaching terms for Dr.Anmar dual-arm learning."""

from isaaclab.managers import SceneEntityCfg

from ...reach.mdp.rewards import (  # noqa: F401
    orientation_command_error,
    orientation_command_tanh,
    pose_command_error_vector,
    pose_command_orientation_error_vector,
    pose_command_errors,
    position_command_error,
    position_command_tanh,
    success_bonus,
    successful_reach,
)


def successful_dual_reach(
    env,
    command_1_name,
    command_2_name,
    position_threshold,
    orientation_threshold,
):
    """Require both tools to satisfy their pose envelopes simultaneously."""
    first = successful_reach(
        env,
        command_1_name,
        position_threshold,
        orientation_threshold,
        robot_cfg=SceneEntityCfg("robot_1"),
        frame_cfg=SceneEntityCfg("ee_1_frame"),
    )
    second = successful_reach(
        env,
        command_2_name,
        position_threshold,
        orientation_threshold,
        robot_cfg=SceneEntityCfg("robot_2"),
        frame_cfg=SceneEntityCfg("ee_2_frame"),
    )
    return first & second
