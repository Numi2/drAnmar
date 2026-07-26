# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-grounded phase state for physical two-instrument handover."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from orbit.surgical.tasks.surgical import mdp_common

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _step_number(env: ManagerBasedRLEnv) -> int:
    value = env.common_step_counter
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def handover_state(
    env: ManagerBasedRLEnv,
    contact_threshold: float = 0.01,
    minimum_lift: float = 0.01,
    presentation_distance: float = 0.05,
    goal_position_threshold: float = 0.025,
    goal_orientation_threshold: float = 0.5,
    maximum_linear_speed: float = 0.1,
    maximum_angular_speed: float = 2.0,
    command_name: str = "ee_1_pose",
) -> dict[str, Any]:
    """Update and return the monotonic physical handover phase.

    Phases are: 0 approach, 1 giver grasp, 2 present, 3 dual grasp,
    4 receiver owns a stable object at its commanded recovery pose. Progress
    requires actual filtered PhysX contacts and cannot be earned by closing jaws
    near an object.
    """
    step = _step_number(env)
    state = getattr(env, "_dr_anmar_handover_state", None)
    if state is None:
        obj: RigidObject = env.scene["object"]
        state = {
            "phase": torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            "rewarded_phase": torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            "start_object_pos": mdp_common.as_torch(obj.data.root_pos_w).clone(),
            "last_reset_step": torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device),
            "last_step": -1,
        }
        setattr(env, "_dr_anmar_handover_state", state)
    obj = env.scene["object"]
    reset = (env.episode_length_buf == 0) & (state["last_reset_step"] != step)
    state["phase"][reset] = 0
    state["rewarded_phase"][reset] = 0
    object_pos_w = mdp_common.as_torch(obj.data.root_pos_w)
    state["start_object_pos"][reset] = object_pos_w[reset]
    state["last_reset_step"][reset] = step
    if state["last_step"] == step and not bool(torch.any(reset)):
        return state

    giver_contact = mdp_common.bilateral_contact(
        env, "robot_2_jaw_1_object_contact", "robot_2_jaw_2_object_contact", contact_threshold
    )
    receiver_contact = mdp_common.bilateral_contact(
        env, "robot_1_jaw_1_object_contact", "robot_1_jaw_2_object_contact", contact_threshold
    )
    receiver_frame: FrameTransformer = env.scene["ee_1_frame"]
    receiver_distance = torch.linalg.vector_norm(
        mdp_common.as_torch(receiver_frame.data.target_pos_w)[:, 0, :] - object_pos_w, dim=-1
    )
    lifted = (object_pos_w[:, 2] - state["start_object_pos"][:, 2]) > minimum_lift
    pos_error, rot_error = mdp_common.object_goal_errors(
        env, command_name, SceneEntityCfg("robot_1"), SceneEntityCfg("object")
    )
    motion = mdp_common.object_motion(env)
    stable = (motion[:, 0] < maximum_linear_speed) & (motion[:, 1] < maximum_angular_speed)

    phase = state["phase"]
    phase[(phase == 0) & giver_contact] = 1
    phase[(phase == 1) & giver_contact & lifted & (receiver_distance < presentation_distance)] = 2
    phase[(phase == 2) & giver_contact & receiver_contact] = 3
    phase[
        (phase == 3)
        & receiver_contact
        & ~giver_contact
        & stable
        & (pos_error < goal_position_threshold)
        & (rot_error < goal_orientation_threshold)
    ] = 4

    state.update(
        {
            "last_step": step,
            "giver_contact": giver_contact,
            "receiver_contact": receiver_contact,
            "receiver_distance": receiver_distance,
            "lifted": lifted,
            "position_error": pos_error,
            "orientation_error": rot_error,
            "motion": motion,
        }
    )
    return state
