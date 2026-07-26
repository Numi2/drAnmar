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
    minimum_height: float = 0.06,
    presentation_distance: float = 0.05,
    required_receiver_only_steps: int = 10,
    command_name: str = "receiver_pose",
) -> dict[str, Any]:
    """Update and return the monotonic physical handover phase.

    Phases are: 0 Arm 1 approach, 1 Arm 1 grasp, 2 lifted presentation,
    3 dual grasp, 4 Arm 2-only ownership. Progress requires actual filtered
    PhysX contacts and cannot be earned by closing jaws near an object.
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
            "receiver_only_consecutive": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "last_step": -1,
        }
        setattr(env, "_dr_anmar_handover_state", state)
    obj = env.scene["object"]
    reset = (env.episode_length_buf == 0) & (state["last_reset_step"] != step)
    state["phase"][reset] = 0
    state["rewarded_phase"][reset] = 0
    state["receiver_only_consecutive"][reset] = 0
    object_pos_w = mdp_common.as_torch(obj.data.root_pos_w)
    state["start_object_pos"][reset] = object_pos_w[reset]
    state["last_reset_step"][reset] = step
    if state["last_step"] == step and not bool(torch.any(reset)):
        return state

    giver_contact = mdp_common.bilateral_contact(
        env, "robot_1_jaw_1_object_contact", "robot_1_jaw_2_object_contact", contact_threshold
    )
    receiver_contact = mdp_common.bilateral_contact(
        env, "robot_2_jaw_1_object_contact", "robot_2_jaw_2_object_contact", contact_threshold
    )
    receiver_frame: FrameTransformer = env.scene["ee_2_frame"]
    receiver_distance = torch.linalg.vector_norm(
        mdp_common.as_torch(receiver_frame.data.target_pos_w)[:, 0, :] - object_pos_w, dim=-1
    )
    lifted = object_pos_w[:, 2] > minimum_height
    pos_error, rot_error = mdp_common.object_goal_errors(
        env, command_name, SceneEntityCfg("robot_2"), SceneEntityCfg("object")
    )
    motion = mdp_common.object_motion(env)

    phase = state["phase"]
    phase[(phase == 0) & giver_contact] = 1
    phase[(phase == 1) & giver_contact & lifted & (receiver_distance < presentation_distance)] = 2
    phase[(phase == 2) & giver_contact & receiver_contact] = 3
    receiver_only = (
        (phase == 3)
        & receiver_contact
        & ~giver_contact
        & lifted
    )
    state["receiver_only_consecutive"][:] = torch.where(
        receiver_only,
        state["receiver_only_consecutive"] + 1,
        torch.zeros_like(state["receiver_only_consecutive"]),
    )
    phase[
        (phase == 3)
        & (state["receiver_only_consecutive"] >= required_receiver_only_steps)
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
