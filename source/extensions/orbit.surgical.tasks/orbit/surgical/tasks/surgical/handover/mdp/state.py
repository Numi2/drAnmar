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
    pickup_clearance: float = 0.01,
    contact_window_steps: int = 5,
    contact_required_steps: int = 3,
    required_receiver_only_steps: int = 10,
    allowed_receiver_contact_flicker_steps: int = 1,
    receiver_follow_tolerance: float = 0.005,
    command_name: str = "receiver_pose",
) -> dict[str, Any]:
    """Update and return the monotonic physical handover phase.

    Phases are: 0 Arm 1 approach, 1 Arm 1 grasp, 2 lifted presentation,
    3 receiver acquisition, 4 Arm 2-only ownership. Pickup and acquisition
    accept bilateral PhysX contact in three of five control steps. Retention
    permits one missing contact frame only while the elevated needle preserves
    its receiver-relative offset.
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
            "giver_contact_history": torch.zeros(
                (env.num_envs, contact_window_steps),
                dtype=torch.bool,
                device=env.device,
            ),
            "receiver_contact_history": torch.zeros(
                (env.num_envs, contact_window_steps),
                dtype=torch.bool,
                device=env.device,
            ),
            "receiver_loss_consecutive": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "giver_release_observed": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "premature_release": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "receiver_retention_failed": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "receiver_acquisition_offset_w": torch.zeros(
                (env.num_envs, 3), dtype=torch.float32, device=env.device
            ),
            "last_step": -1,
        }
        setattr(env, "_dr_anmar_handover_state", state)
    obj = env.scene["object"]
    reset = (env.episode_length_buf == 0) & (state["last_reset_step"] != step)
    state["phase"][reset] = 0
    state["rewarded_phase"][reset] = 0
    state["receiver_only_consecutive"][reset] = 0
    state["giver_contact_history"][reset] = False
    state["receiver_contact_history"][reset] = False
    state["receiver_loss_consecutive"][reset] = 0
    state["giver_release_observed"][reset] = False
    state["premature_release"][reset] = False
    state["receiver_retention_failed"][reset] = False
    state["receiver_acquisition_offset_w"][reset] = 0.0
    object_pos_w = mdp_common.as_torch(obj.data.root_pos_w)
    state["start_object_pos"][reset] = object_pos_w[reset]
    state["last_reset_step"][reset] = step
    if state["last_step"] == step and not bool(torch.any(reset)):
        return state

    giver_contact_now = mdp_common.bilateral_contact(
        env, "robot_1_jaw_1_object_contact", "robot_1_jaw_2_object_contact", contact_threshold
    )
    receiver_contact_now = mdp_common.bilateral_contact(
        env, "robot_2_jaw_1_object_contact", "robot_2_jaw_2_object_contact", contact_threshold
    )
    state["giver_contact_history"] = torch.roll(
        state["giver_contact_history"], shifts=-1, dims=-1
    )
    state["receiver_contact_history"] = torch.roll(
        state["receiver_contact_history"], shifts=-1, dims=-1
    )
    state["giver_contact_history"][:, -1] = giver_contact_now
    state["receiver_contact_history"][:, -1] = receiver_contact_now
    giver_contact = (
        state["giver_contact_history"].sum(dim=-1) >= contact_required_steps
    )
    receiver_contact = (
        state["receiver_contact_history"].sum(dim=-1) >= contact_required_steps
    )
    receiver_frame: FrameTransformer = env.scene["ee_2_frame"]
    receiver_position_w = mdp_common.as_torch(
        receiver_frame.data.target_pos_w
    )[:, 0, :]
    receiver_distance = torch.linalg.vector_norm(
        receiver_position_w - object_pos_w, dim=-1
    )
    clearance = object_pos_w[:, 2] - state["start_object_pos"][:, 2]
    lifted = clearance >= pickup_clearance
    pos_error, rot_error = mdp_common.object_goal_errors(
        env, command_name, SceneEntityCfg("robot_2"), SceneEntityCfg("object")
    )
    motion = mdp_common.object_motion(env)

    phase = state["phase"]
    phase[(phase == 0) & giver_contact] = 1
    phase[(phase == 1) & giver_contact & lifted] = 2
    before_acquisition = (phase >= 1) & (phase < 3)
    giver_gripper_action = mdp_common.as_torch(
        env.action_manager.action
    )
    state["premature_release"] |= (
        before_acquisition
        & (giver_gripper_action[:, 6] > 0.0)
    )
    receiver_acquired = (
        (phase == 2) & giver_contact & receiver_contact
    )
    state["receiver_acquisition_offset_w"][receiver_acquired] = (
        object_pos_w[receiver_acquired]
        - receiver_position_w[receiver_acquired]
    )
    phase[receiver_acquired] = 3
    state["giver_release_observed"] |= (
        (phase == 3) & ~giver_contact_now
    )
    receiver_relative_offset = object_pos_w - receiver_position_w
    receiver_follow_error = torch.linalg.vector_norm(
        receiver_relative_offset - state["receiver_acquisition_offset_w"],
        dim=-1,
    )
    receiver_follows = receiver_follow_error <= receiver_follow_tolerance
    retention_active = (
        (phase == 3)
        & state["giver_release_observed"]
    )
    state["receiver_loss_consecutive"][:] = torch.where(
        retention_active & ~receiver_contact_now,
        state["receiver_loss_consecutive"] + 1,
        torch.zeros_like(state["receiver_loss_consecutive"]),
    )
    receiver_flicker_allowed = (
        state["receiver_loss_consecutive"]
        <= allowed_receiver_contact_flicker_steps
    )
    receiver_only = (
        retention_active
        & lifted
        & receiver_follows
        & (receiver_contact_now | receiver_flicker_allowed)
    )
    state["receiver_retention_failed"] |= (
        retention_active
        & (
            ~lifted
            | ~receiver_follows
            | (
                state["receiver_loss_consecutive"]
                > allowed_receiver_contact_flicker_steps
            )
        )
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
            "giver_contact_now": giver_contact_now,
            "receiver_contact_now": receiver_contact_now,
            "receiver_distance": receiver_distance,
            "clearance": clearance,
            "lifted": lifted,
            "receiver_follows": receiver_follows,
            "receiver_follow_error": receiver_follow_error,
            "needle_dropped": (phase >= 2) & (clearance < 0.005),
            "position_error": pos_error,
            "orientation_error": rot_error,
            "motion": motion,
        }
    )
    return state
