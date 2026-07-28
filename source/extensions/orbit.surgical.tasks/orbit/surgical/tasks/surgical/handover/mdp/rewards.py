# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from orbit.surgical.tasks.surgical import mdp_common

from .state import handover_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def end_effector_object_distance(
    env: ManagerBasedRLEnv, std: float, frame_name: str, minimum_phase: int = 0
) -> torch.Tensor:
    obj: RigidObject = env.scene["object"]
    frame: FrameTransformer = env.scene[frame_name]
    distance = torch.linalg.vector_norm(
        mdp_common.as_torch(frame.data.target_pos_w)[:, 0, :]
        - mdp_common.as_torch(obj.data.root_pos_w),
        dim=-1,
    )
    active = handover_state(env)["phase"] >= minimum_phase
    return active.float() * (1.0 - torch.tanh(distance / std))


def role_end_effector_object_distance(
    env: ManagerBasedRLEnv,
    std: float,
    role: str,
    minimum_phase: int = 0,
) -> torch.Tensor:
    """Distance shaping for the reset-selected giver or receiver."""
    if role not in {"giver", "receiver"}:
        raise ValueError("role must be giver or receiver")
    state = handover_state(env)
    robot_1_position = mdp_common.as_torch(
        env.scene["ee_1_frame"].data.target_pos_w
    )[:, 0, :]
    robot_2_position = mdp_common.as_torch(
        env.scene["ee_2_frame"].data.target_pos_w
    )[:, 0, :]
    use_robot_1 = state["giver_is_robot_1"]
    if role == "receiver":
        use_robot_1 = ~use_robot_1
    role_position = torch.where(
        use_robot_1.unsqueeze(-1),
        robot_1_position,
        robot_2_position,
    )
    obj: RigidObject = env.scene["object"]
    distance = torch.linalg.vector_norm(
        role_position - mdp_common.as_torch(obj.data.root_pos_w),
        dim=-1,
    )
    active = state["phase"] >= minimum_phase
    return active.float() * (1.0 - torch.tanh(distance / std))


def bilateral_grasp(
    env: ManagerBasedRLEnv, sensor_1_name: str, sensor_2_name: str, threshold: float, minimum_phase: int = 0
) -> torch.Tensor:
    contact = mdp_common.bilateral_contact(env, sensor_1_name, sensor_2_name, threshold)
    active = handover_state(env)["phase"] >= minimum_phase
    return (contact & active).float()


def role_bilateral_grasp(
    env: ManagerBasedRLEnv,
    role: str,
    minimum_phase: int = 0,
) -> torch.Tensor:
    """Bilateral grasp shaping for the reset-selected logical role."""
    if role not in {"giver", "receiver"}:
        raise ValueError("role must be giver or receiver")
    state = handover_state(env)
    contact = state[f"{role}_contact"]
    active = state["phase"] >= minimum_phase
    return (contact & active).float()


def phase_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Award each physically completed phase once per episode."""
    state = handover_state(env)
    delta = torch.clamp(
        state["progress_phase"] - state["rewarded_phase"],
        min=0,
    )
    state["rewarded_phase"] = torch.maximum(
        state["rewarded_phase"],
        state["progress_phase"],
    )
    return delta.float()


def potential_based_handover_progress(
    env: ManagerBasedRLEnv,
    gamma: float = 0.995,
) -> torch.Tensor:
    """Dense, bounded progress that cannot change the terminal optimum.

    The returned reward is ``gamma * Phi(s_next) - Phi(s)``.  Potential is
    zeroed at every physical success/failure/timeout terminal, so partial
    approach, lift, or contact cannot retain positive episodic credit after a
    failed outcome.
    """
    if not 0.0 < gamma <= 1.0:
        raise ValueError("potential-shaping gamma must be in (0, 1]")
    state = handover_state(env)
    progress_phase = state["progress_phase"].clamp(0, 4)
    phase = state["phase"]
    approach_progress = 1.0 - torch.tanh(
        state["giver_distance"] / 0.03
    )
    lift_progress = 0.5 * (
        state["clearance"] / 0.01
    ).clamp(0.0, 1.0) + 0.5 * state["giver_custody"].float()
    transfer_progress = 0.5 * (
        1.0 - torch.tanh(state["presentation_error"] / 0.02)
    ) + 0.5 * (
        1.0 - torch.tanh(state["receiver_distance"] / 0.03)
    )
    retention_progress = (
        state["receiver_only_consecutive"].float() / 10.0
    ).clamp(0.0, 1.0)
    stage_progress = torch.where(
        phase == 0,
        approach_progress,
        torch.where(
            phase == 1,
            lift_progress,
            torch.where(
                phase == 2,
                transfer_progress,
                torch.where(
                    phase == 3,
                    retention_progress,
                    torch.zeros_like(approach_progress),
                ),
            ),
        ),
    )
    potential = (
        progress_phase.float() + stage_progress
    ) / 5.0
    timeout = env.episode_length_buf >= (
        env.max_episode_length - 1
    )
    terminal = (
        state["successful_handover"]
        | state["pickup_attempts_exhausted"]
        | state["premature_release"]
        | state["receiver_retention_failed"]
        | state["needle_dropped"]
        | timeout
    )
    next_potential = torch.where(
        terminal,
        torch.zeros_like(potential),
        potential,
    )
    previous = getattr(
        env,
        "_dr_anmar_previous_handover_potential",
        None,
    )
    if previous is None or previous.shape != potential.shape:
        previous = potential.detach().clone()
        env._dr_anmar_previous_handover_potential = previous
    previous_reset_step = getattr(
        env,
        "_dr_anmar_potential_last_reset_step",
        None,
    )
    if (
        previous_reset_step is None
        or previous_reset_step.shape != state["last_reset_step"].shape
    ):
        previous_reset_step = torch.full_like(
            state["last_reset_step"],
            -2,
        )
        env._dr_anmar_potential_last_reset_step = (
            previous_reset_step
        )
    reset = (
        (env.episode_length_buf == 0)
        | (previous_reset_step != state["last_reset_step"])
    )
    shaped = gamma * next_potential - previous
    shaped = torch.where(
        reset,
        torch.zeros_like(shaped),
        shaped,
    )
    previous[:] = torch.where(
        reset,
        potential.detach(),
        next_potential.detach(),
    )
    previous_reset_step[:] = state["last_reset_step"]
    return shaped


def receiver_goal_tracking(env: ManagerBasedRLEnv, position_std: float, orientation_std: float) -> torch.Tensor:
    state = handover_state(env)
    active = state["phase"] >= 3
    return active.float() * (1.0 - torch.tanh(state["position_error"] / position_std)) * (
        1.0 - torch.tanh(state["orientation_error"] / orientation_std)
    )


def stable_dual_grasp(env: ManagerBasedRLEnv, linear_std: float, angular_std: float) -> torch.Tensor:
    state = handover_state(env)
    dual = state["giver_contact"] & state["receiver_contact"]
    return dual.float() * (1.0 - torch.tanh(state["motion"][:, 0] / linear_std)) * (
        1.0 - torch.tanh(state["motion"][:, 1] / angular_std)
    )


def successful_handover(env: ManagerBasedRLEnv) -> torch.Tensor:
    return handover_state(env)["successful_handover"].float()


def contact_force_excess(
    env: ManagerBasedRLEnv, sensor_names: tuple[str, ...], soft_limit: float
) -> torch.Tensor:
    return mdp_common.contact_force_excess(env, sensor_names, soft_limit)


def non_object_contact_force_excess(
    env: ManagerBasedRLEnv, sensor_names: tuple[str, ...], soft_limit: float
) -> torch.Tensor:
    return mdp_common.non_object_contact_force_excess(env, sensor_names, soft_limit)


def rcm_motion(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    return torch.square(mdp_common.rcm_linear_speed(env, robot_cfg))
