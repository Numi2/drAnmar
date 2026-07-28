# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from orbit.surgical.tasks.surgical import mdp_common

from .state import handover_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_PROTECTED_ATTRIBUTION_BODY_NAMES = (
    "psm_main_insertion_link_3",
    "psm_tool_roll_link",
    "psm_tool_pitch_link",
    "psm_tool_yaw_link",
    "psm_tool_gripper1_link",
    "psm_tool_gripper2_link",
)


def _protected_attribution_body_poses(
    env: ManagerBasedRLEnv,
    robot_name: str,
) -> torch.Tensor:
    robot = env.scene[robot_name]
    cache = getattr(
        env,
        "_dr_anmar_protected_attribution_body_ids",
        None,
    )
    if cache is None:
        cache = {}
        env._dr_anmar_protected_attribution_body_ids = cache
    body_ids = cache.get(robot_name)
    if body_ids is None:
        body_ids = tuple(
            robot.body_names.index(body_name)
            for body_name in _PROTECTED_ATTRIBUTION_BODY_NAMES
        )
        cache[robot_name] = body_ids
    positions = mdp_common.as_torch(robot.data.body_pos_w)[:, body_ids, :]
    orientations = mdp_common.as_torch(robot.data.body_quat_w)[:, body_ids, :]
    return torch.cat((positions, orientations), dim=-1)


def successful_handover(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate after ten control steps of receiver-only needle ownership."""
    return handover_state(env)["successful_handover"]


def pickup_attempts_exhausted(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Fail only after the third physical pickup attempt loses custody."""
    return handover_state(env)["pickup_attempts_exhausted"]


def premature_giver_release(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Fail when the selected giver loses custody before receiver acquisition."""
    return handover_state(env)["premature_release"]


def receiver_retention_lost(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Fail when the selected receiver loses the needle during retention."""
    return handover_state(env)["receiver_retention_failed"]


def needle_dropped_after_pickup(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Fail when a previously lifted needle returns to the support surface."""
    return handover_state(env)["needle_dropped"]


def excessive_contact_force(
    env: ManagerBasedRLEnv, sensor_names: tuple[str, ...], hard_limit: float
) -> torch.Tensor:
    return mdp_common.maximum_contact_force(env, sensor_names) > hard_limit


def excessive_non_object_contact_force(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...],
    hard_limit: float,
) -> torch.Tensor:
    force_vectors = torch.stack(
        [
            mdp_common.non_object_contact_force_vector(env, sensor_name)
            for sensor_name in sensor_names
        ],
        dim=1,
    )
    forces = torch.linalg.vector_norm(force_vectors, dim=-1)
    violations = forces.amax(dim=-1) > hard_limit
    # Isaac Lab resets terminal environments inside ``env.step``. Preserve the
    # exact pre-reset force vector so held-out evidence can attribute a safety
    # terminal to the responsible tool and jaw without affecting control.
    terminal_forces = getattr(
        env,
        "_dr_anmar_terminal_protected_surface_forces_n",
        None,
    )
    if terminal_forces is None or terminal_forces.shape != forces.shape:
        terminal_forces = torch.zeros_like(forces)
    terminal_forces[violations] = forces[violations]
    env._dr_anmar_terminal_protected_surface_forces_n = terminal_forces
    terminal_force_vectors = getattr(
        env,
        "_dr_anmar_terminal_protected_surface_force_vectors_w_n",
        None,
    )
    if (
        terminal_force_vectors is None
        or terminal_force_vectors.shape != force_vectors.shape
    ):
        terminal_force_vectors = torch.zeros_like(force_vectors)
    terminal_force_vectors[violations] = force_vectors[violations]
    env._dr_anmar_terminal_protected_surface_force_vectors_w_n = (
        terminal_force_vectors
    )
    for robot_name in ("robot_1", "robot_2"):
        body_poses = _protected_attribution_body_poses(env, robot_name)
        attribute = (
            f"_dr_anmar_terminal_protected_surface_{robot_name}_body_poses_w"
        )
        terminal_body_poses = getattr(env, attribute, None)
        if (
            terminal_body_poses is None
            or terminal_body_poses.shape != body_poses.shape
        ):
            terminal_body_poses = torch.zeros_like(body_poses)
        terminal_body_poses[violations] = body_poses[violations]
        setattr(env, attribute, terminal_body_poses)
    object_position = mdp_common.as_torch(
        env.scene["object"].data.root_pos_w
    )
    terminal_object_position = getattr(
        env,
        "_dr_anmar_terminal_protected_surface_object_position_w",
        None,
    )
    if (
        terminal_object_position is None
        or terminal_object_position.shape != object_position.shape
    ):
        terminal_object_position = torch.zeros_like(object_position)
    terminal_object_position[violations] = object_position[violations]
    env._dr_anmar_terminal_protected_surface_object_position_w = (
        terminal_object_position
    )
    return violations
