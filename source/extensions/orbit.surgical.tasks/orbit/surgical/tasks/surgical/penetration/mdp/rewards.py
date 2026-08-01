# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from orbit.surgical.tasks.surgical import mdp_common

from .state import penetration_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def phase_transition_once(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = penetration_state(env)
    rewarded = state.get("rewarded_phase")
    if rewarded is None:
        rewarded = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    reward = (state["phase"] > rewarded).float()
    state["rewarded_phase"] = torch.maximum(rewarded, state["phase"])
    return reward


def bounded_entry_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = penetration_state(env)
    error = state["measurement"]["entry_error"]
    previous = state.get("previous_entry_error", error)
    state["previous_entry_error"] = error.clone()
    return ((previous - error) / 0.001).clamp(-0.1, 0.1)


def successful_entry(env: ManagerBasedRLEnv) -> torch.Tensor:
    return penetration_state(env)["success"].float()


def bounded_exit_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = penetration_state(env)
    exposure = state["measurement"]["exposed_fraction"]
    previous = state.get("previous_exposed_fraction", exposure)
    state["previous_exposed_fraction"] = exposure.clone()
    return (previous.neg() + exposure).clamp(-0.02, 0.02)


def successful_through_puncture(env: ManagerBasedRLEnv) -> torch.Tensor:
    return penetration_state(env)["success"].float()


def bounded_receiver_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = penetration_state(env)
    distance = state["receiver_distance"]
    previous = state.get("previous_receiver_distance", distance)
    state["previous_receiver_distance"] = distance.clone()
    active = (state["phase"] >= 7) & (state["phase"] <= 9)
    return ((previous - distance) / 0.001).clamp(-0.1, 0.1) * active


def bounded_clearance_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = penetration_state(env)
    embedded = state["measurement"]["embedded_arc_length"]
    previous = state.get("previous_embedded_arc_length", embedded)
    state["previous_embedded_arc_length"] = embedded.clone()
    return ((previous - embedded) / 0.001).clamp(-0.1, 0.1) * (
        state["phase"] >= 10
    )


def successful_pullout(env: ManagerBasedRLEnv) -> torch.Tensor:
    return penetration_state(env)["success"].float()


def normalized_force_overshoot(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = penetration_state(env)
    normalized = state["normal_force"] / state["puncture_force_n"].clamp_min(1.0e-6)
    prepuncture = state["event_count"] == 0
    return torch.square(torch.relu(normalized - 1.0)) * prepuncture


def lateral_slip(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = penetration_state(env)
    tip = state["measurement"]["tip_pos"]
    previous = state.get("previous_tip_pos", tip)
    state["previous_tip_pos"] = tip.clone()
    active = state["phase"] >= 2
    return torch.linalg.vector_norm((tip - previous)[:, :2], dim=-1) * active / 0.00025


def rcm_motion(env: ManagerBasedRLEnv) -> torch.Tensor:
    from isaaclab.managers import SceneEntityCfg

    return mdp_common.rcm_linear_speed(
        env, SceneEntityCfg("robot", body_names="psm_remote_center_link")
    )
