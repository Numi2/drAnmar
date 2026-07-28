# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from orbit.surgical.tasks.surgical import mdp_common

from .state import handover_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
    attribution_sensor_names: tuple[str, ...] = (),
) -> torch.Tensor:
    forces = torch.stack(
        [
            mdp_common.non_object_contact_force_magnitude(env, sensor_name)
            for sensor_name in sensor_names
        ],
        dim=-1,
    )
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
    if attribution_sensor_names:
        attribution_forces = torch.stack(
            [
                mdp_common.filtered_contact_force_magnitudes(
                    env,
                    sensor_name,
                )
                for sensor_name in attribution_sensor_names
            ],
            dim=1,
        )
        terminal_attribution = getattr(
            env,
            "_dr_anmar_terminal_protected_surface_attribution_forces_n",
            None,
        )
        if (
            terminal_attribution is None
            or terminal_attribution.shape != attribution_forces.shape
        ):
            terminal_attribution = torch.zeros_like(attribution_forces)
        terminal_attribution[violations] = attribution_forces[violations]
        env._dr_anmar_terminal_protected_surface_attribution_forces_n = (
            terminal_attribution
        )
    return violations
