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
    return handover_state(env)["phase"] >= 4


def premature_giver_release(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Fail when Arm 1 loses custody before Arm 2 physically acquires it."""
    return handover_state(env)["premature_release"]


def receiver_retention_lost(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Fail when Arm 2 loses the needle during its retention check."""
    return handover_state(env)["receiver_retention_failed"]


def needle_dropped_after_pickup(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Fail when a previously lifted needle returns to the support surface."""
    return handover_state(env)["needle_dropped"]


def excessive_contact_force(
    env: ManagerBasedRLEnv, sensor_names: tuple[str, ...], hard_limit: float
) -> torch.Tensor:
    return mdp_common.maximum_contact_force(env, sensor_names) > hard_limit


def excessive_non_object_contact_force(
    env: ManagerBasedRLEnv, sensor_names: tuple[str, ...], hard_limit: float
) -> torch.Tensor:
    return mdp_common.maximum_non_object_contact_force(env, sensor_names) > hard_limit
