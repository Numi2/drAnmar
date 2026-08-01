# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .state import penetration_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def hard_safety_failure(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = penetration_state(env)
    result = state["hard_failure"]
    if torch.any(result):
        env._dr_anmar_last_hard_failures = [
            tuple(sorted(gate.hard_failures)) for gate in state["gates"]
        ]
    return result


def successful_entry(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = penetration_state(env)
    result = state["success"]
    if torch.any(result):
        env._dr_anmar_last_successful_entry = [
            {
                "event_count": gate.event_count,
                "representation_switch_count": int(
                    state["representation_switch_count"][index]
                ),
                "phase": int(gate.phase),
                "hard_failures": tuple(sorted(gate.hard_failures)),
                "entry_error_m": float(state["measurement"]["entry_error"][index]),
                "tangent_error_deg": float(state["measurement"]["tangent_error"][index]),
                "plane_error_deg": float(state["measurement"]["plane_error"][index]),
                "embedded_depth_m": float(state["measurement"]["embedded_depth"][index]),
                "peak_force_n": gate.peak_force_n,
                "backend_revision": state["backend_metadata"].revision,
                "backend_implementation_sha256": (
                    state["backend_metadata"].implementation_sha256
                ),
            }
            for index, gate in enumerate(state["gates"])
        ]
    return result
