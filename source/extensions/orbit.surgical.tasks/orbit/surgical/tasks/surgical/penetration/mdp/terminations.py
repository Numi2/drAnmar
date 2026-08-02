# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..contract import PunctureReceipt
from ..through_contract import ThroughPunctureReceipt
from ..pullout_contract import PulloutReceipt
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
        env._dr_anmar_last_hard_failure_evidence = [
            {
                "giver_distal_forces_n": [
                    float(value) for value in state["giver_tissue_forces"][index]
                ],
                "receiver_distal_forces_n": [
                    float(value) for value in state["receiver_tissue_forces"][index]
                ],
                "tip_position_m": [
                    float(value) for value in state["measurement"]["tip_pos"][index]
                ],
                "phase": int(state["phase"][index]),
            }
            for index in range(env.num_envs)
        ]
    return result


def successful_entry(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = penetration_state(env)
    result = state["success"]
    if torch.any(result):
        env._dr_anmar_last_successful_entry = [
            PunctureReceipt(
                schema="dr.anmar.tissue-entry-receipt.v1",
                success=True,
                event_count=gate.event_count,
                representation_switch_count=int(
                    state["representation_switch_count"][index]
                ),
                entry_position_m=tuple(
                    float(value) for value in state["measurement"]["tip_pos"][index]
                ),
                entry_error_m=float(state["measurement"]["entry_error"][index]),
                tangent_error_deg=float(state["measurement"]["tangent_error"][index]),
                plane_error_deg=float(state["measurement"]["plane_error"][index]),
                sampled_puncture_force_n=float(state["puncture_force_n"][index]),
                peak_force_n=gate.peak_force_n,
                accumulated_work_j=float(state["accumulated_work"][index]),
                embedded_depth_m=float(state["measurement"]["embedded_depth"][index]),
                phase_sequence=tuple(gate.phase_sequence),
                backend_revision=state["backend_metadata"].revision,
                backend_implementation_sha256=(
                    state["backend_metadata"].implementation_sha256
                ),
                hard_failures=tuple(sorted(gate.hard_failures)),
            ).as_dict()
            | {"phase": int(gate.phase), "custody_valid": bool(state["custody_valid"][index])}
            for index, gate in enumerate(state["gates"])
        ]
    return result


def successful_through_puncture(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = penetration_state(env)
    result = state["success"]
    if torch.any(result):
        env._dr_anmar_last_successful_through_puncture = [
            ThroughPunctureReceipt(
                schema="dr.anmar.tissue-through-puncture-receipt.v3",
                success=True,
                entry_event_count=gate.entry_event_count,
                exit_event_count=gate.exit_event_count,
                right_underside_event_count=int(
                    state["right_underside_event_count"][index]
                ),
                representation_switch_count=int(
                    state["representation_switch_count"][index]
                ),
                entry_error_m=float(gate.entry_error_at_puncture_m),
                exit_error_m=float(state["measurement"]["exit_error"][index]),
                tangent_error_deg=float(gate.tangent_error_at_puncture_deg),
                plane_error_deg=float(gate.plane_error_at_puncture_deg),
                sampled_puncture_force_n=float(state["puncture_force_n"][index]),
                peak_force_n=gate.peak_force_n,
                accumulated_work_j=float(state["accumulated_work"][index]),
                embedded_arc_length_m=float(
                    state["measurement"]["embedded_arc_length"][index]
                ),
                exposed_arc_length_m=float(
                    state["measurement"]["exposed_arc_length"][index]
                ),
                exposed_fraction=float(
                    state["measurement"]["exposed_fraction"][index]
                ),
                phase_sequence=tuple(gate.phase_sequence),
                backend_revision=state["backend_metadata"].revision,
                backend_implementation_sha256=(
                    state["backend_metadata"].implementation_sha256
                ),
                hard_failures=tuple(sorted(gate.hard_failures)),
                entry_slab=state["measurement"]["entry_slab"][index],
                exit_slab=state["measurement"]["exit_slab"][index],
                cross_slab_route_valid=bool(
                    state["measurement"]["cross_slab_route_valid"][index]
                ),
            ).as_dict()
            | {
                "phase": int(gate.phase),
                "custody_valid": bool(state["custody_valid"][index]),
            }
            if bool(result[index])
            else {"success": False}
            for index, gate in enumerate(state["gates"])
        ]
    return result


def successful_pullout(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = penetration_state(env)
    result = state["success"]
    if torch.any(result):
        env._dr_anmar_last_successful_pullout = [
            PulloutReceipt(
                schema="dr.anmar.tissue-puncture-pullout-receipt.v3",
                success=True,
                entry_event_count=gate.entry_event_count,
                exit_event_count=gate.exit_event_count,
                right_underside_event_count=int(
                    state["right_underside_event_count"][index]
                ),
                representation_switch_count=int(
                    state["representation_switch_count"][index]
                ),
                entry_error_m=float(gate.entry_error_at_puncture_m),
                exit_error_m=float(gate.exit_error_at_event_m),
                tangent_error_deg=float(gate.tangent_error_at_puncture_deg),
                plane_error_deg=float(gate.plane_error_at_puncture_deg),
                sampled_puncture_force_n=float(state["puncture_force_n"][index]),
                peak_force_n=gate.peak_force_n,
                accumulated_work_j=float(state["accumulated_work"][index]),
                exposed_fraction=float(
                    state["measurement"]["exposed_fraction"][index]
                ),
                embedded_arc_length_m=float(
                    state["measurement"]["embedded_arc_length"][index]
                ),
                exposed_arc_length_m=float(
                    state["measurement"]["exposed_arc_length"][index]
                ),
                receiver_contact_steps=gate.receiver_contact_steps,
                receiver_pull_steps=gate.receiver_pull_steps,
                receiver_curve_rotation_deg=float(
                    state["receiver_curve_rotation_deg"][index]
                ),
                receiver_curve_center_error_m=float(
                    state["receiver_curve_center_error"][index]
                ),
                receiver_only_clearance_steps=gate.cleared_steps,
                phase_sequence=tuple(gate.phase_sequence),
                backend_revision=state["backend_metadata"].revision,
                backend_implementation_sha256=(
                    state["backend_metadata"].implementation_sha256
                ),
                hard_failures=tuple(sorted(gate.hard_failures)),
                entry_slab=state["measurement"]["entry_slab"][index],
                exit_slab=state["measurement"]["exit_slab"][index],
                cross_slab_route_valid=bool(
                    state["measurement"]["cross_slab_route_valid"][index]
                ),
                tract_support_event_count=int(
                    state["tract_support_event_count"][index]
                ),
                giver_regrasp_complete=bool(
                    state["giver_regrasp_complete"][index]
                ),
            ).as_dict()
            if bool(result[index])
            else {"success": False}
            for index, gate in enumerate(state["gates"])
        ]
    return result
