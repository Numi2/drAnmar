# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Physics-owned contract for a complete curved-needle tissue passage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from math import isfinite
from typing import Any


class ThroughPuncturePhase(IntEnum):
    APPROACH = 0
    ALIGN = 1
    INDENT = 2
    PUNCTURE = 3
    DRIVE = 4
    EXIT = 5
    PRESENT = 6


@dataclass(frozen=True)
class ThroughPunctureThresholds:
    approach_radius_m: float = 0.008
    align_radius_m: float = 0.002
    entry_tolerance_m: float = 0.001
    exit_tolerance_m: float = 0.001
    tangent_error_limit_deg: float = 10.0
    plane_error_limit_deg: float = 10.0
    prepuncture_depth_m: float = 0.0015
    exposed_fraction_min: float = 0.20
    exposed_fraction_max: float = 0.30
    presentation_steps: int = 10
    force_overshoot_fraction: float = 0.25


@dataclass(frozen=True)
class ThroughPunctureMeasurement:
    entry_error_m: float
    exit_error_m: float
    tangent_error_deg: float
    plane_error_deg: float
    indentation_m: float
    embedded_arc_length_m: float
    exposed_arc_length_m: float
    exposed_fraction: float
    normal_force_n: float
    accumulated_work_j: float
    bilateral_custody: bool
    target_region_valid: bool
    tissue_contact: bool
    solver_finite: bool = True
    unintended_jaw_contact: bool = False
    unintended_robot_contact: bool = False
    unintended_surface_crossing: bool = False
    backend_exit_count: int = 0
    backend_right_underside_count: int = 0
    entry_slab: str = "none"
    exit_slab: str = "none"
    cross_slab_route_valid: bool = False
    invalid_exit_route: bool = False
    missing_right_underside_puncture: bool = False


@dataclass
class ThroughPunctureGateState:
    phase: ThroughPuncturePhase = ThroughPuncturePhase.APPROACH
    entry_event_count: int = 0
    exit_event_count: int = 0
    presented_steps: int = 0
    peak_force_n: float = 0.0
    entry_error_at_puncture_m: float | None = None
    tangent_error_at_puncture_deg: float | None = None
    plane_error_at_puncture_deg: float | None = None
    hard_failures: set[str] = field(default_factory=set)
    phase_sequence: list[str] = field(default_factory=lambda: ["approach"])

    @property
    def punctured(self) -> bool:
        return self.entry_event_count == 1

    @property
    def failed(self) -> bool:
        return bool(self.hard_failures)


def advance_through_puncture_gate(
    state: ThroughPunctureGateState,
    measurement: ThroughPunctureMeasurement,
    *,
    puncture_force_n: float,
    thresholds: ThroughPunctureThresholds = ThroughPunctureThresholds(),
) -> ThroughPunctureGateState:
    """Advance a monotonic entry, drive, exit, and presentation state machine."""

    values = (
        measurement.entry_error_m,
        measurement.exit_error_m,
        measurement.tangent_error_deg,
        measurement.plane_error_deg,
        measurement.indentation_m,
        measurement.embedded_arc_length_m,
        measurement.exposed_arc_length_m,
        measurement.exposed_fraction,
        measurement.normal_force_n,
        measurement.accumulated_work_j,
        puncture_force_n,
    )
    if not measurement.solver_finite or not all(isfinite(value) for value in values):
        state.hard_failures.add("nonfinite_solver_state")
        return state
    if not measurement.bilateral_custody:
        state.hard_failures.add("grasp_loss")
    if measurement.unintended_jaw_contact:
        state.hard_failures.add("unintended_jaw_tissue_contact")
    if measurement.unintended_robot_contact:
        state.hard_failures.add("unintended_robot_tissue_contact")
    if measurement.unintended_surface_crossing:
        state.hard_failures.add("unintended_surface_crossing")
    if (
        not state.punctured
        and measurement.tissue_contact
        and not measurement.target_region_valid
    ):
        state.hard_failures.add("off_target_contact")
    if (
        not state.punctured
        and measurement.normal_force_n
        > puncture_force_n * (1.0 + thresholds.force_overshoot_fraction)
    ):
        state.hard_failures.add("prepuncture_force_limit")
    if measurement.backend_exit_count > 1:
        state.hard_failures.add("multiple_exit_events")
    if measurement.backend_right_underside_count > 1:
        state.hard_failures.add("multiple_right_underside_events")
    if measurement.missing_right_underside_puncture:
        state.hard_failures.add("missing_right_underside_puncture")
    if measurement.invalid_exit_route:
        state.hard_failures.add("invalid_cross_slab_exit_route")
    if measurement.exposed_fraction > thresholds.exposed_fraction_max:
        state.hard_failures.add("excessive_exit_exposure")
    if state.failed:
        return state

    state.peak_force_n = max(state.peak_force_n, measurement.normal_force_n)
    aligned = (
        measurement.entry_error_m <= thresholds.align_radius_m
        and measurement.tangent_error_deg <= thresholds.tangent_error_limit_deg
        and measurement.plane_error_deg <= thresholds.plane_error_limit_deg
    )
    next_phase = state.phase
    if state.phase == ThroughPuncturePhase.APPROACH:
        if measurement.entry_error_m <= thresholds.approach_radius_m:
            next_phase = ThroughPuncturePhase.ALIGN
    elif state.phase == ThroughPuncturePhase.ALIGN:
        if aligned:
            next_phase = ThroughPuncturePhase.INDENT
    elif state.phase == ThroughPuncturePhase.INDENT:
        ready = (
            aligned
            and measurement.target_region_valid
            and measurement.tissue_contact
            and measurement.indentation_m >= thresholds.prepuncture_depth_m
            and measurement.normal_force_n >= puncture_force_n
            and measurement.accumulated_work_j > 0.0
        )
        if ready:
            state.entry_event_count += 1
            state.entry_error_at_puncture_m = measurement.entry_error_m
            state.tangent_error_at_puncture_deg = measurement.tangent_error_deg
            state.plane_error_at_puncture_deg = measurement.plane_error_deg
            next_phase = ThroughPuncturePhase.PUNCTURE
    elif state.phase == ThroughPuncturePhase.PUNCTURE:
        if measurement.embedded_arc_length_m > 0.0:
            next_phase = ThroughPuncturePhase.DRIVE
    elif state.phase == ThroughPuncturePhase.DRIVE:
        if (
            measurement.backend_right_underside_count == 1
            and measurement.backend_exit_count == 1
            and measurement.exposed_arc_length_m > 0.0
        ):
            state.exit_event_count = 1
            next_phase = ThroughPuncturePhase.EXIT
    elif state.phase == ThroughPuncturePhase.EXIT:
        if (
            measurement.exit_error_m <= thresholds.exit_tolerance_m
            and measurement.exposed_fraction >= thresholds.exposed_fraction_min
        ):
            next_phase = ThroughPuncturePhase.PRESENT
            state.presented_steps = 1
    elif state.phase == ThroughPuncturePhase.PRESENT:
        exposed = (
            thresholds.exposed_fraction_min
            <= measurement.exposed_fraction
            <= thresholds.exposed_fraction_max
        )
        if exposed and measurement.exit_error_m <= thresholds.exit_tolerance_m:
            state.presented_steps += 1
        else:
            state.presented_steps = 0

    if next_phase < state.phase:
        state.hard_failures.add("invalid_phase_transition")
    elif next_phase != state.phase:
        state.phase = next_phase
        state.phase_sequence.append(next_phase.name.lower())
    return state


def through_puncture_success(
    state: ThroughPunctureGateState,
    measurement: ThroughPunctureMeasurement,
    thresholds: ThroughPunctureThresholds = ThroughPunctureThresholds(),
) -> bool:
    return (
        not state.failed
        and state.entry_event_count == 1
        and state.exit_event_count == 1
        and measurement.backend_exit_count == 1
        and measurement.backend_right_underside_count == 1
        and measurement.entry_slab == "left"
        and measurement.exit_slab == "right"
        and measurement.cross_slab_route_valid
        and state.phase == ThroughPuncturePhase.PRESENT
        and state.presented_steps >= thresholds.presentation_steps
        and state.entry_error_at_puncture_m is not None
        and state.entry_error_at_puncture_m <= thresholds.entry_tolerance_m
        and measurement.exit_error_m <= thresholds.exit_tolerance_m
        and thresholds.exposed_fraction_min
        <= measurement.exposed_fraction
        <= thresholds.exposed_fraction_max
    )


@dataclass(frozen=True)
class ThroughPunctureReceipt:
    schema: str
    success: bool
    entry_event_count: int
    exit_event_count: int
    right_underside_event_count: int
    representation_switch_count: int
    entry_error_m: float
    exit_error_m: float
    tangent_error_deg: float
    plane_error_deg: float
    sampled_puncture_force_n: float
    peak_force_n: float
    accumulated_work_j: float
    embedded_arc_length_m: float
    exposed_arc_length_m: float
    exposed_fraction: float
    phase_sequence: tuple[str, ...]
    backend_revision: str
    backend_implementation_sha256: str
    hard_failures: tuple[str, ...]
    entry_slab: str
    exit_slab: str
    cross_slab_route_valid: bool
    custody_model: str = "pregrasped_pose_coupling"
    evidence_level: str = "simulator_engineering_only"
    clinical_validation: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
