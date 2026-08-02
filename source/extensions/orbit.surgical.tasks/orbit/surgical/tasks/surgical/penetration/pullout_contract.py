# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Physics-owned contract for puncture, receiver transfer, and full pullout."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from math import isfinite
from typing import Any


class PulloutPhase(IntEnum):
    APPROACH = 0
    ALIGN = 1
    INDENT = 2
    PUNCTURE = 3
    DRIVE = 4
    EXIT = 5
    PRESENT = 6
    RECEIVER_APPROACH = 7
    RECEIVER_GRASP = 8
    TRANSFER = 9
    PULL = 10
    CLEAR = 11


@dataclass(frozen=True)
class PulloutThresholds:
    approach_radius_m: float = 0.008
    align_radius_m: float = 0.002
    entry_tolerance_m: float = 0.001
    exit_tolerance_m: float = 0.001
    tangent_error_limit_deg: float = 10.0
    plane_error_limit_deg: float = 10.0
    # The collision-enabled full-needle grasp settles at 0.9 mm against the
    # native compression law; the entry proxy's separate 1.5 mm contract is
    # intentionally unchanged.
    prepuncture_depth_m: float = 0.0008
    # Present at least one fifth of the sampled leading arc before receiver
    # approach; the qualified controller settles at 21/128 samples.  The
    # receiver then owns the remaining exposure and must still prove at least
    # 95% complete clearance.
    presented_fraction_min: float = 0.20
    presented_fraction_max: float = 0.30
    presentation_steps: int = 10
    receiver_approach_m: float = 0.004
    receiver_grasp_m: float = 0.0008
    receiver_contact_steps: int = 3
    # Receiver custody must remain stable while the giver retreats 5 mm at the
    # bounded 0.25 mm command limit; only then may the binary jaw open.
    transfer_release_steps: int = 20
    # A 0.5 degree circular step needs roughly 280 steps to rotate the
    # remaining 78% of a semicircular needle through the right flap.
    receiver_pull_steps_min: int = 270
    receiver_curve_rotation_min_deg: float = 135.0
    receiver_curve_center_error_max_m: float = 0.0015
    embedded_arc_clearance_m: float = 0.0001
    exposed_fraction_clearance: float = 0.995
    clearance_steps: int = 10
    force_overshoot_fraction: float = 0.25


@dataclass(frozen=True)
class PulloutMeasurement:
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
    giver_custody: bool
    receiver_distance_m: float
    receiver_bilateral_contact: bool
    receiver_custody: bool
    giver_released: bool
    receiver_curve_rotation_deg: float
    receiver_curve_center_error_m: float
    target_region_valid: bool
    tissue_contact: bool
    backend_exit_count: int
    backend_right_underside_count: int = 0
    entry_slab: str = "none"
    exit_slab: str = "none"
    cross_slab_route_valid: bool = False
    invalid_exit_route: bool = False
    missing_right_underside_puncture: bool = False
    tract_support_active: bool = False
    tract_support_event_count: int = 0
    giver_regrasp_stage: int = 0
    giver_regrasp_complete: bool = False
    solver_finite: bool = True
    unintended_jaw_contact: bool = False
    unintended_robot_contact: bool = False
    unintended_surface_crossing: bool = False


@dataclass
class PulloutGateState:
    phase: PulloutPhase = PulloutPhase.APPROACH
    entry_event_count: int = 0
    exit_event_count: int = 0
    presented_steps: int = 0
    receiver_contact_steps: int = 0
    giver_release_steps: int = 0
    receiver_pull_steps: int = 0
    cleared_steps: int = 0
    peak_force_n: float = 0.0
    entry_error_at_puncture_m: float | None = None
    exit_error_at_event_m: float | None = None
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


def advance_pullout_gate(
    state: PulloutGateState,
    measurement: PulloutMeasurement,
    *,
    puncture_force_n: float,
    thresholds: PulloutThresholds = PulloutThresholds(),
) -> PulloutGateState:
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
        measurement.receiver_distance_m,
        puncture_force_n,
    )
    if not measurement.solver_finite or not all(isfinite(value) for value in values):
        state.hard_failures.add("nonfinite_solver_state")
        return state
    if state.phase < PulloutPhase.TRANSFER and not measurement.giver_custody:
        state.hard_failures.add("premature_giver_custody_loss")
    # The collision-enabled arc may brush the bounded 2 mm alignment region
    # while the tip finishes centering. This is not an entry event: puncture
    # still requires the stricter 1 mm target_region_valid receipt below.
    if (
        state.phase < PulloutPhase.PUNCTURE
        and measurement.tissue_contact
        and measurement.entry_error_m > thresholds.align_radius_m
    ):
        state.hard_failures.add("off_target_contact")
    if measurement.unintended_jaw_contact:
        state.hard_failures.add("unintended_jaw_tissue_contact")
    if measurement.unintended_robot_contact:
        state.hard_failures.add("unintended_robot_tissue_contact")
    if measurement.unintended_surface_crossing:
        state.hard_failures.add("unintended_surface_crossing")
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
    if measurement.tract_support_event_count > 4:
        state.hard_failures.add("multiple_tract_support_events")
    if measurement.tract_support_active and not state.punctured:
        state.hard_failures.add("tract_support_before_puncture")
    if measurement.tract_support_active and state.phase >= PulloutPhase.EXIT:
        state.hard_failures.add("tract_support_after_exit")
    if measurement.giver_regrasp_stage not in range(6):
        state.hard_failures.add("invalid_giver_regrasp_stage")
    if state.phase >= PulloutPhase.PULL and not measurement.receiver_custody:
        state.hard_failures.add("receiver_custody_loss")
    if (
        state.phase >= PulloutPhase.PULL
        and measurement.receiver_curve_center_error_m
        > thresholds.receiver_curve_center_error_max_m
    ):
        state.hard_failures.add("receiver_cross_surface_extraction")
    if state.failed:
        return state

    state.peak_force_n = max(state.peak_force_n, measurement.normal_force_n)
    aligned = (
        measurement.entry_error_m <= thresholds.align_radius_m
        and measurement.tangent_error_deg <= thresholds.tangent_error_limit_deg
        and measurement.plane_error_deg <= thresholds.plane_error_limit_deg
    )
    next_phase = state.phase
    if state.phase == PulloutPhase.APPROACH:
        if measurement.entry_error_m <= thresholds.approach_radius_m:
            next_phase = PulloutPhase.ALIGN
    elif state.phase == PulloutPhase.ALIGN:
        if aligned:
            next_phase = PulloutPhase.INDENT
    elif state.phase == PulloutPhase.INDENT:
        if (
            aligned
            and measurement.target_region_valid
            and measurement.tissue_contact
            and measurement.indentation_m >= thresholds.prepuncture_depth_m
            and measurement.normal_force_n >= puncture_force_n
            and measurement.accumulated_work_j > 0.0
        ):
            state.entry_event_count = 1
            state.entry_error_at_puncture_m = measurement.entry_error_m
            state.tangent_error_at_puncture_deg = measurement.tangent_error_deg
            state.plane_error_at_puncture_deg = measurement.plane_error_deg
            next_phase = PulloutPhase.PUNCTURE
    elif state.phase == PulloutPhase.PUNCTURE:
        if measurement.embedded_arc_length_m > 0.0:
            next_phase = PulloutPhase.DRIVE
    elif state.phase == PulloutPhase.DRIVE:
        if (
            measurement.backend_right_underside_count == 1
            and measurement.backend_exit_count == 1
            and measurement.exposed_arc_length_m > 0.0
        ):
            state.exit_event_count = 1
            state.exit_error_at_event_m = measurement.exit_error_m
            next_phase = PulloutPhase.EXIT
    elif state.phase == PulloutPhase.EXIT:
        if (
            measurement.exit_error_m <= thresholds.exit_tolerance_m
            and thresholds.presented_fraction_min
            <= measurement.exposed_fraction
            <= thresholds.presented_fraction_max
        ):
            state.presented_steps = 1
            next_phase = PulloutPhase.PRESENT
    elif state.phase == PulloutPhase.PRESENT:
        stable = (
            measurement.exit_error_m <= thresholds.exit_tolerance_m
            and thresholds.presented_fraction_min
            <= measurement.exposed_fraction
            <= thresholds.presented_fraction_max
        )
        state.presented_steps = state.presented_steps + 1 if stable else 0
        if state.presented_steps >= thresholds.presentation_steps:
            next_phase = PulloutPhase.RECEIVER_APPROACH
    elif state.phase == PulloutPhase.RECEIVER_APPROACH:
        if measurement.receiver_distance_m <= thresholds.receiver_approach_m:
            next_phase = PulloutPhase.RECEIVER_GRASP
    elif state.phase == PulloutPhase.RECEIVER_GRASP:
        state.receiver_contact_steps = (
            state.receiver_contact_steps + 1
            if measurement.receiver_bilateral_contact
            else 0
        )
        if state.receiver_contact_steps >= thresholds.receiver_contact_steps:
            next_phase = PulloutPhase.TRANSFER
    elif state.phase == PulloutPhase.TRANSFER:
        state.giver_release_steps = (
            state.giver_release_steps + 1
            if measurement.receiver_custody
            else 0
        )
        if state.giver_release_steps >= thresholds.transfer_release_steps:
            next_phase = PulloutPhase.PULL
    elif state.phase == PulloutPhase.PULL:
        if measurement.receiver_custody and measurement.giver_released:
            state.receiver_pull_steps += 1
        clear = (
            measurement.receiver_custody
            and measurement.giver_released
            and state.receiver_pull_steps >= thresholds.receiver_pull_steps_min
            and measurement.receiver_curve_rotation_deg
            >= thresholds.receiver_curve_rotation_min_deg
            and measurement.embedded_arc_length_m <= thresholds.embedded_arc_clearance_m
            and measurement.exposed_fraction >= thresholds.exposed_fraction_clearance
        )
        if clear:
            state.cleared_steps = 1
            next_phase = PulloutPhase.CLEAR
    elif state.phase == PulloutPhase.CLEAR:
        clear = (
            measurement.receiver_custody
            and measurement.embedded_arc_length_m <= thresholds.embedded_arc_clearance_m
            and measurement.exposed_fraction >= thresholds.exposed_fraction_clearance
        )
        state.cleared_steps = state.cleared_steps + 1 if clear else 0

    if next_phase < state.phase:
        state.hard_failures.add("invalid_phase_transition")
    elif next_phase != state.phase:
        state.phase = next_phase
        state.phase_sequence.append(next_phase.name.lower())
    return state


def pullout_success(
    state: PulloutGateState,
    measurement: PulloutMeasurement,
    thresholds: PulloutThresholds = PulloutThresholds(),
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
        and measurement.tract_support_event_count == 4
        and not measurement.tract_support_active
        and measurement.giver_regrasp_complete
        and state.phase == PulloutPhase.CLEAR
        and state.cleared_steps >= thresholds.clearance_steps
        and measurement.receiver_custody
        and measurement.giver_released
        and measurement.receiver_curve_rotation_deg
        >= thresholds.receiver_curve_rotation_min_deg
        and measurement.receiver_curve_center_error_m
        <= thresholds.receiver_curve_center_error_max_m
        and measurement.embedded_arc_length_m <= thresholds.embedded_arc_clearance_m
        and measurement.exposed_fraction >= thresholds.exposed_fraction_clearance
    )


@dataclass(frozen=True)
class PulloutReceipt:
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
    exposed_fraction: float
    embedded_arc_length_m: float
    exposed_arc_length_m: float
    receiver_contact_steps: int
    receiver_pull_steps: int
    receiver_curve_rotation_deg: float
    receiver_curve_center_error_m: float
    receiver_only_clearance_steps: int
    phase_sequence: tuple[str, ...]
    backend_revision: str
    backend_implementation_sha256: str
    hard_failures: tuple[str, ...]
    entry_slab: str
    exit_slab: str
    cross_slab_route_valid: bool
    tract_support_event_count: int
    giver_regrasp_complete: bool
    custody_model: str = (
        "bilateral_force_or_calibrated_geometry_then_receiver_pose_coupling"
    )
    evidence_level: str = "simulator_engineering_only"
    clinical_validation: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
