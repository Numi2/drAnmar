# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Physics-owned contract for the force-gated tissue-entry task.

This module deliberately has no Isaac Sim dependency.  The gate, evidence
receipt, and promotion decision can therefore be tested and audited outside
Kit while the live task feeds it measurements from PhysX and the native
Dr.Anmar tissue-entry backend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from math import isfinite
from typing import Any, Sequence


class PenetrationPhase(IntEnum):
    APPROACH = 0
    ALIGN = 1
    INDENT = 2
    PUNCTURE = 3
    STABILIZE = 4


HARD_FAILURES = (
    "grasp_loss",
    "off_target_contact",
    "unintended_jaw_tissue_contact",
    "nonfinite_solver_state",
    "invalid_phase_transition",
    "prepuncture_force_limit",
    "unintended_surface_crossing",
)


@dataclass(frozen=True)
class PunctureThresholds:
    approach_radius_m: float = 0.008
    align_radius_m: float = 0.002
    entry_tolerance_m: float = 0.001
    tangent_error_limit_deg: float = 10.0
    plane_error_limit_deg: float = 10.0
    prepuncture_depth_m: float = 0.0015
    depth_min_m: float = 0.0015
    depth_max_m: float = 0.0025
    stabilization_steps: int = 10
    force_overshoot_fraction: float = 0.25


@dataclass(frozen=True)
class PunctureMeasurement:
    entry_error_m: float
    tangent_error_deg: float
    plane_error_deg: float
    indentation_m: float
    embedded_depth_m: float
    normal_force_n: float
    accumulated_work_j: float
    bilateral_custody: bool
    target_region_valid: bool
    tissue_contact: bool
    solver_finite: bool = True
    unintended_jaw_contact: bool = False
    unintended_surface_crossing: bool = False


@dataclass
class PunctureGateState:
    phase: PenetrationPhase = PenetrationPhase.APPROACH
    event_count: int = 0
    stabilized_steps: int = 0
    peak_force_n: float = 0.0
    hard_failures: set[str] = field(default_factory=set)
    phase_sequence: list[str] = field(default_factory=lambda: ["approach"])

    @property
    def punctured(self) -> bool:
        return self.event_count == 1

    @property
    def failed(self) -> bool:
        return bool(self.hard_failures)


def needle_tissue_force_components(
    *,
    indentation_m: float,
    embedded_length_m: float,
    puncture_force_n: float,
    prepuncture_depth_m: float,
    cutting_fraction: float,
    shaft_drag_n_per_m: float,
    sweep_stiffness_n_m2: float,
    swept_area_m2: float,
) -> dict[str, float]:
    """Return the explicit compression/cutting/sweep/friction contact law.

    The task adds these interpretable material terms to the native backend's
    deformation response.  The environment gate still owns the one-shot puncture
    event; this function cannot change puncture state by itself.
    """

    depth = max(0.0, indentation_m)
    if embedded_length_m <= 0.0:
        ratio = min(depth / max(prepuncture_depth_m, 1.0e-9), 1.0)
        compression = puncture_force_n * ratio * ratio
        return {
            "compression_n": compression,
            "cutting_n": 0.0,
            "sweep_n": 0.0,
            "shaft_friction_n": 0.0,
            "total_n": compression,
        }
    cutting = max(0.0, cutting_fraction * puncture_force_n)
    sweep = max(0.0, sweep_stiffness_n_m2 * swept_area_m2)
    shaft = max(0.0, shaft_drag_n_per_m * embedded_length_m)
    return {
        "compression_n": 0.0,
        "cutting_n": cutting,
        "sweep_n": sweep,
        "shaft_friction_n": shaft,
        "total_n": cutting + sweep + shaft,
    }


def advance_puncture_gate(
    state: PunctureGateState,
    measurement: PunctureMeasurement,
    *,
    puncture_force_n: float,
    thresholds: PunctureThresholds = PunctureThresholds(),
) -> PunctureGateState:
    """Advance the monotonic phase machine from measured physical evidence."""

    values = (
        measurement.entry_error_m,
        measurement.tangent_error_deg,
        measurement.plane_error_deg,
        measurement.indentation_m,
        measurement.embedded_depth_m,
        measurement.normal_force_n,
        measurement.accumulated_work_j,
        puncture_force_n,
    )
    if not measurement.solver_finite or not all(isfinite(value) for value in values):
        state.hard_failures.add("nonfinite_solver_state")
        return state
    if not measurement.bilateral_custody:
        state.hard_failures.add("grasp_loss")
    if measurement.tissue_contact and not measurement.target_region_valid:
        state.hard_failures.add("off_target_contact")
    if measurement.unintended_jaw_contact:
        state.hard_failures.add("unintended_jaw_tissue_contact")
    if measurement.unintended_surface_crossing:
        state.hard_failures.add("unintended_surface_crossing")
    if (
        not state.punctured
        and measurement.normal_force_n
        > puncture_force_n * (1.0 + thresholds.force_overshoot_fraction)
    ):
        state.hard_failures.add("prepuncture_force_limit")
    if state.failed:
        return state

    state.peak_force_n = max(state.peak_force_n, measurement.normal_force_n)
    aligned = (
        measurement.entry_error_m <= thresholds.align_radius_m
        and measurement.tangent_error_deg <= thresholds.tangent_error_limit_deg
        and measurement.plane_error_deg <= thresholds.plane_error_limit_deg
    )

    next_phase = state.phase
    if state.phase == PenetrationPhase.APPROACH:
        if measurement.entry_error_m <= thresholds.approach_radius_m:
            next_phase = PenetrationPhase.ALIGN
    elif state.phase == PenetrationPhase.ALIGN:
        if aligned:
            next_phase = PenetrationPhase.INDENT
    elif state.phase == PenetrationPhase.INDENT:
        puncture_ready = (
            aligned
            and measurement.target_region_valid
            and measurement.tissue_contact
            and measurement.indentation_m >= thresholds.prepuncture_depth_m
            and measurement.normal_force_n >= puncture_force_n
            and measurement.accumulated_work_j > 0.0
        )
        if puncture_ready:
            state.event_count += 1
            if state.event_count != 1:
                state.hard_failures.add("invalid_phase_transition")
                return state
            next_phase = PenetrationPhase.PUNCTURE
    elif state.phase == PenetrationPhase.PUNCTURE:
        if thresholds.depth_min_m <= measurement.embedded_depth_m <= thresholds.depth_max_m:
            next_phase = PenetrationPhase.STABILIZE
            state.stabilized_steps = 1
    elif state.phase == PenetrationPhase.STABILIZE:
        if thresholds.depth_min_m <= measurement.embedded_depth_m <= thresholds.depth_max_m:
            state.stabilized_steps += 1
        else:
            state.stabilized_steps = 0

    if next_phase < state.phase:
        state.hard_failures.add("invalid_phase_transition")
    elif next_phase != state.phase:
        state.phase = next_phase
        state.phase_sequence.append(next_phase.name.lower())
    return state


def puncture_success(
    state: PunctureGateState,
    measurement: PunctureMeasurement,
    thresholds: PunctureThresholds = PunctureThresholds(),
) -> bool:
    return (
        not state.failed
        and state.event_count == 1
        and state.phase == PenetrationPhase.STABILIZE
        and state.stabilized_steps >= thresholds.stabilization_steps
        and measurement.entry_error_m <= thresholds.entry_tolerance_m
        and measurement.tangent_error_deg <= thresholds.tangent_error_limit_deg
        and measurement.plane_error_deg <= thresholds.plane_error_limit_deg
        and thresholds.depth_min_m <= measurement.embedded_depth_m <= thresholds.depth_max_m
    )


@dataclass(frozen=True)
class PunctureReceipt:
    schema: str
    success: bool
    event_count: int
    representation_switch_count: int
    entry_position_m: tuple[float, float, float]
    entry_error_m: float
    tangent_error_deg: float
    plane_error_deg: float
    sampled_puncture_force_n: float
    peak_force_n: float
    accumulated_work_j: float
    embedded_depth_m: float
    phase_sequence: tuple[str, ...]
    backend_revision: str
    backend_implementation_sha256: str
    hard_failures: tuple[str, ...]
    custody_model: str = "pregrasped_pose_coupling"
    rigid_needle_collisions_enabled: bool = False
    evidence_level: str = "simulator_engineering_only"
    clinical_validation: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualificationCriteria:
    overall_success_rate: float = 0.80
    per_seed_success_floor: float = 0.75
    entry_rmse_m: float = 0.001
    angle_error_deg: float = 10.0
    force_overshoot_fraction: float = 0.25
    p95_step_ms: float = 20.0
    replay_rmse_m: float = 0.0005
    required_seeds: int = 3
    episodes_per_seed: int = 256


def evaluate_qualification(
    seed_results: Sequence[dict[str, Any]],
    *,
    criteria: QualificationCriteria = QualificationCriteria(),
) -> dict[str, Any]:
    """Fail-closed entry-only promotion decision from isolated seed results."""

    failures: list[str] = []
    if len(seed_results) != criteria.required_seeds:
        failures.append("wrong_seed_count")
    total_successes = 0
    total_episodes = 0
    for result in seed_results:
        episodes = int(result.get("episodes", 0))
        successes = int(result.get("successes", 0))
        total_successes += successes
        total_episodes += episodes
        if episodes != criteria.episodes_per_seed:
            failures.append("wrong_episode_count")
        if successes / max(episodes, 1) < criteria.per_seed_success_floor:
            failures.append("per_seed_success_floor")
        if int(result.get("hard_safety_failures", 0)) != 0:
            failures.append("hard_safety_failure")
        if int(result.get("unintended_crossings", 0)) != 0:
            failures.append("unintended_surface_crossing")
        if float(result.get("entry_rmse_m", float("inf"))) > criteria.entry_rmse_m:
            failures.append("entry_rmse")
        if float(result.get("tangent_error_deg_max", float("inf"))) > criteria.angle_error_deg:
            failures.append("tangent_error")
        if float(result.get("plane_error_deg_max", float("inf"))) > criteria.angle_error_deg:
            failures.append("plane_error")
        if float(result.get("force_overshoot_fraction_max", float("inf"))) > criteria.force_overshoot_fraction:
            failures.append("force_overshoot")
        if float(result.get("physics_step_ms_p95", float("inf"))) > criteria.p95_step_ms:
            failures.append("physics_step_time")
        if float(result.get("replay_rmse_m", float("inf"))) > criteria.replay_rmse_m:
            failures.append("replay_rmse")
        if not bool(result.get("replay_event_sequence_identical", False)):
            failures.append("replay_event_sequence")
    overall = total_successes / max(total_episodes, 1)
    if overall < criteria.overall_success_rate:
        failures.append("overall_success_rate")
    unique = tuple(sorted(set(failures)))
    return {
        "schema": "dr.anmar.puncture-entry-qualification.v1",
        "qualified": not unique,
        "overall_success_rate": overall,
        "episodes": total_episodes,
        "failures": unique,
        "clinical_validation": False,
    }


def learned_policy_beats_baseline(
    learned: dict[str, float], baseline: dict[str, float]
) -> tuple[bool, str]:
    """Apply the declared paired analytical-baseline promotion rule."""

    success_gain = learned["success_rate"] - baseline["success_rate"]
    if success_gain >= 0.05:
        return True, "success_rate_gain"
    if success_gain < -0.02:
        return False, "success_noninferiority"
    entry_improvement = (
        baseline["entry_rmse_m"] - learned["entry_rmse_m"]
    ) / max(baseline["entry_rmse_m"], 1.0e-12)
    force_improvement = (
        baseline["normalized_peak_force"] - learned["normalized_peak_force"]
    ) / max(baseline["normalized_peak_force"], 1.0e-12)
    entry_degradation = -entry_improvement
    force_degradation = -force_improvement
    if entry_improvement >= 0.10 and force_degradation <= 0.05:
        return True, "entry_rmse_improvement"
    if force_improvement >= 0.10 and entry_degradation <= 0.05:
        return True, "normalized_force_improvement"
    return False, "no_required_paired_improvement"


def phase_progress_delta(previous_phase: int, current_phase: int) -> float:
    """Pay each valid phase transition once; never pay for dwelling."""

    return 1.0 if current_phase == previous_phase + 1 else 0.0
