#!/usr/bin/env python3
"""Runtime state and OpenUSD updates for the Dr.Anmar surgical suture.

PhysX owns motion, contact, joint constraints, and breakage.  This controller
only updates physical joint parameters from observed operative events that a
static USD cannot remember: wet relaxation time, knot compaction, needle-driver
crush, and abrasion work.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from dr_anmar_suture_model import (
    DEFAULT_PROFILE_PATH,
    crush_strength_fraction,
    derive,
    load_profile,
    monotonic_tension_force,
    self_friction_coefficient,
    stress_retention,
)


@dataclass
class JointHistory:
    grasp_count: int = 0
    crush_dose: float = 0.0
    abrasion_damage: float = 0.0
    compacted_knot: bool = False
    knot_contact_time_s: float = 0.0
    loaded_time_s: float = 0.0
    current_strain: float = 0.0
    peak_strain: float = 0.0
    failed: bool = False


class SutureRuntime:
    """Stateful material-history controller for one authored suture."""

    def __init__(
        self,
        profile: dict[str, Any],
        *,
        root_path: str = "/World/DrAnmarSuture4_0",
    ) -> None:
        self.profile = profile
        self.derived = derive(profile)
        self.root_path = root_path.rstrip("/")
        self.joints = [JointHistory() for _ in range(self.derived.segment_count)]
        self._last_segment_positions: tuple[tuple[float, float, float], ...] | None = None
        self._last_applied_joint_state: dict[int, tuple[float, float, float]] = {}
        self._last_applied_friction: float | None = None

    def _validate_joint(self, joint_index: int) -> JointHistory:
        if not 0 <= joint_index < len(self.joints):
            raise IndexError(
                f"joint index {joint_index} outside 0..{len(self.joints) - 1}"
            )
        return self.joints[joint_index]

    def record_instrument_grasp(
        self,
        joint_indices: Iterable[int],
        *,
        pressure_pa: float,
        duration_s: float,
    ) -> bool:
        """Record a damaging needle-driver/grasper closure.

        A grasp becomes a crush event when pressure-duration reaches the
        experimental 45 MPa for one second reference dose. Lower-force contacts
        still contribute proportionally through abrasion, but do not count as a
        full crush event.
        """

        damage = self.profile["instrument_damage"]
        reference_pressure = float(damage["reference_crush_pressure_pa"])
        reference_duration = float(damage["reference_crush_duration_s"])
        dose = (
            max(0.0, pressure_pa)
            * max(0.0, duration_s)
            / (reference_pressure * reference_duration)
        )
        full_crush = False
        for joint_index in joint_indices:
            history = self._validate_joint(int(joint_index))
            previous_whole_doses = int(history.crush_dose)
            history.crush_dose += dose
            new_whole_doses = int(history.crush_dose) - previous_whole_doses
            if new_whole_doses > 0:
                history.grasp_count += new_whole_doses
                full_crush = True
            if dose > 0.0 and new_whole_doses == 0:
                history.abrasion_damage = min(
                    0.95, history.abrasion_damage + 0.015 * dose
                )
        return full_crush

    @staticmethod
    def _distance(
        left: Sequence[float],
        right: Sequence[float],
    ) -> float:
        return math.sqrt(
            sum((float(left[axis]) - float(right[axis])) ** 2 for axis in range(3))
        )

    @staticmethod
    def _bend_radius(
        before: Sequence[float],
        center: Sequence[float],
        after: Sequence[float],
    ) -> float:
        """Return the circumradius of three observed segment centers."""

        a = SutureRuntime._distance(center, after)
        b = SutureRuntime._distance(before, after)
        c = SutureRuntime._distance(before, center)
        ab = tuple(float(center[index]) - float(before[index]) for index in range(3))
        ac = tuple(float(after[index]) - float(before[index]) for index in range(3))
        cross = (
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        )
        doubled_area = math.sqrt(sum(value * value for value in cross))
        if doubled_area <= 1.0e-12:
            return math.inf
        return a * b * c / (2.0 * doubled_area)

    def record_instrument_contact(
        self,
        segment_positions: Sequence[Sequence[float]],
        *,
        tool_position: Sequence[float],
        contact_force_n: float,
        duration_s: float,
    ) -> dict[str, Any]:
        """Map one filtered jaw contact to nearby strand joints.

        Isaac Lab owns contact generation. This method only converts the
        filtered force and measured tool pose into a local pressure-duration
        history for the already-authored suture material model.
        """

        runtime = self.profile.get("runtime_detection", {})
        contact_radius = float(
            runtime.get(
                "instrument_localization_radius_m",
                max(0.0015, self.derived.radius_m * 8.0),
            )
        )
        area = float(runtime.get("driver_contact_area_m2_seed", 5.0e-8))
        candidates = [
            (self._distance(position, tool_position), index)
            for index, position in enumerate(segment_positions)
        ]
        localized = [
            index
            for distance, index in sorted(candidates)[:8]
            if distance <= contact_radius
        ]
        if not localized or contact_force_n <= 0.0:
            return {
                "localized_joint_indices": [],
                "pressure_pa": 0.0,
                "full_crush": False,
            }
        pressure = max(0.0, float(contact_force_n)) / max(area, 1.0e-12)
        full_crush = self.record_instrument_grasp(
            localized,
            pressure_pa=pressure,
            duration_s=max(0.0, float(duration_s)),
        )
        return {
            "localized_joint_indices": localized,
            "pressure_pa": pressure,
            "full_crush": full_crush,
        }

    def observe_segment_positions(
        self,
        segment_positions: Sequence[Sequence[float]],
        *,
        dt_s: float,
        representative_self_contact_load_n: float = 0.0,
    ) -> dict[str, Any]:
        """Update strain, tight-bend, self-contact, abrasion, and failure history."""

        positions = tuple(
            tuple(float(value) for value in position[:3])
            for position in segment_positions
        )
        if len(positions) != self.derived.segment_count:
            raise ValueError(
                f"expected {self.derived.segment_count} segment positions, "
                f"received {len(positions)}"
            )
        dt = max(0.0, float(dt_s))
        spacing = self.derived.segment_spacing_m
        strains = [0.0]
        strains.extend(
            max(0.0, self._distance(positions[index - 1], positions[index]) / spacing - 1.0)
            for index in range(1, len(positions))
        )
        self.update_loading(strains, dt_s=dt)

        failure_strain = float(self.profile["tension"]["failure_strain"])
        newly_failed: list[int] = []
        for index, strain in enumerate(strains):
            if strain >= failure_strain and not self.joints[index].failed:
                self.joints[index].failed = True
                newly_failed.append(index)

        runtime = self.profile.get("runtime_detection", {})
        contact_distance = float(
            runtime.get(
                "self_contact_centerline_distance_m",
                float(self.profile["geometry"]["diameter_m"]) * 1.35,
            )
        )
        minimum_separation = int(runtime.get("knot_minimum_index_separation", 12))
        knot_dwell_s = float(runtime.get("knot_compaction_dwell_s", 0.25))
        tight_radius = (
            float(self.profile["knot"]["tight_bend_radius_diameters"])
            * float(self.profile["geometry"]["diameter_m"])
        )
        tight_bends = {
            index
            for index in range(1, len(positions) - 1)
            if self._bend_radius(
                positions[index - 1],
                positions[index],
                positions[index + 1],
            )
            <= tight_radius
        }
        nonadjacent_contacts: list[tuple[int, int]] = []
        knot_candidates: set[int] = set()
        for left in range(len(positions)):
            for right in range(left + minimum_separation, len(positions)):
                if self._distance(positions[left], positions[right]) > contact_distance:
                    continue
                nonadjacent_contacts.append((left, right))
                left_tight = any(abs(left - index) <= 3 for index in tight_bends)
                right_tight = any(abs(right - index) <= 3 for index in tight_bends)
                if left_tight or right_tight:
                    knot_candidates.update(
                        index
                        for center in (left, right)
                        for index in range(max(0, center - 2), min(len(positions), center + 3))
                    )

        for index, history in enumerate(self.joints):
            if index in knot_candidates:
                history.knot_contact_time_s += dt
                if history.knot_contact_time_s >= knot_dwell_s:
                    history.compacted_knot = True
            elif not history.compacted_knot:
                history.knot_contact_time_s = max(0.0, history.knot_contact_time_s - dt)

        abrasion_work_j = 0.0
        if self._last_segment_positions is not None and representative_self_contact_load_n > 0.0:
            for left, right in nonadjacent_contacts:
                left_motion = tuple(
                    positions[left][axis] - self._last_segment_positions[left][axis]
                    for axis in range(3)
                )
                right_motion = tuple(
                    positions[right][axis] - self._last_segment_positions[right][axis]
                    for axis in range(3)
                )
                relative_slip = math.sqrt(
                    sum(
                        (left_motion[axis] - right_motion[axis]) ** 2
                        for axis in range(3)
                    )
                )
                work = max(0.0, representative_self_contact_load_n) * relative_slip
                abrasion_work_j += work
                if work > 0.0:
                    self.record_abrasion((left, right), work_j=work / 2.0)
        self._last_segment_positions = positions
        return {
            "maximum_strain": max(strains, default=0.0),
            "tight_bend_joint_count": len(tight_bends),
            "nonadjacent_self_contact_count": len(nonadjacent_contacts),
            "knot_candidate_joint_count": len(knot_candidates),
            "compacted_knot_joint_count": sum(
                history.compacted_knot for history in self.joints
            ),
            "abrasion_work_j": abrasion_work_j,
            "newly_failed_joint_indices": newly_failed,
        }

    def record_abrasion(self, joint_indices: Iterable[int], work_j: float) -> None:
        threshold = float(
            self.profile["instrument_damage"]["abrasion_work_to_failure_j_seed"]
        )
        increment = max(0.0, float(work_j)) / threshold
        for joint_index in joint_indices:
            history = self._validate_joint(int(joint_index))
            history.abrasion_damage = min(0.95, history.abrasion_damage + increment)

    def set_knot_compaction(
        self, joint_indices: Iterable[int], compacted: bool = True
    ) -> None:
        for joint_index in joint_indices:
            self._validate_joint(int(joint_index)).compacted_knot = bool(compacted)

    def update_loading(
        self,
        joint_strains: Iterable[float],
        *,
        dt_s: float,
    ) -> None:
        strains = list(joint_strains)
        if len(strains) != len(self.joints):
            raise ValueError(
                f"expected {len(self.joints)} strains, received {len(strains)}"
            )
        yield_strain = float(self.profile["tension"]["yield_strain"])
        for history, strain in zip(self.joints, strains):
            positive_strain = max(0.0, float(strain))
            history.current_strain = positive_strain
            history.peak_strain = max(history.peak_strain, positive_strain)
            if positive_strain >= yield_strain * 0.1:
                history.loaded_time_s += max(0.0, float(dt_s))

    def joint_strength_fraction(self, joint_index: int) -> float:
        history = self._validate_joint(joint_index)
        fraction = crush_strength_fraction(self.profile, history.grasp_count)
        fraction *= 1.0 - history.abrasion_damage
        if history.compacted_knot:
            fraction *= float(self.profile["knot"]["nominal_strength_efficiency"])
        return max(0.0, min(1.0, fraction))

    def joint_break_force_n(self, joint_index: int) -> float:
        history = self._validate_joint(joint_index)
        if history.failed:
            return 0.0
        if joint_index == 0:
            baseline = float(self.profile["swage"]["pullout_force_n_seed"])
        else:
            baseline = self.derived.straight_failure_load_n
        return baseline * self.joint_strength_fraction(joint_index)

    def joint_stiffness_n_m(self, joint_index: int) -> float:
        history = self._validate_joint(joint_index)
        if history.current_strain <= 1e-9:
            relaxation = stress_retention(self.profile, history.loaded_time_s)
            return self.derived.axial_joint_stiffness_n_m * relaxation
        force, _ = monotonic_tension_force(
            self.profile,
            history.current_strain,
            elapsed_s=history.loaded_time_s,
            knotted=history.compacted_knot,
            grasp_count=history.grasp_count,
            abrasion_damage=history.abrasion_damage,
        )
        extension_m = history.current_strain * self.derived.segment_spacing_m
        return force / extension_m

    def apply_to_stage(
        self,
        stage: Any,
        *,
        representative_self_contact_load_n: float | None = None,
    ) -> dict[str, Any]:
        """Write current physical history into live OpenUSD joint attributes."""

        updated = 0
        unchanged = 0
        missing: list[str] = []
        minimum_force = math.inf
        minimum_stiffness = math.inf
        for joint_index, history in enumerate(self.joints):
            joint_path = f"{self.root_path}/Joints/J{joint_index:04d}"
            prim = stage.GetPrimAtPath(joint_path)
            if not prim or not prim.IsValid():
                missing.append(joint_path)
                continue
            break_force = self.joint_break_force_n(joint_index)
            stiffness = self.joint_stiffness_n_m(joint_index)
            baseline_torque = (
                self.derived.straight_failure_load_n
                * self.derived.radius_m
                * float(self.profile["knot"]["nominal_strength_efficiency"])
            )
            break_torque = baseline_torque * self.joint_strength_fraction(joint_index)
            state = (break_force, stiffness, break_torque)
            if self._last_applied_joint_state.get(joint_index) == state:
                unchanged += 1
            else:
                prim.GetAttribute("physics:breakForce").Set(break_force)
                prim.GetAttribute("drive:transX:physics:maxForce").Set(break_force)
                prim.GetAttribute("drive:transX:physics:stiffness").Set(stiffness)
                prim.GetAttribute("physics:breakTorque").Set(break_torque)
                self._last_applied_joint_state[joint_index] = state
                updated += 1
            minimum_force = min(minimum_force, break_force)
            minimum_stiffness = min(minimum_stiffness, stiffness)
            if break_force <= 0.0:
                history.failed = True
        friction_coefficient = None
        if representative_self_contact_load_n is not None:
            material_path = f"{self.root_path}/Materials/SutureMaterial"
            material_prim = stage.GetPrimAtPath(material_path)
            if material_prim and material_prim.IsValid():
                friction_coefficient = self_friction_coefficient(
                    self.profile, representative_self_contact_load_n
                )
                if friction_coefficient != self._last_applied_friction:
                    material_prim.GetAttribute("physics:dynamicFriction").Set(
                        friction_coefficient
                    )
                    static_to_dynamic_ratio = float(
                        self.profile["contact"].get(
                            "sampled_static_to_dynamic_ratio",
                            float(self.profile["contact"]["static_friction"])
                            / max(
                                float(
                                    self.profile["contact"][
                                        "dynamic_friction"
                                    ]
                                ),
                                1.0e-9,
                            ),
                        )
                    )
                    material_prim.GetAttribute("physics:staticFriction").Set(
                        min(
                            1.0,
                            friction_coefficient * static_to_dynamic_ratio,
                        )
                    )
                    self._last_applied_friction = friction_coefficient
            else:
                missing.append(material_path)
        return {
            "updated_joints": updated,
            "unchanged_joints": unchanged,
            "missing_joints": missing,
            "minimum_break_force_n": 0.0
            if math.isinf(minimum_force)
            else minimum_force,
            "minimum_axial_stiffness_n_m": 0.0
            if math.isinf(minimum_stiffness)
            else minimum_stiffness,
            "self_contact_friction_coefficient": friction_coefficient,
        }

    def telemetry(self) -> dict[str, Any]:
        damaged = [
            index
            for index, history in enumerate(self.joints)
            if history.grasp_count or history.abrasion_damage or history.compacted_knot
        ]
        return {
            "schema": "dr.anmar.suture-runtime-telemetry.v1",
            "profile_id": self.profile["id"],
            "joint_count": len(self.joints),
            "damaged_joint_count": len(damaged),
            "damaged_joint_indices": damaged,
            "minimum_break_force_n": min(
                self.joint_break_force_n(index) for index in range(len(self.joints))
            ),
            "maximum_loaded_time_s": max(
                history.loaded_time_s for history in self.joints
            ),
            "failed_joint_count": sum(history.failed for history in self.joints),
            "compacted_knot_joint_count": sum(
                history.compacted_knot for history in self.joints
            ),
            "maximum_observed_strain": max(
                history.peak_strain for history in self.joints
            ),
            "clinical_validation": False,
        }


def self_test(profile: dict[str, Any]) -> dict[str, Any]:
    runtime = SutureRuntime(profile)
    baseline = runtime.joint_break_force_n(20)
    crushed = runtime.record_instrument_grasp(
        range(18, 23), pressure_pa=45e6, duration_s=1.0
    )
    after_crush = runtime.joint_break_force_n(20)
    runtime.set_knot_compaction(range(18, 23))
    after_knot = runtime.joint_break_force_n(20)
    strains = [0.0] * runtime.derived.segment_count
    strains[20] = 0.03
    runtime.update_loading(strains, dt_s=7200.0)
    relaxed_stiffness = runtime.joint_stiffness_n_m(20)
    nonlinear_stiffness = runtime.joint_stiffness_n_m(20)
    runtime.record_abrasion((20,), work_j=0.018)
    after_abrasion = runtime.joint_break_force_n(20)
    checks = {
        "crush_event_recorded": crushed,
        "crush_reduces_strength": after_crush < baseline,
        "knot_reduces_strength": after_knot < after_crush,
        "relaxation_reduces_stiffness": relaxed_stiffness
        < runtime.derived.axial_joint_stiffness_n_m,
        "nonlinear_tension_updates_joint_stiffness": nonlinear_stiffness
        != runtime.derived.axial_joint_stiffness_n_m,
        "abrasion_reduces_strength": after_abrasion < after_knot,
        "undamaged_neighbor_preserved": math.isclose(
            runtime.joint_break_force_n(40), baseline
        ),
    }
    return {
        "schema": "dr.anmar.suture-runtime-self-test.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "measurements": {
            "baseline_break_force_n": baseline,
            "after_crush_n": after_crush,
            "after_knot_n": after_knot,
            "after_abrasion_n": after_abrasion,
            "relaxed_stiffness_n_m": relaxed_stiffness,
            "nonlinear_stiffness_n_m": nonlinear_stiffness,
        },
        "clinical_validation": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    profile = load_profile(args.profile)
    report = self_test(profile)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
