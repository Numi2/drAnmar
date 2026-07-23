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
from typing import Any, Iterable

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
    abrasion_damage: float = 0.0
    compacted_knot: bool = False
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
        full_crush = dose >= 1.0
        for joint_index in joint_indices:
            history = self._validate_joint(int(joint_index))
            if full_crush:
                history.grasp_count += max(1, int(dose))
            else:
                history.abrasion_damage = min(
                    0.95, history.abrasion_damage + 0.015 * dose
                )
        return full_crush

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
            prim.GetAttribute("physics:breakForce").Set(break_force)
            prim.GetAttribute("drive:transX:physics:maxForce").Set(break_force)
            prim.GetAttribute("drive:transX:physics:stiffness").Set(stiffness)
            baseline_torque = (
                self.derived.straight_failure_load_n
                * self.derived.radius_m
                * float(self.profile["knot"]["nominal_strength_efficiency"])
            )
            prim.GetAttribute("physics:breakTorque").Set(
                baseline_torque * self.joint_strength_fraction(joint_index)
            )
            minimum_force = min(minimum_force, break_force)
            minimum_stiffness = min(minimum_stiffness, stiffness)
            if break_force <= 0.0:
                history.failed = True
            updated += 1
        friction_coefficient = None
        if representative_self_contact_load_n is not None:
            material_path = f"{self.root_path}/Materials/SutureMaterial"
            material_prim = stage.GetPrimAtPath(material_path)
            if material_prim and material_prim.IsValid():
                friction_coefficient = self_friction_coefficient(
                    self.profile, representative_self_contact_load_n
                )
                material_prim.GetAttribute("physics:dynamicFriction").Set(
                    friction_coefficient
                )
                material_prim.GetAttribute("physics:staticFriction").Set(
                    min(1.0, friction_coefficient * 1.25)
                )
            else:
                missing.append(material_path)
        return {
            "updated_joints": updated,
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
