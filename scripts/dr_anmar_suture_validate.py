#!/usr/bin/env python3
"""Deterministically validate the Dr.Anmar 4-0 suture contract and USD."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from dr_anmar_suture_model import (
    DEFAULT_PROFILE_PATH,
    crush_strength_fraction,
    derive,
    effective_failure_load,
    load_profile,
    monotonic_tension_force,
    self_friction_coefficient,
    stress_retention,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET = (
    REPOSITORY_ROOT
    / "source/extensions/orbit.surgical.assets/data/Props/DrAnmarSuture/DrAnmarSuture4_0.usda"
)


def check(
    checks: dict[str, dict[str, Any]],
    name: str,
    passed: bool,
    measured: Any,
    expected: Any,
) -> None:
    checks[name] = {
        "passed": bool(passed),
        "measured": measured,
        "expected": expected,
    }


def validate(profile: dict[str, Any], asset_text: str) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    derived = derive(profile)
    geometry = profile["geometry"]
    material = profile["material"]
    tension = profile["tension"]
    diameter_range = [float(value) for value in geometry["diameter_range_m"]]
    check(
        checks,
        "true_4_0_diameter",
        diameter_range[0] <= derived.diameter_m <= diameter_range[1],
        derived.diameter_m,
        diameter_range,
    )
    density_range = [float(value) for value in material["density_range_kg_m3"]]
    check(
        checks,
        "polymer_density",
        density_range[0] <= float(material["density_kg_m3"]) <= density_range[1],
        float(material["density_kg_m3"]),
        density_range,
    )
    failure_range = [float(value) for value in tension["straight_failure_range_n"]]
    check(
        checks,
        "straight_failure_load",
        failure_range[0] <= derived.straight_failure_load_n <= failure_range[1],
        derived.straight_failure_load_n,
        failure_range,
    )
    failure_strain_range = [float(value) for value in tension["failure_strain_range"]]
    check(
        checks,
        "elongation_at_break",
        failure_strain_range[0]
        <= float(tension["failure_strain"])
        <= failure_strain_range[1],
        float(tension["failure_strain"]),
        failure_strain_range,
    )
    knot_range = [
        float(value) for value in profile["knot"]["strength_efficiency_range"]
    ]
    knot_efficiency = derived.knot_failure_load_n / derived.straight_failure_load_n
    check(
        checks,
        "knot_strength_reduction",
        knot_range[0] <= knot_efficiency <= knot_range[1],
        knot_efficiency,
        knot_range,
    )
    force_yield, failed_yield = monotonic_tension_force(
        profile, float(tension["yield_strain"])
    )
    force_break, failed_break = monotonic_tension_force(
        profile, float(tension["failure_strain"])
    )
    check(
        checks,
        "nonlinear_tension_curve",
        0.0 < force_yield < force_break <= derived.straight_failure_load_n
        and not failed_yield
        and failed_break,
        {"yield_force_n": force_yield, "break_force_n": force_break},
        "positive yield force below a 20-25 N failed endpoint",
    )
    retained_2h = stress_retention(profile, 7200.0)
    retained_long = stress_retention(profile, 1e9)
    check(
        checks,
        "wet_stress_relaxation",
        retained_long <= retained_2h < 1.0
        and math.isclose(
            retained_long,
            float(profile["viscoelasticity"]["retained_stress_asymptote"]),
            abs_tol=1e-6,
        ),
        {"two_hours": retained_2h, "long_time": retained_long},
        "largest early loss with configured asymptote",
    )
    friction_low = self_friction_coefficient(profile, 0.01)
    friction_high = self_friction_coefficient(profile, 2.0)
    check(
        checks,
        "load_dependent_self_friction",
        0.0 < friction_high < friction_low < 1.0,
        {"mu_at_0_01_n": friction_low, "mu_at_2_n": friction_high},
        "positive coefficient decreasing with load while friction force rises",
    )
    check(
        checks,
        "instrument_crush_damage",
        crush_strength_fraction(profile, 0) == 1.0
        and math.isclose(crush_strength_fraction(profile, 1), 0.661)
        and math.isclose(crush_strength_fraction(profile, 5), 0.383),
        {
            "zero_grasps": crush_strength_fraction(profile, 0),
            "one_grasp": crush_strength_fraction(profile, 1),
            "five_grasps": crush_strength_fraction(profile, 5),
        },
        {"zero_grasps": 1.0, "one_grasp": 0.661, "five_grasps": 0.383},
    )
    check(
        checks,
        "combined_knot_and_crush_failure",
        effective_failure_load(profile, knotted=True, grasp_count=1)
        < derived.knot_failure_load_n,
        effective_failure_load(profile, knotted=True, grasp_count=1),
        f"less than {derived.knot_failure_load_n}",
    )
    segment_defs = len(re.findall(r'def Capsule "S\d{4}"', asset_text))
    joint_defs = len(re.findall(r'def PhysicsJoint "J\d{4}"', asset_text))
    check(
        checks,
        "physical_segment_resolution",
        segment_defs == derived.segment_count,
        segment_defs,
        derived.segment_count,
    )
    check(
        checks,
        "breakable_joint_resolution",
        joint_defs == derived.segment_count
        and asset_text.count("physics:breakForce") == derived.segment_count,
        {
            "joint_defs": joint_defs,
            "break_force_attributes": asset_text.count("physics:breakForce"),
        },
        derived.segment_count,
    )
    required_asset_tokens = [
        "PhysicsRigidBodyAPI",
        "PhysicsCollisionAPI",
        "PhysicsDriveAPI:transX",
        "PhysicsDriveAPI:rotY",
        "PhysicsDriveAPI:rotZ",
        "PhysicsFilteredPairsAPI",
        "physxRigidBody:enableCCD",
        'def Capsule "NeedleInterface"',
        "drAnmar:swageFraction",
    ]
    missing_tokens = [
        token for token in required_asset_tokens if token not in asset_text
    ]
    check(
        checks,
        "runtime_physics_contract",
        not missing_tokens,
        {"missing": missing_tokens},
        required_asset_tokens,
    )
    forbidden_tokens = [
        "Rope.usd",
        "SoftMimicGen",
        "nvidia-strand-ring-threading",
    ]
    present_forbidden = [token for token in forbidden_tokens if token in asset_text]
    check(
        checks,
        "independent_from_current_thread",
        not present_forbidden,
        {"forbidden_references": present_forbidden},
        "no current-thread reference",
    )
    check(
        checks,
        "actual_scale_not_visibility_inflated",
        "drAnmarVisibilityScale" not in asset_text
        and math.isclose(derived.diameter_m, 0.00025),
        derived.diameter_m,
        0.00025,
    )
    evidence = profile.get("evidence", [])
    check(
        checks,
        "primary_research_provenance",
        len(evidence) >= 6
        and all(item.get("url") and item.get("used_for") for item in evidence),
        len(evidence),
        "at least six traceable experimental/computational sources",
    )
    passed = all(item["passed"] for item in checks.values())
    return {
        "schema": "dr.anmar.suture-validation.v1",
        "profile_id": profile["id"],
        "passed": passed,
        "checks": checks,
        "derived": {
            "mass_kg": derived.mass_kg,
            "segment_mass_kg": derived.segment_mass_kg,
            "axial_rigidity_n": derived.axial_rigidity_n,
            "axial_joint_stiffness_n_m": derived.axial_joint_stiffness_n_m,
            "bend_joint_stiffness_n_m_rad": derived.bend_joint_stiffness_n_m_rad,
            "twist_joint_stiffness_n_m_rad": derived.twist_joint_stiffness_n_m_rad,
        },
        "clinical_validation": False,
        "note": "Deterministic engineering validation only; physical bench and clinician validation remain required.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    profile = load_profile(args.profile)
    asset_text = args.asset.read_text(encoding="utf-8")
    report = validate(profile, asset_text)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
