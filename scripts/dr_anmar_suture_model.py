#!/usr/bin/env python3
"""Research-informed constitutive and damage model for the Dr.Anmar suture.

This module has no Isaac Sim dependency.  It is the single source of derived
mechanics used by the OpenUSD author and deterministic validator.  The model is
an engineering simulation contract, not a clinically validated medical model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_PATH = REPOSITORY_ROOT / "physics_next/sutures/dr-anmar-suture-4-0.json"


def load_profile(path: Path = DEFAULT_PROFILE_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def lerp(left: float, right: float, amount: float) -> float:
    return left + (right - left) * amount


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def crush_strength_fraction(profile: dict[str, Any], grasp_count: int) -> float:
    """Piecewise-linear strength retention after instrument crush events."""

    points = profile["instrument_damage"]["crush_strength_remaining"]
    count = max(0, int(grasp_count))
    if count <= int(points[0]["grasps"]):
        return float(points[0]["fraction"])
    for left, right in zip(points, points[1:]):
        left_grasps = int(left["grasps"])
        right_grasps = int(right["grasps"])
        if count <= right_grasps:
            amount = (count - left_grasps) / (right_grasps - left_grasps)
            return lerp(float(left["fraction"]), float(right["fraction"]), amount)
    return float(points[-1]["fraction"])


def stress_retention(profile: dict[str, Any], elapsed_s: float) -> float:
    """Two-term wet-state relaxation with the largest change early in time."""

    visco = profile["viscoelasticity"]
    elapsed = max(0.0, float(elapsed_s))
    asymptote = float(visco["retained_stress_asymptote"])
    fast_weight = float(visco["fast_relaxation_weight"])
    fast = math.exp(-elapsed / float(visco["fast_time_constant_s"]))
    slow = math.exp(-elapsed / float(visco["slow_time_constant_s"]))
    return asymptote + (1.0 - asymptote) * (
        fast_weight * fast + (1.0 - fast_weight) * slow
    )


def self_friction_coefficient(profile: dict[str, Any], normal_load_n: float) -> float:
    """Load-dependent Coulomb coefficient for suture-on-suture contact.

    Experiments show that friction force rises with load while the apparent
    coefficient is not constant and can fall as tension increases.
    """

    friction = profile["contact"]["load_dependent_self_friction"]
    low_load = float(friction["low_load_coefficient"])
    high_load = float(friction["high_load_coefficient"])
    transition = float(friction["transition_normal_load_n"])
    load = max(0.0, float(normal_load_n))
    blend = load / (load + transition)
    return lerp(low_load, high_load, blend)


def effective_failure_load(
    profile: dict[str, Any],
    *,
    knotted: bool = False,
    grasp_count: int = 0,
    abrasion_damage: float = 0.0,
) -> float:
    load = float(profile["tension"]["straight_failure_load_n"])
    if knotted:
        load *= float(profile["knot"]["nominal_strength_efficiency"])
    load *= crush_strength_fraction(profile, grasp_count)
    load *= 1.0 - clamp(float(abrasion_damage), 0.0, 0.95)
    return load


def monotonic_tension_force(
    profile: dict[str, Any],
    strain: float,
    *,
    elapsed_s: float = 0.0,
    knotted: bool = False,
    grasp_count: int = 0,
    abrasion_damage: float = 0.0,
) -> tuple[float, bool]:
    """Return force and failure state for monotonic axial loading.

    The low-strain branch uses the measured 0-3% modulus.  Beyond yield, a
    smooth hardening branch reaches the configured experimental failure load.
    Damage changes the failure envelope, while wet relaxation changes carried
    force at fixed extension.
    """

    geometry = profile["geometry"]
    tension = profile["tension"]
    material = profile["material"]
    extension = max(0.0, float(strain))
    diameter = float(geometry["diameter_m"])
    area = math.pi * diameter * diameter / 4.0
    yield_strain = float(tension["yield_strain"])
    failure_strain = float(tension["failure_strain"])
    failure_load = effective_failure_load(
        profile,
        knotted=knotted,
        grasp_count=grasp_count,
        abrasion_damage=abrasion_damage,
    )
    elastic_force = float(material["initial_axial_modulus_pa"]) * area * extension
    yield_force = min(
        float(material["initial_axial_modulus_pa"]) * area * yield_strain,
        failure_load * 0.7,
    )
    if extension <= yield_strain:
        dry_force = min(elastic_force, yield_force)
    else:
        normalized = clamp(
            (extension - yield_strain) / (failure_strain - yield_strain),
            0.0,
            1.0,
        )
        exponent = float(tension["post_yield_shape_exponent"])
        smooth = normalized**exponent
        dry_force = lerp(yield_force, failure_load, smooth)
    failed = extension >= failure_strain or dry_force >= failure_load
    return min(dry_force, failure_load) * stress_retention(profile, elapsed_s), failed


@dataclass(frozen=True)
class DerivedSuture:
    diameter_m: float
    radius_m: float
    area_m2: float
    length_m: float
    volume_m3: float
    mass_kg: float
    segment_count: int
    segment_spacing_m: float
    segment_mass_kg: float
    axial_rigidity_n: float
    axial_joint_stiffness_n_m: float
    axial_joint_damping_n_s_m: float
    bend_joint_stiffness_n_m_rad: float
    bend_joint_damping_n_m_s_rad: float
    twist_joint_stiffness_n_m_rad: float
    straight_failure_load_n: float
    knot_failure_load_n: float
    swage_segment_count: int


def derive(profile: dict[str, Any]) -> DerivedSuture:
    geometry = profile["geometry"]
    material = profile["material"]
    diameter = float(geometry["diameter_m"])
    radius = diameter / 2.0
    area = math.pi * radius * radius
    length = float(geometry["length_m"])
    volume = area * length
    density = float(material["density_kg_m3"])
    mass = density * volume
    segment_count = int(geometry["segment_count"])
    spacing = float(geometry["segment_spacing_m"])
    segment_mass = mass / segment_count
    axial_rigidity = float(material["initial_axial_modulus_pa"]) * area
    axial_stiffness = axial_rigidity / spacing
    reduced_mass = segment_mass / 2.0
    axial_damping = (
        2.0
        * float(material["axial_damping_ratio"])
        * math.sqrt(axial_stiffness * reduced_mass)
    )
    bend_stiffness = float(material["flexural_rigidity_n_m2"]) / spacing
    polar_inertia = 0.5 * segment_mass * radius * radius
    bend_damping = (
        2.0
        * float(material["bending_damping_ratio"])
        * math.sqrt(max(bend_stiffness * polar_inertia, 0.0))
    )
    twist_stiffness = float(material["torsional_rigidity_n_m2"]) / spacing
    straight_failure = float(profile["tension"]["straight_failure_load_n"])
    knot_failure = straight_failure * float(
        profile["knot"]["nominal_strength_efficiency"]
    )
    swage_segments = max(
        1, round(float(profile["swage"]["transition_length_m"]) / spacing)
    )
    return DerivedSuture(
        diameter_m=diameter,
        radius_m=radius,
        area_m2=area,
        length_m=length,
        volume_m3=volume,
        mass_kg=mass,
        segment_count=segment_count,
        segment_spacing_m=spacing,
        segment_mass_kg=segment_mass,
        axial_rigidity_n=axial_rigidity,
        axial_joint_stiffness_n_m=axial_stiffness,
        axial_joint_damping_n_s_m=axial_damping,
        bend_joint_stiffness_n_m_rad=bend_stiffness,
        bend_joint_damping_n_m_s_rad=bend_damping,
        twist_joint_stiffness_n_m_rad=twist_stiffness,
        straight_failure_load_n=straight_failure,
        knot_failure_load_n=knot_failure,
        swage_segment_count=swage_segments,
    )


def summary(profile: dict[str, Any]) -> dict[str, Any]:
    derived = derive(profile)
    return {
        "profile_id": profile["id"],
        "diameter_m": derived.diameter_m,
        "length_m": derived.length_m,
        "mass_kg": derived.mass_kg,
        "segment_count": derived.segment_count,
        "segment_mass_kg": derived.segment_mass_kg,
        "axial_rigidity_n": derived.axial_rigidity_n,
        "axial_joint_stiffness_n_m": derived.axial_joint_stiffness_n_m,
        "bend_joint_stiffness_n_m_rad": derived.bend_joint_stiffness_n_m_rad,
        "twist_joint_stiffness_n_m_rad": derived.twist_joint_stiffness_n_m_rad,
        "straight_failure_load_n": derived.straight_failure_load_n,
        "knot_failure_load_n": derived.knot_failure_load_n,
        "wet_stress_retention_2h": stress_retention(profile, 7200.0),
        "five_grasp_strength_fraction": crush_strength_fraction(profile, 5),
        "clinical_validation": False,
    }


if __name__ == "__main__":
    print(json.dumps(summary(load_profile()), indent=2, sort_keys=True))
