#!/usr/bin/env python3
"""Research-informed constitutive and damage model for the Dr.Anmar suture.

This module has no Isaac Sim dependency.  It is the single source of derived
mechanics used by the OpenUSD author and deterministic validator.  The model is
an engineering simulation contract, not a clinically validated medical model.
"""

from __future__ import annotations

import json
import math
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_PATH = REPOSITORY_ROOT / "physics_next/sutures/dr-anmar-suture-4-0.json"


def load_profile(path: Path = DEFAULT_PROFILE_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_suture_runtime_profile(
    profile: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, float | int]]:
    """Sample only properties the live material-history controller can enact."""

    sampled = deepcopy(profile)
    generator = random.Random(int(seed))
    tension = sampled["tension"]
    knot = sampled["knot"]
    contact = sampled["contact"]
    visco = sampled["viscoelasticity"]
    damage = sampled["instrument_damage"]
    nominal_static_friction = float(contact["static_friction"])
    nominal_dynamic_friction = float(contact["dynamic_friction"])
    static_friction = generator.uniform(*map(float, contact["static_friction_range"]))
    dynamic_friction = min(
        static_friction,
        generator.uniform(*map(float, contact["dynamic_friction_range"])),
    )
    payload: dict[str, float | int] = {
        "seed": int(seed),
        "straight_failure_load_n": generator.uniform(*map(float, tension["straight_failure_range_n"])),
        "knot_strength_efficiency": generator.uniform(*map(float, knot["strength_efficiency_range"])),
        "static_friction": static_friction,
        "dynamic_friction": dynamic_friction,
        "retained_stress_asymptote": generator.uniform(*map(float, visco["retained_stress_asymptote_range"])),
        "reference_crush_pressure_pa": generator.uniform(*map(float, damage["reference_crush_pressure_range_pa"])),
        "abrasion_work_to_failure_j": generator.uniform(*map(float, damage["abrasion_work_to_failure_range_j"])),
    }
    tension["straight_failure_load_n"] = payload["straight_failure_load_n"]
    knot["nominal_strength_efficiency"] = payload["knot_strength_efficiency"]
    contact["static_friction"] = payload["static_friction"]
    contact["dynamic_friction"] = payload["dynamic_friction"]
    self_friction = contact["load_dependent_self_friction"]
    dynamic_scale = float(payload["dynamic_friction"]) / max(
        nominal_dynamic_friction,
        1.0e-9,
    )
    self_friction["low_load_coefficient"] = min(
        1.0,
        float(self_friction["low_load_coefficient"]) * dynamic_scale,
    )
    self_friction["high_load_coefficient"] = min(
        self_friction["low_load_coefficient"],
        float(self_friction["high_load_coefficient"]) * dynamic_scale,
    )
    contact["sampled_static_to_dynamic_ratio"] = float(payload["static_friction"]) / max(
        float(payload["dynamic_friction"]), 1.0e-9
    )
    contact["nominal_static_to_dynamic_ratio"] = nominal_static_friction / max(nominal_dynamic_friction, 1.0e-9)
    visco["retained_stress_asymptote"] = payload["retained_stress_asymptote"]
    damage["reference_crush_pressure_pa"] = payload["reference_crush_pressure_pa"]
    damage["abrasion_work_to_failure_j_seed"] = payload["abrasion_work_to_failure_j"]
    return sampled, payload


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
    return asymptote + (1.0 - asymptote) * (fast_weight * fast + (1.0 - fast_weight) * slow)


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


@dataclass(frozen=True)
class SutureVisualMesh:
    points: tuple[tuple[float, float, float], ...]
    normals: tuple[tuple[float, float, float], ...]
    face_vertex_counts: tuple[int, ...]
    face_vertex_indices: tuple[int, ...]
    extent_min: tuple[float, float, float]
    extent_max: tuple[float, float, float]
    minimum_radius_m: float
    maximum_radius_m: float


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
    axial_damping = 2.0 * float(material["axial_damping_ratio"]) * math.sqrt(axial_stiffness * reduced_mass)
    bend_stiffness = float(material["flexural_rigidity_n_m2"]) / spacing
    polar_inertia = 0.5 * segment_mass * radius * radius
    bend_damping = 2.0 * float(material["bending_damping_ratio"]) * math.sqrt(max(bend_stiffness * polar_inertia, 0.0))
    twist_stiffness = float(material["torsional_rigidity_n_m2"]) / spacing
    straight_failure = float(profile["tension"]["straight_failure_load_n"])
    knot_failure = straight_failure * float(profile["knot"]["nominal_strength_efficiency"])
    swage_segments = max(1, round(float(profile["swage"]["transition_length_m"]) / spacing))
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


def suture_segment_collision_radius(
    profile: dict[str, Any],
    segment_index: int,
    *,
    derived: DerivedSuture | None = None,
) -> float:
    """Return the authored capsule radius for one swage-tapered segment."""

    model = derived or derive(profile)
    index = int(segment_index)
    if index < 0 or index >= model.segment_count:
        raise IndexError(f"suture segment index out of range: {index}")
    geometry = profile["geometry"]
    swage = profile["swage"]
    swage_fraction = clamp(
        1.0 - index / max(1, model.swage_segment_count - 1),
        0.0,
        1.0,
    )
    swage_radius = float(swage["needle_end_diameter_m"]) / 2.0
    taper_radius = model.radius_m + (swage_radius - model.radius_m) * swage_fraction
    modulation = float(geometry["surface_radius_modulation_fraction"])
    modulation_period = int(geometry["surface_modulation_period_segments"])
    roughness = 1.0 + modulation * math.sin(2.0 * math.pi * index / modulation_period)
    return taper_radius * roughness


def capsule_point_containment_margin(
    point: tuple[float, float, float],
    *,
    radius_m: float,
    cylinder_height_m: float,
) -> float:
    """Return positive clearance when a local point is inside an X-axis capsule."""

    radius = float(radius_m)
    cylinder_height = float(cylinder_height_m)
    if not math.isfinite(radius) or not math.isfinite(cylinder_height) or radius <= 0.0 or cylinder_height < 0.0:
        raise ValueError("capsule dimensions must be finite and nonnegative")
    axial_excess = max(abs(float(point[0])) - cylinder_height / 2.0, 0.0)
    radial_distance = math.hypot(float(point[1]), float(point[2]))
    distance_to_spine = math.hypot(axial_excess, radial_distance)
    return radius - distance_to_spine


def build_suture_visual_mesh(
    profile: dict[str, Any],
    segment_index: int,
    *,
    collision_radius_m: float | None = None,
    derived: DerivedSuture | None = None,
) -> SutureVisualMesh:
    """Build a closed, non-subdivided crossed-carrier relief mesh."""

    model = derived or derive(profile)
    index = int(segment_index)
    collision_radius = (
        suture_segment_collision_radius(profile, index, derived=model)
        if collision_radius_m is None
        else float(collision_radius_m)
    )
    if not math.isfinite(collision_radius) or collision_radius <= 0.0:
        raise ValueError("suture collision radius must be positive")
    visual = profile["geometry"]["visual_representation"]
    axial_samples = int(visual["axial_samples_per_segment"])
    radial_samples = int(visual["radial_samples"])
    if axial_samples < 3 or radial_samples < 12:
        raise ValueError("suture visual mesh resolution is too low")
    carrier_count = int(profile["geometry"]["carrier_count"])
    if carrier_count < 4 or carrier_count % 2:
        raise ValueError("suture carrier count must be even and at least four")
    pitch = float(visual["braid_pitch_m_seed"])
    relief_depth = float(visual["relief_depth_fraction"])
    sharpness = float(visual["crossing_softmax_sharpness"])
    if pitch <= 0.0 or not 0.0 < relief_depth < 0.25 or sharpness <= 0.0:
        raise ValueError("invalid suture braid visual parameters")
    half_carriers = carrier_count // 2
    spacing = model.segment_spacing_m
    left_boundary_radius = (
        collision_radius
        if index == 0
        else min(
            collision_radius,
            suture_segment_collision_radius(
                profile,
                index - 1,
                derived=model,
            ),
        )
    )
    right_boundary_radius = (
        collision_radius
        if index == model.segment_count - 1
        else min(
            collision_radius,
            suture_segment_collision_radius(
                profile,
                index + 1,
                derived=model,
            ),
        )
    )
    segment_start_x = index * spacing
    segment_center_x = (index + 0.5) * spacing

    def smoothstep(amount: float) -> float:
        value = clamp(amount, 0.0, 1.0)
        return value * value * (3.0 - 2.0 * value)

    def envelope_radius(global_x: float) -> float:
        amount = clamp((global_x - segment_start_x) / spacing, 0.0, 1.0)
        if amount <= 0.5:
            return lerp(
                left_boundary_radius,
                collision_radius,
                smoothstep(amount * 2.0),
            )
        return lerp(
            collision_radius,
            right_boundary_radius,
            smoothstep((amount - 0.5) * 2.0),
        )

    def weave(global_x: float, theta: float) -> float:
        axial_phase = 2.0 * math.pi * global_x / pitch
        carrier_phase = float(half_carriers) * theta
        first = 0.5 + 0.5 * math.cos(carrier_phase + axial_phase)
        second = 0.5 + 0.5 * math.cos(carrier_phase - axial_phase)
        raw = math.log(math.exp(sharpness * first) + math.exp(sharpness * second)) / sharpness
        minimum = math.log(2.0) / sharpness
        return clamp(raw - minimum, 0.0, 1.0)

    def surface_radius(global_x: float, theta: float) -> float:
        return envelope_radius(global_x) * (1.0 - relief_depth * (1.0 - weave(global_x, theta)))

    points: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    minimum_radius = math.inf
    maximum_radius = 0.0
    epsilon_x = min(spacing, pitch) * 1.0e-5
    epsilon_theta = 1.0e-5
    for axial_index in range(axial_samples):
        amount = axial_index / (axial_samples - 1)
        local_x = -spacing / 2.0 + spacing * amount
        global_x = segment_center_x + local_x
        for radial_index in range(radial_samples):
            theta = 2.0 * math.pi * radial_index / radial_samples
            cosine = math.cos(theta)
            sine = math.sin(theta)
            radius = surface_radius(global_x, theta)
            minimum_radius = min(minimum_radius, radius)
            maximum_radius = max(maximum_radius, radius)
            points.append((local_x, radius * cosine, radius * sine))
            radial_x = (surface_radius(global_x + epsilon_x, theta) - surface_radius(global_x - epsilon_x, theta)) / (
                2.0 * epsilon_x
            )
            radial_theta = (
                surface_radius(global_x, theta + epsilon_theta) - surface_radius(global_x, theta - epsilon_theta)
            ) / (2.0 * epsilon_theta)
            tangent_x = (
                1.0,
                radial_x * cosine,
                radial_x * sine,
            )
            tangent_theta = (
                0.0,
                radial_theta * cosine - radius * sine,
                radial_theta * sine + radius * cosine,
            )
            normal = (
                tangent_theta[1] * tangent_x[2] - tangent_theta[2] * tangent_x[1],
                tangent_theta[2] * tangent_x[0] - tangent_theta[0] * tangent_x[2],
                tangent_theta[0] * tangent_x[1] - tangent_theta[1] * tangent_x[0],
            )
            length = math.sqrt(sum(component * component for component in normal))
            normals.append(
                (
                    normal[0] / length,
                    normal[1] / length,
                    normal[2] / length,
                )
            )
    left_center = len(points)
    points.append((-spacing / 2.0, 0.0, 0.0))
    normals.append((-1.0, 0.0, 0.0))
    right_center = len(points)
    points.append((spacing / 2.0, 0.0, 0.0))
    normals.append((1.0, 0.0, 0.0))
    face_counts: list[int] = []
    face_indices: list[int] = []
    for axial_index in range(axial_samples - 1):
        left_ring = axial_index * radial_samples
        right_ring = (axial_index + 1) * radial_samples
        for radial_index in range(radial_samples):
            next_radial = (radial_index + 1) % radial_samples
            face_counts.append(4)
            face_indices.extend(
                (
                    left_ring + radial_index,
                    left_ring + next_radial,
                    right_ring + next_radial,
                    right_ring + radial_index,
                )
            )
    last_ring = (axial_samples - 1) * radial_samples
    for radial_index in range(radial_samples):
        next_radial = (radial_index + 1) % radial_samples
        face_counts.append(3)
        face_indices.extend(
            (
                left_center,
                next_radial,
                radial_index,
            )
        )
        face_counts.append(3)
        face_indices.extend(
            (
                right_center,
                last_ring + radial_index,
                last_ring + next_radial,
            )
        )
    extent_min = (
        min(point[0] for point in points),
        min(point[1] for point in points),
        min(point[2] for point in points),
    )
    extent_max = (
        max(point[0] for point in points),
        max(point[1] for point in points),
        max(point[2] for point in points),
    )
    return SutureVisualMesh(
        points=tuple(points),
        normals=tuple(normals),
        face_vertex_counts=tuple(face_counts),
        face_vertex_indices=tuple(face_indices),
        extent_min=extent_min,
        extent_max=extent_max,
        minimum_radius_m=minimum_radius,
        maximum_radius_m=maximum_radius,
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
