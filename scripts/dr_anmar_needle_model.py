#!/usr/bin/env python3
"""Pure parametric geometry and mass model for DrAnmar Needle."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NEEDLE_PROFILE_PATH = (
    REPOSITORY_ROOT / "physics_next/needles/dr-anmar-needle-v1.json"
)


@dataclass(frozen=True)
class DerivedNeedle:
    arc_length_m: float
    arc_radians: float
    curvature_radius_m: float
    body_radius_m: float
    swage_radius_m: float
    tip_radius_m: float
    mass_kg: float
    swage_anchor_m: tuple[float, float, float]
    swage_tangent: tuple[float, float, float]
    visual_vertex_count: int
    visual_face_count: int
    collision_capsule_count: int


@dataclass(frozen=True)
class NeedleMesh:
    points: tuple[tuple[float, float, float], ...]
    face_vertex_counts: tuple[int, ...]
    face_vertex_indices: tuple[int, ...]
    extent_min: tuple[float, float, float]
    extent_max: tuple[float, float, float]


@dataclass(frozen=True)
class NeedleEpisodeParameters:
    seed: int
    mass_kg: float
    static_friction: float
    dynamic_friction: float
    restitution: float
    surface_roughness: float

    def payload(self) -> dict[str, float | int]:
        return {
            "seed": self.seed,
            "mass_kg": self.mass_kg,
            "static_friction": self.static_friction,
            "dynamic_friction": self.dynamic_friction,
            "restitution": self.restitution,
            "surface_roughness": self.surface_roughness,
        }


def load_needle_profile(
    path: Path = DEFAULT_NEEDLE_PROFILE_PATH,
) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def smoothstep(value: float) -> float:
    amount = max(0.0, min(1.0, value))
    return amount * amount * (3.0 - 2.0 * amount)


def radius_at_distance(profile: dict[str, Any], distance_m: float) -> float:
    construction = profile["construction"]
    length = float(construction["centerline_arc_length_m"])
    body_radius = float(construction["body_diameter_m"]) / 2.0
    tip_radius = float(construction["tip_end_diameter_m"]) / 2.0
    swage_radius = float(construction["swage_end_diameter_m"]) / 2.0
    tip_length = float(construction["tip_taper_length_m"])
    swage_length = float(construction["swage_transition_length_m"])
    distance = max(0.0, min(length, float(distance_m)))
    if distance < tip_length:
        amount = smoothstep(distance / tip_length)
        return tip_radius + (body_radius - tip_radius) * amount
    if distance > length - swage_length:
        amount = smoothstep((distance - (length - swage_length)) / swage_length)
        return body_radius + (swage_radius - body_radius) * amount
    return body_radius


def centerline_at(
    profile: dict[str, Any],
    fraction: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    construction = profile["construction"]
    arc_length = float(construction["centerline_arc_length_m"])
    arc_radians = math.pi
    curvature_radius = arc_length / arc_radians
    amount = max(0.0, min(1.0, float(fraction)))
    theta = -math.pi / 2.0 + arc_radians * amount
    point = (
        curvature_radius * math.cos(theta),
        curvature_radius * math.sin(theta),
        0.0,
    )
    tangent = (-math.sin(theta), math.cos(theta), 0.0)
    return point, tangent


def derive_needle(profile: dict[str, Any]) -> DerivedNeedle:
    construction = profile["construction"]
    arc_length = float(construction["centerline_arc_length_m"])
    arc_radians = math.pi
    curvature_radius = arc_length / arc_radians
    body_radius = float(construction["body_diameter_m"]) / 2.0
    swage_radius = float(construction["swage_end_diameter_m"]) / 2.0
    tip_radius = float(construction["tip_end_diameter_m"]) / 2.0
    samples = 2048
    distance_step = arc_length / samples
    volume = 0.0
    for index in range(samples):
        distance = (index + 0.5) * distance_step
        radius = radius_at_distance(profile, distance)
        volume += math.pi * radius * radius * distance_step
    mass = volume * float(profile["material"]["density_kg_m3"])
    swage_anchor, swage_tangent = centerline_at(profile, 1.0)
    centerline_samples = int(construction["visual_centerline_samples"])
    radial_samples = int(construction["visual_radial_samples"])
    ring_count = centerline_samples - 1
    visual_vertex_count = 1 + ring_count * radial_samples
    visual_face_count = radial_samples + (ring_count - 1) * radial_samples + 1
    return DerivedNeedle(
        arc_length_m=arc_length,
        arc_radians=arc_radians,
        curvature_radius_m=curvature_radius,
        body_radius_m=body_radius,
        swage_radius_m=swage_radius,
        tip_radius_m=tip_radius,
        mass_kg=mass,
        swage_anchor_m=swage_anchor,
        swage_tangent=swage_tangent,
        visual_vertex_count=visual_vertex_count,
        visual_face_count=visual_face_count,
        collision_capsule_count=int(construction["collision_capsule_count"]),
    )


def build_needle_mesh(profile: dict[str, Any]) -> NeedleMesh:
    construction = profile["construction"]
    centerline_samples = int(construction["visual_centerline_samples"])
    radial_samples = int(construction["visual_radial_samples"])
    arc_length = float(construction["centerline_arc_length_m"])
    tip_point, _tip_tangent = centerline_at(profile, 0.0)
    points: list[tuple[float, float, float]] = [tip_point]
    for centerline_index in range(1, centerline_samples):
        fraction = centerline_index / (centerline_samples - 1)
        center, _tangent = centerline_at(profile, fraction)
        theta = -math.pi / 2.0 + math.pi * fraction
        radial = (math.cos(theta), math.sin(theta), 0.0)
        radius = radius_at_distance(profile, fraction * arc_length)
        for radial_index in range(radial_samples):
            phi = 2.0 * math.pi * radial_index / radial_samples
            points.append(
                (
                    center[0] + radius * math.cos(phi) * radial[0],
                    center[1] + radius * math.cos(phi) * radial[1],
                    radius * math.sin(phi),
                )
            )
    counts: list[int] = []
    indices: list[int] = []
    first_ring = 1
    for radial_index in range(radial_samples):
        counts.append(3)
        indices.extend(
            (
                0,
                first_ring + radial_index,
                first_ring + (radial_index + 1) % radial_samples,
            )
        )
    ring_count = centerline_samples - 1
    for ring_index in range(ring_count - 1):
        left = 1 + ring_index * radial_samples
        right = left + radial_samples
        for radial_index in range(radial_samples):
            next_radial = (radial_index + 1) % radial_samples
            counts.append(4)
            indices.extend(
                (
                    left + radial_index,
                    right + radial_index,
                    right + next_radial,
                    left + next_radial,
                )
            )
    last_ring = 1 + (ring_count - 1) * radial_samples
    counts.append(radial_samples)
    indices.extend(
        last_ring + radial_index
        for radial_index in reversed(range(radial_samples))
    )
    extent_min = tuple(min(point[axis] for point in points) for axis in range(3))
    extent_max = tuple(max(point[axis] for point in points) for axis in range(3))
    return NeedleMesh(
        points=tuple(points),
        face_vertex_counts=tuple(counts),
        face_vertex_indices=tuple(indices),
        extent_min=extent_min,
        extent_max=extent_max,
    )


def sample_episode_parameters(
    profile: dict[str, Any],
    seed: int,
) -> NeedleEpisodeParameters:
    """Sample the implemented sim-to-real domain with deterministic replay."""

    generator = random.Random(int(seed))
    derived = derive_needle(profile)
    material = profile["material"]
    contact = profile["contact"]
    appearance = profile["appearance"]
    density = generator.uniform(*map(float, material["density_range_kg_m3"]))
    mass = derived.mass_kg * density / float(material["density_kg_m3"])
    static_friction = generator.uniform(
        *map(float, contact["static_friction_range"])
    )
    dynamic_friction = min(
        static_friction,
        generator.uniform(*map(float, contact["dynamic_friction_range"])),
    )
    restitution = generator.uniform(
        *map(float, contact["restitution_range"])
    )
    surface_roughness = generator.uniform(
        *map(float, appearance["roughness_range"])
    )
    return NeedleEpisodeParameters(
        seed=int(seed),
        mass_kg=mass,
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=restitution,
        surface_roughness=surface_roughness,
    )
