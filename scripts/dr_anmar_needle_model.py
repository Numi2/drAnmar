"""Pure parametric geometry and mass model for DrAnmar Needle."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NEEDLE_PROFILE_PATH = REPOSITORY_ROOT / "physics_next/needles/dr-anmar-needle-v1.json"
DEFAULT_MASS_PROPERTY_INTEGRATION_SLICES = 8192


@dataclass(frozen=True)
class NeedleMassProperties:
    integration_slices: int
    volume_m3: float
    mass_kg: float
    center_of_mass_m: tuple[float, float, float]
    inertia_tensor_kg_m2: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    diagonal_inertia_kg_m2: tuple[float, float, float]
    principal_axes_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class DerivedNeedle:
    arc_length_m: float
    arc_radians: float
    curvature_radius_m: float
    body_radius_m: float
    swage_radius_m: float
    tip_radius_m: float
    mass_kg: float
    mass_properties: NeedleMassProperties
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
    normals: tuple[tuple[float, float, float], ...]
    normal_indices: tuple[int, ...]
    extent_min: tuple[float, float, float]
    extent_max: tuple[float, float, float]


@dataclass(frozen=True)
class NeedleCollisionCapsule:
    center_m: tuple[float, float, float]
    axis_direction: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    physical_radius_m: float
    collision_radius_m: float
    cylinder_height_m: float
    chord_length_m: float
    curvature_sagitta_m: float
    visual_seam_margin_m: float
    contact_offset_m: float
    rest_offset_m: float

    @property
    def total_length_m(self) -> float:
        return self.cylinder_height_m + 2.0 * self.collision_radius_m

    @property
    def extent_min(self) -> tuple[float, float, float]:
        half_length = self.total_length_m / 2.0
        return (
            -half_length,
            -self.collision_radius_m,
            -self.collision_radius_m,
        )

    @property
    def extent_max(self) -> tuple[float, float, float]:
        half_length = self.total_length_m / 2.0
        return (
            half_length,
            self.collision_radius_m,
            self.collision_radius_m,
        )


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


def radius_slope_at_distance(
    profile: dict[str, Any],
    distance_m: float,
) -> float:
    """Return dr/ds for the smooth tip and swage taper profile."""

    construction = profile["construction"]
    length = float(construction["centerline_arc_length_m"])
    body_radius = float(construction["body_diameter_m"]) / 2.0
    tip_radius = float(construction["tip_end_diameter_m"]) / 2.0
    swage_radius = float(construction["swage_end_diameter_m"]) / 2.0
    tip_length = float(construction["tip_taper_length_m"])
    swage_length = float(construction["swage_transition_length_m"])
    distance = max(0.0, min(length, float(distance_m)))
    if 0.0 < distance < tip_length:
        amount = distance / tip_length
        return (body_radius - tip_radius) * 6.0 * amount * (1.0 - amount) / tip_length
    if length - swage_length < distance < length:
        amount = (distance - (length - swage_length)) / swage_length
        return (swage_radius - body_radius) * 6.0 * amount * (1.0 - amount) / swage_length
    return 0.0


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


def _normalized(
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    length = math.sqrt(sum(component * component for component in vector))
    if not math.isfinite(length) or length <= 1.0e-18:
        raise ValueError("cannot normalize a zero or non-finite vector")
    return (
        vector[0] / length,
        vector[1] / length,
        vector[2] / length,
    )


def _dot(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    return sum(left[axis] * right[axis] for axis in range(3))


def _cross(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def needle_surface_normal(
    profile: dict[str, Any],
    fraction: float,
    phi: float,
) -> tuple[float, float, float]:
    """Return the analytic outward normal of the tapered swept surface."""

    construction = profile["construction"]
    arc_length = float(construction["centerline_arc_length_m"])
    curvature_radius = arc_length / math.pi
    amount = max(0.0, min(1.0, float(fraction)))
    center, tangent = centerline_at(profile, amount)
    outward_radial = tuple(component / curvature_radius for component in center)
    radius = radius_at_distance(profile, amount * arc_length)
    radius_slope = radius_slope_at_distance(
        profile,
        amount * arc_length,
    )
    cosine = math.cos(phi)
    sine = math.sin(phi)
    cross_section_normal = (
        cosine * outward_radial[0],
        cosine * outward_radial[1],
        sine,
    )
    tangential_scale = 1.0 + radius * cosine / curvature_radius
    return _normalized(
        (
            cross_section_normal[0] - radius_slope * tangent[0] / tangential_scale,
            cross_section_normal[1] - radius_slope * tangent[1] / tangential_scale,
            cross_section_normal[2] - radius_slope * tangent[2] / tangential_scale,
        )
    )


def derive_needle_mass_properties(
    profile: dict[str, Any],
    *,
    integration_slices: int = DEFAULT_MASS_PROPERTY_INTEGRATION_SLICES,
) -> NeedleMassProperties:
    """Integrate the tapered needle solid and diagonalize its inertia tensor.

    Each midpoint slice analytically integrates its complete circular cross
    section normal to the curved centerline. The ``1 + u / R`` toroidal
    Jacobian accounts for the larger swept volume on the outside of the arc.
    The resulting tensor is shifted to the integrated center of mass before
    extracting principal axes.
    """

    if integration_slices < 32:
        raise ValueError("needle mass integration requires at least 32 slices")
    construction = profile["construction"]
    arc_length = float(construction["centerline_arc_length_m"])
    curvature_radius = arc_length / math.pi
    density = float(profile["material"]["density_kg_m3"])
    distance_step = arc_length / integration_slices
    mass = 0.0
    volume = 0.0
    first_moment = [0.0, 0.0, 0.0]
    inertia_origin = [[0.0, 0.0, 0.0] for _ in range(3)]
    for index in range(integration_slices):
        distance = (index + 0.5) * distance_step
        fraction = distance / arc_length
        center, _tangent = centerline_at(profile, fraction)
        radius = radius_at_distance(profile, distance)
        cross_section_area = math.pi * radius * radius
        cross_section_second_moment = math.pi * radius**4 / 4.0
        slice_volume = cross_section_area * distance_step
        slice_mass = density * slice_volume
        volume += slice_volume
        mass += slice_mass
        outward_radial = tuple(component / curvature_radius for component in center)
        binormal = (0.0, 0.0, 1.0)
        radial_first_moment = cross_section_second_moment / curvature_radius
        for axis in range(3):
            first_moment[axis] += (
                density
                * distance_step
                * (cross_section_area * center[axis] + radial_first_moment * outward_radial[axis])
            )

        # Raw second moment of the curved swept cross-section. Odd disk
        # moments vanish except ∫u(1 + u/R)dA = πa⁴/(4R).
        raw_second_moment = [[0.0, 0.0, 0.0] for _ in range(3)]
        for row in range(3):
            for column in range(3):
                raw_second_moment[row][column] = (
                    cross_section_area * center[row] * center[column]
                    + radial_first_moment
                    * (center[row] * outward_radial[column] + outward_radial[row] * center[column])
                    + cross_section_second_moment
                    * (outward_radial[row] * outward_radial[column] + binormal[row] * binormal[column])
                )
        second_moment_trace = sum(raw_second_moment[axis][axis] for axis in range(3))
        for row in range(3):
            for column in range(3):
                inertia_origin[row][column] += (
                    density
                    * distance_step
                    * ((second_moment_trace if row == column else 0.0) - raw_second_moment[row][column])
                )

    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("needle mass integration produced invalid mass")
    center_of_mass = (
        first_moment[0] / mass,
        first_moment[1] / mass,
        first_moment[2] / mass,
    )
    center_squared = sum(component * component for component in center_of_mass)
    inertia_center = [[0.0, 0.0, 0.0] for _ in range(3)]
    for row in range(3):
        for column in range(3):
            identity = 1.0 if row == column else 0.0
            inertia_center[row][column] = inertia_origin[row][column] - mass * (
                center_squared * identity - center_of_mass[row] * center_of_mass[column]
            )
    # Numerical integration preserves this planar symmetry exactly, but
    # symmetrize explicitly to keep future geometry changes deterministic.
    for row in range(3):
        for column in range(row + 1, 3):
            symmetric = 0.5 * (inertia_center[row][column] + inertia_center[column][row])
            inertia_center[row][column] = symmetric
            inertia_center[column][row] = symmetric

    inertia_xx = inertia_center[0][0]
    inertia_xy = inertia_center[0][1]
    inertia_yy = inertia_center[1][1]
    discriminant = math.hypot(
        inertia_xx - inertia_yy,
        2.0 * inertia_xy,
    )
    eigenvalue_low = 0.5 * (inertia_xx + inertia_yy - discriminant)
    eigenvalue_high = 0.5 * (inertia_xx + inertia_yy + discriminant)
    if abs(inertia_xy) > 1.0e-30:
        axis_x = inertia_xy
        axis_y = eigenvalue_low - inertia_xx
        axis_length = math.hypot(axis_x, axis_y)
        axis_x /= axis_length
        axis_y /= axis_length
    elif inertia_xx <= inertia_yy:
        axis_x, axis_y = 1.0, 0.0
    else:
        axis_x, axis_y = 0.0, 1.0
    # Eigenvector sign is physically equivalent; choose one canonical sign so
    # regenerated USD and deterministic reports remain byte-stable.
    if axis_x < 0.0 or (math.isclose(axis_x, 0.0, abs_tol=1.0e-15) and axis_y < 0.0):
        axis_x = -axis_x
        axis_y = -axis_y
    principal_yaw = math.atan2(axis_y, axis_x)
    principal_axes = (
        math.cos(principal_yaw / 2.0),
        0.0,
        0.0,
        math.sin(principal_yaw / 2.0),
    )
    return NeedleMassProperties(
        integration_slices=integration_slices,
        volume_m3=volume,
        mass_kg=mass,
        center_of_mass_m=center_of_mass,
        inertia_tensor_kg_m2=(
            (
                inertia_center[0][0],
                inertia_center[0][1],
                inertia_center[0][2],
            ),
            (
                inertia_center[1][0],
                inertia_center[1][1],
                inertia_center[1][2],
            ),
            (
                inertia_center[2][0],
                inertia_center[2][1],
                inertia_center[2][2],
            ),
        ),
        diagonal_inertia_kg_m2=(
            eigenvalue_low,
            eigenvalue_high,
            inertia_center[2][2],
        ),
        principal_axes_wxyz=principal_axes,
    )


def reconstruct_inertia_tensor(
    diagonal_inertia_kg_m2: tuple[float, float, float],
    principal_axes_wxyz: tuple[float, float, float, float],
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Reconstruct a body-frame tensor from USD principal-axis attributes."""

    quaternion_norm = math.sqrt(sum(value * value for value in principal_axes_wxyz))
    if quaternion_norm <= 0.0:
        raise ValueError("principal-axis quaternion has zero length")
    w, x, y, z = (value / quaternion_norm for value in principal_axes_wxyz)
    rotation = (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ),
        (
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ),
        (
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )

    def tensor_component(row: int, column: int) -> float:
        return sum(rotation[row][axis] * diagonal_inertia_kg_m2[axis] * rotation[column][axis] for axis in range(3))

    return (
        (
            tensor_component(0, 0),
            tensor_component(0, 1),
            tensor_component(0, 2),
        ),
        (
            tensor_component(1, 0),
            tensor_component(1, 1),
            tensor_component(1, 2),
        ),
        (
            tensor_component(2, 0),
            tensor_component(2, 1),
            tensor_component(2, 2),
        ),
    )


def derive_needle(profile: dict[str, Any]) -> DerivedNeedle:
    construction = profile["construction"]
    arc_length = float(construction["centerline_arc_length_m"])
    arc_radians = math.pi
    curvature_radius = arc_length / arc_radians
    body_radius = float(construction["body_diameter_m"]) / 2.0
    swage_radius = float(construction["swage_end_diameter_m"]) / 2.0
    tip_radius = float(construction["tip_end_diameter_m"]) / 2.0
    mass_properties = derive_needle_mass_properties(profile)
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
        mass_kg=mass_properties.mass_kg,
        mass_properties=mass_properties,
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
    tip_point, tip_tangent = centerline_at(profile, 0.0)
    points: list[tuple[float, float, float]] = [tip_point]
    normals: list[tuple[float, float, float]] = [(-tip_tangent[0], -tip_tangent[1], -tip_tangent[2])]
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
            normals.append(
                needle_surface_normal(
                    profile,
                    fraction,
                    phi,
                )
            )
    counts: list[int] = []
    indices: list[int] = []
    normal_indices: list[int] = []
    first_ring = 1
    for radial_index in range(radial_samples):
        counts.append(3)
        triangle = (
            0,
            first_ring + radial_index,
            first_ring + (radial_index + 1) % radial_samples,
        )
        indices.extend(triangle)
        normal_indices.extend(triangle)
    ring_count = centerline_samples - 1
    for ring_index in range(ring_count - 1):
        left = 1 + ring_index * radial_samples
        right = left + radial_samples
        for radial_index in range(radial_samples):
            next_radial = (radial_index + 1) % radial_samples
            counts.append(4)
            quad = (
                left + radial_index,
                right + radial_index,
                right + next_radial,
                left + next_radial,
            )
            indices.extend(quad)
            normal_indices.extend(quad)
    last_ring = 1 + (ring_count - 1) * radial_samples
    counts.append(radial_samples)
    indices.extend(last_ring + radial_index for radial_index in reversed(range(radial_samples)))
    _swage_center, swage_tangent = centerline_at(profile, 1.0)
    cap_normal_index = len(normals)
    normals.append(swage_tangent)
    normal_indices.extend(cap_normal_index for _ in range(radial_samples))
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
    return NeedleMesh(
        points=tuple(points),
        face_vertex_counts=tuple(counts),
        face_vertex_indices=tuple(indices),
        normals=tuple(normals),
        normal_indices=tuple(normal_indices),
        extent_min=extent_min,
        extent_max=extent_max,
    )


def needle_mesh_normal_quality(
    profile: dict[str, Any],
    mesh: NeedleMesh | None = None,
) -> dict[str, float | int | str]:
    """Measure normal validity, winding alignment, and analytic tangency."""

    visual = mesh if mesh is not None else build_needle_mesh(profile)
    if len(visual.normal_indices) != len(visual.face_vertex_indices):
        return {
            "interpolation": "faceVarying",
            "normal_value_count": len(visual.normals),
            "normal_index_count": len(visual.normal_indices),
            "face_corner_count": len(visual.face_vertex_indices),
            "normal_index_count_error": abs(len(visual.normal_indices) - len(visual.face_vertex_indices)),
        }
    normal_lengths = [math.sqrt(_dot(normal, normal)) for normal in visual.normals]
    non_finite_components = sum(not math.isfinite(component) for normal in visual.normals for component in normal)
    used_normal_indices = set(visual.normal_indices)
    invalid_normal_indices = sum(index < 0 or index >= len(visual.normals) for index in visual.normal_indices)
    if invalid_normal_indices:
        return {
            "interpolation": "faceVarying",
            "normal_value_count": len(visual.normals),
            "normal_index_count": len(visual.normal_indices),
            "face_corner_count": len(visual.face_vertex_indices),
            "invalid_normal_index_count": invalid_normal_indices,
        }

    minimum_face_area = math.inf
    minimum_alignment = math.inf
    non_outward_corner_count = 0
    cursor = 0
    for face_count in visual.face_vertex_counts:
        face_indices = visual.face_vertex_indices[cursor : cursor + face_count]
        face_normal_indices = visual.normal_indices[cursor : cursor + face_count]
        cursor += face_count
        origin = visual.points[face_indices[0]]
        area_vector = (0.0, 0.0, 0.0)
        for corner in range(1, face_count - 1):
            left_point = visual.points[face_indices[corner]]
            right_point = visual.points[face_indices[corner + 1]]
            left_edge = (
                left_point[0] - origin[0],
                left_point[1] - origin[1],
                left_point[2] - origin[2],
            )
            right_edge = (
                right_point[0] - origin[0],
                right_point[1] - origin[1],
                right_point[2] - origin[2],
            )
            triangle_area_vector = _cross(left_edge, right_edge)
            area_vector = (
                area_vector[0] + triangle_area_vector[0],
                area_vector[1] + triangle_area_vector[1],
                area_vector[2] + triangle_area_vector[2],
            )
        area_m2 = 0.5 * math.sqrt(_dot(area_vector, area_vector))
        minimum_face_area = min(minimum_face_area, area_m2)
        if area_m2 <= 1.0e-24:
            non_outward_corner_count += face_count
            minimum_alignment = min(minimum_alignment, -1.0)
            continue
        face_normal = _normalized(area_vector)
        for normal_index in face_normal_indices:
            alignment = _dot(
                face_normal,
                visual.normals[normal_index],
            )
            minimum_alignment = min(minimum_alignment, alignment)
            if alignment <= 0.0:
                non_outward_corner_count += 1

    construction = profile["construction"]
    centerline_samples = int(construction["visual_centerline_samples"])
    radial_samples = int(construction["visual_radial_samples"])
    arc_length = float(construction["centerline_arc_length_m"])
    curvature_radius = arc_length / math.pi
    maximum_tangent_error = 0.0
    minimum_radial_alignment = math.inf
    for centerline_index in range(1, centerline_samples):
        fraction = centerline_index / (centerline_samples - 1)
        center, tangent = centerline_at(profile, fraction)
        outward_radial = tuple(component / curvature_radius for component in center)
        radius = radius_at_distance(profile, fraction * arc_length)
        radius_slope = radius_slope_at_distance(profile, fraction * arc_length)
        for radial_index in range(radial_samples):
            phi = 2.0 * math.pi * radial_index / radial_samples
            cosine = math.cos(phi)
            sine = math.sin(phi)
            cross_section_normal = (
                cosine * outward_radial[0],
                cosine * outward_radial[1],
                sine,
            )
            circumferential_tangent = (
                -sine * outward_radial[0],
                -sine * outward_radial[1],
                cosine,
            )
            longitudinal_tangent = (
                (1.0 + radius * cosine / curvature_radius) * tangent[0] + radius_slope * cross_section_normal[0],
                (1.0 + radius * cosine / curvature_radius) * tangent[1] + radius_slope * cross_section_normal[1],
                radius_slope * cross_section_normal[2],
            )
            point_index = 1 + (centerline_index - 1) * radial_samples + radial_index
            normal = visual.normals[point_index]
            maximum_tangent_error = max(
                maximum_tangent_error,
                abs(
                    _dot(
                        normal,
                        _normalized(longitudinal_tangent),
                    )
                ),
                abs(
                    _dot(
                        normal,
                        circumferential_tangent,
                    )
                ),
            )
            minimum_radial_alignment = min(
                minimum_radial_alignment,
                _dot(normal, cross_section_normal),
            )

    cap_normal = visual.normals[-1]
    last_ring = 1 + (centerline_samples - 2) * radial_samples
    maximum_cap_side_dot = max(
        abs(_dot(cap_normal, visual.normals[last_ring + index])) for index in range(radial_samples)
    )
    return {
        "interpolation": "faceVarying",
        "normal_value_count": len(visual.normals),
        "normal_index_count": len(visual.normal_indices),
        "face_corner_count": len(visual.face_vertex_indices),
        "unused_normal_value_count": len(visual.normals) - len(used_normal_indices),
        "invalid_normal_index_count": invalid_normal_indices,
        "non_finite_normal_component_count": non_finite_components,
        "maximum_unit_length_error": max(abs(length - 1.0) for length in normal_lengths),
        "minimum_face_area_m2": minimum_face_area,
        "minimum_face_corner_alignment_dot": minimum_alignment,
        "non_outward_face_corner_count": non_outward_corner_count,
        "maximum_surface_tangent_dot": maximum_tangent_error,
        "minimum_cross_section_outward_dot": minimum_radial_alignment,
        "maximum_swage_cap_side_normal_dot": maximum_cap_side_dot,
    }


def build_needle_collision_capsules(
    profile: dict[str, Any],
) -> tuple[NeedleCollisionCapsule, ...]:
    """Partition the curved needle into bounded compound colliders.

    OpenUSD capsule ``height`` is the cylinder-spine length and excludes both
    spherical caps. Each collider's spine spans its assigned centerline chord;
    adjacent spherical endcaps intentionally overlap at partition boundaries.
    The base radius includes the exact circular-arc sagitta. A final minimal
    uniform seam margin is derived from the authored faces so every polygon is
    contained by at least one convex primitive.
    """

    construction = profile["construction"]
    count = int(construction["collision_capsule_count"])
    arc_length = float(construction["centerline_arc_length_m"])
    curvature_radius = arc_length / math.pi
    capsules: list[NeedleCollisionCapsule] = []
    for index in range(count):
        left_fraction = index / count
        right_fraction = (index + 1) / count
        middle_fraction = (index + 0.5) / count
        left, _ = centerline_at(profile, left_fraction)
        right, _ = centerline_at(profile, right_fraction)
        middle, tangent = centerline_at(profile, middle_fraction)
        chord_length = math.dist(left, right)
        half_chord = chord_length / 2.0
        sagitta = curvature_radius - math.sqrt(max(curvature_radius * curvature_radius - half_chord * half_chord, 0.0))
        physical_radius = max(
            radius_at_distance(
                profile,
                fraction * arc_length,
            )
            for fraction in (
                left_fraction,
                middle_fraction,
                right_fraction,
            )
        )
        collision_radius = physical_radius + sagitta
        cylinder_height = chord_length
        yaw = math.atan2(tangent[1], tangent[0])
        capsules.append(
            NeedleCollisionCapsule(
                center_m=middle,
                axis_direction=tangent,
                orientation_wxyz=(
                    math.cos(yaw / 2.0),
                    0.0,
                    0.0,
                    math.sin(yaw / 2.0),
                ),
                physical_radius_m=physical_radius,
                collision_radius_m=collision_radius,
                cylinder_height_m=cylinder_height,
                chord_length_m=chord_length,
                curvature_sagitta_m=sagitta,
                visual_seam_margin_m=0.0,
                contact_offset_m=0.0,
                rest_offset_m=0.0,
            )
        )
    raw_capsules = tuple(capsules)
    raw_face_margin = min(
        _face_containment_margins(
            build_needle_mesh(profile),
            raw_capsules,
        )
    )
    coverage_epsilon = float(construction["collision_contract"]["coverage_epsilon_m"])
    seam_margin = max(0.0, -raw_face_margin) + coverage_epsilon
    contact_offsets = construction["collision_contract"]["contact_offsets"]
    contact_offset_fraction = float(contact_offsets["collision_radius_fraction"])
    minimum_contact_offset = float(contact_offsets["minimum_m"])
    maximum_contact_offset = float(contact_offsets["maximum_m"])
    rest_offset = float(contact_offsets["rest_offset_m"])
    return tuple(
        replace(
            capsule,
            collision_radius_m=(capsule.collision_radius_m + seam_margin),
            visual_seam_margin_m=seam_margin,
            contact_offset_m=max(
                minimum_contact_offset,
                min(
                    maximum_contact_offset,
                    (capsule.collision_radius_m + seam_margin) * contact_offset_fraction,
                ),
            ),
            rest_offset_m=rest_offset,
        )
        for capsule in raw_capsules
    )


def point_collision_margin_m(
    point: tuple[float, float, float],
    capsule: NeedleCollisionCapsule,
) -> float:
    """Return positive distance inside one capsule and negative outside."""

    half_spine = capsule.cylinder_height_m / 2.0
    start = tuple(capsule.center_m[axis] - capsule.axis_direction[axis] * half_spine for axis in range(3))
    end = tuple(capsule.center_m[axis] + capsule.axis_direction[axis] * half_spine for axis in range(3))
    edge = tuple(end[axis] - start[axis] for axis in range(3))
    relative = tuple(point[axis] - start[axis] for axis in range(3))
    edge_squared = sum(value * value for value in edge)
    amount = (
        0.0
        if edge_squared <= 1.0e-24
        else max(
            0.0,
            min(
                1.0,
                sum(relative[axis] * edge[axis] for axis in range(3)) / edge_squared,
            ),
        )
    )
    closest = tuple(start[axis] + amount * edge[axis] for axis in range(3))
    distance = math.dist(point, closest)
    return capsule.collision_radius_m - distance


def _face_containment_margins(
    visual: NeedleMesh,
    capsules: tuple[NeedleCollisionCapsule, ...],
) -> list[float]:
    point_capsule_margins = [
        [point_collision_margin_m(point, capsule) for capsule in capsules] for point in visual.points
    ]
    face_margins: list[float] = []
    cursor = 0
    for face_count in visual.face_vertex_counts:
        face_indices = visual.face_vertex_indices[cursor : cursor + face_count]
        cursor += face_count
        face_margins.append(
            max(
                min(point_capsule_margins[point_index][capsule_index] for point_index in face_indices)
                for capsule_index in range(len(capsules))
            )
        )
    return face_margins


def needle_mesh_collision_coverage(
    profile: dict[str, Any],
    mesh: NeedleMesh | None = None,
) -> dict[str, float | int]:
    """Measure whether the compound primitive collision covers the render mesh.

    A capsule is convex. A polygon is therefore fully covered when all of its
    vertices are inside the same capsule. Checking that condition for every
    authored face is stronger than sampling vertices or face centroids alone.
    """

    visual = mesh if mesh is not None else build_needle_mesh(profile)
    capsules = build_needle_collision_capsules(profile)
    point_capsule_margins = [
        [point_collision_margin_m(point, capsule) for capsule in capsules] for point in visual.points
    ]
    point_margins = [max(capsule_margins) for capsule_margins in point_capsule_margins]
    face_margins = _face_containment_margins(
        visual,
        capsules,
    )
    tolerance = 1.0e-12
    return {
        "visual_vertex_count": len(visual.points),
        "visual_face_count": len(visual.face_vertex_counts),
        "uncovered_visual_vertex_count": sum(margin < -tolerance for margin in point_margins),
        "uncovered_visual_face_count": sum(margin < -tolerance for margin in face_margins),
        "minimum_visual_vertex_containment_margin_m": min(point_margins),
        "minimum_visual_face_containment_margin_m": min(face_margins),
        "maximum_visual_vertex_containment_margin_m": max(point_margins),
    }


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
    static_friction = generator.uniform(*map(float, contact["static_friction_range"]))
    dynamic_friction = min(
        static_friction,
        generator.uniform(*map(float, contact["dynamic_friction_range"])),
    )
    restitution = generator.uniform(*map(float, contact["restitution_range"]))
    surface_roughness = generator.uniform(*map(float, appearance["roughness_range"]))
    return NeedleEpisodeParameters(
        seed=int(seed),
        mass_kg=mass,
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=restitution,
        surface_roughness=surface_roughness,
    )
