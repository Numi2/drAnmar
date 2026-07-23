#!/usr/bin/env python3
"""Pure geometry, mechanics, and uncertainty model for DrAnmar hemostasis assets."""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HEMOSTASIS_PROFILE_PATH = (
    REPOSITORY_ROOT / "physics_next/vessels/dr-anmar-hemostasis-v1.json"
)
MMHG_TO_PA = 133.32236842105263

Vec3 = tuple[float, float, float]
Tet = tuple[int, int, int, int]
Triangle = tuple[int, int, int]


@dataclass(frozen=True)
class VesselMesh:
    points: tuple[Vec3, ...]
    tetrahedra: tuple[Tet, ...]
    tetrahedron_groups: dict[str, tuple[int, ...]]
    surface_triangles: tuple[Triangle, ...]
    surface_groups: dict[str, tuple[int, ...]]
    extent_min: Vec3
    extent_max: Vec3
    wall_volume_m3: float
    lumen_volume_m3: float
    minimum_tetra_volume_m3: float
    connected_components: int


@dataclass(frozen=True)
class ClipMesh:
    points: tuple[Vec3, ...]
    triangles: tuple[Triangle, ...]
    centerline: tuple[Vec3, ...]
    extent_min: Vec3
    extent_max: Vec3
    centerline_length_m: float
    material_volume_m3: float
    mass_kg: float


@dataclass(frozen=True)
class DerivedHemostasis:
    vessel_point_count: int
    vessel_tetrahedron_count: int
    vessel_surface_triangle_count: int
    vessel_wall_mass_kg: float
    vessel_lumen_volume_ml: float
    inlet_inner_diameter_m: float
    outlet_inner_diameter_m: float
    attachment_node_count: int
    clip_point_count: int
    clip_triangle_count: int
    clip_centerline_segment_count: int
    clip_mass_kg: float


@dataclass(frozen=True)
class HemostasisEpisodeParameters:
    seed: int
    outer_diameter_m: float
    wall_thickness_m: float
    youngs_modulus_pa: float
    axial_prestretch: float
    compliance_fraction_per_100_mmhg: float
    fluid_viscosity_pa_s: float
    static_friction: float
    wetness: float
    clip_yield_strength_pa: float
    clip_closing_force_n: float
    retention_force_n: float
    proof_pressure_mmhg: float

    def payload(self) -> dict[str, float | int]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True)
class ClipClosure:
    applied_force_n: float
    loaded_gap_m: float
    residual_gap_m: float
    plastic_fraction: float
    springback_m: float


@dataclass(frozen=True)
class OcclusionState:
    remaining_lumen_area_fraction: float
    occlusion_fraction: float
    leak_rate_ml_min: float
    crush_damage: float
    qualified_geometry: bool


def load_hemostasis_profile(
    path: Path = DEFAULT_HEMOSTASIS_PROFILE_PATH,
) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def signed_tetra_volume(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> float:
    ab = tuple(b[index] - a[index] for index in range(3))
    ac = tuple(c[index] - a[index] for index in range(3))
    ad = tuple(d[index] - a[index] for index in range(3))
    cross = (
        ac[1] * ad[2] - ac[2] * ad[1],
        ac[2] * ad[0] - ac[0] * ad[2],
        ac[0] * ad[1] - ac[1] * ad[0],
    )
    return sum(ab[index] * cross[index] for index in range(3)) / 6.0


def _triangle_normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3:
    ab = tuple(b[index] - a[index] for index in range(3))
    ac = tuple(c[index] - a[index] for index in range(3))
    return (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )


def _centerline(profile: dict[str, Any], axial_fraction: float) -> Vec3:
    geometry = profile["geometry"]
    length = float(geometry["vessel_length_m"])
    amplitude = float(geometry["centerline_curvature_amplitude_m"])
    centered = 2.0 * axial_fraction - 1.0
    return (
        -length / 2.0 + length * axial_fraction,
        amplitude * (1.0 - centered * centered),
        0.2 * amplitude * math.sin(math.pi * axial_fraction),
    )


def _outer_radius(profile: dict[str, Any], axial_fraction: float) -> float:
    geometry = profile["geometry"]
    seed_radius = float(geometry["outer_diameter_m"]) / 2.0
    taper = float(geometry["taper_fraction"])
    return seed_radius * (1.0 + taper * (0.5 - axial_fraction))


def _orient_surface_face(
    profile: dict[str, Any],
    points: list[Vec3],
    face: Triangle,
    group: str,
) -> Triangle:
    a, b, c = (points[index] for index in face)
    centroid = tuple((a[index] + b[index] + c[index]) / 3.0 for index in range(3))
    normal = _triangle_normal(a, b, c)
    length = float(profile["geometry"]["vessel_length_m"])
    axial_fraction = min(1.0, max(0.0, (centroid[0] + length / 2.0) / length))
    center = _centerline(profile, axial_fraction)
    radial = (0.0, centroid[1] - center[1], centroid[2] - center[2])
    if group == "outer":
        desired = sum(normal[index] * radial[index] for index in range(3)) > 0.0
    elif group == "inner":
        desired = sum(normal[index] * radial[index] for index in range(3)) < 0.0
    elif group == "inlet":
        desired = normal[0] < 0.0
    else:
        desired = normal[0] > 0.0
    return face if desired else (face[0], face[2], face[1])


def build_vessel_mesh(profile: dict[str, Any]) -> VesselMesh:
    geometry = profile["geometry"]
    wall_thickness = float(geometry["wall_thickness_m"])
    ellipticity = float(geometry["ellipticity_fraction"])
    circumferential_cells = int(geometry["circumferential_cells"])
    axial_cells = int(geometry["axial_cells"])
    radial_cells = int(geometry["radial_cells"])
    layer_by_radial_cell = {
        int(layer["radial_cell"]): str(layer["id"]) for layer in profile["wall_layers"]
    }
    if set(layer_by_radial_cell) != set(range(radial_cells)):
        raise ValueError("Vessel wall layers must cover every radial cell exactly once")

    points: list[Vec3] = []
    point_meta: list[tuple[int, int, int]] = []
    point_index: dict[tuple[int, int, int], int] = {}
    for axial_index in range(axial_cells + 1):
        axial_fraction = axial_index / axial_cells
        center = _centerline(profile, axial_fraction)
        outer_radius = _outer_radius(profile, axial_fraction)
        inner_radius = outer_radius - wall_thickness
        if inner_radius <= 0.0:
            raise ValueError("Vessel wall thickness must leave a positive lumen")
        for radial_index in range(radial_cells + 1):
            radial_fraction = radial_index / radial_cells
            radius = inner_radius + wall_thickness * radial_fraction
            for circumferential_index in range(circumferential_cells):
                theta = 2.0 * math.pi * circumferential_index / circumferential_cells
                point = (
                    center[0],
                    center[1] + radius * (1.0 + ellipticity) * math.cos(theta),
                    center[2] + radius * (1.0 - ellipticity) * math.sin(theta),
                )
                point_index[(axial_index, radial_index, circumferential_index)] = len(
                    points
                )
                points.append(point)
                point_meta.append((axial_index, radial_index, circumferential_index))

    def vertex(
        axial_index: int,
        radial_index: int,
        circumferential_index: int,
    ) -> int:
        return point_index[
            (
                axial_index,
                radial_index,
                circumferential_index % circumferential_cells,
            )
        ]

    cell_pattern = (
        (0, 1, 3, 7),
        (0, 3, 2, 7),
        (0, 2, 6, 7),
        (0, 6, 4, 7),
        (0, 4, 5, 7),
        (0, 5, 1, 7),
    )
    tetrahedra: list[Tet] = []
    tetrahedron_groups: dict[str, list[int]] = {
        str(layer["id"]): [] for layer in profile["wall_layers"]
    }
    minimum_volume = math.inf
    wall_volume = 0.0
    for axial_index in range(axial_cells):
        for radial_index in range(radial_cells):
            layer_name = layer_by_radial_cell[radial_index]
            for circumferential_index in range(circumferential_cells):
                next_circumferential = circumferential_index + 1
                corners = (
                    vertex(axial_index, radial_index, circumferential_index),
                    vertex(axial_index + 1, radial_index, circumferential_index),
                    vertex(
                        axial_index,
                        radial_index + 1,
                        circumferential_index,
                    ),
                    vertex(
                        axial_index + 1,
                        radial_index + 1,
                        circumferential_index,
                    ),
                    vertex(
                        axial_index,
                        radial_index,
                        next_circumferential,
                    ),
                    vertex(
                        axial_index + 1,
                        radial_index,
                        next_circumferential,
                    ),
                    vertex(
                        axial_index,
                        radial_index + 1,
                        next_circumferential,
                    ),
                    vertex(
                        axial_index + 1,
                        radial_index + 1,
                        next_circumferential,
                    ),
                )
                for local_tetrahedron in cell_pattern:
                    tetrahedron = tuple(corners[index] for index in local_tetrahedron)
                    volume = signed_tetra_volume(
                        *(points[index] for index in tetrahedron)
                    )
                    if volume < 0.0:
                        tetrahedron = (
                            tetrahedron[1],
                            tetrahedron[0],
                            tetrahedron[2],
                            tetrahedron[3],
                        )
                        volume = -volume
                    if volume <= 1.0e-17:
                        raise ValueError(
                            "Vessel mesh contains a degenerate tetrahedron"
                        )
                    tetrahedron_groups[layer_name].append(len(tetrahedra))
                    tetrahedra.append(tetrahedron)
                    minimum_volume = min(minimum_volume, volume)
                    wall_volume += volume

    faces: dict[tuple[int, int, int], Triangle] = {}
    face_counts: Counter[tuple[int, int, int]] = Counter()
    for a, b, c, d in tetrahedra:
        for face in ((b, c, d), (a, d, c), (a, b, d), (a, c, b)):
            key = tuple(sorted(face))
            face_counts[key] += 1
            faces.setdefault(key, face)
    if any(count not in (1, 2) for count in face_counts.values()):
        raise ValueError("Vessel tetrahedral mesh contains a non-manifold face")

    surface_triangles: list[Triangle] = []
    surface_groups: dict[str, list[int]] = {
        "outer": [],
        "inner": [],
        "inlet": [],
        "outlet": [],
    }
    for key in sorted(key for key, count in face_counts.items() if count == 1):
        face = faces[key]
        metadata = [point_meta[index] for index in face]
        if all(item[1] == 0 for item in metadata):
            group = "inner"
        elif all(item[1] == radial_cells for item in metadata):
            group = "outer"
        elif all(item[0] == 0 for item in metadata):
            group = "inlet"
        elif all(item[0] == axial_cells for item in metadata):
            group = "outlet"
        else:
            raise ValueError("Unclassified vessel boundary triangle")
        oriented = _orient_surface_face(profile, points, face, group)
        surface_groups[group].append(len(surface_triangles))
        surface_triangles.append(oriented)

    lumen_volume = 0.0
    length = float(geometry["vessel_length_m"])
    axial_step = length / axial_cells
    for axial_index in range(axial_cells):
        radii = []
        for station in (axial_index, axial_index + 1):
            axial_fraction = station / axial_cells
            radii.append(_outer_radius(profile, axial_fraction) - wall_thickness)
        area_a = math.pi * radii[0] ** 2 * (1.0 - ellipticity**2)
        area_b = math.pi * radii[1] ** 2 * (1.0 - ellipticity**2)
        lumen_volume += axial_step * (area_a + area_b) / 2.0

    return VesselMesh(
        points=tuple(points),
        tetrahedra=tuple(tetrahedra),
        tetrahedron_groups={
            name: tuple(indices) for name, indices in tetrahedron_groups.items()
        },
        surface_triangles=tuple(surface_triangles),
        surface_groups={
            name: tuple(indices) for name, indices in surface_groups.items()
        },
        extent_min=tuple(min(point[axis] for point in points) for axis in range(3)),
        extent_max=tuple(max(point[axis] for point in points) for axis in range(3)),
        wall_volume_m3=wall_volume,
        lumen_volume_m3=lumen_volume,
        minimum_tetra_volume_m3=minimum_volume,
        connected_components=1,
    )


def _distance(a: Vec3, b: Vec3) -> float:
    return math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(3)))


def build_clip_mesh(profile: dict[str, Any]) -> ClipMesh:
    clip = profile["clip"]
    arm_length = float(clip["arm_length_m"])
    crown_radius = float(clip["crown_radius_m"])
    arm_segments = int(clip["arm_segments"])
    crown_segments = int(clip["crown_segments"])
    section_segments = int(clip["section_segments"])
    section_width = float(clip["section_width_m"])
    section_thickness = float(clip["section_thickness_m"])
    serration_pitch = float(clip["inner_serration_pitch_m"])
    serration_depth = float(clip["inner_serration_depth_m"])
    x_tip = arm_length / 2.0
    x_crown = -arm_length / 2.0

    centerline: list[Vec3] = []
    for index in range(arm_segments + 1):
        fraction = index / arm_segments
        centerline.append(
            (
                x_tip + (x_crown - x_tip) * fraction,
                crown_radius,
                0.0,
            )
        )
    for index in range(1, crown_segments + 1):
        theta = math.pi / 2.0 + math.pi * index / crown_segments
        centerline.append(
            (
                x_crown + crown_radius * math.cos(theta),
                crown_radius * math.sin(theta),
                0.0,
            )
        )
    for index in range(1, arm_segments + 1):
        fraction = index / arm_segments
        centerline.append(
            (
                x_crown + (x_tip - x_crown) * fraction,
                -crown_radius,
                0.0,
            )
        )

    points: list[Vec3] = []
    for centerline_index, center in enumerate(centerline):
        if centerline_index == 0:
            neighbor_before = centerline[0]
            neighbor_after = centerline[1]
        elif centerline_index == len(centerline) - 1:
            neighbor_before = centerline[-2]
            neighbor_after = centerline[-1]
        else:
            neighbor_before = centerline[centerline_index - 1]
            neighbor_after = centerline[centerline_index + 1]
        tangent = tuple(
            neighbor_after[index] - neighbor_before[index] for index in range(3)
        )
        tangent_norm = math.sqrt(sum(value * value for value in tangent))
        tangent = tuple(value / tangent_norm for value in tangent)
        in_plane_normal = (-tangent[1], tangent[0], 0.0)
        on_arm = (
            centerline_index <= arm_segments
            or centerline_index >= arm_segments + crown_segments
        )
        serration_height = 0.0
        if on_arm:
            serration_height = serration_depth * (
                0.5
                + 0.5
                * math.cos(2.0 * math.pi * (center[0] - x_crown) / serration_pitch)
            )
        for section_index in range(section_segments):
            angle = 2.0 * math.pi * section_index / section_segments
            inner_weight = max(0.0, math.cos(angle))
            in_plane_offset = (
                section_width * 0.5 * math.cos(angle) + serration_height * inner_weight
            )
            points.append(
                (
                    center[0] + in_plane_normal[0] * in_plane_offset,
                    center[1] + in_plane_normal[1] * in_plane_offset,
                    center[2] + section_thickness * 0.5 * math.sin(angle),
                )
            )

    triangles: list[Triangle] = []
    ring_count = len(centerline)
    for ring_index in range(ring_count - 1):
        next_ring = ring_index + 1
        for section_index in range(section_segments):
            next_section = (section_index + 1) % section_segments
            a = ring_index * section_segments + section_index
            b = next_ring * section_segments + section_index
            c = next_ring * section_segments + next_section
            d = ring_index * section_segments + next_section
            triangles.extend(((a, b, c), (a, c, d)))
    start_center = len(points)
    points.append(centerline[0])
    end_center = len(points)
    points.append(centerline[-1])
    last_ring_start = (ring_count - 1) * section_segments
    for section_index in range(section_segments):
        next_section = (section_index + 1) % section_segments
        triangles.append((start_center, next_section, section_index))
        triangles.append(
            (
                end_center,
                last_ring_start + section_index,
                last_ring_start + next_section,
            )
        )

    centerline_length = sum(
        _distance(centerline[index], centerline[index + 1])
        for index in range(len(centerline) - 1)
    )
    cross_section_area = math.pi * section_width * section_thickness / 4.0
    material_volume = centerline_length * cross_section_area
    mass = material_volume * float(clip["density_kg_m3"])
    return ClipMesh(
        points=tuple(points),
        triangles=tuple(triangles),
        centerline=tuple(centerline),
        extent_min=tuple(min(point[axis] for point in points) for axis in range(3)),
        extent_max=tuple(max(point[axis] for point in points) for axis in range(3)),
        centerline_length_m=centerline_length,
        material_volume_m3=material_volume,
        mass_kg=mass,
    )


def derive_hemostasis(
    profile: dict[str, Any],
    vessel: VesselMesh | None = None,
    clip: ClipMesh | None = None,
) -> DerivedHemostasis:
    vessel_mesh = vessel or build_vessel_mesh(profile)
    clip_mesh = clip or build_clip_mesh(profile)
    geometry = profile["geometry"]
    length = float(geometry["vessel_length_m"])
    wall_thickness = float(geometry["wall_thickness_m"])
    attachment_width = float(profile["attachments"]["width_m"])
    inlet_inner_radius = _outer_radius(profile, 0.0) - wall_thickness
    outlet_inner_radius = _outer_radius(profile, 1.0) - wall_thickness
    attachment_nodes = sum(
        1
        for point in vessel_mesh.points
        if point[0] <= -length / 2.0 + attachment_width
        or point[0] >= length / 2.0 - attachment_width
    )
    return DerivedHemostasis(
        vessel_point_count=len(vessel_mesh.points),
        vessel_tetrahedron_count=len(vessel_mesh.tetrahedra),
        vessel_surface_triangle_count=len(vessel_mesh.surface_triangles),
        vessel_wall_mass_kg=(
            vessel_mesh.wall_volume_m3
            * float(profile["vessel_material"]["density_kg_m3_seed"])
        ),
        vessel_lumen_volume_ml=vessel_mesh.lumen_volume_m3 * 1.0e6,
        inlet_inner_diameter_m=2.0 * inlet_inner_radius,
        outlet_inner_diameter_m=2.0 * outlet_inner_radius,
        attachment_node_count=attachment_nodes,
        clip_point_count=len(clip_mesh.points),
        clip_triangle_count=len(clip_mesh.triangles),
        clip_centerline_segment_count=len(clip_mesh.centerline) - 1,
        clip_mass_kg=clip_mesh.mass_kg,
    )


def sample_hemostasis_episode_parameters(
    profile: dict[str, Any],
    seed: int,
) -> HemostasisEpisodeParameters:
    generator = random.Random(int(seed))
    geometry = profile["geometry"]
    vessel = profile["vessel_material"]
    lumen = profile["lumen"]
    clip = profile["clip"]
    contact = profile["contact"]
    occlusion = profile["occlusion"]
    return HemostasisEpisodeParameters(
        seed=int(seed),
        outer_diameter_m=generator.uniform(
            *map(float, geometry["outer_diameter_range_m"])
        ),
        wall_thickness_m=generator.uniform(
            *map(float, geometry["wall_thickness_range_m"])
        ),
        youngs_modulus_pa=generator.uniform(
            *map(float, vessel["youngs_modulus_range_pa"])
        ),
        axial_prestretch=generator.uniform(
            *map(float, vessel["axial_prestretch_range"])
        ),
        compliance_fraction_per_100_mmhg=generator.uniform(
            *map(float, vessel["low_pressure_compliance_range"])
        ),
        fluid_viscosity_pa_s=generator.uniform(
            *map(float, lumen["dynamic_viscosity_range_pa_s"])
        ),
        static_friction=generator.uniform(
            *map(float, contact["static_friction_range"])
        ),
        wetness=generator.uniform(*map(float, contact["wetness_range"])),
        clip_yield_strength_pa=generator.uniform(
            *map(float, clip["yield_strength_range_pa"])
        ),
        clip_closing_force_n=generator.uniform(
            *map(float, clip["closing_force_range_n"])
        ),
        retention_force_n=generator.uniform(
            *map(float, occlusion["retention_force_range_n"])
        ),
        proof_pressure_mmhg=generator.uniform(
            *map(float, occlusion["proof_pressure_range_mmhg"])
        ),
    )


def pressure_diameter_m(
    profile: dict[str, Any],
    *,
    pressure_mmhg: float,
    axial_prestretch: float | None = None,
    compliance_fraction_per_100_mmhg: float | None = None,
) -> float:
    vessel = profile["vessel_material"]
    lumen = profile["lumen"]
    reference_pressure = float(lumen["reference_pressure_mmhg"])
    compliance = (
        float(vessel["low_pressure_compliance_fraction_per_100_mmhg_seed"])
        if compliance_fraction_per_100_mmhg is None
        else float(compliance_fraction_per_100_mmhg)
    )
    recruitment = float(vessel["collagen_recruitment_pressure_mmhg_seed"])
    pressure_delta = float(pressure_mmhg) - reference_pressure
    if pressure_delta >= 0.0:
        strain = (
            compliance * recruitment / 100.0 * math.log1p(pressure_delta / recruitment)
        )
    else:
        strain = compliance * pressure_delta / 100.0
    prestretch = (
        float(vessel["axial_prestretch_seed"])
        if axial_prestretch is None
        else float(axial_prestretch)
    )
    axial_coupling = prestretch**-0.12
    return max(
        1.0e-6,
        float(profile["geometry"]["outer_diameter_m"])
        * (1.0 + strain)
        * axial_coupling,
    )


def poiseuille_flow_ml_min(
    *,
    pressure_drop_mmhg: float,
    lumen_radius_m: float,
    length_m: float,
    dynamic_viscosity_pa_s: float,
) -> float:
    if (
        pressure_drop_mmhg <= 0.0
        or lumen_radius_m <= 0.0
        or length_m <= 0.0
        or dynamic_viscosity_pa_s <= 0.0
    ):
        return 0.0
    pressure_pa = pressure_drop_mmhg * MMHG_TO_PA
    flow_m3_s = (
        math.pi
        * pressure_pa
        * lumen_radius_m**4
        / (8.0 * dynamic_viscosity_pa_s * length_m)
    )
    return flow_m3_s * 60.0 * 1.0e6


def clip_closure_state(
    profile: dict[str, Any],
    *,
    applied_force_n: float,
    closing_force_n: float | None = None,
    yield_strength_pa: float | None = None,
) -> ClipClosure:
    clip = profile["clip"]
    force = max(0.0, float(applied_force_n))
    closing_force = (
        float(clip["closing_force_n_seed"])
        if closing_force_n is None
        else max(1.0e-9, float(closing_force_n))
    )
    yield_strength = (
        float(clip["yield_strength_pa_seed"])
        if yield_strength_pa is None
        else max(1.0, float(yield_strength_pa))
    )
    yield_ratio = yield_strength / float(clip["yield_strength_pa_seed"])
    yield_onset_fraction = min(0.62, max(0.18, 0.32 * yield_ratio))
    open_gap = float(clip["inside_gap_m"])
    qualified_gap = float(clip["qualified_residual_gap_m_seed"])
    loading_fraction = min(1.25, force / closing_force)
    loaded_fraction = min(1.0, loading_fraction**0.72)
    loaded_gap = max(
        0.0,
        open_gap - (open_gap - qualified_gap * 0.35) * loaded_fraction,
    )
    plastic_fraction = max(
        0.0,
        min(
            1.0,
            (loading_fraction - yield_onset_fraction) / (1.0 - yield_onset_fraction),
        ),
    )
    residual_gap = open_gap - (open_gap - qualified_gap) * plastic_fraction
    springback = max(0.0, residual_gap - loaded_gap)
    return ClipClosure(
        applied_force_n=force,
        loaded_gap_m=loaded_gap,
        residual_gap_m=residual_gap,
        plastic_fraction=plastic_fraction,
        springback_m=springback,
    )


def vessel_occlusion_state(
    profile: dict[str, Any],
    *,
    clip_gap_m: float,
    pressure_mmhg: float,
    cycles: int = 1,
    applied_force_n: float | None = None,
) -> OcclusionState:
    geometry = profile["geometry"]
    occlusion = profile["occlusion"]
    lumen = profile["lumen"]
    outer_diameter = float(geometry["outer_diameter_m"])
    inner_diameter = outer_diameter - 2.0 * float(geometry["wall_thickness_m"])
    normalized_gap = max(0.0, min(1.0, clip_gap_m / outer_diameter))
    remaining_area_fraction = normalized_gap**2.15
    occlusion_fraction = 1.0 - remaining_area_fraction
    effective_lumen_radius = inner_diameter / 2.0 * remaining_area_fraction**1.5
    leak_rate = poiseuille_flow_ml_min(
        pressure_drop_mmhg=max(0.0, pressure_mmhg),
        lumen_radius_m=effective_lumen_radius,
        length_m=float(profile["clip"]["section_width_m"]),
        dynamic_viscosity_pa_s=float(lumen["dynamic_viscosity_pa_s_seed"]),
    )
    compression_fraction = 1.0 - normalized_gap
    onset = float(occlusion["damage_onset_compression_fraction"])
    damage = 0.0
    if compression_fraction > onset:
        utilization = (compression_fraction - onset) / (1.0 - onset)
        damage = utilization**2 * max(1, int(cycles)) ** 0.35
    if applied_force_n is not None:
        closing_force = float(profile["clip"]["closing_force_n_seed"])
        overload = max(0.0, float(applied_force_n) / closing_force - 1.0)
        damage += 0.4 * overload**1.5
    damage = min(float(occlusion["critical_crush_damage"]), damage)
    target = float(occlusion["target_lumen_area_reduction_fraction"])
    qualified_geometry = occlusion_fraction >= target and leak_rate <= float(
        occlusion["maximum_qualified_leak_rate_ml_min"]
    )
    return OcclusionState(
        remaining_lumen_area_fraction=remaining_area_fraction,
        occlusion_fraction=occlusion_fraction,
        leak_rate_ml_min=leak_rate,
        crush_damage=damage,
        qualified_geometry=qualified_geometry,
    )


def clip_retention_force_n(
    profile: dict[str, Any],
    *,
    closure_fraction: float,
    vessel_diameter_m: float,
    friction_coefficient: float,
    pull_angle_degrees: float,
) -> float:
    occlusion = profile["occlusion"]
    clip = profile["clip"]
    reference_diameter = float(profile["geometry"]["outer_diameter_m"])
    reference_friction = float(clip["surface_friction_seed"])
    diameter_factor = max(0.5, vessel_diameter_m / reference_diameter) ** 0.35
    closure_factor = max(0.0, min(1.0, closure_fraction)) ** 1.4
    friction_factor = max(0.1, friction_coefficient / reference_friction)
    angle_factor = 1.0 + 0.22 * abs(math.sin(math.radians(pull_angle_degrees)))
    return (
        float(occlusion["retention_force_n_seed"])
        * diameter_factor
        * closure_factor
        * friction_factor
        * angle_factor
    )
