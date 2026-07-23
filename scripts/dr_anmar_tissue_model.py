#!/usr/bin/env python3
"""Pure geometry, mechanics, and uncertainty model for DrAnmar Suturable Tissue."""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TISSUE_PROFILE_PATH = (
    REPOSITORY_ROOT / "physics_next/tissues/dr-anmar-suturable-tissue-v1.json"
)


Vec3 = tuple[float, float, float]
Tet = tuple[int, int, int, int]
Triangle = tuple[int, int, int]


@dataclass(frozen=True)
class TissueMesh:
    points: tuple[Vec3, ...]
    tetrahedra: tuple[Tet, ...]
    tetrahedron_groups: dict[str, tuple[int, ...]]
    surface_triangles: tuple[Triangle, ...]
    surface_groups: dict[str, tuple[int, ...]]
    extent_min: Vec3
    extent_max: Vec3
    volume_m3: float
    minimum_tetra_volume_m3: float
    connected_components: int


@dataclass(frozen=True)
class DerivedTissue:
    point_count: int
    tetrahedron_count: int
    surface_triangle_count: int
    mass_kg: float
    rest_wound_gap_bottom_m: float
    rest_wound_gap_top_m: float
    outer_attachment_node_count: int


@dataclass(frozen=True)
class TissueEpisodeParameters:
    seed: int
    density_kg_m3: float
    youngs_modulus_pa: float
    poisson_ratio: float
    damping_ratio: float
    static_friction: float
    dynamic_friction: float
    wetness: float
    puncture_force_n: float
    shaft_drag_n_per_m: float
    reference_pullout_force_n: float
    surface_roughness: float

    def payload(self) -> dict[str, float | int]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True)
class NeedleTissueForce:
    phase: str
    compression_n: float
    cutting_n: float
    shaft_friction_n: float
    total_n: float


def load_tissue_profile(
    path: Path = DEFAULT_TISSUE_PROFILE_PATH,
) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def signed_tetra_volume(
    a: Vec3,
    b: Vec3,
    c: Vec3,
    d: Vec3,
) -> float:
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


def _surface_faces(
    points: list[Vec3],
    tetrahedra: list[Tet],
    component_centers: tuple[Vec3, Vec3],
    point_components: list[int],
) -> list[Triangle]:
    faces: dict[tuple[int, int, int], Triangle] = {}
    counts: Counter[tuple[int, int, int]] = Counter()
    for a, b, c, d in tetrahedra:
        for face in ((b, c, d), (a, d, c), (a, b, d), (a, c, b)):
            key = tuple(sorted(face))
            counts[key] += 1
            faces.setdefault(key, face)
    if any(count not in (1, 2) for count in counts.values()):
        raise ValueError("Tetrahedral mesh contains a non-manifold face")
    surface: list[Triangle] = []
    for key in sorted(key for key, count in counts.items() if count == 1):
        face = faces[key]
        component = point_components[face[0]]
        center = component_centers[component]
        a, b, c = (points[index] for index in face)
        normal = _triangle_normal(a, b, c)
        centroid = tuple((a[index] + b[index] + c[index]) / 3.0 for index in range(3))
        outward = tuple(centroid[index] - center[index] for index in range(3))
        if sum(normal[index] * outward[index] for index in range(3)) < 0.0:
            face = (face[0], face[2], face[1])
        surface.append(face)
    return surface


def build_tissue_mesh(profile: dict[str, Any]) -> TissueMesh:
    geometry = profile["geometry"]
    width = float(geometry["overall_width_m"])
    depth = float(geometry["depth_m"])
    thickness = float(geometry["thickness_m"])
    gap = float(geometry["rest_wound_gap_m"])
    bevel = float(geometry["wound_bevel_m"])
    amplitude = float(geometry["wound_irregularity_amplitude_m"])
    wavelength = float(geometry["wound_irregularity_wavelength_m"])
    cells_x = int(geometry["cells_per_flap_x"])
    cells_y = int(geometry["cells_y"])
    cells_z = int(geometry["cells_z"])
    flap_width = (width - gap) / 2.0
    points: list[Vec3] = []
    point_components: list[int] = []
    point_index: dict[tuple[int, int, int, int], int] = {}

    for component in range(2):
        for z_index in range(cells_z + 1):
            z_fraction = z_index / cells_z
            z = -thickness / 2.0 + thickness * z_fraction
            for y_index in range(cells_y + 1):
                y_fraction = y_index / cells_y
                y = -depth / 2.0 + depth * y_fraction
                center_offset = amplitude * math.sin(
                    2.0 * math.pi * (y + depth / 2.0) / wavelength
                )
                if component == 0:
                    outer_x = -width / 2.0
                    inner_x = -gap / 2.0 + center_offset - bevel * z_fraction
                else:
                    inner_x = gap / 2.0 + center_offset + bevel * z_fraction
                    outer_x = width / 2.0
                for x_index in range(cells_x + 1):
                    fraction = x_index / cells_x
                    if component == 0:
                        x = outer_x + (inner_x - outer_x) * fraction
                    else:
                        x = inner_x + (outer_x - inner_x) * fraction
                    point_index[(component, x_index, y_index, z_index)] = len(points)
                    points.append((x, y, z))
                    point_components.append(component)

    def vertex(
        component: int,
        x_index: int,
        y_index: int,
        z_index: int,
    ) -> int:
        return point_index[(component, x_index, y_index, z_index)]

    cell_pattern = (
        (0, 1, 3, 7),
        (0, 3, 2, 7),
        (0, 2, 6, 7),
        (0, 6, 4, 7),
        (0, 4, 5, 7),
        (0, 5, 1, 7),
    )
    tetrahedra: list[Tet] = []
    minimum_volume = math.inf
    total_volume = 0.0
    for component in range(2):
        for z_index in range(cells_z):
            for y_index in range(cells_y):
                for x_index in range(cells_x):
                    corners = (
                        vertex(component, x_index, y_index, z_index),
                        vertex(component, x_index + 1, y_index, z_index),
                        vertex(component, x_index, y_index + 1, z_index),
                        vertex(component, x_index + 1, y_index + 1, z_index),
                        vertex(component, x_index, y_index, z_index + 1),
                        vertex(component, x_index + 1, y_index, z_index + 1),
                        vertex(component, x_index, y_index + 1, z_index + 1),
                        vertex(component, x_index + 1, y_index + 1, z_index + 1),
                    )
                    for local in cell_pattern:
                        tet = tuple(corners[index] for index in local)
                        volume = signed_tetra_volume(*(points[index] for index in tet))
                        if volume < 0.0:
                            tet = (tet[1], tet[0], tet[2], tet[3])
                            volume = -volume
                        if volume <= 1.0e-16:
                            raise ValueError(
                                "Tissue mesh contains a degenerate tetrahedron"
                            )
                        tetrahedra.append(tet)
                        minimum_volume = min(minimum_volume, volume)
                        total_volume += volume

    component_centers = (
        (-width / 4.0 - gap / 4.0, 0.0, 0.0),
        (width / 4.0 + gap / 4.0, 0.0, 0.0),
    )
    surface = _surface_faces(
        points,
        tetrahedra,
        component_centers,
        point_components,
    )
    tetrahedron_groups: dict[str, list[int]] = {
        str(layer["id"]): [] for layer in profile["layers"]
    }
    for tetrahedron_index, tetrahedron in enumerate(tetrahedra):
        centroid_z = sum(points[index][2] for index in tetrahedron) / 4.0
        depth_fraction = (centroid_z + thickness / 2.0) / thickness
        matching_layers = [
            str(layer["id"])
            for layer in profile["layers"]
            if float(layer["depth_fraction"][0])
            <= depth_fraction
            <= float(layer["depth_fraction"][1])
        ]
        if len(matching_layers) != 1:
            raise ValueError(
                "Every tetrahedron must belong to exactly one tissue layer"
            )
        tetrahedron_groups[matching_layers[0]].append(tetrahedron_index)

    maximum_z = thickness / 2.0
    minimum_z = -thickness / 2.0
    wound_limit = gap / 2.0 + bevel + amplitude + flap_width * 0.02
    groups: dict[str, list[int]] = {
        "surface": [],
        "bulk": [],
        "fascia": [],
        "wound": [],
    }
    for face_index, face in enumerate(surface):
        vertices = [points[index] for index in face]
        if all(abs(point[2] - maximum_z) <= 1.0e-10 for point in vertices):
            group = "surface"
        elif all(abs(point[2] - minimum_z) <= 1.0e-10 for point in vertices):
            group = "fascia"
        elif all(abs(point[0]) <= wound_limit for point in vertices):
            group = "wound"
        else:
            group = "bulk"
        groups[group].append(face_index)

    return TissueMesh(
        points=tuple(points),
        tetrahedra=tuple(tetrahedra),
        tetrahedron_groups={
            name: tuple(indices) for name, indices in tetrahedron_groups.items()
        },
        surface_triangles=tuple(surface),
        surface_groups={name: tuple(indices) for name, indices in groups.items()},
        extent_min=tuple(min(point[axis] for point in points) for axis in range(3)),
        extent_max=tuple(max(point[axis] for point in points) for axis in range(3)),
        volume_m3=total_volume,
        minimum_tetra_volume_m3=minimum_volume,
        connected_components=2,
    )


def derive_tissue(
    profile: dict[str, Any],
    mesh: TissueMesh | None = None,
) -> DerivedTissue:
    geometry = profile["geometry"]
    attachments = profile["attachments"]
    tissue_mesh = mesh or build_tissue_mesh(profile)
    width = float(geometry["overall_width_m"])
    attachment_width = float(attachments["width_m"])
    attachment_nodes = sum(
        1
        for point in tissue_mesh.points
        if point[0] <= -width / 2.0 + attachment_width
        or point[0] >= width / 2.0 - attachment_width
    )
    return DerivedTissue(
        point_count=len(tissue_mesh.points),
        tetrahedron_count=len(tissue_mesh.tetrahedra),
        surface_triangle_count=len(tissue_mesh.surface_triangles),
        mass_kg=(
            tissue_mesh.volume_m3
            * float(profile["intact_tissue"]["density_kg_m3_seed"])
        ),
        rest_wound_gap_bottom_m=float(geometry["rest_wound_gap_m"]),
        rest_wound_gap_top_m=(
            float(geometry["rest_wound_gap_m"]) + 2.0 * float(geometry["wound_bevel_m"])
        ),
        outer_attachment_node_count=attachment_nodes,
    )


def sample_tissue_episode_parameters(
    profile: dict[str, Any],
    seed: int,
) -> TissueEpisodeParameters:
    generator = random.Random(int(seed))
    intact = profile["intact_tissue"]
    contact = profile["contact"]
    puncture = profile["puncture"]
    holding = profile["suture_holding"]
    appearance = profile["appearance"]
    static_friction = generator.uniform(*map(float, contact["static_friction_range"]))
    dynamic_friction = min(
        static_friction,
        generator.uniform(*map(float, contact["dynamic_friction_range"])),
    )
    return TissueEpisodeParameters(
        seed=int(seed),
        density_kg_m3=generator.uniform(*map(float, intact["density_range_kg_m3"])),
        youngs_modulus_pa=generator.uniform(
            *map(float, intact["youngs_modulus_range_pa"])
        ),
        poisson_ratio=generator.uniform(*map(float, intact["poisson_ratio_range"])),
        damping_ratio=generator.uniform(*map(float, intact["damping_ratio_range"])),
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        wetness=generator.uniform(*map(float, contact["wetness_range"])),
        puncture_force_n=generator.uniform(
            *map(float, puncture["puncture_force_range_n"])
        ),
        shaft_drag_n_per_m=generator.uniform(
            *map(float, puncture["shaft_drag_n_per_m_range"])
        ),
        reference_pullout_force_n=generator.uniform(
            *map(float, holding["reference_pullout_force_range_n"])
        ),
        surface_roughness=generator.uniform(*map(float, appearance["roughness_range"])),
    )


def needle_tissue_force(
    profile: dict[str, Any],
    *,
    indentation_m: float,
    embedded_arc_length_m: float = 0.0,
    swept_area_m2: float = 0.0,
    punctured: bool = False,
    episode: TissueEpisodeParameters | None = None,
) -> NeedleTissueForce:
    puncture = profile["puncture"]
    puncture_force = (
        episode.puncture_force_n
        if episode
        else float(puncture["puncture_force_n_seed"])
    )
    shaft_drag = (
        episode.shaft_drag_n_per_m
        if episode
        else float(puncture["shaft_drag_n_per_m_seed"])
    )
    if not punctured:
        normalized = max(
            0.0,
            min(
                1.0,
                float(indentation_m) / float(puncture["prepuncture_depth_m_seed"]),
            ),
        )
        compression = puncture_force * normalized * normalized
        return NeedleTissueForce(
            phase="prepuncture",
            compression_n=compression,
            cutting_n=0.0,
            shaft_friction_n=0.0,
            total_n=compression,
        )
    cutting = puncture_force * float(puncture["cutting_force_fraction_seed"])
    compression = max(0.0, float(swept_area_m2)) * float(
        puncture["sweep_stiffness_n_m2_seed"]
    )
    friction = max(0.0, float(embedded_arc_length_m)) * shaft_drag
    return NeedleTissueForce(
        phase="penetration",
        compression_n=compression,
        cutting_n=cutting,
        shaft_friction_n=friction,
        total_n=compression + cutting + friction,
    )


def suture_holding_capacity_n(
    profile: dict[str, Any],
    *,
    bite_margin_m: float,
    engaged_thickness_m: float,
    fiber_alignment: float = 0.0,
    local_damage: float = 0.0,
    episode: TissueEpisodeParameters | None = None,
) -> float:
    holding = profile["suture_holding"]
    reference_force = (
        episode.reference_pullout_force_n
        if episode
        else float(holding["reference_pullout_force_n_seed"])
    )
    margin_ratio = max(0.0, float(bite_margin_m)) / float(
        holding["reference_bite_margin_m"]
    )
    thickness_ratio = max(0.0, float(engaged_thickness_m)) / float(
        holding["reference_thickness_m"]
    )
    anisotropy = 1.0 + (float(holding["anisotropy_ratio_seed"]) - 1.0) * max(
        -1.0, min(1.0, float(fiber_alignment))
    )
    damage_retention = max(0.0, 1.0 - float(local_damage))
    return (
        reference_force * margin_ratio * thickness_ratio * anisotropy * damage_retention
    )


def cyclic_tear_damage_increment(
    profile: dict[str, Any],
    *,
    tension_n: float,
    holding_capacity_n: float,
    duration_s: float,
) -> float:
    if holding_capacity_n <= 0.0:
        return 1.0
    onset = float(profile["suture_holding"]["cyclic_damage_onset_fraction"])
    utilization = max(0.0, float(tension_n)) / holding_capacity_n
    if utilization <= onset:
        return 0.0
    return (
        (utilization - onset) / max(1.0 - onset, 1.0e-9) * max(0.0, float(duration_s))
    )


def wound_gap_under_tension_m(
    profile: dict[str, Any],
    *,
    total_stitch_tension_n: float,
) -> float:
    closure = profile["wound_closure"]
    rest_gap = float(profile["geometry"]["rest_wound_gap_m"])
    force_scale = float(closure["closure_force_scale_n_seed"])
    minimum_gap = float(closure["minimum_residual_gap_m"])
    closed = rest_gap / (1.0 + max(0.0, float(total_stitch_tension_n)) / force_scale)
    return max(minimum_gap, closed)
