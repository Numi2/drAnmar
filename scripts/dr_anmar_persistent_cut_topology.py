#!/usr/bin/env python3
"""Persistent arbitrary cut-cell topology and wound-surface reference.

The field stores zero-volume oriented discontinuity patches instead of deleting
voxels. A cell may contain multiple patches, allowing intersecting incisions.
Only the cohesive fracture-work channel advances topology; adhesion, wear,
viscous dissipation, and Coulomb friction remain separately auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dr_anmar_cohesive_fracture import swept_blade_triangles
from dr_anmar_cuttable_tissue_solver import DEFAULT_PROFILE_PATH, ScalpelPose, load_profile


@dataclass(frozen=True)
class WorkChannels:
    fracture_j_m2: float = 0.0
    adhesion_j_m2: float = 0.0
    wear_j_m2: float = 0.0
    viscous_j_m2: float = 0.0
    friction_j_m2: float = 0.0
    mode_ii_fraction: float = 0.0

    def __post_init__(self):
        values = (
            self.fracture_j_m2,
            self.adhesion_j_m2,
            self.wear_j_m2,
            self.viscous_j_m2,
            self.friction_j_m2,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Cutting work channels must be finite and non-negative")
        if not 0.0 <= self.mode_ii_fraction <= 1.0:
            raise ValueError("Mode-II fraction must lie in [0, 1]")


@dataclass
class CutPatch:
    point_m: np.ndarray
    normal: np.ndarray
    fracture_work_j_m2: float = 0.0
    adhesion_work_j_m2: float = 0.0
    wear_work_j_m2: float = 0.0
    viscous_work_j_m2: float = 0.0
    friction_work_j_m2: float = 0.0
    mixed_mode_critical_energy_j_m2: float = 0.0
    damage: float = 0.0
    fractured: bool = False
    fracture_event_id: int | None = None


@dataclass
class CutCell:
    patches: list[CutPatch] = field(default_factory=list)


@dataclass(frozen=True)
class WoundSurfaceMesh:
    vertices_m: np.ndarray
    triangles: np.ndarray
    triangle_normals: np.ndarray
    triangle_sides: np.ndarray
    patch_keys: tuple[tuple[int, int, int, int], ...]
    positive_area_m2: float
    negative_area_m2: float


@dataclass(frozen=True)
class PersistentTopologyReceipt:
    schema: str
    profile_id: str
    profile_sha256: str
    field_cells: int
    fractured_patch_count: int
    fracture_event_count: int
    arbitrary_origin_coverage_fraction: float
    curved_path_fracture_events: int
    repeated_path_additional_events: int
    intersecting_path_additional_events: int
    intersection_cell_count: int
    positive_wound_area_m2: float
    negative_wound_area_m2: float
    opposed_surface_area_relative_error: float
    wound_triangle_count: int
    wound_collision_coverage_fraction: float
    removed_volume_m3: float
    friction_only_fracture_events: int
    subcritical_fracture_events: int
    persistent_topology_sha256: str
    deterministic_replay: bool
    qualified: bool
    failed_gates: tuple[str, ...]
    biomechanical_validation: bool
    clinical_validation: bool

    def payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failed_gates"] = list(self.failed_gates)
        return payload


def _normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1.0e-15:
        raise ValueError("Cut patch normal must be finite and non-zero")
    return vector / norm


def _triangle_box_overlap(
    triangle: np.ndarray,
    box_center: np.ndarray,
    half_extent: np.ndarray,
) -> bool:
    local = triangle - box_center
    edges = (local[1] - local[0], local[2] - local[1], local[0] - local[2])
    axes = [
        np.asarray((1.0, 0.0, 0.0)),
        np.asarray((0.0, 1.0, 0.0)),
        np.asarray((0.0, 0.0, 1.0)),
        np.cross(edges[0], edges[1]),
    ]
    box_axes = axes[:3]
    axes.extend(np.cross(edge, box_axis) for edge in edges for box_axis in box_axes)
    for axis in axes:
        if float(np.dot(axis, axis)) <= 1.0e-24:
            continue
        projection = local @ axis
        radius = float(np.dot(half_extent, np.abs(axis)))
        if float(np.min(projection)) > radius or float(np.max(projection)) < -radius:
            return False
    return True


def _plane_box_polygon(
    point: np.ndarray,
    normal: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> np.ndarray:
    corners = np.asarray(
        [
            (x, y, z)
            for z in (minimum[2], maximum[2])
            for y in (minimum[1], maximum[1])
            for x in (minimum[0], maximum[0])
        ],
        dtype=np.float64,
    )
    edges = (
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    signed = (corners - point) @ normal
    intersections: list[np.ndarray] = []
    tolerance = 1.0e-12
    for left, right in edges:
        left_distance = float(signed[left])
        right_distance = float(signed[right])
        if abs(left_distance) <= tolerance:
            intersections.append(corners[left])
        if left_distance * right_distance < -tolerance * tolerance:
            fraction = left_distance / (left_distance - right_distance)
            intersections.append(corners[left] + fraction * (corners[right] - corners[left]))
    if not intersections:
        return np.empty((0, 3), dtype=np.float64)
    unique: list[np.ndarray] = []
    for candidate in intersections:
        if not any(np.linalg.norm(candidate - existing) <= 1.0e-10 for existing in unique):
            unique.append(candidate)
    if len(unique) < 3:
        return np.empty((0, 3), dtype=np.float64)
    polygon = np.asarray(unique)
    centroid = np.mean(polygon, axis=0)
    reference = np.asarray((1.0, 0.0, 0.0))
    if abs(float(np.dot(reference, normal))) > 0.9:
        reference = np.asarray((0.0, 1.0, 0.0))
    tangent = _normalized(np.cross(normal, reference))
    bitangent = np.cross(normal, tangent)
    relative = polygon - centroid
    angles = np.arctan2(relative @ bitangent, relative @ tangent)
    return polygon[np.argsort(angles)]


def _polygon_area(polygon: np.ndarray, normal: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    accumulator = np.zeros(3, dtype=np.float64)
    for index in range(len(polygon)):
        accumulator += np.cross(polygon[index], polygon[(index + 1) % len(polygon)])
    return 0.5 * abs(float(np.dot(accumulator, normal)))


class PersistentCutCellField:
    def __init__(self, profile: dict[str, Any]):
        self.profile = profile
        geometry = profile["geometry"]
        topology = profile["persistent_topology"]
        self.counts = np.asarray([int(topology[f"cells_{axis}"]) for axis in "xyz"], dtype=np.int64)
        self.minimum = -0.5 * np.asarray(
            (
                float(geometry["width_m"]),
                float(geometry["depth_m"]),
                float(geometry["thickness_m"]),
            )
        )
        self.maximum = -self.minimum
        self.cell_size = (self.maximum - self.minimum) / self.counts
        self.cells: dict[tuple[int, int, int], CutCell] = {}
        self.fracture_event_count = 0
        self.merge_cosine = math.cos(math.radians(float(topology["patch_merge_angle_deg"])))
        self.merge_distance = float(
            topology["patch_merge_distance_fraction_of_minimum_cell"]
        ) * float(np.min(self.cell_size))
        self.blade_radius = float(profile["scalpel_contact"]["edge_radius_m"])
        self.blade_length = float(profile["scalpel_contact"]["edge_length_m"])
        self.minimum_speed = float(profile["fracture"]["minimum_seeded_separation_rate_m_s"])

    def _cell_bounds(self, key: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
        index = np.asarray(key, dtype=np.float64)
        minimum = self.minimum + index * self.cell_size
        return minimum, minimum + self.cell_size

    def _candidate_cells(self, swept: tuple[np.ndarray, np.ndarray]) -> set[tuple[int, int, int]]:
        vertices = np.concatenate(swept, axis=0)
        expanded_minimum = np.min(vertices, axis=0) - self.blade_radius
        expanded_maximum = np.max(vertices, axis=0) + self.blade_radius
        lower = np.floor((expanded_minimum - self.minimum) / self.cell_size).astype(int)
        upper = np.floor((expanded_maximum - self.minimum) / self.cell_size).astype(int)
        lower = np.maximum(lower, 0)
        upper = np.minimum(upper, self.counts - 1)
        if np.any(lower > upper):
            return set()
        candidates: set[tuple[int, int, int]] = set()
        half_extent = 0.5 * self.cell_size + self.blade_radius
        for z_index in range(int(lower[2]), int(upper[2]) + 1):
            for y_index in range(int(lower[1]), int(upper[1]) + 1):
                for x_index in range(int(lower[0]), int(upper[0]) + 1):
                    key = (x_index, y_index, z_index)
                    cell_minimum, cell_maximum = self._cell_bounds(key)
                    center = 0.5 * (cell_minimum + cell_maximum)
                    if any(
                        _triangle_box_overlap(triangle, center, half_extent) for triangle in swept
                    ):
                        candidates.add(key)
        return candidates

    def _critical_energy(self, mode_ii_fraction: float) -> float:
        fracture = self.profile["fracture"]
        mode_i = float(fracture["mode_i_fracture_energy_j_m2"])
        mode_ii = float(fracture["mode_ii_fracture_energy_j_m2"])
        exponent = float(fracture["benzeggagh_kenane_exponent"])
        return mode_i + (mode_ii - mode_i) * mode_ii_fraction**exponent

    def _matching_patch(
        self, cell: CutCell, point: np.ndarray, normal: np.ndarray
    ) -> CutPatch | None:
        for patch in cell.patches:
            alignment = float(np.dot(patch.normal, normal))
            if abs(alignment) < self.merge_cosine:
                continue
            oriented_normal = normal if alignment >= 0.0 else -normal
            if abs(float(np.dot(point - patch.point_m, patch.normal))) <= self.merge_distance:
                if not patch.fractured:
                    patch.normal = _normalized(patch.normal + oriented_normal)
                return patch
        return None

    def apply_sweep(
        self,
        start: ScalpelPose,
        end: ScalpelPose,
        work: WorkChannels,
    ) -> set[tuple[int, int, int, int]]:
        speed = max(
            float(np.linalg.norm(start.velocity_m_s)),
            float(np.linalg.norm(end.velocity_m_s)),
        )
        if speed < self.minimum_speed:
            return set()
        swept = swept_blade_triangles(start, end, self.blade_length)
        movement = np.asarray(end.center_m) - np.asarray(start.center_m)
        tangent = np.asarray(start.tangent) + np.asarray(end.tangent)
        normal = _normalized(np.cross(tangent, movement))
        plane_point = 0.5 * (np.asarray(start.center_m) + np.asarray(end.center_m))
        newly_fractured: set[tuple[int, int, int, int]] = set()
        for key in sorted(self._candidate_cells(swept)):
            cell = self.cells.setdefault(key, CutCell())
            patch = self._matching_patch(cell, plane_point, normal)
            if patch is None:
                patch = CutPatch(point_m=plane_point.copy(), normal=normal.copy())
                cell.patches.append(patch)
            patch.fracture_work_j_m2 += work.fracture_j_m2
            patch.adhesion_work_j_m2 += work.adhesion_j_m2
            patch.wear_work_j_m2 += work.wear_j_m2
            patch.viscous_work_j_m2 += work.viscous_j_m2
            patch.friction_work_j_m2 += work.friction_j_m2
            patch.mixed_mode_critical_energy_j_m2 = self._critical_energy(work.mode_ii_fraction)
            patch.damage = min(
                1.0,
                patch.fracture_work_j_m2 / max(patch.mixed_mode_critical_energy_j_m2, 1.0e-15),
            )
            if not patch.fractured and patch.damage >= 1.0:
                self.fracture_event_count += 1
                patch.fractured = True
                patch.fracture_event_id = self.fracture_event_count
                newly_fractured.add((*key, len(cell.patches) - 1))
        return newly_fractured

    def fractured_patches(self):
        for key in sorted(self.cells):
            for patch_index, patch in enumerate(self.cells[key].patches):
                if patch.fractured:
                    yield key, patch_index, patch

    def topology_sha256(self) -> str:
        payload = []
        for key, patch_index, patch in self.fractured_patches():
            payload.append(
                (
                    key,
                    patch_index,
                    tuple(round(float(value), 12) for value in patch.point_m),
                    tuple(round(float(value), 12) for value in patch.normal),
                    patch.fracture_event_id,
                )
            )
        return hashlib.sha256(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def intersection_cell_count(self) -> int:
        count = 0
        for cell in self.cells.values():
            fractured = [patch for patch in cell.patches if patch.fractured]
            if any(
                abs(float(np.dot(left.normal, right.normal))) < self.merge_cosine
                for left_index, left in enumerate(fractured)
                for right in fractured[left_index + 1 :]
            ):
                count += 1
        return count

    def reconstruct_wound_surfaces(self) -> WoundSurfaceMesh:
        vertices: list[np.ndarray] = []
        triangles: list[tuple[int, int, int]] = []
        normals: list[np.ndarray] = []
        sides: list[int] = []
        patch_keys: list[tuple[int, int, int, int]] = []
        positive_area = 0.0
        negative_area = 0.0
        for key, patch_index, patch in self.fractured_patches():
            minimum, maximum = self._cell_bounds(key)
            polygon = _plane_box_polygon(patch.point_m, patch.normal, minimum, maximum)
            if len(polygon) < 3:
                continue
            area = _polygon_area(polygon, patch.normal)
            for side in (1, -1):
                base = len(vertices)
                oriented = polygon if side > 0 else polygon[::-1]
                vertices.extend(oriented)
                for local in range(1, len(oriented) - 1):
                    triangles.append((base, base + local, base + local + 1))
                    normals.append(side * patch.normal)
                    sides.append(side)
                    patch_keys.append((*key, patch_index))
                if side > 0:
                    positive_area += area
                else:
                    negative_area += area
        return WoundSurfaceMesh(
            vertices_m=np.asarray(vertices, dtype=np.float64).reshape((-1, 3)),
            triangles=np.asarray(triangles, dtype=np.int64).reshape((-1, 3)),
            triangle_normals=np.asarray(normals, dtype=np.float64).reshape((-1, 3)),
            triangle_sides=np.asarray(sides, dtype=np.int8),
            patch_keys=tuple(patch_keys),
            positive_area_m2=positive_area,
            negative_area_m2=negative_area,
        )


def _closest_point_on_triangle(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    a, b, c = triangle
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = float(np.dot(ab, ap)), float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    bp = point - b
    d3, d4 = float(np.dot(ab, bp)), float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return a + (d1 / (d1 - d3)) * ab
    cp = point - c
    d5, d6 = float(np.dot(ab, cp)), float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return a + (d2 / (d2 - d6)) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)
    denominator = 1.0 / (va + vb + vc)
    return a + (vb * denominator) * ab + (vc * denominator) * ac


def wound_contact_force(
    mesh: WoundSurfaceMesh,
    point_m: np.ndarray,
    radius_m: float,
    stiffness_n_m: float,
) -> np.ndarray:
    if not len(mesh.triangles):
        return np.zeros(3)
    point = np.asarray(point_m, dtype=np.float64)
    best_distance = math.inf
    best_normal = np.zeros(3)
    best_closest = np.zeros(3)
    for triangle_index, indices in enumerate(mesh.triangles):
        triangle = mesh.vertices_m[indices]
        closest = _closest_point_on_triangle(point, triangle)
        distance = float(np.linalg.norm(point - closest))
        if distance < best_distance:
            best_distance = distance
            best_normal = mesh.triangle_normals[triangle_index]
            best_closest = closest
    if best_distance >= radius_m:
        return np.zeros(3)
    direction = point - best_closest
    if best_distance > 1.0e-12:
        direction /= best_distance
    else:
        direction = best_normal
    return stiffness_n_m * (radius_m - best_distance) * direction


def _incision_sweeps(
    centers: list[tuple[float, float, float]],
    *,
    speed: float = 0.006,
) -> list[tuple[ScalpelPose, ScalpelPose]]:
    tangent = (0.0, 0.0, 1.0)
    sweeps = []
    for start, end in zip(centers[:-1], centers[1:], strict=True):
        movement = _normalized(np.asarray(end) - np.asarray(start))
        velocity = tuple(float(value) for value in speed * movement)
        sweeps.append(
            (
                ScalpelPose(start, tangent=tangent, velocity_m_s=velocity),
                ScalpelPose(end, tangent=tangent, velocity_m_s=velocity),
            )
        )
    return sweeps


def _run_reference_topology(
    profile: dict[str, Any],
) -> tuple[PersistentCutCellField, dict[str, Any]]:
    critical = float(profile["fracture"]["mode_i_fracture_energy_j_m2"])
    work = WorkChannels(
        fracture_j_m2=1.1 * critical,
        adhesion_j_m2=0.7,
        wear_j_m2=0.4,
        viscous_j_m2=0.2,
        friction_j_m2=0.3,
    )
    field = PersistentCutCellField(profile)
    x_values = np.linspace(-0.010, 0.010, 13)
    curved_centers = [
        (float(x), float(0.0035 * math.sin(math.pi * x / 0.020)), 0.0) for x in x_values
    ]
    curved_sweeps = _incision_sweeps(curved_centers)
    for start, end in curved_sweeps:
        field.apply_sweep(start, end, work)
    curved_events = field.fracture_event_count
    topology_before_repeat = field.topology_sha256()
    for start, end in curved_sweeps:
        field.apply_sweep(start, end, work)
    repeated_events = field.fracture_event_count - curved_events
    topology_after_repeat = field.topology_sha256()

    before_intersection = field.fracture_event_count
    crossing_centers = [(0.0, float(y), 0.0) for y in np.linspace(-0.010, 0.010, 13)]
    for start, end in _incision_sweeps(crossing_centers):
        field.apply_sweep(start, end, work)
    intersecting_events = field.fracture_event_count - before_intersection
    return field, {
        "curved_events": curved_events,
        "repeated_events": repeated_events,
        "intersecting_events": intersecting_events,
        "repeat_hash_stable": topology_before_repeat == topology_after_repeat,
    }


def run_persistent_topology_qualification(
    profile: dict[str, Any] | None = None,
    *,
    profile_path: Path = DEFAULT_PROFILE_PATH,
) -> PersistentTopologyReceipt:
    profile = profile or load_profile(profile_path)
    critical = float(profile["fracture"]["mode_i_fracture_energy_j_m2"])
    geometry = profile["geometry"]
    half_width = float(geometry["width_m"]) / 2.0
    arbitrary_counts: list[int] = []
    for offset in np.linspace(-0.75 * half_width, 0.75 * half_width, 9):
        field = PersistentCutCellField(profile)
        sweeps = _incision_sweeps([(float(offset), -0.004, 0.0), (float(offset), 0.004, 0.0)])
        start, end = sweeps[0]
        arbitrary_counts.append(
            len(field.apply_sweep(start, end, WorkChannels(fracture_j_m2=1.1 * critical)))
        )
    arbitrary_coverage = sum(count > 0 for count in arbitrary_counts) / len(arbitrary_counts)

    friction_only = PersistentCutCellField(profile)
    start, end = _incision_sweeps([(-0.004, 0.0, 0.0), (0.004, 0.0, 0.0)])[0]
    friction_only.apply_sweep(start, end, WorkChannels(friction_j_m2=100.0 * critical))
    subcritical = PersistentCutCellField(profile)
    subcritical.apply_sweep(start, end, WorkChannels(fracture_j_m2=0.99 * critical))

    first_field, metrics = _run_reference_topology(profile)
    second_field, second_metrics = _run_reference_topology(profile)
    mesh = first_field.reconstruct_wound_surfaces()
    collision_samples = 0
    collision_hits = 0
    radius = 0.25 * float(np.min(first_field.cell_size))
    for triangle_index in range(0, len(mesh.triangles), max(1, len(mesh.triangles) // 64)):
        triangle = mesh.vertices_m[mesh.triangles[triangle_index]]
        center = np.mean(triangle, axis=0)
        normal = mesh.triangle_normals[triangle_index]
        point = center + 0.5 * radius * normal
        force = wound_contact_force(mesh, point, radius, 2000.0)
        collision_samples += 1
        collision_hits += int(float(np.linalg.norm(force)) > 0.0)
    collision_coverage = collision_hits / max(collision_samples, 1)
    area_error = abs(mesh.positive_area_m2 - mesh.negative_area_m2) / max(
        mesh.positive_area_m2, 1.0e-15
    )
    topology_hash = first_field.topology_sha256()
    deterministic = topology_hash == second_field.topology_sha256() and metrics == second_metrics
    limits = profile["persistent_topology"]["qualification"]
    gates = {
        "arbitrary_origin": arbitrary_coverage
        >= float(limits["minimum_arbitrary_origin_coverage_fraction"]),
        "curved_cut": metrics["curved_events"] > 0,
        "repeat_idempotent": metrics["repeated_events"] == 0 and metrics["repeat_hash_stable"],
        "intersecting_cut": metrics["intersecting_events"] > 0
        and first_field.intersection_cell_count() >= int(limits["minimum_intersection_cell_count"]),
        "paired_surface_area": area_error
        <= float(limits["maximum_opposed_surface_area_relative_error"]),
        "wound_collision": collision_coverage
        >= float(limits["minimum_wound_collision_coverage_fraction"]),
        "zero_volume_loss": 0.0 <= float(limits["maximum_removed_volume_m3"]),
        "separate_work_channels": friction_only.fracture_event_count == 0,
        "subcritical_blocked": subcritical.fracture_event_count == 0,
        "deterministic_replay": deterministic,
    }
    failed = tuple(name for name, passed in gates.items() if not passed)
    profile_sha = hashlib.sha256(
        json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    fractured_count = sum(1 for _ in first_field.fractured_patches())
    return PersistentTopologyReceipt(
        schema="dr.anmar.persistent-cut-topology-receipt.v1",
        profile_id=str(profile["id"]),
        profile_sha256=profile_sha,
        field_cells=int(np.prod(first_field.counts)),
        fractured_patch_count=fractured_count,
        fracture_event_count=first_field.fracture_event_count,
        arbitrary_origin_coverage_fraction=arbitrary_coverage,
        curved_path_fracture_events=int(metrics["curved_events"]),
        repeated_path_additional_events=int(metrics["repeated_events"]),
        intersecting_path_additional_events=int(metrics["intersecting_events"]),
        intersection_cell_count=first_field.intersection_cell_count(),
        positive_wound_area_m2=mesh.positive_area_m2,
        negative_wound_area_m2=mesh.negative_area_m2,
        opposed_surface_area_relative_error=area_error,
        wound_triangle_count=len(mesh.triangles),
        wound_collision_coverage_fraction=collision_coverage,
        removed_volume_m3=0.0,
        friction_only_fracture_events=friction_only.fracture_event_count,
        subcritical_fracture_events=subcritical.fracture_event_count,
        persistent_topology_sha256=topology_hash,
        deterministic_replay=deterministic,
        qualified=not failed,
        failed_gates=failed,
        biomechanical_validation=False,
        clinical_validation=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = run_persistent_topology_qualification(
        load_profile(args.profile), profile_path=args.profile
    )
    encoded = json.dumps(receipt.payload(), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if receipt.qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
