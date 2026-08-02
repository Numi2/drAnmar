#!/usr/bin/env python3
"""Curved implicit-surface cut-cell FEM reference.

Intersected tetrahedra are clipped, not deleted and not snapped to existing
faces.  Intersection nodes are duplicated across the strong discontinuity,
while clipped original faces share a deterministic triangulation.  A settled
connected state is interpolated into the remesh before the wound is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dr_anmar_cuttable_tissue_solver import (
    CuttableTissueReferenceSolver,
    build_regular_tetrahedral_coupon,
    load_profile,
)
from dr_anmar_dynamic_discontinuous_fem import _probe_collision
from dr_anmar_persistent_cut_topology import WoundSurfaceMesh


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CURVED_PROFILE_PATH = (
    REPOSITORY_ROOT / "physics_next/tissues/dr-anmar-dynamic-curved-cut-v1.json"
)
TET_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


@dataclass(frozen=True)
class DynamicCurvedCutReceipt:
    schema: str
    profile_id: str
    profile_sha256: str
    base_profile_sha256: str
    original_node_count: int
    original_tetrahedron_count: int
    cut_original_tetrahedron_count: int
    remeshed_node_count: int
    remeshed_tetrahedron_count: int
    wound_triangle_count: int
    fracture_energy_gate_passed: bool
    zero_removed_volume: bool
    volume_relative_error: float
    mass_relative_error: float
    minimum_subcell_volume_fraction: float
    maximum_level_set_residual_m: float
    positive_wound_area_m2: float
    negative_wound_area_m2: float
    opposed_area_relative_error: float
    unexpected_boundary_face_count: int
    nonmanifold_face_count: int
    curve_midpoint_deviation_from_chord_m: float
    steps: int
    finite: bool
    inverted_tetrahedra_peak: int
    minimum_jacobian: float
    center_of_mass_drift_m: float
    net_momentum_kg_m_s: float
    mean_wound_gap_m: float
    maximum_wound_gap_m: float
    two_sided_collision_coverage_fraction: float
    maximum_probe_surface_crossing_m: float
    topology_sha256: str
    deterministic_trace_sha256: str
    qualified: bool
    failed_gates: tuple[str, ...]
    biomechanical_validation: bool
    clinical_validation: bool

    def payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failed_gates"] = list(self.failed_gates)
        return payload


def load_curved_profile(path: Path = DEFAULT_CURVED_PROFILE_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _level_set(points: np.ndarray, cut: dict[str, Any]) -> np.ndarray:
    x = points[..., 0]
    curve_y = float(cut["center_y_m"]) + float(cut["amplitude_y_m"]) * np.sin(
        math.pi * x / float(cut["wavelength_x_m"])
    )
    return points[..., 1] - curve_y


def _level_gradient(points: np.ndarray, cut: dict[str, Any]) -> np.ndarray:
    x = points[..., 0]
    slope = (
        float(cut["amplitude_y_m"])
        * math.pi
        / float(cut["wavelength_x_m"])
        * np.cos(math.pi * x / float(cut["wavelength_x_m"]))
    )
    gradient = np.stack((-slope, np.ones_like(x), np.zeros_like(x)), axis=-1)
    return gradient / np.linalg.norm(gradient, axis=-1, keepdims=True)


def _triangulate_polygon(descriptors: list[tuple[Any, ...]]) -> list[tuple[int, int, int]]:
    if len(descriptors) < 3:
        return []
    start = min(range(len(descriptors)), key=lambda index: descriptors[index])
    order = list(range(start, len(descriptors))) + list(range(0, start))
    return [(order[0], order[index], order[index + 1]) for index in range(1, len(order) - 1)]


class EmbeddedCurvedCutMesh:
    """Conforming sub-tetrahedralization of one implicit strong discontinuity."""

    def __init__(
        self,
        base_profile: dict[str, Any],
        curved_profile: dict[str, Any],
        settled_position: np.ndarray,
        settled_velocity: np.ndarray,
        settled_prony: np.ndarray,
        settled_elastic: np.ndarray,
    ):
        original_points, original_tets = build_regular_tetrahedral_coupon(base_profile)
        self.original_points = original_points
        self.original_tets = original_tets
        cut = curved_profile["implicit_cut"]
        tolerance = float(cut["level_set_tolerance_m"])
        signed = _level_set(original_points, cut)
        if np.any(np.abs(signed) <= tolerance):
            raise ValueError("Implicit cut must not pass through an original FEM node")
        energy_gate = float(cut["applied_fracture_work_ratio"]) >= float(
            cut["minimum_fracture_work_ratio"]
        )
        if not energy_gate:
            raise ValueError("Curved discontinuity cannot form below the cohesive-energy gate")
        self.fracture_energy_gate_passed = energy_gate

        rest: list[np.ndarray] = [point.copy() for point in original_points]
        current: list[np.ndarray] = [point.copy() for point in settled_position]
        velocity: list[np.ndarray] = [value.copy() for value in settled_velocity]
        edge_nodes: dict[tuple[int, int, int], int] = {}
        node_descriptor: dict[int, tuple[Any, ...]] = {
            index: ("v", index) for index in range(len(original_points))
        }

        def edge_node(a: int, b: int, side: int) -> int:
            lo, hi = sorted((int(a), int(b)))
            key = (lo, hi, side)
            if key in edge_nodes:
                return edge_nodes[key]
            # The sinusoidal level set is nonlinear along x-directed edges.
            # Solve the intersection rather than linearly interpolating phi;
            # the material/state interpolation remains affine on the parent.
            lower_fraction = 0.0
            upper_fraction = 1.0
            lower_sign = float(signed[lo])
            for _ in range(60):
                middle = 0.5 * (lower_fraction + upper_fraction)
                candidate = original_points[lo] + middle * (
                    original_points[hi] - original_points[lo]
                )
                middle_sign = float(_level_set(candidate[None, :], cut)[0])
                if lower_sign * middle_sign <= 0.0:
                    upper_fraction = middle
                else:
                    lower_fraction = middle
                    lower_sign = middle_sign
            fraction = 0.5 * (lower_fraction + upper_fraction)
            point = original_points[lo] + fraction * (original_points[hi] - original_points[lo])
            state = settled_position[lo] + fraction * (settled_position[hi] - settled_position[lo])
            rate = settled_velocity[lo] + fraction * (settled_velocity[hi] - settled_velocity[lo])
            index = len(rest)
            rest.append(point)
            current.append(state)
            velocity.append(rate)
            edge_nodes[key] = index
            node_descriptor[index] = ("e", lo, hi)
            return index

        def clipped_face(face: tuple[int, int, int], side: int) -> list[int]:
            output: list[int] = []
            for slot, local_a in enumerate(face):
                local_b = face[(slot + 1) % len(face)]
                a = int(tet[local_a])
                b = int(tet[local_b])
                inside_a = side * signed[a] > tolerance
                inside_b = side * signed[b] > tolerance
                if inside_a:
                    output.append(a)
                if inside_a != inside_b:
                    output.append(edge_node(a, b, side))
            return output

        new_tets: list[tuple[int, int, int, int]] = []
        parent_tets: list[int] = []
        wound_triangles_by_side: dict[int, list[tuple[int, int, int]]] = {-1: [], 1: []}
        gap_pairs: dict[tuple[int, int], tuple[int, int]] = {}
        cut_original_count = 0
        for parent, tet in enumerate(original_tets):
            local_signed = signed[tet]
            if np.all(local_signed > tolerance) or np.all(local_signed < -tolerance):
                new_tets.append(tuple(int(value) for value in tet))
                parent_tets.append(parent)
                continue
            cut_original_count += 1
            crossing_edges = []
            for local_a, local_b in TET_EDGES:
                a = int(tet[local_a])
                b = int(tet[local_b])
                if signed[a] * signed[b] < 0.0:
                    crossing_edges.append(tuple(sorted((a, b))))
            intersection_points = np.asarray(
                [rest[edge_node(a, b, 1)] for a, b in crossing_edges], dtype=np.float64
            )
            centroid = np.mean(intersection_points, axis=0)
            normal = _level_gradient(centroid[None, :], cut)[0]
            tangent = np.asarray((normal[1], -normal[0], 0.0))
            bitangent = np.asarray((0.0, 0.0, 1.0))
            relative = intersection_points - centroid
            angles = np.arctan2(relative @ bitangent, relative @ tangent)
            ordered_edges = [crossing_edges[index] for index in np.argsort(angles)]
            cut_descriptors = [("e", edge[0], edge[1]) for edge in ordered_edges]
            cut_tris = _triangulate_polygon(cut_descriptors)
            for side in (-1, 1):
                boundary_polygons: list[list[int]] = []
                for local_face in TET_FACES:
                    polygon = clipped_face(local_face, side)
                    if len(polygon) >= 3:
                        boundary_polygons.append(polygon)
                cut_polygon = [edge_node(a, b, side) for a, b in ordered_edges]
                boundary_polygons.append(cut_polygon)
                for edge in ordered_edges:
                    pair = gap_pairs.get(edge, (-1, -1))
                    value = edge_node(edge[0], edge[1], side)
                    gap_pairs[edge] = (value, pair[1]) if side > 0 else (pair[0], value)

                poly_nodes = sorted({node for polygon in boundary_polygons for node in polygon})
                center_index = len(rest)
                rest.append(np.mean(np.asarray([rest[node] for node in poly_nodes]), axis=0))
                current.append(np.mean(np.asarray([current[node] for node in poly_nodes]), axis=0))
                velocity.append(np.mean(np.asarray([velocity[node] for node in poly_nodes]), axis=0))
                node_descriptor[center_index] = ("c", parent, side)
                for polygon_index, polygon in enumerate(boundary_polygons):
                    descriptors = [node_descriptor[node] for node in polygon]
                    local_triangles = (
                        cut_tris if polygon_index == len(boundary_polygons) - 1
                        else _triangulate_polygon(descriptors)
                    )
                    for ia, ib, ic in local_triangles:
                        triangle = (polygon[ia], polygon[ib], polygon[ic])
                        candidate = [center_index, *triangle]
                        matrix = np.stack(
                            (
                                rest[candidate[1]] - rest[candidate[0]],
                                rest[candidate[2]] - rest[candidate[0]],
                                rest[candidate[3]] - rest[candidate[0]],
                            ),
                            axis=1,
                        )
                        if np.linalg.det(matrix) < 0.0:
                            candidate[2], candidate[3] = candidate[3], candidate[2]
                        new_tets.append(tuple(candidate))
                        parent_tets.append(parent)
                        if polygon_index == len(boundary_polygons) - 1:
                            wound_triangles_by_side[side].append(triangle)

        self.rest = np.asarray(rest, dtype=np.float64)
        self.position = np.asarray(current, dtype=np.float64)
        self.velocity = np.asarray(velocity, dtype=np.float64)
        self.tetrahedra = np.asarray(new_tets, dtype=np.int64)
        self.parent_tetrahedra = np.asarray(parent_tets, dtype=np.int64)
        self.cut_original_tetrahedron_count = cut_original_count
        self.wound_triangles_by_side = {
            side: np.asarray(triangles, dtype=np.int64)
            for side, triangles in wound_triangles_by_side.items()
        }
        ordered_pairs = sorted(gap_pairs.items())
        self.gap_plus_nodes = np.asarray([pair[1][0] for pair in ordered_pairs], dtype=np.int64)
        self.gap_minus_nodes = np.asarray([pair[1][1] for pair in ordered_pairs], dtype=np.int64)
        self.gap_rest_points = np.asarray(
            [0.5 * (self.rest[plus] + self.rest[minus]) for plus, minus in zip(
                self.gap_plus_nodes, self.gap_minus_nodes, strict=True
            )]
        )
        self.gap_normals = _level_gradient(self.gap_rest_points, cut)
        self.gap_area = np.zeros(len(self.gap_plus_nodes), dtype=np.float64)
        plus_slot = {int(node): index for index, node in enumerate(self.gap_plus_nodes)}
        for triangle in self.wound_triangles_by_side[1]:
            points = self.rest[triangle]
            area = 0.5 * float(
                np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0]))
            )
            for node in triangle:
                self.gap_area[plus_slot[int(node)]] += area / 3.0
        self.prony_history = settled_prony[self.parent_tetrahedra].copy()
        self.previous_elastic_stress = settled_elastic[self.parent_tetrahedra].copy()
        self._initialize_mechanics(base_profile, curved_profile)

    def _initialize_mechanics(self, base_profile: dict[str, Any], curved_profile: dict[str, Any]):
        self.base_profile = base_profile
        self.curved_profile = curved_profile
        local = self.rest[self.tetrahedra]
        dm = np.stack((local[:, 1] - local[:, 0], local[:, 2] - local[:, 0], local[:, 3] - local[:, 0]), axis=2)
        self.dm_inverse = np.linalg.inv(dm)
        self.rest_volume = np.linalg.det(dm) / 6.0
        if np.any(self.rest_volume <= 0.0):
            raise ValueError("Cut-cell remesh contains a non-positive tetrahedron")
        self.shape_gradients = np.empty((len(self.tetrahedra), 4, 3), dtype=np.float64)
        self.shape_gradients[:, 1:, :] = self.dm_inverse
        self.shape_gradients[:, 0, :] = -np.sum(self.dm_inverse, axis=1)
        density = float(base_profile["material"]["density_kg_m3"])
        self.mass = np.zeros(len(self.rest), dtype=np.float64)
        for local_index in range(4):
            np.add.at(self.mass, self.tetrahedra[:, local_index], density * self.rest_volume / 4.0)
        geometry = base_profile["geometry"]
        half_width = float(geometry["width_m"]) / 2.0
        band = float(geometry["attachment_band_m"])
        self.fixed = np.abs(self.rest[:, 0]) >= half_width - band
        self.fixed_position = self.position[self.fixed].copy()
        self.steps = 0

    def _deformation(self) -> tuple[np.ndarray, np.ndarray]:
        local = self.position[self.tetrahedra]
        ds = np.stack((local[:, 1] - local[:, 0], local[:, 2] - local[:, 0], local[:, 3] - local[:, 0]), axis=2)
        deformation = np.einsum("tij,tjk->tik", ds, self.dm_inverse)
        return deformation, np.linalg.det(deformation)

    def step(self, dt: float, opening_traction_pa: float = 0.0) -> dict[str, Any]:
        material = self.base_profile["material"]
        youngs = float(material["youngs_modulus_pa"])
        poisson = float(material["poisson_ratio"])
        mu = youngs / (2.0 * (1.0 + poisson))
        lam = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        deformation, jacobian = self._deformation()
        inverse_transpose = np.swapaxes(np.linalg.inv(deformation), 1, 2)
        elastic = mu * (deformation - inverse_transpose) + lam * np.log(
            np.maximum(jacobian, 1.0e-9)
        )[:, None, None] * inverse_transpose
        fraction = float(material["prony_relaxation_fraction"])
        decay = math.exp(-dt / float(material["prony_time_constant_s"]))
        self.prony_history = decay * self.prony_history + fraction * (
            elastic - self.previous_elastic_stress
        )
        self.previous_elastic_stress = elastic
        stress = (1.0 - fraction) * elastic + self.prony_history
        local_force = -self.rest_volume[:, None, None] * np.einsum(
            "tij,tkj->tki", stress, self.shape_gradients
        )
        force = np.zeros_like(self.position)
        for local_index in range(4):
            np.add.at(force, self.tetrahedra[:, local_index], local_force[:, local_index])
        # Post-fracture contact is unilateral: separation is traction-free, but
        # the two material sides cannot pass through one another.  This uses
        # the cohesive compression and viscosity terms without healing damage.
        gap = self.gaps()
        relative_velocity = np.sum(
            (self.velocity[self.gap_plus_nodes] - self.velocity[self.gap_minus_nodes])
            * self.gap_normals,
            axis=1,
        )
        fracture = self.base_profile["fracture"]
        compression = float(fracture["compression_stiffness_pa_m"])
        viscosity = float(fracture["cohesive_viscosity_pa_s_m"])
        contact_traction = np.where(
            gap < 0.0,
            -compression * gap - viscosity * np.minimum(relative_velocity, 0.0),
            0.0,
        )
        pair_force = contact_traction[:, None] * self.gap_area[:, None] * self.gap_normals
        np.add.at(force, self.gap_plus_nodes, pair_force)
        np.add.at(force, self.gap_minus_nodes, -pair_force)
        if opening_traction_pa > 0.0:
            for side in (-1, 1):
                for triangle in self.wound_triangles_by_side[side]:
                    points = self.position[triangle]
                    area = 0.5 * float(np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])))
                    normal = _level_gradient(np.mean(self.rest[triangle], axis=0)[None, :], self.curved_profile["implicit_cut"])[0]
                    nodal = side * opening_traction_pa * area * normal / 3.0
                    np.add.at(force, triangle, nodal)
        damping = math.exp(-float(self.curved_profile["solver"]["velocity_damping_per_s"]) * dt)
        self.velocity += dt * force / self.mass[:, None]
        self.velocity *= damping
        self.position += dt * self.velocity
        self.position[self.fixed] = self.fixed_position
        self.velocity[self.fixed] = 0.0
        self.steps += 1
        return {"jacobian": jacobian, "finite": bool(np.isfinite(self.position).all() and np.isfinite(self.velocity).all() and np.isfinite(force).all())}

    def center_of_mass(self) -> np.ndarray:
        return np.sum(self.mass[:, None] * self.position, axis=0) / np.sum(self.mass)

    def momentum(self) -> np.ndarray:
        return np.sum(self.mass[:, None] * self.velocity, axis=0)

    def gaps(self) -> np.ndarray:
        return np.sum((self.position[self.gap_plus_nodes] - self.position[self.gap_minus_nodes]) * self.gap_normals, axis=1)

    def wound_surface_mesh(self) -> WoundSurfaceMesh:
        triangles: list[tuple[int, int, int]] = []
        normals: list[np.ndarray] = []
        sides: list[int] = []
        keys: list[tuple[int, int, int, int]] = []
        areas = {-1: 0.0, 1: 0.0}
        for material_side in (-1, 1):
            collision_side = -material_side
            for index, triangle_tuple in enumerate(self.wound_triangles_by_side[material_side]):
                triangle = list(int(value) for value in triangle_tuple)
                points = self.position[triangle]
                outward = -material_side * _level_gradient(
                    np.mean(self.rest[triangle], axis=0)[None, :], self.curved_profile["implicit_cut"]
                )[0]
                geometric = np.cross(points[1] - points[0], points[2] - points[0])
                if float(np.dot(geometric, outward)) < 0.0:
                    triangle[1], triangle[2] = triangle[2], triangle[1]
                    geometric = -geometric
                triangles.append(tuple(triangle))
                normals.append(geometric / max(float(np.linalg.norm(geometric)), 1.0e-15))
                sides.append(collision_side)
                keys.append((index, collision_side, 0, 0))
                areas[collision_side] += 0.5 * float(np.linalg.norm(geometric))
        return WoundSurfaceMesh(
            vertices_m=self.position.copy(),
            triangles=np.asarray(triangles, dtype=np.int64),
            triangle_normals=np.asarray(normals),
            triangle_sides=np.asarray(sides, dtype=np.int8),
            patch_keys=tuple(keys),
            positive_area_m2=areas[1],
            negative_area_m2=areas[-1],
        )

    def topology_metrics(self) -> tuple[int, int]:
        counts: dict[tuple[int, int, int], int] = {}
        for tet in self.tetrahedra:
            for face in TET_FACES:
                key = tuple(sorted(int(tet[index]) for index in face))
                counts[key] = counts.get(key, 0) + 1
        wound = {
            tuple(sorted(int(value) for value in triangle))
            for triangles in self.wound_triangles_by_side.values()
            for triangle in triangles
        }
        minimum = np.min(self.original_points, axis=0)
        maximum = np.max(self.original_points, axis=0)
        unexpected = 0
        for face, count in counts.items():
            if count != 1 or face in wound:
                continue
            points = self.rest[list(face)]
            on_outer = any(
                np.all(np.isclose(points[:, axis], bound, atol=1.0e-12))
                for axis in range(3)
                for bound in (minimum[axis], maximum[axis])
            )
            unexpected += int(not on_outer)
        return unexpected, sum(count > 2 for count in counts.values())


def _exclude_cut_corridor_from_anchors(
    solver: Any,
    curved: dict[str, Any],
    half_width_m: float | None,
) -> None:
    """Leave the cut/side-face intersection free while retaining nearby anchors."""

    if half_width_m is None:
        return
    if half_width_m <= 0.0:
        raise ValueError("Anchor-exclusion corridor half-width must be positive")
    in_corridor = np.abs(_level_set(solver.rest, curved["implicit_cut"])) <= half_width_m
    solver.fixed = np.asarray(solver.fixed & ~in_corridor, dtype=bool)
    if not np.any(solver.fixed):
        raise ValueError("Cut corridor removed every tissue anchor")
    solver.fixed_position = solver.position[solver.fixed].copy()


def _build_settled_mesh(
    base: dict[str, Any],
    curved: dict[str, Any],
    *,
    anchor_exclusion_half_width_m: float | None = None,
) -> EmbeddedCurvedCutMesh:
    connected = CuttableTissueReferenceSolver(base)
    _exclude_cut_corridor_from_anchors(
        connected, curved, anchor_exclusion_half_width_m
    )
    dt = float(curved["solver"]["connected_settle_time_step_s"])
    for _ in range(int(round(float(curved["solver"]["connected_settle_s"]) / dt))):
        connected.step(dt)
    embedded = EmbeddedCurvedCutMesh(
        base,
        curved,
        connected.position,
        connected.velocity,
        connected.prony_history,
        connected.previous_elastic_stress,
    )
    _exclude_cut_corridor_from_anchors(
        embedded, curved, anchor_exclusion_half_width_m
    )
    return embedded


def run_dynamic_curved_cut_qualification(
    curved_profile: dict[str, Any] | None = None,
    *,
    curved_profile_path: Path = DEFAULT_CURVED_PROFILE_PATH,
) -> DynamicCurvedCutReceipt:
    curved = curved_profile or load_curved_profile(curved_profile_path)
    base = load_profile(REPOSITORY_ROOT / curved["base_profile"])
    base_sha = hashlib.sha256(json.dumps(base, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if base_sha != curved["base_profile_sha256"]:
        raise ValueError("Curved profile is not bound to the current base profile")
    solver = _build_settled_mesh(base, curved)
    original_volume = float(np.sum(np.linalg.det(np.stack((
        solver.original_points[solver.original_tets[:, 1]] - solver.original_points[solver.original_tets[:, 0]],
        solver.original_points[solver.original_tets[:, 2]] - solver.original_points[solver.original_tets[:, 0]],
        solver.original_points[solver.original_tets[:, 3]] - solver.original_points[solver.original_tets[:, 0]],
    ), axis=2)) / 6.0))
    remeshed_volume = float(np.sum(solver.rest_volume))
    density = float(base["material"]["density_kg_m3"])
    expected_mass = density * original_volume
    initial_com = solver.center_of_mass().copy()
    cut = curved["implicit_cut"]
    wound_rest = np.concatenate([
        solver.rest[triangles].reshape((-1, 3)) for triangles in solver.wound_triangles_by_side.values()
    ])
    level_residual = float(np.max(np.abs(_level_set(wound_rest, cut))))
    original_tet_volume = original_volume / len(solver.original_tets)
    minimum_fraction = float(np.min(solver.rest_volume) / original_tet_volume)
    unexpected, nonmanifold = solver.topology_metrics()
    topology_payload = {
        "rest": np.round(solver.rest, 14).tolist(),
        "tetrahedra": solver.tetrahedra.tolist(),
        "wound": {str(side): triangles.tolist() for side, triangles in solver.wound_triangles_by_side.items()},
    }
    topology_sha = hashlib.sha256(json.dumps(topology_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    curve_x = np.linspace(-0.018, 0.018, 101)
    curve_y = float(cut["center_y_m"]) + float(cut["amplitude_y_m"]) * np.sin(
        math.pi * curve_x / float(cut["wavelength_x_m"])
    )
    chord_y = np.linspace(curve_y[0], curve_y[-1], len(curve_y))
    curvature_deviation = float(np.max(np.abs(curve_y - chord_y)))

    settings = curved["solver"]
    dt = float(settings["cut_time_step_s"])
    phases = ((float(settings["opening_load_s"]), float(settings["opening_traction_pa"])), (float(settings["post_load_s"]), 0.0))
    finite = True
    inverted_peak = 0
    minimum_j = math.inf
    trace: list[tuple[Any, ...]] = []
    for phase_index, (duration, traction) in enumerate(phases):
        for local_step in range(int(round(duration / dt))):
            sample = solver.step(dt, traction)
            jacobian = sample["jacobian"]
            finite = finite and bool(sample["finite"])
            inverted_peak = max(inverted_peak, int(np.count_nonzero(jacobian <= 0.0)))
            minimum_j = min(minimum_j, float(np.min(jacobian)))
            if solver.steps % 20 == 0:
                trace.append((phase_index, local_step, round(float(np.mean(solver.gaps())), 10), round(float(np.min(jacobian)), 9), round(float(np.linalg.norm(solver.momentum())), 12)))
    gaps = solver.gaps()
    wound = solver.wound_surface_mesh()
    collision_coverage, maximum_crossing = _probe_collision(wound, curved["wound_collision"])
    volume_error = abs(remeshed_volume - original_volume) / original_volume
    mass_error = abs(float(np.sum(solver.mass)) - expected_mass) / expected_mass
    area_error = abs(wound.positive_area_m2 - wound.negative_area_m2) / max(wound.positive_area_m2, 1.0e-15)
    com_drift = float(np.linalg.norm(solver.center_of_mass() - initial_com))
    momentum = float(np.linalg.norm(solver.momentum()))
    trace_sha = hashlib.sha256(json.dumps(trace, separators=(",", ":")).encode()).hexdigest()
    profile_sha = hashlib.sha256(json.dumps(curved, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    limits = curved["qualification"]
    gates = {
        "fracture_energy": solver.fracture_energy_gate_passed,
        "zero_removed_volume": volume_error <= float(limits["maximum_volume_relative_error"]),
        "mass": mass_error <= float(limits["maximum_mass_relative_error"]),
        "level_set": level_residual <= float(limits["maximum_level_set_residual_m"]),
        "opposed_area": area_error <= float(limits["maximum_opposed_area_relative_error"]),
        "conforming_faces": unexpected <= int(limits["maximum_unexpected_boundary_faces"]),
        "manifold": nonmanifold <= int(limits["maximum_nonmanifold_faces"]),
        "cut_elements": solver.cut_original_tetrahedron_count >= int(limits["minimum_cut_tetrahedra"]),
        "cell_quality": minimum_fraction >= float(limits["minimum_subcell_volume_fraction"]),
        "finite": finite,
        "no_inversion": inverted_peak == 0,
        "jacobian": minimum_j >= float(limits["minimum_jacobian"]),
        "center_of_mass": com_drift <= float(limits["maximum_center_of_mass_drift_m"]),
        "momentum": momentum <= float(limits["maximum_net_momentum_kg_m_s"]),
        "minimum_gap": float(np.mean(gaps)) >= float(limits["minimum_mean_wound_gap_m"]),
        "maximum_gap": float(np.mean(gaps)) <= float(limits["maximum_mean_wound_gap_m"]),
        "two_sided_collision": collision_coverage >= float(limits["minimum_two_sided_collision_coverage_fraction"]),
        "probe_crossing": maximum_crossing <= float(limits["maximum_probe_surface_crossing_m"]),
    }
    failed = tuple(name for name, passed in gates.items() if not passed)
    return DynamicCurvedCutReceipt(
        schema="dr.anmar.dynamic-curved-cut-receipt.v1",
        profile_id=str(curved["id"]), profile_sha256=profile_sha, base_profile_sha256=base_sha,
        original_node_count=len(solver.original_points), original_tetrahedron_count=len(solver.original_tets),
        cut_original_tetrahedron_count=solver.cut_original_tetrahedron_count,
        remeshed_node_count=len(solver.rest), remeshed_tetrahedron_count=len(solver.tetrahedra),
        wound_triangle_count=len(wound.triangles), fracture_energy_gate_passed=solver.fracture_energy_gate_passed,
        zero_removed_volume=volume_error <= float(limits["maximum_volume_relative_error"]),
        volume_relative_error=volume_error, mass_relative_error=mass_error,
        minimum_subcell_volume_fraction=minimum_fraction, maximum_level_set_residual_m=level_residual,
        positive_wound_area_m2=wound.positive_area_m2, negative_wound_area_m2=wound.negative_area_m2,
        opposed_area_relative_error=area_error, unexpected_boundary_face_count=unexpected,
        nonmanifold_face_count=nonmanifold, curve_midpoint_deviation_from_chord_m=curvature_deviation,
        steps=solver.steps, finite=finite, inverted_tetrahedra_peak=inverted_peak,
        minimum_jacobian=minimum_j, center_of_mass_drift_m=com_drift,
        net_momentum_kg_m_s=momentum, mean_wound_gap_m=float(np.mean(gaps)),
        maximum_wound_gap_m=float(np.max(gaps)), two_sided_collision_coverage_fraction=collision_coverage,
        maximum_probe_surface_crossing_m=maximum_crossing, topology_sha256=topology_sha,
        deterministic_trace_sha256=trace_sha, qualified=not failed, failed_gates=failed,
        biomechanical_validation=False, clinical_validation=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_CURVED_PROFILE_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = run_dynamic_curved_cut_qualification(load_curved_profile(args.profile), curved_profile_path=args.profile)
    encoded = json.dumps(receipt.payload(), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if receipt.qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
