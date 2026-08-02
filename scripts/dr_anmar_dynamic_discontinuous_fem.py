#!/usr/bin/env python3
"""Dynamic discontinuous FEM reference for one exact planar incision.

Every tetrahedron owns independent nodal degrees of freedom. Intrinsic
cohesive interfaces keep the intact mesh continuous; interfaces exactly on the
qualified plane are released and become two independently deforming wound
surfaces with one-sided collision. This is deliberately not an arbitrary-cut
runtime: non-conforming cut-cell enrichment remains a later gate.
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

from dr_anmar_cohesive_fracture import build_cohesive_interfaces
from dr_anmar_cuttable_tissue_solver import build_regular_tetrahedral_coupon, load_profile
from dr_anmar_persistent_cut_topology import WoundSurfaceMesh, _closest_point_on_triangle


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DYNAMIC_PROFILE_PATH = (
    REPOSITORY_ROOT / "physics_next/tissues/dr-anmar-dynamic-planar-cut-v1.json"
)


@dataclass(frozen=True)
class DynamicPlanarCutReceipt:
    schema: str
    profile_id: str
    profile_sha256: str
    base_profile_sha256: str
    original_node_count: int
    tetrahedron_count: int
    discontinuous_dof_node_count: int
    cohesive_interface_count: int
    released_interface_count: int
    steps: int
    finite: bool
    inverted_tetrahedra_peak: int
    minimum_jacobian: float
    mass_relative_error: float
    center_of_mass_drift_m: float
    net_momentum_kg_m_s: float
    maximum_intact_interface_jump_m: float
    mean_wound_gap_m: float
    maximum_wound_gap_m: float
    positive_wound_area_m2: float
    negative_wound_area_m2: float
    opposed_area_relative_error: float
    wound_triangle_count: int
    two_sided_collision_coverage_fraction: float
    maximum_probe_surface_crossing_m: float
    deterministic_trace_sha256: str
    qualified: bool
    failed_gates: tuple[str, ...]
    biomechanical_validation: bool
    clinical_validation: bool

    def payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failed_gates"] = list(self.failed_gates)
        return payload


def load_dynamic_profile(path: Path = DEFAULT_DYNAMIC_PROFILE_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class DynamicDiscontinuousFEM:
    def __init__(self, base_profile: dict[str, Any], dynamic_profile: dict[str, Any]):
        self.base_profile = base_profile
        self.dynamic_profile = dynamic_profile
        self.original_points, self.tetrahedra = build_regular_tetrahedral_coupon(base_profile)
        self.tetrahedron_count = len(self.tetrahedra)
        self.rest = self.original_points[self.tetrahedra].reshape((-1, 3)).copy()
        self.position = self.rest.copy()
        prestrain = float(base_profile["geometry"]["prestrain_x"])
        self.position[:, 0] *= 1.0 + prestrain
        self.initial_position = self.position.copy()
        self.velocity = np.zeros_like(self.position)

        local_rest = self.rest.reshape((-1, 4, 3))
        dm = np.stack(
            (
                local_rest[:, 1] - local_rest[:, 0],
                local_rest[:, 2] - local_rest[:, 0],
                local_rest[:, 3] - local_rest[:, 0],
            ),
            axis=2,
        )
        self.dm_inverse = np.linalg.inv(dm)
        self.rest_volume = np.linalg.det(dm) / 6.0
        if np.any(self.rest_volume <= 0.0):
            raise ValueError("Dynamic tetrahedra must have positive rest volume")
        self.shape_gradients = np.empty((self.tetrahedron_count, 4, 3), dtype=np.float64)
        self.shape_gradients[:, 1:, :] = self.dm_inverse
        self.shape_gradients[:, 0, :] = -np.sum(self.dm_inverse, axis=1)

        density = float(base_profile["material"]["density_kg_m3"])
        self.mass = np.repeat(density * self.rest_volume / 4.0, 4)
        original_indices = self.tetrahedra.reshape(-1)
        geometry = base_profile["geometry"]
        half_width = float(geometry["width_m"]) * (1.0 + prestrain) / 2.0
        band = float(geometry["attachment_band_m"])
        self.fixed = np.abs(self.position[:, 0]) >= half_width - band
        self.fixed_position = self.position[self.fixed].copy()
        self.original_indices = original_indices

        interfaces, _ = build_cohesive_interfaces(self.original_points, self.tetrahedra)
        self.interface_nodes = np.asarray(
            [interface.nodes for interface in interfaces], dtype=np.int64
        )
        self.minus_copies = np.empty((len(interfaces), 3), dtype=np.int64)
        self.plus_copies = np.empty_like(self.minus_copies)
        self.rest_interface_normal = np.empty((len(interfaces), 3), dtype=np.float64)
        self.rest_interface_area = np.empty(len(interfaces), dtype=np.float64)
        self.released = np.zeros(len(interfaces), dtype=bool)
        tetrahedron_centroids = np.mean(self.original_points[self.tetrahedra], axis=1)
        for interface_index, interface in enumerate(interfaces):
            left, right = interface.tetrahedra
            normal = np.asarray(interface.normal, dtype=np.float64)
            if np.dot(normal, tetrahedron_centroids[right] - tetrahedron_centroids[left]) < 0.0:
                normal = -normal
            self.rest_interface_normal[interface_index] = normal
            self.rest_interface_area[interface_index] = interface.area_m2
            for vertex_index, original_node in enumerate(interface.nodes):
                left_local = int(np.flatnonzero(self.tetrahedra[left] == original_node)[0])
                right_local = int(np.flatnonzero(self.tetrahedra[right] == original_node)[0])
                self.minus_copies[interface_index, vertex_index] = left * 4 + left_local
                self.plus_copies[interface_index, vertex_index] = right * 4 + right_local

        cut = dynamic_profile["cut_plane"]
        plane_point = np.asarray(cut["point_m"], dtype=np.float64)
        plane_normal = np.asarray(cut["normal"], dtype=np.float64)
        plane_normal /= np.linalg.norm(plane_normal)
        tolerance = float(cut["selection_tolerance_m"])
        interface_points = self.original_points[self.interface_nodes]
        on_plane = np.all(
            np.abs((interface_points - plane_point) @ plane_normal) <= tolerance,
            axis=1,
        )
        alignment = (
            np.abs(np.sum(self.rest_interface_normal * plane_normal[None, :], axis=1))
            >= 1.0 - 1.0e-12
        )
        self.cut_interfaces = on_plane & alignment
        self.released = np.zeros(len(interfaces), dtype=bool)
        if not np.any(self.cut_interfaces):
            raise ValueError("Qualified cut plane must coincide with internal FEM interfaces")
        self.cut_normal = plane_normal
        self.prony_history = np.zeros((self.tetrahedron_count, 3, 3), dtype=np.float64)
        self.previous_elastic_stress = np.zeros_like(self.prony_history)
        self.steps = 0

    def release_cut(self) -> None:
        self.released[self.cut_interfaces] = True

    def _deformation(self) -> tuple[np.ndarray, np.ndarray]:
        local = self.position.reshape((-1, 4, 3))
        ds = np.stack(
            (local[:, 1] - local[:, 0], local[:, 2] - local[:, 0], local[:, 3] - local[:, 0]),
            axis=2,
        )
        deformation = np.einsum("tij,tjk->tik", ds, self.dm_inverse)
        return deformation, np.linalg.det(deformation)

    def _bulk_force(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        material = self.base_profile["material"]
        youngs = float(material["youngs_modulus_pa"])
        poisson = float(material["poisson_ratio"])
        mu = youngs / (2.0 * (1.0 + poisson))
        lam = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        deformation, jacobian = self._deformation()
        safe_j = np.maximum(jacobian, 1.0e-9)
        inverse_transpose = np.swapaxes(np.linalg.inv(deformation), 1, 2)
        elastic = (
            mu * (deformation - inverse_transpose)
            + lam * np.log(safe_j)[:, None, None] * inverse_transpose
        )
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
        return local_force.reshape((-1, 3)), jacobian

    def _cohesive_force(self) -> tuple[np.ndarray, np.ndarray]:
        force = np.zeros_like(self.position)
        minus_position = self.position[self.minus_copies]
        plus_position = self.position[self.plus_copies]
        average_triangle = 0.5 * (minus_position + plus_position)
        area_vector = np.cross(
            average_triangle[:, 1] - average_triangle[:, 0],
            average_triangle[:, 2] - average_triangle[:, 0],
        )
        area_twice = np.linalg.norm(area_vector, axis=1)
        normal = area_vector / np.maximum(area_twice[:, None], 1.0e-15)
        orientation = np.sum(normal * self.rest_interface_normal, axis=1)
        normal = np.where(orientation[:, None] < 0.0, -normal, normal)
        area = 0.5 * area_twice
        jump = np.mean(plus_position - minus_position, axis=1)
        relative_velocity = np.mean(
            self.velocity[self.plus_copies] - self.velocity[self.minus_copies], axis=1
        )
        signed_normal = np.sum(jump * normal, axis=1)
        shear = jump - signed_normal[:, None] * normal
        signed_velocity = np.sum(relative_velocity * normal, axis=1)
        shear_velocity = relative_velocity - signed_velocity[:, None] * normal
        fracture = self.base_profile["fracture"]
        penalty = float(fracture["penalty_stiffness_pa_m"])
        compression = float(fracture["compression_stiffness_pa_m"])
        viscosity = float(fracture["cohesive_viscosity_pa_s_m"])
        traction = -penalty * shear - viscosity * shear_velocity
        intact = ~self.released
        normal_stiffness = np.where(
            signed_normal < 0.0,
            compression,
            np.where(intact, penalty, 0.0),
        )
        normal_damping = np.where(intact, viscosity, 0.0)
        traction += (
            -(normal_stiffness * signed_normal + normal_damping * signed_velocity)[:, None] * normal
        )
        traction[self.released] = np.where(
            (signed_normal[self.released] < 0.0)[:, None],
            -compression * signed_normal[self.released, None] * normal[self.released],
            0.0,
        )
        face_force = area[:, None] * traction / 3.0
        for local in range(3):
            np.add.at(force, self.plus_copies[:, local], face_force)
            np.add.at(force, self.minus_copies[:, local], -face_force)
        return force, jump

    def step(self, dt: float, *, opening_traction_pa: float = 0.0) -> dict[str, Any]:
        force, jacobian = self._bulk_force(dt)
        cohesive_force, jump = self._cohesive_force()
        force += cohesive_force
        if opening_traction_pa > 0.0:
            face_force = (
                opening_traction_pa
                * self.rest_interface_area[self.released, None]
                * self.cut_normal
                / 3.0
            )
            for local in range(3):
                np.add.at(force, self.plus_copies[self.released, local], face_force)
                np.add.at(force, self.minus_copies[self.released, local], -face_force)
        damping = math.exp(-float(self.dynamic_profile["solver"]["velocity_damping_per_s"]) * dt)
        self.velocity += dt * force / self.mass[:, None]
        self.velocity *= damping
        self.position += dt * self.velocity
        self.position[self.fixed] = self.fixed_position
        self.velocity[self.fixed] = 0.0
        self.steps += 1
        return {
            "jacobian": jacobian,
            "jump": jump,
            "finite": bool(
                np.isfinite(self.position).all()
                and np.isfinite(self.velocity).all()
                and np.isfinite(force).all()
            ),
        }

    def center_of_mass(self) -> np.ndarray:
        return np.sum(self.mass[:, None] * self.position, axis=0) / np.sum(self.mass)

    def momentum(self) -> np.ndarray:
        return np.sum(self.mass[:, None] * self.velocity, axis=0)

    def wound_surface_mesh(self) -> WoundSurfaceMesh:
        vertices: list[np.ndarray] = []
        triangles: list[tuple[int, int, int]] = []
        normals: list[np.ndarray] = []
        sides: list[int] = []
        keys: list[tuple[int, int, int, int]] = []
        positive_area = 0.0
        negative_area = 0.0
        for interface_index in np.flatnonzero(self.released):
            for copies, outward, side in (
                (self.minus_copies[interface_index], self.cut_normal, 1),
                (self.plus_copies[interface_index], -self.cut_normal, -1),
            ):
                triangle = self.position[copies]
                geometric = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
                area = 0.5 * float(np.linalg.norm(geometric))
                order = (0, 1, 2)
                if float(np.dot(geometric, outward)) < 0.0:
                    order = (0, 2, 1)
                base = len(vertices)
                vertices.extend(triangle[list(order)])
                triangles.append((base, base + 1, base + 2))
                normals.append(outward.copy())
                sides.append(side)
                keys.append((interface_index, side, 0, 0))
                if side > 0:
                    positive_area += area
                else:
                    negative_area += area
        return WoundSurfaceMesh(
            vertices_m=np.asarray(vertices, dtype=np.float64),
            triangles=np.asarray(triangles, dtype=np.int64),
            triangle_normals=np.asarray(normals, dtype=np.float64),
            triangle_sides=np.asarray(sides, dtype=np.int8),
            patch_keys=tuple(keys),
            positive_area_m2=positive_area,
            negative_area_m2=negative_area,
        )


def one_sided_wound_contact_force(
    mesh: WoundSurfaceMesh,
    point_m: np.ndarray,
    velocity_m_s: np.ndarray,
    radius_m: float,
    stiffness_n_m: float,
    damping_n_s_m: float,
    *,
    side: int,
) -> tuple[np.ndarray, float]:
    candidates = np.flatnonzero(mesh.triangle_sides == side)
    best_distance = math.inf
    best_signed = math.inf
    best_normal = np.zeros(3)
    for triangle_index in candidates:
        triangle = mesh.vertices_m[mesh.triangles[triangle_index]]
        closest = _closest_point_on_triangle(np.asarray(point_m), triangle)
        distance = float(np.linalg.norm(np.asarray(point_m) - closest))
        if distance < best_distance:
            normal = mesh.triangle_normals[triangle_index]
            best_distance = distance
            best_signed = float(np.dot(np.asarray(point_m) - closest, normal))
            best_normal = normal
    penetration = max(0.0, radius_m - best_signed)
    if penetration <= 0.0:
        return np.zeros(3), 0.0
    inward_speed = min(0.0, float(np.dot(np.asarray(velocity_m_s), best_normal)))
    magnitude = stiffness_n_m * penetration - damping_n_s_m * inward_speed
    return magnitude * best_normal, max(0.0, -best_signed)


def _probe_collision(mesh: WoundSurfaceMesh, settings: dict[str, Any]) -> tuple[float, float]:
    radius = float(settings["probe_radius_m"])
    mass = float(settings["probe_mass_kg"])
    speed = float(settings["probe_speed_m_s"])
    stiffness = float(settings["normal_stiffness_n_m"])
    damping = float(settings["normal_damping_n_s_m"])
    dt = float(settings["probe_time_step_s"])
    steps = int(settings["probe_steps"])
    hits = 0
    maximum_crossing = 0.0
    for side in (-1, 1):
        triangle_index = int(np.flatnonzero(mesh.triangle_sides == side)[0])
        triangle = mesh.vertices_m[mesh.triangles[triangle_index]]
        normal = mesh.triangle_normals[triangle_index]
        center = np.mean(triangle, axis=0) + 1.5 * radius * normal
        velocity = -speed * normal
        side_hit = False
        for _ in range(steps):
            force, crossing = one_sided_wound_contact_force(
                mesh,
                center,
                velocity,
                radius,
                stiffness,
                damping,
                side=side,
            )
            side_hit = side_hit or float(np.linalg.norm(force)) > 0.0
            maximum_crossing = max(maximum_crossing, crossing)
            velocity += dt * force / mass
            center += dt * velocity
        hits += int(side_hit)
    return hits / 2.0, maximum_crossing


def run_dynamic_planar_cut_qualification(
    dynamic_profile: dict[str, Any] | None = None,
    *,
    dynamic_profile_path: Path = DEFAULT_DYNAMIC_PROFILE_PATH,
) -> DynamicPlanarCutReceipt:
    dynamic_profile = dynamic_profile or load_dynamic_profile(dynamic_profile_path)
    base_path = REPOSITORY_ROOT / dynamic_profile["base_profile"]
    base_profile = load_profile(base_path)
    base_sha = hashlib.sha256(
        json.dumps(base_profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if base_sha != dynamic_profile["base_profile_sha256"]:
        raise ValueError("Dynamic profile is not bound to the current qualified base profile")
    solver = DynamicDiscontinuousFEM(base_profile, dynamic_profile)
    settings = dynamic_profile["solver"]
    dt = float(settings["time_step_s"])
    phases = (
        ("settle", float(settings["intact_settle_s"]), 0.0),
        ("open", float(settings["opening_load_s"]), float(settings["opening_traction_pa"])),
        ("post", float(settings["post_load_s"]), 0.0),
    )
    initial_mass = float(np.sum(solver.mass))
    initial_center = solver.center_of_mass().copy()
    finite = True
    inverted_peak = 0
    minimum_jacobian = math.inf
    maximum_intact_jump = 0.0
    trace: list[tuple[float, ...]] = []
    for phase, duration, traction in phases:
        if phase == "open":
            solver.release_cut()
        steps = int(round(duration / dt))
        for step in range(steps):
            sample = solver.step(dt, opening_traction_pa=traction)
            jacobian = np.asarray(sample["jacobian"])
            jump = np.asarray(sample["jump"])
            finite = finite and bool(sample["finite"])
            inverted_peak = max(inverted_peak, int(np.count_nonzero(jacobian <= 0.0)))
            minimum_jacobian = min(minimum_jacobian, float(np.min(jacobian)))
            maximum_intact_jump = max(
                maximum_intact_jump,
                float(np.max(np.linalg.norm(jump[~solver.released], axis=1))),
            )
            if solver.steps % 20 == 0:
                tracked_gap = jump[solver.cut_interfaces] @ solver.cut_normal
                trace.append(
                    (
                        phase,
                        step,
                        round(float(np.mean(tracked_gap)), 10),
                        round(float(np.min(jacobian)), 9),
                        round(float(np.linalg.norm(solver.momentum())), 12),
                    )
                )
    _, final_jump = solver._cohesive_force()
    gaps = final_jump[solver.released] @ solver.cut_normal
    mesh = solver.wound_surface_mesh()
    collision_coverage, maximum_crossing = _probe_collision(
        mesh, dynamic_profile["wound_collision"]
    )
    mass_error = abs(float(np.sum(solver.mass)) - initial_mass) / initial_mass
    center_drift = float(np.linalg.norm(solver.center_of_mass() - initial_center))
    momentum = float(np.linalg.norm(solver.momentum()))
    area_error = abs(mesh.positive_area_m2 - mesh.negative_area_m2) / max(
        mesh.positive_area_m2, 1.0e-15
    )
    trace_sha = hashlib.sha256(json.dumps(trace, separators=(",", ":")).encode("utf-8")).hexdigest()
    dynamic_sha = hashlib.sha256(
        json.dumps(dynamic_profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    limits = dynamic_profile["qualification"]
    gates = {
        "finite": finite,
        "no_inversion": inverted_peak == 0,
        "minimum_jacobian": minimum_jacobian >= float(limits["minimum_jacobian"]),
        "mass": mass_error <= float(limits["maximum_mass_relative_error"]),
        "center_of_mass": center_drift <= float(limits["maximum_center_of_mass_drift_m"]),
        "momentum": momentum <= float(limits["maximum_net_momentum_kg_m_s"]),
        "intact_seams": maximum_intact_jump <= float(limits["maximum_intact_interface_jump_m"]),
        "minimum_gap": float(np.mean(gaps)) >= float(limits["minimum_mean_wound_gap_m"]),
        "maximum_gap": float(np.mean(gaps)) <= float(limits["maximum_mean_wound_gap_m"]),
        "opposed_area": area_error <= float(limits["maximum_opposed_area_relative_error"]),
        "two_sided_collision": collision_coverage
        >= float(limits["minimum_two_sided_collision_coverage_fraction"]),
        "probe_crossing": maximum_crossing <= float(limits["maximum_probe_surface_crossing_m"]),
    }
    failed = tuple(name for name, passed in gates.items() if not passed)
    return DynamicPlanarCutReceipt(
        schema="dr.anmar.dynamic-planar-cut-receipt.v1",
        profile_id=str(dynamic_profile["id"]),
        profile_sha256=dynamic_sha,
        base_profile_sha256=base_sha,
        original_node_count=len(solver.original_points),
        tetrahedron_count=solver.tetrahedron_count,
        discontinuous_dof_node_count=len(solver.position),
        cohesive_interface_count=len(solver.released),
        released_interface_count=int(np.count_nonzero(solver.released)),
        steps=solver.steps,
        finite=finite,
        inverted_tetrahedra_peak=inverted_peak,
        minimum_jacobian=minimum_jacobian,
        mass_relative_error=mass_error,
        center_of_mass_drift_m=center_drift,
        net_momentum_kg_m_s=momentum,
        maximum_intact_interface_jump_m=maximum_intact_jump,
        mean_wound_gap_m=float(np.mean(gaps)),
        maximum_wound_gap_m=float(np.max(gaps)),
        positive_wound_area_m2=mesh.positive_area_m2,
        negative_wound_area_m2=mesh.negative_area_m2,
        opposed_area_relative_error=area_error,
        wound_triangle_count=len(mesh.triangles),
        two_sided_collision_coverage_fraction=collision_coverage,
        maximum_probe_surface_crossing_m=maximum_crossing,
        deterministic_trace_sha256=trace_sha,
        qualified=not failed,
        failed_gates=failed,
        biomechanical_validation=False,
        clinical_validation=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_DYNAMIC_PROFILE_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = run_dynamic_planar_cut_qualification(
        load_dynamic_profile(args.profile), dynamic_profile_path=args.profile
    )
    encoded = json.dumps(receipt.payload(), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if receipt.qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
