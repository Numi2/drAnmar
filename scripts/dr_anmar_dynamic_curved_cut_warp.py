#!/usr/bin/env python3
"""Warp CUDA runtime qualification for curved cut-cell FEM dynamics.

The implicit remesh is deterministic CPU preprocessing. All post-cut bulk
force, Prony history, wound compression, opening traction, and time integration
steps execute on the requested Warp device.
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
import warp as wp

from dr_anmar_cuttable_tissue_warp import _clear_force, _neo_hookean_prony_force
from dr_anmar_dynamic_curved_cut_fem import (
    DEFAULT_CURVED_PROFILE_PATH,
    _build_settled_mesh,
    _level_gradient,
    _probe_collision,
    load_curved_profile,
)
from dr_anmar_cuttable_tissue_solver import load_profile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@wp.kernel
def _wound_compression_force(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    plus_nodes: wp.array(dtype=wp.int32),
    minus_nodes: wp.array(dtype=wp.int32),
    normals: wp.array(dtype=wp.vec3),
    nodal_area: wp.array(dtype=float),
    force_x: wp.array(dtype=float),
    force_y: wp.array(dtype=float),
    force_z: wp.array(dtype=float),
    compression_stiffness: float,
    viscosity: float,
):
    pair = wp.tid()
    plus = plus_nodes[pair]
    minus = minus_nodes[pair]
    normal = normals[pair]
    gap = wp.dot(positions[plus] - positions[minus], normal)
    if gap < 0.0:
        speed = wp.dot(velocities[plus] - velocities[minus], normal)
        traction = -compression_stiffness * gap - viscosity * wp.min(speed, 0.0)
        pair_force = traction * nodal_area[pair] * normal
        wp.atomic_add(force_x, plus, pair_force[0])
        wp.atomic_add(force_y, plus, pair_force[1])
        wp.atomic_add(force_z, plus, pair_force[2])
        wp.atomic_add(force_x, minus, -pair_force[0])
        wp.atomic_add(force_y, minus, -pair_force[1])
        wp.atomic_add(force_z, minus, -pair_force[2])


@wp.kernel
def _wound_opening_force(
    positions: wp.array(dtype=wp.vec3),
    triangles: wp.array(dtype=wp.vec3i),
    material_side: wp.array(dtype=wp.int32),
    cut_normals: wp.array(dtype=wp.vec3),
    force_x: wp.array(dtype=float),
    force_y: wp.array(dtype=float),
    force_z: wp.array(dtype=float),
    opening_traction: float,
):
    triangle_index = wp.tid()
    triangle = triangles[triangle_index]
    p0 = positions[triangle[0]]
    p1 = positions[triangle[1]]
    p2 = positions[triangle[2]]
    area = 0.5 * wp.length(wp.cross(p1 - p0, p2 - p0))
    nodal = (
        float(material_side[triangle_index])
        * opening_traction
        * area
        / 3.0
        * cut_normals[triangle_index]
    )
    for local in range(3):
        node = triangle[local]
        wp.atomic_add(force_x, node, nodal[0])
        wp.atomic_add(force_y, node, nodal[1])
        wp.atomic_add(force_z, node, nodal[2])


@wp.kernel
def _integrate_nodes(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    fixed: wp.array(dtype=wp.int32),
    fixed_position: wp.array(dtype=wp.vec3),
    force_x: wp.array(dtype=float),
    force_y: wp.array(dtype=float),
    force_z: wp.array(dtype=float),
    time_step: float,
    damping: float,
):
    node = wp.tid()
    if fixed[node] != 0:
        positions[node] = fixed_position[node]
        velocities[node] = wp.vec3(0.0, 0.0, 0.0)
    else:
        force = wp.vec3(force_x[node], force_y[node], force_z[node])
        velocity = damping * (velocities[node] + time_step * force / mass[node])
        velocities[node] = velocity
        positions[node] = positions[node] + time_step * velocity


@wp.kernel
def _accumulate_jacobian_bounds(
    jacobian: wp.array(dtype=float),
    minimum_jacobian: wp.array(dtype=float),
    inversion_count: wp.array(dtype=wp.int32),
):
    element = wp.tid()
    value = jacobian[element]
    wp.atomic_min(minimum_jacobian, 0, value)
    if value <= 0.0:
        wp.atomic_add(inversion_count, 0, 1)


@dataclass(frozen=True)
class WarpDynamicCurvedCutReceipt:
    schema: str
    profile_id: str
    profile_sha256: str
    topology_sha256: str
    warp_version: str
    device: str
    device_is_cuda: bool
    remeshed_node_count: int
    remeshed_tetrahedron_count: int
    wound_triangle_count: int
    steps: int
    finite: bool
    inversion_observation_count: int
    minimum_jacobian: float
    mass_relative_error: float
    center_of_mass_drift_m: float
    net_momentum_kg_m_s: float
    mean_wound_gap_m: float
    maximum_wound_gap_m: float
    two_sided_collision_coverage_fraction: float
    maximum_probe_surface_crossing_m: float
    qualified: bool
    failed_gates: tuple[str, ...]
    cuda_promotion_pending: bool
    biomechanical_validation: bool
    clinical_validation: bool

    def payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failed_gates"] = list(self.failed_gates)
        return payload


def _device_name(device: str) -> str:
    wp.init()
    return str(wp.get_device(device))


def run_warp_dynamic_curved_cut(
    curved_profile: dict[str, Any] | None = None,
    *,
    curved_profile_path: Path = DEFAULT_CURVED_PROFILE_PATH,
    device: str = "cpu",
) -> WarpDynamicCurvedCutReceipt:
    curved = curved_profile or load_curved_profile(curved_profile_path)
    base = load_profile(REPOSITORY_ROOT / curved["base_profile"])
    solver = _build_settled_mesh(base, curved)
    retained = json.loads(
        (REPOSITORY_ROOT / "physics_next/receipts/dynamic-curved-cut-reference.json").read_text()
    )
    topology_sha = retained["topology_sha256"]
    profile_sha = hashlib.sha256(
        json.dumps(curved, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    device_name = _device_name(device)
    is_cuda = "cuda" in device_name.lower()

    positions = wp.array(solver.position.astype(np.float32), dtype=wp.vec3, device=device)
    velocities = wp.array(solver.velocity.astype(np.float32), dtype=wp.vec3, device=device)
    tetrahedra = wp.array(solver.tetrahedra.astype(np.int32), dtype=wp.vec4i, device=device)
    inverse_rest = wp.array(solver.dm_inverse.astype(np.float32), dtype=wp.mat33, device=device)
    gradients = wp.array(solver.shape_gradients.reshape((-1, 3)).astype(np.float32), dtype=wp.vec3, device=device)
    volumes = wp.array(solver.rest_volume.astype(np.float32), dtype=float, device=device)
    history = wp.array(solver.prony_history.astype(np.float32), dtype=wp.mat33, device=device)
    previous = wp.array(solver.previous_elastic_stress.astype(np.float32), dtype=wp.mat33, device=device)
    masses = wp.array(solver.mass.astype(np.float32), dtype=float, device=device)
    fixed = wp.array(solver.fixed.astype(np.int32), dtype=wp.int32, device=device)
    fixed_positions = wp.array(solver.position.astype(np.float32), dtype=wp.vec3, device=device)
    plus_nodes = wp.array(solver.gap_plus_nodes.astype(np.int32), dtype=wp.int32, device=device)
    minus_nodes = wp.array(solver.gap_minus_nodes.astype(np.int32), dtype=wp.int32, device=device)
    gap_normals = wp.array(solver.gap_normals.astype(np.float32), dtype=wp.vec3, device=device)
    gap_area = wp.array(solver.gap_area.astype(np.float32), dtype=float, device=device)

    wound_triangles_np = np.concatenate(
        (solver.wound_triangles_by_side[-1], solver.wound_triangles_by_side[1]), axis=0
    ).astype(np.int32)
    wound_sides_np = np.concatenate(
        (-np.ones(len(solver.wound_triangles_by_side[-1]), dtype=np.int32),
         np.ones(len(solver.wound_triangles_by_side[1]), dtype=np.int32))
    )
    wound_centroids = np.mean(solver.rest[wound_triangles_np], axis=1)
    wound_normals_np = _level_gradient(wound_centroids, curved["implicit_cut"]).astype(np.float32)
    wound_triangles = wp.array(wound_triangles_np, dtype=wp.vec3i, device=device)
    wound_sides = wp.array(wound_sides_np, dtype=wp.int32, device=device)
    wound_normals = wp.array(wound_normals_np, dtype=wp.vec3, device=device)

    node_count = len(solver.position)
    element_count = len(solver.tetrahedra)
    force_x = wp.zeros(node_count, dtype=float, device=device)
    force_y = wp.zeros(node_count, dtype=float, device=device)
    force_z = wp.zeros(node_count, dtype=float, device=device)
    jacobian = wp.zeros(element_count, dtype=float, device=device)
    minimum_jacobian = wp.full(1, 1.0e9, dtype=float, device=device)
    inversion_count = wp.zeros(1, dtype=wp.int32, device=device)

    material = base["material"]
    fracture = base["fracture"]
    youngs = float(material["youngs_modulus_pa"])
    poisson = float(material["poisson_ratio"])
    shear = youngs / (2.0 * (1.0 + poisson))
    lame = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    dt = float(curved["solver"]["cut_time_step_s"])
    decay = math.exp(-dt / float(material["prony_time_constant_s"]))
    damping = math.exp(-float(curved["solver"]["velocity_damping_per_s"]) * dt)
    phases = (
        (float(curved["solver"]["opening_load_s"]), float(curved["solver"]["opening_traction_pa"])),
        (float(curved["solver"]["post_load_s"]), 0.0),
    )
    steps = 0
    for duration, traction in phases:
        for _ in range(int(round(duration / dt))):
            wp.launch(
                _clear_force,
                dim=node_count,
                inputs=[force_x, force_y, force_z],
                device=device,
            )
            wp.launch(
                _neo_hookean_prony_force,
                dim=element_count,
                inputs=[positions, tetrahedra, inverse_rest, gradients, volumes, history, previous,
                        force_x, force_y, force_z, jacobian, shear, lame,
                        float(material["prony_relaxation_fraction"]), decay],
                device=device,
            )
            wp.launch(
                _wound_compression_force,
                dim=len(solver.gap_plus_nodes),
                inputs=[positions, velocities, plus_nodes, minus_nodes, gap_normals, gap_area,
                        force_x, force_y, force_z, float(fracture["compression_stiffness_pa_m"]),
                        float(fracture["cohesive_viscosity_pa_s_m"])],
                device=device,
            )
            if traction > 0.0:
                wp.launch(
                    _wound_opening_force,
                    dim=len(wound_triangles_np),
                    inputs=[positions, wound_triangles, wound_sides, wound_normals,
                            force_x, force_y, force_z, traction],
                    device=device,
                )
            wp.launch(
                _accumulate_jacobian_bounds,
                dim=element_count,
                inputs=[jacobian, minimum_jacobian, inversion_count],
                device=device,
            )
            wp.launch(
                _integrate_nodes,
                dim=node_count,
                inputs=[positions, velocities, masses, fixed, fixed_positions,
                        force_x, force_y, force_z, dt, damping],
                device=device,
            )
            steps += 1
    wp.synchronize_device(device)

    final_position = positions.numpy().astype(np.float64)
    final_velocity = velocities.numpy().astype(np.float64)
    solver.position = final_position
    solver.velocity = final_velocity
    finite = bool(np.isfinite(final_position).all() and np.isfinite(final_velocity).all())
    gaps = solver.gaps()
    wound = solver.wound_surface_mesh()
    collision_coverage, maximum_crossing = _probe_collision(wound, curved["wound_collision"])
    total_mass = float(np.sum(solver.mass))
    expected_mass = float(base["material"]["density_kg_m3"]) * float(np.sum(solver.rest_volume))
    initial_position = fixed_positions.numpy().astype(np.float64)
    initial_com = np.sum(solver.mass[:, None] * initial_position, axis=0) / total_mass
    final_com = np.sum(solver.mass[:, None] * final_position, axis=0) / total_mass
    momentum = np.sum(solver.mass[:, None] * final_velocity, axis=0)
    min_j = float(minimum_jacobian.numpy()[0])
    inversions = int(inversion_count.numpy()[0])
    limits = curved["qualification"]
    gates = {
        "cuda_device": is_cuda,
        "finite": finite,
        "no_inversion": inversions == 0,
        "jacobian": min_j >= float(limits["minimum_jacobian"]),
        "mass": abs(total_mass - expected_mass) / expected_mass <= float(limits["maximum_mass_relative_error"]),
        "center_of_mass": float(np.linalg.norm(final_com - initial_com)) <= float(limits["maximum_center_of_mass_drift_m"]),
        "momentum": float(np.linalg.norm(momentum)) <= float(limits["maximum_net_momentum_kg_m_s"]),
        "minimum_gap": float(np.mean(gaps)) >= float(limits["minimum_mean_wound_gap_m"]),
        "maximum_gap": float(np.mean(gaps)) <= float(limits["maximum_mean_wound_gap_m"]),
        "two_sided_collision": collision_coverage >= float(limits["minimum_two_sided_collision_coverage_fraction"]),
        "probe_crossing": maximum_crossing <= float(limits["maximum_probe_surface_crossing_m"]),
    }
    failed = tuple(name for name, passed in gates.items() if not passed)
    return WarpDynamicCurvedCutReceipt(
        schema="dr.anmar.warp-dynamic-curved-cut-receipt.v1", profile_id=str(curved["id"]),
        profile_sha256=profile_sha, topology_sha256=topology_sha, warp_version=str(wp.__version__),
        device=device_name, device_is_cuda=is_cuda, remeshed_node_count=node_count,
        remeshed_tetrahedron_count=element_count, wound_triangle_count=len(wound_triangles_np),
        steps=steps, finite=finite, inversion_observation_count=inversions, minimum_jacobian=min_j,
        mass_relative_error=abs(total_mass - expected_mass) / expected_mass,
        center_of_mass_drift_m=float(np.linalg.norm(final_com - initial_com)),
        net_momentum_kg_m_s=float(np.linalg.norm(momentum)), mean_wound_gap_m=float(np.mean(gaps)),
        maximum_wound_gap_m=float(np.max(gaps)), two_sided_collision_coverage_fraction=collision_coverage,
        maximum_probe_surface_crossing_m=maximum_crossing, qualified=not failed, failed_gates=failed,
        cuda_promotion_pending=not (is_cuda and not failed), biomechanical_validation=False,
        clinical_validation=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_CURVED_PROFILE_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = run_warp_dynamic_curved_cut(
        load_curved_profile(args.profile), curved_profile_path=args.profile, device=args.device
    )
    encoded = json.dumps(receipt.payload(), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if receipt.qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
