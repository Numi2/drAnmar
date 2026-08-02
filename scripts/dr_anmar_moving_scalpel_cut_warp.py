#!/usr/bin/env python3
"""Warp CUDA dynamics for the moving-scalpel cohesive fracture front."""

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

from dr_anmar_cuttable_tissue_solver import load_profile
from dr_anmar_cuttable_tissue_warp import _clear_force, _neo_hookean_prony_force
from dr_anmar_dynamic_curved_cut_fem import _probe_collision, load_curved_profile
from dr_anmar_dynamic_curved_cut_warp import _accumulate_jacobian_bounds, _integrate_nodes
from dr_anmar_moving_scalpel_cut_fem import (
    DEFAULT_MOVING_PROFILE_PATH,
    MovingScalpelCutFEM,
    _path_poses,
    _work_channels,
    load_moving_profile,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@wp.kernel
def _moving_cohesive_and_wedge_force(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    plus_nodes: wp.array(dtype=wp.int32),
    minus_nodes: wp.array(dtype=wp.int32),
    normals: wp.array(dtype=wp.vec3),
    nodal_area: wp.array(dtype=float),
    released: wp.array(dtype=wp.int32),
    force_x: wp.array(dtype=float),
    force_y: wp.array(dtype=float),
    force_z: wp.array(dtype=float),
    penalty: float,
    compression: float,
    viscosity: float,
    blade_center: wp.vec3,
    blade_active: int,
    wedge_half_width: float,
    wedge_target_gap: float,
    wedge_stiffness: float,
    wedge_peak_traction: float,
):
    pair = wp.tid()
    plus = plus_nodes[pair]
    minus = minus_nodes[pair]
    normal = normals[pair]
    jump = positions[plus] - positions[minus]
    relative_velocity = velocities[plus] - velocities[minus]
    area = nodal_area[pair]
    pair_force = wp.vec3(0.0, 0.0, 0.0)
    if released[pair] == 0:
        pair_force = area * (-penalty * jump - viscosity * relative_velocity)
    else:
        gap = wp.dot(jump, normal)
        speed = wp.dot(relative_velocity, normal)
        if gap < 0.0:
            traction = -compression * gap - viscosity * wp.min(speed, 0.0)
            pair_force = area * traction * normal
        if blade_active != 0:
            midpoint = 0.5 * (positions[plus] + positions[minus])
            dx = midpoint[0] - blade_center[0]
            dy = midpoint[1] - blade_center[1]
            if wp.sqrt(dx * dx + dy * dy) <= wedge_half_width:
                wedge_traction = wedge_stiffness * wp.max(wedge_target_gap - gap, 0.0)
                wedge_traction = wp.min(wedge_traction, wedge_peak_traction)
                pair_force = pair_force + area * wedge_traction * normal
    wp.atomic_add(force_x, plus, pair_force[0])
    wp.atomic_add(force_y, plus, pair_force[1])
    wp.atomic_add(force_z, plus, pair_force[2])
    wp.atomic_add(force_x, minus, -pair_force[0])
    wp.atomic_add(force_y, minus, -pair_force[1])
    wp.atomic_add(force_z, minus, -pair_force[2])


@dataclass(frozen=True)
class WarpMovingScalpelReceipt:
    schema: str
    profile_id: str
    profile_sha256: str
    warp_version: str
    device: str
    device_is_cuda: bool
    path_segment_count: int
    pseudo_dynamic_step_count: int
    fracture_event_count: int
    released_pair_count: int
    event_trace_sha256: str
    event_trace_matches_cpu: bool
    finite: bool
    inversion_observation_count: int
    minimum_jacobian: float
    mass_relative_error: float
    mean_wound_gap_m: float
    maximum_wound_gap_m: float
    cpu_mean_gap_absolute_error_m: float
    cpu_minimum_jacobian_absolute_error: float
    two_sided_collision_coverage_fraction: float
    maximum_probe_surface_crossing_m: float
    qualified: bool
    failed_gates: tuple[str, ...]
    cuda_promotion_pending: bool
    real_time_transient: bool
    biomechanical_validation: bool
    clinical_validation: bool

    def payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failed_gates"] = list(self.failed_gates)
        return payload


def run_warp_moving_scalpel(
    moving_profile: dict[str, Any] | None = None,
    *,
    moving_profile_path: Path = DEFAULT_MOVING_PROFILE_PATH,
    device: str = "cpu",
) -> WarpMovingScalpelReceipt:
    moving = moving_profile or load_moving_profile(moving_profile_path)
    base = load_profile(REPOSITORY_ROOT / moving["base_profile"])
    curved = load_curved_profile(REPOSITORY_ROOT / moving["embedded_profile"])
    cpu_receipt = json.loads(
        (REPOSITORY_ROOT / "physics_next/receipts/moving-scalpel-cut-reference.json").read_text()
    )
    solver = MovingScalpelCutFEM(base, curved, moving)
    wp.init()
    device_name = str(wp.get_device(device))
    is_cuda = bool(wp.get_device(device).is_cuda)

    mesh = solver.mesh
    positions = wp.array(solver.position.astype(np.float32), dtype=wp.vec3, device=device)
    velocities = wp.array(solver.velocity.astype(np.float32), dtype=wp.vec3, device=device)
    tetrahedra = wp.array(mesh.tetrahedra.astype(np.int32), dtype=wp.vec4i, device=device)
    inverse_rest = wp.array(mesh.dm_inverse.astype(np.float32), dtype=wp.mat33, device=device)
    gradients = wp.array(mesh.shape_gradients.reshape((-1, 3)).astype(np.float32), dtype=wp.vec3, device=device)
    volumes = wp.array(mesh.rest_volume.astype(np.float32), dtype=float, device=device)
    history = wp.array(mesh.prony_history.astype(np.float32), dtype=wp.mat33, device=device)
    previous = wp.array(mesh.previous_elastic_stress.astype(np.float32), dtype=wp.mat33, device=device)
    masses = wp.array(mesh.mass.astype(np.float32), dtype=float, device=device)
    fixed = wp.array(mesh.fixed.astype(np.int32), dtype=wp.int32, device=device)
    fixed_positions = wp.array(mesh.position.astype(np.float32), dtype=wp.vec3, device=device)
    plus_nodes = wp.array(mesh.gap_plus_nodes.astype(np.int32), dtype=wp.int32, device=device)
    minus_nodes = wp.array(mesh.gap_minus_nodes.astype(np.int32), dtype=wp.int32, device=device)
    gap_normals = wp.array(mesh.gap_normals.astype(np.float32), dtype=wp.vec3, device=device)
    gap_area = wp.array(mesh.gap_area.astype(np.float32), dtype=float, device=device)
    released = wp.array(solver.released.astype(np.int32), dtype=wp.int32, device=device)
    node_count = len(mesh.position)
    element_count = len(mesh.tetrahedra)
    force_x = wp.zeros(node_count, dtype=float, device=device)
    force_y = wp.zeros(node_count, dtype=float, device=device)
    force_z = wp.zeros(node_count, dtype=float, device=device)
    jacobian = wp.zeros(element_count, dtype=float, device=device)
    minimum_jacobian = wp.full(1, 1.0e9, dtype=float, device=device)
    inversion_count = wp.zeros(1, dtype=wp.int32, device=device)

    material = base["material"]
    fracture = base["fracture"]
    qs = moving["quasi_static_solver"]
    youngs = float(material["youngs_modulus_pa"])
    poisson = float(material["poisson_ratio"])
    shear = youngs / (2.0 * (1.0 + poisson))
    lame = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    dt = float(qs["pseudo_time_step_s"])
    decay = math.exp(-dt / float(material["prony_time_constant_s"]))
    damping = math.exp(-float(qs["velocity_damping_per_s"]) * dt)

    def dynamic_step(blade_center: np.ndarray | None):
        wp.launch(_clear_force, dim=node_count, inputs=[force_x, force_y, force_z], device=device)
        wp.launch(
            _neo_hookean_prony_force,
            dim=element_count,
            inputs=[positions, tetrahedra, inverse_rest, gradients, volumes, history, previous,
                    force_x, force_y, force_z, jacobian, shear, lame,
                    float(material["prony_relaxation_fraction"]), decay], device=device,
        )
        center = wp.vec3(0.0, 0.0, 0.0) if blade_center is None else wp.vec3(*map(float, blade_center))
        wp.launch(
            _moving_cohesive_and_wedge_force,
            dim=len(mesh.gap_plus_nodes),
            inputs=[positions, velocities, plus_nodes, minus_nodes, gap_normals, gap_area, released,
                    force_x, force_y, force_z, float(fracture["penalty_stiffness_pa_m"]),
                    float(fracture["compression_stiffness_pa_m"]), float(fracture["cohesive_viscosity_pa_s_m"]),
                    center, int(blade_center is not None), float(qs["blade_wedge_half_width_m"]),
                    float(qs["blade_wedge_target_gap_m"]), float(qs["blade_wedge_stiffness_pa_m"]),
                    float(qs["blade_wedge_peak_traction_pa"])], device=device,
        )
        wp.launch(_accumulate_jacobian_bounds, dim=element_count, inputs=[jacobian, minimum_jacobian, inversion_count], device=device)
        wp.launch(
            _integrate_nodes, dim=node_count,
            inputs=[positions, velocities, masses, fixed, fixed_positions, force_x, force_y, force_z, dt, damping],
            device=device,
        )

    poses = _path_poses(moving, curved)
    work = _work_channels(base, moving)
    steps = 0
    for segment, (start, end) in enumerate(zip(poses[:-1], poses[1:], strict=True)):
        solver.advance_blade(segment, start, end, work)
        released.assign(solver.released.astype(np.int32))
        blade_center = np.asarray(end.center_m, dtype=np.float64)
        for _ in range(int(qs["relaxation_steps_per_segment"])):
            dynamic_step(blade_center)
            steps += 1
    for _ in range(int(qs["post_cut_relaxation_steps"])):
        dynamic_step(None)
        steps += 1
    wp.synchronize_device(device)

    solver.position[:] = positions.numpy().astype(np.float64)
    solver.velocity[:] = velocities.numpy().astype(np.float64)
    finite = bool(np.isfinite(solver.position).all() and np.isfinite(solver.velocity).all())
    wound = solver.released_wound_mesh()
    collision, crossing = _probe_collision(wound, curved["wound_collision"])
    gaps = np.sum(
        (solver.position[mesh.gap_plus_nodes[solver.released]] - solver.position[mesh.gap_minus_nodes[solver.released]])
        * mesh.gap_normals[solver.released], axis=1
    )
    total_mass = float(np.sum(mesh.mass))
    expected_mass = float(base["material"]["density_kg_m3"]) * float(np.sum(mesh.rest_volume))
    mass_error = abs(total_mass - expected_mass) / expected_mass
    min_j = float(minimum_jacobian.numpy()[0])
    inversions = int(inversion_count.numpy()[0])
    mean_gap = float(np.mean(gaps))
    trace_sha = hashlib.sha256(json.dumps(solver.event_trace, separators=(",", ":")).encode()).hexdigest()
    event_match = trace_sha == cpu_receipt["event_trace_sha256"]
    gap_error = abs(mean_gap - float(cpu_receipt["mean_wound_gap_m"]))
    jacobian_error = abs(min_j - float(cpu_receipt["minimum_jacobian"]))
    limits = moving["qualification"]
    gates = {
        "cuda_device": is_cuda,
        "event_trace": event_match,
        "event_count": solver.field.fracture_event_count == int(cpu_receipt["fracture_event_count"]),
        "release_count": int(np.count_nonzero(solver.released)) == int(cpu_receipt["released_pair_count"]),
        "finite": finite,
        "no_inversion": inversions == 0,
        "jacobian": min_j >= float(limits["minimum_jacobian"]),
        "mass": mass_error <= float(limits["maximum_mass_relative_error"]),
        "minimum_gap": mean_gap >= float(limits["minimum_mean_wound_gap_m"]),
        "maximum_gap": mean_gap <= float(limits["maximum_mean_wound_gap_m"]),
        "cpu_gap_parity": gap_error <= 0.00001,
        "cpu_jacobian_parity": jacobian_error <= 0.01,
        "collision": collision >= float(limits["minimum_two_sided_collision_coverage_fraction"]),
        "probe_crossing": crossing <= float(limits["maximum_probe_surface_crossing_m"]),
    }
    failed = tuple(name for name, passed in gates.items() if not passed)
    profile_sha = hashlib.sha256(json.dumps(moving, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return WarpMovingScalpelReceipt(
        schema="dr.anmar.warp-moving-scalpel-cut-receipt.v1", profile_id=str(moving["id"]),
        profile_sha256=profile_sha, warp_version=str(wp.__version__), device=device_name,
        device_is_cuda=is_cuda, path_segment_count=len(poses) - 1, pseudo_dynamic_step_count=steps,
        fracture_event_count=solver.field.fracture_event_count,
        released_pair_count=int(np.count_nonzero(solver.released)), event_trace_sha256=trace_sha,
        event_trace_matches_cpu=event_match, finite=finite, inversion_observation_count=inversions,
        minimum_jacobian=min_j, mass_relative_error=mass_error, mean_wound_gap_m=mean_gap,
        maximum_wound_gap_m=float(np.max(gaps)), cpu_mean_gap_absolute_error_m=gap_error,
        cpu_minimum_jacobian_absolute_error=jacobian_error,
        two_sided_collision_coverage_fraction=collision, maximum_probe_surface_crossing_m=crossing,
        qualified=not failed, failed_gates=failed, cuda_promotion_pending=not (is_cuda and not failed),
        real_time_transient=False, biomechanical_validation=False, clinical_validation=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_MOVING_PROFILE_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = run_warp_moving_scalpel(load_moving_profile(args.profile), moving_profile_path=args.profile, device=args.device)
    encoded = json.dumps(receipt.payload(), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if receipt.qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
