#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical cross-backend tissue-coupon benchmark for physics-next.

Run this only inside the isolated Isaac Sim 6 / Isaac Lab 3 environment.  It
uses the same geometry, material seed, kinematic attachment, pull trajectory,
time step, and metrics for PhysX FEM and Newton VBD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from isaaclab_tasks.utils.sim_launcher import launch_simulation


parser = argparse.ArgumentParser(description="Dr.Anmar physics-next tissue benchmark")
parser.add_argument("--backend", choices=("physx", "newton"), default="physx")
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--iterations", type=int, default=8)
parser.add_argument("--substeps", type=int, default=10)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--replay-reference", type=Path)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--viz", "--visualizer", dest="visualizer", choices=("kit", "newton", "none"), default=None)
args_cli = parser.parse_args()
args_cli.visualizer_explicit = any(
    token == "--viz" or token == "--visualizer" or token.startswith("--viz=") or token.startswith("--visualizer=")
    for token in sys.argv[1:]
)
if args_cli.visualizer == "none":
    args_cli.visualizer = None

import numpy as np
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObjectCfg


YOUNGS_MODULUS_PA = 18_000.0
POISSON_RATIO = 0.47
DENSITY_KG_M3 = 1_060.0
DT = 0.01
PARTICLE_RADIUS_M = 0.005
TOOL_RADIUS_M = 0.008
SOFT_CONTACT_MARGIN_M = 0.008
SOFT_CONTACT_STIFFNESS_N_M = 75_000.0
SOFT_CONTACT_DAMPING = 1.0e-4
SOFT_CONTACT_FRICTION = 0.22


def smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile)) if values else 0.0


def create_physics():
    if args_cli.backend == "newton":
        from isaaclab_newton.physics import NewtonCfg
        from isaaclab_contrib.deformable import VBDSolverCfg

        return NewtonCfg(
            solver_cfg=VBDSolverCfg(
                iterations=10,
                particle_enable_self_contact=False,
                particle_collision_detection_interval=-1,
            ),
            num_substeps=10,
            use_cuda_graph=True,
        )
    from isaaclab_physx.physics import PhysxCfg

    return PhysxCfg()


def create_physx_tissue():
    # Importing the concrete asset class starts the Isaac Sim extension path,
    # so it must remain inside the already-launched PhysX runtime.
    from isaaclab.assets import DeformableObject
    from isaaclab_physx.sim.schemas import PhysxDeformableBodyPropertiesCfg
    from isaaclab_physx.sim.spawners.materials import PhysxDeformableBodyMaterialCfg

    deformable_props = PhysxDeformableBodyPropertiesCfg()
    material = PhysxDeformableBodyMaterialCfg(
        density=DENSITY_KG_M3,
        youngs_modulus=YOUNGS_MODULUS_PA,
        poissons_ratio=POISSON_RATIO,
        dynamic_friction=0.22,
        static_friction=0.34,
    )
    return DeformableObject(
        DeformableObjectCfg(
            prim_path="/World/DrAnmarTissueCoupon",
            spawn=sim_utils.MeshCuboidCfg(
                size=(0.12, 0.08, 0.05),
                deformable_props=deformable_props,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.48, 0.08, 0.06),
                    roughness=0.62,
                ),
                physics_material=material,
            ),
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.08)),
        )
    )


def benchmark_result(
    *,
    backend: str,
    device: str,
    software: dict,
    samples: int,
    finite_samples: int,
    step_ms: list[float],
    peak_displacement: float,
    peak_speed: float,
    residual: float,
    max_bbox_volume_error: float,
    volume_error_fraction: float | None,
    element_volume_error_fraction_p95_peak: float | None,
    element_volume_error_fraction_max_peak: float | None,
    inverted_tetrahedra_peak: int | None,
    contact_penetration_m_max: float | None,
    tool_force_n_peak: float | None,
    tool_force_n_integral: float | None,
    contact_samples: int | None,
    simulation_nodes: int,
    attachment_nodes: int,
    retraction_nodes: int,
    runtime_path: str,
    solver_settings: dict,
) -> dict:
    return {
        "schema": "dr.anmar.physics-benchmark-result.v1",
        "benchmark": "liver-retraction",
        "benchmark_scope": "solver_smoke_before_patient_asset_promotion",
        "backend": backend,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "steps_requested": max(args_cli.steps, 500),
        "steps_completed": samples,
        "dt_s": DT,
        "device": device,
        "software": software,
        "runtime_path": runtime_path,
        "solver_settings": solver_settings,
        "material_profile": "liver_research_default_2026_07",
        "calibration_status": "research_defaults_unvalidated",
        "clinical_validation": False,
        "metrics": {
            "finite_state_fraction": finite_samples / max(samples, 1),
            "physics_step_ms_p50": percentile(step_ms, 50),
            "physics_step_ms_p95": percentile(step_ms, 95),
            "physics_step_ms_max": max(step_ms, default=0.0),
            "tissue_displacement_m_peak": peak_displacement,
            "tissue_speed_m_s_peak": peak_speed,
            "recovery_residual_m": residual,
            "bbox_volume_error_fraction_max_proxy": max_bbox_volume_error,
            "volume_error_fraction": volume_error_fraction,
            "element_volume_error_fraction_p95_peak": element_volume_error_fraction_p95_peak,
            "element_volume_error_fraction_max_peak": element_volume_error_fraction_max_peak,
            "inverted_tetrahedra_peak": inverted_tetrahedra_peak,
            "contact_penetration_m_max": contact_penetration_m_max,
            "tool_force_n_peak": tool_force_n_peak,
            "tool_force_n_integral": tool_force_n_integral,
            "contact_samples": contact_samples,
            "simulation_nodes": simulation_nodes,
            "attachment_nodes": attachment_nodes,
            "retraction_nodes": retraction_nodes,
        },
        "limitations": [
            "Procedural tissue coupon; the separately authored patient-specific liver TetMesh is not promoted by this result.",
            "Global and per-element tetrahedral volume errors are engineering smoke metrics, not biomechanical validation.",
            "Material values are uncalibrated research seeds.",
        ]
        + (
            ["Rigid-tool contact penetration and force gates were not exercised by this backend."]
            if contact_penetration_m_max is None
            else [
                "Tool force is the net normal penalty reaction reconstructed from Newton's per-contact penalty stiffness; tangential friction is excluded from this scalar diagnostic.",
                "The contact fixture is a kinematic spherical research probe, not a clinically calibrated instrument jaw.",
            ]
        ),
    }


def run_physx(sim_cfg: sim_utils.SimulationCfg) -> dict:
    torch.manual_seed(7)
    np.random.seed(7)
    print(json.dumps({"physics_next_stage": "create_simulation_context", "backend": args_cli.backend}), flush=True)
    sim = sim_utils.SimulationContext(sim_cfg)
    print(json.dumps({"physics_next_stage": "simulation_context_ready", "backend": args_cli.backend}), flush=True)
    light_cfg = sim_utils.DomeLightCfg(intensity=1200.0)
    light_cfg.func("/World/Light", light_cfg)
    tissue = create_physx_tissue()
    sim.reset()
    tissue.reset()

    default_state = tissue.data.default_nodal_state_w.torch.clone()
    default_pos = default_state[..., :3].clone()
    targets = tissue.data.nodal_kinematic_target.torch.clone()
    targets[..., :3] = default_pos
    targets[..., 3] = 1.0

    x = default_pos[0, :, 0]
    base_mask = x <= torch.min(x) + 0.008
    grasp_mask = x >= torch.max(x) - 0.008
    if int(base_mask.sum()) < 1 or int(grasp_mask.sum()) < 1:
        raise RuntimeError("The benchmark mesh did not expose attachment nodes")
    targets[0, base_mask, 3] = 0.0

    warmup_steps = 50
    pull_start = 100
    pull_end = 300
    hold_end = 380
    release_end = 420
    requested_steps = max(args_cli.steps, 500)
    step_ms: list[float] = []
    peak_displacement = 0.0
    peak_speed = 0.0
    finite_samples = 0
    samples = 0
    initial_extents = torch.max(default_pos, dim=1).values - torch.min(default_pos, dim=1).values
    initial_bbox_volume = float(torch.prod(initial_extents[0]).item())
    max_bbox_volume_error = 0.0

    for step in range(requested_steps):
        targets[0, base_mask, :3] = default_pos[0, base_mask]
        targets[0, base_mask, 3] = 0.0
        if pull_start <= step < pull_end:
            progress = smoothstep((step - pull_start) / max(1, pull_end - pull_start))
            offset = torch.tensor((0.025 * progress, 0.0, 0.010 * progress), device=sim.device)
            targets[0, grasp_mask, :3] = default_pos[0, grasp_mask] + offset
            targets[0, grasp_mask, 3] = 0.0
        elif pull_end <= step < release_end:
            offset = torch.tensor((0.025, 0.0, 0.010), device=sim.device)
            targets[0, grasp_mask, :3] = default_pos[0, grasp_mask] + offset
            targets[0, grasp_mask, 3] = 0.0 if step < hold_end else 1.0
        else:
            targets[0, grasp_mask, 3] = 1.0

        tissue.write_nodal_kinematic_target_to_sim_index(targets)
        tissue.write_data_to_sim()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        sim.step(render=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        tissue.update(DT)

        current = tissue.data.nodal_pos_w.torch
        velocity = tissue.data.nodal_vel_w.torch
        finite = bool(torch.isfinite(current).all() and torch.isfinite(velocity).all())
        finite_samples += int(finite)
        samples += 1
        displacement = float(torch.linalg.vector_norm(current - default_pos, dim=-1).max().item())
        speed = float(torch.linalg.vector_norm(velocity, dim=-1).max().item())
        peak_displacement = max(peak_displacement, displacement)
        peak_speed = max(peak_speed, speed)
        extents = torch.max(current, dim=1).values - torch.min(current, dim=1).values
        bbox_volume = float(torch.prod(extents[0]).item())
        max_bbox_volume_error = max(
            max_bbox_volume_error,
            abs(bbox_volume - initial_bbox_volume) / max(initial_bbox_volume, 1e-9),
        )
        if step >= warmup_steps:
            step_ms.append(elapsed_ms)
        if not finite:
            break

    final_pos = tissue.data.nodal_pos_w.torch
    free_mask = ~base_mask
    residual = float(
        torch.linalg.vector_norm(final_pos[0, free_mask] - default_pos[0, free_mask], dim=-1).max().item()
    )
    return benchmark_result(
        backend="physx_fem",
        device=str(sim.device),
        software={
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        samples=samples,
        finite_samples=finite_samples,
        step_ms=step_ms,
        peak_displacement=peak_displacement,
        peak_speed=peak_speed,
        residual=residual,
        max_bbox_volume_error=max_bbox_volume_error,
        volume_error_fraction=None,
        element_volume_error_fraction_p95_peak=None,
        element_volume_error_fraction_max_peak=None,
        inverted_tetrahedra_peak=None,
        contact_penetration_m_max=None,
        tool_force_n_peak=None,
        tool_force_n_integral=None,
        contact_samples=None,
        simulation_nodes=int(default_pos.shape[1]),
        attachment_nodes=int(base_mask.sum().item()),
        retraction_nodes=int(grasp_mask.sum().item()),
        runtime_path="isaac_sim_kit_physx_deformable",
        solver_settings={"iterations": None, "substeps": None},
    )


@wp.kernel
def apply_newton_kinematic_targets(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    rest_q: wp.array(dtype=wp.vec3),
    rest_inv_mass: wp.array(dtype=float),
    constraint_kind: wp.array(dtype=wp.int32),
    offset_control: wp.array(dtype=wp.vec3),
    grasp_control: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    kind = constraint_kind[index]
    if kind == 1:
        particle_q[index] = rest_q[index]
        particle_qd[index] = wp.vec3(0.0, 0.0, 0.0)
        particle_inv_mass[index] = 0.0
    elif kind == 2 and grasp_control[0] == 1:
        particle_q[index] = rest_q[index] + offset_control[0]
        particle_qd[index] = wp.vec3(0.0, 0.0, 0.0)
        particle_inv_mass[index] = 0.0
    else:
        particle_inv_mass[index] = rest_inv_mass[index]


@wp.kernel
def apply_newton_tool_target(
    shape_transform: wp.array(dtype=wp.transform),
    tool_shape: int,
    position_control: wp.array(dtype=wp.vec3),
):
    position = position_control[0]
    shape_transform[tool_shape] = wp.transform(position, wp.quat_identity())


@wp.kernel
def measure_newton_tool_contact(
    contact_count: wp.array(dtype=wp.int32),
    contact_particle: wp.array(dtype=wp.int32),
    contact_shape: wp.array(dtype=wp.int32),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    shape_body: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    contact_penalty_k: wp.array(dtype=float),
    substep_slot: int,
    penetration_by_substep: wp.array(dtype=float),
    force_x_by_substep: wp.array(dtype=float),
    force_y_by_substep: wp.array(dtype=float),
    force_z_by_substep: wp.array(dtype=float),
    contacts_by_substep: wp.array(dtype=wp.int32),
):
    contact_index = wp.tid()
    if contact_index >= contact_count[0]:
        return

    particle_index = contact_particle[contact_index]
    shape_index = contact_shape[contact_index]
    body_index = shape_body[shape_index]
    body_transform = wp.transform_identity()
    if body_index >= 0:
        body_transform = body_q[body_index]
    body_point = wp.transform_point(body_transform, contact_body_pos[contact_index])
    normal = contact_normal[contact_index]
    penetration = -(
        wp.dot(normal, particle_q[particle_index] - body_point)
        - particle_radius[particle_index]
    )
    if penetration <= 0.0:
        return

    # This is exactly the elastic normal term used by Newton VBD's
    # _compute_body_particle_contact_force. Damping and tangential friction
    # are deliberately excluded from the scalar normal-reaction diagnostic.
    normal_force = normal * (penetration * contact_penalty_k[contact_index])
    wp.atomic_max(penetration_by_substep, substep_slot, penetration)
    wp.atomic_add(force_x_by_substep, substep_slot, normal_force[0])
    wp.atomic_add(force_y_by_substep, substep_slot, normal_force[1])
    wp.atomic_add(force_z_by_substep, substep_slot, normal_force[2])
    wp.atomic_add(contacts_by_substep, substep_slot, 1)


def run_newton() -> dict:
    """Run VBD through Newton directly so the solver stays genuinely kitless."""
    import newton

    np.random.seed(7)
    device = args_cli.device
    shear = YOUNGS_MODULUS_PA / (2.0 * (1.0 + POISSON_RATIO))
    lame = YOUNGS_MODULUS_PA * POISSON_RATIO / (
        (1.0 + POISSON_RATIO) * (1.0 - 2.0 * POISSON_RATIO)
    )
    builder = newton.ModelBuilder(gravity=0.0)
    builder.add_soft_grid(
        pos=wp.vec3(-0.06, -0.04, 0.055),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=12,
        dim_y=8,
        dim_z=5,
        cell_x=0.01,
        cell_y=0.01,
        cell_z=0.01,
        density=DENSITY_KG_M3,
        k_mu=shear,
        k_lambda=lame,
        # Newton's k_damp is a solver stiffness term, not the dimensionless
        # damping ratio stored in the canonical material contract. This is an
        # explicit, still-unvalidated backend adapter seed.
        k_damp=0.01,
        particle_radius=PARTICLE_RADIUS_M,
    )
    # A world-attached shape whose transform is driven each frame is Newton's
    # lightest kinematic obstacle representation. It participates in the same
    # particle-rigid collision and VBD contact law without adding unnecessary
    # rigid-body solve work to this soft-tissue benchmark.
    tool_shape = builder.add_shape_sphere(
        -1,
        xform=wp.transform((0.0, 0.0, 0.125), wp.quat_identity()),
        radius=TOOL_RADIUS_M,
        label="DrAnmarContactProbeTip",
    )
    builder.color()
    model = builder.finalize(device=device)
    model.soft_contact_ke = SOFT_CONTACT_STIFFNESS_N_M
    model.soft_contact_kd = SOFT_CONTACT_DAMPING
    model.soft_contact_mu = SOFT_CONTACT_FRICTION
    model.shape_material_ke.fill_(SOFT_CONTACT_STIFFNESS_N_M)
    model.shape_material_kd.fill_(SOFT_CONTACT_DAMPING)
    model.shape_material_mu.fill_(SOFT_CONTACT_FRICTION)
    collision_pipeline = newton.CollisionPipeline(
        model,
        soft_contact_margin=SOFT_CONTACT_MARGIN_M,
        deterministic=True,
    )
    solver = newton.solvers.SolverVBD(
        model=model,
        iterations=max(1, args_cli.iterations),
        particle_enable_self_contact=False,
        particle_enable_tile_solve=False,
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = collision_pipeline.contacts()

    default_pos_np = np.asarray(model.particle_q.numpy(), dtype=np.float32)
    default_pos = wp.array(default_pos_np, dtype=wp.vec3, device=device)
    default_inv_mass = wp.clone(model.particle_inv_mass)
    x = default_pos_np[:, 0]
    base_mask = x <= float(np.min(x)) + 0.008
    grasp_mask = x >= float(np.max(x)) - 0.008
    if int(np.sum(base_mask)) < 1 or int(np.sum(grasp_mask)) < 1:
        raise RuntimeError("The Newton benchmark mesh did not expose attachment nodes")
    constraint_kind_np = np.zeros(default_pos_np.shape[0], dtype=np.int32)
    constraint_kind_np[base_mask] = 1
    constraint_kind_np[grasp_mask] = 2
    constraint_kind = wp.array(constraint_kind_np, dtype=wp.int32, device=device)

    warmup_steps = 50
    pull_start = 100
    pull_end = 300
    hold_end = 380
    requested_steps = max(args_cli.steps, 500)
    step_ms: list[float] = []
    peak_displacement = 0.0
    peak_speed = 0.0
    finite_samples = 0
    samples = 0
    initial_extents = np.max(default_pos_np, axis=0) - np.min(default_pos_np, axis=0)
    initial_bbox_volume = float(np.prod(initial_extents))
    max_bbox_volume_error = 0.0
    tet_indices = np.asarray(model.tet_indices.numpy(), dtype=np.int64)

    def signed_tet_volumes(positions: np.ndarray) -> np.ndarray:
        a = positions[tet_indices[:, 0]]
        b = positions[tet_indices[:, 1]]
        c = positions[tet_indices[:, 2]]
        d = positions[tet_indices[:, 3]]
        return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0

    rest_tet_volumes = signed_tet_volumes(default_pos_np.astype(np.float64))
    valid_tets = np.abs(rest_tet_volumes) > 1.0e-12
    if not bool(np.all(valid_tets)):
        raise RuntimeError("The Newton benchmark mesh contains degenerate tetrahedra")
    rest_total_volume = float(np.sum(np.abs(rest_tet_volumes)))
    volume_error_fraction = 0.0
    element_volume_error_fraction_p95_peak = 0.0
    element_volume_error_fraction_max_peak = 0.0
    inverted_tetrahedra_peak = 0
    contact_penetration_m_max = 0.0
    tool_force_n_peak = 0.0
    tool_force_n_integral = 0.0
    contact_samples = 0
    substeps = max(1, args_cli.substeps)
    offset_control = wp.zeros(1, dtype=wp.vec3, device=device)
    grasp_control = wp.zeros(1, dtype=wp.int32, device=device)
    tool_position_control = wp.array([(0.0, 0.0, 0.125)], dtype=wp.vec3, device=device)
    contact_penetration_by_substep = wp.zeros(substeps, dtype=float, device=device)
    contact_force_x_by_substep = wp.zeros(substeps, dtype=float, device=device)
    contact_force_y_by_substep = wp.zeros(substeps, dtype=float, device=device)
    contact_force_z_by_substep = wp.zeros(substeps, dtype=float, device=device)
    contact_count_by_substep = wp.zeros(substeps, dtype=wp.int32, device=device)

    def simulate_frame() -> None:
        source = state_0
        destination = state_1
        # Rebuild the conservative particle/rigid contact set once per 10 ms
        # frame. Newton still evaluates and resolves those contacts on every
        # 1 ms solver substep; repeating broad/narrow phase ten times for this
        # slowly moving kinematic probe only adds discovery overhead.
        wp.launch(
            apply_newton_tool_target,
            dim=1,
            inputs=[
                model.shape_transform,
                tool_shape,
                tool_position_control,
            ],
            device=device,
        )
        collision_pipeline.collide(source, contacts)
        for substep in range(substeps):
            wp.launch(
                apply_newton_kinematic_targets,
                dim=model.particle_count,
                inputs=[
                    source.particle_q,
                    source.particle_qd,
                    model.particle_inv_mass,
                    default_pos,
                    default_inv_mass,
                    constraint_kind,
                    offset_control,
                    grasp_control,
                ],
                device=device,
            )
            source.clear_forces()
            solver.step(source, destination, control, contacts, DT / substeps)
            wp.launch(
                measure_newton_tool_contact,
                dim=contacts.soft_contact_max,
                inputs=[
                    contacts.soft_contact_count,
                    contacts.soft_contact_particle,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_normal,
                    destination.particle_q,
                    model.particle_radius,
                    model.shape_body,
                    source.body_q,
                    solver.body_particle_contact_penalty_k,
                    substep,
                ],
                outputs=[
                    contact_penetration_by_substep,
                    contact_force_x_by_substep,
                    contact_force_y_by_substep,
                    contact_force_z_by_substep,
                    contact_count_by_substep,
                ],
                device=device,
            )
            source, destination = destination, source

    # Compile once, then capture a full 10-substep frame. The control arrays
    # remain mutable between replays, eliminating Python launch overhead while
    # preserving the exact same trajectory and solver work.
    simulate_frame()
    wp.synchronize()
    cuda_graph = None
    if wp.get_device(device).is_cuda:
        try:
            with wp.ScopedCapture(device=device) as capture:
                simulate_frame()
            cuda_graph = capture.graph
        except (RuntimeError, ValueError):
            cuda_graph = None

    for step in range(requested_steps):
        if step < 10:
            tool_z = 0.125
        elif step < 45:
            progress = smoothstep((step - 10) / 35.0)
            tool_z = 0.125 - 0.008 * progress
        elif step < 65:
            tool_z = 0.117
        elif step < 95:
            progress = smoothstep((step - 65) / 30.0)
            tool_z = 0.117 + 0.008 * progress
        else:
            tool_z = 0.125

        if pull_start <= step < pull_end:
            progress = smoothstep((step - pull_start) / max(1, pull_end - pull_start))
            offset = (0.025 * progress, 0.0, 0.010 * progress)
            grasp_active = 1
        elif pull_end <= step < hold_end:
            offset = (0.025, 0.0, 0.010)
            grasp_active = 1
        else:
            offset = (0.0, 0.0, 0.0)
            grasp_active = 0

        offset_control.assign(np.asarray([offset], dtype=np.float32))
        grasp_control.assign(np.asarray([grasp_active], dtype=np.int32))
        tool_position_control.assign(np.asarray([(0.0, 0.0, tool_z)], dtype=np.float32))
        contact_penetration_by_substep.zero_()
        contact_force_x_by_substep.zero_()
        contact_force_y_by_substep.zero_()
        contact_force_z_by_substep.zero_()
        contact_count_by_substep.zero_()
        wp.synchronize()
        started = time.perf_counter()
        if cuda_graph is not None:
            wp.capture_launch(cuda_graph)
        else:
            simulate_frame()
        wp.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        penetration_samples = np.asarray(contact_penetration_by_substep.numpy(), dtype=np.float64)
        force_samples = np.column_stack(
            (
                np.asarray(contact_force_x_by_substep.numpy(), dtype=np.float64),
                np.asarray(contact_force_y_by_substep.numpy(), dtype=np.float64),
                np.asarray(contact_force_z_by_substep.numpy(), dtype=np.float64),
            )
        )
        frame_contact_count = int(np.sum(np.asarray(contact_count_by_substep.numpy(), dtype=np.int64)))
        frame_force = float(np.linalg.norm(force_samples, axis=1).max())
        contact_penetration_m_max = max(contact_penetration_m_max, float(np.max(penetration_samples)))
        tool_force_n_peak = max(tool_force_n_peak, frame_force)
        tool_force_n_integral += frame_force * DT
        contact_samples += frame_contact_count

        current = np.asarray(state_0.particle_q.numpy(), dtype=np.float64)
        velocity = np.asarray(state_0.particle_qd.numpy(), dtype=np.float64)
        finite = bool(np.isfinite(current).all() and np.isfinite(velocity).all())
        finite_samples += int(finite)
        samples += 1
        peak_displacement = max(peak_displacement, float(np.linalg.norm(current - default_pos_np, axis=-1).max()))
        peak_speed = max(peak_speed, float(np.linalg.norm(velocity, axis=-1).max()))
        extents = np.max(current, axis=0) - np.min(current, axis=0)
        bbox_volume = float(np.prod(extents))
        max_bbox_volume_error = max(
            max_bbox_volume_error,
            abs(bbox_volume - initial_bbox_volume) / max(initial_bbox_volume, 1e-9),
        )
        current_tet_volumes = signed_tet_volumes(current)
        element_volume_errors = (
            np.abs(np.abs(current_tet_volumes) - np.abs(rest_tet_volumes))
            / np.abs(rest_tet_volumes)
        )
        volume_error_fraction = max(
            volume_error_fraction,
            abs(float(np.sum(np.abs(current_tet_volumes))) - rest_total_volume)
            / max(rest_total_volume, 1.0e-12),
        )
        element_volume_error_fraction_p95_peak = max(
            element_volume_error_fraction_p95_peak,
            float(np.percentile(element_volume_errors, 95)),
        )
        element_volume_error_fraction_max_peak = max(
            element_volume_error_fraction_max_peak,
            float(np.max(element_volume_errors)),
        )
        inverted_tetrahedra_peak = max(
            inverted_tetrahedra_peak,
            int(np.count_nonzero(current_tet_volumes * rest_tet_volumes <= 0.0)),
        )
        if step >= warmup_steps:
            step_ms.append(elapsed_ms)
        if not finite:
            break

    final_pos = np.asarray(state_0.particle_q.numpy(), dtype=np.float64)
    residual = float(np.linalg.norm(final_pos[~base_mask] - default_pos_np[~base_mask], axis=-1).max())
    result = benchmark_result(
        backend="newton_vbd",
        device=device,
        software={
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "warp": getattr(wp, "__version__", None),
            "newton": getattr(newton, "__version__", None),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        samples=samples,
        finite_samples=finite_samples,
        step_ms=step_ms,
        peak_displacement=peak_displacement,
        peak_speed=peak_speed,
        residual=residual,
        max_bbox_volume_error=max_bbox_volume_error,
        volume_error_fraction=volume_error_fraction,
        element_volume_error_fraction_p95_peak=element_volume_error_fraction_p95_peak,
        element_volume_error_fraction_max_peak=element_volume_error_fraction_max_peak,
        inverted_tetrahedra_peak=inverted_tetrahedra_peak,
        contact_penetration_m_max=contact_penetration_m_max,
        tool_force_n_peak=tool_force_n_peak,
        tool_force_n_integral=tool_force_n_integral,
        contact_samples=contact_samples,
        simulation_nodes=int(default_pos_np.shape[0]),
        attachment_nodes=int(np.sum(base_mask)),
        retraction_nodes=int(np.sum(grasp_mask)),
        runtime_path=(
            "newton_solver_vbd_kitless_cuda_graph"
            if cuda_graph is not None
            else "newton_solver_vbd_kitless_direct"
        ),
        solver_settings={
            "iterations": max(1, args_cli.iterations),
            "substeps": substeps,
            "cuda_graph": cuda_graph is not None,
            "k_damp_adapter_seed": 0.01,
            "contact_fixture": "kinematic_spherical_probe",
            "particle_radius_m": PARTICLE_RADIUS_M,
            "tool_radius_m": TOOL_RADIUS_M,
            "soft_contact_margin_m": SOFT_CONTACT_MARGIN_M,
            "soft_contact_stiffness_n_m": SOFT_CONTACT_STIFFNESS_N_M,
            "soft_contact_damping": SOFT_CONTACT_DAMPING,
            "soft_contact_friction": SOFT_CONTACT_FRICTION,
            "tool_force_diagnostic": "net_elastic_normal_penalty_reaction",
        },
    )
    final_state_sha256 = hashlib.sha256(np.asarray(final_pos, dtype="<f8").tobytes()).hexdigest()
    replay_reference = None
    replay_exact_match = None
    replay_rmse = None
    if args_cli.replay_reference:
        reference_path = args_cli.replay_reference.expanduser().resolve()
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        replay_reference = str(reference_path)
        replay_exact_match = reference.get("final_state_sha256") == final_state_sha256
        # Exact byte equality proves an RMSE of zero. A mismatch remains
        # unmeasured until the trajectory-state artifacts are compared.
        replay_rmse = 0.0 if replay_exact_match else None
    result["final_state_sha256"] = final_state_sha256
    result["deterministic_replay"] = {
        "reference": replay_reference,
        "exact_state_hash_match": replay_exact_match,
        "position_rmse_m": replay_rmse,
    }
    result["metrics"]["deterministic_replay_position_rmse_m"] = replay_rmse
    return result


def main() -> None:
    if args_cli.backend == "newton":
        print(json.dumps({"physics_next_stage": "runtime_ready", "backend": "newton", "kitless": True}), flush=True)
        result = run_newton()
        args_cli.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.expanduser().resolve().write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return

    sim_cfg = sim_utils.SimulationCfg(
        dt=DT,
        gravity=(0.0, 0.0, 0.0),
        device=args_cli.device,
        physics=create_physics(),
        render_interval=20,
    )
    # The launcher inspects the resolved physics config. Newton VBD remains
    # entirely kitless; PhysX starts Kit because that backend requires it.
    env_cfg = SimpleNamespace(sim=sim_cfg)
    print(json.dumps({"physics_next_stage": "launch_runtime", "backend": args_cli.backend}), flush=True)
    with launch_simulation(env_cfg, args_cli):
        print(json.dumps({"physics_next_stage": "runtime_ready", "backend": args_cli.backend}), flush=True)
        result = run_physx(sim_cfg)
    args_cli.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.expanduser().resolve().write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
