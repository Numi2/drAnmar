#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Load an authored patient liver TetMesh into Newton VBD.

This is deliberately a solver-integration smoke test, not a calibration or
promotion benchmark. Attachment bands are selected geometrically because no
clinician-reviewed ligament/vascular attachment map exists yet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import newton
import numpy as np
import warp as wp


YOUNGS_MODULUS_PA = 18_000.0
POISSON_RATIO = 0.47
DENSITY_KG_M3 = 1_060.0
DT = 0.01


@wp.kernel
def apply_geometric_fixture(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    rest_q: wp.array(dtype=wp.vec3),
    rest_inv_mass: wp.array(dtype=float),
    constraint_kind: wp.array(dtype=wp.int32),
    offset_control: wp.array(dtype=wp.vec3),
    pull_active: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    kind = constraint_kind[index]
    if kind == 1:
        particle_q[index] = rest_q[index]
        particle_qd[index] = wp.vec3(0.0)
        particle_inv_mass[index] = 0.0
    elif kind == 2 and pull_active[0] == 1:
        particle_q[index] = rest_q[index] + offset_control[0]
        particle_qd[index] = wp.vec3(0.0)
        particle_inv_mass[index] = 0.0
    else:
        particle_inv_mass[index] = rest_inv_mass[index]


def signed_tet_volumes(positions: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    a = positions[tetrahedra[:, 0]]
    b = positions[tetrahedra[:, 1]]
    c = positions[tetrahedra[:, 2]]
    d = positions[tetrahedra[:, 3]]
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile)) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Dr.Anmar patient liver Newton smoke test")
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=2)
    args = parser.parse_args()

    mesh_path = args.mesh.expanduser().resolve()
    payload = np.load(mesh_path)
    vertices_world = np.asarray(payload["vertices_m"], dtype=np.float32)
    tetrahedra = np.asarray(payload["tetrahedra"], dtype=np.int32)
    if vertices_world.ndim != 2 or vertices_world.shape[1] != 3:
        raise ValueError("vertices_m must have shape (N, 3)")
    if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
        raise ValueError("tetrahedra must have shape (M, 4)")
    if int(tetrahedra.min()) < 0 or int(tetrahedra.max()) >= vertices_world.shape[0]:
        raise ValueError("tetrahedra contain out-of-range vertex indices")
    if not bool(np.isfinite(vertices_world).all()):
        raise ValueError("vertices_m contain non-finite coordinates")

    # Preserve the original patient-space bounds in the report while centering
    # the actual solver state to improve floating-point conditioning.
    source_bounds_min = vertices_world.min(axis=0)
    source_bounds_max = vertices_world.max(axis=0)
    source_center = 0.5 * (source_bounds_min + source_bounds_max)
    vertices = vertices_world - source_center
    rest_volumes = signed_tet_volumes(vertices.astype(np.float64), tetrahedra)
    if bool(np.any(np.abs(rest_volumes) <= 1.0e-12)):
        raise ValueError("patient liver TetMesh contains degenerate tetrahedra")

    shear = YOUNGS_MODULUS_PA / (2.0 * (1.0 + POISSON_RATIO))
    lame = YOUNGS_MODULUS_PA * POISSON_RATIO / (
        (1.0 + POISSON_RATIO) * (1.0 - 2.0 * POISSON_RATIO)
    )
    tet_mesh = newton.TetMesh(vertices, tetrahedra.reshape(-1))
    builder = newton.ModelBuilder(gravity=0.0)
    builder.add_soft_mesh(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        mesh=tet_mesh,
        density=DENSITY_KG_M3,
        k_mu=shear,
        k_lambda=lame,
        k_damp=0.01,
        particle_radius=0.004,
    )
    coloring_started = time.perf_counter()
    builder.color()
    coloring_s = time.perf_counter() - coloring_started
    model = builder.finalize(device=args.device)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=max(1, args.iterations),
        particle_enable_self_contact=False,
        particle_enable_tile_solve=False,
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    rest_q = wp.array(vertices, dtype=wp.vec3, device=args.device)
    rest_inv_mass = wp.clone(model.particle_inv_mass)
    x = vertices[:, 0]
    span_x = max(float(np.ptp(x)), 1.0e-6)
    fixture_mask = x <= float(np.min(x)) + 0.025 * span_x
    pull_mask = x >= float(np.max(x)) - 0.025 * span_x
    if int(fixture_mask.sum()) < 1 or int(pull_mask.sum()) < 1:
        raise RuntimeError("geometric patient-liver fixture bands are empty")
    constraint_kind_np = np.zeros(vertices.shape[0], dtype=np.int32)
    constraint_kind_np[fixture_mask] = 1
    constraint_kind_np[pull_mask] = 2
    constraint_kind = wp.array(constraint_kind_np, dtype=wp.int32, device=args.device)
    offset_control = wp.zeros(1, dtype=wp.vec3, device=args.device)
    pull_active = wp.zeros(1, dtype=wp.int32, device=args.device)

    requested_steps = max(2, args.steps)
    step_ms: list[float] = []
    finite_samples = 0
    peak_displacement = 0.0
    for step in range(requested_steps):
        progress = min(1.0, (step + 1) / max(1.0, requested_steps * 0.6))
        offset_control.assign(np.asarray([(0.002 * progress, 0.0, 0.0005 * progress)], dtype=np.float32))
        pull_active.assign(np.asarray([1 if step < requested_steps - 2 else 0], dtype=np.int32))
        wp.launch(
            apply_geometric_fixture,
            dim=model.particle_count,
            inputs=[
                state_0.particle_q,
                state_0.particle_qd,
                model.particle_inv_mass,
                rest_q,
                rest_inv_mass,
                constraint_kind,
                offset_control,
                pull_active,
            ],
            device=args.device,
        )
        state_0.clear_forces()
        wp.synchronize()
        started = time.perf_counter()
        solver.step(state_0, state_1, control, contacts, DT)
        wp.synchronize()
        step_ms.append((time.perf_counter() - started) * 1000.0)
        state_0, state_1 = state_1, state_0
        current = np.asarray(state_0.particle_q.numpy(), dtype=np.float64)
        velocity = np.asarray(state_0.particle_qd.numpy(), dtype=np.float64)
        finite = bool(np.isfinite(current).all() and np.isfinite(velocity).all())
        finite_samples += int(finite)
        peak_displacement = max(peak_displacement, float(np.linalg.norm(current - vertices, axis=1).max()))
        if not finite:
            break

    final_q = np.asarray(state_0.particle_q.numpy(), dtype=np.float64)
    final_volumes = signed_tet_volumes(final_q, tetrahedra)
    volume_error = abs(float(np.sum(np.abs(final_volumes))) - float(np.sum(np.abs(rest_volumes)))) / max(
        float(np.sum(np.abs(rest_volumes))), 1.0e-12
    )
    inverted = int(np.count_nonzero(final_volumes * rest_volumes <= 0.0))
    result = {
        "schema": "dr.anmar.patient-tetmesh-smoke.v1",
        "benchmark_scope": "patient_asset_solver_smoke_not_promotion",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": "newton_vbd",
        "mesh": str(mesh_path),
        "mesh_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        "device": args.device,
        "software": {
            "newton": getattr(newton, "__version__", None),
            "warp": getattr(wp, "__version__", None),
        },
        "asset": {
            "vertices": int(vertices.shape[0]),
            "tetrahedra": int(tetrahedra.shape[0]),
            "source_bounds_min_m": source_bounds_min.tolist(),
            "source_bounds_max_m": source_bounds_max.tolist(),
            "solver_center_offset_m": source_center.tolist(),
        },
        "solver_settings": {
            "steps": requested_steps,
            "dt_s": DT,
            "iterations": max(1, args.iterations),
            "substeps": 1,
            "material_profile": "liver_research_default_2026_07",
            "attachment_selection": "geometric_extreme_x_bands_not_anatomical",
            "fixture_nodes": int(fixture_mask.sum()),
            "pull_nodes": int(pull_mask.sum()),
        },
        "metrics": {
            "steps_completed": len(step_ms),
            "finite_state_fraction": finite_samples / max(len(step_ms), 1),
            "coloring_s": coloring_s,
            "physics_step_ms_p50": percentile(step_ms, 50),
            "physics_step_ms_p95": percentile(step_ms, 95),
            "physics_step_ms_max": max(step_ms, default=0.0),
            "tissue_displacement_m_peak": peak_displacement,
            "volume_error_fraction_final": volume_error,
            "inverted_tetrahedra_final": inverted,
        },
        "final_state_sha256": hashlib.sha256(np.asarray(final_q, dtype="<f8").tobytes()).hexdigest(),
        "calibration_status": "research_defaults_unvalidated",
        "clinical_validation": False,
        "promotion_allowed": False,
        "limitations": [
            "This checks that the authored patient liver TetMesh loads, colors, advances and remains finite in Newton VBD.",
            "Geometric attachment bands are not anatomical boundary conditions and must not be used for biomechanical claims.",
            "No patient-specific material calibration or validation is applied.",
        ],
    }
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["metrics"]["finite_state_fraction"] == 1.0 and inverted == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
