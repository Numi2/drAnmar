#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
# pyright: reportInvalidTypeForm=false, reportAttributeAccessIssue=false

"""Qualify the canonical needle-ready tissue TetMesh in NVIDIA Newton VBD.

This is an intact-deformation and rigid-soft contact qualification.  It does
not claim puncture, topology change, calibrated tissue mechanics, or clinical
validity.
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
from pxr import Usd


DEFAULT_ASSET_ROOT = (
    Path(__file__).resolve().parents[1]
    / "source/extensions/orbit.surgical.assets"
    / "data/Props/SurgicalTissue/NeedleReadyTissueUnit"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
YOUNGS_MODULUS_PA = 180_000.0
POISSON_RATIO = 0.47
DENSITY_KG_M3 = 1_050.0
DT = 0.001


@wp.kernel
def apply_constraints(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    rest_q: wp.array(dtype=wp.vec3),
    rest_inv_mass: wp.array(dtype=float),
    constraint_kind: wp.array(dtype=wp.int32),
    retraction_offset: wp.array(dtype=wp.vec3),
    retraction_active: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    kind = constraint_kind[index]
    if kind == 1:
        particle_q[index] = rest_q[index]
        particle_qd[index] = wp.vec3(0.0)
        particle_inv_mass[index] = 0.0
    elif kind == 2 and retraction_active[0] == 1:
        particle_q[index] = rest_q[index] + retraction_offset[0]
        particle_qd[index] = wp.vec3(0.0)
        particle_inv_mass[index] = 0.0
    else:
        particle_inv_mass[index] = rest_inv_mass[index]


@wp.kernel
def set_probe_transform(
    shape_transform: wp.array(dtype=wp.transform),
    shape_index: int,
    position: wp.array(dtype=wp.vec3),
):
    shape_transform[shape_index] = wp.transform(
        position[0], wp.quat_identity()
    )


def smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def signed_tet_volumes(
    positions: np.ndarray, tetrahedra: np.ndarray
) -> np.ndarray:
    a = positions[tetrahedra[:, 0]]
    b = positions[tetrahedra[:, 1]]
    c = positions[tetrahedra[:, 2]]
    d = positions[tetrahedra[:, 3]]
    return np.einsum(
        "ij,ij->i", np.cross(b - a, c - a), d - a
    ) / 6.0


def load_tetmesh(
    usd_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"OpenUSD could not open {usd_path}")
    tet_prims = [
        prim for prim in stage.Traverse() if prim.GetTypeName() == "TetMesh"
    ]
    if len(tet_prims) != 1:
        raise ValueError(
            f"expected exactly one TetMesh, found {len(tet_prims)}"
        )
    prim = tet_prims[0]
    points = np.asarray(prim.GetAttribute("points").Get(), dtype=np.float32)
    tetrahedra = np.asarray(
        prim.GetAttribute("tetVertexIndices").Get(), dtype=np.int32
    )
    component_ids = np.asarray(
        prim.GetAttribute("drAnmar:pointComponentIds").Get(),
        dtype=np.int32,
    )
    node_sets = {}
    for name in (
        "anchor_outer",
        "safe_bite_top",
        "contact_roi_top",
        "wound_edge_top",
    ):
        node_sets[name] = np.asarray(
            prim.GetAttribute(f"drAnmar:nodeSet:{name}").Get(),
            dtype=np.int32,
        )
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("TetMesh points must have shape (N, 3)")
    if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
        raise ValueError("TetMesh tetrahedra must have shape (M, 4)")
    if component_ids.shape != (points.shape[0],):
        raise ValueError("point component IDs do not match TetMesh points")
    if int(tetrahedra.min()) < 0 or int(tetrahedra.max()) >= len(points):
        raise ValueError("TetMesh contains an out-of-range index")
    if not bool(np.isfinite(points).all()):
        raise ValueError("TetMesh contains non-finite points")
    return points, tetrahedra, node_sets, component_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-root", type=Path, default=DEFAULT_ASSET_ROOT
    )
    parser.add_argument(
        "--lod",
        choices=("training", "contact", "validation"),
        default="contact",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument("--contact-probe", action="store_true")
    parser.add_argument("--replay-reference", type=Path)
    parser.add_argument(
        "--volume-audit-interval",
        type=int,
        default=1,
        help="Audit all tetrahedra every N frames; 0 audits the final frame only.",
    )
    args = parser.parse_args()
    if args.instances < 1:
        raise ValueError("instances must be positive")
    if args.volume_audit_interval < 0:
        raise ValueError("volume audit interval cannot be negative")
    if args.contact_probe and args.instances != 1:
        raise ValueError("contact probe qualification requires one instance")

    asset_root = args.asset_root.expanduser().resolve()
    usd_path = asset_root / f"needle_ready_tissue_{args.lod}.usda"
    physics_profile = json.loads(
        (asset_root / "physics_profile.json").read_text(encoding="utf-8")
    )
    points, tetrahedra, node_sets, component_ids = load_tetmesh(usd_path)
    rest_volumes_single = signed_tet_volumes(
        points.astype(np.float64), tetrahedra
    )
    if bool(np.any(rest_volumes_single <= 1.0e-16)):
        raise ValueError("authored TetMesh contains non-positive tetrahedra")

    shear = YOUNGS_MODULUS_PA / (2.0 * (1.0 + POISSON_RATIO))
    lame = YOUNGS_MODULUS_PA * POISSON_RATIO / (
        (1.0 + POISSON_RATIO) * (1.0 - 2.0 * POISSON_RATIO)
    )
    particle_radius = float(
        physics_profile["intact_deformation"][
            "particle_radius_m_by_lod"
        ][args.lod]
    )
    device_handle = wp.get_device(args.device)
    initial_free_memory = int(device_handle.free_memory)
    minimum_free_memory = initial_free_memory
    tet_mesh = newton.TetMesh(points, tetrahedra.reshape(-1))
    template_builder = newton.ModelBuilder(
        gravity=wp.vec3(0.0, 0.0, -9.81)
    )
    template_builder.add_soft_mesh(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        mesh=tet_mesh,
        density=DENSITY_KG_M3,
        k_mu=shear,
        k_lambda=lame,
        k_damp=float(
            physics_profile["intact_deformation"][
                "newton_k_damp_adapter_seed"
            ]
        ),
        particle_radius=particle_radius,
    )

    probe_shape = None
    probe_radius = 0.004
    probe_x = 0.009
    probe_rest_z = float(points[:, 2].max()) + probe_radius + 0.004
    if args.contact_probe:
        probe_shape = template_builder.add_shape_sphere(
            -1,
            xform=wp.transform(
                (probe_x, 0.0, probe_rest_z), wp.quat_identity()
            ),
            radius=probe_radius,
            label="DrAnmarNeedleReadyTissueContactProbe",
        )

    coloring_started = time.perf_counter()
    template_builder.color()
    coloring_s = time.perf_counter() - coloring_started
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, -9.81))
    replication_started = time.perf_counter()
    builder.replicate(
        template_builder,
        args.instances,
        spacing=(0.0, 0.0, 0.0),
    )
    replication_s = time.perf_counter() - replication_started
    finalize_started = time.perf_counter()
    model = builder.finalize(device=args.device)
    finalize_s = time.perf_counter() - finalize_started
    minimum_free_memory = min(
        minimum_free_memory, int(device_handle.free_memory)
    )
    model.soft_contact_ke = 75_000.0
    model.soft_contact_kd = 1.0e-4
    model.soft_contact_mu = 0.22
    model.shape_material_ke.fill_(75_000.0)
    model.shape_material_kd.fill_(1.0e-4)
    model.shape_material_mu.fill_(0.22)
    collision_pipeline = newton.CollisionPipeline(
        model,
        soft_contact_margin=max(0.0015, particle_radius),
        deterministic=True,
    )
    solver = newton.solvers.SolverVBD(
        model,
        iterations=max(1, args.iterations),
        particle_enable_self_contact=False,
        particle_enable_tile_solve=False,
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = collision_pipeline.contacts()

    rest_q_np = np.asarray(model.particle_q.numpy(), dtype=np.float32)
    rest_q = wp.array(rest_q_np, dtype=wp.vec3, device=args.device)
    rest_inv_mass = wp.clone(model.particle_inv_mass)
    point_count = points.shape[0]
    constraint_kind_np = np.zeros(rest_q_np.shape[0], dtype=np.int32)
    retraction_single = node_sets["safe_bite_top"][
        component_ids[node_sets["safe_bite_top"]] == 1
    ]
    if len(retraction_single) == 0:
        raise RuntimeError("right-flap safe-bite node set is empty")
    for instance in range(args.instances):
        offset = instance * point_count
        constraint_kind_np[offset + node_sets["anchor_outer"]] = 1
        constraint_kind_np[offset + retraction_single] = 2
    constraint_kind = wp.array(
        constraint_kind_np, dtype=wp.int32, device=args.device
    )
    retraction_offset = wp.zeros(1, dtype=wp.vec3, device=args.device)
    retraction_active = wp.zeros(1, dtype=wp.int32, device=args.device)
    probe_position = wp.array(
        [(probe_x, 0.0, probe_rest_z)],
        dtype=wp.vec3,
        device=args.device,
    )

    requested_steps = max(20, args.steps)
    substeps = max(1, args.substeps)
    step_ms: list[float] = []
    finite_samples = 0
    peak_displacement = 0.0
    peak_speed = 0.0
    peak_volume_error = 0.0
    inverted_peak = 0
    contact_candidates_peak = 0
    geometric_contact_samples = 0
    maximum_geometric_penetration = 0.0
    rest_tets_all = np.concatenate(
        [
            tetrahedra + instance * point_count
            for instance in range(args.instances)
        ],
        axis=0,
    )
    rest_volumes = signed_tet_volumes(
        rest_q_np.astype(np.float64), rest_tets_all
    )
    rest_total_volume = float(np.sum(np.abs(rest_volumes)))

    for step in range(requested_steps):
        phase = step / max(1, requested_steps - 1)
        if phase < 0.35:
            progress = smoothstep(phase / 0.35)
            offset = (-0.001 * progress, 0.0, 0.002 * progress)
            retract = 1
        elif phase < 0.60:
            offset = (-0.001, 0.0, 0.002)
            retract = 1
        else:
            offset = (0.0, 0.0, 0.0)
            retract = 0
        retraction_offset.assign(np.asarray([offset], dtype=np.float32))
        retraction_active.assign(np.asarray([retract], dtype=np.int32))

        if args.contact_probe:
            assert probe_shape is not None
            if phase < 0.20:
                probe_z = probe_rest_z
            elif phase < 0.45:
                progress = smoothstep((phase - 0.20) / 0.25)
                probe_z = probe_rest_z - 0.0048 * progress
            elif phase < 0.65:
                probe_z = probe_rest_z - 0.0048
            else:
                progress = smoothstep((phase - 0.65) / 0.35)
                probe_z = probe_rest_z - 0.0048 * (1.0 - progress)
            probe_position.assign(
                np.asarray([(probe_x, 0.0, probe_z)], dtype=np.float32)
            )
            wp.launch(
                set_probe_transform,
                dim=1,
                inputs=[
                    model.shape_transform,
                    int(probe_shape),
                    probe_position,
                ],
                device=args.device,
            )

        wp.synchronize()
        started = time.perf_counter()
        for _ in range(substeps):
            wp.launch(
                apply_constraints,
                dim=model.particle_count,
                inputs=[
                    state_0.particle_q,
                    state_0.particle_qd,
                    model.particle_inv_mass,
                    rest_q,
                    rest_inv_mass,
                    constraint_kind,
                    retraction_offset,
                    retraction_active,
                ],
                device=args.device,
            )
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(
                state_0,
                state_1,
                control,
                contacts,
                DT / substeps,
            )
            state_0, state_1 = state_1, state_0
        wp.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        minimum_free_memory = min(
            minimum_free_memory, int(device_handle.free_memory)
        )
        if step >= min(10, requested_steps // 4):
            step_ms.append(elapsed_ms)

        current = np.asarray(state_0.particle_q.numpy(), dtype=np.float64)
        velocity = np.asarray(state_0.particle_qd.numpy(), dtype=np.float64)
        finite = bool(
            np.isfinite(current).all() and np.isfinite(velocity).all()
        )
        finite_samples += int(finite)
        if not finite:
            break
        peak_displacement = max(
            peak_displacement,
            float(np.linalg.norm(current - rest_q_np, axis=1).max()),
        )
        peak_speed = max(
            peak_speed, float(np.linalg.norm(velocity, axis=1).max())
        )
        audit_this_frame = (
            args.volume_audit_interval > 0
            and (
                step % args.volume_audit_interval == 0
                or step == requested_steps - 1
            )
        )
        if audit_this_frame:
            volumes = signed_tet_volumes(current, rest_tets_all)
            inverted_peak = max(
                inverted_peak,
                int(np.count_nonzero(volumes * rest_volumes <= 0.0)),
            )
            peak_volume_error = max(
                peak_volume_error,
                abs(float(np.sum(np.abs(volumes))) - rest_total_volume)
                / max(rest_total_volume, 1.0e-12),
            )
        contact_candidates_peak = max(
            contact_candidates_peak,
            int(np.asarray(contacts.soft_contact_count.numpy())[0]),
        )
        if args.contact_probe:
            probe_now = np.asarray(probe_position.numpy())[0]
            distances = np.linalg.norm(current - probe_now, axis=1)
            penetrations = probe_radius + particle_radius - distances
            active = penetrations > 0.0
            geometric_contact_samples += int(np.count_nonzero(active))
            if bool(np.any(active)):
                maximum_geometric_penetration = max(
                    maximum_geometric_penetration,
                    float(np.max(penetrations[active])),
                )

    final_q = np.asarray(state_0.particle_q.numpy(), dtype=np.float64)
    final_volumes = signed_tet_volumes(final_q, rest_tets_all)
    inverted_peak = max(
        inverted_peak,
        int(np.count_nonzero(final_volumes * rest_volumes <= 0.0)),
    )
    peak_volume_error = max(
        peak_volume_error,
        abs(float(np.sum(np.abs(final_volumes))) - rest_total_volume)
        / max(rest_total_volume, 1.0e-12),
    )
    free_mask = constraint_kind_np != 1
    recovery_residual = float(
        np.linalg.norm(final_q[free_mask] - rest_q_np[free_mask], axis=1).max()
    )
    completed = finite_samples
    contact_passed = (
        not args.contact_probe
        or (
            contact_candidates_peak > 0
            and geometric_contact_samples > 0
            and maximum_geometric_penetration > 0.0
        )
    )
    final_state_sha256 = hashlib.sha256(
        np.asarray(final_q, dtype="<f8").tobytes()
    ).hexdigest()
    replay_reference = None
    replay_exact_match = None
    if args.replay_reference is not None:
        reference_path = args.replay_reference.expanduser().resolve()
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        replay_reference = portable_path(reference_path)
        replay_exact_match = (
            reference.get("final_state_sha256") == final_state_sha256
        )
    runtime_passed = (
        completed == requested_steps
        and inverted_peak == 0
        and contact_passed
        and replay_exact_match is not False
    )
    result = {
        "schema": "dr.anmar.needle-ready-tissue-newton-qualification.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "intact_deformation_and_rigid_soft_contact_only",
        "asset": {
            "path": portable_path(usd_path),
            "sha256": hashlib.sha256(usd_path.read_bytes()).hexdigest(),
            "lod": args.lod,
            "instances": args.instances,
            "points_per_instance": int(point_count),
            "tetrahedra_per_instance": int(len(tetrahedra)),
            "total_particles": int(model.particle_count),
            "total_tetrahedra": int(len(rest_tets_all)),
        },
        "runtime": {
            "backend": "nvidia_newton_vbd",
            "device": args.device,
            "newton": getattr(newton, "__version__", None),
            "warp": getattr(wp, "__version__", None),
            "coloring_s": coloring_s,
            "replication_s": replication_s,
            "finalize_s": finalize_s,
            "gpu_memory_total_bytes": int(device_handle.total_memory),
            "gpu_memory_free_initial_bytes": initial_free_memory,
            "gpu_memory_free_minimum_bytes": minimum_free_memory,
            "gpu_memory_delta_peak_bytes": (
                initial_free_memory - minimum_free_memory
            ),
        },
        "solver": {
            "dt_s": DT,
            "steps_requested": requested_steps,
            "substeps": substeps,
            "iterations": max(1, args.iterations),
            "youngs_modulus_pa_seed": YOUNGS_MODULUS_PA,
            "poisson_ratio_seed": POISSON_RATIO,
            "density_kg_m3_seed": DENSITY_KG_M3,
            "particle_radius_m": particle_radius,
            "gravity_m_s2": [0.0, 0.0, -9.81],
            "contact_probe": bool(args.contact_probe),
            "volume_audit_interval": args.volume_audit_interval,
        },
        "metrics": {
            "steps_completed": completed,
            "finite_state_fraction": completed / requested_steps,
            "physics_step_ms_p50": percentile(step_ms, 50),
            "physics_step_ms_p95": percentile(step_ms, 95),
            "physics_step_ms_max": max(step_ms, default=0.0),
            "peak_displacement_m": peak_displacement,
            "peak_speed_m_s": peak_speed,
            "peak_global_volume_error_fraction": peak_volume_error,
            "inverted_tetrahedra_peak": inverted_peak,
            "recovery_residual_m": recovery_residual,
            "contact_candidates_peak": contact_candidates_peak,
            "geometric_contact_samples": geometric_contact_samples,
            "maximum_geometric_penetration_m": maximum_geometric_penetration,
        },
        "final_state_sha256": final_state_sha256,
        "deterministic_replay": {
            "reference": replay_reference,
            "exact_final_state_hash_match": replay_exact_match,
        },
        "runtime_gate_passed": runtime_passed,
        "calibration_status": "research_defaults_unvalidated",
        "clinical_validation": False,
        "promotion_boundaries": {
            "geometry_and_intact_runtime": runtime_passed,
            "puncture": False,
            "persistent_tract": False,
            "thread_passage": False,
            "damage_and_tear": False,
            "physical_calibration": False,
            "clinical_use": False,
        },
        "limitations": [
            "Kinematic outer fixtures and retraction nodes are engineering qualification boundary conditions, not anatomy.",
            "The spherical contact probe is not a needle-tip puncture model.",
            "Material values are uncalibrated research seeds.",
            "No topology change, persistent tract, thread passage, damage, or clinical claim is established.",
        ],
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if runtime_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
