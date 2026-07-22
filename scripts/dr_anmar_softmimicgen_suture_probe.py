#!/usr/bin/env python3
"""Qualify NVIDIA's SoftMimicGen FEM strand with a native rigid needle.

This probe never writes deformable points. PhysX owns the strand, needle,
attachment, gravity, contact, and stepping. The probe remains separate from
doctor-facing rooms until the complete promotion gate passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--asset-dir", type=Path, required=True)
parser.add_argument("--needle-usd", type=Path, required=True)
parser.add_argument("--steps", type=int, default=160)
parser.add_argument("--output", type=Path)
parser.add_argument("--gate", choices=("core", "promotion"), default="core")
parser.add_argument(
    "--free-strand-only",
    action="store_true",
    help="Run the unmodified SoftMimicGen strand without the needle boundary.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import carb
import numpy as np
import omni.usd
import torch
from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext
from isaacsim.core.simulation_manager import SimulationManager
from omni.physx import get_physx_interface
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics


ROPE_SHA256 = "a4af501095060bbe4b73781a54f5f28c51af9e5cce2b8c35e8379881eee7ee5e"
RING_SHA256 = "4326018069784689639ec95558e3598bc71c7e55d000243058e86e576cd522f9"
THREAD_PATH = Sdf.Path("/World/Thread/Xform")
NEEDLE_PATH = Sdf.Path("/World/Needle")
RING_PATH = Sdf.Path("/World/Ring")
ATTACHMENT_PATH = Sdf.Path("/World/Thread/NeedleAttachment")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_transform(
    prim,
    *,
    translation: tuple[float, float, float],
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    xform.AddScaleOp().Set(Gf.Vec3d(*scale))


def add_reference(stage, path: Sdf.Path, asset: Path, *, translation, scale=(1.0, 1.0, 1.0)):
    prim = stage.DefinePrim(path, "Xform")
    prim.GetReferences().AddReference(str(asset.resolve()))
    set_transform(prim, translation=translation, scale=scale)
    return prim


def rotation_matrix_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def author_scene(stage, rope: Path, ring: Path, needle: Path, *, attach_needle: bool) -> None:
    add_reference(
        stage,
        Sdf.Path("/World/Thread"),
        rope,
        translation=(0.0, 0.0, 0.085),
        scale=(0.2, 0.2, 0.2),
    )
    # The released needle is a 10 cm curved rigid asset. At its 0.4 training
    # scale, this pose overlaps one physical end with the strand endpoint so
    # PhysX can discover the attachment tetrahedra against real collision.
    needle_prim = add_reference(
        stage,
        NEEDLE_PATH,
        needle,
        translation=(0.076, -0.017, 0.085),
        scale=(0.4, 0.4, 0.4),
    )
    rigid = UsdPhysics.RigidBodyAPI.Apply(needle_prim)
    rigid.CreateRigidBodyEnabledAttr(True)
    mass = UsdPhysics.MassAPI.Apply(needle_prim)
    mass.CreateMassAttr(0.02)
    physx_rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(needle_prim)
    physx_rigid.CreateDisableGravityAttr(True)
    physx_rigid.CreateLinearDampingAttr(2.0)
    physx_rigid.CreateAngularDampingAttr(8.0)

    add_reference(
        stage,
        RING_PATH,
        ring,
        translation=(0.0, 0.0, 0.035),
    )

    # Author the same native schemas used by NVIDIA's PhysX deformable-body
    # attachment demo. The optional UI command that wraps these schemas is not
    # loaded in Isaac Lab's lean headless experience.
    if attach_needle:
        attachment = PhysxSchema.PhysxPhysicsAttachment.Define(stage, ATTACHMENT_PATH)
        attachment.GetActor0Rel().SetTargets([THREAD_PATH])
        attachment.GetActor1Rel().SetTargets([NEEDLE_PATH])
        auto_attachment = PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment.GetPrim())
        # The surgical needle is much thinner than the strand's FEM cell size.
        # PhysX therefore needs a surgical-scale proximity tolerance to select
        # the terminal vertices against the actual needle collision volume.
        auto_attachment.CreateDeformableVertexOverlapOffsetAttr(0.003)
        auto_attachment.CreateCollisionFilteringOffsetAttr(0.004)


def main() -> int:
    rope = args.asset_dir / "Rope.usd"
    ring = args.asset_dir / "Ring.usd"
    required = {rope: ROPE_SHA256, ring: RING_SHA256}
    for path, digest in required.items():
        if not path.is_file() or file_sha256(path) != digest:
            raise RuntimeError(f"Pinned SoftMimicGen asset is absent or changed: {path}")
    if not args.needle_usd.is_file():
        raise RuntimeError(f"Needle USD not found: {args.needle_usd}")

    carb.settings.get_settings().set_bool("/persistent/physics/enableDeformableBeta", False)
    sim = SimulationContext(
        SimulationCfg(
            dt=0.005,
            render_interval=1000,
            device=args.device,
            use_fabric=False,
            # This is a one-strand qualification scene, not a 4096-environment
            # trainer. Tight capacities prevent the probe from reserving the
            # production worker's GPU budget on a shared 4090.
            physx=PhysxCfg(
                gpu_max_rigid_contact_count=2**16,
                gpu_max_rigid_patch_count=2**14,
                gpu_found_lost_pairs_capacity=2**16,
                gpu_found_lost_aggregate_pairs_capacity=2**16,
                gpu_total_aggregate_pairs_capacity=2**16,
                gpu_collision_stack_size=2**24,
                gpu_heap_capacity=2**24,
                gpu_temp_buffer_capacity=2**22,
                gpu_max_soft_body_contacts=2**16,
                gpu_max_particle_contacts=2**14,
            ),
        )
    )
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.Scene.Get(stage, "/physicsScene").GetGravityMagnitudeAttr().Set(9.81)
    author_scene(stage, rope, ring, args.needle_usd, attach_needle=not args.free_strand_only)
    sim.reset()

    physics_view = SimulationManager.get_physics_sim_view()
    soft_view = physics_view.create_soft_body_view(str(THREAD_PATH))
    rigid_view = physics_view.create_rigid_body_view(str(NEEDLE_PATH))
    if soft_view._backend is None or rigid_view._backend is None:
        raise RuntimeError("PhysX failed to create the native thread or needle tensor view")

    # Isaac Lab's official environment resets imported nodal velocities before
    # every episode. A raw USD reference can otherwise retain authored solver
    # velocity state, so mirror that native reset contract exactly once here.
    imported_nodal_velocity = soft_view.get_sim_nodal_velocities().clone()
    all_indices = torch.tensor((0,), dtype=torch.int32, device=args.device)
    soft_view.set_sim_nodal_velocities(
        torch.zeros_like(imported_nodal_velocity), indices=all_indices
    )
    rigid_view.set_velocities(
        torch.zeros((1, 6), dtype=torch.float32, device=args.device), indices=all_indices
    )
    initial_nodes = soft_view.get_sim_nodal_positions().cpu().numpy()[0].astype(np.float64)
    initial_needle = rigid_view.get_transforms().clone()
    initial_needle_position = initial_needle[0, :3].detach().cpu().numpy().astype(np.float64)
    initial_needle_quaternion = initial_needle[0, 3:7].detach().cpu().numpy().astype(np.float64)
    attached_indices = np.flatnonzero(initial_nodes[:, 0] >= initial_nodes[:, 0].max() - 0.004)
    initial_attached_mean = initial_nodes[attached_indices].mean(axis=0)
    force_start = max(20, args.steps // 5)
    force_end = max(force_start + 1, (args.steps * 3) // 5)
    force = torch.tensor(((0.003, 0.0, 0.0),), dtype=torch.float32, device=args.device)
    indices = all_indices

    for step in range(max(1, args.steps)):
        if force_start <= step < force_end:
            rigid_view.apply_forces_and_torques_at_position(
                force,
                None,
                None,
                indices,
                True,
            )
        sim.step(render=False)

    get_physx_interface().update_transformations(True, True, True, False)
    final_nodes = soft_view.get_sim_nodal_positions().cpu().numpy()[0].astype(np.float64)
    final_needle = rigid_view.get_transforms().clone()
    final_needle_position = final_needle[0, :3].detach().cpu().numpy().astype(np.float64)
    final_needle_quaternion = final_needle[0, 3:7].detach().cpu().numpy().astype(np.float64)
    final_attached_mean = final_nodes[attached_indices].mean(axis=0)

    thread_prim = stage.GetPrimAtPath(THREAD_PATH)
    attachment_prim = stage.GetPrimAtPath(ATTACHMENT_PATH)
    ring_collision = stage.GetPrimAtPath("/World/Ring/Torus")
    schemas = sorted(thread_prim.GetAppliedSchemas())
    attachment_schemas = sorted(attachment_prim.GetAppliedSchemas()) if attachment_prim.IsValid() else []
    self_collision = bool(thread_prim.GetAttribute("physxDeformable:selfCollision").Get())
    needle_displacement = float(
        torch.linalg.vector_norm(final_needle[0, :3] - initial_needle[0, :3]).item()
    )
    attached_relative_initial = float(
        np.linalg.norm(initial_attached_mean - initial_needle_position)
    )
    attached_relative_final = float(np.linalg.norm(final_attached_mean - final_needle_position))
    local_attachment = rotation_matrix_xyzw(initial_needle_quaternion).T @ (
        initial_attached_mean - initial_needle_position
    )
    predicted_attachment = (
        final_needle_position + rotation_matrix_xyzw(final_needle_quaternion) @ local_attachment
    )
    attachment_follow_error = float(np.linalg.norm(final_attached_mean - predicted_attachment))
    free_end = initial_nodes[:, 0] <= initial_nodes[:, 0].min() + 0.003
    free_end_drop = float(initial_nodes[free_end, 2].mean() - final_nodes[free_end, 2].mean())
    attachment_follows = bool(
        needle_displacement > 0.003
        and attachment_follow_error < 0.008
        and abs(attached_relative_final - attached_relative_initial) < 0.008
    )
    core_passed = bool(
        "PhysxDeformableBodyAPI" in schemas
        and "PhysxCollisionAPI" in schemas
        and not args.free_strand_only
        and attachment_prim.IsValid()
        and "PhysxAutoAttachmentAPI" in attachment_schemas
        and ring_collision.IsValid()
        and needle_displacement > 0.003
        and free_end_drop > 0.0001
        and attachment_follows
    )
    promotion_checks = {
        "free_strand_gravity": free_end_drop > 0.0001,
        "needle_attachment_follows": attachment_follows,
        "ring_tube_and_center_contact": False,
        "needle_ring_collision": False,
        "strand_self_contact": self_collision,
        "ungrasped_native_dynamics": False,
        "deterministic_reset": False,
        "ui_cannot_write_physics": True,
    }
    promotion_passed = all(promotion_checks.values())
    report = {
        "schema": "dr.anmar.softmimicgen-suture-probe.v1",
        "source_revision": "c9d146ba57358a544167de8ebe946caaac8f6220",
        "physics_authority": "NVIDIA PhysX",
        "physics_dt_s": 0.005,
        "steps": int(args.steps),
        "gate": args.gate,
        "assets": {path.name: file_sha256(path) for path in required},
        "official_strand_native_fem": "PhysxDeformableBodyAPI" in schemas,
        "official_strand_self_collision": self_collision,
        "native_attachment_api": "PhysxAutoAttachmentAPI" in attachment_schemas,
        "scenario": "free_strand" if args.free_strand_only else "needle_attachment",
        "initial_node_bounds_m": {
            "min": [round(float(value), 6) for value in initial_nodes.min(axis=0)],
            "max": [round(float(value), 6) for value in initial_nodes.max(axis=0)],
        },
        "final_node_bounds_m": {
            "min": [round(float(value), 6) for value in final_nodes.min(axis=0)],
            "max": [round(float(value), 6) for value in final_nodes.max(axis=0)],
        },
        "attachment_children": [str(child.GetPath()) for child in attachment_prim.GetChildren()]
        if attachment_prim.IsValid()
        else [],
        "ring_native_collision": ring_collision.IsValid(),
        "needle_displacement_m": round(needle_displacement, 6),
        "free_end_drop_m": round(free_end_drop, 6),
        "attached_relative_initial_m": round(attached_relative_initial, 6),
        "attached_relative_final_m": round(attached_relative_final, 6),
        "attachment_follow_error_m": round(attachment_follow_error, 6),
        "attachment_follows": attachment_follows,
        "authored_deformable_point_writes_after_reset": 0,
        "imported_nodal_velocity_max_mps": round(
            float(imported_nodal_velocity.abs().max().item()), 6
        ),
        "authored_nodal_velocity_reset_writes": 1,
        "core_passed": core_passed,
        "promotion_checks": promotion_checks,
        "promotion_passed": promotion_passed,
        "upstream_limitations": [
            "The released SoftMimicGen task grasps the strand directly and has no needle attachment.",
            "The released Rope.usd has self-collision disabled and does not qualify knot mechanics.",
        ],
    }
    report["passed"] = core_passed if args.gate == "core" else promotion_passed
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
