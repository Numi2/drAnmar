#!/usr/bin/env python3
"""Headless native-PhysX qualification probe for Dr.Anmar suture mechanics.

This is deliberately separate from the operating room. It proves each native
physics requirement or exits non-zero with measured evidence. It does not
author, project, or score a desired strand shape.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Qualify native PhysX suture behavior")
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import numpy as np
import carb
import omni.usd
import torch
from isaaclab.sim import SimulationCfg, SimulationContext
from isaacsim.core.simulation_manager import SimulationManager
from omni.physx import get_physx_interface, get_physxunittests_interface
from omni.physx.scripts import deformableUtils, physicsUtils
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, Vt


def beam_tetrahedra(
    segments: int = 28,
    length_m: float = 0.12,
    half_width_m: float = 0.00065,
    height_m: float = 0.08,
) -> tuple[list[Gf.Vec3f], list[Gf.Vec4i]]:
    """Create a connected square-prism tetrahedral mesh for a thin suture."""
    points: list[Gf.Vec3f] = []
    for index in range(segments + 1):
        x = -length_m + length_m * index / segments
        for y, z in (
            (-half_width_m, -half_width_m),
            (half_width_m, -half_width_m),
            (half_width_m, half_width_m),
            (-half_width_m, half_width_m),
        ):
            points.append(Gf.Vec3f(x, y, height_m + z))

    tetrahedra: list[Gf.Vec4i] = []
    pattern = (
        (0, 1, 3, 4),
        (1, 2, 3, 6),
        (1, 4, 5, 6),
        (3, 4, 6, 7),
        (1, 3, 4, 6),
    )
    point_array = np.asarray(points, dtype=np.float64)
    for segment in range(segments):
        base = segment * 4
        for local in pattern:
            indices = [base + value for value in local]
            a, b, c, d = point_array[indices]
            signed_volume = float(np.linalg.det(np.stack((b - a, c - a, d - a))) / 6.0)
            if signed_volume < 0.0:
                indices[2], indices[3] = indices[3], indices[2]
            tetrahedra.append(Gf.Vec4i(*indices))
    return points, tetrahedra


def set_translation(prim, translation: tuple[float, float, float]) -> None:
    xformable = UsdGeom.Xformable(prim)
    operations = xformable.GetOrderedXformOps()
    translate = next((op for op in operations if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
    if translate is None:
        translate = xformable.AddTranslateOp()
    translate.Set(Gf.Vec3d(*translation))


def create_native_suture(stage):
    body_path = Sdf.Path("/World/NativeSuture")
    points, tetrahedra = beam_tetrahedra()
    flat_tetrahedra = [index for tet in tetrahedra for index in tet]
    surface_points, surface_indices = deformableUtils.extractTriangleSurfaceFromTetra(
        points,
        flat_tetrahedra,
    )
    mesh = UsdGeom.Mesh.Define(stage, body_path)
    mesh.GetPointsAttr().Set(surface_points)
    mesh.GetFaceVertexCountsAttr().Set([3] * (len(surface_indices) // 3))
    mesh.GetFaceVertexIndicesAttr().Set(surface_indices)
    if not deformableUtils.add_physx_deformable_body(
        stage,
        body_path,
        collision_rest_points=points,
        collision_indices=flat_tetrahedra,
        collision_simplification=False,
        simulation_rest_points=points,
        simulation_indices=flat_tetrahedra,
        solver_position_iteration_count=16,
        vertex_velocity_damping=0.05,
        self_collision=True,
        self_collision_filter_distance=0.0013,
    ):
        raise RuntimeError("Native PhysX deformable-body authoring failed")

    material_path = Sdf.Path("/World/NativeSutureMaterial")
    if not deformableUtils.add_deformable_body_material(
        stage,
        material_path,
        damping_scale=0.05,
        density=1100.0,
        dynamic_friction=0.5,
        elasticity_damping=0.05,
        poissons_ratio=0.35,
        youngs_modulus=1.2e6,
    ):
        raise RuntimeError("Native PhysX deformable material authoring failed")
    physicsUtils.add_physics_material_to_prim(
        stage,
        mesh.GetPrim(),
        material_path,
    )

    anchor_path = Sdf.Path("/World/NeedleAnchor")
    anchor = UsdGeom.Cube.Define(stage, anchor_path)
    anchor.CreateSizeAttr(0.006)
    set_translation(anchor.GetPrim(), (0.0025, 0.0, 0.08))
    UsdPhysics.CollisionAPI.Apply(anchor.GetPrim())
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(anchor.GetPrim())
    rigid_api.CreateKinematicEnabledAttr(True)

    attachment_path = Sdf.Path("/World/SutureNeedleAttachment")
    attachment = PhysxSchema.PhysxPhysicsAttachment.Define(stage, attachment_path)
    attachment.GetActor0Rel().SetTargets([body_path])
    attachment.GetActor1Rel().SetTargets([anchor_path])
    PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment.GetPrim())
    return mesh, anchor, attachment_path, len(tetrahedra)


def main() -> int:
    deformable_setting = "/persistent/physics/enableDeformableBeta"
    carb.settings.get_settings().set_bool(deformable_setting, False)
    sim = SimulationContext(
        SimulationCfg(
            dt=0.005,
            render_interval=4,
            device=args.device,
            use_fabric=False,
        )
    )
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.Scene.Get(stage, "/physicsScene").GetGravityMagnitudeAttr().Set(9.81)

    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.CreateSizeAttr(1.0)
    ground.AddScaleOp().Set(Gf.Vec3f(0.5, 0.5, 0.0025))
    set_translation(ground.GetPrim(), (-0.05, 0.0, -0.0025))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    body, anchor, attachment_path, tetrahedron_count = create_native_suture(stage)
    sim.reset()
    physics_view = SimulationManager.get_physics_sim_view()
    soft_body_view = physics_view.create_soft_body_view("/World/NativeSuture")
    if soft_body_view._backend is None:
        raise RuntimeError("PhysX did not create a native soft-body view for the suture")
    initial = soft_body_view.get_sim_nodal_positions().cpu().numpy()[0].astype(np.float64)
    rigid_body_view = physics_view.create_rigid_body_view("/World/NeedleAnchor")
    if rigid_body_view._backend is None:
        raise RuntimeError("PhysX did not create a native rigid-body view for the needle anchor")
    initial_anchor = rigid_body_view.get_transforms().clone()

    for step in range(max(1, args.steps)):
        if step >= max(1, args.steps) // 3:
            target = rigid_body_view.get_transforms().clone()
            target[0, :3] = torch.tensor((0.0225, 0.0, 0.095), device=target.device)
            rigid_body_view.set_transforms(
                target,
                indices=torch.tensor((0,), dtype=torch.int32, device=target.device),
            )
        # A render/update pass is required for PhysX to publish simulated
        # deformable node positions back to the USD TetMesh attributes.
        sim.step(render=True)

    get_physx_interface().update_transformations(True, True, True, False)
    final = soft_body_view.get_sim_nodal_positions().cpu().numpy()[0].astype(np.float64)
    final_anchor = rigid_body_view.get_transforms().clone()
    tail = np.flatnonzero(initial[:, 0] <= initial[:, 0].min() + 0.0045)
    attached = np.flatnonzero(initial[:, 0] >= initial[:, 0].max() - 0.0045)
    attachment_prim = stage.GetPrimAtPath(attachment_path)
    attachment_children = [str(child.GetTypeName()) for child in attachment_prim.GetChildren()]
    attachment_apis = sorted(attachment_prim.GetAppliedSchemas())
    native_apis = sorted(body.GetPrim().GetAppliedSchemas())
    sim_apis = native_apis
    report = {
        "schema": "dr.anmar.native-suture-probe.v1",
        "physics_dt_s": 0.005,
        "native_deformable_runtime": "PhysxDeformableBodyAPI",
        "steps": int(args.steps),
        "native_body_api": "PhysxDeformableBodyAPI" in native_apis,
        "native_volume_sim_api": bool(
            body.GetPrim().GetAttribute("physxDeformable:simulationIndices").Get()
        ),
        "collision_api": "PhysxCollisionAPI" in sim_apis,
        "attachment_prim_valid": bool(attachment_prim and attachment_prim.IsValid()),
        "native_attachment_api": "PhysxAutoAttachmentAPI" in attachment_apis,
        "native_attachment_children": attachment_children,
        "point_count": int(len(final)),
        "native_soft_body_count": int(soft_body_view.count),
        "tetrahedron_count": int(tetrahedron_count),
        "tail_drop_m": round(float(initial[tail, 2].mean() - final[tail, 2].mean()), 6),
        "attached_end_displacement_m": round(
            float(np.linalg.norm(final[attached].mean(axis=0) - initial[attached].mean(axis=0))),
            6,
        ),
        "needle_anchor_displacement_m": round(
            float(torch.linalg.vector_norm(final_anchor[0, :3] - initial_anchor[0, :3]).item()),
            6,
        ),
        "authored_point_writes_after_reset": 0,
        "physx_statistics": get_physxunittests_interface().get_physics_stats(),
    }
    report["passed"] = bool(
        report["native_body_api"]
        and report["native_volume_sim_api"]
        and report["collision_api"]
        and report["attachment_prim_valid"]
        and report["native_attachment_api"]
        and report["tail_drop_m"] > 0.0001
        and report["attached_end_displacement_m"] > 0.005
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
