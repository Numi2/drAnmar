#!/usr/bin/env python3
"""Headless native-PhysX qualification probe for Dr.Anmar suture mechanics.

This is deliberately separate from the operating room.  It proves that the
thread is a native volume-deformable body, that one end is physically attached
to a kinematic rigid body, and that the remaining nodes respond to gravity.  It
does not author, project, or score a desired strand shape.
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
import omni.kit.commands
import omni.usd
from isaaclab.sim import SimulationCfg, SimulationContext
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt


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
    body = UsdGeom.Xform.Define(stage, body_path)
    body.GetPrim().ApplyAPI("OmniPhysicsDeformableBodyAPI")

    mesh_path = body_path.AppendPath("sim_mesh")
    mesh = UsdGeom.TetMesh.Define(stage, mesh_path)
    points, tetrahedra = beam_tetrahedra()
    mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))
    mesh.GetTetVertexIndicesAttr().Set(Vt.Vec4iArray(tetrahedra))
    mesh.GetPrim().ApplyAPI("OmniPhysicsVolumeDeformableSimAPI")
    mesh.GetPrim().GetAttribute("omniphysics:restShapePoints").Set(Vt.Vec3fArray(points))
    mesh.GetPrim().GetAttribute("omniphysics:restTetVtxIndices").Set(Vt.Vec4iArray(tetrahedra))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh.GetSurfaceFaceVertexIndicesAttr().Set(
        UsdGeom.TetMesh.ComputeSurfaceFaces(mesh, Usd.TimeCode.Default())
    )

    material_path = Sdf.Path("/World/NativeSutureMaterial")
    material_prim = stage.DefinePrim(material_path, "Material")
    material = UsdShade.Material(material_prim)
    material_prim.ApplyAPI("OmniPhysicsDeformableMaterialAPI")
    material_prim.GetAttribute("omniphysics:density").Set(1100.0)
    material_prim.GetAttribute("omniphysics:youngsModulus").Set(1.2e6)
    material_prim.GetAttribute("omniphysics:poissonsRatio").Set(0.35)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
        material,
        UsdShade.Tokens.weakerThanDescendants,
        "physics",
    )

    anchor_path = Sdf.Path("/World/NeedleAnchor")
    anchor = UsdGeom.Cube.Define(stage, anchor_path)
    anchor.CreateSizeAttr(0.006)
    set_translation(anchor.GetPrim(), (0.0025, 0.0, 0.08))
    UsdPhysics.CollisionAPI.Apply(anchor.GetPrim())
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(anchor.GetPrim())
    rigid_api.CreateKinematicEnabledAttr(True)

    attachment_path = Sdf.Path("/World/SutureNeedleAttachment")
    success = omni.kit.commands.execute(
        "CreateAutoDeformableAttachment",
        target_attachment_path=attachment_path,
        attachable0_path=body_path,
        attachable1_path=anchor_path,
    )
    if success is False:
        raise RuntimeError("CreateAutoDeformableAttachment returned false")
    return body, mesh, anchor, attachment_path


def main() -> int:
    sim = SimulationContext(
        SimulationCfg(
            dt=0.005,
            render_interval=4,
            device=args.device,
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

    body, mesh, anchor, attachment_path = create_native_suture(stage)
    initial = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    sim.reset()

    for step in range(max(1, args.steps)):
        if step == max(1, args.steps) // 3:
            set_translation(anchor.GetPrim(), (0.0225, 0.0, 0.095))
        sim.step(render=False)

    final = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    tail = np.arange(0, 4)
    attached = np.arange(len(final) - 4, len(final))
    attachment_prim = stage.GetPrimAtPath(attachment_path)
    native_apis = sorted(body.GetPrim().GetAppliedSchemas())
    sim_apis = sorted(mesh.GetPrim().GetAppliedSchemas())
    report = {
        "schema": "dr.anmar.native-suture-probe.v1",
        "physics_dt_s": 0.005,
        "steps": int(args.steps),
        "native_body_api": "OmniPhysicsDeformableBodyAPI" in native_apis,
        "native_volume_sim_api": "OmniPhysicsVolumeDeformableSimAPI" in sim_apis,
        "collision_api": "PhysicsCollisionAPI" in sim_apis,
        "attachment_prim_valid": bool(attachment_prim and attachment_prim.IsValid()),
        "point_count": int(len(final)),
        "tetrahedron_count": int(len(mesh.GetTetVertexIndicesAttr().Get())),
        "tail_drop_m": round(float(initial[tail, 2].mean() - final[tail, 2].mean()), 6),
        "attached_end_displacement_m": round(
            float(np.linalg.norm(final[attached].mean(axis=0) - initial[attached].mean(axis=0))),
            6,
        ),
        "authored_point_writes_after_reset": 0,
    }
    report["passed"] = bool(
        report["native_body_api"]
        and report["native_volume_sim_api"]
        and report["collision_api"]
        and report["attachment_prim_valid"]
        and report["tail_drop_m"] > 0.0001
        and report["attached_end_displacement_m"] > 0.005
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    simulation_app.close()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
