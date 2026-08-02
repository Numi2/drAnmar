#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Author the puncture-task needle with one native PhysX surface-FEM suture.

The task deliberately manages exactly one rigid needle body.  This asset keeps
that ABI and replaces the older maximal-coordinate chain with a single
triangulated deformable strand attached at the authored swage frame.

Run this script with the pinned Isaac Lab Python so the installed Omni Physics
schemas, rather than hand-written schema guesses, author the USD layer.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument(
    "--output",
    type=Path,
    default=(
        Path(__file__).resolve().parents[1]
        / "source/extensions/orbit.surgical.assets/data/Props/SurgicalClosure/Needle"
        / "dranmar_needle_thread_fem.usda"
    ),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from omni.physx.scripts import deformableUtils  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402


# The penetration task scales this component uniformly by 1.5.  Dimensions in
# this layer are pre-compensated so the simulated result remains a 180 mm long,
# 0.25 mm diameter 4-0 strand with 0.5 mm axial discretization.
TASK_SCALE = 1.5
THREAD_LENGTH_M = 0.180 / TASK_SCALE
THREAD_DIAMETER_M = 0.00025 / TASK_SCALE
THREAD_WIDTH_M = (math.pi * 0.00025 / 4.0) / TASK_SCALE
THREAD_SEGMENT_M = 0.0005 / TASK_SCALE
THREAD_SEGMENTS = round(THREAD_LENGTH_M / THREAD_SEGMENT_M)
SWAGE_ANCHOR_M = (0.0, 0.00700281749604, 0.0)
SWAGE_ATTACHED_STATIONS = 3


def _centreline(distance_m: float) -> tuple[float, float, float]:
    """Compact, tangent-continuous strand rest pose with exact arc length."""

    lead = 0.012 / TASK_SCALE
    if distance_m <= lead:
        return (-distance_m, SWAGE_ANCHOR_M[1], 0.0)

    remainder = THREAD_LENGTH_M - lead
    angle_span = 1.75 * math.pi
    radius = remainder / angle_span
    angle = math.pi / 2.0 + (distance_m - lead) / radius
    centre_x = -lead
    centre_y = SWAGE_ANCHOR_M[1] - radius
    return (
        centre_x + radius * math.cos(angle),
        centre_y + radius * math.sin(angle),
        0.0,
    )


def _make_strand_mesh(stage: Usd.Stage, path: Sdf.Path) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, path)
    # Keep the mesh in the compound asset's coordinate frame.  The deformable
    # body API terminates rigid-body shape collection, while inheriting this
    # transform is necessary for task-authored position, rotation, and scale
    # to place the strand at the needle's swage before PhysX creates the hard
    # two-way attachment.

    points: list[Gf.Vec3f] = []
    for station in range(THREAD_SEGMENTS + 1):
        centre = _centreline(station * THREAD_SEGMENT_M)
        points.extend(
            (
                Gf.Vec3f(centre[0], centre[1], -0.5 * THREAD_WIDTH_M),
                Gf.Vec3f(centre[0], centre[1], 0.5 * THREAD_WIDTH_M),
            )
        )

    counts: list[int] = []
    indices: list[int] = []
    for station in range(THREAD_SEGMENTS):
        lower = 2 * station
        upper = lower + 2
        counts.extend((3, 3))
        indices.extend((lower, upper, lower + 1, lower + 1, upper, upper + 1))

    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.42, 0.08, 0.55)])
    mesh.CreateExtentAttr(
        UsdGeom.PointBased.ComputeExtent(points)
    )
    if not deformableUtils.set_physics_surface_deformable_body(stage, path):
        raise RuntimeError("failed to author the native surface deformable strand")
    return mesh


def _make_material(stage: Usd.Stage, path: Sdf.Path) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path.AppendChild("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.42, 0.08, 0.55)
    )
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.42)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    # Independent shell stiffnesses preserve the braided profile's strong
    # axial response without making its bending response needle-stiff.
    deformableUtils.add_surface_deformable_material(
        stage,
        path,
        density=1300.0,
        static_friction=0.42,
        dynamic_friction=0.32,
        youngs_modulus=7.6639e9,
        poissons_ratio=0.36,
        surface_thickness=THREAD_DIAMETER_M,
        surface_stretch_stiffness=7.6639e9,
        surface_shear_stiffness=2.8176e9,
        surface_bend_stiffness=2.08e8,
    )
    prim = material.GetPrim()
    prim.ApplyAPI("PhysxDeformableMaterialAPI")
    prim.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(0.08)
    prim.ApplyAPI("PhysxSurfaceDeformableMaterialAPI")
    prim.GetAttribute("physxDeformableMaterial:bendDamping").Set(0.12)
    return material


def _make_needle_grip_material(
    stage: Usd.Stage, path: Sdf.Path
) -> UsdShade.Material:
    """Own the validated high-friction jaw material inside the compound asset."""

    material = UsdShade.Material.Define(stage, path)
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(2.0)
    physics.CreateDynamicFrictionAttr(1.5)
    physics.CreateRestitutionAttr(0.0)
    physx = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx.CreateFrictionCombineModeAttr("max")
    return material


def _author_attachment(
    stage: Usd.Stage,
    mesh: UsdGeom.Mesh,
    asset_root_path: Sdf.Path,
    rigid_path: Sdf.Path,
) -> None:
    print("defining low-level swage attachment", flush=True)
    attachment = stage.DefinePrim(
        asset_root_path.AppendChild("ThreadSwageAttachment"),
        "OmniPhysicsVtxXformAttachment",
    )
    print("low-level swage attachment defined", flush=True)
    attachment.GetAttribute("omniphysics:attachmentEnabled").Set(True)
    attachment.GetRelationship("omniphysics:src0").SetTargets([mesh.GetPath()])
    attachment.GetRelationship("omniphysics:src1").SetTargets([rigid_path])

    attached_indices = list(range(2 * SWAGE_ATTACHED_STATIONS))
    points = mesh.GetPointsAttr().Get()
    attachment.GetAttribute("omniphysics:vtxIndicesSrc0").Set(attached_indices)
    attachment.GetAttribute("omniphysics:localPositionsSrc1").Set(
        [points[index] for index in attached_indices]
    )


def author(output: Path) -> None:
    print(f"authoring {output}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    print("stage created", flush=True)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetMetadata("kilogramsPerUnit", 1.0)

    root_path = Sdf.Path("/DrAnmarNeedle")
    root = UsdGeom.Xform.Define(stage, root_path)
    root.GetPrim().SetAssetInfoByKey("name", "DrAnmarNeedleThreadFEM")
    root.GetPrim().SetAssetInfoByKey("version", "0.1.0")
    root.GetPrim().SetCustomDataByKey(
        "drAnmar:representation", "single_rigid_needle_plus_surface_fem_suture"
    )
    root.GetPrim().SetCustomDataByKey(
        "drAnmar:rigidBodyPath", "NeedleRigid"
    )
    root.GetPrim().SetCustomDataByKey("drAnmar:clinicalValidation", False)
    stage.SetDefaultPrim(root.GetPrim())

    # A neutral asset root is required by Omni Physics: a deformable cannot be
    # a transform-inheriting child of a rigid body.  Both physics actors remain
    # siblings in the same authored coordinate frame, so task placement and
    # scale affect them identically without a reset-xform teleport.
    rigid_path = root_path.AppendChild("NeedleRigid")
    needle_rigid = UsdGeom.Xform.Define(stage, rigid_path)
    needle_rigid.GetPrim().GetReferences().AddReference(
        "./dranmar_needle_entry_proxy.usda", root_path
    )
    # The original rigid-only scene overrode every collider below /Needle with
    # one material.  That would also overwrite the new deformable's material.
    # Move the exact validated jaw friction values into the asset and scope the
    # binding to the rigid needle subtree instead.
    grip_material = _make_needle_grip_material(
        stage, root_path.AppendChild("NeedleGripPhysics")
    )
    if not UsdShade.MaterialBindingAPI.Apply(needle_rigid.GetPrim()).Bind(
        grip_material,
        UsdShade.Tokens.strongerThanDescendants,
        "physics",
    ):
        raise RuntimeError("failed to bind the needle grip material")
    print("needle reference composed", flush=True)

    mesh = _make_strand_mesh(stage, root_path.AppendChild("ThreadFEM"))
    print("surface mesh authored", flush=True)
    # Keep the material beside the rigid and deformable actors under their
    # neutral compound root.  Isaac Lab's manager walks that common root when
    # cloning, so both the body and its bound material remain in the view.
    material = _make_material(stage, root_path.AppendChild("ThreadMaterial"))
    print("surface material authored", flush=True)
    binding_prim = mesh.GetPrim()
    binding_api = UsdShade.MaterialBindingAPI.Apply(binding_prim)
    # Author a schema-owned, physics-purpose binding.  A hand-written custom
    # relationship looks equivalent in USDA text, but Omni Physics does not
    # discover it when Isaac Lab clones the deformable into its tensor view.
    if not binding_api.Bind(
        material,
        UsdShade.Tokens.weakerThanDescendants,
        "physics",
    ):
        raise RuntimeError("failed to bind ThreadMaterial for physics")
    physics_binding = binding_api.GetDirectBindingRel("physics")
    if physics_binding.GetTargets() != [material.GetPath()]:
        raise RuntimeError("ThreadMaterial physics binding target was not authored")
    print("surface material bound", flush=True)

    body = mesh.GetPrim()
    if not body.ApplyAPI("PhysxSurfaceDeformableBodyAPI"):
        raise RuntimeError("failed to apply PhysxSurfaceDeformableBodyAPI")
    print("PhysX surface body API applied", flush=True)
    body.GetAttribute("physxDeformableBody:solverPositionIterationCount").Set(32)
    body.GetAttribute("physxDeformableBody:linearDamping").Set(0.015)
    body.GetAttribute("physxDeformableBody:maxLinearVelocity").Set(0.35)
    body.GetAttribute("physxDeformableBody:settlingDamping").Set(2.0)
    body.GetAttribute("physxDeformableBody:settlingThreshold").Set(0.01)
    body.GetAttribute("physxDeformableBody:sleepThreshold").Set(0.001)
    body.GetAttribute("physxDeformableBody:maxDepenetrationVelocity").Set(0.10)
    body.GetAttribute("physxDeformableBody:selfCollision").Set(True)
    body.GetAttribute("physxDeformableBody:selfCollisionFilterDistance").Set(
        2.0 * THREAD_DIAMETER_M
    )
    body.GetAttribute("physxDeformableBody:enableSpeculativeCCD").Set(True)
    body.GetAttribute("physxDeformableBody:disableGravity").Set(False)
    body.GetAttribute("physxDeformableBody:collisionPairUpdateFrequency").Set(2)
    body.GetAttribute("physxDeformableBody:collisionIterationMultiplier").Set(2)

    _author_attachment(stage, mesh, root_path, rigid_path)
    print("swage attachment authored", flush=True)
    stage.GetRootLayer().documentation = (
        "Dr.Anmar 4-0 needle-thread experiment: one task-compatible rigid needle "
        "and one NVIDIA Omni Physics surface-FEM strand. Simulator engineering "
        "asset; not biomechanical or clinical validation."
    )
    stage.GetRootLayer().Save()


try:
    author(args.output.resolve())
    print(args.output.resolve())
finally:
    simulation_app.close()
