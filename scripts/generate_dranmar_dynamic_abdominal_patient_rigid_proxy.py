#!/usr/bin/env python3
"""Regenerate the abdominal patient's rigid perception/planning proxy.

The proxy is authored through native OpenUSD APIs from the canonical component
layers and anatomy manifest. This avoids hand-built USDA delimiter bugs while
keeping the proxy geometry synchronized with the modular patient anatomy.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = (
    REPOSITORY_ROOT
    / "source/extensions/orbit.surgical.assets/data/Props/Patients"
    / "DynamicAbdominalPatient"
)
ANATOMY_MANIFEST = ASSET_ROOT / "anatomy_manifest.json"
OUTPUT_PATH = ASSET_ROOT / "dranmar_dynamic_abdominal_patient_rigid_proxy.usda"

PROXY_ROOT = "DrAnmarDynamicAbdominalPatientRigidProxy"
ASSET_ID = "dranmar-dynamic-abdominal-patient-v1"
BODY_WIDTH = 0.38
BODY_LENGTH = 0.54
BODY_DEPTH = 0.22


def _set_applied_schemas(prim: Any, Sdf: Any, schemas: list[str]) -> None:
    schema_list = Sdf.TokenListOp()
    schema_list.prependedItems = schemas
    if not prim.SetMetadata("apiSchemas", schema_list):
        raise RuntimeError(f"Unable to set applied schemas on {prim.GetPath()}")


def _openusd() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Native OpenUSD Python bindings are required. Install the locked "
            "authoring dependency with: python3 -m pip install usd-core==25.11"
        ) from exc
    return Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def _copy_component_visual(
    *,
    stage: Any,
    component: dict[str, Any],
    copied_materials: set[str],
    modules: tuple[Any, Any, Any, Any, Any, Any],
) -> None:
    Gf, Sdf, _, UsdGeom, _, UsdShade = modules
    source_path = ASSET_ROOT / component["asset_path"]
    source_layer = Sdf.Layer.FindOrOpen(str(source_path))
    if source_layer is None:
        raise RuntimeError(f"OpenUSD could not open component layer: {source_path}")

    root_prim = component["root_prim"]
    component_id = component["id"]
    material_name = component["material"]
    source_visual = Sdf.Path(f"/{root_prim}/Geometry/Visual")
    target_visual = Sdf.Path(f"/{PROXY_ROOT}/Visuals/{component_id}")
    if not source_layer.GetPrimAtPath(source_visual):
        raise RuntimeError(f"{source_path}: missing {source_visual}")
    if not Sdf.CopySpec(
        source_layer,
        source_visual,
        stage.GetRootLayer(),
        target_visual,
    ):
        raise RuntimeError(f"Unable to copy {source_visual} from {source_path}")

    if material_name not in copied_materials:
        source_material = Sdf.Path(f"/{root_prim}/Looks/{material_name}")
        target_material = Sdf.Path(f"/{PROXY_ROOT}/Looks/{material_name}")
        if not source_layer.GetPrimAtPath(source_material):
            raise RuntimeError(f"{source_path}: missing {source_material}")
        if not Sdf.CopySpec(
            source_layer,
            source_material,
            stage.GetRootLayer(),
            target_material,
        ):
            raise RuntimeError(
                f"Unable to copy {source_material} from {source_path}"
            )
        copied_materials.add(material_name)

    visual_prim = stage.GetPrimAtPath(str(target_visual))
    visual_prim.RemoveProperty("material:binding")
    material = UsdShade.Material.Get(
        stage, f"/{PROXY_ROOT}/Looks/{material_name}"
    )
    UsdShade.MaterialBindingAPI.Apply(visual_prim).Bind(material)

    transformable = UsdGeom.Xformable(visual_prim)
    transformable.ClearXformOpOrder()
    translation = component["translation_m"]
    orientation = component["orientation_wxyz"]
    transformable.AddTranslateOp().Set(Gf.Vec3d(*translation))
    transformable.AddOrientOp().Set(
        Gf.Quatf(float(orientation[0]), Gf.Vec3f(*orientation[1:]))
    )


def _author_proxy(destination: Path) -> None:
    Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade = _openusd()
    manifest = json.loads(ANATOMY_MANIFEST.read_text(encoding="utf-8"))
    components = [
        component
        for component in manifest["components"]
        if component["id"] != "adhesions"
    ]

    stage = Usd.Stage.CreateNew(str(destination))
    if stage is None:
        raise RuntimeError(f"OpenUSD could not create layer: {destination}")
    stage.GetRootLayer().documentation = (
        "Rigid perception and scene-composition proxy for the DrAnmar dynamic "
        "abdominal patient. Regenerated from canonical modular anatomy."
    )
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetMetadata("kilogramsPerUnit", 1.0)

    root = UsdGeom.Xform.Define(stage, f"/{PROXY_ROOT}").GetPrim()
    stage.SetDefaultPrim(root)
    Usd.ModelAPI(root).SetKind("component")
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid_body.CreateRigidBodyEnabledAttr(True)
    rigid_body.CreateKinematicEnabledAttr(True)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(65.0)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, -0.03))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1.65, 0.92, 1.88))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0))
    _set_applied_schemas(
        root,
        Sdf,
        ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"],
    )
    root.CreateAttribute(
        "drAnmar:assetId", Sdf.ValueTypeNames.String, custom=True
    ).Set(f"{ASSET_ID}:rigid_proxy")
    root.CreateAttribute(
        "drAnmar:clinicalValidation", Sdf.ValueTypeNames.Bool, custom=True
    ).Set(False)
    root.CreateAttribute(
        "drAnmar:sourceGenerator", Sdf.ValueTypeNames.String, custom=True
    ).Set("scripts/generate_dranmar_dynamic_abdominal_patient_rigid_proxy.py")

    UsdGeom.Scope.Define(stage, f"/{PROXY_ROOT}/Looks")
    UsdGeom.Scope.Define(stage, f"/{PROXY_ROOT}/Visuals")
    UsdGeom.Scope.Define(stage, f"/{PROXY_ROOT}/Collisions")

    copied_materials: set[str] = set()
    modules = (Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade)
    for component in components:
        _copy_component_visual(
            stage=stage,
            component=component,
            copied_materials=copied_materials,
            modules=modules,
        )

    physics_material = UsdShade.Material.Define(
        stage, f"/{PROXY_ROOT}/Looks/TablePhysics"
    )
    physics_api = UsdPhysics.MaterialAPI.Apply(physics_material.GetPrim())
    physics_api.CreateStaticFrictionAttr(0.50)
    physics_api.CreateDynamicFrictionAttr(0.40)
    physics_api.CreateRestitutionAttr(0.01)

    collider = UsdGeom.Cube.Define(
        stage, f"/{PROXY_ROOT}/Collisions/TorsoCollider"
    )
    collider.CreateSizeAttr(1.0)
    collider.CreatePurposeAttr(UsdGeom.Tokens.guide)
    collider.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    collision_api = UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    collision_api.CreateCollisionEnabledAttr(True)
    collider_xform = UsdGeom.Xformable(collider.GetPrim())
    collider_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.01))
    collider_xform.AddScaleOp().Set(
        Gf.Vec3d(BODY_WIDTH, BODY_LENGTH, BODY_DEPTH)
    )
    UsdShade.MaterialBindingAPI.Apply(collider.GetPrim()).Bind(
        physics_material,
        UsdShade.Tokens.weakerThanDescendants,
        "physics",
    )
    _set_applied_schemas(
        collider.GetPrim(),
        Sdf,
        ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"],
    )

    stage.GetRootLayer().Save()
    # OpenUSD's text exporter emits a second trailing newline. Normalize that
    # harmless formatting artifact so generation stays git-clean.
    destination.write_bytes(destination.read_bytes().rstrip(b"\n") + b"\n")


def regenerate(*, check: bool = False) -> bool:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{OUTPUT_PATH.stem}.",
        suffix=OUTPUT_PATH.suffix,
        dir=OUTPUT_PATH.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    temporary_path.unlink()
    try:
        _author_proxy(temporary_path)
        matches = OUTPUT_PATH.is_file() and filecmp.cmp(
            temporary_path, OUTPUT_PATH, shallow=False
        )
        if check:
            return matches
        os.replace(temporary_path, OUTPUT_PATH)
        return True
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the checked-in proxy differs from regenerated output",
    )
    args = parser.parse_args()
    current = regenerate(check=args.check)
    print(
        json.dumps(
            {
                "asset": str(OUTPUT_PATH.relative_to(REPOSITORY_ROOT)),
                "current": current,
                "mode": "check" if args.check else "write",
            },
            indent=2,
        )
    )
    if not current:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
