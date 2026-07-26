#!/usr/bin/env python3
"""Replace legacy remote MDL bindings with portable UsdPreviewSurface materials."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from pxr import Gf, Sdf, Usd, UsdShade
except ImportError as exc:  # pragma: no cover - exercised by the dependency gate
    raise SystemExit(
        "Native OpenUSD Python bindings are required. Install "
        "scripts/requirements_openusd_validation.txt."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "source/extensions/orbit.surgical.assets/data"
SOURCE_LAYERS = (
    "Props/Surgical_block/block.usd",
    "Props/Surgical_needle/needle.usd",
    "Props/Surgical_needle/needle_sdf.usd",
    "Props/Table/table.usd",
    "Robots/STAR/star.usd",
    "Robots/dVRK/ECM/ecm.usd",
    "Robots/dVRK/PSM/psm.usd",
    "Robots/dVRK/PSM/psm_col.usd",
)


def _preview_parameters(shader: UsdShade.Shader) -> tuple[Gf.Vec3f, float, float, float]:
    prim_name = shader.GetPrim().GetPath().pathString.lower()
    diffuse = shader.GetInput("diffuse_color_constant").Get()
    if diffuse is None:
        diffuse = Gf.Vec3f(0.58, 0.61, 0.66)
    else:
        diffuse = Gf.Vec3f(float(diffuse[0]), float(diffuse[1]), float(diffuse[2]))

    is_steel = "steel" in prim_name or "silver" in prim_name
    metallic = 0.9 if is_steel else 0.0
    roughness = 0.24 if is_steel else 0.46
    opacity = 0.34 if "translucent" in prim_name else 1.0
    return diffuse, metallic, roughness, opacity


def _convert_layer(path: Path) -> int:
    stage = Usd.Stage.Open(str(path.resolve()), Usd.Stage.LoadNone)
    if stage is None:
        raise RuntimeError(f"Native OpenUSD could not open {path}")
    converted = 0
    for prim in list(stage.Traverse()):
        if not prim.IsA(UsdShade.Shader):
            continue
        source_asset = prim.GetAttribute("info:mdl:sourceAsset")
        if not source_asset or not source_asset.HasAuthoredValueOpinion():
            continue
        shader = UsdShade.Shader(prim)
        diffuse, metallic, roughness, opacity = _preview_parameters(shader)
        for prop in list(prim.GetProperties()):
            name = prop.GetName()
            if (
                name.startswith("info:mdl:")
                or name.startswith("inputs:")
                or name.startswith("outputs:")
                or name == "info:id"
            ):
                prim.RemoveProperty(name)
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(diffuse)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)

        material_prim = prim.GetParent()
        if material_prim.IsA(UsdShade.Material):
            for prop in list(material_prim.GetProperties()):
                if prop.GetName().startswith("outputs:mdl:"):
                    material_prim.RemoveProperty(prop.GetName())
            material = UsdShade.Material(material_prim)
            material.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface"
            )
        converted += 1
    if converted:
        if not stage.GetRootLayer().Save():
            raise RuntimeError(f"Could not save converted material layer {path}")
    return converted


def _legacy_materials() -> list[str]:
    findings: list[str] = []
    for relative in SOURCE_LAYERS:
        path = DATA_ROOT / relative
        stage = Usd.Stage.Open(str(path.resolve()), Usd.Stage.LoadNone)
        if stage is None:
            findings.append(f"{relative}: unopenable")
            continue
        for prim in stage.Traverse():
            if not prim.IsA(UsdShade.Shader):
                continue
            source_asset = prim.GetAttribute("info:mdl:sourceAsset")
            if source_asset and source_asset.HasAuthoredValueOpinion():
                findings.append(f"{relative}:{prim.GetPath()}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if any catalog-owned layer still authors a legacy MDL material",
    )
    args = parser.parse_args()
    if not args.check:
        total = 0
        for relative in SOURCE_LAYERS:
            converted = _convert_layer(DATA_ROOT / relative)
            total += converted
            print(f"{relative}: converted {converted} material(s)")
        print(f"Converted {total} legacy materials to UsdPreviewSurface.")
    findings = _legacy_materials()
    if findings:
        for finding in findings:
            print(f"legacy material: {finding}")
        return 1
    print("Portable material check passed: no catalog-owned legacy MDL bindings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
