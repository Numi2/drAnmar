# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Create dependency-clean OpenUSD geometry layers using the Isaac runtime."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher

from dr_anmar_openusd import (
    ANATOMY_REFERENCE_CENTER_M,
    ANATOMY_REFERENCE_EXTENT_M,
    GEOMETRY_SCHEMA,
    geometry_cache_key,
)


DATA_ROOT = Path(os.environ.get("DR_ANMAR_ROOT", Path.home() / ".local/share/dr-anmar")).expanduser()
REPO_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Sanitize the installed Dr.Anmar room geometry.")
parser.add_argument("--anatomy_root", type=Path, default=DATA_ROOT / "assets/sufia_bc")
parser.add_argument("--output_root", type=Path, default=DATA_ROOT / "scenes/openusd/_geometry")
parser.add_argument(
    "--table",
    type=Path,
    default=REPO_ROOT / "source/extensions/orbit.surgical.assets/data/Props/Table/table.usd",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils


SOURCE_NAMES = {
    "Over_GRP_Room_Additions_merged.usd",
    "Over_GRP_CeilingLamps_merged.usd",
    "models_topo_blender.usdc",
}


def normalize_anatomy(stage: Usd.Stage) -> dict[str, object]:
    """Fit inconsistent source units into the known-good CT anatomy envelope."""

    root = stage.GetDefaultPrim()
    if not root or not root.IsA(UsdGeom.Xform):
        raise RuntimeError("An anatomy layer must have an Xform default prim")
    root_xformable = UsdGeom.Xformable(root)
    existing_ops = root_xformable.GetOrderedXformOps()
    if existing_ops:
        raise RuntimeError(
            f"Anatomy default prim {root.GetPath()} already has transforms: "
            f"{[op.GetOpName() for op in existing_ops]}"
        )
    cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bounds = cache.ComputeWorldBound(root).ComputeAlignedRange()
    minimum = bounds.GetMin()
    maximum = bounds.GetMax()
    extents = [float(maximum[index] - minimum[index]) for index in range(3)]
    source_extent = max(extents)
    if not math.isfinite(source_extent) or source_extent <= 1e-6:
        raise RuntimeError(f"Anatomy layer has invalid bounds: {list(minimum)} to {list(maximum)}")
    source_center = [(float(minimum[index]) + float(maximum[index])) * 0.5 for index in range(3)]
    scale = ANATOMY_REFERENCE_EXTENT_M / source_extent
    translation = [
        ANATOMY_REFERENCE_CENTER_M[index] - source_center[index] * scale
        for index in range(3)
    ]
    normalized_children = 0
    for child in root.GetChildren():
        if not child.IsA(UsdGeom.Xform):
            continue
        child_xformable = UsdGeom.Xformable(child)
        child_ops = child_xformable.GetOrderedXformOps()
        translate_op = child_xformable.AddTranslateOp(opSuffix="drAnmarNormalize")
        scale_op = child_xformable.AddScaleOp(opSuffix="drAnmarNormalize")
        translate_op.Set(Gf.Vec3d(*translation))
        scale_op.Set(Gf.Vec3f(scale, scale, scale))
        # Keep normalization below the default-prim composition boundary.  A
        # caller may author its own scale on the referenced root (as Isaac's
        # USD spawner does); child-level ops remain composed and cannot be
        # masked by that stronger root xformOpOrder.
        child_xformable.SetXformOpOrder([translate_op, scale_op, *child_ops])
        normalized_children += 1
    if not normalized_children:
        raise RuntimeError("An anatomy layer contains no child Xforms to normalize")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    normalized_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    normalized_bounds = normalized_cache.ComputeWorldBound(root).ComputeAlignedRange()
    normalized_extent = max(
        float(normalized_bounds.GetMax()[index] - normalized_bounds.GetMin()[index])
        for index in range(3)
    )
    if not math.isclose(normalized_extent, ANATOMY_REFERENCE_EXTENT_M, rel_tol=1e-5, abs_tol=1e-5):
        raise RuntimeError(
            f"Anatomy normalization produced {normalized_extent} m; "
            f"expected {ANATOMY_REFERENCE_EXTENT_M} m"
        )
    layer_data = dict(stage.GetRootLayer().customLayerData)
    layer_data.update(
        {
            "drAnmarGeometrySchema": GEOMETRY_SCHEMA,
            "drAnmarNormalizationScale": scale,
            "drAnmarSourceExtent": source_extent,
            "drAnmarTargetExtentMeters": ANATOMY_REFERENCE_EXTENT_M,
        }
    )
    stage.GetRootLayer().customLayerData = layer_data
    return {
        "source_bounds_min": [float(value) for value in minimum],
        "source_bounds_max": [float(value) for value in maximum],
        "source_extent": source_extent,
        "normalization_scale": scale,
        "normalization_translation": translation,
        "normalized_child_count": normalized_children,
        "normalized_extent_m": normalized_extent,
        "target_extent_m": ANATOMY_REFERENCE_EXTENT_M,
    }


def sanitize(source: Path, destination: Path) -> dict[str, object]:
    if destination.is_file():
        _, _, unresolved = UsdUtils.ComputeAllDependencies(str(destination))
        if not unresolved:
            return {"source": str(source), "output": str(destination), "cached": True, "unresolved": []}
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.stem}.tmp.usdc")
    temporary.unlink(missing_ok=True)
    source_stage = Usd.Stage.Open(str(source))
    if source_stage is None:
        raise RuntimeError(f"Could not open source OpenUSD geometry: {source}")
    flattened = source_stage.Flatten()
    if not flattened.Export(str(temporary)):
        raise RuntimeError(f"Could not flatten OpenUSD geometry: {source}")
    clean_stage = Usd.Stage.Open(str(temporary))
    if clean_stage is None:
        raise RuntimeError(f"Could not reopen flattened OpenUSD geometry: {temporary}")
    material_paths = []
    for prim in clean_stage.TraverseAll():
        if prim.IsA(UsdShade.Material) or prim.IsA(UsdShade.Shader) or prim.GetTypeName() == "NodeGraph":
            material_paths.append(prim.GetPath())
            continue
        for relationship in list(prim.GetRelationships()):
            if relationship.GetName().startswith("material:binding"):
                prim.RemoveProperty(relationship.GetName())
        for attribute in list(prim.GetAttributes()):
            if attribute.GetTypeName() in (Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray):
                prim.RemoveProperty(attribute.GetName())
    for path in sorted(material_paths, key=lambda value: len(str(value)), reverse=True):
        clean_stage.RemovePrim(path)
    normalization = normalize_anatomy(clean_stage) if source.name == "models_topo_blender.usdc" else None
    clean_stage.GetRootLayer().Save()
    os.replace(temporary, destination)
    _, _, unresolved = UsdUtils.ComputeAllDependencies(str(destination))
    unresolved_strings = [str(path) for path in unresolved]
    if unresolved_strings:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Sanitized geometry still has unresolved dependencies: {unresolved_strings}")
    return {
        "source": str(source),
        "output": str(destination),
        "cached": False,
        "unresolved": [],
        "normalization": normalization,
    }


def main() -> None:
    anatomy_root = args.anatomy_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    sources = sorted({path.resolve() for path in anatomy_root.rglob("*") if path.is_file() and path.name in SOURCE_NAMES})
    table = args.table.expanduser().resolve()
    if not table.is_file():
        raise RuntimeError(f"ORBIT-Surgical table USD does not exist: {table}")
    sources.append(table)
    if not sources:
        raise RuntimeError(f"No installed operating-room USD geometry found under {anatomy_root}")
    results = []
    completed_hashes: set[str] = set()
    for source in sources:
        source_hash = geometry_cache_key(source)
        if source_hash in completed_hashes:
            continue
        completed_hashes.add(source_hash)
        results.append(sanitize(source, output_root / f"{source_hash}.usdc"))
    print(
        json.dumps(
            {
                "schema": GEOMETRY_SCHEMA,
                "source_count": len(sources),
                "unique_geometry_count": len(results),
                "outputs": results,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
