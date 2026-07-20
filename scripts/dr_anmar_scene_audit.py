# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect an installed OpenUSD anatomy stage through the Isaac runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Audit one installed Dr.Anmar OpenUSD scene.")
parser.add_argument("--scene", type=Path, required=True, nargs="+")
parser.add_argument("--summary_only", action="store_true")
parser.add_argument("--output", type=Path)
parser.add_argument("--max_default_extent_m", type=float)
parser.add_argument("--max_anatomy_extent_m", type=float)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdUtils


def audit_scene(scene: Path) -> dict[str, object]:
    scene = scene.expanduser().resolve()
    stage = Usd.Stage.Open(str(scene))
    if stage is None:
        raise RuntimeError(f"Unable to open {scene}")
    cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    world_bound = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    default_prim = stage.GetDefaultPrim()
    default_xform = UsdGeom.Xformable(default_prim) if default_prim and default_prim.IsA(UsdGeom.Xformable) else None
    cameras = []
    top_level = []
    default_children = []
    meshes = []
    for prim in stage.Traverse():
        if prim.GetParent() == stage.GetPseudoRoot():
            top_xform = UsdGeom.Xformable(prim) if prim.IsA(UsdGeom.Xform) else None
            top_level.append(
                {
                    "path": str(prim.GetPath()),
                    "type": prim.GetTypeName(),
                    "xform_ops": [op.GetOpName() for op in top_xform.GetOrderedXformOps()] if top_xform else [],
                }
            )
        if default_prim and prim.GetParent() == default_prim:
            child_xform = UsdGeom.Xformable(prim) if prim.IsA(UsdGeom.Xform) else None
            default_children.append(
                {
                    "path": str(prim.GetPath()),
                    "type": prim.GetTypeName(),
                    "xform_ops": [op.GetOpName() for op in child_xform.GetOrderedXformOps()] if child_xform else [],
                }
            )
        if prim.IsA(UsdGeom.Camera):
            cameras.append(str(prim.GetPath()))
        if prim.IsA(UsdGeom.Mesh):
            mesh_range = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            meshes.append(
                {
                    "path": str(prim.GetPath()),
                    "bounds_min": list(mesh_range.GetMin()),
                    "bounds_max": list(mesh_range.GetMax()),
                }
            )
    dependency_layers, dependency_assets, unresolved_paths = UsdUtils.ComputeAllDependencies(str(scene))
    world_extent = [float(world_bound.GetMax()[index] - world_bound.GetMin()[index]) for index in range(3)]
    anatomy_prim = stage.GetPrimAtPath("/DrAnmarDigitalTwin/Anatomy")
    anatomy_bounds = None
    anatomy_extent = None
    if anatomy_prim:
        anatomy_range = cache.ComputeWorldBound(anatomy_prim).ComputeAlignedRange()
        anatomy_bounds = {
            "min": list(anatomy_range.GetMin()),
            "max": list(anatomy_range.GetMax()),
        }
        anatomy_extent = [
            float(anatomy_range.GetMax()[index] - anatomy_range.GetMin()[index])
            for index in range(3)
        ]
    return {
        "scene": str(scene),
        "default_prim": str(default_prim.GetPath()) if default_prim else None,
        "default_xform_ops": [op.GetOpName() for op in default_xform.GetOrderedXformOps()] if default_xform else [],
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "up_axis": UsdGeom.GetStageUpAxis(stage),
        "bounds_min": list(world_bound.GetMin()),
        "bounds_max": list(world_bound.GetMax()),
        "bounds_extent": world_extent,
        "anatomy_bounds": anatomy_bounds,
        "anatomy_extent": anatomy_extent,
        "cameras": cameras,
        "top_level_prims": top_level,
        "default_children": default_children,
        "mesh_count": len(meshes),
        "dependency_layer_count": len(dependency_layers),
        "dependency_asset_count": len(dependency_assets),
        "unresolved_paths": [str(path) for path in unresolved_paths],
        "meshes": [] if args.summary_only else meshes,
    }


def main() -> None:
    results = [audit_scene(scene) for scene in args.scene]
    payload = results[0] if len(results) == 1 else results
    serialized = json.dumps(payload, indent=2)
    print(serialized, flush=True)
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    failures = []
    for result in results:
        if args.max_default_extent_m and max(result["bounds_extent"]) > args.max_default_extent_m:
            failures.append(f"{result['scene']}: default extent exceeds {args.max_default_extent_m} m")
        anatomy_extent = result.get("anatomy_extent")
        if args.max_anatomy_extent_m and anatomy_extent and max(anatomy_extent) > args.max_anatomy_extent_m:
            failures.append(f"{result['scene']}: anatomy extent exceeds {args.max_anatomy_extent_m} m")
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
