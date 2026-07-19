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
    cameras = []
    top_level = []
    meshes = []
    for prim in stage.Traverse():
        if prim.GetParent() == stage.GetPseudoRoot():
            top_level.append({"path": str(prim.GetPath()), "type": prim.GetTypeName()})
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
    return {
        "scene": str(scene),
        "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "up_axis": UsdGeom.GetStageUpAxis(stage),
        "bounds_min": list(world_bound.GetMin()),
        "bounds_max": list(world_bound.GetMax()),
        "cameras": cameras,
        "top_level_prims": top_level,
        "mesh_count": len(meshes),
        "dependency_layer_count": len(dependency_layers),
        "dependency_asset_count": len(dependency_assets),
        "unresolved_paths": [str(path) for path in unresolved_paths],
        "meshes": [] if args.summary_only else meshes,
    }


def main() -> None:
    results = [audit_scene(scene) for scene in args.scene]
    print(json.dumps(results[0] if len(results) == 1 else results, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
