#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Extract a normalized OpenUSD organ surface for physics-next meshing.

This first stage is intentionally lossless with respect to the source points and
face winding.  It reports topology defects instead of silently repairing them;
tetrahedralization and attachment authoring happen only after this evidence is
recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Prepare one Dr.Anmar organ for physics-next")
parser.add_argument("--scene", type=Path, required=True)
parser.add_argument("--prim", default="/root/Liver_topo_blender/Liver_topo_blender")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Gf, Usd, UsdGeom


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def triangulate(counts, indices) -> list[tuple[int, int, int]]:
    triangles: list[tuple[int, int, int]] = []
    cursor = 0
    for count in counts:
        face = [int(value) for value in indices[cursor : cursor + int(count)]]
        cursor += int(count)
        if len(face) < 3:
            continue
        triangles.extend((face[0], face[index], face[index + 1]) for index in range(1, len(face) - 1))
    return triangles


def main() -> None:
    scene = args.scene.expanduser().resolve()
    output = args.output.expanduser().resolve()
    stage = Usd.Stage.Open(str(scene))
    if stage is None:
        raise RuntimeError(f"Could not open anatomy stage: {scene}")
    prim = stage.GetPrimAtPath(args.prim)
    if not prim or not prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"The requested prim is not a mesh: {args.prim}")
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get() or []
    counts = mesh.GetFaceVertexCountsAttr().Get() or []
    indices = mesh.GetFaceVertexIndicesAttr().Get() or []
    triangles = triangulate(counts, indices)
    if not points or not triangles:
        raise RuntimeError("The selected organ mesh has no usable topology")

    transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    world_points = [transform.Transform(Gf.Vec3d(*point)) for point in points]
    edge_counts: Counter[tuple[int, int]] = Counter()
    for triangle in triangles:
        for start, end in zip(triangle, (triangle[1], triangle[2], triangle[0])):
            edge_counts[tuple(sorted((start, end)))] += 1
    boundary_edges = sum(1 for count in edge_counts.values() if count == 1)
    non_manifold_edges = sum(1 for count in edge_counts.values() if count > 2)

    output.mkdir(parents=True, exist_ok=True)
    surface_path = output / "surface.obj"
    surface_lines = [f"# Dr.Anmar physics-next source: {scene}", f"o {prim.GetName()}"]
    surface_lines.extend(f"v {point[0]:.9g} {point[1]:.9g} {point[2]:.9g}" for point in world_points)
    surface_lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in triangles)
    temporary_surface = surface_path.with_suffix(".obj.tmp")
    temporary_surface.write_text("\n".join(surface_lines) + "\n", encoding="utf-8")
    temporary_surface.replace(surface_path)

    minimum = [min(float(point[axis]) for point in world_points) for axis in range(3)]
    maximum = [max(float(point[axis]) for point in world_points) for axis in range(3)]
    result = {
        "schema": "dr.anmar.physics-surface-extraction.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_usd": str(scene),
        "source_sha256": sha256(scene),
        "source_prim": str(prim.GetPath()),
        "surface_obj": str(surface_path),
        "surface_sha256": sha256(surface_path),
        "points": len(world_points),
        "source_faces": len(counts),
        "triangles": len(triangles),
        "bounds_min_m": minimum,
        "bounds_max_m": maximum,
        "extent_m": [maximum[axis] - minimum[axis] for axis in range(3)],
        "boundary_edges": boundary_edges,
        "non_manifold_edges": non_manifold_edges,
        "watertight_surface_candidate": boundary_edges == 0 and non_manifold_edges == 0,
        "next_stage": "tetrahedralize_and_author_render_to_sim_mapping",
        "calibration_status": "research_defaults_unvalidated",
        "clinical_validation": False,
    }
    manifest_path = output / "extraction.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

