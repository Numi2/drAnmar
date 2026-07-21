#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Author a backend-neutral OpenUSD TetMesh from a canonical NPZ volume."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, Vt


parser = argparse.ArgumentParser(description="Author a Dr.Anmar OpenUSD tetrahedral asset")
parser.add_argument("--mesh", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--asset-id", default="DrAnmarLiverInteractive8mm")
parser.add_argument("--canonical", type=Path)
args_cli = parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    mesh_path = args_cli.mesh.expanduser().resolve()
    with np.load(mesh_path) as archive:
        vertices = np.asarray(archive["vertices_m"], dtype=np.float32)
        tetrahedra = np.asarray(archive["tetrahedra"], dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices_m must have shape (N, 3)")
    if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
        raise ValueError("tetrahedra must have shape (M, 4)")
    if int(tetrahedra.min()) < 0 or int(tetrahedra.max()) >= len(vertices):
        raise ValueError("tetrahedra contain an out-of-range vertex index")

    a, b, c, d = (vertices[tetrahedra[:, index]].astype(np.float64) for index in range(4))
    signed_volume = np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) / 6.0
    if np.any(signed_volume <= 0.0):
        raise ValueError("TetMesh input must use positive right-handed orientation")

    output = args_cli.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp.usda")
    stage = Usd.Stage.CreateNew(str(temporary))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, f"/{args_cli.asset_id}")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey("drAnmarClinicalValidation", False)
    root.GetPrim().SetCustomDataByKey("drAnmarCalibrationStatus", "research_defaults_unvalidated")
    root.GetPrim().SetCustomDataByKey("drAnmarSourceMeshSha256", sha256(mesh_path))

    tet_mesh = UsdGeom.TetMesh.Define(stage, f"/{args_cli.asset_id}/sim_mesh")
    tet_mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices))
    tet_mesh.GetTetVertexIndicesAttr().Set(
        Vt.Vec4iArray([Gf.Vec4i(*(int(value) for value in tet)) for tet in tetrahedra])
    )
    surface_faces = UsdGeom.TetMesh.ComputeSurfaceFaces(tet_mesh, Usd.TimeCode.Default())
    if not surface_faces:
        raise RuntimeError("OpenUSD could not derive the TetMesh surface")
    tet_mesh.GetSurfaceFaceVertexIndicesAttr().Set(surface_faces)
    tet_mesh.GetPrim().SetMetadata("documentation", "Canonical Dr.Anmar simulation mesh; research-only")

    visual = UsdGeom.Mesh.Define(stage, f"/{args_cli.asset_id}/vis_mesh")
    visual.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices))
    visual.GetFaceVertexCountsAttr().Set([3] * len(surface_faces))
    visual.GetFaceVertexIndicesAttr().Set([int(value) for face in surface_faces for value in face])
    visual.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    visual.CreateDisplayColorAttr().Set([Gf.Vec3f(0.48, 0.08, 0.06)])

    stage.GetRootLayer().Save()
    temporary.replace(output)
    report = {
        "schema": "dr.anmar.physics-tet-usd.v1",
        "mesh_npz": str(mesh_path),
        "mesh_npz_sha256": sha256(mesh_path),
        "usd": str(output),
        "usd_sha256": sha256(output),
        "vertices": int(len(vertices)),
        "tetrahedra": int(len(tetrahedra)),
        "surface_triangles": int(len(surface_faces)),
        "meters_per_unit": 1.0,
        "up_axis": "Z",
        "backend_neutral": True,
        "calibration_status": "research_defaults_unvalidated",
        "clinical_validation": False,
    }
    if args_cli.canonical:
        canonical_path = args_cli.canonical.expanduser().resolve()
        canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
        representation = canonical.get("simulation_representation")
        if not isinstance(representation, dict):
            raise ValueError("Canonical asset has no simulation representation")
        if Path(representation.get("path", "")).expanduser().resolve() != mesh_path:
            raise ValueError("Canonical asset points to a different tetrahedral mesh")
        representation["usd_path"] = str(output)
        representation["usd_sha256"] = report["usd_sha256"]
        temporary_canonical = canonical_path.with_suffix(".json.tmp")
        temporary_canonical.write_text(json.dumps(canonical, indent=2) + "\n", encoding="utf-8")
        temporary_canonical.replace(canonical_path)
        report["canonical_asset"] = str(canonical_path)
        report["canonical_asset_sha256"] = sha256(canonical_path)
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
