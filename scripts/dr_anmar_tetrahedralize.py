#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Create a reproducible tetrahedral candidate from an extracted organ surface."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytetwild
from scipy.spatial import cKDTree


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("v "):
            vertices.append([float(value) for value in line.split()[1:4]])
        elif line.startswith("f "):
            face = [int(value.split("/")[0]) - 1 for value in line.split()[1:]]
            if len(face) != 3:
                raise ValueError("The extracted physics surface must already be triangular")
            triangles.append(face)
    if not vertices or not triangles:
        raise ValueError(f"No triangle surface found in {path}")
    return np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int32)


def write_vtk(path: Path, vertices: np.ndarray, tetrahedra: np.ndarray) -> None:
    lines = [
        "# vtk DataFile Version 3.0",
        "Dr.Anmar physics-next tetrahedral organ",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        f"POINTS {len(vertices)} float",
    ]
    lines.extend(f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g}" for point in vertices)
    lines.append(f"CELLS {len(tetrahedra)} {len(tetrahedra) * 5}")
    lines.extend(f"4 {tet[0]} {tet[1]} {tet[2]} {tet[3]}" for tet in tetrahedra)
    lines.append(f"CELL_TYPES {len(tetrahedra)}")
    lines.extend("10" for _ in tetrahedra)
    temporary = path.with_suffix(".vtk.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tetrahedralize a Dr.Anmar organ surface")
    parser.add_argument("--extraction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--edge-length-m", type=float, default=0.008)
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Meshing threads; one is the reproducible default, zero uses every core",
    )
    args = parser.parse_args()

    extraction_path = args.extraction.expanduser().resolve()
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    if extraction.get("schema") != "dr.anmar.physics-surface-extraction.v1":
        raise ValueError("Unsupported surface extraction manifest")
    if not extraction.get("watertight_surface_candidate"):
        raise ValueError("Refusing to tetrahedralize a surface with recorded topology defects")
    surface_path = Path(extraction["surface_obj"]).expanduser().resolve()
    if sha256(surface_path) != extraction.get("surface_sha256"):
        raise ValueError("Surface hash does not match the extraction manifest")
    if not 0.001 <= args.edge_length_m <= 0.03:
        raise ValueError("edge length must be between 1 mm and 30 mm")

    surface_vertices, surface_triangles = read_obj(surface_path)
    vertices, tetrahedra = pytetwild.tetrahedralize(
        surface_vertices,
        surface_triangles,
        edge_length_abs=args.edge_length_m,
        optimize=True,
        simplify=True,
        epsilon=1e-4,
        coarsen=False,
        num_threads=args.threads,
        quiet=True,
    )
    vertices = np.asarray(vertices, dtype=np.float64)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
        raise RuntimeError("pytetwild returned an unexpected tetrahedral mesh")

    a = vertices[tetrahedra[:, 0]]
    b = vertices[tetrahedra[:, 1]]
    c = vertices[tetrahedra[:, 2]]
    d = vertices[tetrahedra[:, 3]]
    signed_volume = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0
    absolute_volume = np.abs(signed_volume)
    zero_volume = absolute_volume <= 1e-15
    if np.any(zero_volume):
        raise RuntimeError(f"Tetrahedralization produced {int(zero_volume.sum())} zero-volume elements")
    negative_orientation = signed_volume < 0.0
    input_negative_orientation_count = int(np.sum(negative_orientation))
    if input_negative_orientation_count:
        # fTetWild and the downstream solvers use opposite conventions.  Make
        # the stored volume orientation explicit and uniform so it cannot be
        # interpreted differently by PhysX FEM and Newton VBD.
        first = tetrahedra[negative_orientation, 0].copy()
        tetrahedra[negative_orientation, 0] = tetrahedra[negative_orientation, 1]
        tetrahedra[negative_orientation, 1] = first
        signed_volume[negative_orientation] *= -1.0

    tet_vertices = vertices[tetrahedra]
    edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    max_edge = np.maximum.reduce(
        [np.linalg.norm(tet_vertices[:, start] - tet_vertices[:, end], axis=1) for start, end in edge_pairs]
    )
    normalized_volume_quality = 6.0 * np.sqrt(2.0) * signed_volume / np.maximum(max_edge**3, 1e-24)

    nearest_distance, nearest_index = cKDTree(vertices).query(surface_vertices, k=1)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    mesh_path = output / "simulation-tetrahedra.npz"
    temporary_mesh = mesh_path.with_suffix(".npz.tmp")
    with temporary_mesh.open("wb") as stream:
        np.savez_compressed(
            stream,
            vertices_m=vertices.astype(np.float32),
            tetrahedra=tetrahedra,
            render_to_sim_nearest=nearest_index.astype(np.int32),
            render_to_sim_distance_m=np.asarray(nearest_distance, dtype=np.float32),
        )
    temporary_mesh.replace(mesh_path)
    vtk_path = output / "simulation-tetrahedra.vtk"
    write_vtk(vtk_path, vertices, tetrahedra)

    material_path = (Path(__file__).resolve().parents[1] / "physics_next/materials/liver-research-default.json").resolve()
    result = {
        "schema": "dr.anmar.physics-tetrahedralization.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_extraction": str(extraction_path),
        "source_surface_sha256": extraction["surface_sha256"],
        "edge_length_m": args.edge_length_m,
        "vertices": int(len(vertices)),
        "tetrahedra": int(len(tetrahedra)),
        "input_negative_orientation_tetrahedra": input_negative_orientation_count,
        "stored_negative_orientation_tetrahedra": int(np.sum(signed_volume < 0.0)),
        "volume_m3": float(absolute_volume.sum()),
        "tetra_volume_m3_min": float(absolute_volume.min()),
        "tetra_volume_m3_median": float(np.median(absolute_volume)),
        "tetra_volume_m3_max": float(absolute_volume.max()),
        "normalized_volume_quality_min": float(normalized_volume_quality.min()),
        "normalized_volume_quality_p01": float(np.percentile(normalized_volume_quality, 1)),
        "normalized_volume_quality_median": float(np.median(normalized_volume_quality)),
        "render_mapping_distance_m_max": float(np.max(nearest_distance)),
        "render_mapping_distance_m_p95": float(np.percentile(nearest_distance, 95)),
        "mesh_npz": str(mesh_path),
        "mesh_npz_sha256": sha256(mesh_path),
        "mesh_vtk": str(vtk_path),
        "mesh_vtk_sha256": sha256(vtk_path),
        "material_profile": str(material_path),
        "material_profile_sha256": sha256(material_path),
        "attachments_status": "pending_anatomical_review",
        "calibration_status": "research_defaults_unvalidated",
        "clinical_validation": False,
    }
    result_path = output / "tetrahedralization.json"
    temporary_result = result_path.with_suffix(".json.tmp")
    temporary_result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary_result.replace(result_path)

    canonical = {
        "schema": "dr.anmar.physics-asset.v1",
        "id": "ct-liver-prostate-bladder_liver_candidate",
        "render_mesh_usd": extraction["source_usd"],
        "collision_mesh_usd": extraction["source_usd"],
        "simulation_representation": {
            "kind": "tetrahedral_fem",
            "path": str(mesh_path),
            "units": "metres-kilograms-seconds"
        },
        "material_regions": [{"id": "parenchyma", "material": str(material_path)}],
        "attachments": [],
        "vascular_graph": None,
        "mapping": {
            "render_to_sim": f"{mesh_path}::render_to_sim_nearest",
            "collision_to_sim": f"{mesh_path}::render_to_sim_nearest"
        },
        "calibration": {
            "status": "research_defaults_unvalidated",
            "dataset_sha256": None
        },
        "promotion_status": "attachments_and_benchmarks_pending",
        "clinical_validation": False
    }
    canonical_path = output / "canonical-asset.json"
    temporary_canonical = canonical_path.with_suffix(".json.tmp")
    temporary_canonical.write_text(json.dumps(canonical, indent=2) + "\n", encoding="utf-8")
    temporary_canonical.replace(canonical_path)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
