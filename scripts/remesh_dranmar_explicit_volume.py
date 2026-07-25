#!/usr/bin/env python3
"""Rebuild selected patient TetMeshes from their watertight render surfaces.

This is an offline asset-authoring tool.  It uses TetGen's radius-edge and
dihedral quality controls, rejects inverted or ill-conditioned elements, then
replaces only the ``SimulationTetMesh`` block in the selected anatomy USDA.
The committed USDA files have no runtime dependency on TetGen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import tetgen
import trimesh


ROOT = Path(__file__).resolve().parents[1]
PATIENT_ROOT = (
    ROOT
    / "source/extensions/orbit.surgical.assets/data/Props/Patients"
    / "DynamicAbdominalPatient"
)

COMPONENT_MAX_VOLUME_M3 = {
    "liver": 2.0e-7,
    "bladder": 1.0e-7,
    "stomach": 1.0e-7,
    "pancreas": 2.0e-8,
}
COMPONENT_SOLVER_RADII_M = {
    "liver": (0.115, 0.072, 0.035),
    "bladder": (0.042, 0.037, 0.029),
    "stomach": (0.04759774, 0.07049008, 0.03955537),
}
COMPONENT_SOLVER_CENTERS_M = {
    "stomach": (-0.04416688, -0.00538743, -0.00358315),
}
COMPONENT_VOXEL_ELLIPSOID = {
    "pancreas": {
        "radii_m": (0.072, 0.018, 0.012),
        "pitch_m": 0.003,
    },
}


def _voxel_ellipsoid_surface(
    radii: tuple[float, float, float],
    pitch: float,
) -> trimesh.Trimesh:
    """Build a watertight, near-isotropic boundary for a thin ellipsoid."""

    radii_array = np.asarray(radii, dtype=np.float64)
    counts = np.ceil(2.0 * radii_array / pitch).astype(np.int32)
    counts += counts % 2
    origin = -counts.astype(np.float64) * pitch / 2.0
    occupied: set[tuple[int, int, int]] = set()
    for index in np.ndindex(tuple(int(value) for value in counts)):
        center = origin + (np.asarray(index, dtype=np.float64) + 0.5) * pitch
        if float(np.sum((center / radii_array) ** 2)) <= 1.0:
            occupied.add(index)

    face_specs = (
        ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
        ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
        ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
        ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
        ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
        ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
    )
    vertices: list[np.ndarray] = []
    vertex_indices: dict[tuple[int, int, int], int] = {}
    faces: list[tuple[int, int, int]] = []
    for cell in occupied:
        for direction, corners in face_specs:
            neighbor = tuple(cell[axis] + direction[axis] for axis in range(3))
            if neighbor in occupied:
                continue
            quad: list[int] = []
            for corner in corners:
                key = tuple(cell[axis] + corner[axis] for axis in range(3))
                if key not in vertex_indices:
                    vertex_indices[key] = len(vertices)
                    vertices.append(origin + np.asarray(key) * pitch)
                quad.append(vertex_indices[key])
            faces.extend(
                (
                    (quad[0], quad[1], quad[2]),
                    (quad[0], quad[2], quad[3]),
                )
            )
    surface = trimesh.Trimesh(
        vertices=np.asarray(vertices),
        faces=np.asarray(faces),
        process=False,
    )
    if not surface.is_watertight or not surface.is_winding_consistent:
        raise RuntimeError("Generated voxel ellipsoid is not a closed manifold")
    return surface


def _surface_faces(tetrahedra: np.ndarray) -> np.ndarray:
    counts: Counter[tuple[int, int, int]] = Counter()
    oriented: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for a, b, c, d in tetrahedra.tolist():
        for face in ((a, c, b), (a, b, d), (a, d, c), (b, c, d)):
            key = tuple(sorted(face))
            counts[key] += 1
            oriented[key] = face
    return np.asarray(
        [oriented[key] for key, count in counts.items() if count == 1],
        dtype=np.int32,
    )


def _tet_quality(
    vertices: np.ndarray,
    tetrahedra: np.ndarray,
) -> dict[str, float | int]:
    elements = vertices[tetrahedra]
    bases = np.stack(
        (
            elements[:, 1] - elements[:, 0],
            elements[:, 2] - elements[:, 0],
            elements[:, 3] - elements[:, 0],
        ),
        axis=2,
    )
    signed_volumes = np.linalg.det(bases) / 6.0
    conditions = np.linalg.cond(bases)
    return {
        "point_count": int(len(vertices)),
        "tetrahedron_count": int(len(tetrahedra)),
        "minimum_signed_volume_m3": float(signed_volumes.min()),
        "median_signed_volume_m3": float(np.median(signed_volumes)),
        "maximum_condition_number": float(conditions.max()),
        "condition_number_p99": float(np.quantile(conditions, 0.99)),
    }


def _format_vector(values: np.ndarray) -> str:
    return "(" + ", ".join(f"{float(value):.10g}" for value in values) + ")"


def _tetmesh_block(
    vertices: np.ndarray,
    tetrahedra: np.ndarray,
    surface_faces: np.ndarray,
) -> str:
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    points = ",\n".join(
        f"            {_format_vector(point)}" for point in vertices
    )
    elements = ",\n".join(
        "            (" + ", ".join(str(int(index)) for index in tet) + ")"
        for tet in tetrahedra
    )
    faces = ",\n".join(
        "            (" + ", ".join(str(int(index)) for index in face) + ")"
        for face in surface_faces
    )
    return f'''        def TetMesh "SimulationTetMesh"
        {{
            custom string drAnmar:role = "volume_deformable_simulation_mesh"
            uniform token purpose = "guide"
            token visibility = "invisible"
            float3[] extent = [{_format_vector(minimum)}, {_format_vector(maximum)}]
            point3f[] points = [
{points}
            ]
            int4[] tetVertexIndices = [
{elements}
            ]
            int3[] surfaceFaceVertexIndices = [
{faces}
            ]
        }}'''


def _replace_tetmesh_block(text: str, replacement: str) -> str:
    marker = '        def TetMesh "SimulationTetMesh"'
    if marker not in text:
        geometry_marker = '    def Xform "Geometry"'
        geometry_start = text.index(geometry_marker)
        opening = text.index("{", geometry_start)
        depth = 0
        for index in range(opening, len(text)):
            character = text[index]
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return (
                        text[:index]
                        + "\n"
                        + replacement
                        + "\n"
                        + text[index:]
                    )
        raise ValueError("Unterminated Geometry block")
    start = text.index(marker)
    opening = text.index("{", start)
    depth = 0
    end = None
    for index in range(opening, len(text)):
        character = text[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end is None:
        raise ValueError("Unterminated SimulationTetMesh block")
    return text[:start] + replacement + text[end:]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _refresh_patient_manifests(results: list[dict[str, object]]) -> None:
    anatomy_path = PATIENT_ROOT / "anatomy_manifest.json"
    anatomy = json.loads(anatomy_path.read_text(encoding="utf-8"))
    by_id = {entry["id"]: entry for entry in anatomy["components"]}
    for result in results:
        component = str(result["component"])
        quality = result["quality"]
        by_id[component]["mechanics"] = "volume_deformable"
        by_id[component]["tet_vertices"] = quality["point_count"]
        by_id[component]["tetrahedra"] = quality["tetrahedron_count"]
    anatomy_path.write_text(
        json.dumps(anatomy, indent=2) + "\n",
        encoding="utf-8",
    )

    release_manifest_path = PATIENT_ROOT / "asset_manifest.json"
    release_manifest = json.loads(
        release_manifest_path.read_text(encoding="utf-8")
    )
    changed_relative_paths = {
        f"anatomy/dranmar_{result['component']}.usda"
        for result in results
    }
    changed_relative_paths.add("anatomy_manifest.json")
    prefix = "assets/Props/Patients/DynamicAbdominalPatient/"
    entries = {
        entry["path"][len(prefix) :]: entry
        for entry in release_manifest["files"]
        if entry["path"].startswith(prefix)
    }
    for relative_path in changed_relative_paths:
        source = PATIENT_ROOT / relative_path
        entry = entries[relative_path]
        entry["bytes"] = source.stat().st_size
        entry["sha256"] = _sha256(source)
    release_manifest_path.write_text(
        json.dumps(release_manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def remesh_component(
    component: str,
    *,
    maximum_condition: float,
) -> dict[str, object]:
    if component not in COMPONENT_MAX_VOLUME_M3:
        raise ValueError(f"No remeshing profile for {component}")
    glb_path = PATIENT_ROOT / "glb" / f"{component}.glb"
    usd_path = PATIENT_ROOT / "anatomy" / f"dranmar_{component}.usda"
    scene = trimesh.load(glb_path, force="scene", process=False)
    surface = trimesh.util.concatenate(tuple(scene.geometry.values()))
    if not surface.is_watertight:
        raise ValueError(f"{component} render surface is not watertight")
    if component in COMPONENT_VOXEL_ELLIPSOID:
        profile = COMPONENT_VOXEL_ELLIPSOID[component]
        surface = _voxel_ellipsoid_surface(
            tuple(profile["radii_m"]),
            float(profile["pitch_m"]),
        )
        solver_surface = "watertight_voxel_ellipsoid"
        solver_radii = list(profile["radii_m"])
        solver_center_m = [0.0, 0.0, 0.0]
        solver_pitch_m = float(profile["pitch_m"])
    elif component in COMPONENT_SOLVER_RADII_M:
        # The detailed superellipsoid render surfaces intentionally
        # concentrate vertices at their poles.  Using that topology as a FEM
        # boundary forces sliver tetrahedra.  Keep the established solver
        # extents but use a regular icosphere boundary, then bind the detailed
        # render mesh to it.
        surface = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        surface.vertices *= np.asarray(
            COMPONENT_SOLVER_RADII_M[component],
            dtype=np.float64,
        )
        solver_center = np.asarray(
            COMPONENT_SOLVER_CENTERS_M.get(component, (0.0, 0.0, 0.0)),
            dtype=np.float64,
        )
        surface.vertices += solver_center
        solver_surface = "regular_icosphere_subdivision_2"
        solver_radii = list(COMPONENT_SOLVER_RADII_M[component])
        solver_center_m = solver_center.tolist()
        solver_pitch_m = None
    else:
        solver_surface = "watertight_render_topology"
        solver_radii = None
        solver_center_m = None
        solver_pitch_m = None

    mesher = tetgen.TetGen(
        np.asarray(surface.vertices, dtype=np.float64),
        np.asarray(surface.faces, dtype=np.int32),
    )
    vertices, tetrahedra, *_ = mesher.tetrahedralize(
        order=1,
        quality=True,
        minratio=1.1,
        mindihedral=10.0,
        maxvolume=COMPONENT_MAX_VOLUME_M3[component],
        steinerleft=100_000,
        verbose=False,
    )
    vertices = np.asarray(vertices, dtype=np.float64)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int32)

    elements = vertices[tetrahedra]
    bases = np.stack(
        (
            elements[:, 1] - elements[:, 0],
            elements[:, 2] - elements[:, 0],
            elements[:, 3] - elements[:, 0],
        ),
        axis=2,
    )
    negative = np.linalg.det(bases) < 0.0
    if np.any(negative):
        first = tetrahedra[negative, 0].copy()
        tetrahedra[negative, 0] = tetrahedra[negative, 1]
        tetrahedra[negative, 1] = first

    quality = _tet_quality(vertices, tetrahedra)
    if quality["minimum_signed_volume_m3"] <= 0.0:
        raise RuntimeError(f"{component} contains non-positive tetrahedra")
    if quality["maximum_condition_number"] > maximum_condition:
        raise RuntimeError(
            f"{component} maximum condition "
            f"{quality['maximum_condition_number']} exceeds "
            f"{maximum_condition}"
        )

    replacement = _tetmesh_block(
        vertices,
        tetrahedra,
        _surface_faces(tetrahedra),
    )
    original = usd_path.read_text(encoding="utf-8")
    updated = _replace_tetmesh_block(original, replacement).replace(
        'drAnmar:mechanics = "volume_deformable_cooked_from_surface"',
        'drAnmar:mechanics = "volume_deformable"',
        1,
    )
    usd_path.write_text(updated, encoding="utf-8")
    return {
        "component": component,
        "source_surface": str(glb_path.relative_to(ROOT)),
        "asset": str(usd_path.relative_to(ROOT)),
        "tetgen": {
            "solver_surface": solver_surface,
            "solver_surface_faces": int(len(surface.faces)),
            "solver_radii_m": solver_radii,
            "solver_center_m": solver_center_m,
            "solver_pitch_m": solver_pitch_m,
            "minratio": 1.1,
            "mindihedral_deg": 10.0,
            "maximum_tetrahedron_volume_m3": (
                COMPONENT_MAX_VOLUME_M3[component]
            ),
        },
        "quality": quality,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component",
        action="append",
        choices=tuple(COMPONENT_MAX_VOLUME_M3),
        required=True,
    )
    parser.add_argument("--maximum-condition", type=float, default=100.0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    results = [
        remesh_component(
            component,
            maximum_condition=args.maximum_condition,
        )
        for component in args.component
    ]
    _refresh_patient_manifests(results)
    payload = {
        "schema": "dr.anmar.explicit-volume-remeshing.v1",
        "status": "pass",
        "clinical_validation": False,
        "components": results,
    }
    serialized = json.dumps(payload, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
