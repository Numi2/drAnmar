#!/usr/bin/env python3
"""Static package inspection for the DrAnmar dynamic abdominal patient.

This requires native OpenUSD parsing and composition, then inspects package
structure, source conventions, explicit tetrahedral meshes, and GLB/PNG/JSON
payloads. It deliberately does not execute authored physiology outcomes and
does not replace scene-derived Isaac Sim, PhysX, CUDA, sensor, physical-bench,
or clinical evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import py_compile
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = (
    REPOSITORY_ROOT
    / "source/extensions/orbit.surgical.assets/data/Props/Patients"
    / "DynamicAbdominalPatient"
)
RUNTIME_PATH = (
    REPOSITORY_ROOT
    / "source/extensions/orbit.surgical.assets/orbit/surgical/assets"
    / "dynamic_abdominal_patient.py"
)
PROFILE_PATH = (
    REPOSITORY_ROOT
    / "physics_next/dynamic-patient/dranmar-dynamic-abdominal-patient-v1.json"
)
OPENUSD_VALIDATOR_PATH = REPOSITORY_ROOT / "scripts/validate_openusd_layers.py"

FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.strip()


def _native_openusd_checks() -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(
        "dranmar_native_openusd_validation", OPENUSD_VALIDATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load native OpenUSD layer validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.require_openusd_layers()


def _extract_array(text: str, declaration: str, start: int = 0) -> str:
    at = text.index(declaration, start)
    opening = text.index("[", at + len(declaration))
    depth = 0
    for index in range(opening, len(text)):
        char = text[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index]
    raise ValueError(f"Unclosed array after {declaration!r}")


def _parse_vec3(body: str) -> np.ndarray:
    matches = re.findall(rf"\(\s*({FLOAT})\s*,\s*({FLOAT})\s*,\s*({FLOAT})\s*\)", body)
    return np.asarray(
        [[float(a), float(b), float(c)] for a, b, c in matches], dtype=np.float64
    )


def _parse_int4(body: str) -> np.ndarray:
    matches = re.findall(
        r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", body
    )
    return np.asarray(
        [[int(a), int(b), int(c), int(d)] for a, b, c, d in matches], dtype=np.int64
    )


def _validate_tet_asset(
    path: Path, expected_vertices: int, expected_tets: int
) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    start = text.index('def TetMesh "SimulationTetMesh"')
    points = _parse_vec3(_extract_array(text, "point3f[] points =", start))
    tets = _parse_int4(_extract_array(text, "int4[] tetVertexIndices =", start))
    if len(points) != expected_vertices or len(tets) != expected_tets:
        raise AssertionError(
            f"{path.name}: tet counts differ: vertices {len(points)}/{expected_vertices}, "
            f"tets {len(tets)}/{expected_tets}"
        )
    if tets.size and (int(tets.min()) < 0 or int(tets.max()) >= len(points)):
        raise AssertionError(f"{path.name}: tetrahedral index outside point array")
    a = points[tets[:, 0]]
    b = points[tets[:, 1]]
    c = points[tets[:, 2]]
    d = points[tets[:, 3]]
    signed = np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) / 6.0
    abs_volume = np.abs(signed)
    if not np.all(np.isfinite(abs_volume)) or np.any(abs_volume <= 1.0e-16):
        raise AssertionError(
            f"{path.name}: non-positive or non-finite tetrahedral volume"
        )
    return {
        "vertices": int(len(points)),
        "tetrahedra": int(len(tets)),
        "total_volume_m3": float(abs_volume.sum()),
        "minimum_tet_volume_m3": float(abs_volume.min()),
        "maximum_tet_volume_m3": float(abs_volume.max()),
        "orientation_counts": {
            "positive": int(np.count_nonzero(signed > 0.0)),
            "negative": int(np.count_nonzero(signed < 0.0)),
        },
    }


def _validate_laparotomy_wound(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    starts = [
        match.start()
        for match in re.finditer(
            r'def TetMesh "SimulationTetMesh"',
            text,
        )
    ]
    if len(starts) != 10:
        raise AssertionError(
            f"{path.name}: expected 10 wound-margin TetMeshes, got "
            f"{len(starts)}"
        )
    minimum_volume = math.inf
    total_tetrahedra = 0
    for start in starts:
        points = _parse_vec3(
            _extract_array(text, "point3f[] points =", start)
        )
        tets = _parse_int4(
            _extract_array(text, "int4[] tetVertexIndices =", start)
        )
        if len(points) != 350 or len(tets) != 864:
            raise AssertionError(
                f"{path.name}: each wound margin must contain 350 points "
                f"and 864 tetrahedra, got {len(points)} and {len(tets)}"
            )
        a = points[tets[:, 0]]
        b = points[tets[:, 1]]
        c = points[tets[:, 2]]
        d = points[tets[:, 3]]
        signed = (
            np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))
            / 6.0
        )
        if not np.all(np.isfinite(signed)) or np.any(signed <= 1.0e-16):
            raise AssertionError(
                f"{path.name}: wound TetMesh contains a non-positive or "
                "non-finite tetrahedron"
            )
        minimum_volume = min(minimum_volume, float(signed.min()))
        total_tetrahedra += len(tets)
    return {
        "body_count": len(starts),
        "points_per_body": 350,
        "tetrahedra_per_body": 864,
        "total_tetrahedra": total_tetrahedra,
        "minimum_signed_volume_m3": minimum_volume,
    }


def _static_payload_checks() -> dict[str, Any]:
    openusd = _native_openusd_checks()
    usda_files = sorted(ASSET_ROOT.rglob("*.usda"))
    json_files = sorted(ASSET_ROOT.rglob("*.json")) + [PROFILE_PATH]
    glb_files = sorted(ASSET_ROOT.rglob("*.glb"))
    png_files = sorted(ASSET_ROOT.rglob("*.png"))
    python_files = [
        RUNTIME_PATH,
        Path(__file__).resolve(),
        REPOSITORY_ROOT / "scripts/generate_dranmar_laparotomy_wound.py",
        REPOSITORY_ROOT
        / "scripts/generate_dranmar_dynamic_abdominal_patient_rigid_proxy.py",
        REPOSITORY_ROOT / "scripts/refresh_dynamic_patient_asset_manifest.py",
        OPENUSD_VALIDATOR_PATH,
        REPOSITORY_ROOT / "examples/dynamic_abdominal_patient_scene.py",
        REPOSITORY_ROOT / "examples/end_to_end_procedure.py",
    ]

    required_integration_files = [
        ASSET_ROOT / "dranmar_dynamic_abdominal_patient.usda",
        ASSET_ROOT / "dranmar_dynamic_abdominal_patient_rigid_proxy.usda",
        ASSET_ROOT / "dranmar_dynamic_abdominal_patient_operating_scene.usda",
        PROFILE_PATH,
        RUNTIME_PATH,
        REPOSITORY_ROOT / "docs/DYNAMIC_PATIENT_LAPAROTOMY.md",
        REPOSITORY_ROOT / "docs/DYNAMIC_PATIENT_README.md",
        *python_files[1:],
    ]
    missing_integration_files = [
        str(path) for path in required_integration_files if not path.is_file()
    ]
    if missing_integration_files:
        raise AssertionError(
            f"Repository integration is incomplete: {missing_integration_files}"
        )

    source_checks: dict[str, Any] = {}
    missing_references: list[dict[str, str]] = []
    for path in usda_files:
        text = path.read_text(encoding="utf-8")
        if re.search(r"quat[fdh]?\s+[^=]+\s*=\s*\([^,]+,\s*\(", text):
            raise AssertionError(f"{path}: nested quaternion syntax")
        if re.search(r'^\s*over\s+"[^"]+"\s*\{.*\}\s*$', text, flags=re.MULTILINE):
            raise AssertionError(f"{path}: one-line over declaration")
        for reference in re.findall(r"@([^@]+)@", text):
            if "://" in reference:
                continue
            resolved = (path.parent / reference).resolve()
            if not resolved.exists():
                missing_references.append({"source": str(path), "reference": reference})
        source_checks[path.relative_to(REPOSITORY_ROOT).as_posix()] = {
            "bytes": path.stat().st_size,
            "flat_quaternion_count": len(
                re.findall(r"quat[fdh]?\s+[^=]+\s*=\s*\([^()]+\)", text)
            ),
        }
    if missing_references:
        raise AssertionError(
            f"Missing relative USD references: {missing_references[:5]}"
        )

    for path in json_files:
        json.loads(path.read_text(encoding="utf-8"))
    for path in png_files:
        with Image.open(path) as image:
            image.verify()
    glb_metrics: dict[str, Any] = {}
    for path in glb_files:
        scene = trimesh.load(path, force="scene", process=False)
        meshes = list(scene.geometry.values())
        vertices = sum(len(mesh.vertices) for mesh in meshes)
        faces = sum(len(mesh.faces) for mesh in meshes)
        if vertices <= 0 or faces <= 0:
            raise AssertionError(f"{path}: empty GLB")
        for mesh in meshes:
            if not np.all(np.isfinite(np.asarray(mesh.vertices, dtype=float))):
                raise AssertionError(f"{path}: non-finite GLB vertices")
        glb_metrics[path.relative_to(REPOSITORY_ROOT).as_posix()] = {
            "geometry_count": len(meshes),
            "vertices": vertices,
            "triangles": faces,
        }
    for path in python_files:
        py_compile.compile(str(path), doraise=True)

    anatomy = json.loads(
        (ASSET_ROOT / "anatomy_manifest.json").read_text(encoding="utf-8")
    )
    asset_manifest = json.loads(
        (ASSET_ROOT / "asset_manifest.json").read_text(encoding="utf-8")
    )
    manifest_paths = {
        str(entry["path"]) for entry in asset_manifest["files"]
    }
    discovered_asset_paths = {
        "assets/"
        + path.relative_to(
            REPOSITORY_ROOT / "source/extensions/orbit.surgical.assets/data"
        ).as_posix()
        for path in ASSET_ROOT.rglob("*")
        if path.is_file() and path.name != "asset_manifest.json"
    }
    if manifest_paths != discovered_asset_paths:
        raise AssertionError(
            "Dynamic-patient asset manifest file inventory drifted from its "
            "authoritative asset directory"
        )
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    anatomy_summary = {
        "component_count": len(anatomy["components"]),
        "explicit_tet_component_count": sum(
            int(component.get("tetrahedra", 0)) > 0
            for component in anatomy["components"]
        ),
        "explicit_tet_vertices": sum(
            int(component.get("tet_vertices", 0))
            for component in anatomy["components"]
            if int(component.get("tetrahedra", 0)) > 0
        ),
        "explicit_tet_tetrahedra": sum(
            int(component.get("tetrahedra", 0))
            for component in anatomy["components"]
        ),
        "visual_vertex_count": sum(
            int(component.get("visual_vertices", 0))
            for component in anatomy["components"]
        ),
        "visual_triangle_count": sum(
            int(component.get("visual_triangles", 0))
            for component in anatomy["components"]
        ),
    }
    declared_summary = {
        key: int(profile["anatomy"][key])
        for key in anatomy_summary
    }
    if declared_summary != anatomy_summary:
        raise AssertionError(
            "Dynamic-patient profile anatomy totals drifted from the authoritative "
            f"manifest: declared={declared_summary}, derived={anatomy_summary}"
        )
    tet_metrics: dict[str, Any] = {}
    volume_cooking_sources: dict[str, Any] = {}
    deformable_hierarchy_roots: dict[str, bool] = {}
    access_layer_variant_paths: dict[str, bool] = {}
    access_layers = {
        "skin",
        "subcutaneous_fat",
        "fascia",
        "abdominal_wall",
        "peritoneum",
    }
    for component in anatomy["components"]:
        component_path = ASSET_ROOT / component["asset_path"]
        component_text = component_path.read_text(encoding="utf-8")
        mechanics = str(component.get("mechanics", ""))
        expects_hierarchy = mechanics in {
            "volume_deformable",
            "volume_deformable_cooked_from_surface",
            "segmented_volume_deformable",
            "surface_deformable",
            "segmented_surface_deformable",
        }
        if expects_hierarchy:
            hierarchy_ok = 'def Xform "Geometry"' in component_text
            deformable_hierarchy_roots[component["id"]] = hierarchy_ok
            if not hierarchy_ok:
                raise AssertionError(
                    f"{component_path.name}: deformable Geometry root is not an Xform"
                )

        if component["id"] in access_layers:
            hierarchy_targets_geometry = (
                'variantSet "access_state"' in component_text
                and component_text.count('over "Geometry"') >= 2
                and component_text.count('over "Visual"') >= 2
                and component_text.count('over "OpenVisual"') >= 2
            )
            access_layer_variant_paths[component["id"]] = hierarchy_targets_geometry
            if not hierarchy_targets_geometry:
                raise AssertionError(
                    f"{component_path.name}: access_state variants must target "
                    "Geometry/Visual and Geometry/OpenVisual"
                )

        if int(component.get("tetrahedra", 0)) > 0:
            tet_metrics[component["id"]] = _validate_tet_asset(
                component_path,
                int(component["tet_vertices"]),
                int(component["tetrahedra"]),
            )

        if "volume" in mechanics or mechanics == "attached_rigid_or_deformable":
            glb_path = ASSET_ROOT / "glb" / f"{component['id']}.glb"
            scene = trimesh.load(glb_path, force="scene", process=False)
            watertight = bool(scene.geometry) and all(
                mesh.is_watertight for mesh in scene.geometry.values()
            )
            volume_cooking_sources[component["id"]] = {
                "watertight": watertight,
                "geometry_count": len(scene.geometry),
            }
            if not watertight:
                raise AssertionError(
                    f"{glb_path.name}: volume cooking source is not watertight"
                )

    main_text = (ASSET_ROOT / "dranmar_dynamic_abdominal_patient.usda").read_text(
        encoding="utf-8"
    )
    laparotomy_metrics = _validate_laparotomy_wound(
        ASSET_ROOT / "anatomy/dranmar_laparotomy_wound.usda"
    )
    required_variant_tokens = [
        'prepend variantSets = "access_state"',
        'variants = { string access_state = "intact" }',
        'variantSet "access_state"',
        'string access_state = "open"',
    ]
    if not all(token in main_text for token in required_variant_tokens):
        raise AssertionError(
            "Top-level patient access-state variant contract is incomplete"
        )

    package_init = RUNTIME_PATH.with_name("__init__.py").read_text(encoding="utf-8")
    if "from .dynamic_abdominal_patient import *" not in package_init:
        raise AssertionError(
            "Dynamic patient runtime is not exported by the asset package"
        )
    portfolio = json.loads(
        (REPOSITORY_ROOT / "physics_next/dr-anmar-assets.json").read_text(
            encoding="utf-8"
        )
    )
    matching_assets = [
        asset
        for asset in portfolio.get("assets", [])
        if asset.get("id") == "dranmar-dynamic-abdominal-patient-v1"
    ]
    if len(matching_assets) != 1:
        raise AssertionError(
            "DrAnmar portfolio must contain exactly one dynamic-patient entry"
        )

    return {
        "native_openusd": openusd,
        "usda_count": len(usda_files),
        "json_count": len(json_files),
        "glb_count": len(glb_files),
        "png_count": len(png_files),
        "python_count": len(python_files),
        "usd_sources": source_checks,
        "glb_metrics": glb_metrics,
        "anatomy_summary": anatomy_summary,
        "asset_manifest_file_count": len(manifest_paths),
        "tet_metrics": tet_metrics,
        "laparotomy_wound_metrics": laparotomy_metrics,
        "access_layer_variant_paths": access_layer_variant_paths,
        "deformable_hierarchy_roots": deformable_hierarchy_roots,
        "volume_cooking_sources": volume_cooking_sources,
        "relative_references_complete": True,
        "top_level_access_variant_complete": True,
        "repository_registration_complete": True,
    }


def validate(source_parent_revision: str | None = None) -> dict[str, Any]:
    static = _static_payload_checks()
    submodule_root = REPOSITORY_ROOT / "source/extensions/orbit.surgical.assets"
    return {
        "schema": "dr.anmar.dynamic-abdominal-patient-model-sanity.v1",
        "asset_id": "dranmar-dynamic-abdominal-patient-v1",
        "version": "0.1.0",
        "passed": True,
        "passed_scope": "checks_executed_by_this_validator_only",
        "overall_qualified": False,
        "static": static,
        "source_control": {
            "parent_revision": source_parent_revision or _git_revision(REPOSITORY_ROOT),
            "asset_submodule_revision": _git_revision(submodule_root),
        },
        "input_hashes": {
            "asset_payload_tree_sha256": _sha256_tree(ASSET_ROOT),
            "profile_sha256": _sha256_file(PROFILE_PATH),
            "runtime_sha256": _sha256_file(RUNTIME_PATH),
            "validator_sha256": _sha256_file(Path(__file__).resolve()),
            "laparotomy_generator_sha256": _sha256_file(
                REPOSITORY_ROOT / "scripts/generate_dranmar_laparotomy_wound.py"
            ),
            "rigid_proxy_generator_sha256": _sha256_file(
                REPOSITORY_ROOT
                / "scripts/generate_dranmar_dynamic_abdominal_patient_rigid_proxy.py"
            ),
            "asset_manifest_refresher_sha256": _sha256_file(
                REPOSITORY_ROOT
                / "scripts/refresh_dynamic_patient_asset_manifest.py"
            ),
        },
        # Retain this key for report consumers. Native OpenUSD parsing and
        # composition are executed above and therefore are not listed here.
        "not_executed": [
            "Isaac Sim execution",
            "PhysX CUDA deformable cooking",
            "PBD fluid execution",
            "RTX sensor execution",
            "scene-derived physiology and patient coupling",
            "physical bench calibration",
            "clinical validation",
        ],
        "intended_use": "simulation_training",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "physics_next/benchmarks"
            / "dranmar-dynamic-abdominal-patient-validation.json"
        ),
    )
    parser.add_argument(
        "--source-parent-revision",
        help="implementation revision represented by retained evidence",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate with the retained source revision and require byte equality",
    )
    args = parser.parse_args()
    retained: dict[str, Any] | None = None
    source_parent_revision = args.source_parent_revision
    if args.check:
        if not args.output.is_file():
            raise SystemExit(f"Retained report is missing: {args.output}")
        retained = json.loads(args.output.read_text(encoding="utf-8"))
        source_parent_revision = retained["source_control"]["parent_revision"]
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_parent_revision, "HEAD"],
            cwd=REPOSITORY_ROOT,
            timeout=15,
        )
        if ancestor.returncode != 0:
            raise SystemExit(
                "Retained dynamic-patient evidence revision is not an ancestor of HEAD"
            )
    report = validate(source_parent_revision=source_parent_revision)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.check:
        assert retained is not None
        current = args.output.read_text(encoding="utf-8")
        if current != serialized:
            raise SystemExit(
                "Retained dynamic-patient evidence is stale; regenerate it from "
                "the recorded implementation revision"
            )
        print(
            json.dumps(
                {
                    "passed": True,
                    "byte_for_byte_reproducible": True,
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "passed_scope": report["passed_scope"],
                "overall_qualified": report["overall_qualified"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
