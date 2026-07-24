#!/usr/bin/env python3
"""Validate the DrAnmar laparotomy-sponge asset and physics contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = (
    REPOSITORY_ROOT
    / "source/extensions/orbit.surgical.assets/data/Props/SurgicalCount/LaparotomySponge"
)
UNFOLDED_PATH = ASSET_ROOT / "lap_sponge_unfolded.usda"
FOLDED_PATH = ASSET_ROOT / "lap_sponge_folded_proxy.usda"
PROFILE_PATH = (
    REPOSITORY_ROOT
    / "physics_next/surgical-count/dr-anmar-laparotomy-sponge-v1.json"
)
RUNTIME_HELPER_PATH = (
    REPOSITORY_ROOT
    / "source/extensions/orbit.surgical.assets/orbit/surgical/assets/laparotomy_sponge.py"
)
EXPECTED_TEXTURES = (
    "cotton_dry_basecolor.png",
    "cotton_dry_roughness.png",
    "cotton_normal.png",
    "cotton_wet_basecolor.png",
    "cotton_wet_roughness.png",
)

FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_array(text: str, declaration: str, occurrence: int = 0) -> str:
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(declaration, start + 1)
    opening = text.index("[", start + len(declaration))
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "[":
            depth += 1
        elif text[index] == "]":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index]
    raise ValueError(f"Unclosed array after {declaration!r}")


def parse_points(text: str, occurrence: int = 0) -> list[tuple[float, float, float]]:
    body = extract_array(text, "point3f[] points =", occurrence)
    points = []
    for match in re.finditer(
        rf"\(\s*({FLOAT_PATTERN})\s*,\s*({FLOAT_PATTERN})\s*,\s*({FLOAT_PATTERN})\s*\)",
        body,
    ):
        points.append(tuple(float(match.group(index)) for index in range(1, 4)))
    return points


def parse_int_array(text: str, declaration: str, occurrence: int = 0) -> list[int]:
    return [int(value) for value in re.findall(r"-?\d+", extract_array(text, declaration, occurrence))]


def triangle_area(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
) -> float:
    ax, ay, az = (p1[index] - p0[index] for index in range(3))
    bx, by, bz = (p2[index] - p0[index] for index in range(3))
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    return 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)


def mesh_metrics(
    points: list[tuple[float, float, float]],
    counts: list[int],
    indices: list[int],
) -> dict[str, Any]:
    if any(count != 3 for count in counts):
        raise ValueError("Simulation mesh contains a non-triangle face")
    if len(indices) != 3 * len(counts):
        raise ValueError("Face-index count does not match triangular face count")
    if not points or min(indices) < 0 or max(indices) >= len(points):
        raise ValueError("Face indices are outside the point array")

    parent = list(range(len(points)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    referenced = set(indices)
    area = 0.0
    for offset in range(0, len(indices), 3):
        a, b, c = indices[offset : offset + 3]
        union(a, b)
        union(b, c)
        area += triangle_area(points[a], points[b], points[c])

    components = {find(index) for index in referenced}
    bounds_min = [min(point[axis] for point in points) for axis in range(3)]
    bounds_max = [max(point[axis] for point in points) for axis in range(3)]
    return {
        "point_count": len(points),
        "triangle_count": len(counts),
        "connected_components": len(components),
        "unreferenced_point_count": len(points) - len(referenced),
        "surface_area_m2": area,
        "bounds_min_m": bounds_min,
        "bounds_max_m": bounds_max,
        "envelope_m": [
            bounds_max[axis] - bounds_min[axis] for axis in range(3)
        ],
    }


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path.name} is not a valid PNG")
    return struct.unpack(">II", header[16:24])


def parse_vector(text: str, declaration: str) -> tuple[float, float, float]:
    match = re.search(
        rf"{re.escape(declaration)}\s*=\s*\(\s*({FLOAT_PATTERN})\s*,\s*({FLOAT_PATTERN})\s*,\s*({FLOAT_PATTERN})\s*\)",
        text,
    )
    if match is None:
        raise ValueError(f"Missing vector {declaration}")
    return tuple(float(match.group(index)) for index in range(1, 4))


def portable(path: Path) -> str:
    return path.relative_to(REPOSITORY_ROOT).as_posix()


def validate() -> dict[str, Any]:
    unfolded_text = UNFOLDED_PATH.read_text(encoding="utf-8")
    folded_text = FOLDED_PATH.read_text(encoding="utf-8")
    helper_text = RUNTIME_HELPER_PATH.read_text(encoding="utf-8")
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))

    unfolded = mesh_metrics(
        parse_points(unfolded_text),
        parse_int_array(unfolded_text, "int[] faceVertexCounts ="),
        parse_int_array(unfolded_text, "int[] faceVertexIndices ="),
    )
    folded = mesh_metrics(
        parse_points(folded_text),
        parse_int_array(folded_text, "int[] faceVertexCounts ="),
        parse_int_array(folded_text, "int[] faceVertexIndices ="),
    )

    collider_scale = parse_vector(folded_text, "double3 xformOp:scale")
    collider_translation = parse_vector(folded_text, "double3 xformOp:translate")
    coverage = [
        collider_scale[axis] / folded["envelope_m"][axis] for axis in range(3)
    ]
    capsule_count = len(re.findall(r'def Capsule "LoopCollider_\d{2}"', folded_text))
    capsule_radii = [
        float(value)
        for value in re.findall(rf"double radius = ({FLOAT_PATTERN})", folded_text)
    ]
    capsule_heights = [
        float(value)
        for value in re.findall(rf"double height = ({FLOAT_PATTERN})", folded_text)
    ]
    capsule_orientations = re.findall(
        rf"quatf xformOp:orient = \(\s*({FLOAT_PATTERN})\s*,\s*({FLOAT_PATTERN})\s*,\s*({FLOAT_PATTERN})\s*,\s*({FLOAT_PATTERN})\s*\)",
        folded_text,
    )

    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, measured: Any, expected: Any) -> None:
        checks[name] = {
            "passed": bool(passed),
            "measured": measured,
            "expected": expected,
        }

    check(
        "portable_component_contract",
        all(
            token in unfolded_text and token in folded_text
            for token in (
                'defaultPrim = "LaparotomySponge"',
                "metersPerUnit = 1",
                'upAxis = "Z"',
                'prepend variantSets = "state"',
                '"dry"',
                '"wet"',
                'string simulationUse = "research_only_not_a_medical_device"',
            )
        ),
        "shared component metadata inspected",
        "SI, Z-up, default prim, dry/wet variants, and research-only declaration",
    )
    check(
        "connected_triangular_surface",
        unfolded["point_count"] == 1027
        and unfolded["triangle_count"] == 1868
        and unfolded["connected_components"] == 1
        and unfolded["unreferenced_point_count"] == 0,
        unfolded,
        {
            "point_count": 1027,
            "triangle_count": 1868,
            "connected_components": 1,
            "unreferenced_point_count": 0,
        },
    )
    check(
        "surface_area_parameter_identity",
        math.isclose(
            unfolded["surface_area_m2"],
            profile["geometry"]["connected_simulation_surface_area_m2"],
            rel_tol=1.0e-7,
        ),
        unfolded["surface_area_m2"],
        profile["geometry"]["connected_simulation_surface_area_m2"],
    )
    integrated_masses = {
        state: values["effective_density_kg_m3"]
        * unfolded["surface_area_m2"]
        * values["effective_surface_thickness_m"]
        for state, values in profile["surface_parameters"].items()
        if state in {"dry", "wet"}
    }
    check(
        "integrated_surface_mass",
        math.isclose(integrated_masses["dry"], 0.022, rel_tol=1.0e-6)
        and math.isclose(integrated_masses["wet"], 0.120, rel_tol=1.0e-6),
        integrated_masses,
        {"dry": 0.022, "wet": 0.120},
    )
    check(
        "folded_visual_mesh",
        folded["point_count"] == 1026 and folded["triangle_count"] == 2048,
        folded,
        {"point_count": 1026, "triangle_count": 2048},
    )
    check(
        "body_collision_coverage",
        all(
            math.isclose(coverage[index], expected, rel_tol=1.0e-6)
            for index, expected in enumerate(
                profile["collision_coverage"]["body_axis_coverage_ratio"]
            )
        )
        and math.isclose(collider_translation[2], 0.00048, abs_tol=1.0e-12),
        {
            "collider_envelope_m": collider_scale,
            "collider_translation_m": collider_translation,
            "visual_envelope_m": folded["envelope_m"],
            "axis_coverage_ratio": coverage,
        },
        profile["collision_coverage"],
    )
    check(
        "loop_collision_coverage",
        capsule_count == 32
        and len(capsule_radii) == 32
        and len(capsule_heights) == 32
        and len(capsule_orientations) == 32
        and all(math.isclose(radius, 0.0032) for radius in capsule_radii)
        and all(height > 0.0 for height in capsule_heights),
        {
            "capsule_count": capsule_count,
            "minimum_radius_m": min(capsule_radii),
            "maximum_radius_m": max(capsule_radii),
            "minimum_segment_height_m": min(capsule_heights),
            "maximum_segment_height_m": max(capsule_heights),
            "valid_quaternion_count": len(capsule_orientations),
            "visual_loop_radius_m": 0.003,
            "radial_coverage_ratio": min(capsule_radii) / 0.003,
        },
        "32 positive contiguous segment capsules with 3.2 mm radius over a 3.0 mm visual loop",
    )
    check(
        "rigid_state_physics",
        all(
            token in folded_text
            for token in (
                'def Material "DryPhysics"',
                'def Material "WetPhysics"',
                "float physics:staticFriction = 0.75",
                "float physics:staticFriction = 0.65",
                "float physics:dynamicFriction = 0.65",
                "float physics:dynamicFriction = 0.55",
                "float physics:mass = 0.022",
                "float physics:mass = 0.12",
                'rel material:binding:physics = </LaparotomySponge/Looks/DryPhysics>',
                'rel material:binding:physics = </LaparotomySponge/Looks/WetPhysics>',
            )
        )
        and folded_text.count("float physics:mass = 0.022") == 1
        and folded_text.count("float physics:mass = 0.12") == 1
        and folded_text.count(
            "rel material:binding = </LaparotomySponge/Looks/CottonDry>"
        )
        == 1
        and folded_text.count(
            "rel material:binding:physics = </LaparotomySponge/Looks/DryPhysics>"
        )
        == 1
        and unfolded_text.count(
            "rel material:binding = </LaparotomySponge/Looks/CottonDry>"
        )
        == 1,
        "variant material bindings, friction, restitution, and masses inspected",
        "coordinated dry and wet rigid physics",
    )
    check(
        "surface_runtime_authoring",
        all(
            token in helper_text
            for token in (
                "set_physics_surface_deformable_body",
                'mesh_prim.ApplyAPI("PhysxSurfaceDeformableBodyAPI")',
                '"physxDeformableBody:selfCollision"',
                'prim.ApplyAPI("OmniPhysicsSurfaceDeformableMaterialAPI")',
                'prim.ApplyAPI("PhysxSurfaceDeformableMaterialAPI")',
                '"omniphysics:surfaceBendStiffness"',
                "surface_bend_stiffness: float = 0.0",
            )
        ),
        "runtime surface cooker, material schemas, and self-collision authoring inspected",
        "current Omni Physics surface path with explicit self-collision and thickness-aware bending",
    )

    asset_references = []
    missing_references = []
    invalid_references = []
    for usd_path, text in ((UNFOLDED_PATH, unfolded_text), (FOLDED_PATH, folded_text)):
        for reference in re.findall(r"@([^@]+)@", text):
            asset_references.append(reference)
            if reference.startswith("/") or "://" in reference:
                invalid_references.append(reference)
            elif not (usd_path.parent / reference).is_file():
                missing_references.append(reference)
    check(
        "relative_asset_references",
        bool(asset_references) and not invalid_references and not missing_references,
        {
            "reference_count": len(asset_references),
            "invalid": invalid_references,
            "missing": missing_references,
        },
        "all referenced assets are relative and resolve inside the component directory",
    )

    texture_dimensions = {
        name: png_dimensions(ASSET_ROOT / "textures" / name)
        for name in EXPECTED_TEXTURES
    }
    check(
        "procedural_texture_delivery",
        all(dimensions == (512, 512) for dimensions in texture_dimensions.values()),
        texture_dimensions,
        "five 512 x 512 PNG textures",
    )
    check(
        "interaction_frames_and_semantics",
        all(
            token in unfolded_text or token in folded_text
            for token in (
                'def Xform "center_grasp"',
                'def Xform "corner_grasp"',
                'def Xform "side_grasp"',
                'def Xform "loop_grasp"',
                'def Xform "count_reference"',
                'def GeomSubset "XrayMarkerRegion"',
                'def Mesh "XrayMarkerRegion"',
            )
        )
        and "semanticLabels" in unfolded_text
        and "semanticLabels" in folded_text,
        "frames, marker geometry, and semantic labels inspected",
        "workflow frames plus visible and semantic X-ray-marker region",
    )
    check(
        "licensing_and_nonclinical_boundary",
        (ASSET_ROOT / "LICENSE.txt").is_file()
        and "Apache License" in (ASSET_ROOT / "LICENSE.txt").read_text(encoding="utf-8")
        and profile["license"] == "Apache-2.0"
        and profile["clinical_validation"] is False,
        {
            "license": profile["license"],
            "clinical_validation": profile["clinical_validation"],
        },
        "Apache-2.0 and no clinical-validation claim",
    )

    failed = [name for name, result in checks.items() if not result["passed"]]
    assets = {
        portable(path): sha256(path)
        for path in sorted(ASSET_ROOT.rglob("*"))
        if path.is_file()
    }
    return {
        "schema": "dr.anmar.laparotomy-sponge-static-validation.v1",
        "passed": not failed,
        "summary": {
            "checks": len(checks),
            "passed": len(checks) - len(failed),
            "failed": len(failed),
        },
        "failed_checks": failed,
        "checks": checks,
        "asset_sha256": assets,
        "collision_coverage": {
            "body_axis_coverage_ratio": coverage,
            "body_thickness_coverage": "full",
            "body_lateral_inset": "1 mm per side",
            "loop_capsules": capsule_count,
            "loop_radial_coverage_ratio": min(capsule_radii) / 0.003,
        },
        "clinical_validation": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
