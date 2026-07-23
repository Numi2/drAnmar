#!/usr/bin/env python3
"""Deterministically author DrAnmar vessel and vascular-clip OpenUSD assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from dr_anmar_hemostasis_model import (
    DEFAULT_HEMOSTASIS_PROFILE_PATH,
    ClipMesh,
    VesselMesh,
    build_clip_mesh,
    build_vessel_mesh,
    derive_hemostasis,
    load_hemostasis_profile,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_DIRECTORY = REPOSITORY_ROOT / "assets/dr_anmar/hemostasis"
DEFAULT_VESSEL_OUTPUT = DEFAULT_ASSET_DIRECTORY / "DrAnmarVessel.usda"
DEFAULT_TET_OUTPUT = DEFAULT_ASSET_DIRECTORY / "DrAnmarVessel.tet.usda"
DEFAULT_CLIP_OUTPUT = DEFAULT_ASSET_DIRECTORY / "DrAnmarVascularClip.usda"
DEFAULT_REPORT = DEFAULT_ASSET_DIRECTORY / "DrAnmarHemostasis.report.json"
PACKAGE_NAME = "DrAnmar Hemostasis Vessel and Clip"
PACKAGE_ID = "dr-anmar-hemostasis"
PACKAGE_VERSION = "1.0.0"
VESSEL_NAME = "DrAnmar Vessel"
VESSEL_ID = "dr-anmar-vessel"
CLIP_NAME = "DrAnmar Vascular Clip"
CLIP_ID = "dr-anmar-vascular-clip"
VESSEL_ROOT = "DrAnmarVessel"
TET_ROOT = "DrAnmarVesselTet"
CLIP_ROOT = "DrAnmarVascularClip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def usd_float(value: float) -> str:
    return f"{value:.12g}"


def usd_vec(values: tuple[float, float, float]) -> str:
    return "(" + ", ".join(usd_float(value) for value in values) + ")"


def _preview_material(
    *,
    root: str,
    name: str,
    color: tuple[float, float, float],
    roughness: float,
    metallic: float = 0.0,
) -> str:
    return f'''def Material "{name}"
{{
    def Shader "PreviewSurface"
    {{
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor = {usd_vec(color)}
        float inputs:metallic = {usd_float(metallic)}
        float inputs:roughness = {usd_float(roughness)}
        token outputs:surface
    }}
    token outputs:surface.connect = </{root}/Materials/{name}/PreviewSurface.outputs:surface>
}}'''


def vessel_materials_block(root: str, profile: dict[str, Any]) -> str:
    appearance = profile["appearance"]
    definitions = [
        _preview_material(
            root=root,
            name="Outer",
            color=tuple(appearance["outer_color_seed"]),
            roughness=float(appearance["roughness_seed"]),
        ),
        _preview_material(
            root=root,
            name="Inner",
            color=tuple(appearance["intima_color_seed"]),
            roughness=float(appearance["roughness_seed"]) * 0.7,
        ),
        _preview_material(
            root=root,
            name="Inlet",
            color=tuple(appearance["media_color_seed"]),
            roughness=float(appearance["roughness_seed"]),
        ),
        _preview_material(
            root=root,
            name="Outlet",
            color=tuple(appearance["media_color_seed"]),
            roughness=float(appearance["roughness_seed"]),
        ),
    ]
    return (
        '    def Scope "Materials"\n    {\n'
        + "\n\n".join(
            "        " + definition.replace("\n", "\n        ")
            for definition in definitions
        )
        + "\n    }"
    )


def clip_materials_block(root: str, profile: dict[str, Any]) -> str:
    appearance = profile["appearance"]
    definition = _preview_material(
        root=root,
        name="ClipMetal",
        color=tuple(appearance["clip_color_seed"]),
        roughness=0.24,
        metallic=float(appearance["metallic_clip"]),
    )
    return (
        '    def Scope "Materials"\n    {\n'
        + "        "
        + definition.replace("\n", "\n        ")
        + "\n    }"
    )


def mesh_attributes(
    points: tuple[tuple[float, float, float], ...],
    triangles: tuple[tuple[int, int, int], ...],
    extent_min: tuple[float, float, float],
    extent_max: tuple[float, float, float],
    *,
    indentation: int = 8,
) -> str:
    prefix = " " * indentation
    encoded_points = (",\n" + prefix).join(usd_vec(point) for point in points)
    counts = ", ".join("3" for _ in triangles)
    indices = ", ".join(str(index) for triangle in triangles for index in triangle)
    return f"""float3[] extent = [{usd_vec(extent_min)}, {usd_vec(extent_max)}]
int[] faceVertexCounts = [{counts}]
int[] faceVertexIndices = [{indices}]
point3f[] points = [
{prefix}{encoded_points}
]
uniform token subdivisionScheme = "none"
uniform token orientation = "rightHanded\""""


def vessel_subset_blocks(
    vessel: VesselMesh,
    *,
    root: str,
    indentation: int = 8,
) -> str:
    prefix = " " * indentation
    blocks: list[str] = []
    for group, indices in vessel.surface_groups.items():
        material_name = group.title()
        encoded = ", ".join(str(index) for index in indices)
        blocks.append(
            f'''def GeomSubset "{material_name}Faces" (
    prepend apiSchemas = ["MaterialBindingAPI"]
)
{{
    uniform token elementType = "face"
    uniform token familyName = "materialBind"
    int[] indices = [{encoded}]
    rel material:binding = </{root}/Materials/{material_name}>
}}'''
        )
    return "\n\n".join(prefix + block.replace("\n", "\n" + prefix) for block in blocks)


def vessel_custom_data(
    profile: dict[str, Any],
    vessel: VesselMesh,
    *,
    representation: str,
) -> str:
    derived = derive_hemostasis(profile, vessel, build_clip_mesh(profile))
    return f'''customData = {{
        string drAnmarAssetId = "{VESSEL_ID}"
        string drAnmarAssetName = "{VESSEL_NAME}"
        string drAnmarAssetVersion = "{PACKAGE_VERSION}"
        string drAnmarAuthorship = "Independent Dr.Anmar hollow vessel geometry, topology and mechanics contract"
        bool drAnmarClinicalValidation = false
        int drAnmarConnectedComponents = {vessel.connected_components}
        bool drAnmarHasOpenLumen = true
        string drAnmarProfileId = "{profile["id"]}"
        string drAnmarRepresentation = "{representation}"
        int drAnmarSimToRealGapCount = {len(profile["sim_to_real"]["gaps"])}
        int drAnmarSurfaceTriangleCount = {derived.vessel_surface_triangle_count}
        int drAnmarTetrahedronCount = {derived.vessel_tetrahedron_count}
        int drAnmarVertexCount = {derived.vessel_point_count}
        string drAnmarWallTopology = "single_watertight_three_layer_hollow_tube"
        string drAnmarStatus = "{profile["status"]}"
    }}'''


def author_vessel_surface(
    profile: dict[str, Any],
    vessel: VesselMesh,
) -> str:
    return f"""#usda 1.0
(
    defaultPrim = "{VESSEL_ROOT}"
    doc = "{VESSEL_NAME}: hollow three-layer deformable vascular research asset; not clinically validated."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{VESSEL_ROOT}" (
    {vessel_custom_data(profile, vessel, representation="watertight_hollow_surface_for_native_physx_tetrahedral_cooking")}
)
{{
{vessel_materials_block(VESSEL_ROOT, profile)}

    def Mesh "Wall" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        uniform token physics:approximation = "none"
        {mesh_attributes(vessel.points, vessel.surface_triangles, vessel.extent_min, vessel.extent_max)}

{vessel_subset_blocks(vessel, root=VESSEL_ROOT)}
    }}
}}
"""


def author_vessel_tetmesh(
    profile: dict[str, Any],
    vessel: VesselMesh,
) -> str:
    layer_order = ("intima", "media", "adventitia")
    layer_names = ", ".join(f'"{name}"' for name in layer_order)
    layer_ids_by_tetrahedron = [0] * len(vessel.tetrahedra)
    for layer_id, layer_name in enumerate(layer_order):
        for tetrahedron_index in vessel.tetrahedron_groups[layer_name]:
            layer_ids_by_tetrahedron[tetrahedron_index] = layer_id
    layer_ids = ", ".join(str(value) for value in layer_ids_by_tetrahedron)
    points = ",\n            ".join(usd_vec(point) for point in vessel.points)
    tetrahedra = ",\n            ".join(
        "(" + ", ".join(str(index) for index in tetrahedron) + ")"
        for tetrahedron in vessel.tetrahedra
    )
    surface = ",\n            ".join(
        "(" + ", ".join(str(index) for index in face) + ")"
        for face in vessel.surface_triangles
    )
    return f"""#usda 1.0
(
    defaultPrim = "{TET_ROOT}"
    doc = "{VESSEL_NAME}: explicit hollow-wall tetrahedral simulation mesh; not clinically validated."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{TET_ROOT}" (
    {vessel_custom_data(profile, vessel, representation="explicit_openusd_tetmesh_with_layer_ids_and_lumen_surface")}
)
{{
{vessel_materials_block(TET_ROOT, profile)}

    def TetMesh "Simulation"
    {{
        float3[] extent = [{usd_vec(vessel.extent_min)}, {usd_vec(vessel.extent_max)}]
        point3f[] points = [
            {points}
        ]
        int3[] surfaceFaceVertexIndices = [
            {surface}
        ]
        int4[] tetVertexIndices = [
            {tetrahedra}
        ]
        custom uniform token[] drAnmar:tetLayerNames = [{layer_names}]
        custom int[] drAnmar:tetLayerIds = [{layer_ids}]
    }}

    def Mesh "Visual" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        {mesh_attributes(vessel.points, vessel.surface_triangles, vessel.extent_min, vessel.extent_max)}

{vessel_subset_blocks(vessel, root=TET_ROOT)}
    }}
}}
"""


def clip_custom_data(
    profile: dict[str, Any],
    clip: ClipMesh,
) -> str:
    return f'''customData = {{
        string drAnmarAssetId = "{CLIP_ID}"
        string drAnmarAssetName = "{CLIP_NAME}"
        string drAnmarAssetVersion = "{PACKAGE_VERSION}"
        string drAnmarAuthorship = "Independent Dr.Anmar one-piece swept clip geometry, collision and plasticity contract"
        bool drAnmarClinicalValidation = false
        int drAnmarCollisionSegmentCount = {len(clip.centerline) - 1}
        string drAnmarConstruction = "one_piece_u_clip_with_swept_elliptical_section_and_inner_serrations"
        string drAnmarMaterialProxy = "{profile["clip"]["material_proxy"]}"
        string drAnmarPlasticFormingBackend = "{profile["clip"]["plastic_forming_backend_required"]}"
        string drAnmarProfileId = "{profile["id"]}"
        string drAnmarRepresentation = "high_resolution_serrated_mesh_with_centerline_capsule_collision"
        int drAnmarSimToRealGapCount = {len(profile["sim_to_real"]["gaps"])}
        int drAnmarTriangleCount = {len(clip.triangles)}
        int drAnmarVertexCount = {len(clip.points)}
        string drAnmarStatus = "{profile["status"]}"
    }}'''


def clip_collision_blocks(
    profile: dict[str, Any],
    clip: ClipMesh,
) -> str:
    radius = float(profile["clip"]["section_thickness_m"]) / 2.0
    blocks: list[str] = []
    for index, (start, end) in enumerate(zip(clip.centerline, clip.centerline[1:])):
        midpoint = tuple((start[axis] + end[axis]) / 2.0 for axis in range(3))
        delta = tuple(end[axis] - start[axis] for axis in range(3))
        length = math.sqrt(sum(value * value for value in delta))
        angle_degrees = math.degrees(math.atan2(delta[1], delta[0]))
        blocks.append(
            f"""def Capsule "Segment{index:03d}" (
    prepend apiSchemas = ["PhysicsCollisionAPI"]
)
{{
    uniform token axis = "X"
    double height = {usd_float(length)}
    double radius = {usd_float(radius)}
    double3 xformOp:translate = {usd_vec(midpoint)}
    float xformOp:rotateZ = {usd_float(angle_degrees)}
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateZ"]
}}"""
        )
    return "\n\n".join(
        "        " + block.replace("\n", "\n        ") for block in blocks
    )


def author_clip(
    profile: dict[str, Any],
    clip: ClipMesh,
) -> str:
    mass = clip.mass_kg
    return f"""#usda 1.0
(
    defaultPrim = "{CLIP_ROOT}"
    doc = "{CLIP_NAME}: one-piece vascular clip research asset with explicit plastic-forming qualification boundary."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{CLIP_ROOT}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    {clip_custom_data(profile, clip)}
)
{{
    bool physics:kinematicEnabled = false
    float physics:mass = {usd_float(mass)}

{clip_materials_block(CLIP_ROOT, profile)}

    def Mesh "Visual" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        rel material:binding = </{CLIP_ROOT}/Materials/ClipMetal>
        {mesh_attributes(clip.points, clip.triangles, clip.extent_min, clip.extent_max)}
    }}

    def Xform "Collision"
    {{
{clip_collision_blocks(profile, clip)}
    }}
}}
"""


def build_report(
    profile_path: Path,
    profile: dict[str, Any],
    vessel: VesselMesh,
    clip: ClipMesh,
    vessel_path: Path,
    tet_path: Path,
    clip_path: Path,
) -> dict[str, Any]:
    derived = derive_hemostasis(profile, vessel, clip)
    return {
        "schema": "dr.anmar.hemostasis-asset-report.v1",
        "name": PACKAGE_NAME,
        "asset_id": PACKAGE_ID,
        "asset_version": PACKAGE_VERSION,
        "profile": str(profile_path.resolve()),
        "profile_id": profile["id"],
        "vessel_asset": str(vessel_path.resolve()),
        "vessel_asset_sha256": sha256(vessel_path),
        "vessel_tetmesh_asset": str(tet_path.resolve()),
        "vessel_tetmesh_asset_sha256": sha256(tet_path),
        "clip_asset": str(clip_path.resolve()),
        "clip_asset_sha256": sha256(clip_path),
        "vessel_point_count": derived.vessel_point_count,
        "vessel_tetrahedron_count": derived.vessel_tetrahedron_count,
        "vessel_surface_triangle_count": (derived.vessel_surface_triangle_count),
        "vessel_connected_components": vessel.connected_components,
        "vessel_wall_volume_m3": vessel.wall_volume_m3,
        "vessel_wall_mass_kg": derived.vessel_wall_mass_kg,
        "vessel_lumen_volume_ml": derived.vessel_lumen_volume_ml,
        "vessel_minimum_tetra_volume_m3": (vessel.minimum_tetra_volume_m3),
        "vessel_inlet_inner_diameter_m": (derived.inlet_inner_diameter_m),
        "vessel_outlet_inner_diameter_m": (derived.outlet_inner_diameter_m),
        "vessel_attachment_node_count": derived.attachment_node_count,
        "vessel_surface_groups": {
            name: len(indices) for name, indices in vessel.surface_groups.items()
        },
        "vessel_tetrahedron_groups": {
            name: len(indices) for name, indices in vessel.tetrahedron_groups.items()
        },
        "clip_point_count": derived.clip_point_count,
        "clip_triangle_count": derived.clip_triangle_count,
        "clip_centerline_segment_count": (derived.clip_centerline_segment_count),
        "clip_centerline_length_m": clip.centerline_length_m,
        "clip_material_volume_m3": clip.material_volume_m3,
        "clip_mass_kg": derived.clip_mass_kg,
        "stable_capabilities": [
            "vessel_geometry",
            "open_lumen_geometry",
            "wall_layer_identity",
            "rigid_clip_manipulation",
            "contact_geometry",
            "pressure_diameter_reduced_order_model",
            "occlusion_and_retention_reduced_order_model",
        ],
        "gated_capabilities": [
            "plastic_clip_forming",
            "two_way_clip_vessel_closure",
            "pulsatile_flow",
            "pressure_tight_seal",
            "bleeding",
            "rupture",
            "clinical_use",
        ],
        "sim_to_real_gap_count": len(profile["sim_to_real"]["gaps"]),
        "clinical_validation": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_HEMOSTASIS_PROFILE_PATH,
    )
    parser.add_argument(
        "--vessel-output",
        type=Path,
        default=DEFAULT_VESSEL_OUTPUT,
    )
    parser.add_argument(
        "--tet-output",
        type=Path,
        default=DEFAULT_TET_OUTPUT,
    )
    parser.add_argument(
        "--clip-output",
        type=Path,
        default=DEFAULT_CLIP_OUTPUT,
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    profile = load_hemostasis_profile(args.profile)
    vessel = build_vessel_mesh(profile)
    clip = build_clip_mesh(profile)
    for path in (
        args.vessel_output,
        args.tet_output,
        args.clip_output,
        args.report,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.vessel_output.write_text(
        author_vessel_surface(profile, vessel),
        encoding="utf-8",
    )
    args.tet_output.write_text(
        author_vessel_tetmesh(profile, vessel),
        encoding="utf-8",
    )
    args.clip_output.write_text(
        author_clip(profile, clip),
        encoding="utf-8",
    )
    report = build_report(
        args.profile,
        profile,
        vessel,
        clip,
        args.vessel_output,
        args.tet_output,
        args.clip_output,
    )
    args.report.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
