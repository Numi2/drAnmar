#!/usr/bin/env python3
"""Deterministically author DrAnmar Suturable Tissue OpenUSD representations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from dr_anmar_tissue_model import (
    DEFAULT_TISSUE_PROFILE_PATH,
    TissueMesh,
    build_tissue_mesh,
    derive_tissue,
    load_tissue_profile,
    signed_tetra_volume,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "assets/dr_anmar/tissue/DrAnmarSuturableTissue.usda"
DEFAULT_TET_OUTPUT = DEFAULT_OUTPUT.with_name("DrAnmarSuturableTissue.tet.usda")
DEFAULT_LEFT_FLAP_OUTPUT = DEFAULT_OUTPUT.with_name(
    "DrAnmarSuturableTissue.left.usda"
)
DEFAULT_RIGHT_FLAP_OUTPUT = DEFAULT_OUTPUT.with_name(
    "DrAnmarSuturableTissue.right.usda"
)
DEFAULT_LEFT_FLAP_TET_OUTPUT = DEFAULT_OUTPUT.with_name(
    "DrAnmarSuturableTissue.left.tet.usda"
)
DEFAULT_RIGHT_FLAP_TET_OUTPUT = DEFAULT_OUTPUT.with_name(
    "DrAnmarSuturableTissue.right.tet.usda"
)
DEFAULT_REPORT = DEFAULT_OUTPUT.with_suffix(".report.json")
ASSET_NAME = "DrAnmar Suturable Tissue"
ASSET_ID = "dr-anmar-suturable-tissue"
ASSET_VERSION = "1.0.0"
ROOT_PRIM = "DrAnmarSuturableTissue"
TET_ROOT_PRIM = "DrAnmarSuturableTissueTet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def usd_float(value: float) -> str:
    return f"{value:.12g}"


def usd_vec(values: tuple[float, float, float]) -> str:
    return "(" + ", ".join(usd_float(value) for value in values) + ")"


def materials_block(root_prim: str, profile: dict[str, Any]) -> str:
    appearance = profile["appearance"]
    colors = {
        "Surface": tuple(appearance["surface_color_seed"]),
        "Bulk": tuple(appearance["bulk_color_seed"]),
        "Fascia": tuple(appearance["fascia_color_seed"]),
        "Wound": tuple(appearance["wound_color_seed"]),
    }
    blocks: list[str] = []
    for name, color in colors.items():
        roughness = float(appearance["roughness_seed"])
        blocks.append(
            f'''def Material "{name}"
{{
    def Shader "PreviewSurface"
    {{
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor = {usd_vec(color)}
        float inputs:metallic = 0
        float inputs:roughness = {usd_float(roughness)}
        token outputs:surface
    }}
    token outputs:surface.connect = </{root_prim}/Materials/{name}/PreviewSurface.outputs:surface>
}}'''
        )
    return (
        '    def Scope "Materials"\n    {\n'
        + "\n\n".join(
            "        " + block.replace("\n", "\n        ") for block in blocks
        )
        + "\n    }"
    )


def mesh_attributes(mesh: TissueMesh, *, indentation: int = 8) -> str:
    prefix = " " * indentation
    points = (",\n" + prefix).join(usd_vec(point) for point in mesh.points)
    counts = ", ".join("3" for _ in mesh.surface_triangles)
    indices = ", ".join(
        str(index) for triangle in mesh.surface_triangles for index in triangle
    )
    return f"""float3[] extent = [{usd_vec(mesh.extent_min)}, {usd_vec(mesh.extent_max)}]
int[] faceVertexCounts = [{counts}]
int[] faceVertexIndices = [{indices}]
point3f[] points = [
{prefix}{points}
]
uniform token subdivisionScheme = "none"
uniform token orientation = "rightHanded"
uniform token subsetFamily:materialBind:familyType = "partition\""""


def subset_blocks(
    mesh: TissueMesh,
    *,
    root_prim: str,
    indentation: int = 8,
) -> str:
    prefix = " " * indentation
    blocks: list[str] = []
    for group, indices in mesh.surface_groups.items():
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
    rel material:binding = </{root_prim}/Materials/{material_name}>
}}'''
        )
    return "\n\n".join(prefix + block.replace("\n", "\n" + prefix) for block in blocks)


def custom_data(
    profile: dict[str, Any],
    mesh: TissueMesh,
    *,
    representation: str,
) -> str:
    derived = derive_tissue(profile, mesh)
    return f'''customData = {{
        string drAnmarAssetId = "{ASSET_ID}"
        string drAnmarAssetName = "{ASSET_NAME}"
        string drAnmarAssetVersion = "{ASSET_VERSION}"
        string drAnmarAuthorship = "Independent Dr.Anmar procedural geometry, topology, mechanics and calibration contract"
        bool drAnmarClinicalValidation = false
        int drAnmarConnectedComponents = {mesh.connected_components}
        string drAnmarProfileId = "{profile["id"]}"
        string drAnmarRepresentation = "{representation}"
        int drAnmarSimToRealGapCount = {len(profile["sim_to_real"]["gaps"])}
        int drAnmarSurfaceTriangleCount = {derived.surface_triangle_count}
        int drAnmarTetrahedronCount = {derived.tetrahedron_count}
        int drAnmarVertexCount = {derived.point_count}
        string drAnmarWoundTopology = "two_disconnected_watertight_flaps_with_open_incision"
        string drAnmarStatus = "{profile["status"]}"
    }}'''


def author_surface(profile: dict[str, Any], mesh: TissueMesh) -> str:
    root = ROOT_PRIM
    return f"""#usda 1.0
(
    defaultPrim = "{root}"
    doc = "{ASSET_NAME}: two-flap open-incision deformable tissue research asset; not clinically validated."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}" (
    {custom_data(profile, mesh, representation="watertight_surface_for_native_physx_tetrahedral_cooking")}
)
{{
{materials_block(root, profile)}

    def Mesh "Surface" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        uniform token physics:approximation = "none"
        {mesh_attributes(mesh)}

{subset_blocks(mesh, root_prim=root)}
    }}
}}
"""


def component_mesh(mesh: TissueMesh, component: int) -> TissueMesh:
    """Extract one disconnected flap and remap all topology deterministically."""

    if component not in (0, 1):
        raise ValueError("Tissue component must be 0 or 1")
    midpoint = len(mesh.points) // 2
    start = 0 if component == 0 else midpoint
    stop = midpoint if component == 0 else len(mesh.points)
    old_to_new = {
        old_index: old_index - start
        for old_index in range(start, stop)
    }
    points = mesh.points[start:stop]
    kept_tetrahedra = [
        (old_index, tetrahedron)
        for old_index, tetrahedron in enumerate(mesh.tetrahedra)
        if all(index in old_to_new for index in tetrahedron)
    ]
    kept_triangles = [
        (old_index, triangle)
        for old_index, triangle in enumerate(mesh.surface_triangles)
        if all(index in old_to_new for index in triangle)
    ]
    tetrahedron_index_map = {
        old_index: new_index
        for new_index, (old_index, _) in enumerate(kept_tetrahedra)
    }
    triangle_index_map = {
        old_index: new_index
        for new_index, (old_index, _) in enumerate(kept_triangles)
    }
    tetrahedra = tuple(
        tuple(old_to_new[index] for index in tetrahedron)
        for _, tetrahedron in kept_tetrahedra
    )
    surface_triangles = tuple(
        tuple(old_to_new[index] for index in triangle)
        for _, triangle in kept_triangles
    )
    volumes = [
        abs(
            signed_tetra_volume(
                *(points[index] for index in tetrahedron)
            )
        )
        for tetrahedron in tetrahedra
    ]
    return TissueMesh(
        points=points,
        tetrahedra=tetrahedra,
        tetrahedron_groups={
            name: tuple(
                tetrahedron_index_map[index]
                for index in indices
                if index in tetrahedron_index_map
            )
            for name, indices in mesh.tetrahedron_groups.items()
        },
        surface_triangles=surface_triangles,
        surface_groups={
            name: tuple(
                triangle_index_map[index]
                for index in indices
                if index in triangle_index_map
            )
            for name, indices in mesh.surface_groups.items()
        },
        extent_min=tuple(
            min(point[axis] for point in points)
            for axis in range(3)
        ),
        extent_max=tuple(
            max(point[axis] for point in points)
            for axis in range(3)
        ),
        volume_m3=sum(volumes),
        minimum_tetra_volume_m3=min(volumes),
        connected_components=1,
    )


def author_tetmesh(profile: dict[str, Any], mesh: TissueMesh) -> str:
    root = TET_ROOT_PRIM
    layer_order = ("fascia", "bulk", "surface")
    layer_names = ", ".join(f'"{name}"' for name in layer_order)
    layer_ids_by_tetrahedron = [0] * len(mesh.tetrahedra)
    for layer_id, layer_name in enumerate(layer_order):
        for tetrahedron_index in mesh.tetrahedron_groups[layer_name]:
            layer_ids_by_tetrahedron[tetrahedron_index] = layer_id
    layer_ids = ", ".join(str(value) for value in layer_ids_by_tetrahedron)
    points = ",\n            ".join(usd_vec(point) for point in mesh.points)
    tetrahedra = ",\n            ".join(
        "(" + ", ".join(str(index) for index in tet) + ")" for tet in mesh.tetrahedra
    )
    surface = ",\n            ".join(
        "(" + ", ".join(str(index) for index in face) + ")"
        for face in mesh.surface_triangles
    )
    material = profile["intact_tissue"]
    contact = profile["contact"]
    solver = profile["solver"]
    mass_kg = derive_tissue(profile, mesh).mass_kg
    tet_attributes = f'''float3[] extent = [{usd_vec(mesh.extent_min)}, {usd_vec(mesh.extent_max)}]
point3f[] points = [
    {points}
]
int3[] surfaceFaceVertexIndices = [
    {surface}
]
int4[] tetVertexIndices = [
    {tetrahedra}
]'''
    simulation_rest_attributes = f'''point3f[] omniphysics:restShapePoints = [
    {points}
]
int4[] omniphysics:restTetVtxIndices = [
    {tetrahedra}
]'''
    return f"""#usda 1.0
(
    defaultPrim = "{root}"
    doc = "{ASSET_NAME}: explicit backend-neutral tetrahedral simulation mesh; not clinically validated."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}" (
    prepend apiSchemas = ["OmniPhysicsDeformableBodyAPI", "PhysxBaseDeformableBodyAPI", "MaterialBindingAPI"]
    {custom_data(profile, mesh, representation="explicit_openusd_tetmesh_with_surface_visualization")}
)
{{
{materials_block(root, profile)}

    def Material "FEMPhysics" (
        prepend apiSchemas = ["OmniPhysicsBaseMaterialAPI", "OmniPhysicsDeformableMaterialAPI", "PhysxDeformableMaterialAPI"]
    )
    {{
        float omniphysics:density = {usd_float(float(material["density_kg_m3_seed"]))}
        float omniphysics:dynamicFriction = {usd_float(float(contact["dynamic_friction_seed"]))}
        float omniphysics:staticFriction = {usd_float(float(contact["static_friction_seed"]))}
        float omniphysics:youngsModulus = {usd_float(float(material["youngs_modulus_pa_seed"]))}
        float omniphysics:poissonsRatio = {usd_float(float(material["poisson_ratio_seed"]))}
        float physxDeformableMaterial:elasticityDamping = {usd_float(float(material["damping_ratio_seed"]))}
    }}

    float omniphysics:mass = {usd_float(mass_kg)}
    bool physxDeformableBody:disableGravity = true
    bool physxDeformableBody:selfCollision = {str(bool(solver["self_collision"])).lower()}
    int physxDeformableBody:solverPositionIterationCount = {int(solver["position_iterations"])}
    float physxDeformableBody:linearDamping = {usd_float(float(solver["vertex_velocity_damping"]))}
    rel material:binding:physics = </{root}/FEMPhysics>

    def TetMesh "SimulationMesh" (
        prepend apiSchemas = ["OmniPhysicsVolumeDeformableSimAPI", "OmniPhysicsDeformablePoseAPI:bind"]
    )
    {{
        {tet_attributes.replace(chr(10), chr(10) + "        ")}
        {simulation_rest_attributes.replace(chr(10), chr(10) + "        ")}
        custom uniform token[] drAnmar:tetLayerNames = [{layer_names}]
        custom int[] drAnmar:tetLayerIds = [{layer_ids}]
    }}

    def TetMesh "CollisionMesh" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "OmniPhysicsDeformablePoseAPI:bind"]
    )
    {{
        {tet_attributes.replace(chr(10), chr(10) + "        ")}
        float physxCollision:contactOffset = 0.0005
        float physxCollision:restOffset = 0.0001
        uniform token purpose = "guide"
    }}

    def Mesh "Visual" (
        prepend apiSchemas = ["MaterialBindingAPI", "OmniPhysicsDeformablePoseAPI:bind"]
    )
    {{
        {mesh_attributes(mesh)}

{subset_blocks(mesh, root_prim=root)}
    }}
}}
"""


def build_report(
    profile_path: Path,
    profile: dict[str, Any],
    mesh: TissueMesh,
    surface_path: Path,
    tet_path: Path,
) -> dict[str, Any]:
    derived = derive_tissue(profile, mesh)
    return {
        "schema": "dr.anmar.suturable-tissue-asset-report.v2",
        "name": ASSET_NAME,
        "asset_id": ASSET_ID,
        "asset_version": ASSET_VERSION,
        "profile": portable_path(profile_path),
        "profile_id": profile["id"],
        "surface_asset": portable_path(surface_path),
        "surface_asset_sha256": sha256(surface_path),
        "tetmesh_asset": portable_path(tet_path),
        "tetmesh_asset_sha256": sha256(tet_path),
        "point_count": derived.point_count,
        "tetrahedron_count": derived.tetrahedron_count,
        "surface_triangle_count": derived.surface_triangle_count,
        "connected_components": mesh.connected_components,
        "volume_m3": mesh.volume_m3,
        "mass_kg": derived.mass_kg,
        "minimum_tetra_volume_m3": mesh.minimum_tetra_volume_m3,
        "rest_wound_gap_bottom_m": derived.rest_wound_gap_bottom_m,
        "rest_wound_gap_top_m": derived.rest_wound_gap_top_m,
        "outer_attachment_node_count": derived.outer_attachment_node_count,
        "surface_groups": {
            name: len(indices) for name, indices in mesh.surface_groups.items()
        },
        "tetrahedron_groups": {
            name: len(indices) for name, indices in mesh.tetrahedron_groups.items()
        },
        "stable_physx_layer_proxy": profile["stable_physx_proxy"],
        "stable_capabilities": [
            "intact_deformation",
            "two_way_contact",
            "grasping",
            "retraction",
            "wound_edge_approximation",
        ],
        "gated_capabilities": [
            "arbitrary_puncture",
            "persistent_tract",
            "thread_passage",
            "cutting",
        ],
        "sim_to_real_gap_count": len(profile["sim_to_real"]["gaps"]),
        "clinical_validation": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_TISSUE_PROFILE_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--tet-output",
        type=Path,
        default=DEFAULT_TET_OUTPUT,
    )
    parser.add_argument(
        "--left-flap-output",
        type=Path,
        default=DEFAULT_LEFT_FLAP_OUTPUT,
    )
    parser.add_argument(
        "--right-flap-output",
        type=Path,
        default=DEFAULT_RIGHT_FLAP_OUTPUT,
    )
    parser.add_argument(
        "--left-flap-tet-output",
        type=Path,
        default=DEFAULT_LEFT_FLAP_TET_OUTPUT,
    )
    parser.add_argument(
        "--right-flap-tet-output",
        type=Path,
        default=DEFAULT_RIGHT_FLAP_TET_OUTPUT,
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    profile = load_tissue_profile(args.profile)
    mesh = build_tissue_mesh(profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.tet_output.parent.mkdir(parents=True, exist_ok=True)
    args.left_flap_output.parent.mkdir(parents=True, exist_ok=True)
    args.right_flap_output.parent.mkdir(parents=True, exist_ok=True)
    args.left_flap_tet_output.parent.mkdir(parents=True, exist_ok=True)
    args.right_flap_tet_output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(author_surface(profile, mesh), encoding="utf-8")
    args.tet_output.write_text(
        author_tetmesh(profile, mesh),
        encoding="utf-8",
    )
    left_flap_mesh = component_mesh(mesh, 0)
    right_flap_mesh = component_mesh(mesh, 1)
    args.left_flap_output.write_text(
        author_surface(profile, left_flap_mesh),
        encoding="utf-8",
    )
    args.right_flap_output.write_text(
        author_surface(profile, right_flap_mesh),
        encoding="utf-8",
    )
    args.left_flap_tet_output.write_text(
        author_tetmesh(profile, left_flap_mesh),
        encoding="utf-8",
    )
    args.right_flap_tet_output.write_text(
        author_tetmesh(profile, right_flap_mesh),
        encoding="utf-8",
    )
    report = build_report(
        args.profile,
        profile,
        mesh,
        args.output,
        args.tet_output,
    )
    report["physx_flap_assets"] = {
        "left": portable_path(args.left_flap_output),
        "left_sha256": sha256(args.left_flap_output),
        "right": portable_path(args.right_flap_output),
        "right_sha256": sha256(args.right_flap_output),
        "left_tetmesh": portable_path(args.left_flap_tet_output),
        "left_tetmesh_sha256": sha256(args.left_flap_tet_output),
        "right_tetmesh": portable_path(args.right_flap_tet_output),
        "right_tetmesh_sha256": sha256(args.right_flap_tet_output),
        "purpose": (
            "one connected explicit collision/simulation TetMesh hierarchy per wound flap"
        ),
    }
    args.report.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
