#!/usr/bin/env python3
"""Author the independent Dr.Anmar 4-0 suture as a layered OpenUSD asset.

Closed braided meshes provide segment-local render detail. Hidden primitive
capsules and D6 joints form the portable PhysX compatibility payload; the
promoted knotting runtime consumes the same centerline and material profile
through the NVIDIA Warp backend. No NVIDIA Rope.usd data is read or referenced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from dr_anmar_needle_model import (
    DEFAULT_NEEDLE_PROFILE_PATH,
    build_needle_collision_capsules,
    build_needle_mesh,
    derive_needle,
    load_needle_profile,
    needle_mesh_collision_coverage,
    needle_mesh_normal_quality,
)
from dr_anmar_suture_integration import (
    DR_ANMAR_NEEDLE_ASSET_ID,
    DR_ANMAR_NEEDLE_ASSET_PATH,
    DR_ANMAR_NEEDLE_ASSET_VERSION,
    DR_ANMAR_NEEDLE_BASE_ASSET_PATH,
    DR_ANMAR_NEEDLE_GEOMETRY_ASSET_PATH,
    DR_ANMAR_NEEDLE_MATERIALS_ASSET_PATH,
    DR_ANMAR_NEEDLE_NAME,
    DR_ANMAR_NEEDLE_PHYSICS_ASSET_PATH,
    DR_ANMAR_NEEDLE_PHYSX_ASSET_PATH,
    DR_ANMAR_NEEDLE_ROOT_PRIM,
    SUTURE_ASSET_PATH,
    SUTURE_BASE_ASSET_PATH,
    SUTURE_GEOMETRY_ASSET_PATH,
    SUTURE_MATERIALS_ASSET_PATH,
    SUTURE_NEEDLE_INTERFACE_CENTER_M,
    SUTURE_NORMAL_ROUGHNESS_TEXTURE_PATH,
    SUTURE_PHYSICS_ASSET_PATH,
    SUTURE_PHYSX_ASSET_PATH,
)
from dr_anmar_suture_model import (
    DEFAULT_PROFILE_PATH,
    SutureInterfaceVisualMesh,
    SutureRigidBodyMassProperties,
    SutureVisualMesh,
    build_suture_interface_visual_mesh,
    build_suture_material_texture,
    build_suture_visual_mesh,
    capsule_point_containment_margin,
    derive,
    encode_suture_material_texture_png,
    load_profile,
    suture_interface_mass_properties,
    suture_segment_collision_radius,
    suture_segment_mass_properties,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = SUTURE_ASSET_PATH
DEFAULT_BASE_OUTPUT = SUTURE_BASE_ASSET_PATH
DEFAULT_GEOMETRY_OUTPUT = SUTURE_GEOMETRY_ASSET_PATH
DEFAULT_MATERIALS_OUTPUT = SUTURE_MATERIALS_ASSET_PATH
DEFAULT_TEXTURE_OUTPUT = SUTURE_NORMAL_ROUGHNESS_TEXTURE_PATH
DEFAULT_PHYSICS_OUTPUT = SUTURE_PHYSICS_ASSET_PATH
DEFAULT_PHYSX_OUTPUT = SUTURE_PHYSX_ASSET_PATH
DEFAULT_NEEDLE_OUTPUT = DR_ANMAR_NEEDLE_ASSET_PATH
DEFAULT_NEEDLE_BASE_OUTPUT = DR_ANMAR_NEEDLE_BASE_ASSET_PATH
DEFAULT_NEEDLE_GEOMETRY_OUTPUT = DR_ANMAR_NEEDLE_GEOMETRY_ASSET_PATH
DEFAULT_NEEDLE_MATERIALS_OUTPUT = DR_ANMAR_NEEDLE_MATERIALS_ASSET_PATH
DEFAULT_NEEDLE_PHYSICS_OUTPUT = DR_ANMAR_NEEDLE_PHYSICS_ASSET_PATH
DEFAULT_NEEDLE_PHYSX_OUTPUT = DR_ANMAR_NEEDLE_PHYSX_ASSET_PATH


def usd_float(value: float) -> str:
    return f"{value:.12g}"


def usd_vec(values: tuple[float, float, float]) -> str:
    return "(" + ", ".join(usd_float(value) for value in values) + ")"


def usd_vec2(values: tuple[float, float]) -> str:
    return "(" + ", ".join(usd_float(value) for value in values) + ")"


def usd_quat(values: tuple[float, float, float, float]) -> str:
    return "(" + usd_float(values[0]) + ", " + ", ".join(usd_float(value) for value in values[1:]) + ")"


def indent(text: str, spaces: int = 4) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line else "" for line in text.splitlines())


def anchored_reference_path(path: str) -> str:
    """Make a local USD dependency explicitly relative to its owning layer."""

    if path.startswith(("./", "../")):
        return path
    if path.startswith("/") or "://" in path:
        raise ValueError(f"asset dependency must be local and relative: {path}")
    return f"./{path}"


def swage_interface_geometry_block(
    *,
    name: str,
    x_m: float,
    radius_m: float,
    cylinder_height_m: float,
    color: tuple[float, float, float],
    mesh: SutureInterfaceVisualMesh,
    collider_purpose: str,
    collider_visibility: str,
) -> str:
    """Author an explicit swage render mesh with a primitive collider."""

    total_half_length = cylinder_height_m / 2.0 + radius_m
    points = ",\n            ".join(usd_vec(point) for point in mesh.points)
    normals = ",\n            ".join(usd_vec(normal) for normal in mesh.normals)
    face_counts = ", ".join(str(value) for value in mesh.face_vertex_counts)
    face_indices = ", ".join(str(value) for value in mesh.face_vertex_indices)
    return f"""def Xform "{name}"
{{
    double3 xformOp:translate = {usd_vec((x_m, 0.0, 0.0))}
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Mesh "Visual"
    {{
        uniform bool doubleSided = false
        float3[] extent = [{usd_vec(mesh.extent_min)}, {usd_vec(mesh.extent_max)}]
        int[] faceVertexCounts = [{face_counts}]
        int[] faceVertexIndices = [{face_indices}]
        point3f[] points = [
            {points}
        ]
        normal3f[] normals = [
            {normals}
        ] (
            interpolation = "vertex"
        )
        color3f[] primvars:displayColor = [{usd_vec(color)}]
        uniform token subdivisionScheme = "none"
    }}

    def Capsule "Collision"
    {{
        uniform token axis = "X"
        float height = {usd_float(cylinder_height_m)}
        float radius = {usd_float(radius_m)}
        float3[] extent = [{usd_vec((-total_half_length, -radius_m, -radius_m))}, {usd_vec((total_half_length, radius_m, radius_m))}]
        uniform token purpose = "{collider_purpose}"
        token visibility = "{collider_visibility}"
    }}
}}"""


def braided_segment_geometry_block(
    *,
    name: str,
    x_m: float,
    collision_radius_m: float,
    cylinder_height_m: float,
    color: tuple[float, float, float],
    mesh: SutureVisualMesh,
    collider_purpose: str,
    collider_visibility: str,
) -> str:
    """Author one rigid segment with a detailed render mesh and primitive collider."""

    total_half_length = cylinder_height_m / 2.0 + collision_radius_m
    points = ",\n            ".join(usd_vec(point) for point in mesh.points)
    normals = ",\n            ".join(usd_vec(normal) for normal in mesh.normals)
    tangents = ",\n            ".join(usd_vec(tangent) for tangent in mesh.tangents)
    binormals = ",\n            ".join(usd_vec(binormal) for binormal in mesh.binormals)
    tangent_frame_indices = ", ".join(str(value) for value in mesh.tangent_frame_indices)
    texcoords = ",\n            ".join(usd_vec2(texcoord) for texcoord in mesh.texcoords)
    texcoord_indices = ", ".join(str(value) for value in mesh.texcoord_indices)
    face_counts = ", ".join(str(value) for value in mesh.face_vertex_counts)
    face_indices = ", ".join(str(value) for value in mesh.face_vertex_indices)
    return f"""def Xform "{name}"
{{
    double3 xformOp:translate = {usd_vec((x_m, 0.0, 0.0))}
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Mesh "Visual"
    {{
        uniform bool doubleSided = false
        float3[] extent = [{usd_vec(mesh.extent_min)}, {usd_vec(mesh.extent_max)}]
        int[] faceVertexCounts = [{face_counts}]
        int[] faceVertexIndices = [{face_indices}]
        point3f[] points = [
            {points}
        ]
        normal3f[] normals = [
            {normals}
        ] (
            interpolation = "faceVarying"
        )
        int[] primvars:normals:indices = [{tangent_frame_indices}]
        vector3f[] primvars:tangents = [
            {tangents}
        ] (
            interpolation = "faceVarying"
        )
        int[] primvars:tangents:indices = [{tangent_frame_indices}]
        vector3f[] primvars:binormals = [
            {binormals}
        ] (
            interpolation = "faceVarying"
        )
        int[] primvars:binormals:indices = [{tangent_frame_indices}]
        texCoord2f[] primvars:st = [
            {texcoords}
        ] (
            interpolation = "faceVarying"
        )
        int[] primvars:st:indices = [{texcoord_indices}]
        color3f[] primvars:displayColor = [{usd_vec(color)}]
        uniform token subdivisionScheme = "none"
    }}

    def Capsule "Collision"
    {{
        uniform token axis = "X"
        float height = {usd_float(cylinder_height_m)}
        float radius = {usd_float(collision_radius_m)}
        float3[] extent = [{usd_vec((-total_half_length, -collision_radius_m, -collision_radius_m))}, {usd_vec((total_half_length, collision_radius_m, collision_radius_m))}]
        uniform token purpose = "{collider_purpose}"
        token visibility = "{collider_visibility}"
    }}
}}"""


def capsule_physics_block(
    *,
    name: str,
    mass_properties: SutureRigidBodyMassProperties,
    kinematic: bool,
    filtered_pair: str | None,
    physics_material_path: str | None = None,
) -> str:
    schemas = ['"PhysicsRigidBodyAPI"', '"PhysicsMassAPI"']
    if filtered_pair:
        schemas.append('"PhysicsFilteredPairsAPI"')
    filtered = f"\n    rel physics:filteredPairs = <{filtered_pair}>" if filtered_pair else ""
    collision_schemas = ['"PhysicsCollisionAPI"']
    if physics_material_path:
        collision_schemas.append('"MaterialBindingAPI"')
    binding = f"\n        rel material:binding:physics = <{physics_material_path}>" if physics_material_path else ""
    return f"""over "{name}" (
    prepend apiSchemas = [{", ".join(schemas)}]
)
{{
    bool physics:rigidBodyEnabled = true
    bool physics:kinematicEnabled = {"true" if kinematic else "false"}
    float physics:mass = {usd_float(mass_properties.mass_kg)}
    {filtered.lstrip()}

    over "Collision" (
        prepend apiSchemas = [{", ".join(collision_schemas)}]
    )
    {{
        bool physics:collisionEnabled = true
        {binding.lstrip()}
    }}
}}"""


def capsule_physx_block(
    *,
    name: str,
    radius_m: float,
    profile: dict[str, Any],
) -> str:
    material = profile["material"]
    solver = profile["solver"]
    offset_contract = profile["contact"]["contact_offsets"]
    contact_offset = max(
        float(offset_contract["minimum_m"]),
        min(
            float(offset_contract["maximum_m"]),
            radius_m * float(offset_contract["collision_radius_fraction"]),
        ),
    )
    return f"""over "{name}" (
    prepend apiSchemas = ["PhysxRigidBodyAPI"]
)
{{
    bool physxRigidBody:enableCCD = true
    bool physxRigidBody:enableSpeculativeCCD = {"true" if profile["contact"]["speculative_ccd"] else "false"}
    float physxRigidBody:linearDamping = {usd_float(float(material["linear_velocity_damping"]))}
    float physxRigidBody:angularDamping = {usd_float(float(material["angular_velocity_damping"]))}
    int physxRigidBody:solverPositionIterationCount = {int(solver["position_iterations"])}
    int physxRigidBody:solverVelocityIterationCount = {int(solver["velocity_iterations"])}
    float physxRigidBody:maxDepenetrationVelocity = 0.25
    over "Collision" (
        prepend apiSchemas = ["PhysxCollisionAPI"]
    )
    {{
        float physxCollision:contactOffset = {usd_float(contact_offset)}
        float physxCollision:restOffset = {usd_float(float(offset_contract["rest_offset_m"]))}
    }}
}}"""


def joint_block(
    *,
    name: str,
    body0: str,
    body1: str,
    half_spacing_m: float,
    extension_limit_m: float,
    axial_stiffness_n_m: float,
    axial_damping_n_s_m: float,
    bend_stiffness_n_m_rad: float,
    bend_damping_n_m_s_rad: float,
    twist_stiffness_n_m_rad: float,
    break_force_n: float,
    break_torque_n_m: float,
    swage_fraction: float,
) -> str:
    schemas = [
        '"PhysicsLimitAPI:transX"',
        '"PhysicsLimitAPI:transY"',
        '"PhysicsLimitAPI:transZ"',
        '"PhysicsDriveAPI:transX"',
        '"PhysicsDriveAPI:rotX"',
        '"PhysicsDriveAPI:rotY"',
        '"PhysicsDriveAPI:rotZ"',
    ]
    # Angular USD drives are expressed per degree, while the profile stores
    # the rod stiffness and damping per radian.
    per_degree = math.pi / 180.0
    bend_per_degree = bend_stiffness_n_m_rad * per_degree
    bend_damping_per_degree = bend_damping_n_m_s_rad * per_degree
    twist_per_degree = twist_stiffness_n_m_rad * per_degree
    return f"""def PhysicsJoint "{name}" (
    prepend apiSchemas = [{", ".join(schemas)}]
)
{{
    rel physics:body0 = <{body0}>
    rel physics:body1 = <{body1}>
    point3f physics:localPos0 = {usd_vec((half_spacing_m, 0.0, 0.0))}
    point3f physics:localPos1 = {usd_vec((-half_spacing_m, 0.0, 0.0))}
    quatf physics:localRot0 = (1, 0, 0, 0)
    quatf physics:localRot1 = (1, 0, 0, 0)
    bool physics:collisionEnabled = false
    float physics:breakForce = {usd_float(break_force_n)}
    float physics:breakTorque = {usd_float(break_torque_n_m)}
    float limit:transX:physics:low = {usd_float(-extension_limit_m)}
    float limit:transX:physics:high = {usd_float(extension_limit_m)}
    float limit:transY:physics:low = 1
    float limit:transY:physics:high = -1
    float limit:transZ:physics:low = 1
    float limit:transZ:physics:high = -1
    uniform token drive:transX:physics:type = "force"
    float drive:transX:physics:targetPosition = 0
    float drive:transX:physics:targetVelocity = 0
    float drive:transX:physics:stiffness = {usd_float(axial_stiffness_n_m)}
    float drive:transX:physics:damping = {usd_float(axial_damping_n_s_m)}
    float drive:transX:physics:maxForce = {usd_float(break_force_n)}
    uniform token drive:rotX:physics:type = "force"
    float drive:rotX:physics:targetPosition = 0
    float drive:rotX:physics:targetVelocity = 0
    float drive:rotX:physics:stiffness = {usd_float(twist_per_degree)}
    float drive:rotX:physics:damping = {usd_float(bend_damping_per_degree)}
    float drive:rotX:physics:maxForce = {usd_float(break_torque_n_m)}
    uniform token drive:rotY:physics:type = "force"
    float drive:rotY:physics:targetPosition = 0
    float drive:rotY:physics:targetVelocity = 0
    float drive:rotY:physics:stiffness = {usd_float(bend_per_degree)}
    float drive:rotY:physics:damping = {usd_float(bend_damping_per_degree)}
    float drive:rotY:physics:maxForce = {usd_float(break_torque_n_m)}
    uniform token drive:rotZ:physics:type = "force"
    float drive:rotZ:physics:targetPosition = 0
    float drive:rotZ:physics:targetVelocity = 0
    float drive:rotZ:physics:stiffness = {usd_float(bend_per_degree)}
    float drive:rotZ:physics:damping = {usd_float(bend_damping_per_degree)}
    float drive:rotZ:physics:maxForce = {usd_float(break_torque_n_m)}
    custom float drAnmar:swageFraction = {usd_float(swage_fraction)}
}}"""


def author(
    profile: dict[str, Any],
    *,
    base_reference: str,
    geometry_sublayer_reference: str,
    materials_sublayer_reference: str,
    physics_payload_reference: str,
    physx_payload_reference: str,
    neutral_physics_sublayer_reference: str,
    texture_reference: str,
) -> tuple[str, str, str, str, str, str]:
    """Author interface, base, geometry, materials, neutral physics, and PhysX layers."""

    derived = derive(profile)
    geometry = profile["geometry"]
    tension = profile["tension"]
    contact = profile["contact"]
    swage = profile["swage"]
    visual_representation = geometry["visual_representation"]
    appearance = profile["appearance"]
    material_texture = appearance["normal_roughness_texture"]
    model_identity = profile["asset_structure"]["model_identity"]
    color = (
        float(geometry["color_rgb"][0]),
        float(geometry["color_rgb"][1]),
        float(geometry["color_rgb"][2]),
    )
    spacing = derived.segment_spacing_m
    root = "/DrAnmarSuture4_0"
    suture_visual_path = f"{root}/Looks/SutureVisual"
    swage_visual_path = f"{root}/Looks/SwageVisual"
    suture_physics_path = f"{root}/Materials/SutureMaterial"
    swage_physics_path = f"{root}/Materials/SwageSteel"
    swage_radius = float(swage["needle_end_diameter_m"]) / 2.0
    interface_height = spacing
    segment_mass_properties = suture_segment_mass_properties(
        profile,
        derived=derived,
    )
    interface_mass_properties = suture_interface_mass_properties(
        profile,
        derived=derived,
    )
    geometry_blocks: list[str] = [
        swage_interface_geometry_block(
            name="NeedleInterface",
            x_m=-spacing / 2.0,
            radius_m=swage_radius,
            cylinder_height_m=interface_height,
            color=(0.58, 0.61, 0.66),
            mesh=build_suture_interface_visual_mesh(
                profile,
                derived=derived,
            ),
            collider_purpose=str(visual_representation["collider_purpose"]),
            collider_visibility=str(visual_representation["collider_visibility"]),
        )
    ]
    segment_geometry_blocks: list[str] = []
    segment_physics_blocks: list[str] = []
    segment_physx_blocks: list[str] = []
    for index in range(derived.segment_count):
        radius = suture_segment_collision_radius(
            profile,
            index,
            derived=derived,
        )
        visual_mesh = build_suture_visual_mesh(
            profile,
            index,
            collision_radius_m=radius,
            derived=derived,
        )
        shade = 0.92 + 0.08 * ((index % 3) / 2.0)
        segment_color = (
            max(0.0, min(1.0, color[0] * shade)),
            max(0.0, min(1.0, color[1] * shade)),
            max(0.0, min(1.0, color[2] * shade)),
        )
        previous_path = f"{root}/NeedleInterface" if index == 0 else f"{root}/Segments/S{index - 1:04d}"
        cylinder_height = spacing
        segment_geometry_blocks.append(
            braided_segment_geometry_block(
                name=f"S{index:04d}",
                x_m=(index + 0.5) * spacing,
                collision_radius_m=radius,
                cylinder_height_m=cylinder_height,
                color=segment_color,
                mesh=visual_mesh,
                collider_purpose=str(visual_representation["collider_purpose"]),
                collider_visibility=str(visual_representation["collider_visibility"]),
            )
        )
        segment_physics_blocks.append(
            capsule_physics_block(
                name=f"S{index:04d}",
                mass_properties=segment_mass_properties,
                kinematic=False,
                filtered_pair=previous_path,
                physics_material_path=suture_physics_path,
            )
        )
        segment_physx_blocks.append(
            capsule_physx_block(
                name=f"S{index:04d}",
                radius_m=radius,
                profile=profile,
            )
        )
    joint_blocks: list[str] = []
    extension_limit = spacing * float(tension["joint_extension_limit_fraction"])
    break_torque = (
        derived.straight_failure_load_n * derived.radius_m * float(profile["knot"]["nominal_strength_efficiency"])
    )
    for joint_index in range(derived.segment_count):
        is_swage_joint = joint_index == 0
        segment_index = joint_index
        body0 = f"{root}/NeedleInterface" if is_swage_joint else f"{root}/Segments/S{segment_index - 1:04d}"
        body1 = f"{root}/Segments/S{segment_index:04d}"
        swage_fraction = clamp01(1.0 - segment_index / max(1, derived.swage_segment_count - 1))
        axial_multiplier = lerp1(1.0, float(swage["axial_stiffness_multiplier"]), swage_fraction)
        bend_multiplier = lerp1(1.0, float(swage["bending_stiffness_multiplier"]), swage_fraction)
        failure_load = float(swage["pullout_force_n_seed"]) if is_swage_joint else derived.straight_failure_load_n
        joint_blocks.append(
            joint_block(
                name=f"J{joint_index:04d}",
                body0=body0,
                body1=body1,
                half_spacing_m=spacing / 2.0,
                extension_limit_m=extension_limit,
                axial_stiffness_n_m=derived.axial_joint_stiffness_n_m * axial_multiplier,
                axial_damping_n_s_m=derived.axial_joint_damping_n_s_m * math.sqrt(axial_multiplier),
                bend_stiffness_n_m_rad=derived.bend_joint_stiffness_n_m_rad * bend_multiplier,
                bend_damping_n_m_s_rad=derived.bend_joint_damping_n_m_s_rad * math.sqrt(bend_multiplier),
                twist_stiffness_n_m_rad=derived.twist_joint_stiffness_n_m_rad * bend_multiplier,
                break_force_n=failure_load,
                break_torque_n_m=break_torque,
                swage_fraction=swage_fraction,
            )
        )
    custom_data = {
        "drAnmarAssetVersion": profile["version"],
        "drAnmarClinicalValidation": False,
        "drAnmarCanonicalAssetPackage": "assets/dr_anmar",
        "drAnmarIndependentAsset": True,
        "drAnmarLayerContract": "interface_references_base_with_public_none_physics_physx_payload_variants",
        "drAnmarMassPropertyContract": "explicit_mass_with_native_collider_derived_inertia",
        "drAnmarProfileId": profile["id"],
        "drAnmarRepresentation": "discrete_cosserat_rod_with_braided_render_mesh_and_capsule_colliders",
        "drAnmarStatus": profile["status"],
    }
    custom_lines = "\n".join(
        f"        {'bool' if isinstance(value, bool) else 'string'} {key} = "
        + (("true" if value else "false") if isinstance(value, bool) else json.dumps(str(value)))
        for key, value in custom_data.items()
    )
    interface_geometry_source = indent(geometry_blocks[0])
    segment_geometry_source = indent("\n\n".join(segment_geometry_blocks), 8)
    segment_physics_source = indent("\n\n".join(segment_physics_blocks), 8)
    joint_source = indent("\n\n".join(joint_blocks), 8)
    segment_physx_source = indent("\n\n".join(segment_physx_blocks), 8)
    entry_layer = f"""#usda 1.0
(
    defaultPrim = "DrAnmarSuture4_0"
    doc = "Dr.Anmar 4-0 suture interface with public NVIDIA-style Physics payload variants; not clinically validated."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "DrAnmarSuture4_0" (
    prepend references = @{base_reference}@
    variants = {{
        string Physics = "physx"
    }}
    append variantSets = "Physics"
)
{{
    variantSet "Physics" = {{
        "none" {{
        }}
        "physics" (
            prepend payload = @{physics_payload_reference}@
        ) {{
        }}
        "physx" (
            prepend payload = @{physx_payload_reference}@
        ) {{
        }}
    }}
}}
"""
    base_layer = f"""#usda 1.0
(
    subLayers = [
        @{materials_sublayer_reference}@,
        @{geometry_sublayer_reference}@
    ]
    defaultPrim = "DrAnmarSuture4_0"
    doc = "Dr.Anmar 4-0 suture base identity, hierarchy, geometry, and visual materials."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "DrAnmarSuture4_0" (
    prepend apiSchemas = ["{model_identity["semantic_schema_instance"]}"]
    assetInfo = {{
        string name = "{model_identity["asset_info_name"]}"
        string version = "{profile["version"]}"
    }}
    customData = {{
{custom_lines}
    }}
    displayName = "{model_identity["display_name"]}"
    kind = "{model_identity["kind"]}"
)
{{
    token[] {model_identity["semantic_label_attribute"]} = {json.dumps(model_identity["wikidata_qcodes"])}

    def Scope "Looks"
    {{
    }}

    def Scope "Materials"
    {{
    }}

    def Scope "Segments"
    {{
    }}

    def Scope "Joints"
    {{
    }}
}}
"""
    geometry_layer = f"""#usda 1.0
(
    defaultPrim = "DrAnmarSuture4_0"
    doc = "Dr.Anmar 4-0 suture binary braided visual-mesh and primitive-collider geometry layer."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

over "DrAnmarSuture4_0"
{{
{interface_geometry_source}

    over "Segments"
    {{
{segment_geometry_source}
    }}
}}
"""
    materials_layer = f"""#usda 1.0
(
    defaultPrim = "DrAnmarSuture4_0"
    doc = "Dr.Anmar 4-0 suture visual look-development and inherited binding layer."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

over "DrAnmarSuture4_0"
{{
    over "Looks"
    {{
        def Material "SutureVisual"
        {{
            string inputs:frame:tangentsPrimvarName = "{material_texture["tangent_frame"]["tangent_primvar"]}"
            string inputs:frame:binormalsPrimvarName = "{material_texture["tangent_frame"]["binormal_primvar"]}"
            string inputs:frame:stPrimvarName = "{material_texture["tangent_frame"]["st_primvar"]}"

            def Shader "PreviewSurface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = {usd_vec(color)}
                float inputs:ior = {usd_float(float(appearance["ior_seed"]))}
                float inputs:metallic = {usd_float(float(appearance["metallic_seed"]))}
                normal3f inputs:normal.connect = <{suture_visual_path}/BraidNormalRoughness.outputs:rgb>
                float inputs:roughness.connect = <{suture_visual_path}/BraidNormalRoughness.outputs:a>
                token outputs:surface
            }}
            def Shader "PrimvarReader_st"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                string inputs:varname.connect = <{suture_visual_path}.inputs:frame:stPrimvarName>
                float2 outputs:result
            }}
            def Shader "BraidNormalRoughness"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @{texture_reference}@
                float4 inputs:bias = (-1, -1, -1, 0)
                float4 inputs:scale = (2, 2, 2, 1)
                token inputs:sourceColorSpace = "{material_texture["source_color_space"]}"
                float2 inputs:st.connect = <{suture_visual_path}/PrimvarReader_st.outputs:result>
                token inputs:wrapS = "{material_texture["wrap_s"]}"
                token inputs:wrapT = "{material_texture["wrap_t"]}"
                float outputs:a
                float3 outputs:rgb
            }}
            token outputs:surface.connect = <{suture_visual_path}/PreviewSurface.outputs:surface>
        }}

        def Material "SwageVisual"
        {{
            def Shader "PreviewSurface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.58, 0.61, 0.66)
                float inputs:metallic = 0.86
                float inputs:roughness = 0.24
                token outputs:surface
            }}
            token outputs:surface.connect = <{swage_visual_path}/PreviewSurface.outputs:surface>
        }}
    }}

    over "NeedleInterface" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        rel material:binding = <{swage_visual_path}>
    }}

    over "Segments" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        rel material:binding = <{suture_visual_path}>
    }}
}}
"""
    interface_physics = capsule_physics_block(
        name="NeedleInterface",
        mass_properties=interface_mass_properties,
        kinematic=True,
        filtered_pair=f"{root}/Segments/S0000",
        physics_material_path=swage_physics_path,
    )
    neutral_physics_layer = f"""#usda 1.0
(
    defaultPrim = "DrAnmarSuture4_0"
    doc = "Dr.Anmar 4-0 suture engine-neutral rigid-body, collision, material, and D6-joint layer."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

over "DrAnmarSuture4_0"
{{
    over "Materials"
    {{
        def Material "SutureMaterial" (
            prepend apiSchemas = ["PhysicsMaterialAPI"]
        )
        {{
            float physics:staticFriction = {usd_float(float(contact["static_friction"]))}
            float physics:dynamicFriction = {usd_float(float(contact["dynamic_friction"]))}
            float physics:restitution = {usd_float(float(contact["restitution"]))}
        }}

        def Material "SwageSteel" (
            prepend apiSchemas = ["PhysicsMaterialAPI"]
        )
        {{
            float physics:staticFriction = 0.35
            float physics:dynamicFriction = 0.25
            float physics:restitution = 0
        }}
    }}

{indent(interface_physics)}

    over "Segments"
    {{
{segment_physics_source}
    }}

    over "Joints"
    {{
{joint_source}
    }}
}}
"""
    interface_physx = capsule_physx_block(
        name="NeedleInterface",
        radius_m=swage_radius,
        profile=profile,
    )
    physx_layer = f"""#usda 1.0
(
    subLayers = [
        @{neutral_physics_sublayer_reference}@
    ]
    defaultPrim = "DrAnmarSuture4_0"
    doc = "Dr.Anmar 4-0 suture PhysX-specific CCD, damping, solver, contact-offset, and combine-mode layer."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

over "DrAnmarSuture4_0"
{{
    over "Materials"
    {{
        over "SutureMaterial" (
            prepend apiSchemas = ["PhysxMaterialAPI"]
        )
        {{
            uniform token physxMaterial:frictionCombineMode = "{contact["combine_mode"]}"
        }}

        over "SwageSteel" (
            prepend apiSchemas = ["PhysxMaterialAPI"]
        )
        {{
            uniform token physxMaterial:frictionCombineMode = "{contact["combine_mode"]}"
        }}
    }}

{indent(interface_physx)}

    over "Segments"
    {{
{segment_physx_source}
    }}
}}
"""
    return entry_layer, base_layer, geometry_layer, materials_layer, neutral_physics_layer, physx_layer


def author_dr_anmar_needle(
    suture_profile: dict[str, Any],
    needle_profile: dict[str, Any],
    *,
    base_reference: str,
    suture_reference: str,
    geometry_sublayer_reference: str,
    materials_sublayer_reference: str,
    physics_payload_reference: str,
    physx_payload_reference: str,
    neutral_physics_sublayer_reference: str,
) -> tuple[str, str, str, str, str, str]:
    """Author interface, base, geometry, materials, neutral physics, and PhysX layers."""

    derived_needle = derive_needle(needle_profile)
    mass_properties = derived_needle.mass_properties
    mesh = build_needle_mesh(needle_profile)
    contact = needle_profile["contact"]
    solver = needle_profile["solver"]
    render_collision = needle_profile["construction"]["collision_contract"]["render_collision_separation"]
    model_identity = needle_profile["construction"]["layer_organization"]["model_identity"]
    swage_anchor = derived_needle.swage_anchor_m
    swage_tangent = derived_needle.swage_tangent
    swage_yaw = math.atan2(swage_tangent[1], swage_tangent[0])
    swage_orientation = (
        math.cos(swage_yaw / 2.0),
        0.0,
        0.0,
        math.sin(swage_yaw / 2.0),
    )
    rotated_interface_center = (
        swage_tangent[0] * SUTURE_NEEDLE_INTERFACE_CENTER_M[0],
        swage_tangent[1] * SUTURE_NEEDLE_INTERFACE_CENTER_M[0],
        0.0,
    )
    suture_translation = (
        swage_anchor[0] - rotated_interface_center[0],
        swage_anchor[1] - rotated_interface_center[1],
        swage_anchor[2] - rotated_interface_center[2],
    )
    root = f"/{DR_ANMAR_NEEDLE_ROOT_PRIM}"
    material_organization = needle_profile["material"]["usd_organization"]
    visual_material_path = f"{root}/{material_organization['scope']}/{material_organization['visual_material']}"
    physics_material_path = f"{root}/{material_organization['scope']}/{material_organization['physics_material']}"
    mesh_points = ",\n            ".join(usd_vec(point) for point in mesh.points)
    mesh_normals = ",\n            ".join(usd_vec(normal) for normal in mesh.normals)
    face_counts = ", ".join(str(value) for value in mesh.face_vertex_counts)
    face_indices = ", ".join(str(value) for value in mesh.face_vertex_indices)
    normal_indices = ", ".join(str(value) for value in mesh.normal_indices)
    neutral_collision_blocks: list[str] = []
    physx_collision_blocks: list[str] = []
    collision_capsules = build_needle_collision_capsules(needle_profile)
    for index, capsule in enumerate(collision_capsules):
        neutral_collision_blocks.append(f"""def Capsule "C{index:03d}" (
    prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]
)
{{
    uniform token axis = "X"
    float height = {usd_float(capsule.cylinder_height_m)}
    float radius = {usd_float(capsule.collision_radius_m)}
    float3[] extent = [{usd_vec(capsule.extent_min)}, {usd_vec(capsule.extent_max)}]
    uniform token purpose = "{render_collision["collider_purpose"]}"
    token visibility = "{render_collision["collider_visibility"]}"
    rel {render_collision["collider_physics_material_binding"]} = <{physics_material_path}>
    bool physics:collisionEnabled = true
    quatf xformOp:orient = {usd_quat(capsule.orientation_wxyz)}
    double3 xformOp:translate = {usd_vec(capsule.center_m)}
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
}}""")
        physx_collision_blocks.append(f"""over "C{index:03d}" (
    prepend apiSchemas = ["PhysxCollisionAPI"]
)
{{
    float physxCollision:contactOffset = {usd_float(capsule.contact_offset_m)}
    float physxCollision:restOffset = {usd_float(capsule.rest_offset_m)}
}}""")
    neutral_collisions = indent("\n\n".join(neutral_collision_blocks), 12)
    physx_collisions = indent("\n\n".join(physx_collision_blocks), 12)
    sim_to_real_gap_count = len(needle_profile["sim_to_real"]["gaps"])
    implemented_randomization_count = len(needle_profile["sim_to_real"]["implemented_randomization_on_episode_reset"])
    entry_layer = f"""#usda 1.0
(
    defaultPrim = "{DR_ANMAR_NEEDLE_ROOT_PRIM}"
    doc = "{DR_ANMAR_NEEDLE_NAME} interface with synchronized public NVIDIA-style Physics payload variants; not clinically validated."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{DR_ANMAR_NEEDLE_ROOT_PRIM}" (
    prepend references = @{base_reference}@
    variants = {{
        string Physics = "physx"
    }}
    append variantSets = "Physics"
)
{{
    variantSet "Physics" = {{
        "none" {{
            over "Suture" (
                variants = {{
                    string Physics = "none"
                }}
            )
            {{
            }}
        }}
        "physics" (
            prepend payload = @{physics_payload_reference}@
        ) {{
            over "Suture" (
                variants = {{
                    string Physics = "physics"
                }}
            )
            {{
            }}
        }}
        "physx" (
            prepend payload = @{physx_payload_reference}@
        ) {{
            over "Suture" (
                variants = {{
                    string Physics = "physx"
                }}
            )
            {{
            }}
        }}
    }}
}}
"""

    base_layer = f"""#usda 1.0
(
    subLayers = [
        @{materials_sublayer_reference}@,
        @{geometry_sublayer_reference}@
    ]
    defaultPrim = "{DR_ANMAR_NEEDLE_ROOT_PRIM}"
    doc = "{DR_ANMAR_NEEDLE_NAME} base identity, hierarchy, suture assembly, geometry, and visual materials."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{DR_ANMAR_NEEDLE_ROOT_PRIM}" (
    prepend apiSchemas = ["{model_identity["semantic_schema_instance"]}"]
    assetInfo = {{
        string name = "{model_identity["asset_info_name"]}"
        string version = "{DR_ANMAR_NEEDLE_ASSET_VERSION}"
    }}
    customData = {{
        string drAnmarAssetId = "{DR_ANMAR_NEEDLE_ASSET_ID}"
        string drAnmarAssetName = "{DR_ANMAR_NEEDLE_NAME}"
        string drAnmarAssetVersion = "{DR_ANMAR_NEEDLE_ASSET_VERSION}"
        string drAnmarAuthorship = "Independent Dr.Anmar geometry, collision, instrument composition and suture physics"
        bool drAnmarClinicalValidation = false
        string drAnmarGeometrySource = "independently_generated_parametric_geometry"
        int drAnmarMassPropertyIntegrationSlices = {mass_properties.integration_slices}
        string drAnmarContactOffsetContract = "scale_aware_physx_engine_layer_authoring"
        string drAnmarNormalContract = "analytic_taper_and_curvature_aware_indexed_face_varying_primvar"
        string drAnmarRenderCollisionContract = "separate_visual_mesh_and_guide_purpose_invisible_compound_colliders"
        string drAnmarMaterialContract = "top_level_looks_with_separate_visual_and_physics_materials"
        string drAnmarLayerContract = "interface_references_base_with_public_none_physics_physx_payload_variants"
        string drAnmarNeedleProfileId = "{needle_profile["id"]}"
        string drAnmarRepresentation = "high_resolution_mesh_with_compound_capsule_collision"
        string drAnmarCollisionContract = "curvature_sagitta_bounded_capsules_with_explicit_extents"
        int drAnmarResetRandomizationCount = {implemented_randomization_count}
        int drAnmarSimToRealGapCount = {sim_to_real_gap_count}
        string drAnmarSutureProfileId = "{suture_profile["id"]}"
        string drAnmarSuturePhysicsProvenance = "Independent Dr.Anmar 4-0 discrete Cosserat rod"
        string drAnmarSwageConnection = "fixed_needle_to_interface_then_breakable_pullout_joint"
        string drAnmarStatus = "{needle_profile["status"]}"
    }}
    displayName = "{model_identity["display_name"]}"
    kind = "{model_identity["kind"]}"
)
{{
    token[] {model_identity["semantic_label_attribute"]} = {json.dumps(model_identity["assembly_wikidata_qcodes"])}

    def Xform "Needle" (
        prepend apiSchemas = ["{model_identity["semantic_schema_instance"]}"]
        kind = "{model_identity["child_model_kind"]}"
    )
    {{
        token[] {model_identity["semantic_label_attribute"]} = {json.dumps(model_identity["needle_wikidata_qcodes"])}

        def Xform "SutureAnchor"
        {{
            quatf xformOp:orient = {usd_quat(swage_orientation)}
            double3 xformOp:translate = {usd_vec(swage_anchor)}
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
        }}
    }}

    def Xform "Suture" (
        kind = "{model_identity["child_model_kind"]}"
        prepend references = @{suture_reference}@
    )
    {{
        double3 xformOp:translate = {usd_vec(suture_translation)}
        quatf xformOp:orient = {usd_quat(swage_orientation)}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
    }}
}}
"""

    geometry_layer = f"""#usda 1.0
(
    defaultPrim = "{DR_ANMAR_NEEDLE_ROOT_PRIM}"
    doc = "{DR_ANMAR_NEEDLE_NAME} binary visual geometry layer. Composed by DrAnmarNeedle.usda."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

over "{DR_ANMAR_NEEDLE_ROOT_PRIM}"
{{
    over "Needle"
    {{
        def Mesh "Visual"
        {{
            uniform bool doubleSided = false
            float3[] extent = [{usd_vec(mesh.extent_min)}, {usd_vec(mesh.extent_max)}]
            int[] faceVertexCounts = [{face_counts}]
            int[] faceVertexIndices = [{face_indices}]
            point3f[] points = [
            {mesh_points}
            ]
            normal3f[] normals = [
            {mesh_normals}
            ] (
                interpolation = "faceVarying"
            )
            int[] primvars:normals:indices = [{normal_indices}]
            uniform token subdivisionScheme = "none"
        }}
    }}
}}
"""

    materials_layer = f"""#usda 1.0
(
    defaultPrim = "{DR_ANMAR_NEEDLE_ROOT_PRIM}"
    doc = "{DR_ANMAR_NEEDLE_NAME} visual look-development and binding layer. Composed by DrAnmarNeedle.usda."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

over "{DR_ANMAR_NEEDLE_ROOT_PRIM}"
{{
    def Scope "{material_organization["scope"]}"
    {{
        def Material "{material_organization["visual_material"]}"
        {{
            def Shader "PreviewSurface"
            {{
                uniform token info:id = "{material_organization["visual_shader"]}"
                color3f inputs:diffuseColor = (0.53, 0.58, 0.64)
                float inputs:metallic = {usd_float(float(needle_profile["appearance"]["metallic_seed"]))}
                float inputs:roughness = {usd_float(float(needle_profile["appearance"]["roughness_seed"]))}
                token outputs:surface
            }}
            token outputs:surface.connect = <{visual_material_path}/PreviewSurface.outputs:surface>
        }}
    }}

    over "Needle"
    {{
        over "Visual" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {{
            rel material:binding = <{visual_material_path}>
        }}
    }}
}}
"""

    physics_layer = f"""#usda 1.0
(
    defaultPrim = "{DR_ANMAR_NEEDLE_ROOT_PRIM}"
    doc = "{DR_ANMAR_NEEDLE_NAME} engine-neutral local physics layer. Composed by DrAnmarNeedle_physx.usda."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

over "{DR_ANMAR_NEEDLE_ROOT_PRIM}"
{{
    over "{material_organization["scope"]}"
    {{
        def Material "{material_organization["physics_material"]}" (
            prepend apiSchemas = ["PhysicsMaterialAPI"]
        )
        {{
            float physics:staticFriction = {usd_float(float(contact["static_friction_seed"]))}
            float physics:dynamicFriction = {usd_float(float(contact["dynamic_friction_seed"]))}
            float physics:restitution = {usd_float(float(contact["restitution_seed"]))}
        }}
    }}

    over "Needle" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {{
        bool physics:rigidBodyEnabled = true
        bool physics:kinematicEnabled = false
        float physics:mass = {usd_float(derived_needle.mass_kg)}
        point3f physics:centerOfMass = {usd_vec(mass_properties.center_of_mass_m)}
        float3 physics:diagonalInertia = {usd_vec(mass_properties.diagonal_inertia_kg_m2)}
        quatf physics:principalAxes = {usd_quat(mass_properties.principal_axes_wxyz)}

        def Scope "Collision"
        {{
{neutral_collisions}
        }}
    }}

    over "Suture"
    {{
        over "NeedleInterface"
        {{
            bool physics:kinematicEnabled = false
        }}
    }}

    def PhysicsFixedJoint "FactorySwage"
    {{
        rel physics:body0 = <{root}/Needle>
        rel physics:body1 = <{root}/Suture/NeedleInterface>
        point3f physics:localPos0 = {usd_vec(swage_anchor)}
        point3f physics:localPos1 = (0, 0, 0)
        quatf physics:localRot0 = {usd_quat(swage_orientation)}
        quatf physics:localRot1 = (1, 0, 0, 0)
        bool physics:collisionEnabled = false
    }}
}}
"""

    physx_layer = f"""#usda 1.0
(
    subLayers = [
        @{neutral_physics_sublayer_reference}@
    ]
    defaultPrim = "{DR_ANMAR_NEEDLE_ROOT_PRIM}"
    doc = "{DR_ANMAR_NEEDLE_NAME} PhysX-specific tuning layer. Composed by DrAnmarNeedle.usda."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

over "{DR_ANMAR_NEEDLE_ROOT_PRIM}"
{{
    over "{material_organization["scope"]}"
    {{
        over "{material_organization["physics_material"]}" (
            prepend apiSchemas = ["PhysxMaterialAPI"]
        )
        {{
            uniform token physxMaterial:frictionCombineMode = "{contact["combine_mode"]}"
        }}
    }}

    over "Needle" (
        prepend apiSchemas = ["PhysxRigidBodyAPI"]
    )
    {{
        bool physxRigidBody:enableCCD = {"true" if solver["ccd"] else "false"}
        int physxRigidBody:solverPositionIterationCount = {int(solver["position_iterations"])}
        int physxRigidBody:solverVelocityIterationCount = {int(solver["velocity_iterations"])}
        float physxRigidBody:maxDepenetrationVelocity = {usd_float(float(solver["max_depenetration_velocity_m_s"]))}

        over "Collision"
        {{
{physx_collisions}
        }}
    }}
}}
"""
    return entry_layer, base_layer, geometry_layer, materials_layer, physics_layer, physx_layer


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def lerp1(left: float, right: float, amount: float) -> float:
    return left + (right - left) * amount


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


def write_usdc(text: str, output: Path, usdcat_command: str) -> None:
    usdcat_path = shutil.which(usdcat_command)
    if usdcat_path is None:
        try:
            from pxr import Sdf
        except ImportError as exc:
            raise RuntimeError(
                "OpenUSD usdcat or the pxr Python bindings are required to "
                f"author binary geometry: {usdcat_command}"
            ) from exc
        layer = Sdf.Layer.CreateAnonymous(f"{output.stem}.usda")
        if not layer.ImportFromString(text):
            raise RuntimeError(f"OpenUSD rejected generated geometry for {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        if not layer.Export(str(output), args={"format": "usdc"}):
            raise RuntimeError(f"OpenUSD could not export binary geometry to {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dr_anmar_usdc_", dir=output.parent) as temporary_directory:
        temporary_root = Path(temporary_directory)
        source = temporary_root / f"{output.stem}.usda"
        binary = temporary_root / output.name
        source.write_text(text, encoding="utf-8")
        subprocess.run(
            [
                usdcat_path,
                str(source),
                "--out",
                str(binary),
                "--usdFormat",
                "usdc",
            ],
            check=True,
        )
        binary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument(
        "--needle-profile",
        type=Path,
        default=DEFAULT_NEEDLE_PROFILE_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--base-output",
        type=Path,
        default=DEFAULT_BASE_OUTPUT,
    )
    parser.add_argument(
        "--geometry-output",
        type=Path,
        default=DEFAULT_GEOMETRY_OUTPUT,
    )
    parser.add_argument(
        "--materials-output",
        type=Path,
        default=DEFAULT_MATERIALS_OUTPUT,
    )
    parser.add_argument(
        "--texture-output",
        type=Path,
        default=DEFAULT_TEXTURE_OUTPUT,
    )
    parser.add_argument(
        "--physics-output",
        type=Path,
        default=DEFAULT_PHYSICS_OUTPUT,
    )
    parser.add_argument(
        "--physx-output",
        type=Path,
        default=DEFAULT_PHYSX_OUTPUT,
    )
    parser.add_argument(
        "--needle-output",
        "--assembly-output",
        dest="needle_output",
        type=Path,
        default=DEFAULT_NEEDLE_OUTPUT,
    )
    parser.add_argument(
        "--needle-base-output",
        type=Path,
        default=DEFAULT_NEEDLE_BASE_OUTPUT,
    )
    parser.add_argument(
        "--needle-geometry-output",
        type=Path,
        default=DEFAULT_NEEDLE_GEOMETRY_OUTPUT,
    )
    parser.add_argument(
        "--needle-materials-output",
        type=Path,
        default=DEFAULT_NEEDLE_MATERIALS_OUTPUT,
    )
    parser.add_argument(
        "--needle-physics-output",
        type=Path,
        default=DEFAULT_NEEDLE_PHYSICS_OUTPUT,
    )
    parser.add_argument(
        "--needle-physx-output",
        type=Path,
        default=DEFAULT_NEEDLE_PHYSX_OUTPUT,
    )
    parser.add_argument(
        "--usdcat",
        default=shutil.which("usdcat") or "usdcat",
    )
    parser.add_argument(
        "--segment-count",
        type=int,
        help="Author a native interactive LOD with this many physical segments while preserving length and diameter.",
    )
    args = parser.parse_args()
    profile = load_profile(args.profile)
    if args.segment_count is not None:
        if args.segment_count < 16:
            parser.error("--segment-count must be at least 16")
        profile = json.loads(json.dumps(profile))
        profile["geometry"]["segment_count"] = args.segment_count
        profile["geometry"]["segment_spacing_m"] = (
            float(profile["geometry"]["length_m"]) / args.segment_count
        )
    needle_profile = load_needle_profile(args.needle_profile)
    output = args.output.expanduser().resolve()
    base_output = args.base_output.expanduser().resolve()
    geometry_output = args.geometry_output.expanduser().resolve()
    materials_output = args.materials_output.expanduser().resolve()
    texture_output = args.texture_output.expanduser().resolve()
    physics_output = args.physics_output.expanduser().resolve()
    physx_output = args.physx_output.expanduser().resolve()
    needle_output = args.needle_output.expanduser().resolve()
    needle_base_output = args.needle_base_output.expanduser().resolve()
    needle_geometry_output = args.needle_geometry_output.expanduser().resolve()
    needle_materials_output = args.needle_materials_output.expanduser().resolve()
    needle_physics_output = args.needle_physics_output.expanduser().resolve()
    needle_physx_output = args.needle_physx_output.expanduser().resolve()
    for layer_output in (
        output,
        base_output,
        geometry_output,
        materials_output,
        texture_output,
        physics_output,
        physx_output,
    ):
        layer_output.parent.mkdir(parents=True, exist_ok=True)
    suture_base_reference = anchored_reference_path(Path(os.path.relpath(base_output, start=output.parent)).as_posix())
    suture_geometry_reference = anchored_reference_path(
        Path(os.path.relpath(geometry_output, start=base_output.parent)).as_posix()
    )
    suture_materials_reference = anchored_reference_path(
        Path(os.path.relpath(materials_output, start=base_output.parent)).as_posix()
    )
    texture_reference = anchored_reference_path(
        Path(
            os.path.relpath(
                texture_output,
                start=materials_output.parent,
            )
        ).as_posix()
    )
    suture_physics_reference = anchored_reference_path(
        Path(os.path.relpath(physics_output, start=output.parent)).as_posix()
    )
    suture_physx_reference = anchored_reference_path(
        Path(os.path.relpath(physx_output, start=output.parent)).as_posix()
    )
    suture_neutral_physics_reference = anchored_reference_path(
        Path(
            os.path.relpath(
                physics_output,
                start=physx_output.parent,
            )
        ).as_posix()
    )
    (
        suture_entry_text,
        suture_base_text,
        suture_geometry_text,
        suture_materials_text,
        suture_physics_text,
        suture_physx_text,
    ) = author(
        profile,
        base_reference=suture_base_reference,
        geometry_sublayer_reference=suture_geometry_reference,
        materials_sublayer_reference=suture_materials_reference,
        physics_payload_reference=suture_physics_reference,
        physx_payload_reference=suture_physx_reference,
        neutral_physics_sublayer_reference=suture_neutral_physics_reference,
        texture_reference=texture_reference,
    )
    suture_temporary = output.with_suffix(output.suffix + ".tmp")
    suture_base_temporary = base_output.with_suffix(base_output.suffix + ".tmp")
    suture_materials_temporary = materials_output.with_suffix(materials_output.suffix + ".tmp")
    suture_texture_temporary = texture_output.with_suffix(texture_output.suffix + ".tmp")
    suture_physics_temporary = physics_output.with_suffix(physics_output.suffix + ".tmp")
    suture_physx_temporary = physx_output.with_suffix(physx_output.suffix + ".tmp")
    suture_temporary.write_text(suture_entry_text, encoding="utf-8")
    suture_base_temporary.write_text(suture_base_text, encoding="utf-8")
    suture_materials_temporary.write_text(suture_materials_text, encoding="utf-8")
    suture_texture_temporary.write_bytes(encode_suture_material_texture_png(build_suture_material_texture(profile)))
    suture_physics_temporary.write_text(suture_physics_text, encoding="utf-8")
    suture_physx_temporary.write_text(suture_physx_text, encoding="utf-8")
    write_usdc(suture_geometry_text, geometry_output, args.usdcat)
    suture_texture_temporary.replace(texture_output)
    suture_materials_temporary.replace(materials_output)
    suture_physics_temporary.replace(physics_output)
    suture_physx_temporary.replace(physx_output)
    suture_base_temporary.replace(base_output)
    suture_temporary.replace(output)
    needle_output.parent.mkdir(parents=True, exist_ok=True)
    needle_base_output.parent.mkdir(parents=True, exist_ok=True)
    needle_geometry_output.parent.mkdir(parents=True, exist_ok=True)
    needle_materials_output.parent.mkdir(parents=True, exist_ok=True)
    needle_physics_output.parent.mkdir(parents=True, exist_ok=True)
    needle_physx_output.parent.mkdir(parents=True, exist_ok=True)
    needle_temporary = needle_output.with_suffix(needle_output.suffix + ".tmp")
    needle_base_temporary = needle_base_output.with_suffix(needle_base_output.suffix + ".tmp")
    needle_materials_temporary = needle_materials_output.with_suffix(needle_materials_output.suffix + ".tmp")
    needle_physics_temporary = needle_physics_output.with_suffix(needle_physics_output.suffix + ".tmp")
    needle_physx_temporary = needle_physx_output.with_suffix(needle_physx_output.suffix + ".tmp")
    suture_reference = anchored_reference_path(Path(os.path.relpath(output, start=needle_output.parent)).as_posix())
    base_reference = anchored_reference_path(
        Path(os.path.relpath(needle_base_output, start=needle_output.parent)).as_posix()
    )
    geometry_sublayer_reference = anchored_reference_path(
        Path(
            os.path.relpath(
                needle_geometry_output,
                start=needle_base_output.parent,
            )
        ).as_posix()
    )
    materials_sublayer_reference = anchored_reference_path(
        Path(
            os.path.relpath(
                needle_materials_output,
                start=needle_base_output.parent,
            )
        ).as_posix()
    )
    physics_payload_reference = anchored_reference_path(
        Path(
            os.path.relpath(
                needle_physics_output,
                start=needle_output.parent,
            )
        ).as_posix()
    )
    physx_payload_reference = anchored_reference_path(
        Path(
            os.path.relpath(
                needle_physx_output,
                start=needle_output.parent,
            )
        ).as_posix()
    )
    neutral_physics_sublayer_reference = anchored_reference_path(
        Path(
            os.path.relpath(
                needle_physics_output,
                start=needle_physx_output.parent,
            )
        ).as_posix()
    )
    (
        needle_entry_text,
        needle_base_text,
        needle_geometry_text,
        needle_materials_text,
        needle_physics_text,
        needle_physx_text,
    ) = author_dr_anmar_needle(
        profile,
        needle_profile,
        base_reference=base_reference,
        suture_reference=suture_reference,
        geometry_sublayer_reference=geometry_sublayer_reference,
        materials_sublayer_reference=materials_sublayer_reference,
        physics_payload_reference=physics_payload_reference,
        physx_payload_reference=physx_payload_reference,
        neutral_physics_sublayer_reference=neutral_physics_sublayer_reference,
    )
    needle_temporary.write_text(needle_entry_text, encoding="utf-8")
    needle_base_temporary.write_text(needle_base_text, encoding="utf-8")
    needle_materials_temporary.write_text(needle_materials_text, encoding="utf-8")
    needle_physics_temporary.write_text(needle_physics_text, encoding="utf-8")
    needle_physx_temporary.write_text(needle_physx_text, encoding="utf-8")
    write_usdc(needle_geometry_text, needle_geometry_output, args.usdcat)
    needle_materials_temporary.replace(needle_materials_output)
    needle_physics_temporary.replace(needle_physics_output)
    needle_physx_temporary.replace(needle_physx_output)
    needle_base_temporary.replace(needle_base_output)
    needle_temporary.replace(needle_output)
    derived = derive(profile)
    segment_mass_properties = suture_segment_mass_properties(
        profile,
        derived=derived,
    )
    interface_mass_properties = suture_interface_mass_properties(
        profile,
        derived=derived,
    )
    suture_visual_vertex_count = 0
    suture_visual_face_count = 0
    suture_visual_texcoord_count = 0
    suture_visual_texcoord_index_count = 0
    suture_visual_tangent_frame_value_count = 0
    suture_visual_tangent_frame_index_count = 0
    suture_visual_minimum_radius_ratio = math.inf
    suture_visual_maximum_radius_ratio = 0.0
    suture_minimum_visual_collision_margin_m = math.inf
    suture_interface_visual_mesh = build_suture_interface_visual_mesh(
        profile,
        derived=derived,
    )
    suture_interface_radius = float(profile["swage"]["needle_end_diameter_m"]) / 2.0
    suture_interface_minimum_collision_margin_m = min(
        capsule_point_containment_margin(
            point,
            radius_m=suture_interface_radius,
            cylinder_height_m=derived.segment_spacing_m,
        )
        for point in suture_interface_visual_mesh.points
    )
    for segment_index in range(derived.segment_count):
        collision_radius = suture_segment_collision_radius(
            profile,
            segment_index,
            derived=derived,
        )
        suture_visual_mesh = build_suture_visual_mesh(
            profile,
            segment_index,
            collision_radius_m=collision_radius,
            derived=derived,
        )
        suture_visual_vertex_count += len(suture_visual_mesh.points)
        suture_visual_face_count += len(suture_visual_mesh.face_vertex_counts)
        suture_visual_texcoord_count += len(suture_visual_mesh.texcoords)
        suture_visual_texcoord_index_count += len(suture_visual_mesh.texcoord_indices)
        suture_visual_tangent_frame_value_count += len(suture_visual_mesh.normals)
        suture_visual_tangent_frame_index_count += len(suture_visual_mesh.tangent_frame_indices)
        suture_visual_minimum_radius_ratio = min(
            suture_visual_minimum_radius_ratio,
            suture_visual_mesh.minimum_radius_m / collision_radius,
        )
        suture_visual_maximum_radius_ratio = max(
            suture_visual_maximum_radius_ratio,
            suture_visual_mesh.maximum_radius_m / collision_radius,
        )
        suture_minimum_visual_collision_margin_m = min(
            suture_minimum_visual_collision_margin_m,
            *(
                capsule_point_containment_margin(
                    point,
                    radius_m=collision_radius,
                    cylinder_height_m=derived.segment_spacing_m,
                )
                for point in suture_visual_mesh.points
            ),
        )
    derived_needle = derive_needle(needle_profile)
    collision_capsules = build_needle_collision_capsules(needle_profile)
    needle_mesh = build_needle_mesh(needle_profile)
    report = {
        "schema": "dr.anmar.suture-asset-report.v15",
        "profile": portable_path(args.profile),
        "asset": portable_path(output),
        "asset_sha256": sha256(output),
        "suture_asset_version": profile["version"],
        "suture_entry_bytes": output.stat().st_size,
        "suture_base": portable_path(base_output),
        "suture_base_sha256": sha256(base_output),
        "suture_geometry": portable_path(geometry_output),
        "suture_geometry_format": "usdc",
        "suture_geometry_sha256": sha256(geometry_output),
        "suture_geometry_bytes": geometry_output.stat().st_size,
        "suture_interface_visual_mesh_schema": profile["geometry"]["visual_representation"][
            "needle_interface_visual_schema"
        ],
        "suture_interface_visual_vertex_count": len(suture_interface_visual_mesh.points),
        "suture_interface_visual_face_count": len(suture_interface_visual_mesh.face_vertex_counts),
        "suture_interface_minimum_collision_margin_m": suture_interface_minimum_collision_margin_m,
        "suture_visual_mesh_schema": profile["geometry"]["visual_representation"]["visual_schema"],
        "suture_visual_vertices_per_segment": suture_visual_vertex_count // derived.segment_count,
        "suture_visual_faces_per_segment": suture_visual_face_count // derived.segment_count,
        "suture_visual_total_vertices": suture_visual_vertex_count,
        "suture_visual_total_faces": suture_visual_face_count,
        "suture_visual_texcoords_per_segment": suture_visual_texcoord_count // derived.segment_count,
        "suture_visual_total_texcoords": suture_visual_texcoord_count,
        "suture_visual_texcoord_indices_per_segment": suture_visual_texcoord_index_count // derived.segment_count,
        "suture_visual_total_texcoord_indices": suture_visual_texcoord_index_count,
        "suture_visual_tangent_frame_values_per_segment": (
            suture_visual_tangent_frame_value_count // derived.segment_count
        ),
        "suture_visual_tangent_frame_indices_per_segment": (
            suture_visual_tangent_frame_index_count // derived.segment_count
        ),
        "suture_visual_total_tangent_frame_values_per_channel": suture_visual_tangent_frame_value_count,
        "suture_visual_total_tangent_frame_indices_per_channel": suture_visual_tangent_frame_index_count,
        "suture_visual_minimum_radius_ratio": suture_visual_minimum_radius_ratio,
        "suture_visual_maximum_radius_ratio": suture_visual_maximum_radius_ratio,
        "suture_collider_cylinder_height_m": derived.segment_spacing_m,
        "suture_minimum_visual_collision_margin_m": suture_minimum_visual_collision_margin_m,
        "suture_mass_property_contract": profile["geometry"]["mass_properties"],
        "suture_segment_mass_properties": {
            "mass_kg": segment_mass_properties.mass_kg,
            "center_of_mass_m": segment_mass_properties.center_of_mass_m,
            "diagonal_inertia_kg_m2": segment_mass_properties.diagonal_inertia_kg_m2,
            "principal_axes_wxyz": segment_mass_properties.principal_axes_wxyz,
        },
        "suture_interface_mass_properties": {
            "mass_kg": interface_mass_properties.mass_kg,
            "center_of_mass_m": interface_mass_properties.center_of_mass_m,
            "diagonal_inertia_kg_m2": interface_mass_properties.diagonal_inertia_kg_m2,
            "principal_axes_wxyz": interface_mass_properties.principal_axes_wxyz,
        },
        "suture_render_collision_separation": profile["geometry"]["visual_representation"],
        "suture_appearance": profile["appearance"],
        "suture_normal_roughness_texture": portable_path(texture_output),
        "suture_normal_roughness_texture_sha256": sha256(texture_output),
        "suture_normal_roughness_texture_bytes": texture_output.stat().st_size,
        "suture_materials": portable_path(materials_output),
        "suture_materials_sha256": sha256(materials_output),
        "suture_physics": portable_path(physics_output),
        "suture_physics_sha256": sha256(physics_output),
        "suture_physx": portable_path(physx_output),
        "suture_physx_sha256": sha256(physx_output),
        "suture_layer_contract": profile["asset_structure"],
        "suture_model_identity": profile["asset_structure"]["model_identity"],
        "dr_anmar_needle_name": DR_ANMAR_NEEDLE_NAME,
        "dr_anmar_needle_asset_id": DR_ANMAR_NEEDLE_ASSET_ID,
        "dr_anmar_needle_asset_version": DR_ANMAR_NEEDLE_ASSET_VERSION,
        "dr_anmar_needle": portable_path(needle_output),
        "dr_anmar_needle_sha256": sha256(needle_output),
        "dr_anmar_needle_entry_bytes": needle_output.stat().st_size,
        "dr_anmar_needle_base": portable_path(needle_base_output),
        "dr_anmar_needle_base_sha256": sha256(needle_base_output),
        "dr_anmar_needle_geometry": portable_path(needle_geometry_output),
        "dr_anmar_needle_geometry_format": "usdc",
        "dr_anmar_needle_geometry_sha256": sha256(needle_geometry_output),
        "dr_anmar_needle_geometry_bytes": needle_geometry_output.stat().st_size,
        "dr_anmar_needle_materials": portable_path(needle_materials_output),
        "dr_anmar_needle_materials_sha256": sha256(needle_materials_output),
        "dr_anmar_needle_physics": portable_path(needle_physics_output),
        "dr_anmar_needle_physics_sha256": sha256(needle_physics_output),
        "dr_anmar_needle_physx": portable_path(needle_physx_output),
        "dr_anmar_needle_physx_sha256": sha256(needle_physx_output),
        "dr_anmar_needle_layer_contract": needle_profile["construction"]["layer_organization"],
        "dr_anmar_needle_model_identity": needle_profile["construction"]["layer_organization"]["model_identity"],
        "needle_profile": portable_path(args.needle_profile),
        "needle_profile_id": needle_profile["id"],
        "needle_geometry_source": needle_profile["construction"]["geometry_source"],
        "needle_arc_length_m": derived_needle.arc_length_m,
        "needle_curvature_radius_m": derived_needle.curvature_radius_m,
        "needle_body_diameter_m": derived_needle.body_radius_m * 2.0,
        "needle_mass_kg": derived_needle.mass_kg,
        "needle_mass_property_integration_slices": derived_needle.mass_properties.integration_slices,
        "needle_center_of_mass_m": list(derived_needle.mass_properties.center_of_mass_m),
        "needle_diagonal_inertia_kg_m2": list(derived_needle.mass_properties.diagonal_inertia_kg_m2),
        "needle_principal_axes_wxyz": list(derived_needle.mass_properties.principal_axes_wxyz),
        "needle_visual_vertex_count": derived_needle.visual_vertex_count,
        "needle_visual_normal_quality": needle_mesh_normal_quality(
            needle_profile,
            needle_mesh,
        ),
        "needle_collision_capsule_count": derived_needle.collision_capsule_count,
        "needle_collision_contract": needle_profile["construction"]["collision_contract"],
        "needle_material_organization": needle_profile["material"]["usd_organization"],
        "needle_render_collision_separation": needle_profile["construction"]["collision_contract"][
            "render_collision_separation"
        ],
        "needle_collision_guide_purpose_count": derived_needle.collision_capsule_count,
        "needle_collision_invisible_count": derived_needle.collision_capsule_count,
        "needle_collision_physics_material_binding_count": derived_needle.collision_capsule_count,
        "needle_collision_max_curvature_sagitta_m": max(capsule.curvature_sagitta_m for capsule in collision_capsules),
        "needle_collision_visual_seam_margin_m": max(capsule.visual_seam_margin_m for capsule in collision_capsules),
        "needle_collision_max_chord_length_error_m": max(
            abs(capsule.cylinder_height_m - capsule.chord_length_m) for capsule in collision_capsules
        ),
        "needle_contact_offset_policy": needle_profile["construction"]["collision_contract"]["contact_offsets"],
        "needle_contact_offset_range_m": [
            min(capsule.contact_offset_m for capsule in collision_capsules),
            max(capsule.contact_offset_m for capsule in collision_capsules),
        ],
        "needle_rest_offset_range_m": [
            min(capsule.rest_offset_m for capsule in collision_capsules),
            max(capsule.rest_offset_m for capsule in collision_capsules),
        ],
        "needle_collision_visual_mesh_coverage": needle_mesh_collision_coverage(
            needle_profile,
            needle_mesh,
        ),
        "needle_swage_anchor_m": list(derived_needle.swage_anchor_m),
        "needle_sim_to_real_gap_count": len(needle_profile["sim_to_real"]["gaps"]),
        "suture_sim_to_real_gap_count": len(profile["sim_to_real"]["gaps"]),
        "swage_connection": "fixed_needle_to_interface_then_breakable_pullout_joint",
        "representation": (
            "braided_visual_meshes_on_rigid_xforms_with_hidden_capsule_colliders_and_breakable_d6_cosserat_joints"
        ),
        "segment_count": derived.segment_count,
        "segment_count_override": args.segment_count,
        "joint_count": derived.segment_count,
        "diameter_m": derived.diameter_m,
        "length_m": derived.length_m,
        "mass_kg": derived.mass_kg,
        "straight_failure_load_n": derived.straight_failure_load_n,
        "knot_failure_load_n": derived.knot_failure_load_n,
        "runtime_material_history_controller": None,
        "runtime_physics_authority": "OpenUSD_PhysX",
        "runtime_observation_source": profile["runtime_detection"]["observation_source"],
        "runtime_self_contact_broadphase": profile["runtime_detection"]["self_contact_broadphase"],
        "clinical_validation": False,
        "independent_from_current_thread": True,
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
