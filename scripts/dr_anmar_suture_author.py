#!/usr/bin/env python3
"""Author the independent Dr.Anmar 4-0 suture as an OpenUSD physics asset.

The asset is a high-resolution discrete Cosserat rod: every visible capsule is
also its collision body, while D6 joints independently model axial stretch,
bending, torsion, damping, and overload breakage.  No NVIDIA Rope.usd data is
read or referenced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from dr_anmar_needle_model import (
    DEFAULT_NEEDLE_PROFILE_PATH,
    build_needle_mesh,
    centerline_at,
    derive_needle,
    load_needle_profile,
    radius_at_distance,
)
from dr_anmar_suture_integration import (
    DR_ANMAR_NEEDLE_ASSET_ID,
    DR_ANMAR_NEEDLE_ASSET_PATH,
    DR_ANMAR_NEEDLE_ASSET_VERSION,
    DR_ANMAR_NEEDLE_NAME,
    DR_ANMAR_NEEDLE_ROOT_PRIM,
    SUTURE_NEEDLE_INTERFACE_CENTER_M,
)
from dr_anmar_suture_model import DEFAULT_PROFILE_PATH, derive, load_profile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "assets/dr_anmar/suture/DrAnmarSuture4_0.usda"
)
DEFAULT_NEEDLE_OUTPUT = DR_ANMAR_NEEDLE_ASSET_PATH


def usd_float(value: float) -> str:
    return f"{value:.12g}"


def usd_vec(values: tuple[float, float, float]) -> str:
    return "(" + ", ".join(usd_float(value) for value in values) + ")"


def usd_quat(values: tuple[float, float, float, float]) -> str:
    return (
        "("
        + usd_float(values[0])
        + ", "
        + ", ".join(usd_float(value) for value in values[1:])
        + ")"
    )


def indent(text: str, spaces: int = 4) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line else "" for line in text.splitlines())


def capsule_block(
    *,
    name: str,
    x_m: float,
    radius_m: float,
    cylinder_height_m: float,
    mass_kg: float,
    color: tuple[float, float, float],
    material_path: str,
    kinematic: bool,
    filtered_pair: str | None,
    profile: dict[str, Any],
) -> str:
    material = profile["material"]
    schemas = [
        '"PhysicsCollisionAPI"',
        '"PhysicsRigidBodyAPI"',
        '"PhysicsMassAPI"',
        '"PhysxRigidBodyAPI"',
        '"MaterialBindingAPI"',
    ]
    if filtered_pair:
        schemas.append('"PhysicsFilteredPairsAPI"')
    total_half_length = cylinder_height_m / 2.0 + radius_m
    filtered = (
        f"\n    rel physics:filteredPairs = <{filtered_pair}>" if filtered_pair else ""
    )
    return f'''def Capsule "{name}" (
    prepend apiSchemas = [{", ".join(schemas)}]
)
{{
    uniform token axis = "X"
    float height = {usd_float(cylinder_height_m)}
    float radius = {usd_float(radius_m)}
    float3[] extent = [{usd_vec((-total_half_length, -radius_m, -radius_m))}, {usd_vec((total_half_length, radius_m, radius_m))}]
    color3f[] primvars:displayColor = [{usd_vec(color)}]
    rel material:binding = <{material_path}>
    bool physics:collisionEnabled = true
    bool physics:rigidBodyEnabled = true
    bool physics:kinematicEnabled = {"true" if kinematic else "false"}
    float physics:mass = {usd_float(mass_kg)}
    bool physxRigidBody:enableCCD = true
    float physxRigidBody:linearDamping = {usd_float(float(material["linear_velocity_damping"]))}
    float physxRigidBody:angularDamping = {usd_float(float(material["angular_velocity_damping"]))}
    float physxRigidBody:maxDepenetrationVelocity = 0.25
    double3 xformOp:translate = {usd_vec((x_m, 0.0, 0.0))}
    uniform token[] xformOpOrder = ["xformOp:translate"]{filtered}
}}'''


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
    return f'''def PhysicsJoint "{name}" (
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
}}'''


def author(profile: dict[str, Any]) -> str:
    derived = derive(profile)
    geometry = profile["geometry"]
    tension = profile["tension"]
    contact = profile["contact"]
    swage = profile["swage"]
    color = tuple(float(value) for value in geometry["color_rgb"])
    spacing = derived.segment_spacing_m
    base_radius = derived.radius_m
    root = "/DrAnmarSuture4_0"
    material_path = f"{root}/Materials/SutureMaterial"
    steel_path = f"{root}/Materials/SwageSteel"
    blocks: list[str] = []
    blocks.append(
        f"""def Scope "Materials"
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
}}"""
    )
    swage_radius = float(swage["needle_end_diameter_m"]) / 2.0
    blocks.append(
        capsule_block(
            name="NeedleInterface",
            x_m=-spacing / 2.0,
            radius_m=swage_radius,
            cylinder_height_m=max(spacing - 2.0 * swage_radius, spacing * 0.05),
            mass_kg=max(derived.segment_mass_kg * 8.0, 1e-7),
            color=(0.58, 0.61, 0.66),
            material_path=steel_path,
            kinematic=True,
            filtered_pair=f"{root}/Segments/S0000",
            profile=profile,
        ).replace('def Capsule "NeedleInterface"', 'def Capsule "NeedleInterface"')
    )
    segment_blocks: list[str] = []
    modulation = float(geometry["surface_radius_modulation_fraction"])
    modulation_period = int(geometry["surface_modulation_period_segments"])
    for index in range(derived.segment_count):
        swage_fraction = clamp01(1.0 - index / max(1, derived.swage_segment_count - 1))
        taper_radius = base_radius + (swage_radius - base_radius) * swage_fraction
        roughness = 1.0 + modulation * math.sin(
            2.0 * math.pi * index / modulation_period
        )
        radius = taper_radius * roughness
        segment_color = tuple(
            max(0.0, min(1.0, component * (0.92 + 0.08 * ((index % 3) / 2.0))))
            for component in color
        )
        previous_path = (
            f"{root}/NeedleInterface"
            if index == 0
            else f"{root}/Segments/S{index - 1:04d}"
        )
        segment_blocks.append(
            capsule_block(
                name=f"S{index:04d}",
                x_m=(index + 0.5) * spacing,
                radius_m=radius,
                cylinder_height_m=max(spacing - 2.0 * radius, spacing * 0.05),
                mass_kg=derived.segment_mass_kg,
                color=segment_color,
                material_path=material_path,
                kinematic=False,
                filtered_pair=previous_path,
                profile=profile,
            )
        )
    blocks.append(
        'def Scope "Segments"\n{\n' + indent("\n\n".join(segment_blocks)) + "\n}"
    )
    joint_blocks: list[str] = []
    extension_limit = spacing * float(tension["joint_extension_limit_fraction"])
    break_torque = (
        derived.straight_failure_load_n
        * derived.radius_m
        * float(profile["knot"]["nominal_strength_efficiency"])
    )
    for joint_index in range(derived.segment_count):
        is_swage_joint = joint_index == 0
        segment_index = joint_index
        body0 = (
            f"{root}/NeedleInterface"
            if is_swage_joint
            else f"{root}/Segments/S{segment_index - 1:04d}"
        )
        body1 = f"{root}/Segments/S{segment_index:04d}"
        swage_fraction = clamp01(
            1.0 - segment_index / max(1, derived.swage_segment_count - 1)
        )
        axial_multiplier = lerp1(
            1.0, float(swage["axial_stiffness_multiplier"]), swage_fraction
        )
        bend_multiplier = lerp1(
            1.0, float(swage["bending_stiffness_multiplier"]), swage_fraction
        )
        failure_load = (
            float(swage["pullout_force_n_seed"])
            if is_swage_joint
            else derived.straight_failure_load_n
        )
        joint_blocks.append(
            joint_block(
                name=f"J{joint_index:04d}",
                body0=body0,
                body1=body1,
                half_spacing_m=spacing / 2.0,
                extension_limit_m=extension_limit,
                axial_stiffness_n_m=derived.axial_joint_stiffness_n_m
                * axial_multiplier,
                axial_damping_n_s_m=derived.axial_joint_damping_n_s_m
                * math.sqrt(axial_multiplier),
                bend_stiffness_n_m_rad=derived.bend_joint_stiffness_n_m_rad
                * bend_multiplier,
                bend_damping_n_m_s_rad=derived.bend_joint_damping_n_m_s_rad
                * math.sqrt(bend_multiplier),
                twist_stiffness_n_m_rad=derived.twist_joint_stiffness_n_m_rad
                * bend_multiplier,
                break_force_n=failure_load,
                break_torque_n_m=break_torque,
                swage_fraction=swage_fraction,
            )
        )
    blocks.append('def Scope "Joints"\n{\n' + indent("\n\n".join(joint_blocks)) + "\n}")
    custom_data = {
        "drAnmarClinicalValidation": False,
        "drAnmarCanonicalAssetPackage": "assets/dr_anmar",
        "drAnmarIndependentAsset": True,
        "drAnmarProfileId": profile["id"],
        "drAnmarRepresentation": "discrete_cosserat_rod",
        "drAnmarStatus": profile["status"],
    }
    custom_lines = "\n".join(
        f"        {'bool' if isinstance(value, bool) else 'string'} {key} = "
        + (
            ("true" if value else "false")
            if isinstance(value, bool)
            else json.dumps(str(value))
        )
        for key, value in custom_data.items()
    )
    body = "\n\n".join(indent(block) for block in blocks)
    return f"""#usda 1.0
(
    defaultPrim = "DrAnmarSuture4_0"
    doc = "Independent research-grade 4-0 braided surgical-suture asset; not clinically validated."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "DrAnmarSuture4_0" (
    customData = {{
{custom_lines}
    }}
)
{{
{body}
}}
"""


def author_dr_anmar_needle(
    suture_profile: dict[str, Any],
    needle_profile: dict[str, Any],
    *,
    suture_reference: str,
) -> str:
    """Author independent needle geometry and attach the Dr.Anmar suture."""

    derived_needle = derive_needle(needle_profile)
    mesh = build_needle_mesh(needle_profile)
    contact = needle_profile["contact"]
    solver = needle_profile["solver"]
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
    suture_translation = tuple(
        swage_anchor[index] - rotated_interface_center[index]
        for index in range(3)
    )
    root = f"/{DR_ANMAR_NEEDLE_ROOT_PRIM}"
    steel_material_path = f"{root}/Materials/NeedleSteel"
    mesh_points = ",\n            ".join(usd_vec(point) for point in mesh.points)
    face_counts = ", ".join(str(value) for value in mesh.face_vertex_counts)
    face_indices = ", ".join(str(value) for value in mesh.face_vertex_indices)
    collision_blocks: list[str] = []
    collision_count = derived_needle.collision_capsule_count
    for index in range(collision_count):
        left_fraction = index / collision_count
        right_fraction = (index + 1) / collision_count
        middle_fraction = (index + 0.5) / collision_count
        left, _left_tangent = centerline_at(needle_profile, left_fraction)
        right, _right_tangent = centerline_at(needle_profile, right_fraction)
        middle, tangent = centerline_at(needle_profile, middle_fraction)
        chord_length = math.dist(left, right)
        radius = radius_at_distance(
            needle_profile,
            middle_fraction * derived_needle.arc_length_m,
        )
        yaw = math.atan2(tangent[1], tangent[0])
        orientation = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
        collision_blocks.append(
            f'''def Capsule "C{index:03d}" (
    prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]
)
{{
    uniform token axis = "X"
    float height = {usd_float(chord_length)}
    float radius = {usd_float(radius)}
    rel material:binding = <{steel_material_path}>
    bool physics:collisionEnabled = true
    quatf xformOp:orient = {usd_quat(orientation)}
    double3 xformOp:translate = {usd_vec(middle)}
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
}}'''
        )
    collisions = indent("\n\n".join(collision_blocks), 8)
    sim_to_real_gap_count = len(needle_profile["sim_to_real"]["gaps"])
    implemented_randomization_count = len(
        needle_profile["sim_to_real"][
            "implemented_randomization_on_episode_reset"
        ]
    )
    return f"""#usda 1.0
(
    defaultPrim = "{DR_ANMAR_NEEDLE_ROOT_PRIM}"
    doc = "{DR_ANMAR_NEEDLE_NAME}: independently generated research-grade curved taper-point needle with factory-swaged Dr.Anmar 4-0 suture; not clinically validated."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{DR_ANMAR_NEEDLE_ROOT_PRIM}" (
    customData = {{
        string drAnmarAssetId = "{DR_ANMAR_NEEDLE_ASSET_ID}"
        string drAnmarAssetName = "{DR_ANMAR_NEEDLE_NAME}"
        string drAnmarAssetVersion = "{DR_ANMAR_NEEDLE_ASSET_VERSION}"
        string drAnmarAuthorship = "Independent Dr.Anmar geometry, collision, instrument composition and suture physics"
        bool drAnmarClinicalValidation = false
        string drAnmarGeometrySource = "independently_generated_parametric_geometry"
        string drAnmarNeedleProfileId = "{needle_profile["id"]}"
        string drAnmarRepresentation = "high_resolution_mesh_with_compound_capsule_collision"
        int drAnmarResetRandomizationCount = {implemented_randomization_count}
        int drAnmarSimToRealGapCount = {sim_to_real_gap_count}
        string drAnmarSutureProfileId = "{suture_profile["id"]}"
        string drAnmarSuturePhysicsProvenance = "Independent Dr.Anmar 4-0 discrete Cosserat rod"
        string drAnmarSwageConnection = "fixed_needle_to_interface_then_breakable_pullout_joint"
        string drAnmarStatus = "{needle_profile["status"]}"
    }}
)
{{
    def Scope "Materials"
    {{
        def Material "NeedleSteel" (
            prepend apiSchemas = ["PhysicsMaterialAPI"]
        )
        {{
            float physics:staticFriction = {usd_float(float(contact["static_friction_seed"]))}
            float physics:dynamicFriction = {usd_float(float(contact["dynamic_friction_seed"]))}
            float physics:restitution = {usd_float(float(contact["restitution_seed"]))}

            def Shader "PreviewSurface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.53, 0.58, 0.64)
                float inputs:metallic = {usd_float(float(needle_profile["appearance"]["metallic_seed"]))}
                float inputs:roughness = {usd_float(float(needle_profile["appearance"]["roughness_seed"]))}
                token outputs:surface
            }}
            token outputs:surface.connect = </{DR_ANMAR_NEEDLE_ROOT_PRIM}/Materials/NeedleSteel/PreviewSurface.outputs:surface>
        }}
    }}

    def Xform "Needle" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
    )
    {{
        bool physics:rigidBodyEnabled = true
        bool physics:kinematicEnabled = false
        float physics:mass = {usd_float(derived_needle.mass_kg)}
        bool physxRigidBody:enableCCD = {"true" if solver["ccd"] else "false"}
        int physxRigidBody:solverPositionIterationCount = {int(solver["position_iterations"])}
        int physxRigidBody:solverVelocityIterationCount = {int(solver["velocity_iterations"])}
        float physxRigidBody:maxDepenetrationVelocity = {usd_float(float(solver["max_depenetration_velocity_m_s"]))}

        def Mesh "Visual" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {{
            uniform bool doubleSided = false
            float3[] extent = [{usd_vec(mesh.extent_min)}, {usd_vec(mesh.extent_max)}]
            int[] faceVertexCounts = [{face_counts}]
            int[] faceVertexIndices = [{face_indices}]
            point3f[] points = [
            {mesh_points}
            ]
            uniform token subdivisionScheme = "none"
            rel material:binding = <{steel_material_path}>
        }}

        def Scope "Collision"
        {{
{collisions}
        }}

        def Xform "SutureAnchor"
        {{
            quatf xformOp:orient = {usd_quat(swage_orientation)}
            double3 xformOp:translate = {usd_vec(swage_anchor)}
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
        }}
    }}

    def Xform "Suture" (
        prepend references = @{suture_reference}@
    )
    {{
        double3 xformOp:translate = {usd_vec(suture_translation)}
        quatf xformOp:orient = {usd_quat(swage_orientation)}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]

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
        "--needle-output",
        "--assembly-output",
        dest="needle_output",
        type=Path,
        default=DEFAULT_NEEDLE_OUTPUT,
    )
    args = parser.parse_args()
    profile = load_profile(args.profile)
    needle_profile = load_needle_profile(args.needle_profile)
    output = args.output.expanduser().resolve()
    needle_output = args.needle_output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(author(profile), encoding="utf-8")
    temporary.replace(output)
    needle_output.parent.mkdir(parents=True, exist_ok=True)
    needle_temporary = needle_output.with_suffix(needle_output.suffix + ".tmp")
    suture_reference = Path(
        os.path.relpath(output, start=needle_output.parent)
    ).as_posix()
    needle_temporary.write_text(
        author_dr_anmar_needle(
            profile,
            needle_profile,
            suture_reference=suture_reference,
        ),
        encoding="utf-8",
    )
    needle_temporary.replace(needle_output)
    derived = derive(profile)
    derived_needle = derive_needle(needle_profile)
    report = {
        "schema": "dr.anmar.suture-asset-report.v2",
        "profile": str(args.profile.resolve()),
        "asset": str(output),
        "asset_sha256": sha256(output),
        "dr_anmar_needle_name": DR_ANMAR_NEEDLE_NAME,
        "dr_anmar_needle_asset_id": DR_ANMAR_NEEDLE_ASSET_ID,
        "dr_anmar_needle_asset_version": DR_ANMAR_NEEDLE_ASSET_VERSION,
        "dr_anmar_needle": str(needle_output),
        "dr_anmar_needle_sha256": sha256(needle_output),
        "needle_profile": str(args.needle_profile.resolve()),
        "needle_profile_id": needle_profile["id"],
        "needle_geometry_source": needle_profile["construction"]["geometry_source"],
        "needle_arc_length_m": derived_needle.arc_length_m,
        "needle_curvature_radius_m": derived_needle.curvature_radius_m,
        "needle_body_diameter_m": derived_needle.body_radius_m * 2.0,
        "needle_mass_kg": derived_needle.mass_kg,
        "needle_visual_vertex_count": derived_needle.visual_vertex_count,
        "needle_collision_capsule_count": derived_needle.collision_capsule_count,
        "needle_swage_anchor_m": list(derived_needle.swage_anchor_m),
        "sim_to_real_gap_count": len(needle_profile["sim_to_real"]["gaps"]),
        "swage_connection": "fixed_needle_to_interface_then_breakable_pullout_joint",
        "representation": "visible_collision_capsules_with_breakable_d6_cosserat_joints",
        "segment_count": derived.segment_count,
        "joint_count": derived.segment_count,
        "diameter_m": derived.diameter_m,
        "length_m": derived.length_m,
        "mass_kg": derived.mass_kg,
        "straight_failure_load_n": derived.straight_failure_load_n,
        "knot_failure_load_n": derived.knot_failure_load_n,
        "clinical_validation": False,
        "independent_from_current_thread": True,
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
