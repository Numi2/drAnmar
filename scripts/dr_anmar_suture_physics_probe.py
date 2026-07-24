#!/usr/bin/env python3
"""Headless native-PhysX smoke probe for the independent Dr.Anmar suture."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from dr_anmar_needle_model import build_needle_collision_capsules, build_needle_mesh, derive_needle, load_needle_profile
from dr_anmar_suture_integration import DR_ANMAR_NEEDLE_ASSET_ID, DR_ANMAR_NEEDLE_ASSET_VERSION, DR_ANMAR_NEEDLE_NAME
from dr_anmar_suture_model import build_suture_interface_visual_mesh
from dr_anmar_suture_model import derive as derive_suture
from dr_anmar_suture_model import load_profile as load_suture_profile

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--asset", type=Path, required=True)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import numpy as np  # noqa: E402

import omni.usd  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext  # noqa: E402


def rotate_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate one vector by the PhysX tensor API's XYZW quaternion."""

    vector_part = quaternion[:3]
    scalar_part = quaternion[3]
    doubled_cross = 2.0 * np.cross(vector_part, vector)
    return (
        vector
        + scalar_part * doubled_cross
        + np.cross(
            vector_part,
            doubled_cross,
        )
    )


def main() -> int:
    if not args.asset.is_file():
        raise FileNotFoundError(args.asset)
    needle_profile = load_needle_profile()
    suture_profile = load_suture_profile()
    derived_suture = derive_suture(suture_profile)
    expected_suture_interface_mesh = build_suture_interface_visual_mesh(
        suture_profile,
        derived=derived_suture,
    )
    derived_needle = derive_needle(needle_profile)
    expected_collision_capsules = build_needle_collision_capsules(needle_profile)
    expected_needle_mesh = build_needle_mesh(needle_profile)
    sim = SimulationContext(
        SimulationCfg(
            dt=0.0005,
            render_interval=16,
            device=args.device,
            use_fabric=False,
            physx=PhysxCfg(
                solver_type=1,
                min_position_iteration_count=16,
                max_position_iteration_count=32,
                min_velocity_iteration_count=2,
                max_velocity_iteration_count=8,
                enable_ccd=True,
                gpu_max_rigid_contact_count=2**18,
                gpu_max_rigid_patch_count=2**16,
                gpu_found_lost_pairs_capacity=2**18,
                gpu_found_lost_aggregate_pairs_capacity=2**18,
                gpu_total_aggregate_pairs_capacity=2**18,
                gpu_collision_stack_size=2**25,
                gpu_heap_capacity=2**26,
                gpu_temp_buffer_capacity=2**24,
            ),
        )
    )
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    scene = UsdPhysics.Scene.Get(stage, "/physicsScene")
    scene.GetGravityMagnitudeAttr().Set(9.81)

    root_path = "/World/DrAnmarNeedle"
    root = stage.DefinePrim(root_path, "Xform")
    root.GetReferences().AddReference(str(args.asset.resolve()))
    needle_physics_variant_selection = root.GetVariantSets().GetVariantSet("Physics").GetVariantSelection()
    suture_variant_prim = stage.GetPrimAtPath(f"{root_path}/Suture")
    suture_physics_variant_selection = (
        suture_variant_prim.GetVariantSets().GetVariantSet("Physics").GetVariantSelection()
        if suture_variant_prim.IsValid()
        else None
    )
    physics_variant_contract_valid = bool(
        needle_physics_variant_selection == "physx" and suture_physics_variant_selection == "physx"
    )
    xform = UsdGeom.Xformable(root)
    xform.AddTranslateOp().Set(Gf.Vec3d(-0.09, 0.0, 0.06))

    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.CreateSizeAttr(1.0)
    ground.AddScaleOp().Set(Gf.Vec3f(0.3, 0.2, 0.002))
    ground.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.002))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    sim.reset()
    physics_view = SimulationManager.get_physics_sim_view()
    assembly = stage.GetPrimAtPath(f"{root_path}/Suture/Segments/S0000").IsValid()
    segment_pattern = f"{root_path}/Suture/Segments/S*" if assembly else f"{root_path}/Segments/S*"
    joint_prefix = f"{root_path}/Suture/Joints/" if assembly else f"{root_path}/Joints/"
    segments = physics_view.create_rigid_body_view(segment_pattern)
    if segments._backend is None or segments.count != 360:
        raise RuntimeError(f"PhysX created {segments.count if segments._backend else 0} of 360 suture bodies")
    needle = None
    interface = None
    initial_swage_distance_m = None
    if assembly:
        needle = physics_view.create_rigid_body_view(f"{root_path}/Needle")
        interface = physics_view.create_rigid_body_view(f"{root_path}/Suture/NeedleInterface")
        if needle._backend is None or needle.count != 1 or interface._backend is None or interface.count != 1:
            raise RuntimeError("PhysX did not create the needle and swage rigid bodies")
        initial_needle = needle.get_transforms().cpu().numpy().astype(np.float64)[0]
        initial_interface = interface.get_transforms().cpu().numpy().astype(np.float64)[0]
        initial_anchor = initial_needle[:3] + rotate_xyzw(
            initial_needle[3:7],
            np.asarray(derived_needle.swage_anchor_m, dtype=np.float64),
        )
        initial_swage_distance_m = float(np.linalg.norm(initial_anchor - initial_interface[:3]))
    initial = segments.get_transforms().cpu().numpy().astype(np.float64)
    for _ in range(max(1, args.steps)):
        sim.step(render=False)
    final = segments.get_transforms().cpu().numpy().astype(np.float64)
    finite = bool(np.isfinite(final).all())
    free_end_drop = float(initial[-1, 2] - final[-1, 2])
    displacement = np.linalg.norm(final[:, :3] - initial[:, :3], axis=1)
    final_swage_distance_m = None
    if needle is not None and interface is not None:
        final_needle = needle.get_transforms().cpu().numpy().astype(np.float64)[0]
        final_interface = interface.get_transforms().cpu().numpy().astype(np.float64)[0]
        final_anchor = final_needle[:3] + rotate_xyzw(
            final_needle[3:7],
            np.asarray(derived_needle.swage_anchor_m, dtype=np.float64),
        )
        final_swage_distance_m = float(np.linalg.norm(final_anchor - final_interface[:3]))
    joint_count = sum(
        prim.GetTypeName() == "PhysicsJoint"
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(joint_prefix)
    )
    factory_swage = stage.GetPrimAtPath(f"{root_path}/FactorySwage")
    needle_collision_capsules = []
    needle_collision_extent_count = None
    needle_friction_combine_mode = None
    needle_authored_mass_kg = None
    needle_center_of_mass_m = None
    needle_diagonal_inertia_kg_m2 = None
    needle_principal_axes_wxyz = None
    needle_mass_properties_match_geometry = None
    needle_physx_collision_api_count = None
    needle_newton_collision_api_count = None
    needle_physx_contact_offset_range_m = None
    needle_physx_rest_offset_range_m = None
    needle_physx_contact_offsets_match_profile = None
    needle_engine_schema_isolation_valid = None
    needle_visual_normal_value_count = None
    needle_visual_normal_index_count = None
    needle_visual_normal_interpolation = None
    needle_visual_normals_valid = None
    needle_collision_guide_purpose_count = None
    needle_collision_invisible_count = None
    needle_collision_physics_material_binding_count = None
    needle_render_collision_separation_valid = None
    needle_material_organization_valid = None
    needle_base_layer_name = None
    needle_geometry_layer_name = None
    needle_materials_layer_name = None
    needle_neutral_physics_layer_name = None
    needle_physx_layer_name = None
    needle_asset_structure_source_ownership_valid = None
    suture_geometry_layer_name = None
    suture_base_layer_name = None
    suture_materials_layer_name = None
    suture_neutral_physics_layer_name = None
    suture_physx_layer_name = None
    suture_asset_structure_source_ownership_valid = None
    suture_physx_collision_api_count = None
    suture_hybrid_ccd_body_count = None
    suture_physx_contact_offset_range_m = None
    suture_physx_rest_offset_range_m = None
    suture_physx_contact_offsets_match_profile = None
    suture_material_bindings_valid = None
    suture_visual_mesh_count = None
    suture_visual_mesh_vertex_count = None
    suture_visual_normals_valid_count = None
    suture_visual_uv_value_count = None
    suture_visual_uv_index_count = None
    suture_visual_uv_valid_count = None
    suture_material_texture_path = None
    suture_material_texture_exists = None
    suture_pbr_material_graph_valid = None
    suture_collision_capsule_count = None
    suture_collision_guide_purpose_count = None
    suture_collision_invisible_count = None
    suture_collision_physics_material_binding_count = None
    suture_collider_cylinder_height_range_m = None
    suture_minimum_visual_collision_margin_m = None
    suture_interface_minimum_visual_collision_margin_m = None
    suture_interface_visual_mesh_valid = None
    suture_render_collision_separation_valid = None
    if assembly:
        layer_organization = needle_profile["construction"]["layer_organization"]
        needle_base_layer_name = str(layer_organization["base_layer"])
        needle_geometry_layer_name = str(layer_organization["geometry_layer"])
        needle_materials_layer_name = str(layer_organization["materials_layer"])
        needle_neutral_physics_layer_name = str(layer_organization["physics_layer"])
        needle_physx_layer_name = str(layer_organization["physx_layer"])
        local_base_path = args.asset.resolve().parent / needle_base_layer_name
        local_geometry_path = args.asset.resolve().parent / needle_geometry_layer_name
        local_materials_path = args.asset.resolve().parent / needle_materials_layer_name
        local_physics_path = args.asset.resolve().parent / needle_neutral_physics_layer_name
        local_physx_path = args.asset.resolve().parent / needle_physx_layer_name
        entry_layer_text = args.asset.read_text(encoding="utf-8")
        base_layer_text = local_base_path.read_text(encoding="utf-8")
        geometry_stage = Usd.Stage.Open(str(local_geometry_path))
        if geometry_stage is None:
            raise RuntimeError(f"Could not open the needle geometry layer: {local_geometry_path}")
        geometry_layer_text = geometry_stage.GetRootLayer().ExportToString()
        materials_layer_text = local_materials_path.read_text(encoding="utf-8")
        physics_layer_text = local_physics_path.read_text(encoding="utf-8")
        physx_layer_text = local_physx_path.read_text(encoding="utf-8")
        entry_physics_properties = re.findall(
            r"\b(?:physics:|physx[A-Za-z]*:|newton:)[A-Za-z][A-Za-z0-9_]*",
            entry_layer_text,
        )
        entry_physics_schemas = re.findall(
            r'"((?:Physics|Physx|Newton)[A-Za-z0-9_]*API)"',
            entry_layer_text,
        )
        entry_physics_typed_prims = re.findall(
            r"\bdef\s+(Physics[A-Za-z0-9_]+)\s+\"",
            entry_layer_text,
        )
        base_physics_properties = re.findall(
            r"\b(?:physics:|physx[A-Za-z]*:|newton:)[A-Za-z][A-Za-z0-9_]*",
            base_layer_text,
        )
        base_physics_schemas = re.findall(
            r'"((?:Physics|Physx|Newton)[A-Za-z0-9_]*API)"',
            base_layer_text,
        )
        neutral_engine_properties = re.findall(
            r"\b(?:physx[A-Za-z]*:|newton:)[A-Za-z][A-Za-z0-9_]*",
            physics_layer_text,
        )
        neutral_engine_schemas = re.findall(
            r'"((?:Physx|Newton)[A-Za-z0-9_]*API)"',
            physics_layer_text,
        )
        physx_neutral_properties = re.findall(
            r"\bphysics:[A-Za-z][A-Za-z0-9_]*",
            physx_layer_text,
        )
        physx_newton_properties = re.findall(
            r"\bnewton:[A-Za-z][A-Za-z0-9_]*",
            physx_layer_text,
        )
        physx_newton_schemas = re.findall(
            r'"(Newton[A-Za-z0-9_]*API)"',
            physx_layer_text,
        )
        needle_asset_structure_source_ownership_valid = bool(
            layer_organization["entry_layer"] == args.asset.name
            and needle_base_layer_name.endswith("_base.usda")
            and needle_geometry_layer_name.endswith("_geometry.usd")
            and layer_organization["geometry_format"] == "usdc"
            and local_geometry_path.read_bytes()[:8] == b"PXR-USDC"
            and needle_materials_layer_name.endswith("_materials.usda")
            and needle_neutral_physics_layer_name.endswith("_physics.usda")
            and needle_physx_layer_name.endswith("_physx.usda")
            and f"@{needle_base_layer_name}@" in entry_layer_text
            and f"@{needle_physx_layer_name}@" in entry_layer_text
            and f"@{needle_neutral_physics_layer_name}@" in entry_layer_text
            and f"@{needle_materials_layer_name}@" not in entry_layer_text
            and f"@{needle_geometry_layer_name}@" not in entry_layer_text
            and f"@{needle_materials_layer_name}@" in base_layer_text
            and f"@{needle_geometry_layer_name}@" in base_layer_text
            and f"@{needle_neutral_physics_layer_name}@" in physx_layer_text
            and 'append variantSets = "Physics"' in entry_layer_text
            and entry_layer_text.count("prepend payload =") == 2
            and entry_layer_text.count('over "Suture" (') == 3
            and layer_organization["variant_choices"] == ["none", "physics", "physx"]
            and layer_organization["default_runtime"] == "physx"
            and len(entry_layer_text.encode("utf-8")) <= int(layer_organization["entry_layer_max_bytes"])
            and not entry_physics_properties
            and not entry_physics_schemas
            and not entry_physics_typed_prims
            and not base_physics_properties
            and not base_physics_schemas
            and 'def Mesh "Visual"' not in entry_layer_text
            and "point3f[] points" not in entry_layer_text
            and 'def Material "NeedleSteelVisual"' not in entry_layer_text
            and 'def Shader "PreviewSurface"' not in entry_layer_text
            and 'def Mesh "Visual"' in geometry_layer_text
            and "point3f[] points" in geometry_layer_text
            and "faceVertexIndices" in geometry_layer_text
            and "primvars:normals" in geometry_layer_text
            and "apiSchemas" not in geometry_layer_text
            and "material:binding" not in geometry_layer_text
            and 'def Material "' not in geometry_layer_text
            and 'def Shader "' not in geometry_layer_text
            and 'def Scope "Looks"' in materials_layer_text
            and 'def Material "NeedleSteelVisual"' in materials_layer_text
            and 'def Shader "PreviewSurface"' in materials_layer_text
            and '"MaterialBindingAPI"' in materials_layer_text
            and "rel material:binding" in materials_layer_text
            and "point3f[] points" not in materials_layer_text
            and "faceVertexIndices" not in materials_layer_text
            and "physics:" not in materials_layer_text
            and "physx" not in materials_layer_text
            and "newton:" not in materials_layer_text
            and not neutral_engine_properties
            and not neutral_engine_schemas
            and not physx_neutral_properties
            and not physx_newton_properties
            and not physx_newton_schemas
            and 'def Material "NeedleSteelPhysics"' in physics_layer_text
            and '"PhysicsMaterialAPI"' in physics_layer_text
            and '"PhysicsRigidBodyAPI", "PhysicsMassAPI"' in physics_layer_text
            and 'def Scope "Collision"' in physics_layer_text
            and 'over "NeedleInterface"' in physics_layer_text
            and 'def PhysicsFixedJoint "FactorySwage"' in physics_layer_text
            and '"PhysxMaterialAPI"' in physx_layer_text
            and '"PhysxRigidBodyAPI"' in physx_layer_text
            and '"PhysxCollisionAPI"' in physx_layer_text
            and "physxCollision:contactOffset" in physx_layer_text
            and "physxCollision:restOffset" in physx_layer_text
            and 'def Mesh "Visual"' not in physics_layer_text
            and 'def Mesh "Visual"' not in physx_layer_text
            and "point3f[] points" not in physics_layer_text
            and "point3f[] points" not in physx_layer_text
            and 'def Shader "PreviewSurface"' not in physics_layer_text
            and 'def Shader "PreviewSurface"' not in physx_layer_text
            and "prepend references =" not in physics_layer_text
            and "prepend references =" not in physx_layer_text
        )
        suture_layer_organization = suture_profile["asset_structure"]
        suture_base_layer_name = str(suture_layer_organization["base_layer"])
        suture_geometry_layer_name = str(suture_layer_organization["geometry_layer"])
        suture_materials_layer_name = str(suture_layer_organization["materials_layer"])
        suture_neutral_physics_layer_name = str(suture_layer_organization["physics_layer"])
        suture_physx_layer_name = str(suture_layer_organization["physx_layer"])
        suture_directory = args.asset.resolve().parent.parent / "suture"
        suture_entry_path = suture_directory / str(suture_layer_organization["entry_layer"])
        suture_base_path = suture_directory / suture_base_layer_name
        suture_geometry_path = suture_directory / suture_geometry_layer_name
        suture_materials_path = suture_directory / suture_materials_layer_name
        suture_physics_path = suture_directory / suture_neutral_physics_layer_name
        suture_physx_path = suture_directory / suture_physx_layer_name
        suture_material_texture_path = str(
            (
                suture_materials_path.parent
                / str(suture_profile["appearance"]["normal_roughness_texture"]["relative_path"])
            ).resolve()
        )
        suture_material_texture_file = Path(suture_material_texture_path)
        suture_material_texture_exists = bool(
            suture_material_texture_file.is_file()
            and suture_material_texture_file.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        )
        suture_entry_text = suture_entry_path.read_text(encoding="utf-8")
        suture_base_text = suture_base_path.read_text(encoding="utf-8")
        suture_geometry_stage = Usd.Stage.Open(str(suture_geometry_path))
        if suture_geometry_stage is None:
            raise RuntimeError(f"Could not open the suture geometry layer: {suture_geometry_path}")
        suture_geometry_text = suture_geometry_stage.GetRootLayer().ExportToString()
        suture_materials_text = suture_materials_path.read_text(encoding="utf-8")
        suture_physics_text = suture_physics_path.read_text(encoding="utf-8")
        suture_physx_text = suture_physx_path.read_text(encoding="utf-8")
        suture_asset_structure_source_ownership_valid = bool(
            suture_geometry_path.read_bytes()[:8] == b"PXR-USDC"
            and f"@{suture_base_layer_name}@" in suture_entry_text
            and f"@{suture_physx_layer_name}@" in suture_entry_text
            and f"@{suture_neutral_physics_layer_name}@" in suture_entry_text
            and f"@{suture_geometry_layer_name}@" not in suture_entry_text
            and f"@{suture_materials_layer_name}@" not in suture_entry_text
            and f"@{suture_geometry_layer_name}@" in suture_base_text
            and f"@{suture_materials_layer_name}@" in suture_base_text
            and f"@{suture_neutral_physics_layer_name}@" in suture_physx_text
            and 'append variantSets = "Physics"' in suture_entry_text
            and suture_entry_text.count("prepend payload =") == 2
            and suture_layer_organization["variant_choices"] == ["none", "physics", "physx"]
            and suture_layer_organization["default_runtime"] == "physx"
            and len(suture_entry_text.encode("utf-8")) <= int(suture_layer_organization["entry_layer_max_bytes"])
            and not re.findall(
                r"\b(?:physics:|physx[A-Za-z]*:|newton:)[A-Za-z][A-Za-z0-9_]*",
                suture_entry_text,
            )
            and not re.findall(
                r"\b(?:physics:|physx[A-Za-z]*:|newton:)[A-Za-z][A-Za-z0-9_]*",
                suture_base_text,
            )
            and not re.findall(
                r'"((?:Physics|Physx|Newton)[A-Za-z0-9_]*API)"',
                suture_base_text,
            )
            and 'def Xform "NeedleInterface"' in suture_geometry_text
            and len(re.findall(r'def Xform "S\d{4}"', suture_geometry_text)) == 360
            and len(re.findall(r'def Mesh "Visual"', suture_geometry_text)) == 361
            and len(re.findall(r'def Capsule "Visual"', suture_geometry_text)) == 0
            and len(re.findall(r'def Capsule "Collision"', suture_geometry_text)) == 361
            and suture_geometry_text.count('uniform token purpose = "guide"') == 361
            and suture_geometry_text.count('token visibility = "invisible"') == 361
            and suture_geometry_text.count('uniform token subdivisionScheme = "none"') == 361
            and suture_geometry_text.count("normal3f[] primvars:normals") == 361
            and suture_geometry_text.count('interpolation = "vertex"') == 361
            and suture_geometry_text.count("texCoord2f[] primvars:st") == 360
            and suture_geometry_text.count("int[] primvars:st:indices") == 360
            and suture_geometry_text.count('interpolation = "faceVarying"') == 360
            and "apiSchemas" not in suture_geometry_text
            and "material:binding" not in suture_geometry_text
            and "physics:" not in suture_geometry_text
            and "physx" not in suture_geometry_text
            and suture_materials_text.count('def Material "') == 2
            and suture_materials_text.count('def Shader "PreviewSurface"') == 2
            and suture_materials_text.count('uniform token info:id = "UsdPrimvarReader_float2"') == 1
            and suture_materials_text.count('string inputs:varname = "st"') == 1
            and suture_materials_text.count('uniform token info:id = "UsdUVTexture"') == 1
            and ("asset inputs:file = @./textures/DrAnmarSuture4_0_braid_normal_roughness.png@")
            in suture_materials_text
            and 'token inputs:sourceColorSpace = "raw"' in suture_materials_text
            and suture_material_texture_exists
            and "physics:" not in suture_materials_text
            and "physx" not in suture_materials_text
            and '"Physx' not in suture_physics_text
            and "physx" not in suture_physics_text
            and '"Newton' not in suture_physics_text
            and "newton:" not in suture_physics_text
            and "physics:rigidBodyEnabled" not in suture_physx_text
            and "physics:mass" not in suture_physx_text
            and '"PhysicsRigidBodyAPI"' not in suture_physx_text
            and '"Newton' not in suture_physx_text
            and "newton:" not in suture_physx_text
        )
        suture_segment_prims = [
            stage.GetPrimAtPath(f"{root_path}/Suture/Segments/S{index:04d}") for index in range(360)
        ]
        suture_interface_prim = stage.GetPrimAtPath(f"{root_path}/Suture/NeedleInterface")
        suture_body_prims = [suture_interface_prim, *suture_segment_prims]
        suture_collision_prims = [stage.GetPrimAtPath(f"{prim.GetPath()}/Collision") for prim in suture_body_prims]
        suture_visual_prims = [stage.GetPrimAtPath(f"{prim.GetPath()}/Visual") for prim in suture_segment_prims]
        suture_interface_visual_prim = stage.GetPrimAtPath(f"{root_path}/Suture/NeedleInterface/Visual")
        suture_physx_collision_api_count = sum(
            "PhysxCollisionAPI" in prim.GetAppliedSchemas() for prim in suture_collision_prims
        )
        suture_hybrid_ccd_body_count = sum(
            bool(prim.GetAttribute("physxRigidBody:enableCCD").Get())
            and bool(prim.GetAttribute("physxRigidBody:enableSpeculativeCCD").Get())
            for prim in suture_body_prims
        )
        suture_contact_offsets = [
            float(prim.GetAttribute("physxCollision:contactOffset").Get()) for prim in suture_collision_prims
        ]
        suture_rest_offsets = [
            float(prim.GetAttribute("physxCollision:restOffset").Get()) for prim in suture_collision_prims
        ]
        suture_physx_contact_offset_range_m = [
            min(suture_contact_offsets),
            max(suture_contact_offsets),
        ]
        suture_physx_rest_offset_range_m = [
            min(suture_rest_offsets),
            max(suture_rest_offsets),
        ]
        suture_offset_contract = suture_profile["contact"]["contact_offsets"]
        suture_expected_contact_offsets = [
            max(
                float(suture_offset_contract["minimum_m"]),
                min(
                    float(suture_offset_contract["maximum_m"]),
                    float(UsdGeom.Capsule(prim).GetRadiusAttr().Get())
                    * float(suture_offset_contract["collision_radius_fraction"]),
                ),
            )
            for prim in suture_collision_prims
        ]
        suture_physx_contact_offsets_match_profile = bool(
            np.allclose(
                suture_contact_offsets,
                suture_expected_contact_offsets,
                rtol=1.0e-6,
                atol=1.0e-12,
            )
            and np.allclose(
                suture_rest_offsets,
                float(suture_offset_contract["rest_offset_m"]),
                rtol=0.0,
                atol=1.0e-12,
            )
        )
        suture_segments_scope = stage.GetPrimAtPath(f"{root_path}/Suture/Segments")
        suture_visual_material_path = f"{root_path}/Suture/Looks/SutureVisual"
        suture_physics_material_path = f"{root_path}/Suture/Materials/SutureMaterial"
        swage_visual_material_path = f"{root_path}/Suture/Looks/SwageVisual"
        swage_physics_material_path = f"{root_path}/Suture/Materials/SwageSteel"
        suture_physics_material_prim = stage.GetPrimAtPath(suture_physics_material_path)
        swage_physics_material_prim = stage.GetPrimAtPath(swage_physics_material_path)
        suture_preview_shader_path = f"{suture_visual_material_path}/PreviewSurface"
        suture_primvar_reader_path = f"{suture_visual_material_path}/PrimvarReader_st"
        suture_texture_shader_path = f"{suture_visual_material_path}/BraidNormalRoughness"
        suture_preview_shader = stage.GetPrimAtPath(suture_preview_shader_path)
        suture_primvar_reader = stage.GetPrimAtPath(suture_primvar_reader_path)
        suture_texture_shader = stage.GetPrimAtPath(suture_texture_shader_path)
        suture_texture_asset = suture_texture_shader.GetAttribute("inputs:file").Get()
        suture_pbr_material_graph_valid = bool(
            suture_preview_shader.GetTypeName() == "Shader"
            and str(suture_preview_shader.GetAttribute("info:id").Get()) == "UsdPreviewSurface"
            and suture_primvar_reader.GetTypeName() == "Shader"
            and str(suture_primvar_reader.GetAttribute("info:id").Get()) == "UsdPrimvarReader_float2"
            and str(suture_primvar_reader.GetAttribute("inputs:varname").GetTypeName()) == "string"
            and str(suture_primvar_reader.GetAttribute("inputs:varname").Get()) == "st"
            and suture_texture_shader.GetTypeName() == "Shader"
            and str(suture_texture_shader.GetAttribute("info:id").Get()) == "UsdUVTexture"
            and getattr(suture_texture_asset, "path", "") == "./textures/DrAnmarSuture4_0_braid_normal_roughness.png"
            and str(suture_texture_shader.GetAttribute("inputs:sourceColorSpace").Get()) == "raw"
            and str(suture_texture_shader.GetAttribute("inputs:wrapS").Get()) == "repeat"
            and str(suture_texture_shader.GetAttribute("inputs:wrapT").Get()) == "repeat"
            and [str(path) for path in suture_preview_shader.GetAttribute("inputs:normal").GetConnections()]
            == [f"{suture_texture_shader_path}.outputs:rgb"]
            and [str(path) for path in suture_preview_shader.GetAttribute("inputs:roughness").GetConnections()]
            == [f"{suture_texture_shader_path}.outputs:a"]
            and [str(path) for path in suture_texture_shader.GetAttribute("inputs:st").GetConnections()]
            == [f"{suture_primvar_reader_path}.outputs:result"]
            and suture_material_texture_exists
        )
        suture_visual_mesh_count = sum(prim.IsValid() and prim.GetTypeName() == "Mesh" for prim in suture_visual_prims)
        suture_visual_mesh_vertex_count = sum(
            len(UsdGeom.Mesh(prim).GetPointsAttr().Get()) for prim in suture_visual_prims
        )
        suture_visual_normals_valid_count = 0
        suture_visual_uv_value_count = 0
        suture_visual_uv_index_count = 0
        suture_visual_uv_valid_count = 0
        for prim in suture_visual_prims:
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            normals = mesh.GetNormalsAttr().Get()
            normal_array = np.asarray(normals, dtype=np.float64)
            if (
                len(points) == len(normals)
                and mesh.GetNormalsAttr().GetMetadata("interpolation") == "vertex"
                and mesh.GetSubdivisionSchemeAttr().Get() == "none"
                and np.isfinite(normal_array).all()
                and np.allclose(
                    np.linalg.norm(normal_array, axis=1),
                    1.0,
                    rtol=0.0,
                    atol=2.0e-5,
                )
            ):
                suture_visual_normals_valid_count += 1
            st_attribute = prim.GetAttribute("primvars:st")
            st_index_attribute = prim.GetAttribute("primvars:st:indices")
            texture_coordinates = st_attribute.Get()
            texture_coordinate_indices = st_index_attribute.Get()
            if texture_coordinates is not None and texture_coordinate_indices is not None:
                texture_coordinate_array = np.asarray(
                    texture_coordinates,
                    dtype=np.float64,
                )
                texture_coordinate_index_array = np.asarray(
                    texture_coordinate_indices,
                    dtype=np.int64,
                )
                suture_visual_uv_value_count += len(texture_coordinates)
                suture_visual_uv_index_count += len(texture_coordinate_indices)
                if (
                    str(st_attribute.GetTypeName()) == "texCoord2f[]"
                    and st_attribute.GetMetadata("interpolation") == "faceVarying"
                    and len(texture_coordinates) == 441
                    and len(texture_coordinate_indices) == 1440
                    and len(texture_coordinate_indices) == len(mesh.GetFaceVertexIndicesAttr().Get())
                    and np.isfinite(texture_coordinate_array).all()
                    and np.all(texture_coordinate_index_array >= 0)
                    and np.all(texture_coordinate_index_array < len(texture_coordinates))
                    and np.isclose(
                        texture_coordinate_array[:, 1].min(),
                        0.0,
                        rtol=0.0,
                        atol=1.0e-7,
                    )
                    and np.isclose(
                        texture_coordinate_array[:, 1].max(),
                        1.0,
                        rtol=0.0,
                        atol=1.0e-7,
                    )
                ):
                    suture_visual_uv_valid_count += 1
        suture_collision_capsule_count = sum(
            prim.IsValid() and prim.GetTypeName() == "Capsule" for prim in suture_collision_prims
        )
        suture_collision_guide_purpose_count = sum(
            str(UsdGeom.Imageable(prim).GetPurposeAttr().Get()) == "guide" for prim in suture_collision_prims
        )
        suture_collision_invisible_count = sum(
            str(UsdGeom.Imageable(prim).GetVisibilityAttr().Get()) == "invisible" for prim in suture_collision_prims
        )
        suture_collider_heights = [
            float(UsdGeom.Capsule(prim).GetHeightAttr().Get()) for prim in suture_collision_prims
        ]
        suture_collider_cylinder_height_range_m = [
            min(suture_collider_heights),
            max(suture_collider_heights),
        ]
        suture_minimum_visual_collision_margin_m = np.inf
        for visual_prim, collision_prim in zip(
            suture_visual_prims,
            suture_collision_prims[1:],
            strict=True,
        ):
            points = np.asarray(
                UsdGeom.Mesh(visual_prim).GetPointsAttr().Get(),
                dtype=np.float64,
            )
            radius = float(UsdGeom.Capsule(collision_prim).GetRadiusAttr().Get())
            cylinder_height = float(UsdGeom.Capsule(collision_prim).GetHeightAttr().Get())
            axial_excess = np.maximum(np.abs(points[:, 0]) - cylinder_height / 2.0, 0.0)
            radial_distance = np.linalg.norm(points[:, 1:3], axis=1)
            distance_to_spine = np.hypot(axial_excess, radial_distance)
            suture_minimum_visual_collision_margin_m = min(
                suture_minimum_visual_collision_margin_m,
                float(np.min(radius - distance_to_spine)),
            )
        interface_visual_mesh = UsdGeom.Mesh(suture_interface_visual_prim)
        interface_points = np.asarray(
            interface_visual_mesh.GetPointsAttr().Get(),
            dtype=np.float64,
        )
        interface_normals = np.asarray(
            interface_visual_mesh.GetNormalsAttr().Get(),
            dtype=np.float64,
        )
        interface_face_counts = np.asarray(
            interface_visual_mesh.GetFaceVertexCountsAttr().Get(),
            dtype=np.int64,
        )
        interface_face_indices = np.asarray(
            interface_visual_mesh.GetFaceVertexIndicesAttr().Get(),
            dtype=np.int64,
        )
        interface_collision = UsdGeom.Capsule(suture_collision_prims[0])
        interface_collision_radius = float(interface_collision.GetRadiusAttr().Get())
        interface_collision_height = float(interface_collision.GetHeightAttr().Get())
        interface_axial_excess = np.maximum(
            np.abs(interface_points[:, 0]) - interface_collision_height / 2.0,
            0.0,
        )
        interface_radial_distance = np.linalg.norm(
            interface_points[:, 1:3],
            axis=1,
        )
        interface_distance_to_spine = np.hypot(
            interface_axial_excess,
            interface_radial_distance,
        )
        suture_interface_minimum_visual_collision_margin_m = float(
            np.min(interface_collision_radius - interface_distance_to_spine)
        )
        interface_edge_counts: dict[tuple[int, int], int] = {}
        interface_face_cursor = 0
        for face_count in interface_face_counts.tolist():
            face = interface_face_indices[interface_face_cursor : interface_face_cursor + face_count].tolist()
            interface_face_cursor += face_count
            for left, right in zip(
                face,
                (*face[1:], face[0]),
                strict=True,
            ):
                edge = (min(left, right), max(left, right))
                interface_edge_counts[edge] = interface_edge_counts.get(edge, 0) + 1
        expected_interface_points = np.asarray(
            expected_suture_interface_mesh.points,
            dtype=np.float64,
        )
        expected_interface_exit_x = float(expected_interface_points[:, 0].max())
        expected_interface_exit_radius = float(
            np.linalg.norm(
                expected_interface_points[
                    np.isclose(
                        expected_interface_points[:, 0],
                        expected_interface_exit_x,
                        rtol=0.0,
                        atol=1.0e-15,
                    )
                ][:, 1:3],
                axis=1,
            ).max()
        )
        interface_exit_x = float(interface_points[:, 0].max())
        interface_exit_radius = float(
            np.linalg.norm(
                interface_points[
                    np.isclose(
                        interface_points[:, 0],
                        interface_exit_x,
                        rtol=0.0,
                        atol=1.0e-9,
                    )
                ][:, 1:3],
                axis=1,
            ).max()
        )
        suture_interface_visual_mesh_valid = bool(
            suture_interface_visual_prim.GetTypeName() == "Mesh"
            and interface_points.shape == expected_interface_points.shape
            and interface_normals.shape == interface_points.shape
            and len(interface_face_counts) == len(expected_suture_interface_mesh.face_vertex_counts)
            and len(interface_face_indices) == len(expected_suture_interface_mesh.face_vertex_indices)
            and int(interface_face_counts.sum()) == len(interface_face_indices)
            and np.all(interface_face_counts >= 3)
            and np.all(interface_face_indices >= 0)
            and np.all(interface_face_indices < len(interface_points))
            and np.isfinite(interface_points).all()
            and np.isfinite(interface_normals).all()
            and np.allclose(
                np.linalg.norm(interface_normals, axis=1),
                1.0,
                rtol=0.0,
                atol=2.0e-5,
            )
            and np.allclose(
                interface_points,
                expected_interface_points,
                rtol=1.0e-6,
                atol=1.0e-10,
            )
            and all(count == 2 for count in interface_edge_counts.values())
            and np.isclose(
                interface_exit_x,
                expected_interface_exit_x,
                rtol=0.0,
                atol=1.0e-9,
            )
            and np.isclose(
                interface_exit_radius,
                expected_interface_exit_radius,
                rtol=1.0e-6,
                atol=1.0e-10,
            )
            and suture_interface_minimum_visual_collision_margin_m
            >= -float(
                suture_profile["geometry"]["visual_representation"]["binary_visual_point_containment_tolerance_m"]
            )
            and interface_visual_mesh.GetSubdivisionSchemeAttr().Get() == "none"
            and str(UsdGeom.Imageable(suture_interface_visual_prim).GetPurposeAttr().Get()) == "default"
            and str(UsdGeom.Imageable(suture_interface_visual_prim).GetVisibilityAttr().Get()) == "inherited"
            and "PhysicsCollisionAPI" not in suture_interface_visual_prim.GetAppliedSchemas()
            and "PhysxCollisionAPI" not in suture_interface_visual_prim.GetAppliedSchemas()
            and not suture_interface_visual_prim.GetRelationship("material:binding:physics").HasAuthoredTargets()
        )
        expected_suture_physics_material_paths = [
            swage_physics_material_path,
            *([suture_physics_material_path] * len(suture_segment_prims)),
        ]
        suture_collision_physics_material_binding_count = sum(
            [str(target) for target in prim.GetRelationship("material:binding:physics").GetTargets()] == [expected_path]
            and not prim.GetRelationship("material:binding").HasAuthoredTargets()
            for prim, expected_path in zip(
                suture_collision_prims,
                expected_suture_physics_material_paths,
                strict=True,
            )
        )
        suture_material_bindings_valid = bool(
            [str(target) for target in suture_segments_scope.GetRelationship("material:binding").GetTargets()]
            == [suture_visual_material_path]
            and [str(target) for target in suture_interface_prim.GetRelationship("material:binding").GetTargets()]
            == [swage_visual_material_path]
            and not suture_segments_scope.GetRelationship("material:binding:physics").HasAuthoredTargets()
            and not suture_interface_prim.GetRelationship("material:binding:physics").HasAuthoredTargets()
            and suture_collision_physics_material_binding_count == 361
            and "PhysicsMaterialAPI" in suture_physics_material_prim.GetAppliedSchemas()
            and "PhysxMaterialAPI" in suture_physics_material_prim.GetAppliedSchemas()
            and "PhysicsMaterialAPI" in swage_physics_material_prim.GetAppliedSchemas()
            and "PhysxMaterialAPI" in swage_physics_material_prim.GetAppliedSchemas()
            and str(suture_physics_material_prim.GetAttribute("physxMaterial:frictionCombineMode").Get()) == "max"
            and str(swage_physics_material_prim.GetAttribute("physxMaterial:frictionCombineMode").Get()) == "max"
        )
        expected_visual_vertices_per_segment = (
            int(suture_profile["geometry"]["visual_representation"]["axial_samples_per_segment"])
            * int(suture_profile["geometry"]["visual_representation"]["radial_samples"])
            + 2
        )
        suture_render_collision_separation_valid = bool(
            suture_visual_mesh_count == 360
            and suture_visual_mesh_vertex_count == 360 * expected_visual_vertices_per_segment
            and suture_visual_normals_valid_count == 360
            and suture_visual_uv_value_count == 360 * 441
            and suture_visual_uv_index_count == 360 * 1440
            and suture_visual_uv_valid_count == 360
            and suture_pbr_material_graph_valid
            and suture_collision_capsule_count == 361
            and suture_collision_guide_purpose_count == 361
            and suture_collision_invisible_count == 361
            and suture_collision_physics_material_binding_count == 361
            and np.allclose(
                suture_collider_heights,
                derived_suture.segment_spacing_m,
                rtol=0.0,
                atol=1.0e-12,
            )
            and suture_minimum_visual_collision_margin_m
            >= -float(
                suture_profile["geometry"]["visual_representation"]["binary_visual_point_containment_tolerance_m"]
            )
            and suture_interface_visual_mesh_valid
            and all(
                str(UsdGeom.Imageable(prim).GetPurposeAttr().Get()) == "default"
                and str(UsdGeom.Imageable(prim).GetVisibilityAttr().Get()) == "inherited"
                and "PhysicsCollisionAPI" not in prim.GetAppliedSchemas()
                and "PhysxCollisionAPI" not in prim.GetAppliedSchemas()
                and not prim.GetRelationship("material:binding:physics").HasAuthoredTargets()
                for prim in [
                    suture_interface_visual_prim,
                    *suture_visual_prims,
                ]
            )
        )
        needle_collision_capsules = [
            prim
            for prim in stage.Traverse()
            if prim.GetTypeName() == "Capsule" and str(prim.GetPath()).startswith(f"{root_path}/Needle/Collision/C")
        ]
        needle_collision_extent_count = sum(
            UsdGeom.Capsule(prim).GetExtentAttr().HasAuthoredValueOpinion() for prim in needle_collision_capsules
        )
        needle_physx_collision_api_count = sum(
            "PhysxCollisionAPI" in prim.GetAppliedSchemas() for prim in needle_collision_capsules
        )
        needle_newton_collision_api_count = sum(
            "NewtonCollisionAPI" in prim.GetAppliedSchemas() for prim in needle_collision_capsules
        )
        expected_visual_material_path = f"{root_path}/Looks/NeedleSteelVisual"
        expected_physics_material_path = f"{root_path}/Looks/NeedleSteelPhysics"
        needle_collision_guide_purpose_count = sum(
            str(UsdGeom.Imageable(prim).GetPurposeAttr().Get()) == "guide" for prim in needle_collision_capsules
        )
        needle_collision_invisible_count = sum(
            str(UsdGeom.Imageable(prim).GetVisibilityAttr().Get()) == "invisible" for prim in needle_collision_capsules
        )
        needle_collision_physics_material_binding_count = sum(
            [str(target) for target in prim.GetRelationship("material:binding:physics").GetTargets()]
            == [expected_physics_material_path]
            and not prim.GetRelationship("material:binding").HasAuthoredTargets()
            for prim in needle_collision_capsules
        )
        physx_contact_offsets = [
            float(prim.GetAttribute("physxCollision:contactOffset").Get()) for prim in needle_collision_capsules
        ]
        physx_rest_offsets = [
            float(prim.GetAttribute("physxCollision:restOffset").Get()) for prim in needle_collision_capsules
        ]
        needle_physx_contact_offset_range_m = [
            min(physx_contact_offsets),
            max(physx_contact_offsets),
        ]
        needle_physx_rest_offset_range_m = [
            min(physx_rest_offsets),
            max(physx_rest_offsets),
        ]
        expected_contact_offsets = [capsule.contact_offset_m for capsule in expected_collision_capsules]
        expected_rest_offsets = [capsule.rest_offset_m for capsule in expected_collision_capsules]
        needle_physx_contact_offsets_match_profile = bool(
            np.isfinite(
                [
                    *physx_contact_offsets,
                    *physx_rest_offsets,
                ]
            ).all()
            and np.allclose(
                physx_contact_offsets,
                expected_contact_offsets,
                rtol=1.0e-6,
                atol=1.0e-12,
            )
            and np.allclose(
                physx_rest_offsets,
                expected_rest_offsets,
                rtol=0.0,
                atol=1.0e-12,
            )
        )
        needle_physics_material = stage.GetPrimAtPath(expected_physics_material_path)
        needle_friction_combine_mode = needle_physics_material.GetAttribute("physxMaterial:frictionCombineMode").Get()
        mass_api = UsdPhysics.MassAPI(stage.GetPrimAtPath(f"{root_path}/Needle"))
        needle_authored_mass_kg = float(mass_api.GetMassAttr().Get())
        center_of_mass = mass_api.GetCenterOfMassAttr().Get()
        diagonal_inertia = mass_api.GetDiagonalInertiaAttr().Get()
        principal_axes = mass_api.GetPrincipalAxesAttr().Get()
        principal_imaginary = principal_axes.GetImaginary()
        needle_center_of_mass_m = [float(center_of_mass[index]) for index in range(3)]
        needle_diagonal_inertia_kg_m2 = [float(diagonal_inertia[index]) for index in range(3)]
        needle_principal_axes_wxyz = [
            float(principal_axes.GetReal()),
            *(float(principal_imaginary[index]) for index in range(3)),
        ]
        expected_mass_properties = derived_needle.mass_properties
        needle_mass_properties_match_geometry = bool(
            np.isfinite(
                [
                    needle_authored_mass_kg,
                    *needle_center_of_mass_m,
                    *needle_diagonal_inertia_kg_m2,
                    *needle_principal_axes_wxyz,
                ]
            ).all()
            and needle_authored_mass_kg > 0.0
            and all(value > 0.0 for value in needle_diagonal_inertia_kg_m2)
            and np.isclose(
                needle_authored_mass_kg,
                derived_needle.mass_kg,
                rtol=1.0e-6,
                atol=0.0,
            )
            and np.allclose(
                needle_center_of_mass_m,
                expected_mass_properties.center_of_mass_m,
                rtol=1.0e-6,
                atol=1.0e-12,
            )
            and np.allclose(
                needle_diagonal_inertia_kg_m2,
                expected_mass_properties.diagonal_inertia_kg_m2,
                rtol=1.0e-6,
                atol=0.0,
            )
            and np.allclose(
                needle_principal_axes_wxyz,
                expected_mass_properties.principal_axes_wxyz,
                rtol=1.0e-6,
                atol=1.0e-7,
            )
            and np.isclose(
                np.linalg.norm(needle_principal_axes_wxyz),
                1.0,
                rtol=0.0,
                atol=1.0e-6,
            )
        )
        visual_prim = stage.GetPrimAtPath(f"{root_path}/Needle/Visual")
        needle_render_collision_separation_valid = bool(
            needle_collision_guide_purpose_count == derived_needle.collision_capsule_count
            and needle_collision_invisible_count == derived_needle.collision_capsule_count
            and needle_collision_physics_material_binding_count == derived_needle.collision_capsule_count
            and str(UsdGeom.Imageable(visual_prim).GetPurposeAttr().Get()) == "default"
            and str(UsdGeom.Imageable(visual_prim).GetVisibilityAttr().Get()) == "inherited"
            and [str(target) for target in visual_prim.GetRelationship("material:binding").GetTargets()]
            == [expected_visual_material_path]
            and not visual_prim.GetRelationship("material:binding:physics").HasAuthoredTargets()
        )
        looks_prim = stage.GetPrimAtPath(f"{root_path}/Looks")
        visual_material_prim = stage.GetPrimAtPath(expected_visual_material_path)
        physics_material_prim = stage.GetPrimAtPath(expected_physics_material_path)
        visual_shader_prim = stage.GetPrimAtPath(f"{expected_visual_material_path}/PreviewSurface")
        needle_material_organization_valid = bool(
            looks_prim.IsValid()
            and looks_prim.GetTypeName() == "Scope"
            and not stage.GetPrimAtPath(f"{root_path}/Materials").IsValid()
            and len([child for child in looks_prim.GetChildren() if child.GetTypeName() == "Material"]) == 2
            and visual_material_prim.GetTypeName() == "Material"
            and physics_material_prim.GetTypeName() == "Material"
            and "PhysicsMaterialAPI" not in visual_material_prim.GetAppliedSchemas()
            and "PhysxMaterialAPI" not in visual_material_prim.GetAppliedSchemas()
            and "PhysicsMaterialAPI" in physics_material_prim.GetAppliedSchemas()
            and "PhysxMaterialAPI" in physics_material_prim.GetAppliedSchemas()
            and visual_shader_prim.GetTypeName() == "Shader"
            and str(visual_shader_prim.GetAttribute("info:id").Get()) == "UsdPreviewSurface"
            and not stage.GetPrimAtPath(f"{expected_physics_material_path}/PreviewSurface").IsValid()
        )
        needle_prim = stage.GetPrimAtPath(f"{root_path}/Needle")
        needle_engine_schema_isolation_valid = bool(
            "PhysicsRigidBodyAPI" in needle_prim.GetAppliedSchemas()
            and "PhysicsMassAPI" in needle_prim.GetAppliedSchemas()
            and "PhysxRigidBodyAPI" in needle_prim.GetAppliedSchemas()
            and all("Newton" not in schema for schema in needle_prim.GetAppliedSchemas())
            and needle_physx_collision_api_count == derived_needle.collision_capsule_count
            and needle_newton_collision_api_count == 0
            and all("PhysicsCollisionAPI" in prim.GetAppliedSchemas() for prim in needle_collision_capsules)
            and all(
                not prim.GetAttribute("newton:contactGap").IsValid()
                and not prim.GetAttribute("newton:contactMargin").IsValid()
                for prim in needle_collision_capsules
            )
            and "PhysicsMaterialAPI" in physics_material_prim.GetAppliedSchemas()
            and "PhysxMaterialAPI" in physics_material_prim.GetAppliedSchemas()
            and all("Newton" not in schema for schema in physics_material_prim.GetAppliedSchemas())
        )
        normal_attribute = visual_prim.GetAttribute("primvars:normals")
        normal_index_attribute = visual_prim.GetAttribute("primvars:normals:indices")
        authored_normals = normal_attribute.Get()
        authored_normal_indices = normal_index_attribute.Get()
        if authored_normals is None or authored_normal_indices is None:
            raise RuntimeError("The needle visual mesh is missing indexed normals")
        needle_visual_normal_value_count = len(authored_normals)
        needle_visual_normal_index_count = len(authored_normal_indices)
        needle_visual_normal_interpolation = str(normal_attribute.GetMetadata("interpolation"))
        normal_values = np.asarray(
            authored_normals,
            dtype=np.float64,
        )
        normal_indices = np.asarray(
            authored_normal_indices,
            dtype=np.int64,
        )
        expected_normal_values = np.asarray(
            expected_needle_mesh.normals,
            dtype=np.float64,
        )
        expected_normal_indices = np.asarray(
            expected_needle_mesh.normal_indices,
            dtype=np.int64,
        )
        needle_visual_normals_valid = bool(
            needle_visual_normal_interpolation == "faceVarying"
            and not visual_prim.GetAttribute("normals").HasAuthoredValueOpinion()
            and normal_values.shape == expected_normal_values.shape
            and normal_indices.shape == expected_normal_indices.shape
            and np.isfinite(normal_values).all()
            and np.allclose(
                np.linalg.norm(normal_values, axis=1),
                1.0,
                rtol=0.0,
                atol=1.0e-4,
            )
            and np.allclose(
                normal_values,
                expected_normal_values,
                rtol=1.0e-6,
                atol=1.0e-7,
            )
            and np.array_equal(
                normal_indices,
                expected_normal_indices,
            )
        )
    report = {
        "schema": "dr.anmar.needle-native-physx-probe.v13",
        "asset_name": DR_ANMAR_NEEDLE_NAME if assembly else "DrAnmar Suture 4-0",
        "asset_id": DR_ANMAR_NEEDLE_ASSET_ID if assembly else "dr-anmar-suture-4-0",
        "asset_version": DR_ANMAR_NEEDLE_ASSET_VERSION if assembly else None,
        "asset": str(args.asset.resolve()),
        "physics_dt_s": 0.0005,
        "steps": int(args.steps),
        "segment_count": int(segments.count),
        "joint_count": int(joint_count),
        "factory_swage": bool(factory_swage.IsValid()) if assembly else None,
        "needle_collision_capsule_count": len(needle_collision_capsules) if assembly else None,
        "needle_collision_explicit_extent_count": needle_collision_extent_count,
        "needle_friction_combine_mode": needle_friction_combine_mode,
        "needle_authored_mass_kg": needle_authored_mass_kg,
        "needle_center_of_mass_m": needle_center_of_mass_m,
        "needle_diagonal_inertia_kg_m2": needle_diagonal_inertia_kg_m2,
        "needle_principal_axes_wxyz": needle_principal_axes_wxyz,
        "needle_mass_properties_match_geometry": needle_mass_properties_match_geometry,
        "needle_physx_collision_api_count": needle_physx_collision_api_count,
        "needle_newton_collision_api_count": needle_newton_collision_api_count,
        "needle_physx_contact_offset_range_m": needle_physx_contact_offset_range_m,
        "needle_physx_rest_offset_range_m": needle_physx_rest_offset_range_m,
        "needle_physx_contact_offsets_match_profile": needle_physx_contact_offsets_match_profile,
        "needle_engine_schema_isolation_valid": needle_engine_schema_isolation_valid,
        "needle_visual_normal_value_count": needle_visual_normal_value_count,
        "needle_visual_normal_index_count": needle_visual_normal_index_count,
        "needle_visual_normal_interpolation": needle_visual_normal_interpolation,
        "needle_visual_normals_valid": needle_visual_normals_valid,
        "needle_collision_guide_purpose_count": needle_collision_guide_purpose_count,
        "needle_collision_invisible_count": needle_collision_invisible_count,
        "needle_collision_physics_material_binding_count": needle_collision_physics_material_binding_count,
        "needle_render_collision_separation_valid": needle_render_collision_separation_valid,
        "needle_material_organization_valid": needle_material_organization_valid,
        "needle_physics_variant_selection": needle_physics_variant_selection,
        "suture_physics_variant_selection": suture_physics_variant_selection,
        "physics_variant_contract_valid": physics_variant_contract_valid,
        "needle_base_layer_name": needle_base_layer_name,
        "needle_geometry_layer_name": needle_geometry_layer_name,
        "needle_materials_layer_name": needle_materials_layer_name,
        "needle_neutral_physics_layer_name": needle_neutral_physics_layer_name,
        "needle_physx_layer_name": needle_physx_layer_name,
        "needle_asset_structure_source_ownership_valid": needle_asset_structure_source_ownership_valid,
        "suture_base_layer_name": suture_base_layer_name,
        "suture_geometry_layer_name": suture_geometry_layer_name,
        "suture_materials_layer_name": suture_materials_layer_name,
        "suture_neutral_physics_layer_name": suture_neutral_physics_layer_name,
        "suture_physx_layer_name": suture_physx_layer_name,
        "suture_asset_structure_source_ownership_valid": suture_asset_structure_source_ownership_valid,
        "suture_physx_collision_api_count": suture_physx_collision_api_count,
        "suture_hybrid_ccd_body_count": suture_hybrid_ccd_body_count,
        "suture_physx_contact_offset_range_m": suture_physx_contact_offset_range_m,
        "suture_physx_rest_offset_range_m": suture_physx_rest_offset_range_m,
        "suture_physx_contact_offsets_match_profile": suture_physx_contact_offsets_match_profile,
        "suture_material_bindings_valid": suture_material_bindings_valid,
        "suture_visual_mesh_count": suture_visual_mesh_count,
        "suture_visual_mesh_vertex_count": suture_visual_mesh_vertex_count,
        "suture_visual_normals_valid_count": suture_visual_normals_valid_count,
        "suture_visual_uv_value_count": suture_visual_uv_value_count,
        "suture_visual_uv_index_count": suture_visual_uv_index_count,
        "suture_visual_uv_valid_count": suture_visual_uv_valid_count,
        "suture_material_texture_path": suture_material_texture_path,
        "suture_material_texture_exists": suture_material_texture_exists,
        "suture_pbr_material_graph_valid": suture_pbr_material_graph_valid,
        "suture_collision_capsule_count": suture_collision_capsule_count,
        "suture_collision_guide_purpose_count": suture_collision_guide_purpose_count,
        "suture_collision_invisible_count": suture_collision_invisible_count,
        "suture_collision_physics_material_binding_count": suture_collision_physics_material_binding_count,
        "suture_collider_cylinder_height_range_m": suture_collider_cylinder_height_range_m,
        "suture_minimum_visual_collision_margin_m": suture_minimum_visual_collision_margin_m,
        "suture_interface_minimum_visual_collision_margin_m": suture_interface_minimum_visual_collision_margin_m,
        "suture_interface_visual_mesh_valid": suture_interface_visual_mesh_valid,
        "suture_render_collision_separation_valid": suture_render_collision_separation_valid,
        "initial_swage_distance_m": initial_swage_distance_m,
        "final_swage_distance_m": final_swage_distance_m,
        "finite_transforms": finite,
        "free_end_drop_m": free_end_drop,
        "maximum_segment_displacement_m": float(displacement.max()),
        "native_rigid_contact_bodies": int(segments.count),
        "authored_pose_writes_after_reset": 0,
        "current_thread_modified": False,
        "clinical_validation": False,
    }
    report["passed"] = bool(
        finite
        and report["segment_count"] == 360
        and report["joint_count"] == 360
        and free_end_drop > 0.0001
        and report["maximum_segment_displacement_m"] < 0.5
        and (
            not assembly
            or (
                report["factory_swage"]
                and report["needle_collision_capsule_count"] == derived_needle.collision_capsule_count
                and report["needle_collision_explicit_extent_count"] == derived_needle.collision_capsule_count
                and report["needle_friction_combine_mode"] == "max"
                and report["needle_mass_properties_match_geometry"]
                and report["needle_physx_collision_api_count"] == derived_needle.collision_capsule_count
                and report["needle_newton_collision_api_count"] == 0
                and report["needle_physx_contact_offsets_match_profile"]
                and report["needle_engine_schema_isolation_valid"]
                and report["needle_visual_normal_value_count"] == len(expected_needle_mesh.normals)
                and report["needle_visual_normal_index_count"] == len(expected_needle_mesh.normal_indices)
                and report["needle_visual_normals_valid"]
                and report["needle_collision_guide_purpose_count"] == derived_needle.collision_capsule_count
                and report["needle_collision_invisible_count"] == derived_needle.collision_capsule_count
                and report["needle_collision_physics_material_binding_count"] == derived_needle.collision_capsule_count
                and report["needle_render_collision_separation_valid"]
                and report["needle_material_organization_valid"]
                and report["physics_variant_contract_valid"]
                and report["needle_asset_structure_source_ownership_valid"]
                and report["suture_asset_structure_source_ownership_valid"]
                and report["suture_physx_collision_api_count"] == 361
                and report["suture_hybrid_ccd_body_count"] == 361
                and report["suture_physx_contact_offsets_match_profile"]
                and report["suture_material_bindings_valid"]
                and report["suture_visual_mesh_count"] == 360
                and report["suture_visual_normals_valid_count"] == 360
                and report["suture_visual_uv_value_count"] == 360 * 441
                and report["suture_visual_uv_index_count"] == 360 * 1440
                and report["suture_visual_uv_valid_count"] == 360
                and report["suture_material_texture_exists"]
                and report["suture_pbr_material_graph_valid"]
                and report["suture_collision_capsule_count"] == 361
                and report["suture_collision_guide_purpose_count"] == 361
                and report["suture_collision_invisible_count"] == 361
                and report["suture_collision_physics_material_binding_count"] == 361
                and report["suture_minimum_visual_collision_margin_m"] is not None
                and report["suture_minimum_visual_collision_margin_m"]
                >= -float(
                    suture_profile["geometry"]["visual_representation"]["binary_visual_point_containment_tolerance_m"]
                )
                and report["suture_interface_minimum_visual_collision_margin_m"] is not None
                and report["suture_interface_minimum_visual_collision_margin_m"]
                >= -float(
                    suture_profile["geometry"]["visual_representation"]["binary_visual_point_containment_tolerance_m"]
                )
                and report["suture_interface_visual_mesh_valid"]
                and report["suture_render_collision_separation_valid"]
                and initial_swage_distance_m is not None
                and initial_swage_distance_m < 0.0001
                and final_swage_distance_m is not None
                and final_swage_distance_m < 0.0005
            )
        )
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
