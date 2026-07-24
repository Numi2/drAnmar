#!/usr/bin/env python3
"""Headless native-PhysX smoke probe for the independent Dr.Anmar suture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dr_anmar_needle_model import derive_needle, load_needle_profile
from dr_anmar_suture_integration import DR_ANMAR_NEEDLE_ASSET_ID, DR_ANMAR_NEEDLE_ASSET_VERSION, DR_ANMAR_NEEDLE_NAME

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
from pxr import Gf, UsdGeom, UsdPhysics  # noqa: E402

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
    derived_needle = derive_needle(needle_profile)
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
    if assembly:
        needle_collision_capsules = [
            prim
            for prim in stage.Traverse()
            if prim.GetTypeName() == "Capsule" and str(prim.GetPath()).startswith(f"{root_path}/Needle/Collision/C")
        ]
        needle_collision_extent_count = sum(
            UsdGeom.Capsule(prim).GetExtentAttr().HasAuthoredValueOpinion() for prim in needle_collision_capsules
        )
        needle_material = stage.GetPrimAtPath(f"{root_path}/Materials/NeedleSteel")
        needle_friction_combine_mode = needle_material.GetAttribute("physxMaterial:frictionCombineMode").Get()
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
    report = {
        "schema": "dr.anmar.needle-native-physx-probe.v2",
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
