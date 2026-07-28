#!/usr/bin/env python3
"""Legacy CUDA controller exercise for the DrAnmar SafePlane asset.

This script still injects caller-authored force, work, fluid, energy, visibility,
and injury-adjacent values into the former outcome API. Those public entry
points now intentionally fail closed until a SafePlane SceneEvidenceEnvelope and
shared-mechanics bridge exists. Retain this file as a migration fixture; it is
not a current validator or outcome-evidence producer.

Run through Isaac Lab:

    ./isaaclab.sh -p examples/validate_safeplane_dissection_runtime.py \
        --headless --device cuda:0 --representation standalone
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import math
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = (
    ROOT
    / "source/extensions/orbit.surgical.assets/orbit/surgical/assets"
    / "safeplane_dissection_robot.py"
)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--representation", choices=("standalone", "franka"), required=True)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import carb
import numpy as np
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaacsim.core.simulation_manager import PhysxGpuCfg, PhysxScene
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt


def load_helper():
    spec = importlib.util.spec_from_file_location(
        "dranmar_safeplane_dissection_runtime",
        HELPER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load runtime helper from {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def world_transform(stage, path: str):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing transform prim: {path}")
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )


def wxyz(quaternion):
    imaginary = quaternion.GetImaginary()
    return (
        float(quaternion.GetReal()),
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
    )


def main() -> int:
    if args.steps < 20:
        raise ValueError("--steps must be at least 20")

    engine_errors: list[str] = []
    carb_logging = carb.logging.acquire_logging()

    def record_engine_error(source, level, _filename, _line_number, message):
        if level >= carb.logging.LEVEL_ERROR and len(engine_errors) < 30:
            engine_errors.append(f"{source}: {message.strip()}")

    logger_handle = carb_logging.add_logger(record_engine_error)
    helper = load_helper()
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device)
    )
    physx_scene = PhysxScene(sim.cfg.physics_prim_path)
    physx_scene.set_gpu_configuration(
        PhysxGpuCfg(gpu_max_deformable_surface_contacts=2**24)
    )
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/GroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    if args.representation == "standalone":
        root_path = "/World/SafePlaneDissectionTool"
        robot = Articulation(
            helper.make_tool_cfg(
                root_path,
                position=(0.0, 0.0, 0.10),
            )
        )
        tool_path = root_path
    else:
        root_path = "/World/Robot"
        robot = Articulation(
            helper.make_franka_safeplane_dissection_robot_cfg(
                prim_path=root_path,
            )
        )
        tool_path = f"{root_path}/DrAnmarSafePlaneDissectionTool"

    # Register the layered tissue to the authored TCP before the one and only
    # physics reset. Rebuilding physics after an Articulation tensor view has
    # initialized invalidates that view in Isaac Lab 6.1.
    stage = omni.usd.get_context().get_stage()
    tcp_transform = world_transform(stage, helper.frame_path(tool_path, "safeplane_tcp"))
    tissue_origin = tcp_transform.Transform(Gf.Vec3d(0.0, 0.0, -0.025))
    tissue_orientation = wxyz(tcp_transform.ExtractRotationQuat())

    tissue_root = "/World/DrAnmarSafePlaneTissue"
    helper.spawn_tissue_demo(
        tissue_root,
        translation=tuple(float(value) for value in tissue_origin),
        orientation_wxyz=tissue_orientation,
    )
    deformables = helper.apply_tissue_surface_deformables(
        tissue_root,
        self_collision=False,
        stage=stage,
    )
    sequence = helper.SafePlaneDissectionSequenceController(
        tissue_root=tissue_root,
        tool_path=tool_path,
    )
    connections = sequence.initialize_physical_connections(stage=stage)
    fixture_paths = connections["target_bed_fixtures"]
    traction_paths = [
        cell.attachment_path for cell in connections["traction"]
    ]
    bridge_paths = [
        path
        for state in sequence.bridges.states.values()
        for path in state.attachment_paths
    ]
    protected_paths = connections["protected_structures"]
    if tuple(map(len, (fixture_paths, traction_paths, bridge_paths, protected_paths))) != (2, 8, 56, 6):
        raise RuntimeError(
            "Unexpected physical-connection counts: "
            f"fixtures={len(fixture_paths)}, traction={len(traction_paths)}, "
            f"bridges={len(bridge_paths)}, protected={len(protected_paths)}"
        )

    attachment_paths = fixture_paths + traction_paths + bridge_paths + protected_paths
    attachment_prims = [stage.GetPrimAtPath(path) for path in attachment_paths]
    if any(not prim.IsValid() for prim in attachment_prims):
        raise RuntimeError("Not all physical attachments were authored")
    attachment_types = [prim.GetTypeName() for prim in attachment_prims]
    for prim in attachment_prims:
        if not prim.GetRelationship("omniphysics:src0").GetTargets():
            raise RuntimeError(f"Attachment has no deformable source: {prim.GetPath()}")
        if not prim.GetRelationship("omniphysics:src1").GetTargets():
            raise RuntimeError(f"Attachment has no rigid source: {prim.GetPath()}")

    corrected_joint_frames = []
    for index in sorted(sequence.bridges.states):
        path = f"{tissue_root}/AdhesionBridges/Bridge_{index:02d}/ContinuityJoint"
        joint = UsdPhysics.FixedJoint.Get(stage, path)
        if not joint:
            raise RuntimeError(f"Missing adhesion continuity joint: {path}")
        local0 = joint.GetLocalPos0Attr().Get()
        local1 = joint.GetLocalPos1Attr().Get()
        values = (tuple(local0), tuple(local1))
        if not (
            np.allclose(local0, (0.0, 0.0, -0.011), atol=1.0e-7)
            and np.allclose(local1, (0.0, 0.0, 0.011), atol=1.0e-7)
        ):
            raise RuntimeError(f"Adhesion bridge joint does not meet at its midpoint: {path} {values}")
        corrected_joint_frames.append(values)

    helper.ensure_dissection_particle_system(
        stage=stage,
        physics_scene_path=sim.cfg.physics_prim_path,
    )
    sim.reset()
    robot.update(sim.get_physics_dt())
    initialized_tcp = world_transform(
        stage, helper.frame_path(tool_path, "safeplane_tcp")
    ).ExtractTranslation()
    initialized_plane = world_transform(stage, tissue_root).Transform(
        Gf.Vec3d(0.0, 0.0, 0.025)
    )
    registration_correction_m = float(
        (initialized_tcp - initialized_plane).GetLength()
    )
    if registration_correction_m > 0.003:
        initialized_tcp_transform = world_transform(
            stage, helper.frame_path(tool_path, "safeplane_tcp")
        )
        desired_tissue_transform = Gf.Matrix4d(1.0)
        desired_tissue_transform.SetTranslate(Gf.Vec3d(0.0, 0.0, -0.025))
        desired_tissue_transform = (
            desired_tissue_transform * initialized_tcp_transform
        )
        tissue_xform = UsdGeom.Xformable(stage.GetPrimAtPath(tissue_root))
        tissue_xform.ClearXformOpOrder()
        tissue_xform.AddTransformOp().Set(desired_tissue_transform)
        initialized_plane = world_transform(stage, tissue_root).Transform(
            Gf.Vec3d(0.0, 0.0, 0.025)
        )
    registration_error_m = float(
        (initialized_tcp - initialized_plane).GetLength()
    )
    if registration_error_m > 0.003:
        raise RuntimeError(
            f"Tissue-to-TCP registration changed across reset: {registration_error_m} m"
        )
    initial_steps = 10
    for _ in range(initial_steps):
        sim.step(render=not args.headless)
        robot.update(sim.get_physics_dt())

    force_events = sequence.traction.update_force(1.4, 1.4, stage=stage)
    if force_events:
        raise RuntimeError(f"Nominal traction unexpectedly released cells: {force_events}")

    vessel_point = sequence.protected.topology()["vessel"]["centerline_m"][2]
    blocked_energy = sequence.energy_action(
        vessel_point,
        dt=0.1,
        contact_force_n=1.5,
        requested_power_w=22.0,
        stage=stage,
    )
    blocked_scissors = sequence.scissors_action(
        vessel_point,
        0.0,
        stage=stage,
    )
    if not blocked_energy.get("blocked") or blocked_scissors.get("released"):
        raise RuntimeError("Protected-structure interlocks did not block dangerous actions")
    if not all(state.intact for state in sequence.protected.states.values()):
        raise RuntimeError("A protected structure changed state during blocked actions")

    release_modes = {
        "blunt_spreading": [],
        "hydrodissection": [],
        "low_energy_dissection": [],
        "guarded_scissors": [],
    }
    safety_reroutes = []
    loose_states = [
        state for state in sequence.bridges.states.values()
        if state.bridge_class == "loose_connective_fibre"
    ]
    for offset, state in enumerate(loose_states):
        if not sequence.protected.evaluate_action(state.position, "blunt")["authorized"]:
            raise RuntimeError(f"Loose bridge {state.index} is inside a protected clearance")
        if offset % 2 == 0:
            released = sequence.bridges.apply_blunt_work(
                state.position,
                state.mechanical_threshold_j * 1.01,
                radius_m=1.0e-4,
                stage=stage,
            )
            mode = "blunt_spreading"
        else:
            released = sequence.bridges.apply_hydro_volume(
                state.position,
                state.hydro_threshold_ml * 1.01,
                radius_m=1.0e-4,
                stage=stage,
            )
            mode = "hydrodissection"
        if state.index not in released:
            raise RuntimeError(f"{mode} failed to release loose bridge {state.index}")
        release_modes[mode].append(state.index)

    vascular_states = [
        state for state in sequence.bridges.states.values()
        if state.bridge_class == "vascularized_adhesion"
    ]
    for state in vascular_states:
        energy_safety = sequence.protected.evaluate_action(state.position, "energy")
        if energy_safety["authorized"]:
            released = sequence.bridges.apply_energy(
                state.position,
                state.energy_threshold_j * 1.01,
                radius_m=1.0e-4,
                stage=stage,
            )
            mode = "low_energy_dissection"
        else:
            hydro_safety = sequence.protected.evaluate_action(state.position, "hydro")
            if not hydro_safety["authorized"]:
                raise RuntimeError(
                    f"Vascularized bridge {state.index} has no safe release modality"
                )
            released = sequence.bridges.apply_hydro_volume(
                state.position,
                state.hydro_threshold_ml / 0.62 * 1.01,
                radius_m=1.0e-4,
                stage=stage,
            )
            mode = "hydrodissection"
            safety_reroutes.append({
                "bridge_index": state.index,
                "blocked_modality": "low_energy_dissection",
                "selected_modality": "hydrodissection",
                "nearest_structure": energy_safety["nearest_structure"],
                "distance_m": energy_safety["distance_m"],
                "required_energy_clearance_m": energy_safety["minimum_clearance_m"],
            })
        if state.index not in released:
            raise RuntimeError(f"{mode} failed to release vascularized bridge {state.index}")
        release_modes[mode].append(state.index)

    dense_states = [
        state for state in sequence.bridges.states.values()
        if state.bridge_class == "dense_fibrous_band"
    ]
    for state in dense_states:
        cut = sequence.scissors_action(
            state.position,
            0.010,
            stage=stage,
        )
        if not cut.get("authorized") or not cut.get("released") or cut.get("bridge_index") != state.index:
            raise RuntimeError(f"Guarded scissors failed to release dense bridge {state.index}: {cut}")
        release_modes["guarded_scissors"].append(state.index)

    released_count = sum(state.released for state in sequence.bridges.states.values())
    if released_count != 28:
        raise RuntimeError(f"Expected 28 released adhesion bridges, got {released_count}")
    remaining_joint_paths = [
        f"{tissue_root}/AdhesionBridges/Bridge_{index:02d}/ContinuityJoint"
        for index in sequence.bridges.states
        if (
            stage.GetPrimAtPath(
                f"{tissue_root}/AdhesionBridges/Bridge_{index:02d}/ContinuityJoint"
            ).IsValid()
            and stage.GetPrimAtPath(
                f"{tissue_root}/AdhesionBridges/Bridge_{index:02d}/ContinuityJoint"
            ).IsActive()
        )
    ]
    if remaining_joint_paths:
        raise RuntimeError(f"Released bridge joints remain: {remaining_joint_paths}")

    completion = sequence.verify(visibility_fraction=0.95, traction_stable=True)
    if not completion["complete"]:
        raise RuntimeError(f"Connectivity completion verifier failed: {completion}")

    ledger = helper.FluidLedger()
    emission = helper.emit_hydro_burst(
        tool_path,
        ledger,
        requested_ml=0.22,
        stage=stage,
    )
    if emission["particle_count"] < 7:
        raise RuntimeError(f"Hydro burst emitted too few particles: {emission}")
    particle_prim = stage.GetPrimAtPath(emission["particle_set_path"])
    points = UsdGeom.Points(particle_prim)
    positions = list(points.GetPointsAttr().Get() or [])
    throat = world_transform(
        stage, helper.frame_path(tool_path, "suction_center")
    ).ExtractTranslation()
    for index in range(7):
        positions[index] = Gf.Vec3f(throat)
    points.GetPointsAttr().Set(Vt.Vec3fArray(positions))
    suction = sequence.suction.update_particles(
        tool_path,
        ledger,
        dt=0.01,
        opening=1.0,
        stage=stage,
    )
    if suction["captured"] < 7 or suction["aspirated_ml"] <= 0.0:
        raise RuntimeError(f"Annular suction did not capture injected particles: {suction}")
    if abs(ledger.balance_error_ml) > 1.0e-9:
        raise RuntimeError(f"Fluid conservation error: {ledger.snapshot()}")

    left_released = sequence.traction.release_side("left", stage=stage)
    right_released = sequence.traction.release_side("right", stage=stage)
    if len(left_released) != 4 or len(right_released) != 4:
        raise RuntimeError("Traction attachments were not fully released")
    active_attachment_paths = [
        path
        for path in fixture_paths + bridge_paths + protected_paths
        if stage.GetPrimAtPath(path).IsValid()
    ]
    if len(active_attachment_paths) != 64:
        raise RuntimeError(f"Expected 64 retained attachments, got {len(active_attachment_paths)}")
    if any(stage.GetPrimAtPath(path).IsValid() for path in traction_paths):
        raise RuntimeError("Released traction attachment prims remain")

    for _ in range(args.steps - initial_steps):
        sim.step(render=not args.headless)
        robot.update(sim.get_physics_dt())

    joint_pos = helper.tensor_value(robot.data.joint_pos).detach().cpu().numpy()
    if not np.isfinite(joint_pos).all():
        raise RuntimeError("Non-finite articulation state after CUDA simulation")
    joint_names = list(robot.joint_names)
    expected_joint_count = 17 if args.representation == "standalone" else 24
    if len(joint_names) != expected_joint_count:
        raise RuntimeError(
            f"Expected {expected_joint_count} joints, got {len(joint_names)}: {joint_names}"
        )
    missing_tool = sorted(set(helper.TOOL_JOINTS.values()) - set(joint_names))
    if missing_tool:
        raise RuntimeError(f"Tool articulation is missing joints: {missing_tool}")
    if args.representation == "franka":
        missing_arm = sorted(
            {f"panda_joint{index}" for index in range(1, 8)} - set(joint_names)
        )
        if missing_arm:
            raise RuntimeError(f"Franka articulation is missing arm joints: {missing_arm}")

    mount_contract = None
    if args.representation == "franka":
        mount_path = f"{root_path}/dranmar_safeplane_mount_joint"
        mount = UsdPhysics.FixedJoint.Get(stage, mount_path)
        body0 = [str(path) for path in mount.GetBody0Rel().GetTargets()]
        rotation = mount.GetLocalRot0Attr().Get()
        rotation_values = wxyz(rotation)
        expected = (
            math.cos(math.radians(-45.0) / 2.0),
            0.0,
            0.0,
            math.sin(math.radians(-45.0) / 2.0),
        )
        if len(body0) != 1 or not body0[0].endswith("/panda_link8"):
            raise RuntimeError(f"Franka mount body is not panda_link8: {body0}")
        if not (
            np.allclose(rotation_values, expected, atol=1.0e-5)
            or np.allclose(rotation_values, tuple(-value for value in expected), atol=1.0e-5)
        ):
            raise RuntimeError(
                f"Franka mount does not preserve the stock -45 degree frame: {rotation_values}"
            )
        mount_contract = {
            "body0": body0[0],
            "local_rotation_wxyz": rotation_values,
            "stock_hand_frame_preserved": True,
        }

    required_deformable_schemas = {
        "OmniPhysicsDeformableBodyAPI",
        "OmniPhysicsSurfaceDeformableSimAPI",
        "PhysxSurfaceDeformableBodyAPI",
    }
    deformable_schemas: dict[str, list[str]] = {}
    for mesh_path in deformables["mesh_paths"]:
        prim = stage.GetPrimAtPath(mesh_path)
        applied = list(prim.GetAppliedSchemas())
        missing = sorted(required_deformable_schemas - set(applied))
        if missing:
            raise RuntimeError(f"{mesh_path} omitted deformable schemas: {missing}")
        deformable_schemas[mesh_path.rsplit("/", 1)[-1]] = applied

    carb_logging.remove_logger(logger_handle)
    if engine_errors:
        raise RuntimeError("Isaac runtime emitted engine errors:\n" + "\n".join(engine_errors))

    result = {
        "schema": "dranmar.safeplane-dissection-runtime-diagnostic.v2",
        "status": "controller_exercise_only",
        "qualification_scope": (
            "composition_interlocks_ledgers_and_controller_threshold_logic"
        ),
        "representation": args.representation,
        "steps": args.steps,
        "device": args.device,
        "gpu_max_deformable_surface_contacts": (
            physx_scene.get_gpu_configuration().gpu_max_deformable_surface_contacts
        ),
        "isaaclab_distribution_version": distribution_version("isaaclab"),
        "isaacsim_distribution_version": distribution_version("isaacsim"),
        "joint_count": len(joint_names),
        "joint_names": joint_names,
        "finite_joint_state": True,
        "engine_error_count": 0,
        "deformable_applied_schemas": deformable_schemas,
        "surface_self_collision": deformables["self_collision"],
        "tissue_tcp_registration_error_m": registration_error_m,
        "tissue_tcp_registration_correction_m": registration_correction_m,
        "attachment_type_names": sorted(set(attachment_types)),
        "fixture_attachment_count": len(fixture_paths),
        "traction_attachment_count": len(traction_paths),
        "bridge_attachment_count": len(bridge_paths),
        "protected_structure_attachment_count": len(protected_paths),
        "retained_attachment_count": len(active_attachment_paths),
        "corrected_adhesion_joint_count": len(corrected_joint_frames),
        "released_bridge_count": released_count,
        "physical_bridge_release_qualified": False,
        "bridge_release_input_source": (
            "direct_controller_threshold_injection_at_authored_coordinates"
        ),
        "release_modes": release_modes,
        "safety_reroutes": safety_reroutes,
        "protected_structures_intact": all(
            state.intact for state in sequence.protected.states.values()
        ),
        "blocked_energy_interlock": bool(blocked_energy.get("blocked")),
        "blocked_scissors_interlock": not bool(blocked_scissors.get("released")),
        "completion": completion,
        "fluid_emission": emission,
        "fluid_suction": suction,
        "physical_hydro_suction_qualified": False,
        "fluid_capture_input_source": (
            "emitted_particle_positions_moved_to_authored_throat_for_"
            "capacity_and_ledger_controller_exercise"
        ),
        "fluid_ledger": ledger.snapshot(),
        "franka_mount_contract": mount_contract,
        "clinical_validation": False,
        "medical_device": False,
    }
    payload = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        try:
            simulation_app.close(skip_cleanup=True)
        except TypeError:
            simulation_app.close()
        raise SystemExit(exit_code)
