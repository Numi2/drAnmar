#!/usr/bin/env python3
"""Headless CUDA smoke test for the wound-preparation system.

Run through Isaac Lab:

    ./isaaclab.sh -p examples/validate_wound_preparation_runtime.py \
        --headless --device cuda:0 --representation standalone

Use ``--representation franka`` to qualify the combined Panda-link8 payload.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = (
    ROOT
    / "source/extensions/orbit.surgical.assets/orbit/surgical/assets"
    / "wound_preparation_robot.py"
)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--representation", choices=("standalone", "franka"), required=True)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--render-width", type=int, default=160)
parser.add_argument("--render-height", type=int, default=120)
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import carb
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaacsim.core.simulation_manager import PhysxGpuCfg, PhysxScene
from pxr import Usd, UsdGeom
from dranmar_native_qualification import (
    attach_registered_camera_prims,
    capture_registered_camera_frames,
    command_phase_targets,
    spawn_fixed_standalone_articulation,
)


def load_helper():
    spec = importlib.util.spec_from_file_location(
        "dranmar_wound_preparation_runtime",
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


def tensor_value(value):
    return value.torch if hasattr(value, "torch") else value


def main() -> int:
    if args.steps <= 0:
        raise ValueError("--steps must be greater than zero")

    engine_errors: list[str] = []
    carb_logging = carb.logging.acquire_logging()

    def record_engine_error(source, level, _filename, _line_number, message):
        if level >= carb.logging.LEVEL_ERROR and len(engine_errors) < 20:
            engine_errors.append(f"{source}: {message.strip()}")

    logger_handle = carb_logging.add_logger(record_engine_error)
    helper = load_helper()
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device)
    )
    physx_scene = PhysxScene(sim.cfg.physics_prim_path)
    physx_scene.set_gpu_configuration(
        PhysxGpuCfg(gpu_max_deformable_surface_contacts=2**21)
    )
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/GroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    stage = omni.usd.get_context().get_stage()
    qualification_mount = None
    if args.representation == "standalone":
        root_path = "/World/WoundPreparationTool"
        robot, tool_path, qualification_mount = (
            spawn_fixed_standalone_articulation(
                stage,
                tool_cfg=helper.make_tool_cfg(
                    root_path,
                    irrigation_state="loaded",
                    collection_state="empty",
                    position=(0.0, 0.0, 0.45),
                ),
            )
        )
    else:
        root_path = "/World/Robot"
        robot = Articulation(
            helper.make_franka_wound_preparation_robot_cfg(
                prim_path=root_path,
                irrigation_state="loaded",
                collection_state="empty",
            )
        )
        tool_path = f"{root_path}/DrAnmarWoundPreparationTool"

    tcp_prim = stage.GetPrimAtPath(
        helper.frame_path(tool_path, "wound_preparation_tcp")
    )
    if not tcp_prim.IsValid():
        raise RuntimeError("Wound-preparation TCP is missing")
    tcp_world = UsdGeom.Xformable(tcp_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    wound_translation = tuple(
        float(value) for value in tcp_world.ExtractTranslation()
    )
    wound_root = "/World/DrAnmarWoundBed"
    helper.spawn_wound_bed_demo(wound_root, translation=wound_translation)
    deformable = helper.apply_wound_surface_deformable(wound_root, stage=stage)
    attachments = helper.attach_demo_debris(wound_root, stage=stage)
    attachment_prims = [stage.GetPrimAtPath(path) for path in attachments.values()]
    if len(attachment_prims) != 7 or any(not prim.IsValid() for prim in attachment_prims):
        raise RuntimeError("The wound demo did not author seven valid debris attachments")
    attachment_types = [prim.GetTypeName() for prim in attachment_prims]
    if any(not type_name for type_name in attachment_types):
        raise RuntimeError(f"Debris attachment type is missing: {attachment_types}")
    particle_paths = helper.ensure_irrigation_particle_system(stage=stage)
    ledger = helper.FluidLedger()
    emitted = helper.emit_irrigation_burst(
        tool_path,
        ledger,
        requested_ml=0.10,
        random_seed=17,
        stage=stage,
        particle_set_path=particle_paths["particle_set_path"],
    )
    camera_paths = attach_registered_camera_prims(
        stage,
        tool_path=tool_path,
        frame_path=helper.frame_path,
        camera_names=helper.REGISTERED_CAMERA_FRAMES,
    )

    sim.reset()
    robot.update(sim.get_physics_dt())
    joint_names = list(robot.joint_names)
    missing_joints = sorted(set(helper.TOOL_JOINTS.values()) - set(joint_names))
    if missing_joints:
        raise RuntimeError(f"Tool articulation is missing joints: {missing_joints}")
    if args.representation == "franka":
        missing_arm = sorted(
            {f"panda_joint{index}" for index in range(1, 8)}
            - set(joint_names)
        )
        if missing_arm:
            raise RuntimeError(f"Franka articulation is missing arm joints: {missing_arm}")

    canonical_phases = (
        "inspect",
        "contact",
        "pre_rinse",
        "aspirate",
        "debride",
        "post_rinse",
        "dry",
        "verify",
    )
    steps_per_phase = max(30, args.steps // len(canonical_phases))
    phase_convergence = {}
    for phase in canonical_phases:
        authored = helper.phase_targets(phase)
        rotor_velocity = authored.pop("debridement_rotor_joint_velocity")
        phase_convergence[phase] = command_phase_targets(
            robot,
            sim,
            joint_names=joint_names,
            targets=authored,
            velocity_targets={
                "debridement_rotor_joint": rotor_velocity,
            },
            steps=steps_per_phase,
            render=not args.headless,
            maximum_velocity_error=8.0,
            phase_name=phase,
        )
    _, rendered_sensor_evidence = capture_registered_camera_frames(
        sim,
        camera_paths,
        width=args.render_width,
        height=args.render_height,
    )
    joint_pos = tensor_value(robot.data.joint_pos).detach().cpu().numpy()
    if not np.isfinite(joint_pos).all():
        raise RuntimeError("Non-finite articulation joint state after workflow")

    particle_prim = stage.GetPrimAtPath(particle_paths["particle_set_path"])
    particle_points = list(UsdGeom.Points(particle_prim).GetPointsAttr().Get() or [])
    if len(particle_points) != emitted["particle_count"]:
        raise RuntimeError(
            f"Particle set contains {len(particle_points)} points; "
            f"expected {emitted['particle_count']}"
        )

    wound_mesh = stage.GetPrimAtPath(deformable["mesh_path"])
    applied_schemas = list(wound_mesh.GetAppliedSchemas())
    required_deformable_schemas = {
        "OmniPhysicsDeformableBodyAPI",
        "OmniPhysicsSurfaceDeformableSimAPI",
        "PhysxSurfaceDeformableBodyAPI",
    }
    missing_schemas = sorted(required_deformable_schemas - set(applied_schemas))
    if missing_schemas:
        raise RuntimeError(f"Surface deformable cooking omitted schemas: {missing_schemas}")
    carb_logging.remove_logger(logger_handle)
    if engine_errors:
        raise RuntimeError("Isaac runtime emitted engine errors:\n" + "\n".join(engine_errors))

    result = {
        "schema": "dr.anmar.wound-preparation-cuda-qualification.v2",
        "status": "diagnostic_complete",
        "qualification_scope": (
            "native_authored_mechanism_particle_and_sensor_execution"
        ),
        "representation": args.representation,
        "steps": args.steps,
        "device": args.device,
        "gpu_max_deformable_surface_contacts": (
            physx_scene.get_gpu_configuration().gpu_max_deformable_surface_contacts
        ),
        "isaaclab_distribution_version": distribution_version("isaaclab"),
        "isaacsim_distribution_version": distribution_version("isaacsim"),
        "tool_joint_names": joint_names,
        "qualification_fixed_joint": qualification_mount,
        "phase_convergence": phase_convergence,
        "registered_camera_count": len(camera_paths),
        "rendered_sensor_evidence": rendered_sensor_evidence,
        "finite_joint_state": True,
        "engine_error_count": 0,
        "deformable_applied_schemas": applied_schemas,
        "attachment_count": len(attachments),
        "attachment_types": attachment_types,
        "particle_count": len(particle_points),
        "fluid": ledger.snapshot(),
        "debridement_release_qualified": False,
        "debridement_release_blocker": (
            "contact_work_requires_runtime_contact_force_and_tangential_"
            "velocity_measurements_not_inferred_from_joint_targets"
        ),
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
