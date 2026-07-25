#!/usr/bin/env python3
"""Headless CUDA qualification for adaptive hemostasis."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / (
    "source/extensions/orbit.surgical.assets/orbit/surgical/assets/"
    "adaptive_hemostasis_robot.py"
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

import carb
import numpy as np
import omni.usd
from pxr import Gf, Usd, UsdGeom
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaacsim.core.simulation_manager import PhysxGpuCfg, PhysxScene
from dranmar_native_qualification import (
    attach_registered_camera_prims,
    capture_registered_camera_frames,
    command_phase_targets,
    spawn_fixed_standalone_articulation,
)


def load_helper():
    spec = importlib.util.spec_from_file_location("dranmar_hemostasis_runtime", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def tensor_value(value):
    return value.torch if hasattr(value, "torch") else value


def translation_matrix(position):
    matrix = Gf.Matrix4d(1.0)
    matrix.SetTranslate(Gf.Vec3d(*position))
    return matrix


def main() -> int:
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    engine_errors = []
    carb_logging = carb.logging.acquire_logging()

    def record_error(source, level, _filename, _line, message):
        if level >= carb.logging.LEVEL_ERROR and len(engine_errors) < 30:
            engine_errors.append(f"{source}: {message.strip()}")

    logger_handle = carb_logging.add_logger(record_error)
    helper = load_helper()
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device)
    )
    physx_scene = PhysxScene(sim.cfg.physics_prim_path)
    physx_scene.set_gpu_configuration(
        PhysxGpuCfg(gpu_max_deformable_surface_contacts=2**24)
    )
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/GroundPlane", ground)
    light = sim_utils.DomeLightCfg(intensity=2500.0)
    light.func("/World/Light", light)

    stage = omni.usd.get_context().get_stage()
    qualification_mount = None
    if args.representation == "standalone":
        root_path = "/World/AdaptiveHemostasisTool"
        robot, tool_path, qualification_mount = (
            spawn_fixed_standalone_articulation(
                stage,
                tool_cfg=helper.make_tool_cfg(
                    root_path, position=(0.0, 0.0, 0.10)
                ),
            )
        )
    else:
        root_path = "/World/Robot"
        robot = Articulation(
            helper.make_franka_adaptive_hemostasis_robot_cfg(prim_path=root_path)
        )
        tool_path = f"{root_path}/DrAnmarAdaptiveHemostasisTool"
    tcp_prim = stage.GetPrimAtPath(helper.frame_path(tool_path, "hemostasis_tcp"))
    if not tcp_prim.IsValid():
        raise RuntimeError("Hemostasis TCP is missing")
    tcp_world = UsdGeom.Xformable(tcp_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    tcp_translation = tcp_world.ExtractTranslation()
    vessel_translation = tuple(float(value) for value in tcp_translation)

    vessel_root = "/World/DrAnmarBleedingVessel"
    helper.spawn_vessel_demo(vessel_root, translation=vessel_translation)
    deformable = helper.apply_vessel_surface_deformable(vessel_root, stage=stage)
    fixture_attachments = helper.anchor_vessel_fixture(vessel_root, stage=stage)
    compression = helper.TemporaryCompressionController(
        tool_path, f"{vessel_root}/VesselWall"
    )
    compression_attachments = compression.engage(stage=stage)
    compression_status = compression.update_loads(1.8, 1.8, stage=stage)

    clip = helper.deploy_formed_clip(
        "/World/DeployedHemostaticClip",
        translation_matrix((vessel_translation[0], vessel_translation[1], vessel_translation[2] + 0.006)),
        f"{vessel_root}/VesselWall",
        stage=stage,
    )
    clip_retention = helper.ClipRetentionController()
    clip_bond = clip_retention.register(clip)
    if clip_retention.apply_load(clip_bond, 2.8, stage=stage):
        raise RuntimeError("Clip released at, rather than above, its threshold")

    patch_controller = helper.HemostaticPatchBondController()
    patch_bond = patch_controller.deploy(
        "/World/DeployedHemostaticPatch",
        translation_matrix((vessel_translation[0] + 0.004, vessel_translation[1], vessel_translation[2] + 0.004)),
        f"{vessel_root}/VesselWall",
        stage=stage,
    )
    patch_controller.update(30.0)
    if patch_bond.cure_fraction != 1.0:
        raise RuntimeError("Patch did not reach its provisional cured state")

    attachment_paths = (
        fixture_attachments + compression_attachments
        + clip["attachment_paths"] + patch_bond.attachment_paths
    )
    if len(attachment_paths) != 14:
        raise RuntimeError(f"Expected 14 attachments, got {len(attachment_paths)}")
    attachment_prims = [stage.GetPrimAtPath(path) for path in attachment_paths]
    for prim in attachment_prims:
        if not prim.IsValid() or prim.GetTypeName() != "OmniPhysicsVtxXformAttachment":
            raise RuntimeError(f"Invalid current-schema attachment: {prim.GetPath()}")
        for relationship in ("omniphysics:src0", "omniphysics:src1"):
            if not prim.GetRelationship(relationship).GetTargets():
                raise RuntimeError(f"Attachment relationship missing: {prim.GetPath()}")

    particle_paths = helper.ensure_blood_particle_system(stage=stage)
    ledger = helper.HemorrhageLedger(initial_reservoir_ml=1.0, reservoir_ml=1.0)
    positions = [
        (
            vessel_translation[0] + (0.003 + i * 0.0002 if i < 4 else 0.025),
            vessel_translation[1],
            vessel_translation[2] + 0.002,
        )
        for i in range(8)
    ]
    burst = helper.emit_blood_burst(
        positions, [(0.25, 0.0, 0.1)] * len(positions), ledger=ledger, stage=stage
    )
    if burst["particle_count"] != 8 or abs(ledger.conservation_error_ml) > 1e-9:
        raise RuntimeError(f"Blood ledger/particle mismatch: {burst} {ledger.snapshot()}")
    suction_prim = stage.GetPrimAtPath(
        helper.frame_path(tool_path, "suction_center")
    )
    suction_world = UsdGeom.Xformable(
        suction_prim
    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    suction_center = tuple(
        float(value) for value in suction_world.ExtractTranslation()
    )
    # The emitted PBD set itself, not a separate synthetic NumPy array, is
    # passed through annular suction and its removals debit the same ledger.
    suction = helper.AnnularSuctionController(suction_center)
    suction_evidence = suction.update_particle_set(
        1.0 / 120.0,
        ledger,
        stage=stage,
        particles_path=particle_paths["particles_path"],
    )
    if suction_evidence["captured_particle_count"] <= 0:
        raise RuntimeError(
            f"Annular suction did not capture emitted PBD blood: {suction_evidence}"
        )
    remaining_particles = list(
        UsdGeom.Points(
            stage.GetPrimAtPath(particle_paths["particles_path"])
        ).GetPointsAttr().Get()
        or []
    )
    if len(remaining_particles) != suction_evidence["active_particle_count"]:
        raise RuntimeError("PBD particle-set count diverged from suction evidence")

    # Retained physical attachments are evidence that the clip and patch
    # carriers exist.  No calibrated mapping from attachment count to defect
    # occlusion exists, so the reduced-order verifier must fail closed instead
    # of receiving nominal 0.999 effectiveness values.
    sequence = helper.AdaptiveHemostasisSequenceController()
    sequence.set_compression(0.0)
    sequence.set_clip_occlusion(0.0)
    sequence.set_patch_seal(0.0)
    sequence.transition("pressure_challenge")
    sequence.transition("verify")
    for _ in range(51):
        verification = sequence.update_verification(0.1)
    if not verification["complete"] or verification["passed"]:
        raise RuntimeError(
            f"Uncalibrated attachment effectiveness did not fail closed: {verification}"
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
    missing = sorted(set(helper.TOOL_JOINTS.values()) - set(joint_names))
    if missing:
        raise RuntimeError(f"Missing tool joints: {missing}")
    if args.representation == "franka":
        missing_arm = sorted(
            {f"panda_joint{i}" for i in range(1, 8)} - set(joint_names)
        )
        if missing_arm:
            raise RuntimeError(f"Missing Franka joints: {missing_arm}")
    canonical_phases = tuple(helper.PHASE_TARGETS)
    steps_per_phase = max(30, args.steps // len(canonical_phases))
    phase_convergence = {}
    for phase in canonical_phases:
        phase_convergence[phase] = command_phase_targets(
            robot,
            sim,
            joint_names=joint_names,
            targets=helper.phase_targets(phase),
            steps=steps_per_phase,
            render=not args.headless,
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
        bad = np.argwhere(~np.isfinite(joint_pos))
        raise RuntimeError(f"Non-finite articulation state at indices {bad.tolist()}")

    vessel_prim = stage.GetPrimAtPath(deformable["mesh_path"])
    applied_schemas = list(vessel_prim.GetAppliedSchemas())
    required = {
        "OmniPhysicsDeformableBodyAPI",
        "OmniPhysicsSurfaceDeformableSimAPI",
        "PhysxSurfaceDeformableBodyAPI",
    }
    if required - set(applied_schemas):
        raise RuntimeError(f"Missing deformable schemas: {sorted(required-set(applied_schemas))}")
    carb_logging.remove_logger(logger_handle)
    if engine_errors:
        raise RuntimeError("Isaac emitted engine errors:\n" + "\n".join(engine_errors))

    result = {
        "schema": "dr.anmar.adaptive-hemostasis-cuda-qualification.v2",
        "status": "fail_closed_as_designed",
        "qualification_scope": (
            "native_authored_mechanism_attachment_particle_suction_sensor_"
            "and_fail_closed_verification"
        ),
        "representation": args.representation,
        "steps": args.steps,
        "device": args.device,
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
        "attachment_count": len(attachment_paths),
        "attachment_type": "OmniPhysicsVtxXformAttachment",
        "blood_particle_count": burst["particle_count"],
        "blood_particle_count_after_suction": len(remaining_particles),
        "blood_suction": suction_evidence,
        "blood_ledger": ledger.snapshot(),
        "compression_mode": compression_status["mode"],
        "patch_cure_fraction": patch_bond.cure_fraction,
        "residual_flow_ml_min_unsealed_baseline": verification["flow_ml_min"],
        "average_flow_ml_min_unsealed_baseline": verification["average_flow_ml_min"],
        "intended_hemostasis_efficacy_qualified": False,
        "efficacy_blocker": (
            "clip_and_patch_attachment_retention_is_physical_but_occlusion_"
            "and_sealing_effectiveness_are_not_scene_measured_or_calibrated"
        ),
        "clinical_validation": False,
        "medical_device": False,
    }
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    try:
        code = main()
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
        raise SystemExit(code)
