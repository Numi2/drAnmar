#!/usr/bin/env python3
"""Headless CUDA qualification for the DrAnmar seal-and-divide system."""
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
    "adaptive_seal_divide_robot.py"
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
from pxr import Gf, Usd, UsdGeom
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaacsim.core.simulation_manager import PhysxGpuCfg, PhysxScene


def load_helper():
    spec = importlib.util.spec_from_file_location(
        "dranmar_seal_divide_runtime", HELPER_PATH
    )
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


def translated_local(matrix, x):
    result = Gf.Matrix4d(matrix)
    result.SetTranslateOnly(matrix.Transform(Gf.Vec3d(x, 0.0, 0.0)))
    return result


def assert_current_attachments(stage, paths):
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or prim.GetTypeName() != "OmniPhysicsVtxXformAttachment":
            raise RuntimeError(f"Invalid current-schema attachment: {path}")
        for relationship in ("omniphysics:src0", "omniphysics:src1"):
            if not prim.GetRelationship(relationship).GetTargets():
                raise RuntimeError(f"Attachment relationship missing: {path}")


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
    PhysxScene(sim.cfg.physics_prim_path).set_gpu_configuration(
        PhysxGpuCfg(gpu_max_deformable_surface_contacts=2**24)
    )
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/GroundPlane", ground)
    light = sim_utils.DomeLightCfg(intensity=2500.0)
    light.func("/World/Light", light)

    if args.representation == "standalone":
        root_path = "/World/AdaptiveSealDivideTool"
        robot = Articulation(helper.make_tool_cfg(root_path, position=(0.0, 0.0, 0.10)))
        tool_path = root_path
    else:
        root_path = "/World/Robot"
        robot = Articulation(
            helper.make_franka_adaptive_seal_divide_robot_cfg(prim_path=root_path)
        )
        tool_path = f"{root_path}/DrAnmarAdaptiveSealDivideTool"

    stage = omni.usd.get_context().get_stage()
    tcp_prim = stage.GetPrimAtPath(helper.frame_path(tool_path, "seal_divide_tcp"))
    if not tcp_prim.IsValid():
        raise RuntimeError("Seal-and-divide TCP is missing")
    tcp_world = UsdGeom.Xformable(tcp_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    tcp_translation = tuple(float(value) for value in tcp_world.ExtractTranslation())

    vessel_root = "/World/DrAnmarSealDivideVessel"
    helper.spawn_vessel_demo(vessel_root, translation=tcp_translation)
    deformables = helper.apply_vessel_surface_deformables(
        vessel_root, self_collision=False, stage=stage
    )
    fixture_attachments = helper.anchor_vessel_distal_ends(vessel_root, stage=stage)

    bridge = helper.BridgeAttachmentController(vessel_root)
    bridge_cells = bridge.engage(stage=stage)
    bridge_attachments = [
        path for cell in bridge_cells for path in cell.attachment_paths
    ]
    if len(bridge_attachments) != 16:
        raise RuntimeError(f"Expected 16 bridge attachments: {bridge_attachments}")

    compression = helper.DualZoneCompressionController(tool_path, vessel_root)
    compression_attachments = compression.engage(stage=stage)
    compression_status = compression.update_force(9.0, 9.0, stage=stage)
    if compression_status["mode"] != "controlled":
        raise RuntimeError(f"Unexpected compression mode: {compression_status}")

    sequence = helper.AdaptiveSealDivideSequenceController()
    sequence.transition("seal")
    for _ in range(5000):
        energy_report = sequence.update_energy(0.01, 9.0, 9.0)
        if (
            energy_report["both_ready"]
            and energy_report["flows"]["left_ml_min"] < 0.1
            and energy_report["flows"]["right_ml_min"] < 0.1
        ):
            break
    else:
        raise RuntimeError(f"Nominal seal did not qualify: {energy_report}")

    bands = helper.TissueSealBandController()
    left_bond = bands.deploy(
        "/World/LeftRetainedSealBand",
        translated_local(tcp_world, -0.0065),
        f"{vessel_root}/LeftVesselWall",
        stage=stage,
    )
    right_bond = bands.deploy(
        "/World/RightRetainedSealBand",
        translated_local(tcp_world, 0.0065),
        f"{vessel_root}/RightVesselWall",
        stage=stage,
    )
    bands.update_maturity(left_bond, sequence.energy.left.maturity, stage=stage)
    bands.update_maturity(right_bond, sequence.energy.right.maturity, stage=stage)
    if bands.apply_load(left_bond, bands.break_force_n(left_bond), stage=stage):
        raise RuntimeError("Left seal band failed at its threshold")
    if bands.apply_load(right_bond, bands.break_force_n(right_bond), stage=stage):
        raise RuntimeError("Right seal band failed at its threshold")
    seal_band_attachments = left_bond.attachment_paths + right_bond.attachment_paths

    sequence.transition("verify_seal")
    pre_guard = helper.BladeInterlockController().evaluate(
        sequence.energy, sequence.leak, 9.0, 9.0, False
    )
    if pre_guard["authorized"] or "blade_guard_not_retracted" not in pre_guard["reasons"]:
        raise RuntimeError(f"Blade interlock failed closed-state proof: {pre_guard}")
    sequence.transition("retract_guard")
    division = helper.TissueDivisionController(bridge)
    division_report = division.advance(
        1.0,
        energy=sequence.energy,
        leak=sequence.leak,
        upper_force_n=9.0,
        lower_force_n=9.0,
        guard_retracted=True,
        stage=stage,
    )
    if not division_report["authorized"] or not division_report["division_complete"]:
        raise RuntimeError(f"Interlocked division failed: {division_report}")
    if any(stage.GetPrimAtPath(path).IsValid() for path in bridge_attachments):
        raise RuntimeError("Bridge attachments remained after complete division")

    sequence.transition("release")
    compression.release(stage=stage)
    if any(stage.GetPrimAtPath(path).IsValid() for path in compression_attachments):
        raise RuntimeError("Temporary compression attachments were not released")
    active_attachments = fixture_attachments + seal_band_attachments
    if (
        len(fixture_attachments) != 2
        or len(compression_attachments) != 4
        or len(seal_band_attachments) != 4
        or len(active_attachments) != 6
    ):
        raise RuntimeError(
            "Unexpected attachment counts: "
            f"fixture={len(fixture_attachments)} "
            f"compression={len(compression_attachments)} "
            f"seal_bands={len(seal_band_attachments)} "
            f"active={len(active_attachments)}"
        )
    assert_current_attachments(stage, active_attachments)

    sim.reset()
    for _ in range(args.steps):
        sim.step(render=not args.headless)
        robot.update(sim.get_physics_dt())
    joint_pos = tensor_value(robot.data.joint_pos).detach().cpu().numpy()
    if not np.isfinite(joint_pos).all():
        bad = np.argwhere(~np.isfinite(joint_pos))
        raise RuntimeError(f"Non-finite articulation state: {bad.tolist()}")
    joint_names = list(robot.joint_names)
    expected_joint_count = 8 if args.representation == "standalone" else 15
    if len(joint_names) != expected_joint_count:
        raise RuntimeError(
            f"Expected {expected_joint_count} joints, got {len(joint_names)}: {joint_names}"
        )
    missing = sorted(set(helper.TOOL_JOINTS.values()) - set(joint_names))
    if missing:
        raise RuntimeError(f"Missing tool joints: {missing}")
    if args.representation == "franka":
        missing_arm = sorted(
            {f"panda_joint{i}" for i in range(1, 8)} - set(joint_names)
        )
        if missing_arm:
            raise RuntimeError(f"Missing Franka joints: {missing_arm}")

    required = {
        "OmniPhysicsDeformableBodyAPI",
        "OmniPhysicsSurfaceDeformableSimAPI",
        "PhysxSurfaceDeformableBodyAPI",
    }
    deformable_schemas = {}
    for mesh_path in deformables["mesh_paths"]:
        applied = list(stage.GetPrimAtPath(mesh_path).GetAppliedSchemas())
        if required - set(applied):
            raise RuntimeError(
                f"{mesh_path} missing schemas: {sorted(required-set(applied))}"
            )
        deformable_schemas[mesh_path.rsplit("/", 1)[-1]] = applied

    carb_logging.remove_logger(logger_handle)
    if engine_errors:
        raise RuntimeError("Isaac emitted engine errors:\n" + "\n".join(engine_errors))

    result = {
        "schema": "dr.anmar.adaptive-seal-divide-cuda-smoke.v1",
        "status": "pass",
        "representation": args.representation,
        "steps": args.steps,
        "device": args.device,
        "isaaclab_distribution_version": distribution_version("isaaclab"),
        "isaacsim_distribution_version": distribution_version("isaacsim"),
        "joint_names": joint_names,
        "joint_count": len(joint_names),
        "finite_joint_state": True,
        "engine_error_count": 0,
        "deformable_applied_schemas": deformable_schemas,
        "self_collision": deformables["self_collision"],
        "fixture_attachment_count": len(fixture_attachments),
        "bridge_attachment_count_created": len(bridge_attachments),
        "bridge_attachment_count_released": len(bridge_attachments),
        "compression_attachment_count_created": len(compression_attachments),
        "compression_released": True,
        "seal_band_attachment_count": len(seal_band_attachments),
        "active_attachment_count": len(active_attachments),
        "attachment_type": "OmniPhysicsVtxXformAttachment",
        "left_seal": {
            "temperature_c": sequence.energy.left.temperature_c,
            "energy_j": sequence.energy.left.energy_j,
            "maturity": sequence.energy.left.maturity,
            "impedance_ohm": sequence.energy.left.impedance_ohm,
        },
        "right_seal": {
            "temperature_c": sequence.energy.right.temperature_c,
            "energy_j": sequence.energy.right.energy_j,
            "maturity": sequence.energy.right.maturity,
            "impedance_ohm": sequence.energy.right.impedance_ohm,
        },
        "predicted_stump_flows_ml_min": sequence.leak.flows(),
        "blade_interlock": division_report,
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
