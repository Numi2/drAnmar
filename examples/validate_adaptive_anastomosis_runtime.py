#!/usr/bin/env python3
"""Headless CUDA qualification for the DrAnmar adaptive anastomosis system."""
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
    "adaptive_anastomosis_robot.py"
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
from pxr import Usd, UsdGeom
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaacsim.core.simulation_manager import PhysxGpuCfg, PhysxScene


def load_helper():
    spec = importlib.util.spec_from_file_location(
        "dranmar_anastomosis_runtime", HELPER_PATH
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

    if args.representation == "standalone":
        root_path = "/World/AdaptiveAnastomosisTool"
        robot = Articulation(
            helper.make_tool_cfg(root_path, position=(0.0, 0.0, 0.10))
        )
        tool_path = root_path
    else:
        root_path = "/World/Robot"
        robot = Articulation(
            helper.make_franka_adaptive_anastomosis_robot_cfg(prim_path=root_path)
        )
        tool_path = f"{root_path}/DrAnmarAdaptiveAnastomosisTool"

    stage = omni.usd.get_context().get_stage()
    tcp_prim = stage.GetPrimAtPath(helper.frame_path(tool_path, "anastomosis_tcp"))
    if not tcp_prim.IsValid():
        raise RuntimeError("Anastomosis TCP is missing")
    tcp_world = UsdGeom.Xformable(tcp_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    tcp_translation = tuple(float(value) for value in tcp_world.ExtractTranslation())

    tissue_root = "/World/DrAnmarHollowTissue"
    helper.spawn_hollow_tissue_demo(
        tissue_root, state="initial", translation=tcp_translation
    )
    deformables = helper.apply_hollow_tissue_surface_deformables(
        tissue_root, self_collision=False, stage=stage
    )
    fixture_attachments = helper.anchor_hollow_tissue_distal_ends(
        tissue_root, stage=stage
    )

    capture = helper.BilateralTissueCaptureController(
        tool_path,
        deformables["LeftTissue"],
        deformables["RightTissue"],
    )
    capture_paths = capture.engage(stage=stage)
    capture_attachments = capture_paths["left"] + capture_paths["right"]
    capture_status = capture.update_loads(1.6, 1.6, stage=stage)
    if capture_status["mode"] != "controlled":
        raise RuntimeError(f"Unexpected capture mode: {capture_status}")

    deployments = helper.deploy_staple_ring(
        "/World/DeployedStapleRing",
        tcp_world,
        deformables["LeftTissue"],
        deformables["RightTissue"],
        stage=stage,
    )
    retention = helper.StapleRingRetentionController()
    retention.register(deployments)
    if retention.apply_loads([1.4] * len(deployments), stage=stage):
        raise RuntimeError("A staple released at, rather than above, its threshold")
    if retention.retained_fraction != 1.0:
        raise RuntimeError("Staple ring was not fully retained")
    staple_attachments = [
        path for deployment in deployments for path in deployment["attachment_paths"]
    ]

    capture.release(stage=stage)
    if any(stage.GetPrimAtPath(path).IsValid() for path in capture_attachments):
        raise RuntimeError("Temporary capture attachments were not released")

    collar_controller = helper.ReinforcementCollarBondController()
    collar_bond = collar_controller.deploy(
        "/World/DeployedReinforcementCollar",
        tcp_world,
        deformables["LeftTissue"],
        deformables["RightTissue"],
        stage=stage,
    )
    collar_controller.update(45.0)
    if collar_bond.cure_fraction != 1.0:
        raise RuntimeError("Reinforcement collar did not reach cured state")
    if collar_controller.bonded_fraction(collar_bond) != 1.0:
        raise RuntimeError("Reinforcement collar lost a sector unexpectedly")

    active_attachment_paths = (
        fixture_attachments + staple_attachments + collar_bond.attachment_paths
    )
    if len(capture_attachments) != 12 or len(active_attachment_paths) != 66:
        raise RuntimeError(
            "Unexpected attachment counts: "
            f"capture={len(capture_attachments)} active={len(active_attachment_paths)}"
        )
    for path in active_attachment_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or prim.GetTypeName() != "OmniPhysicsVtxXformAttachment":
            raise RuntimeError(f"Invalid current-schema attachment: {path}")
        for relationship in ("omniphysics:src0", "omniphysics:src1"):
            if not prim.GetRelationship(relationship).GetTargets():
                raise RuntimeError(f"Attachment relationship missing: {path}")

    def usd_world_points(mesh_path):
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        mesh = UsdGeom.Mesh(mesh_prim)
        transform = UsdGeom.Xformable(
            mesh_prim
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return [
            tuple(transform.Transform(point))
            for point in (mesh.GetPointsAttr().Get() or [])
        ]

    seam_geometry = helper.measure_lumen_seam_geometry(
        usd_world_points(deformables["LeftTissue"]),
        usd_world_points(deformables["RightTissue"]),
    )
    patency = helper.LumenPatencyController().evaluate(
        seam_geometry["radial_samples_m"],
        centerline_offset_m=seam_geometry["centerline_offset_m"],
        axis_error_deg=seam_geometry["axis_error_deg"],
    )

    helper.ensure_leak_particle_system(stage=stage)
    ledger = helper.LeakTestLedger(initial_reservoir_ml=1.0)
    ledger.inject(0.5)
    particle_positions = [
        (
            tcp_translation[0],
            tcp_translation[1] + 0.003 + i * 0.0002,
            tcp_translation[2] - 0.002,
        )
        for i in range(8)
    ]
    burst = helper.emit_leak_particles(
        particle_positions,
        [(0.0, -0.08, -0.03)] * len(particle_positions),
        ledger=ledger,
        stage=stage,
    )
    if burst["particle_count"] != 8:
        raise RuntimeError(f"Leak particle emission mismatch: {burst}")
    if abs(ledger.conservation_error_ml) > 1e-9:
        raise RuntimeError(f"Leak ledger drifted after emission: {ledger.snapshot()}")

    sequence = helper.AdaptiveAnastomosisSequenceController()
    sequence.transition("pressurize")
    for _ in range(20):
        sequence.leak_test.update(
            0.1,
            pump_flow_ml_s=0.2,
            edge_gap_m=seam_geometry["edge_gap_m"],
            retained_staple_fraction=retention.retained_fraction,
            collar_bond_fraction=collar_controller.bonded_fraction(collar_bond),
        )
        if sequence.leak_test.pressure_pa >= sequence.leak_test.target_pressure_pa:
            break
    sequence.transition("verify")
    for _ in range(81):
        verification = sequence.leak_test.update(
            0.1,
            edge_gap_m=seam_geometry["edge_gap_m"],
            retained_staple_fraction=retention.retained_fraction,
            collar_bond_fraction=collar_controller.bonded_fraction(collar_bond),
        )
    if not sequence.leak_test.complete or sequence.leak_test.passed:
        raise RuntimeError(
            "Open scene-derived seam did not fail closed: "
            f"last={verification} avg={sequence.leak_test.average_leak_ml_min}"
        )
    ledger.leak(sequence.leak_test.integrated_leak_ml)
    ledger.collect(ledger.active_leak_ml)
    if abs(ledger.conservation_error_ml) > 1e-9:
        raise RuntimeError(f"Leak ledger drifted after verification: {ledger.snapshot()}")

    sim.reset()
    for _ in range(args.steps):
        sim.step(render=not args.headless)
        robot.update(sim.get_physics_dt())
    joint_pos = tensor_value(robot.data.joint_pos).detach().cpu().numpy()
    if not np.isfinite(joint_pos).all():
        bad = np.argwhere(~np.isfinite(joint_pos))
        raise RuntimeError(f"Non-finite articulation state at indices {bad.tolist()}")
    joint_names = list(robot.joint_names)
    expected_joint_count = 14 if args.representation == "standalone" else 21
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
    for side, mesh_path in deformables.items():
        applied = list(stage.GetPrimAtPath(mesh_path).GetAppliedSchemas())
        if required - set(applied):
            raise RuntimeError(
                f"{side} missing deformable schemas: {sorted(required-set(applied))}"
            )
        deformable_schemas[side] = applied

    carb_logging.remove_logger(logger_handle)
    if engine_errors:
        raise RuntimeError("Isaac emitted engine errors:\n" + "\n".join(engine_errors))

    result = {
        "schema": "dr.anmar.adaptive-anastomosis-runtime-diagnostic.v2",
        "status": "controller_exercise_only",
        "qualification_scope": (
            "composition_attachments_ledgers_and_fail_closed_open_seam_model"
        ),
        "representation": args.representation,
        "steps": args.steps,
        "device": args.device,
        "isaaclab_distribution_version": distribution_version("isaaclab"),
        "isaacsim_distribution_version": distribution_version("isaacsim"),
        "tool_joint_names": joint_names,
        "joint_count": len(joint_names),
        "finite_joint_state": True,
        "engine_error_count": 0,
        "deformable_applied_schemas": deformable_schemas,
        "fixture_attachment_count": len(fixture_attachments),
        "capture_attachment_count_created": len(capture_attachments),
        "capture_released": True,
        "staple_count": len(deployments),
        "staple_attachment_count": len(staple_attachments),
        "retained_staple_fraction": retention.retained_fraction,
        "collar_attachment_count": len(collar_bond.attachment_paths),
        "collar_cure_fraction": collar_bond.cure_fraction,
        "active_attachment_count": len(active_attachment_paths),
        "attachment_type": "OmniPhysicsVtxXformAttachment",
        "leak_particle_count": burst["particle_count"],
        "leak_ledger": ledger.snapshot(),
        "patency": {
            "minimum_radius_m": patency.minimum_radius_m,
            "mean_radius_m": patency.mean_radius_m,
            "area_fraction": patency.area_fraction,
            "centerline_offset_m": patency.centerline_offset_m,
            "axis_error_deg": patency.axis_error_deg,
            "passed": patency.passed,
            "source": "authored_usd_nodes_before_physics",
        },
        "seam_geometry": seam_geometry,
        "pressure_decay": {
            "target_pressure_pa": sequence.leak_test.target_pressure_pa,
            "final_pressure_pa": sequence.leak_test.pressure_pa,
            "observation_s": sequence.leak_test.elapsed_s,
            "average_leak_ml_min": sequence.leak_test.average_leak_ml_min,
            "peak_leak_ml_min": sequence.leak_test.peak_leak_ml_min,
            "integrated_leak_ml": sequence.leak_test.integrated_leak_ml,
            "passed": sequence.leak_test.passed,
        },
        "intended_anastomosis_efficacy_qualified": False,
        "efficacy_blocker": (
            "runtime_seam_apposition_patency_and_calibrated_pressure_flow_"
            "measurements_are_not_available"
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
