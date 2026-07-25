#!/usr/bin/env python3
"""Headless CUDA qualification for the DrAnmar perfusion-viability system."""

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
HELPER_PATH = (
    ROOT
    / "source/extensions/orbit.surgical.assets/orbit/surgical/assets"
    / "perfusion_viability_robot.py"
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
from pxr import Gf, Usd, UsdGeom, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation


def load_helper():
    spec = importlib.util.spec_from_file_location(
        "dranmar_perfusion_viability_runtime", HELPER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {HELPER_PATH}")
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


def world_translation(stage, path: str):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        root = "/" + path.strip("/").split("/", 1)[0]
        found = [
            str(item.GetPath())
            for item in stage.Traverse()
            if str(item.GetPath()).startswith(root)
        ][:30]
        raise RuntimeError(f"Missing transform prim: {path}; composed prims={found}")
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    ).ExtractTranslation()


def main() -> int:
    if args.steps < 20:
        raise ValueError("--steps must be at least 20")
    engine_errors: list[str] = []
    carb_logging = carb.logging.acquire_logging()

    def record_error(source, level, _filename, _line, message):
        if level >= carb.logging.LEVEL_ERROR and len(engine_errors) < 30:
            engine_errors.append(f"{source}: {message.strip()}")

    logger_handle = carb_logging.add_logger(record_error)
    helper = load_helper()
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device)
    )
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/GroundPlane", ground)
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light.func("/World/Light", light)

    if args.representation == "standalone":
        root_path = "/World/PerfusionViabilityTool"
        robot = Articulation(
            helper.make_tool_cfg(root_path, position=(0.0, 0.0, 0.10))
        )
        tool_path = root_path
    else:
        root_path = "/World/Robot"
        robot = Articulation(
            helper.make_franka_perfusion_viability_robot_cfg(prim_path=root_path)
        )
        tool_path = f"{root_path}/DrAnmarPerfusionViabilityTool"

    stage = omni.usd.get_context().get_stage()
    tcp_path = helper.frame_path(tool_path, "perfusion_tcp")
    tcp_translation = world_translation(stage, tcp_path)
    tissue_translation = tuple(
        float(value)
        for value in (
            tcp_translation[0],
            tcp_translation[1],
            tcp_translation[2] - 0.008,
        )
    )
    tissue_root = "/World/DrAnmarPerfusedTissue"
    helper.spawn_tissue_demo(
        tissue_root,
        condition="anastomotic_stenosis",
        translation=tissue_translation,
    )
    camera_paths = helper.attach_camera_prims(stage, tool_path)
    if len(camera_paths) != 6:
        raise RuntimeError(f"Expected six registered camera prims: {camera_paths}")
    for path in camera_paths.values():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
            raise RuntimeError(f"Invalid registered camera prim: {path}")

    if args.representation == "franka":
        mount_joint = UsdPhysics.FixedJoint.Get(
            stage, f"{root_path}/dranmar_perfusion_mount_joint"
        )
        if not mount_joint:
            raise RuntimeError("Franka payload mount joint is missing")
        body0 = mount_joint.GetBody0Rel().GetTargets()
        body1 = mount_joint.GetBody1Rel().GetTargets()
        if (
            len(body0) != 1
            or body0[0].name != "panda_link8"
            or len(body1) != 1
            or str(body1[0]) != f"{tool_path}/Links/Mount"
        ):
            raise RuntimeError(
                f"Unexpected Franka mount relationship: body0={body0}, body1={body1}"
            )

    sim.reset()
    robot.update(sim.get_physics_dt())
    initial_tcp = world_translation(stage, tcp_path)
    tissue_surface = world_translation(stage, tissue_root) + Gf.Vec3d(0, 0, 0.008)
    registration_error_m = float((initial_tcp - tissue_surface).GetLength())
    if registration_error_m > 0.003:
        tissue_prim = stage.GetPrimAtPath(tissue_root)
        tissue_xform = UsdGeom.Xformable(tissue_prim)
        tissue_xform.ClearXformOpOrder()
        tissue_xform.AddTranslateOp().Set(
            Gf.Vec3d(initial_tcp[0], initial_tcp[1], initial_tcp[2] - 0.008)
        )
        tissue_surface = world_translation(stage, tissue_root) + Gf.Vec3d(0, 0, 0.008)
        registration_error_m = float((initial_tcp - tissue_surface).GetLength())
    if registration_error_m > 0.003:
        raise RuntimeError(
            f"Perfused tissue is not registered to the TCP: {registration_error_m} m"
        )

    joint_names = list(robot.joint_names)
    expected_joint_count = 12 if args.representation == "standalone" else 19
    if len(joint_names) != expected_joint_count:
        raise RuntimeError(
            f"Expected {expected_joint_count} joints, got {len(joint_names)}: "
            f"{joint_names}"
        )
    missing_tool = sorted(set(helper.TOOL_JOINTS.values()) - set(joint_names))
    if missing_tool:
        raise RuntimeError(f"Missing perfusion tool joints: {missing_tool}")
    if args.representation == "franka":
        missing_arm = sorted(
            {f"panda_joint{index}" for index in range(1, 8)} - set(joint_names)
        )
        if missing_arm:
            raise RuntimeError(f"Missing Franka arm joints: {missing_arm}")

    phases_checked = []
    steps_per_phase = max(1, args.steps // len(helper.TASK_PHASES))
    for phase in helper.TASK_PHASES:
        phase_contract = helper.phase_targets(phase)
        target = tensor_value(robot.data.default_joint_pos).clone()
        for name, value in phase_contract.items():
            target[:, joint_names.index(name)] = value
        robot.set_joint_position_target(target)
        for _ in range(steps_per_phase):
            robot.write_data_to_sim()
            sim.step(render=not args.headless)
            robot.update(sim.get_physics_dt())
        phases_checked.append(phase)

    joint_pos = tensor_value(robot.data.joint_pos).detach().cpu().numpy()
    if not np.isfinite(joint_pos).all():
        bad = np.argwhere(~np.isfinite(joint_pos))
        raise RuntimeError(f"Non-finite articulation state: {bad.tolist()}")

    flow_solver = helper.VascularFlowSolver()
    conservation = {}
    for condition in sorted(helper.VALID_CONDITIONS):
        flow = flow_solver.solve(condition)
        conservation[condition] = flow.conservation_error_ml_s
        if abs(flow.conservation_error_ml_s) > 1.0e-8:
            raise RuntimeError(
                f"Flow conservation failed for {condition}: "
                f"{flow.conservation_error_ml_s}"
            )

    closed_loop = {}
    verifier = helper.ClosedLoopPerfusionVerifier()
    for condition in (
        "arterial_occlusion",
        "venous_congestion",
        "anastomotic_stenosis",
        "branch_leak",
        "retraction_ischemia",
        "dressing_compression",
    ):
        scan = verifier.scan_intervene_rescan(
            condition, duration_s=18.0, dt_s=0.15
        )
        before = scan["before"].assessment
        after = scan["after"].assessment
        arrays = (
            scan["before"].maps.flow_index,
            scan["before"].maps.icg_intensity,
            scan["before"].maps.icg_extravascular,
            scan["before"].maps.speckle_perfusion,
            scan["before"].maps.temperature_c,
            scan["before"].maps.oxygenation_fraction,
            scan["before"].maps.doppler_speed_m_s,
            scan["before"].maps.ultrasound_patency,
            scan["before"].maps.confidence,
        )
        if any(array.shape != (4, 6) or not np.isfinite(array).all() for array in arrays):
            raise RuntimeError(f"Invalid multimodal map for {condition}")
        if scan["after_condition"] != "recovered" or scan["viability_gain"] <= 0.0:
            raise RuntimeError(
                f"Closed-loop recovery failed for {condition}: "
                f"gain={scan['viability_gain']}"
            )
        closed_loop[condition] = {
            "cause": before.likely_cause,
            "action": scan["action"],
            "before_viability": before.global_viability_score,
            "after_viability": after.global_viability_score,
            "viability_gain": scan["viability_gain"],
            "nonperfused_fraction_reduction": scan[
                "nonperfused_fraction_reduction"
            ],
        }

    carb_logging.remove_logger(logger_handle)
    if engine_errors:
        raise RuntimeError("Isaac emitted engine errors:\n" + "\n".join(engine_errors))

    result = {
        "schema": "dr.anmar.perfusion-viability-cuda-smoke.v1",
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
        "phases_checked": phases_checked,
        "registered_camera_count": len(camera_paths),
        "tissue_registration_error_m": registration_error_m,
        "flow_conservation_error_ml_s": conservation,
        "closed_loop": closed_loop,
        "research_only": True,
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
