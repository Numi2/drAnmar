#!/usr/bin/env python3
"""Headless CUDA smoke test for the atraumatic-exposure system.

Run through Isaac Lab:

    ./isaaclab.sh -p examples/validate_atraumatic_exposure_runtime.py \
        --headless --device cuda:0 --representation standalone --pad-type fenestrated
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
    / "atraumatic_exposure_robot.py"
)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--representation", choices=("standalone", "franka"), required=True)
parser.add_argument("--pad-type", choices=("fenestrated", "microcup"), required=True)
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


def load_helper():
    spec = importlib.util.spec_from_file_location(
        "dranmar_atraumatic_exposure_runtime",
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
        PhysxGpuCfg(gpu_max_deformable_surface_contacts=2**23)
    )
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/GroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    if args.representation == "standalone":
        root_path = "/World/AtraumaticExposureTool"
        robot = Articulation(
            helper.make_tool_cfg(
                root_path,
                pad_type=args.pad_type,
                position=(0.0, 0.0, 0.10),
            )
        )
        tool_path = root_path
        tissue_translation = (0.0, 0.0, 0.285)
    else:
        root_path = "/World/Robot"
        robot = Articulation(
            helper.make_franka_exposure_robot_cfg(
                prim_path=root_path,
                pad_type=args.pad_type,
            )
        )
        tool_path = f"{root_path}/DrAnmarAtraumaticExposureTool"
        tissue_translation = (0.54, 0.0, 0.0)

    tissue_root = "/World/DrAnmarExposureTissue"
    helper.spawn_exposure_tissue_demo(tissue_root, translation=tissue_translation)
    stage = omni.usd.get_context().get_stage()
    deformables = helper.apply_exposure_tissue_surface_deformables(
        tissue_root,
        stage=stage,
    )
    anchor_paths = helper.anchor_tissue_outer_bands(tissue_root, stage=stage)
    sequence = helper.ExposureSequenceController(
        tool_path=tool_path,
        left_tissue_path=f"{tissue_root}/LeftFlap",
        right_tissue_path=f"{tissue_root}/RightFlap",
        stage=stage,
    )
    sequence.set_phase("capture")
    capture_paths = [cell.attachment_path for cell in sequence.capture.active_cells()]
    attachment_paths = anchor_paths + capture_paths
    if len(anchor_paths) != 2 or len(capture_paths) != 12:
        raise RuntimeError(
            f"Expected 2 anchors and 12 capture cells, got "
            f"{len(anchor_paths)} and {len(capture_paths)}"
        )
    attachment_prims = [stage.GetPrimAtPath(path) for path in attachment_paths]
    if any(not prim.IsValid() for prim in attachment_prims):
        raise RuntimeError("Not all tissue attachments were authored")
    attachment_types = [prim.GetTypeName() for prim in attachment_prims]
    if any(not value for value in attachment_types):
        raise RuntimeError(f"Attachment type is missing: {attachment_types}")
    for prim in attachment_prims:
        if not prim.GetRelationship("omniphysics:src0").GetTargets():
            raise RuntimeError(f"Attachment has no deformable source: {prim.GetPath()}")
        if not prim.GetRelationship("omniphysics:src1").GetTargets():
            raise RuntimeError(f"Attachment has no rigid source: {prim.GetPath()}")

    control = sequence.hold_update(
        dt=1.0 / 120.0,
        visible_fraction=helper.ROIExposureEstimator.from_edge_gap(0.028),
        left_compression_m=-0.0008,
        right_compression_m=-0.0009,
    )
    if not all(np.isfinite(list(control["joint_targets"].values()))):
        raise RuntimeError("Force/visibility controller returned a non-finite target")

    sim.reset()
    for _ in range(args.steps):
        sim.step(render=not args.headless)
        robot.update(sim.get_physics_dt())

    joint_pos = tensor_value(robot.data.joint_pos).detach().cpu().numpy()
    if not np.isfinite(joint_pos).all():
        raise RuntimeError("Non-finite articulation joint state after simulation")
    joint_names = list(robot.joint_names)
    missing_joints = sorted(set(helper.TOOL_JOINTS.values()) - set(joint_names))
    if missing_joints:
        raise RuntimeError(f"Tool articulation is missing joints: {missing_joints}")
    if args.representation == "franka":
        missing_arm = sorted(
            {f"panda_joint{index}" for index in range(1, 8)} - set(joint_names)
        )
        if missing_arm:
            raise RuntimeError(f"Franka articulation is missing arm joints: {missing_arm}")

    deformable_schemas: dict[str, list[str]] = {}
    required_deformable_schemas = {
        "OmniPhysicsDeformableBodyAPI",
        "OmniPhysicsSurfaceDeformableSimAPI",
        "PhysxSurfaceDeformableBodyAPI",
    }
    for side, metadata in deformables["flaps"].items():
        mesh_prim = stage.GetPrimAtPath(metadata["mesh_path"])
        applied = list(mesh_prim.GetAppliedSchemas())
        missing = sorted(required_deformable_schemas - set(applied))
        if missing:
            raise RuntimeError(f"{side} surface deformable omitted schemas: {missing}")
        deformable_schemas[side] = applied

    carb_logging.remove_logger(logger_handle)
    if engine_errors:
        raise RuntimeError("Isaac runtime emitted engine errors:\n" + "\n".join(engine_errors))

    result = {
        "schema": "dr.anmar.atraumatic-exposure-cuda-smoke.v1",
        "status": "pass",
        "representation": args.representation,
        "pad_type": args.pad_type,
        "steps": args.steps,
        "device": args.device,
        "gpu_max_deformable_surface_contacts": (
            physx_scene.get_gpu_configuration().gpu_max_deformable_surface_contacts
        ),
        "isaaclab_distribution_version": distribution_version("isaaclab"),
        "isaacsim_distribution_version": distribution_version("isaacsim"),
        "tool_joint_names": joint_names,
        "finite_joint_state": True,
        "engine_error_count": 0,
        "deformable_applied_schemas": deformable_schemas,
        "anchor_attachment_count": len(anchor_paths),
        "capture_attachment_count": len(capture_paths),
        "attachment_types": sorted(set(attachment_types)),
        "controller_mode": control["mode"],
        "controller_visible_fraction": control["visible_fraction"],
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
