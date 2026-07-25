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
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from pxr import UsdGeom


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

    helper = load_helper()
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device)
    )
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/GroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    if args.representation == "standalone":
        root_path = "/World/WoundPreparationTool"
        robot = Articulation(
            helper.make_tool_cfg(
                root_path,
                irrigation_state="loaded",
                collection_state="empty",
                position=(0.0, 0.0, 0.45),
            )
        )
        tool_path = root_path
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

    wound_root = "/World/DrAnmarWoundBed"
    helper.spawn_wound_bed_demo(wound_root, translation=(0.55, 0.0, 0.02))
    stage = omni.usd.get_context().get_stage()
    deformable = helper.apply_wound_surface_deformable(wound_root, stage=stage)
    attachments = helper.attach_demo_debris(wound_root, stage=stage)
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

    result = {
        "schema": "dr.anmar.wound-preparation-cuda-smoke.v1",
        "status": "pass",
        "representation": args.representation,
        "steps": args.steps,
        "device": args.device,
        "isaaclab_distribution_version": distribution_version("isaaclab"),
        "isaacsim_distribution_version": distribution_version("isaacsim"),
        "tool_joint_names": joint_names,
        "finite_joint_state": True,
        "deformable_applied_schemas": applied_schemas,
        "attachment_count": len(attachments),
        "particle_count": len(particle_points),
        "fluid": ledger.snapshot(),
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
