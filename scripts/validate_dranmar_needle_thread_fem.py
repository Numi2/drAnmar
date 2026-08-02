#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Qualify the compound Dr.Anmar needle-thread asset on the pinned PhysX runtime.

This is intentionally an asset-only gate.  It does not instantiate or retune
the penetration task: the existing task must continue to see one rigid needle,
while PhysX independently advances one native surface-deformable strand.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET = (
    REPOSITORY_ROOT
    / "source/extensions/orbit.surgical.assets/data/Props/SurgicalClosure/Needle"
    / "dranmar_needle_thread_fem.usda"
)

parser = argparse.ArgumentParser()
parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--dt", type=float, default=0.002)
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import (  # noqa: E402
    DeformableObject,
    DeformableObjectCfg,
    RigidObject,
    RigidObjectCfg,
)
from isaaclab_physx.physics import PhysxCfg  # noqa: E402


TASK_SCALE = 1.5
SWAGE_ANCHOR_M = (0.0, 0.00700281749604, 0.0)
SWAGE_ATTACHED_VERTICES = 6
EXPECTED_POINTS = 722
EXPECTED_TRIANGLES = 720


def _schema_count(stage: Usd.Stage, schema_name: str) -> int:
    return sum(schema_name in prim.GetAppliedSchemas() for prim in stage.Traverse())


def _type_count(stage: Usd.Stage, type_name: str) -> int:
    return sum(prim.GetTypeName() == type_name for prim in stage.Traverse())


def _triangle_count(mesh: UsdGeom.Mesh) -> int:
    counts = mesh.GetFaceVertexCountsAttr().Get() or []
    if any(count != 3 for count in counts):
        raise RuntimeError("thread collision mesh contains non-triangular faces")
    return len(counts)


def _world_points(stage: Usd.Stage, mesh: UsdGeom.Mesh) -> torch.Tensor:
    matrix = UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    points = [matrix.Transform(Gf.Vec3d(point)) for point in mesh.GetPointsAttr().Get()]
    return torch.tensor(points, dtype=torch.float32, device=args.device).unsqueeze(0)


def validate() -> dict:
    asset_path = args.asset.resolve()
    if not asset_path.is_file():
        raise FileNotFoundError(asset_path)

    sim_cfg = sim_utils.SimulationCfg(
        dt=args.dt,
        device=args.device,
        gravity=(0.0, 0.0, -9.81),
        physics=PhysxCfg(),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim_utils.DomeLightCfg(intensity=1800.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=1800.0))

    needle = RigidObject(
        RigidObjectCfg(
            prim_path="/World/Needle",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(asset_path),
                scale=(TASK_SCALE, TASK_SCALE, TASK_SCALE),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25)),
        )
    )
    thread = DeformableObject(
        DeformableObjectCfg(
            prim_path="/World/Needle/ThreadFEM",
            spawn=None,
        )
    )

    stage = sim.stage
    root_prim = stage.GetPrimAtPath("/World/Needle")
    rigid_prim = stage.GetPrimAtPath("/World/Needle/NeedleRigid")
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Needle/ThreadFEM"))
    if not root_prim or not rigid_prim or not mesh:
        raise RuntimeError("compound needle-thread prims did not compose")

    rigid_count = _schema_count(stage, "PhysicsRigidBodyAPI")
    deformable_count = _schema_count(stage, "OmniPhysicsDeformableBodyAPI")
    joint_count = sum(
        _type_count(stage, type_name)
        for type_name in ("PhysicsJoint", "PhysicsFixedJoint", "PhysicsRevoluteJoint", "PhysicsPrismaticJoint")
    )
    point_count = len(mesh.GetPointsAttr().Get() or [])
    triangle_count = _triangle_count(mesh)
    expected_world = _world_points(stage, mesh)

    # Hold the rigid needle fixed while gravity tests the strand and swage.
    UsdPhysics.RigidBodyAPI(rigid_prim).CreateKinematicEnabledAttr(True)
    sim.reset()
    if not needle.is_initialized or not thread.is_initialized:
        raise RuntimeError("PhysX tensor views did not initialize")

    thread.update(args.dt)
    initial = thread.data.nodal_state_w.torch.clone()
    peak_speed = 0.0
    speed_samples: list[float] = []
    finite_steps = 0
    for _ in range(args.steps):
        sim.step(render=False)
        thread.update(args.dt)
        state = thread.data.nodal_state_w.torch
        if torch.isfinite(state).all().item():
            finite_steps += 1
        step_speed = float(torch.linalg.vector_norm(state[..., 3:], dim=-1).max().item())
        speed_samples.append(step_speed)
        peak_speed = max(peak_speed, step_speed)

    final = thread.data.nodal_state_w.torch.clone()
    simulated_nodes = final.shape[1]
    expected_world = expected_world[:, :simulated_nodes]
    attachment_error = torch.linalg.vector_norm(
        final[:, :SWAGE_ATTACHED_VERTICES, :3]
        - expected_world[:, :SWAGE_ATTACHED_VERTICES],
        dim=-1,
    )
    free_displacement = torch.linalg.vector_norm(
        final[:, SWAGE_ATTACHED_VERTICES:, :3]
        - initial[:, SWAGE_ATTACHED_VERTICES:, :3],
        dim=-1,
    )

    receipt = {
        "schema": "dr.anmar.needle-thread-asset-qualification.v1",
        "asset": str(asset_path),
        "scope": "simulator_engineering_not_biomechanical_or_clinical_validation",
        "device": args.device,
        "dt_s": args.dt,
        "steps": args.steps,
        "topology": {
            "rigid_bodies": rigid_count,
            "surface_deformables": deformable_count,
            "joints": joint_count,
            "points": point_count,
            "triangles": triangle_count,
            "physx_simulation_nodes": simulated_nodes,
        },
        "metrics": {
            "finite_step_fraction": finite_steps / max(args.steps, 1),
            "swage_attachment_error_m_max": float(attachment_error.max().item()),
            "free_strand_displacement_m_max": float(free_displacement.max().item()),
            "nodal_speed_m_s_max": peak_speed,
            "nodal_speed_m_s_tail_max": max(speed_samples[3 * len(speed_samples) // 4 :], default=0.0),
            "nodal_speed_m_s_final": speed_samples[-1] if speed_samples else 0.0,
        },
    }
    failures = []
    if rigid_count != 1:
        failures.append(f"expected one rigid body, found {rigid_count}")
    if deformable_count != 1:
        failures.append(f"expected one deformable body, found {deformable_count}")
    if joint_count != 0:
        failures.append(f"legacy joint-chain schemas remain: {joint_count}")
    if point_count != EXPECTED_POINTS or triangle_count != EXPECTED_TRIANGLES:
        failures.append(
            f"unexpected strand topology: {point_count} points/{triangle_count} triangles"
        )
    if simulated_nodes != EXPECTED_POINTS:
        failures.append(f"PhysX exposes {simulated_nodes} nodes, expected {EXPECTED_POINTS}")
    if finite_steps != args.steps:
        failures.append("non-finite nodal state")
    if float(attachment_error.max().item()) > 0.00025:
        failures.append("swage attachment drift exceeds one strand diameter")
    if float(free_displacement.max().item()) < 0.00005:
        failures.append("free strand did not respond to gravity")
    if not math.isfinite(peak_speed) or peak_speed > 2.0:
        failures.append("nodal velocity indicates an unstable strand")
    if max(speed_samples[3 * len(speed_samples) // 4 :], default=0.0) > 1.0:
        failures.append("strand did not settle into a bounded velocity regime")

    receipt["passed"] = not failures
    receipt["failures"] = failures
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if failures:
        raise RuntimeError("; ".join(failures))
    return receipt


try:
    validate()
finally:
    simulation_app.close()
