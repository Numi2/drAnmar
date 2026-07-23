#!/usr/bin/env python3
"""Headless native-PhysX smoke probe for the independent Dr.Anmar suture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics  # noqa: E402


def main() -> int:
    if not args.asset.is_file():
        raise FileNotFoundError(args.asset)
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

    root = stage.DefinePrim("/World/IndependentDrAnmarSuture", "Xform")
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
    segments = physics_view.create_rigid_body_view(
        "/World/IndependentDrAnmarSuture/Segments/S*"
    )
    if segments._backend is None or segments.count != 360:
        raise RuntimeError(
            f"PhysX created {segments.count if segments._backend else 0} of 360 suture bodies"
        )
    initial = segments.get_transforms().cpu().numpy().astype(np.float64)
    for _ in range(max(1, args.steps)):
        sim.step(render=False)
    final = segments.get_transforms().cpu().numpy().astype(np.float64)
    finite = bool(np.isfinite(final).all())
    free_end_drop = float(initial[-1, 2] - final[-1, 2])
    displacement = np.linalg.norm(final[:, :3] - initial[:, :3], axis=1)
    joint_count = sum(
        prim.GetTypeName() == "PhysicsJoint"
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith("/World/IndependentDrAnmarSuture/Joints/")
    )
    report = {
        "schema": "dr.anmar.suture-native-physx-probe.v1",
        "asset": str(args.asset.resolve()),
        "physics_dt_s": 0.0005,
        "steps": int(args.steps),
        "segment_count": int(segments.count),
        "joint_count": int(joint_count),
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
