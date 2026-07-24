#!/usr/bin/env python3
"""CUDA smoke test for both DrAnmar laparotomy-sponge representations.

Examples:
    python scripts/dr_anmar_laparotomy_sponge_smoke.py \
        --headless --device cuda:0 --representation rigid --state dry

    python scripts/dr_anmar_laparotomy_sponge_smoke.py \
        --headless --device cuda:0 --representation surface --state wet
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


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = (
    REPOSITORY_ROOT
    / "source/extensions/orbit.surgical.assets/orbit/surgical/assets/laparotomy_sponge.py"
)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--representation", choices=("rigid", "surface"), required=True)
parser.add_argument("--state", choices=("dry", "wet"), required=True)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import numpy as np
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObject, RigidObject


def load_helper():
    spec = importlib.util.spec_from_file_location(
        "dr_anmar_laparotomy_sponge_runtime",
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


def torch_value(value):
    """Return a torch tensor from Isaac Lab tensor or ProxyArray values."""

    return value.torch if hasattr(value, "torch") else value


def rigid_run(helper, sim, stage, root_path: str) -> dict[str, object]:
    sponge = RigidObject(
        helper.make_rigid_proxy_cfg(
            root_path,
            state=args.state,
            position=(0.0, 0.0, 0.35),
        )
    )
    helper.set_state_variant(stage, root_path, args.state)
    semantic_backends = helper.apply_default_labels(root_path)
    root_prim = stage.GetPrimAtPath(root_path)
    selected_mass_kg = float(root_prim.GetAttribute("physics:mass").Get())
    physics_material_targets = [
        str(target)
        for target in stage.GetPrimAtPath(
            f"{root_path}/Colliders"
        ).GetRelationship("material:binding:physics").GetTargets()
    ]
    expected_mass_kg = helper.SURFACE_PRESETS[args.state].mass_kg
    expected_material = (
        f"{root_path}/Looks/{'DryPhysics' if args.state == 'dry' else 'WetPhysics'}"
    )
    if abs(selected_mass_kg - expected_mass_kg) > 1.0e-7:
        raise RuntimeError(
            f"state={args.state} selected mass {selected_mass_kg}, "
            f"expected {expected_mass_kg}"
        )
    if expected_material not in physics_material_targets:
        raise RuntimeError(
            f"state={args.state} selected physics materials "
            f"{physics_material_targets}, expected {expected_material}"
        )

    sim.reset()
    for _ in range(args.steps):
        sim.step(render=not args.headless)
        sponge.update(sim.get_physics_dt())
    state = torch_value(sponge.data.root_state_w).detach().cpu().numpy()
    if not np.isfinite(state).all():
        raise RuntimeError("Non-finite rigid state after simulation")
    return {
        "selected_mass_kg": selected_mass_kg,
        "physics_material_targets": physics_material_targets,
        "root_state_w": state.tolist(),
        "finite_state": True,
        "semantic_backends": semantic_backends,
    }


def surface_run(helper, sim, stage, root_path: str) -> dict[str, object]:
    helper.spawn_unfolded_reference(
        root_path,
        state=args.state,
        translation=(0.0, 0.0, 0.50),
    )
    authored = helper.apply_surface_deformable(
        stage,
        root_path,
        state=args.state,
        self_collision=True,
    )
    semantic_backends = helper.apply_default_labels(root_path)
    mesh_path = f"{root_path}/SimulationMesh"
    mesh_prim = stage.GetPrimAtPath(mesh_path)
    applied_schemas = list(mesh_prim.GetAppliedSchemas())
    self_collision_enabled = bool(
        mesh_prim.GetAttribute("physxDeformableBody:selfCollision").Get()
    )
    required_schemas = {
        "OmniPhysicsDeformableBodyAPI",
        "OmniPhysicsSurfaceDeformableSimAPI",
        "PhysxSurfaceDeformableBodyAPI",
    }
    missing = sorted(required_schemas - set(applied_schemas))
    if missing:
        raise RuntimeError(
            f"Surface cooking omitted required schemas {missing}; "
            f"applied={applied_schemas}"
        )
    if not self_collision_enabled:
        raise RuntimeError("Surface self-collision was not enabled")

    deformable = DeformableObject(helper.make_surface_view_cfg(mesh_path))
    sim.reset()
    initial = torch_value(deformable.data.nodal_pos_w).detach().cpu().numpy().copy()
    for _ in range(args.steps):
        sim.step(render=not args.headless)
        deformable.update(sim.get_physics_dt())
    final = torch_value(deformable.data.nodal_pos_w).detach().cpu().numpy()
    if not np.isfinite(final).all():
        raise RuntimeError("Non-finite surface nodal positions after simulation")
    if int(final.shape[-2]) != 1027:
        raise RuntimeError(f"Surface view exposed {final.shape[-2]} nodes, expected 1027")
    return {
        "node_count": int(final.shape[-2]),
        "max_displacement_m": float(
            np.linalg.norm(final - initial, axis=-1).max()
        ),
        "finite_state": True,
        "applied_schemas": applied_schemas,
        "self_collision_enabled": self_collision_enabled,
        "authored": {
            key: str(value)
            for key, value in authored.items()
            if key != "preset"
        },
        "semantic_backends": semantic_backends,
    }


def main() -> int:
    helper = load_helper()
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device)
    )
    if args.representation == "rigid":
        sim.set_camera_view([0.65, 0.65, 0.45], [0.0, 0.0, 0.10])
    else:
        sim.set_camera_view([0.8, 0.8, 0.65], [0.0, 0.0, 0.20])
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/GroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(
        intensity=2500.0,
        color=(0.8, 0.8, 0.8),
    )
    light_cfg.func("/World/Light", light_cfg)

    root_path = "/World/LaparotomySponge"
    stage = omni.usd.get_context().get_stage()
    if args.representation == "rigid":
        result = rigid_run(helper, sim, stage, root_path)
    else:
        result = surface_run(helper, sim, stage, root_path)

    report = {
        "schema": "dr.anmar.laparotomy-sponge-cuda-smoke.v1",
        "status": "pass",
        "representation": args.representation,
        "state": args.state,
        "steps": args.steps,
        "device": args.device,
        "isaaclab_distribution_version": distribution_version("isaaclab"),
        "isaacsim_distribution_version": distribution_version("isaacsim"),
        "result": result,
        "clinical_validation": False,
    }
    payload = json.dumps(report, indent=2) + "\n"
    output = args.output
    if output is None:
        output = (
            REPOSITORY_ROOT
            / "run/qualification"
            / f"laparotomy-sponge-{args.representation}-{args.state}.json"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
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
