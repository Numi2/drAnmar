#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn the canonical needle-ready TetMesh through Isaac Lab and Newton."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from isaaclab.app import launch_simulation


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--asset-extension",
    type=Path,
    default=(
        Path(__file__).resolve().parents[1]
        / "source/extensions/orbit.surgical.assets"
    ),
)
parser.add_argument(
    "--lod",
    choices=("training", "contact", "validation"),
    default="contact",
)
parser.add_argument("--steps", type=int, default=12)
parser.add_argument("--iterations", type=int, default=4)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument(
    "--viz",
    "--visualizer",
    dest="visualizer",
    choices=("kit", "newton", "none"),
    default=None,
)
args_cli = parser.parse_args()
args_cli.visualizer_explicit = any(
    token == "--viz"
    or token == "--visualizer"
    or token.startswith("--viz=")
    or token.startswith("--visualizer=")
    for token in sys.argv[1:]
)
if args_cli.visualizer == "none":
    args_cli.visualizer = None

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import DeformableObject  # noqa: E402
from isaaclab_contrib.deformable import VBDSolverCfg  # noqa: E402
from isaaclab_newton.physics import NewtonCfg  # noqa: E402


DT = 0.001
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def main() -> int:
    extension = args_cli.asset_extension.expanduser().resolve()
    if not extension.is_dir():
        raise FileNotFoundError(f"asset extension not found: {extension}")
    sys.path.insert(0, str(extension))
    from orbit.surgical.assets.needle_ready_tissue import (
        make_needle_ready_tissue_cfg,
        needle_ready_tissue_usd,
    )

    usd_path = needle_ready_tissue_usd(args_cli.lod).resolve()
    physics = NewtonCfg(
        solver_cfg=VBDSolverCfg(
            iterations=max(1, args_cli.iterations),
            particle_enable_self_contact=False,
            particle_collision_detection_interval=-1,
        ),
        num_substeps=2,
        use_cuda_graph=True,
    )
    sim_cfg = sim_utils.SimulationCfg(
        dt=DT,
        gravity=(0.0, 0.0, 0.0),
        device=args_cli.device,
        physics=physics,
        render_interval=20,
    )
    env_cfg = SimpleNamespace(sim=sim_cfg)
    step_ms: list[float] = []
    finite_samples = 0
    samples = 0
    node_count = 0
    default_hash = None
    final_hash = None
    with launch_simulation(env_cfg, args_cli):
        sim = sim_utils.SimulationContext(sim_cfg)
        tissue = DeformableObject(
            make_needle_ready_tissue_cfg(
                lod=args_cli.lod,
                prim_path="/World/NeedleReadyTissue",
                position=(0.0, 0.0, 0.02),
            )
        )
        sim.reset()
        tissue.reset()
        default = tissue.data.default_nodal_state_w.torch.clone()
        node_count = int(default.shape[1])
        default_hash = hashlib.sha256(
            np.asarray(default.cpu(), dtype="<f8").tobytes()
        ).hexdigest()
        for _ in range(max(2, args_cli.steps)):
            tissue.write_data_to_sim()
            torch.cuda.synchronize()
            started = time.perf_counter()
            sim.step(render=False)
            torch.cuda.synchronize()
            step_ms.append((time.perf_counter() - started) * 1000.0)
            tissue.update(DT)
            current = tissue.data.nodal_state_w.torch
            finite = bool(torch.isfinite(current).all())
            finite_samples += int(finite)
            samples += 1
            if not finite:
                break
        final = tissue.data.nodal_state_w.torch
        final_hash = hashlib.sha256(
            np.asarray(final.cpu(), dtype="<f8").tobytes()
        ).hexdigest()

    passed = samples == max(2, args_cli.steps) and finite_samples == samples
    result = {
        "schema": "dr.anmar.needle-ready-tissue-isaaclab-smoke.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "asset": {
            "path": portable_path(usd_path),
            "sha256": hashlib.sha256(usd_path.read_bytes()).hexdigest(),
            "lod": args_cli.lod,
            "nodal_count": node_count,
        },
        "runtime": {
            "isaac_sim": "6.0.1.0",
            "isaac_lab_revision": "51104d55d46192f9c981f2b63007d5156e141cec",
            "backend": "nvidia_newton_vbd_via_isaac_lab",
            "device": args_cli.device,
        },
        "solver": {
            "dt_s": DT,
            "substeps": 2,
            "iterations": max(1, args_cli.iterations),
            "cuda_graph": True,
        },
        "metrics": {
            "steps_completed": samples,
            "finite_state_fraction": finite_samples / max(samples, 1),
            "physics_step_ms_p50": float(np.percentile(step_ms, 50)),
            "physics_step_ms_p95": float(np.percentile(step_ms, 95)),
            "physics_step_ms_max": max(step_ms, default=0.0),
        },
        "default_state_sha256": default_hash,
        "final_state_sha256": final_hash,
        "spawn_gate_passed": passed,
        "scope": "spawn_reset_and_finite_step_only",
        "clinical_validation": False,
        "limitations": [
            "This proves Isaac Lab discovery, spawn, reset and stepping of the canonical TetMesh.",
            "The direct Newton qualification separately exercises fixture, retraction, release and rigid-soft contact.",
            "No puncture, topology, material calibration or clinical claim is established.",
        ],
    }
    output = args_cli.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
