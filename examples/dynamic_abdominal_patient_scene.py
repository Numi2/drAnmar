#!/usr/bin/env python3
"""Interactive Dr.Anmar patient with a retractable midline laparotomy.

This is the real patient hierarchy and real Dr.Anmar atraumatic-exposure tool,
not the isolated tissue-demo asset. The open patient loads five bilateral
explicit-TetMesh wound layers. Six distributed capture cells on each exposure
pad lift and retract the full-thickness wound margins while the central
operative field stays clear for the camera.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument(
    "--release_after_s",
    type=float,
    default=0.0,
    help="Release both wound margins after this simulated time; 0 holds open.",
)
parser.add_argument(
    "--steps",
    type=int,
    default=0,
    help="Exit after this many simulation steps; 0 runs interactively.",
)
parser.add_argument(
    "--capture_path",
    type=Path,
    help="Optional PNG path for the actual retracted patient scene.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(ROOT / "source/extensions/orbit.surgical.assets"),
)


import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sensors.camera import Camera, CameraCfg  # noqa: E402
from PIL import Image  # noqa: E402
from orbit.surgical.assets.atraumatic_exposure_robot import (  # noqa: E402
    make_tool_cfg,
    phase_targets,
)
from orbit.surgical.assets.dynamic_abdominal_patient import (  # noqa: E402
    DynamicSurgicalPatient,
    apply_laparotomy_wound_deformables,
    capture_laparotomy_wound_edges,
    configure_patient_internal_collision_filter,
    release_laparotomy_wound_edges,
    spawn_patient,
)


PATIENT_PATH = "/World/Patient"
TOOL_PATH = "/World/DrAnmarAtraumaticExposureTool"


def _target_tensor(
    tool: Articulation,
    targets: dict[str, float],
) -> torch.Tensor:
    result = tool.data.default_joint_pos.clone()
    joint_names = list(tool.joint_names)
    for name, value in targets.items():
        result[:, joint_names.index(name)] = float(value)
    return result


def _rgb_image(value) -> Image.Image:
    if hasattr(value, "torch"):
        value = value.torch
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and float(
            array.max(initial=0.0)
        ) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def main() -> None:
    if not math.isfinite(args.release_after_s) or args.release_after_s < 0.0:
        raise ValueError("--release_after_s must be finite and non-negative")
    if args.steps < 0:
        raise ValueError("--steps must be non-negative")
    if args.capture_path is not None and not args.enable_cameras:
        raise ValueError(
            "--capture_path requires AppLauncher --enable_cameras"
        )

    log_dir = ROOT / "run/laparotomy"
    log_dir.mkdir(parents=True, exist_ok=True)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=1.0 / 120.0,
            device=args.device,
            log_dir=str(log_dir),
        )
    )
    light_cfg = sim_utils.DomeLightCfg(
        intensity=2_200.0,
        color=(0.86, 0.88, 0.92),
    )
    light_cfg.func("/World/Light", light_cfg)

    spawn_patient(PATIENT_PATH, access_state="open")
    wound_routes = apply_laparotomy_wound_deformables(PATIENT_PATH)
    if any(
        edge["route"] != "current_explicit_tetmesh_volume_hierarchy"
        for layer in wound_routes.values()
        for edge in layer.values()
    ):
        raise RuntimeError(
            f"Patient wound mechanics did not initialize: {wound_routes}"
        )
    print(
        "[laparotomy] activated 10 full-thickness TetMesh wound edges",
        flush=True,
    )
    configure_patient_internal_collision_filter(PATIENT_PATH)

    # Rotate the tool 180 degrees about X so the pads approach from above.
    # This keeps the arms out of the patient volume and leaves the midline
    # operative corridor visible from the overhead camera.
    tool = Articulation(
        make_tool_cfg(
            prim_path=TOOL_PATH,
            pad_type="fenestrated",
            position=(0.0, 0.0, 0.289),
            orientation_wxyz=(0.0, 1.0, 0.0, 0.0),
        )
    )
    capture_paths = capture_laparotomy_wound_edges(
        PATIENT_PATH,
        TOOL_PATH,
    )
    if len(capture_paths) != 60:
        raise RuntimeError(
            f"Expected 60 distributed wound-edge bonds, got "
            f"{len(capture_paths)}"
        )
    print(
        "[laparotomy] connected 60 distributed capture bonds to the "
        "Dr.Anmar exposure pads",
        flush=True,
    )

    camera = None
    if args.capture_path is not None:
        camera = Camera(
            CameraCfg(
                prim_path="/World/LaparotomyCamera",
                update_period=0.0,
                height=900,
                width=900,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=32.0,
                    focus_distance=0.55,
                    horizontal_aperture=24.0,
                    clipping_range=(0.01, 3.0),
                ),
            )
        )

    sim.set_camera_view(
        eye=(0.0, -0.62, 0.43),
        target=(0.0, 0.0, 0.055),
    )
    sim.reset()
    tool.reset()
    if camera is not None:
        camera.set_world_poses_from_view(
            torch.tensor(
                [[0.0, -0.62, 0.43]],
                dtype=torch.float32,
                device=args.device,
            ),
            torch.tensor(
                [[0.0, 0.0, 0.055]],
                dtype=torch.float32,
                device=args.device,
            ),
        )
    patient = DynamicSurgicalPatient()
    patient.set_procedure_stage("exposed")

    capture = _target_tensor(tool, phase_targets("capture"))
    retract = _target_tensor(tool, phase_targets("retract"))
    hold = _target_tensor(tool, phase_targets("hold"))
    elapsed = 0.0
    released = False
    step_count = 0

    while app.is_running():
        if args.steps and step_count >= args.steps:
            break
        dt = sim.get_physics_dt()
        elapsed += dt
        step_count += 1
        if elapsed < 1.0:
            targets = capture
        elif elapsed < 5.0:
            fraction = (elapsed - 1.0) / 4.0
            targets = capture + (retract - capture) * fraction
        else:
            targets = hold

        if (
            args.release_after_s > 0.0
            and elapsed >= args.release_after_s
            and not released
        ):
            release_laparotomy_wound_edges(PATIENT_PATH)
            targets = _target_tensor(tool, phase_targets("release"))
            released = True

        tool.set_joint_position_target(targets)
        tool.write_data_to_sim()
        patient.step(dt)
        sim.step(render=camera is not None)
        tool.update(dt)
        if camera is not None:
            camera.update(dt, force_recompute=True)
    print(
        f"[laparotomy] completed {step_count} actual patient-scene steps",
        flush=True,
    )
    if camera is not None:
        args.capture_path.parent.mkdir(parents=True, exist_ok=True)
        _rgb_image(camera.data.output["rgb"]).save(args.capture_path)
        print(
            f"[laparotomy] wrote actual scene frame to "
            f"{args.capture_path}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        try:
            app.close(skip_cleanup=True)
        except TypeError:
            app.close()
