#!/usr/bin/env python3
"""Render a reproducible Isaac Lab GIF for one Dr.Anmar surgical robot.

Run with the Isaac Lab Python launcher, for example:

    ./isaaclab.sh -p scripts/capture_dranmar_robot_gif.py \
        --headless --enable_cameras --device cuda:0 \
        --robot wound-preparation \
        --output docs/screenshots/robots/wound-preparation-isaac-lab.gif

The GIFs are documentation media, not qualification evidence. Runtime
qualification remains the responsibility of the robot-specific validation
programs under ``examples/``.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSET_MODULE_ROOT = (
    ROOT / "source/extensions/orbit.surgical.assets/orbit/surgical/assets"
)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--robot",
    choices=(
        "wound-preparation",
        "atraumatic-exposure",
        "adaptive-hemostasis",
        "adaptive-anastomosis",
        "adaptive-seal-divide",
    ),
    required=True,
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--width", type=int, default=720)
parser.add_argument("--height", type=int, default=450)
parser.add_argument("--frames-per-transition", type=int, default=7)
parser.add_argument("--hold-frames", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors.camera import Camera, CameraCfg
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class RobotCapture:
    module_name: str
    display_name: str
    phases: tuple[str, ...]
    spawn_height_m: float
    camera_eye: tuple[float, float, float]
    camera_target: tuple[float, float, float]
    variants: dict[str, str]


CAPTURES = {
    "wound-preparation": RobotCapture(
        module_name="wound_preparation_robot",
        display_name="Wound Preparation Robot",
        phases=("inspect", "contact", "pre_rinse", "debride", "aspirate", "post_rinse"),
        spawn_height_m=0.45,
        camera_eye=(0.68, 0.56, 0.58),
        camera_target=(0.0, 0.0, 0.34),
        variants={"irrigation_state": "loaded", "collection_state": "empty"},
    ),
    "atraumatic-exposure": RobotCapture(
        module_name="atraumatic_exposure_robot",
        display_name="Atraumatic Exposure Robot",
        phases=("stowed", "approach", "deploy", "contact", "capture", "retract", "hold", "release"),
        spawn_height_m=0.45,
        camera_eye=(0.75, 0.62, 0.65),
        camera_target=(0.0, 0.0, 0.40),
        variants={"pad_type": "fenestrated"},
    ),
    "adaptive-hemostasis": RobotCapture(
        module_name="adaptive_hemostasis_robot",
        display_name="Adaptive Hemostasis Robot",
        phases=("inspect", "clear", "compress", "clip", "release_compression", "patch", "verify"),
        spawn_height_m=0.45,
        camera_eye=(0.70, 0.58, 0.62),
        camera_target=(0.0, 0.0, 0.39),
        variants={
            "clip_state": "loaded",
            "patch_state": "loaded",
            "irrigation_state": "full",
            "collection_state": "empty",
        },
    ),
    "adaptive-anastomosis": RobotCapture(
        module_name="adaptive_anastomosis_robot",
        display_name="Adaptive Anastomosis Robot",
        phases=("inspect", "capture", "align", "approximate", "evert", "staple", "reinforce", "pressurize", "verify"),
        spawn_height_m=0.45,
        camera_eye=(0.78, 0.65, 0.65),
        camera_target=(0.0, 0.0, 0.39),
        variants={
            "staple_state": "loaded",
            "collar_state": "loaded",
            "test_medium_state": "full",
        },
    ),
    "adaptive-seal-divide": RobotCapture(
        module_name="adaptive_seal_divide_robot",
        display_name="Adaptive Seal-and-Divide Robot",
        phases=("inspect", "center", "compress", "seal", "verify_seal", "retract_guard", "divide", "release"),
        spawn_height_m=0.45,
        camera_eye=(0.70, 0.58, 0.62),
        camera_target=(0.0, 0.0, 0.39),
        variants={
            "cartridge_state": "fresh",
            "saline_state": "full",
            "collection_state": "empty",
            "energy_state": "ready",
        },
    ),
}


def load_asset_module(name: str):
    path = ASSET_MODULE_ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"dranmar_capture_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Dr.Anmar asset module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def font(size: int, *, bold: bool = False):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def tensor_rgb_to_image(value) -> Image.Image:
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
        if np.issubdtype(array.dtype, np.floating) and float(array.max(initial=0.0)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def annotate(image: Image.Image, capture: RobotCapture, phase: str) -> Image.Image:
    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((0, 0, canvas.width, 58), fill=(5, 14, 24, 218))
    draw.rectangle((0, canvas.height - 34, canvas.width, canvas.height), fill=(5, 14, 24, 205))
    title_size = 19 if len(capture.display_name) > 25 else 22
    draw.text(
        (22, 10 if title_size == 22 else 13),
        f"Dr.Anmar · {capture.display_name}",
        font=font(title_size, bold=True),
        fill=(242, 248, 252, 255),
    )
    phase_label = phase.replace("_", " ").upper()
    phase_box = draw.textbbox((0, 0), phase_label, font=font(15, bold=True))
    phase_width = phase_box[2] - phase_box[0]
    draw.rounded_rectangle(
        (canvas.width - phase_width - 42, 13, canvas.width - 18, 45),
        radius=11,
        fill=(10, 143, 163, 235),
    )
    draw.text(
        (canvas.width - phase_width - 30, 20),
        phase_label,
        font=font(15, bold=True),
        fill=(255, 255, 255, 255),
    )
    draw.text(
        (22, canvas.height - 26),
        "Isaac Lab · CUDA simulation visualization · non-clinical research asset",
        font=font(14),
        fill=(218, 230, 238, 255),
    )
    return canvas.convert("RGB")


def target_vector(robot: Articulation, phase_targets: dict[str, float]) -> tuple[torch.Tensor, torch.Tensor]:
    position = robot.data.joint_pos.torch.clone()
    velocity = torch.zeros_like(position)
    name_to_index = {name: index for index, name in enumerate(robot.joint_names)}
    for target_name, value in phase_targets.items():
        is_velocity = target_name.endswith("_velocity")
        joint_name = target_name.removesuffix("_velocity")
        index = name_to_index.get(joint_name)
        if index is None:
            continue
        numeric_value = float(value)
        if joint_name.endswith("_carousel_joint") and abs(numeric_value) > 2.0 * math.pi:
            numeric_value = math.radians(numeric_value)
        if is_velocity:
            velocity[:, index] = numeric_value
        else:
            position[:, index] = numeric_value
    return position, velocity


def main() -> None:
    if args.width < 320 or args.height < 240:
        raise ValueError("GIF dimensions must be at least 320 x 240")
    if args.frames_per_transition < 1 or args.hold_frames < 1:
        raise ValueError("Frame counts must be positive")

    capture = CAPTURES[args.robot]
    helper = load_asset_module(capture.module_name)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=1.0 / 60.0,
            device=args.device,
            gravity=(0.0, 0.0, 0.0),
        )
    )
    ground_cfg = sim_utils.GroundPlaneCfg(color=(0.035, 0.045, 0.055))
    ground_cfg.func("/World/GroundPlane", ground_cfg)
    dome_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.78, 0.84, 0.92))
    dome_cfg.func("/World/DomeLight", dome_cfg)
    key_cfg = sim_utils.DistantLightCfg(intensity=2600.0, color=(1.0, 0.94, 0.86), angle=18.0)
    key_cfg.func("/World/KeyLight", key_cfg, translation=(1.0, 1.0, 2.0))

    robot = Articulation(
        helper.make_tool_cfg(
            "/World/DrAnmarRobot",
            position=(0.0, 0.0, capture.spawn_height_m),
            **capture.variants,
        )
    )
    camera = Camera(
        CameraCfg(
            prim_path="/World/DocumentationCamera",
            update_period=0.0,
            height=args.height,
            width=args.width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=34.0,
                focus_distance=0.35,
                horizontal_aperture=22.0,
                clipping_range=(0.01, 5.0),
            ),
        )
    )

    sim.reset()
    camera.set_world_poses_from_view(
        torch.tensor([capture.camera_eye], device=args.device),
        torch.tensor([capture.camera_target], device=args.device),
    )
    for _ in range(12):
        sim.step(render=True)
        robot.update(sim.get_physics_dt())
        camera.update(sim.get_physics_dt(), force_recompute=True)

    frames: list[Image.Image] = []
    previous_position = robot.data.joint_pos.torch.clone()
    for phase in capture.phases:
        print(f"Capturing {capture.display_name}: {phase}", flush=True)
        end_position, end_velocity = target_vector(robot, helper.phase_targets(phase))
        for step in range(1, args.frames_per_transition + 1):
            blend = 0.5 - 0.5 * math.cos(math.pi * step / args.frames_per_transition)
            position = previous_position + (end_position - previous_position) * blend
            robot.write_joint_position_to_sim_index(position=position)
            robot.write_joint_velocity_to_sim_index(velocity=end_velocity)
            sim.step(render=True)
            robot.update(sim.get_physics_dt())
            camera.update(sim.get_physics_dt(), force_recompute=True)
            frame = tensor_rgb_to_image(camera.data.output["rgb"])
            frames.append(annotate(frame, capture, phase))
        for _ in range(args.hold_frames):
            robot.write_joint_position_to_sim_index(position=end_position)
            robot.write_joint_velocity_to_sim_index(velocity=end_velocity)
            sim.step(render=True)
            robot.update(sim.get_physics_dt())
            camera.update(sim.get_physics_dt(), force_recompute=True)
            frame = tensor_rgb_to_image(camera.data.output["rgb"])
            frames.append(annotate(frame, capture, phase))
        previous_position = end_position.clone()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=115,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(
        f"Wrote {len(frames)} Isaac Lab frames for {capture.display_name} "
        f"to {args.output}"
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        exit_code = 1
    else:
        exit_code = 0
    try:
        simulation_app.close()
    finally:
        raise SystemExit(exit_code)
