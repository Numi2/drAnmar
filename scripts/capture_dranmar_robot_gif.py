#!/usr/bin/env python3
"""Render a reproducible Isaac Lab GIF for one Franka-mounted Dr.Anmar robot.

Run with the Isaac Lab Python launcher, for example:

    ./isaaclab.sh -p scripts/capture_dranmar_robot_gif.py \
        --headless --enable_cameras --device cuda:0 \
        --robot wound-preparation \
        --output docs/screenshots/robots/wound-preparation-isaac-lab.gif

Each scene uses the same complete Franka composition and procedure fixture as
the corresponding CUDA qualification program. The GIFs are documentation
media, not qualification evidence. Runtime qualification remains the
responsibility of the robot-specific validation programs under ``examples/``.
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
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors.camera import Camera, CameraCfg
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, UsdGeom


@dataclass(frozen=True)
class RobotCapture:
    module_name: str
    display_name: str
    phases: tuple[str, ...]
    variants: dict[str, str]
    task_label: str
    tcp_offset_m: float


CAPTURES = {
    "wound-preparation": RobotCapture(
        module_name="wound_preparation_robot",
        display_name="Wound Preparation Robot",
        phases=("inspect", "contact", "pre_rinse", "debride", "aspirate", "post_rinse"),
        variants={"irrigation_state": "loaded", "collection_state": "empty"},
        task_label="WOUND BED · IRRIGATION / ASPIRATION / DEBRIDEMENT",
        tcp_offset_m=0.172,
    ),
    "atraumatic-exposure": RobotCapture(
        module_name="atraumatic_exposure_robot",
        display_name="Atraumatic Exposure Robot",
        phases=("stowed", "approach", "deploy", "contact", "capture", "retract", "hold", "release"),
        variants={"pad_type": "fenestrated"},
        task_label="BILATERAL SOFT-TISSUE EXPOSURE · FENESTRATED PADS",
        tcp_offset_m=0.184,
    ),
    "adaptive-hemostasis": RobotCapture(
        module_name="adaptive_hemostasis_robot",
        display_name="Adaptive Hemostasis Robot",
        phases=("inspect", "clear", "compress", "clip", "release_compression", "patch", "verify"),
        variants={
            "clip_state": "loaded",
            "patch_state": "loaded",
            "irrigation_state": "full",
            "collection_state": "empty",
        },
        task_label="VESSEL HEMOSTASIS · COMPRESSION / CLIP / PATCH",
        tcp_offset_m=0.184,
    ),
    "adaptive-anastomosis": RobotCapture(
        module_name="adaptive_anastomosis_robot",
        display_name="Adaptive Anastomosis Robot",
        phases=("inspect", "capture", "align", "approximate", "evert", "staple", "reinforce", "pressurize", "verify"),
        variants={
            "staple_state": "loaded",
            "collar_state": "loaded",
            "test_medium_state": "full",
        },
        task_label="HOLLOW-TISSUE ANASTOMOSIS · STAPLE / REINFORCE / TEST",
        tcp_offset_m=0.205,
    ),
    "adaptive-seal-divide": RobotCapture(
        module_name="adaptive_seal_divide_robot",
        display_name="Adaptive Seal-and-Divide Robot",
        phases=("inspect", "center", "compress", "seal", "verify_seal", "retract_guard", "divide", "release"),
        variants={
            "cartridge_state": "fresh",
            "saline_state": "full",
            "collection_state": "empty",
            "energy_state": "ready",
        },
        task_label="VESSEL SEALING AND DIVISION · INTERLOCKED BLADE",
        tcp_offset_m=0.190,
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
    draw.rectangle((0, canvas.height - 50, canvas.width, canvas.height), fill=(5, 14, 24, 218))
    title_size = 16 if len(capture.display_name) > 25 else 19
    draw.text(
        (22, 14 if title_size == 19 else 16),
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
        (22, canvas.height - 42),
        capture.task_label,
        font=font(13, bold=True),
        fill=(128, 226, 236, 255),
    )
    draw.text(
        (22, canvas.height - 23),
        "COMPLETE FRANKA ASSEMBLY · ISAAC LAB / CUDA · NON-CLINICAL RESEARCH",
        font=font(12),
        fill=(218, 230, 238, 255),
    )
    return canvas.convert("RGB")


def spawn_franka_and_task(capture: RobotCapture, helper):
    root_path = "/World/Robot"
    stage = omni.usd.get_context().get_stage()

    if args.robot == "wound-preparation":
        tool_path = f"{root_path}/DrAnmarWoundPreparationTool"
        robot_cfg = helper.make_franka_wound_preparation_robot_cfg(
            prim_path=root_path,
            **capture.variants,
        )
    elif args.robot == "atraumatic-exposure":
        tool_path = f"{root_path}/DrAnmarAtraumaticExposureTool"
        robot_cfg = helper.make_franka_exposure_robot_cfg(
            prim_path=root_path,
            **capture.variants,
        )
    elif args.robot == "adaptive-hemostasis":
        tool_path = f"{root_path}/DrAnmarAdaptiveHemostasisTool"
        robot_cfg = helper.make_franka_adaptive_hemostasis_robot_cfg(
            prim_path=root_path,
            **capture.variants,
        )
    elif args.robot == "adaptive-anastomosis":
        tool_path = f"{root_path}/DrAnmarAdaptiveAnastomosisTool"
        robot_cfg = helper.make_franka_adaptive_anastomosis_robot_cfg(
            prim_path=root_path,
            **capture.variants,
        )
    else:
        tool_path = f"{root_path}/DrAnmarAdaptiveSealDivideTool"
        robot_cfg = helper.make_franka_adaptive_seal_divide_robot_cfg(
            prim_path=root_path,
            **capture.variants,
        )

    robot = Articulation(robot_cfg)

    if args.robot == "wound-preparation":
        helper.spawn_wound_bed_demo("/World/ProcedureFixture")
    elif args.robot == "atraumatic-exposure":
        helper.spawn_exposure_tissue_demo("/World/ProcedureFixture")
    elif args.robot == "adaptive-hemostasis":
        helper.spawn_vessel_demo("/World/ProcedureFixture")
    elif args.robot == "adaptive-anastomosis":
        helper.spawn_hollow_tissue_demo(
            "/World/ProcedureFixture",
            state="initial",
        )
    elif args.robot == "adaptive-seal-divide":
        helper.spawn_vessel_demo("/World/ProcedureFixture")

    return robot, tool_path


def rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    w = float(quaternion[0])
    xyz = np.asarray(quaternion[1:4], dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    return (
        vector * (w * w - float(np.dot(xyz, xyz)))
        + 2.0 * xyz * float(np.dot(xyz, vector))
        + 2.0 * w * np.cross(xyz, vector)
    )


def align_fixture_to_mounted_tcp(
    robot: Articulation,
    tcp_offset_m: float,
) -> np.ndarray:
    mount_index = robot.body_names.index("Mount")
    mount_position = (
        robot.data.body_pos_w.torch[0, mount_index].detach().cpu().numpy()
    )
    mount_quaternion = (
        robot.data.body_quat_w.torch[0, mount_index].detach().cpu().numpy()
    )
    tcp = mount_position + rotate_wxyz(
        mount_quaternion,
        np.asarray((0.0, 0.0, tcp_offset_m), dtype=np.float64),
    )
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(
        "/World/ProcedureFixture"
    )
    if not prim or not prim.IsValid():
        raise RuntimeError("Procedure fixture was not spawned")
    xformable = UsdGeom.Xformable(prim)
    translate_set = False
    orient_set = False
    for operation in xformable.GetOrderedXformOps():
        if operation.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            operation.Set(Gf.Vec3d(*[float(value) for value in tcp]))
            translate_set = True
        elif operation.GetOpType() == UsdGeom.XformOp.TypeOrient:
            operation.Set(
                Gf.Quatd(
                    float(mount_quaternion[0]),
                    float(mount_quaternion[1]),
                    float(mount_quaternion[2]),
                    float(mount_quaternion[3]),
                )
            )
            orient_set = True
    if not translate_set:
        xformable.AddTranslateOp().Set(
            Gf.Vec3d(*[float(value) for value in tcp])
        )
    if not orient_set:
        xformable.AddOrientOp().Set(
            Gf.Quatd(
                float(mount_quaternion[0]),
                float(mount_quaternion[1]),
                float(mount_quaternion[2]),
                float(mount_quaternion[3]),
            )
        )
    return tcp.astype(np.float32)


def camera_for_points(
    points: np.ndarray,
    *,
    distance_scale: float,
    minimum_distance: float,
    direction_xyz: tuple[float, float, float] = (1.0, 1.0, 0.62),
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    center = (minimum + maximum) * 0.5
    extent = maximum - minimum
    direction = np.asarray(direction_xyz, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    distance = max(minimum_distance, float(np.max(extent)) * distance_scale)
    eye = center + direction * distance
    return eye.astype(np.float32), center.astype(np.float32)


def update_task_visual_state(phase: str) -> None:
    if args.robot != "adaptive-anastomosis":
        return
    root = omni.usd.get_context().get_stage().GetPrimAtPath(
        "/World/ProcedureFixture"
    )
    variants = root.GetVariantSets().GetVariantSet("state")
    if not variants.IsValid():
        return
    if phase in {"align", "approximate", "evert"}:
        state = "aligned"
    elif phase in {"staple", "reinforce", "pressurize", "verify"}:
        state = "completed"
    else:
        state = "initial"
    variants.SetVariantSelection(state)


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

    table_cfg = sim_utils.CuboidCfg(
        size=(1.10, 0.76, 0.045),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.055, 0.12, 0.15),
            metallic=0.05,
            roughness=0.28,
        ),
    )
    table_cfg.func(
        "/World/ProcedureTable",
        table_cfg,
        translation=(0.35, 0.0, -0.035),
    )
    backdrop_cfg = sim_utils.CuboidCfg(
        size=(3.0, 0.04, 2.0),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.018, 0.052, 0.075),
            metallic=0.0,
            roughness=0.42,
        ),
    )
    backdrop_cfg.func(
        "/World/Backdrop",
        backdrop_cfg,
        translation=(0.30, -0.82, 0.78),
    )
    robot, _tool_path = spawn_franka_and_task(capture, helper)
    camera = Camera(
        CameraCfg(
            prim_path="/World/DocumentationCamera",
            update_period=0.0,
            height=args.height,
            width=args.width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=0.35,
                horizontal_aperture=22.0,
                clipping_range=(0.01, 5.0),
            ),
        )
    )

    sim.reset()
    tcp = align_fixture_to_mounted_tcp(robot, capture.tcp_offset_m)
    sim.reset()
    body_positions = robot.data.body_pos_w.torch[0].detach().cpu().numpy()
    tool_body_ids = [
        index
        for index, name in enumerate(robot.body_names)
        if not name.startswith("panda_link")
    ]
    tool_positions = body_positions[tool_body_ids]
    fixture_context = np.stack(
        (
            tcp + np.asarray((-0.14, -0.14, -0.10), dtype=np.float32),
            tcp + np.asarray((0.14, 0.14, 0.10), dtype=np.float32),
        )
    )
    wide_eye, wide_target = camera_for_points(
        np.concatenate((body_positions, fixture_context), axis=0),
        distance_scale=3.2,
        minimum_distance=1.55,
    )
    action_eye, action_target = camera_for_points(
        np.concatenate((tool_positions, fixture_context), axis=0),
        distance_scale=2.5,
        minimum_distance=0.50,
        direction_xyz=(0.4, 1.0, 0.18),
    )
    action_distance = float(np.linalg.norm(action_eye - action_target))
    action_direction = np.asarray((0.4, 1.0, 0.18), dtype=np.float32)
    action_direction /= np.linalg.norm(action_direction)
    action_target = (
        np.mean(tool_positions, axis=0) * 0.65 + tcp * 0.35
    ).astype(np.float32)
    action_eye = action_target + action_direction * action_distance
    camera.set_world_poses_from_view(
        torch.tensor([wide_eye], device=args.device),
        torch.tensor([wide_target], device=args.device),
    )
    for _ in range(12):
        sim.step(render=True)
        robot.update(sim.get_physics_dt())
        camera.update(sim.get_physics_dt(), force_recompute=True)

    frames: list[Image.Image] = []
    for _ in range(max(8, args.hold_frames * 3)):
        sim.step(render=True)
        robot.update(sim.get_physics_dt())
        camera.update(sim.get_physics_dt(), force_recompute=True)
        frame = tensor_rgb_to_image(camera.data.output["rgb"])
        frames.append(annotate(frame, capture, "FRANKA MOUNTED"))

    previous_position = robot.data.joint_pos.torch.clone()
    previous_eye = wide_eye
    previous_target = wide_target
    for phase in capture.phases:
        print(f"Capturing {capture.display_name}: {phase}", flush=True)
        update_task_visual_state(phase)
        end_position, end_velocity = target_vector(robot, helper.phase_targets(phase))
        for step in range(1, args.frames_per_transition + 1):
            blend = 0.5 - 0.5 * math.cos(math.pi * step / args.frames_per_transition)
            position = previous_position + (end_position - previous_position) * blend
            eye = previous_eye + (action_eye - previous_eye) * blend
            target = previous_target + (action_target - previous_target) * blend
            camera.set_world_poses_from_view(
                torch.tensor([eye], device=args.device),
                torch.tensor([target], device=args.device),
            )
            robot.write_joint_position_to_sim_index(position=position)
            robot.write_joint_velocity_to_sim_index(velocity=end_velocity)
            sim.step(render=True)
            robot.update(sim.get_physics_dt())
            camera.update(sim.get_physics_dt(), force_recompute=True)
            frame = tensor_rgb_to_image(camera.data.output["rgb"])
            frames.append(annotate(frame, capture, phase))
        for _ in range(args.hold_frames):
            camera.set_world_poses_from_view(
                torch.tensor([action_eye], device=args.device),
                torch.tensor([action_target], device=args.device),
            )
            robot.write_joint_position_to_sim_index(position=end_position)
            robot.write_joint_velocity_to_sim_index(velocity=end_velocity)
            sim.step(render=True)
            robot.update(sim.get_physics_dt())
            camera.update(sim.get_physics_dt(), force_recompute=True)
            frame = tensor_rgb_to_image(camera.data.output["rgb"])
            frames.append(annotate(frame, capture, phase))
        previous_position = end_position.clone()
        previous_eye = action_eye
        previous_target = action_target

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
