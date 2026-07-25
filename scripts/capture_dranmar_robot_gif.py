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
parser.add_argument(
    "--standalone",
    action="store_true",
    help="Render the local articulated tool workcell without fetching the Franka asset.",
)
parser.add_argument(
    "--franka-usd",
    type=Path,
    help="Use a local cached Franka USD instead of the external asset service.",
)
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
try:
    from isaaclab_physx.physics import PhysxGpuCfg, PhysxScene
except ImportError:
    from isaaclab_physx.physics import PhysxCfg

    PhysxGpuCfg = None
    PhysxScene = None
else:
    PhysxCfg = None
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics


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

WOUND_PREPARATION_DOCUMENTATION_ARM_POSE = {
    "panda_joint1": -1.0641,
    "panda_joint2": -0.9560,
    "panda_joint3": 0.5321,
    "panda_joint4": -2.1003,
    "panda_joint5": 0.0159,
    "panda_joint6": 2.1233,
    "panda_joint7": 0.5403,
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


def tensor_value(value):
    return value.torch if hasattr(value, "torch") else value


def write_joint_positions(robot: Articulation, position: torch.Tensor) -> None:
    writer = getattr(robot, "write_joint_position_to_sim_index", None)
    if writer is not None:
        writer(position=position)
    else:
        robot.write_joint_position_to_sim(position)


def write_joint_velocities(robot: Articulation, velocity: torch.Tensor) -> None:
    writer = getattr(robot, "write_joint_velocity_to_sim_index", None)
    if writer is not None:
        writer(velocity=velocity)
    else:
        robot.write_joint_velocity_to_sim(velocity)


def set_joint_position_targets(
    robot: Articulation,
    position: torch.Tensor,
) -> None:
    writer = getattr(robot, "set_joint_position_target_index", None)
    if writer is not None:
        writer(position=position)
    else:
        robot.set_joint_position_target(position)


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
        (
            "STANDALONE ARTICULATED WORKCELL · ISAAC LAB / CUDA · NON-CLINICAL RESEARCH"
            if args.standalone
            else "COMPLETE FRANKA ASSEMBLY · ISAAC LAB / CUDA · NON-CLINICAL RESEARCH"
        ),
        font=font(12),
        fill=(218, 230, 238, 255),
    )
    return canvas.convert("RGB")


def spawn_franka_and_task(capture: RobotCapture, helper):
    root_path = "/World/Robot"
    stage = omni.usd.get_context().get_stage()

    if args.standalone:
        if args.robot != "wound-preparation":
            raise ValueError("--standalone currently supports wound-preparation")
        root_path = "/World/WoundPreparationTool"
        tool_path = root_path
        robot_cfg = helper.make_tool_cfg(
            prim_path=root_path,
            irrigation_state="loaded",
            collection_state="empty",
            position=(0.45, 0.0, 0.45),
            orientation_wxyz=(0.0, 1.0, 0.0, 0.0),
        )
    elif args.robot == "wound-preparation":
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

    if args.robot == "wound-preparation" and not args.standalone:
        robot_cfg.init_state.joint_pos.update(
            WOUND_PREPARATION_DOCUMENTATION_ARM_POSE
        )
    if args.franka_usd is not None and not args.standalone:
        franka_usd = args.franka_usd.expanduser().resolve()
        if not franka_usd.is_file():
            raise FileNotFoundError(f"Local Franka USD does not exist: {franka_usd}")
        robot_cfg.spawn.usd_path = str(franka_usd)
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
    *,
    align_orientation_to_mount: bool = True,
) -> np.ndarray:
    mount_index = robot.body_names.index("Mount")
    mount_position = (
        tensor_value(robot.data.body_pos_w)[0, mount_index].detach().cpu().numpy()
    )
    mount_quaternion = (
        tensor_value(robot.data.body_quat_w)[0, mount_index].detach().cpu().numpy()
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
    fixture_quaternion = (
        mount_quaternion
        if align_orientation_to_mount
        else np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32)
    )
    translate_set = False
    orient_set = False
    for operation in xformable.GetOrderedXformOps():
        if operation.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            operation.Set(Gf.Vec3d(*[float(value) for value in tcp]))
            translate_set = True
        elif operation.GetOpType() == UsdGeom.XformOp.TypeOrient:
            operation.Set(
                Gf.Quatd(
                    float(fixture_quaternion[0]),
                    float(fixture_quaternion[1]),
                    float(fixture_quaternion[2]),
                    float(fixture_quaternion[3]),
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
                float(fixture_quaternion[0]),
                float(fixture_quaternion[1]),
                float(fixture_quaternion[2]),
                float(fixture_quaternion[3]),
            )
        )
    return tcp.astype(np.float32)


def set_prim_translation(prim_path: str, translation: np.ndarray) -> None:
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing prim for documentation layout: {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    for operation in xformable.GetOrderedXformOps():
        if operation.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            operation.Set(Gf.Vec3d(*[float(value) for value in translation]))
            return
    xformable.AddTranslateOp().Set(
        Gf.Vec3d(*[float(value) for value in translation])
    )


def position_franka_for_downward_workcell(
    robot: Articulation,
) -> torch.Tensor:
    arm_indices = [
        robot.joint_names.index(f"panda_joint{index}")
        for index in range(1, 8)
    ]
    mount_index = robot.body_names.index("Mount")
    joint_state = tensor_value(robot.data.joint_pos).clone()
    arm_state = (
        joint_state[0, arm_indices].detach().cpu().numpy().astype(np.float64)
    )
    limits = (
        tensor_value(robot.data.soft_joint_pos_limits)[0, arm_indices]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    target = np.concatenate(
        (
            np.asarray((0.45, 0.0, 0.48), dtype=np.float64),
            np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
        )
    )
    row_scale = np.asarray((2.5, 2.5, 2.5, 1.0, 1.0, 1.0))

    def apply_and_measure(values: np.ndarray) -> np.ndarray:
        updated = joint_state.clone()
        updated[0, arm_indices] = torch.as_tensor(
            values,
            dtype=updated.dtype,
            device=updated.device,
        )
        write_joint_positions(robot, updated)
        write_joint_velocities(robot, torch.zeros_like(updated))
        # PhysX refreshes forward kinematics when the body-pose tensor is read
        # after a direct joint-state write. Stepping here would let the PD
        # drives move toward an earlier target and corrupt the finite
        # difference Jacobian.
        position = (
            tensor_value(robot.data.body_pos_w)[0, mount_index]
            .detach()
            .cpu()
            .numpy()
        )
        quaternion = (
            tensor_value(robot.data.body_quat_w)[0, mount_index]
            .detach()
            .cpu()
            .numpy()
        )
        return np.concatenate(
            (
                position,
                rotate_wxyz(quaternion, np.asarray((0.0, 0.0, 1.0))),
            )
        )

    epsilon = 5.0e-3
    for _ in range(36):
        feature = apply_and_measure(arm_state)
        residual = (feature - target) * row_scale
        if float(np.linalg.norm(residual)) < 1.5e-3:
            break
        jacobian = np.zeros((feature.size, arm_state.size), dtype=np.float64)
        for column in range(arm_state.size):
            perturbed = arm_state.copy()
            perturbed[column] += epsilon
            jacobian[:, column] = (
                apply_and_measure(perturbed) - feature
            ) / epsilon
        jacobian *= row_scale[:, None]
        normal = jacobian.T @ jacobian + np.eye(arm_state.size) * 1.0e-2
        delta = -np.linalg.solve(normal, jacobian.T @ residual)
        delta = np.clip(delta, -0.16, 0.16)
        arm_state = np.clip(
            arm_state + delta,
            limits[:, 0] + 1.0e-3,
            limits[:, 1] - 1.0e-3,
        )

    final_feature = apply_and_measure(arm_state)
    if final_feature[5] > -0.985:
        raise RuntimeError(
            "Unable to align the Franka wound-preparation approach axis downward: "
            f"mount={final_feature[:3].round(4).tolist()} "
            f"approach={final_feature[3:6].round(4).tolist()}"
        )
    final_state = tensor_value(robot.data.joint_pos).clone()
    set_joint_position_targets(robot, final_state)
    print(
        "Franka documentation pose: "
        f"mount={final_feature[:3].round(4).tolist()} "
        f"approach={final_feature[3:6].round(4).tolist()}",
        flush=True,
    )
    return final_state


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
    position = tensor_value(robot.data.joint_pos).clone()
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
    sim_cfg_kwargs = {
        "dt": 1.0 / 60.0,
        "device": args.device,
        "gravity": (0.0, 0.0, 0.0),
    }
    if PhysxCfg is not None:
        sim_cfg_kwargs["physics"] = PhysxCfg(
            gpu_max_soft_body_contacts=2**21,
        )
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(**sim_cfg_kwargs)
    )
    stage = omni.usd.get_context().get_stage()
    physics_scene = UsdPhysics.Scene.Get(stage, sim.cfg.physics_prim_path)
    if not physics_scene:
        physics_scene = UsdPhysics.Scene.Define(stage, sim.cfg.physics_prim_path)
    physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())
    physx_scene_api.CreateGpuMaxDeformableSurfaceContactsAttr(2**21)
    if PhysxScene is not None and PhysxGpuCfg is not None:
        PhysxScene(sim.cfg.physics_prim_path).set_gpu_configuration(
            PhysxGpuCfg(gpu_max_deformable_surface_contacts=2**21)
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
    franka_documentation_joint_state = None
    if args.robot == "wound-preparation" and not args.standalone:
        franka_documentation_joint_state = position_franka_for_downward_workcell(
            robot
        )
    tcp = align_fixture_to_mounted_tcp(
        robot,
        capture.tcp_offset_m,
        align_orientation_to_mount=args.robot != "wound-preparation",
    )
    wound_runtime = None
    if args.robot == "wound-preparation":
        set_prim_translation(
            "/World/ProcedureTable",
            tcp + np.asarray((0.08, 0.0, -0.055), dtype=np.float32),
        )
        stage = omni.usd.get_context().get_stage()
        helper.apply_wound_surface_deformable(
            "/World/ProcedureFixture",
            stage=stage,
        )
        helper.attach_demo_debris(
            "/World/ProcedureFixture",
            stage=stage,
        )
        particle_paths = helper.ensure_irrigation_particle_system(stage=stage)
        wound_runtime = {
            "stage": stage,
            "particle_set_path": particle_paths["particle_set_path"],
            "ledger": helper.FluidLedger(),
            "suction": helper.SuctionFieldController(),
        }
    sim.reset()
    if franka_documentation_joint_state is not None:
        write_joint_positions(robot, franka_documentation_joint_state)
        write_joint_velocities(
            robot,
            torch.zeros_like(franka_documentation_joint_state),
        )
        set_joint_position_targets(robot, franka_documentation_joint_state)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim.get_physics_dt())
    body_positions = (
        tensor_value(robot.data.body_pos_w)[0].detach().cpu().numpy()
    )
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
        direction_xyz=(0.55, 0.75, 0.48),
    )
    action_distance = float(np.linalg.norm(action_eye - action_target))
    action_direction = np.asarray((0.58, 0.72, 0.64), dtype=np.float32)
    action_direction /= np.linalg.norm(action_direction)
    if args.robot == "wound-preparation":
        mount_position = body_positions[robot.body_names.index("Mount")]
        # Frame the complete distal mechanism and the wound together. A TCP-only
        # target crops the vertically mounted tool above the sensor on a
        # downward workcell pose.
        action_target = ((mount_position + tcp) * 0.5).astype(np.float32)
        action_distance = 0.72
    else:
        action_target = (
            np.mean(tool_positions, axis=0) * 0.65 + tcp * 0.35
        ).astype(np.float32)
    action_eye = action_target + action_direction * action_distance
    camera.set_world_poses_from_view(
        torch.as_tensor(np.asarray([wide_eye]), device=args.device),
        torch.as_tensor(np.asarray([wide_target]), device=args.device),
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
        frames.append(
            annotate(
                frame,
                capture,
                "ARTICULATED WORKCELL" if args.standalone else "FRANKA MOUNTED",
            )
        )

    previous_position = tensor_value(robot.data.joint_pos).clone()
    previous_eye = wide_eye
    previous_target = wide_target
    for phase in capture.phases:
        print(f"Capturing {capture.display_name}: {phase}", flush=True)
        update_task_visual_state(phase)
        if wound_runtime is not None and phase in {"pre_rinse", "post_rinse"}:
            helper.emit_irrigation_burst(
                _tool_path,
                wound_runtime["ledger"],
                requested_ml=0.12,
                random_seed=17 if phase == "pre_rinse" else 29,
                stage=wound_runtime["stage"],
                particle_set_path=wound_runtime["particle_set_path"],
            )
        end_position, end_velocity = target_vector(robot, helper.phase_targets(phase))
        for step in range(1, args.frames_per_transition + 1):
            blend = 0.5 - 0.5 * math.cos(math.pi * step / args.frames_per_transition)
            position = previous_position + (end_position - previous_position) * blend
            eye = previous_eye + (action_eye - previous_eye) * blend
            target = previous_target + (action_target - previous_target) * blend
            camera.set_world_poses_from_view(
                torch.as_tensor(np.asarray([eye]), device=args.device),
                torch.as_tensor(np.asarray([target]), device=args.device),
            )
            write_joint_positions(robot, position)
            write_joint_velocities(robot, end_velocity)
            sim.step(render=True)
            if wound_runtime is not None and phase in {
                "aspirate",
                "debride",
                "post_rinse",
            }:
                wound_runtime["suction"].update_particles(
                    _tool_path,
                    wound_runtime["ledger"],
                    dt=sim.get_physics_dt(),
                    opening=1.0,
                    stage=wound_runtime["stage"],
                    particle_set_path=wound_runtime["particle_set_path"],
                )
            robot.update(sim.get_physics_dt())
            camera.update(sim.get_physics_dt(), force_recompute=True)
            frame = tensor_rgb_to_image(camera.data.output["rgb"])
            frames.append(annotate(frame, capture, phase))
        for _ in range(args.hold_frames):
            camera.set_world_poses_from_view(
                torch.as_tensor(np.asarray([action_eye]), device=args.device),
                torch.as_tensor(np.asarray([action_target]), device=args.device),
            )
            write_joint_positions(robot, end_position)
            write_joint_velocities(robot, end_velocity)
            sim.step(render=True)
            if wound_runtime is not None and phase in {
                "aspirate",
                "debride",
                "post_rinse",
            }:
                wound_runtime["suction"].update_particles(
                    _tool_path,
                    wound_runtime["ledger"],
                    dt=sim.get_physics_dt(),
                    opening=1.0,
                    stage=wound_runtime["stage"],
                    particle_set_path=wound_runtime["particle_set_path"],
                )
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
