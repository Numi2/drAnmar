#!/usr/bin/env python3
"""Headless NVIDIA Isaac Lab qualification for the DrAnmar oncology system."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = (
    ROOT
    / "source/extensions/orbit.surgical.assets/orbit/surgical/assets"
    / "oncologic_resection.py"
)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument(
    "--representation", choices=("standalone", "franka"), required=True
)
parser.add_argument(
    "--tissue-mode",
    choices=("dynamic-volume", "demo"),
    default="dynamic-volume",
    help=(
        "Use the native GPU volume-deformable Dynamic Patient liver or the "
        "lighter registered oncology visualization substrate."
    ),
)
parser.add_argument("--steps-per-phase", type=int, default=16)
parser.add_argument(
    "--liver-tool-clearance",
    type=float,
    default=0.055,
    help=(
        "Axial clearance in metres between the resection TCP and the "
        "volume-liver target during non-contact qualification."
    ),
)
parser.add_argument(
    "--deformable-bench-offset-x",
    type=float,
    default=0.45,
    help=(
        "Lateral separation from the registered target used for the native "
        "non-contact deformable stability lane."
    ),
)
parser.add_argument(
    "--maximum-liver-displacement",
    type=float,
    default=0.025,
)
parser.add_argument(
    "--maximum-liver-speed",
    type=float,
    default=0.50,
)
parser.add_argument("--render-width", type=int, default=160)
parser.add_argument("--render-height", type=int, default=120)
parser.add_argument("--skip-rendered-sensors", action="store_true")
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import carb
import numpy as np
import omni.physics.tensors as physics_tensors
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors.camera import Camera, CameraCfg


def load_helper():
    spec = importlib.util.spec_from_file_location(
        "dranmar_oncology_native_runtime", HELPER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {HELPER_PATH}")
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
    candidate = value.torch if hasattr(value, "torch") else value
    return candidate() if callable(candidate) else candidate


def world_translation(stage, path: str) -> Gf.Vec3d:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing transform prim: {path}")
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    ).ExtractTranslation()


def create_camera_sensor(
    camera_path: str,
    *,
    width: int,
    height: int,
    with_depth: bool,
):
    stage = omni.usd.get_context().get_stage()
    camera_prim = UsdGeom.Camera.Get(stage, camera_path)
    clipping = camera_prim.GetClippingRangeAttr().Get()
    data_types = ["rgb"]
    if with_depth:
        data_types.append("distance_to_camera")
    return Camera(
        CameraCfg(
            prim_path=camera_path,
            update_period=0.0,
            height=height,
            width=width,
            data_types=data_types,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=float(camera_prim.GetFocalLengthAttr().Get()),
                focus_distance=float(
                    camera_prim.GetFocusDistanceAttr().Get() or 0.4
                ),
                horizontal_aperture=float(
                    camera_prim.GetHorizontalApertureAttr().Get()
                ),
                clipping_range=(
                    float(clipping[0]),
                    float(clipping[1]),
                ),
            ),
        )
    )


def capture_registered_sensors(sim, camera_paths, *, width: int, height: int):
    evidence = {}
    acquisition_order = []
    depth_evidence = None

    def as_numpy(value):
        value = tensor_value(value)
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    for name, path in camera_paths.items():
        sensor = create_camera_sensor(
            path,
            width=width,
            height=height,
            with_depth=name == "rgb_camera_left",
        )
        if not sensor.is_initialized:
            sensor._initialize_callback(None)
        for _ in range(4):
            sim.step(render=True)
            sensor.update(sim.get_physics_dt(), force_recompute=True)
        rgb = as_numpy(sensor.data.output["rgb"])[0, ..., :3]
        if rgb.shape != (height, width, 3) or not np.isfinite(rgb).all():
            raise RuntimeError(f"Invalid rendered frame for {name}: {rgb.shape}")
        variation = float(np.std(rgb.astype(np.float32)))
        if variation <= 0.25:
            raise RuntimeError(
                f"Rendered frame {name} lacks scene variation: {variation}"
            )
        evidence[name] = {
            "shape": list(rgb.shape),
            "mean": float(np.mean(rgb)),
            "standard_deviation": variation,
        }
        if name == "rgb_camera_left":
            depth = as_numpy(sensor.data.output["distance_to_camera"])[
                0, ..., 0
            ]
            finite = depth[np.isfinite(depth)]
            if depth.shape != (height, width) or finite.size == 0:
                raise RuntimeError("Registered RGB depth capture is invalid")
            depth_evidence = {
                "shape": list(depth.shape),
                "finite_fraction": float(finite.size / depth.size),
                "minimum_m": float(np.min(finite)),
                "maximum_m": float(np.max(finite)),
            }
        acquisition_order.append(name)
        sensor.__del__()
        sensor._renderer = None
        sensor._render_data = None
        del sensor
        gc.collect()
    if depth_evidence is None:
        raise RuntimeError("Left RGB depth evidence is missing")
    evidence["depth"] = depth_evidence
    evidence["acquisition_policy"] = {
        "mode": "serialized_one_camera_at_a_time",
        "maximum_concurrent_camera_pipelines": 1,
        "order": acquisition_order,
        "fusion_requirement": (
            "timestamped buffering or interpolation to a common time "
            "followed by the 50 ms skew gate"
        ),
    }
    return evidence


def main() -> int:
    if args.steps_per_phase <= 0:
        raise ValueError("--steps-per-phase must be positive")
    if (
        not np.isfinite(args.liver_tool_clearance)
        or args.liver_tool_clearance < 0.030
    ):
        raise ValueError("--liver-tool-clearance must be at least 0.030 m")
    if (
        not np.isfinite(args.deformable_bench_offset_x)
        or args.deformable_bench_offset_x < 0.30
    ):
        raise ValueError("--deformable-bench-offset-x must be at least 0.30 m")
    if (
        not np.isfinite(args.maximum_liver_displacement)
        or args.maximum_liver_displacement <= 0.0
    ):
        raise ValueError("--maximum-liver-displacement must be positive")
    if (
        not np.isfinite(args.maximum_liver_speed)
        or args.maximum_liver_speed <= 0.0
    ):
        raise ValueError("--maximum-liver-speed must be positive")
    engine_errors = []
    logging = carb.logging.acquire_logging()

    def record_error(source, level, _filename, _line, message):
        if level >= carb.logging.LEVEL_ERROR and len(engine_errors) < 40:
            engine_errors.append(f"{source}: {message.strip()}")

    logger_handle = logging.add_logger(record_error)
    helper = load_helper()
    log_dir = ROOT / "run/qualification/logs/oncologic_resection"
    log_dir.mkdir(parents=True, exist_ok=True)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=1.0 / 120.0,
            device=args.device,
            log_dir=str(log_dir),
        )
    )
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/GroundPlane", ground)
    light = sim_utils.DomeLightCfg(intensity=2800.0)
    light.func("/World/Light", light)

    if args.representation == "standalone":
        root_path = "/World/OncologyTool"
        robot = Articulation(
            helper.make_tool_cfg(root_path, position=(0.0, 0.0, 0.10))
        )
        tool_path = root_path
    else:
        root_path = "/World/Robot"
        robot = Articulation(
            helper.make_franka_oncology_robot_cfg(prim_path=root_path)
        )
        tool_path = f"{root_path}/DrAnmarTumorResectionTool"

    stage = omni.usd.get_context().get_stage()
    tool_center = world_translation(
        stage, helper.frame_path(tool_path, "tool_center")
    )
    deformable_route = None
    tissue_registration = None
    if args.tissue_mode == "dynamic-volume":
        target_center = Gf.Vec3d(0.055, 0.005, 0.020)
        resection_tcp = world_translation(
            stage, helper.frame_path(tool_path, "resection_tcp")
        )
        registered_target = resection_tcp - Gf.Vec3d(
            0.0, 0.0, args.liver_tool_clearance
        )
        qualification_target = registered_target + Gf.Vec3d(
            args.deformable_bench_offset_x, 0.0, 0.0
        )
        liver_translation = qualification_target - target_center
        deformable_route = helper.spawn_oncology_volume_liver(
            "/World/OncologyLiver",
            position=tuple(float(value) for value in liver_translation),
            stage=stage,
        )
        tissue_registration = {
            "resection_tcp_world_m": [
                float(value) for value in resection_tcp
            ],
            "liver_target_world_m": [
                float(value) for value in registered_target
            ],
            "qualification_target_world_m": [
                float(value) for value in qualification_target
            ],
            "axial_clearance_m": args.liver_tool_clearance,
            "lateral_bench_offset_m": args.deformable_bench_offset_x,
            "qualification_mode": (
                "non_contact_stability_with_registered_target_recorded"
            ),
        }
    else:
        tumor_center = Gf.Vec3d(0.03, -0.004, 0.012)
        liver_translation = tool_center - tumor_center
        liver_cfg = sim_utils.UsdFileCfg(
            usd_path=str(helper.LIVER_DEMO_USD),
            variants={
                "procedure_state": "mapped",
                "pathology_state": "multifocal",
            },
            activate_contact_sensors=True,
        )
        liver_cfg.func(
            "/World/OncologyLiver",
            liver_cfg,
            translation=tuple(float(value) for value in liver_translation),
        )
    camera_paths = helper.attach_camera_prims(stage, tool_path)
    if set(camera_paths) != set(helper.REGISTERED_CAMERA_FRAMES):
        raise RuntimeError(f"Registered camera mismatch: {camera_paths}")
    for path in camera_paths.values():
        camera = UsdGeom.Camera.Get(stage, path)
        if not camera:
            raise RuntimeError(f"Missing registered USD camera: {path}")

    if args.representation == "franka":
        mount = UsdPhysics.FixedJoint.Get(
            stage, f"{root_path}/dranmar_oncology_mount_joint"
        )
        body0 = mount.GetBody0Rel().GetTargets() if mount else []
        body1 = mount.GetBody1Rel().GetTargets() if mount else []
        if (
            len(body0) != 1
            or body0[0].name != "panda_link8"
            or len(body1) != 1
            or str(body1[0]) != f"{tool_path}/Links/Mount"
        ):
            raise RuntimeError(
                f"Unexpected Franka mount: body0={body0}, body1={body1}"
            )

    sim.reset()
    robot.update(sim.get_physics_dt())
    deformable_view = None
    deformable_initial = None
    deformable_body_settings = None
    if deformable_route is not None:
        physics_view = physics_tensors.create_simulation_view("torch")
        physics_view.set_subspace_roots("/")
        deformable_view = physics_view.create_volume_deformable_body_view(
            deformable_route["body_prim_path"]
        )
        if deformable_view.count != 1:
            raise RuntimeError(
                "Expected one live oncology liver volume body at "
                f"{deformable_route['body_prim_path']}, found "
                f"{deformable_view.count}"
            )
        deformable_initial = (
            deformable_view.get_simulation_nodal_positions()
            .detach()
            .cpu()
            .numpy()[0]
            .astype(np.float64, copy=True)
        )
        if (
            deformable_initial.ndim != 2
            or deformable_initial.shape[1] != 3
            or not np.isfinite(deformable_initial).all()
        ):
            raise RuntimeError(
                f"Invalid initial liver nodal state: "
                f"{deformable_initial.shape}"
            )
        body_prim = stage.GetPrimAtPath(deformable_route["body_prim_path"])
        deformable_body_settings = {
            name: (
                body_prim.GetAttribute(name).Get()
                if body_prim.GetAttribute(name).IsValid()
                else None
            )
            for name in (
                "physxDeformableBody:disableGravity",
                "physxDeformableBody:selfCollision",
                "physxDeformableBody:solverPositionIterationCount",
                "physxDeformableBody:vertexVelocityDamping",
            )
        }
    joint_names = list(robot.joint_names)
    expected = 22 if args.representation == "standalone" else 29
    if len(joint_names) != expected:
        raise RuntimeError(
            f"Expected {expected} joints, got {len(joint_names)}: {joint_names}"
        )
    missing = sorted(set(helper.TOOL_JOINTS.values()) - set(joint_names))
    if missing:
        raise RuntimeError(f"Missing oncology tool joints: {missing}")

    maximum_joint_error = 0.0
    phase_evidence = []
    for phase in helper.TASK_PHASES:
        target = tensor_value(robot.data.default_joint_pos).clone()
        for name, value in helper.phase_targets(phase).items():
            target[:, joint_names.index(name)] = value
        for _ in range(args.steps_per_phase):
            robot.set_joint_position_target(target)
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
        measured = tensor_value(robot.data.joint_pos)
        if not bool(np.isfinite(measured.detach().cpu().numpy()).all()):
            raise RuntimeError(f"Non-finite joint state in phase {phase}")
        indices = [joint_names.index(name) for name in helper.TOOL_JOINTS.values()]
        error = float(
            np.max(
                np.abs(
                    (
                        measured[:, indices] - target[:, indices]
                    ).detach().cpu().numpy()
                )
            )
        )
        maximum_joint_error = max(maximum_joint_error, error)
        phase_evidence.append({"phase": phase, "maximum_joint_error": error})

    rendered = None
    if not args.skip_rendered_sensors:
        rendered = capture_registered_sensors(
            sim,
            camera_paths,
            width=args.render_width,
            height=args.render_height,
        )

    authored_mass = 0.0
    for prim in stage.Traverse():
        if prim.GetPath().HasPrefix(tool_path):
            attribute = prim.GetAttribute("physics:mass")
            if attribute and attribute.HasAuthoredValueOpinion():
                authored_mass += float(attribute.Get())
    if abs(authored_mass - 2.5534) > 1.0e-5:
        raise RuntimeError(f"Unexpected composed payload mass: {authored_mass}")

    deformable_evidence = None
    if deformable_view is not None and deformable_initial is not None:
        final_positions = (
            deformable_view.get_simulation_nodal_positions()
            .detach()
            .cpu()
            .numpy()[0]
            .astype(np.float64, copy=True)
        )
        final_velocities = (
            deformable_view.get_simulation_nodal_velocities()
            .detach()
            .cpu()
            .numpy()[0]
            .astype(np.float64, copy=True)
        )
        if (
            not np.isfinite(final_positions).all()
            or not np.isfinite(final_velocities).all()
        ):
            raise RuntimeError("Oncology liver produced non-finite nodal state")
        maximum_displacement = float(
            np.linalg.norm(
                final_positions - deformable_initial, axis=-1
            ).max()
        )
        maximum_speed = float(
            np.linalg.norm(final_velocities, axis=-1).max()
        )
        if maximum_displacement > args.maximum_liver_displacement:
            raise RuntimeError(
                "Oncology liver exceeded the displacement stability gate: "
                f"{maximum_displacement:.6f} m > "
                f"{args.maximum_liver_displacement:.6f} m; "
                f"body_settings={deformable_body_settings}; "
                f"initial_bounds="
                f"{deformable_initial.min(axis=0).tolist()}.."
                f"{deformable_initial.max(axis=0).tolist()}; "
                f"final_bounds={final_positions.min(axis=0).tolist()}.."
                f"{final_positions.max(axis=0).tolist()}"
            )
        if maximum_speed > args.maximum_liver_speed:
            raise RuntimeError(
                "Oncology liver exceeded the velocity stability gate: "
                f"{maximum_speed:.6f} m/s > "
                f"{args.maximum_liver_speed:.6f} m/s"
            )
        deformable_evidence = {
            "route": deformable_route["route"],
            "body_prim_path": deformable_route["body_prim_path"],
            "simulation_mesh_path": deformable_route["simulation_mesh_path"],
            "nodal_point_count": int(final_positions.shape[0]),
            "maximum_displacement_m": maximum_displacement,
            "maximum_allowed_displacement_m": (
                args.maximum_liver_displacement
            ),
            "maximum_speed_m_s": maximum_speed,
            "maximum_allowed_speed_m_s": args.maximum_liver_speed,
            "body_settings": deformable_body_settings,
            "registration": tissue_registration,
            "finite_state": True,
            "continuous_mechanics": "native_gpu_volume_deformable",
            "irreversible_topology": (
                "registered_discrete_resection_graph"
            ),
            "constitutive_validation": False,
        }

    ignored_error_fragments = (
        "Could not load the dynamic library",
        "omni.usd.schema",
    )
    fatal_errors = [
        message
        for message in engine_errors
        if not any(fragment in message for fragment in ignored_error_fragments)
    ]
    if fatal_errors:
        raise RuntimeError(f"Isaac/PhysX errors: {fatal_errors[:8]}")

    report = {
        "schema": "dr.anmar.oncology-native-simulator-evidence.v1",
        "status": "pass",
        "representation": args.representation,
        "tissue_mode": args.tissue_mode,
        "device": args.device,
        "isaac_lab": distribution_version("isaaclab"),
        "isaac_sim": distribution_version("isaacsim"),
        "joint_count": len(joint_names),
        "tool_joint_count": len(helper.TOOL_JOINTS),
        "authored_payload_mass_kg": authored_mass,
        "phases": phase_evidence,
        "phase_sweep_contract": (
            "finite_bounded_setpoint_acceptance_not_convergence_benchmark"
        ),
        "maximum_tool_joint_error": maximum_joint_error,
        "registered_camera_count": len(camera_paths),
        "rendered_sensor_evidence": rendered,
        "deformable_liver_evidence": deformable_evidence,
        "engine_error_count": 0,
        "clinical_validation": False,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    logging.remove_logger(logger_handle)
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
