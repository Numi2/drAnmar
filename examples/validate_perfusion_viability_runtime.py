#!/usr/bin/env python3
"""Headless CUDA qualification for the DrAnmar perfusion-viability system."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = (
    ROOT
    / "source/extensions/orbit.surgical.assets/orbit/surgical/assets"
    / "perfusion_viability_robot.py"
)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--representation", choices=("standalone", "franka"), required=True)
parser.add_argument("--steps", type=int, default=260)
parser.add_argument("--arm-pose-steps", type=int, default=48)
parser.add_argument("--render-width", type=int, default=160)
parser.add_argument("--render-height", type=int, default=120)
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import carb
import numpy as np
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors.camera import Camera, CameraCfg


def load_helper():
    spec = importlib.util.spec_from_file_location(
        "dranmar_perfusion_viability_runtime", HELPER_PATH
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
    return value.torch if hasattr(value, "torch") else value


def world_translation(stage, path: str):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        root = "/" + path.strip("/").split("/", 1)[0]
        found = [
            str(item.GetPath())
            for item in stage.Traverse()
            if str(item.GetPath()).startswith(root)
        ][:30]
        raise RuntimeError(f"Missing transform prim: {path}; composed prims={found}")
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    ).ExtractTranslation()


def create_kinematic_fixture(stage, path: str, position, scale):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddScaleOp().Set(Gf.Vec3d(*scale))
    cube.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    body = UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    body.CreateRigidBodyEnabledAttr().Set(True)
    cube.GetPrim().CreateAttribute(
        "physics:kinematicEnabled", Sdf.ValueTypeNames.Bool
    ).Set(True)
    return path


def create_registered_camera_sensor(name, camera_path, *, width: int, height: int):
    stage = omni.usd.get_context().get_stage()
    camera_prim = UsdGeom.Camera.Get(stage, camera_path)
    clipping = camera_prim.GetClippingRangeAttr().Get()
    data_types = ["rgb"]
    if name == "rgb_left_camera":
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


def capture_registered_camera_frames(sim, camera_paths, *, width: int, height: int):
    """Capture registered cameras serially with one live RTX sensor pipeline."""

    camera_frames = {}
    depth_frame = None
    acquisition_order = []

    def as_numpy(value):
        value = tensor_value(value)
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    for name, camera_path in camera_paths.items():
        sensor = create_registered_camera_sensor(
            name,
            camera_path,
            width=width,
            height=height,
        )
        # Physics is already ready, so initialize this one camera directly
        # rather than resetting the articulation between modalities.
        if not sensor.is_initialized:
            sensor._initialize_callback(None)
        for _ in range(4):
            sim.step(render=True)
            sensor.update(sim.get_physics_dt(), force_recompute=True)
        camera_frames[name] = as_numpy(sensor.data.output["rgb"])[0].copy()
        if name == "rgb_left_camera":
            depth_frame = (
                as_numpy(sensor.data.output["distance_to_camera"])[0, ..., 0]
                .copy()
            )
        acquisition_order.append(name)
        # Release the renderer and callbacks before constructing the next
        # modality camera. Isaac Lab otherwise retains render products.
        sensor.__del__()
        sensor._renderer = None
        sensor._render_data = None
        del sensor
        gc.collect()

    if depth_frame is None:
        raise RuntimeError("Serialized left-camera depth capture is missing")
    evidence = {}
    for name, frame in camera_frames.items():
        if frame.ndim != 3 or frame.shape[:2] != (height, width):
            raise RuntimeError(
                f"Rendered camera {name} has unexpected shape {frame.shape}"
            )
        rgb = frame[..., :3].astype(np.float32)
        variation = float(np.std(rgb))
        if not np.isfinite(rgb).all() or variation <= 0.25:
            raise RuntimeError(
                f"Rendered camera {name} lacks finite image variation: "
                f"std={variation}"
            )
        evidence[name] = {
            "shape": list(frame.shape),
            "mean": float(np.mean(rgb)),
            "standard_deviation": variation,
        }
    if depth_frame.shape != (height, width):
        raise RuntimeError(
            f"Rendered depth has unexpected shape {depth_frame.shape}"
        )
    finite_depth = depth_frame[np.isfinite(depth_frame)]
    if finite_depth.size == 0:
        raise RuntimeError("Rendered depth contains no finite samples")
    evidence["depth"] = {
        "shape": list(depth_frame.shape),
        "finite_fraction": float(finite_depth.size / depth_frame.size),
        "minimum_m": float(np.min(finite_depth)),
        "maximum_m": float(np.max(finite_depth)),
    }
    evidence["acquisition_policy"] = {
        "mode": "serialized_one_camera_at_a_time",
        "maximum_concurrent_camera_pipelines": 1,
        "render_steps_per_camera": 4,
        "order": acquisition_order,
        "alignment": "timestamped_continuous_simulation_capture",
        "estimated_serial_span_s": (
            len(acquisition_order) * 4 * sim.get_physics_dt()
        ),
        "operational_note": (
            "live fusion must buffer or interpolate each timestamped frame "
            "to a common fusion time and apply the runtime skew gate"
        ),
    }
    return camera_frames, depth_frame, evidence


def main() -> int:
    if args.steps < 20:
        raise ValueError("--steps must be at least 20")
    if args.arm_pose_steps < 10:
        raise ValueError("--arm-pose-steps must be at least 10")
    if args.render_width < 32 or args.render_height < 32:
        raise ValueError("render dimensions must be at least 32 pixels")
    engine_errors: list[str] = []
    carb_logging = carb.logging.acquire_logging()

    def record_error(source, level, _filename, _line, message):
        if level >= carb.logging.LEVEL_ERROR and len(engine_errors) < 30:
            engine_errors.append(f"{source}: {message.strip()}")

    logger_handle = carb_logging.add_logger(record_error)
    helper = load_helper()
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device)
    )
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/GroundPlane", ground)
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light.func("/World/Light", light)

    if args.representation == "standalone":
        root_path = "/World/PerfusionViabilityTool"
        robot = Articulation(
            helper.make_tool_cfg(root_path, position=(0.0, 0.0, 0.10))
        )
        tool_path = root_path
    else:
        root_path = "/World/Robot"
        robot = Articulation(
            helper.make_franka_perfusion_viability_robot_cfg(prim_path=root_path)
        )
        tool_path = f"{root_path}/DrAnmarPerfusionViabilityTool"

    stage = omni.usd.get_context().get_stage()
    tcp_path = helper.frame_path(tool_path, "perfusion_tcp")
    tcp_translation = world_translation(stage, tcp_path)
    tissue_translation = tuple(
        float(value)
        for value in (
            tcp_translation[0],
            tcp_translation[1],
            tcp_translation[2] - 0.008,
        )
    )
    tissue_root = "/World/DrAnmarPerfusedTissue"
    helper.spawn_tissue_demo(
        tissue_root,
        condition="anastomotic_stenosis",
        translation=tissue_translation,
    )
    deformable = helper.apply_perfused_tissue_surface_deformable(
        tissue_root, self_collision=False, stage=stage
    )
    surface_path = deformable["mesh_path"]
    fixture_paths = []
    attachment_types = []
    for side, x_offset in (("Left", -0.087), ("Right", 0.087)):
        fixture_path = f"/World/PerfusedTissueFixture{side}"
        create_kinematic_fixture(
            stage,
            fixture_path,
            (
                tissue_translation[0] + x_offset,
                tissue_translation[1],
                tissue_translation[2] + 0.004,
            ),
            (0.012, 0.128, 0.012),
        )
        attachment_path = f"{tissue_root}/RuntimeAttachments/{side}Edge"
        attachment_types.append(
            helper.create_perfused_tissue_fixture_attachment(
                surface_path,
                fixture_path,
                attachment_path,
                stage=stage,
            )
        )
        fixture_paths.append(fixture_path)
    camera_paths = helper.attach_camera_prims(stage, tool_path)
    if len(camera_paths) != 6:
        raise RuntimeError(f"Expected six registered camera prims: {camera_paths}")
    for path in camera_paths.values():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
            raise RuntimeError(f"Invalid registered camera prim: {path}")

    if args.representation == "franka":
        mount_joint = UsdPhysics.FixedJoint.Get(
            stage, f"{root_path}/dranmar_perfusion_mount_joint"
        )
        if not mount_joint:
            raise RuntimeError("Franka payload mount joint is missing")
        body0 = mount_joint.GetBody0Rel().GetTargets()
        body1 = mount_joint.GetBody1Rel().GetTargets()
        if (
            len(body0) != 1
            or body0[0].name != "panda_link8"
            or len(body1) != 1
            or str(body1[0]) != f"{tool_path}/Links/Mount"
        ):
            raise RuntimeError(
                f"Unexpected Franka mount relationship: body0={body0}, body1={body1}"
            )

    sim.reset()
    robot.update(sim.get_physics_dt())
    initial_tcp = world_translation(stage, tcp_path)
    tissue_surface = world_translation(stage, tissue_root) + Gf.Vec3d(0, 0, 0.008)
    registration_error_m = float((initial_tcp - tissue_surface).GetLength())
    if registration_error_m > 0.003:
        tissue_prim = stage.GetPrimAtPath(tissue_root)
        tissue_xform = UsdGeom.Xformable(tissue_prim)
        tissue_xform.ClearXformOpOrder()
        tissue_xform.AddTranslateOp().Set(
            Gf.Vec3d(initial_tcp[0], initial_tcp[1], initial_tcp[2] - 0.008)
        )
        tissue_surface = world_translation(stage, tissue_root) + Gf.Vec3d(0, 0, 0.008)
        registration_error_m = float((initial_tcp - tissue_surface).GetLength())
    if registration_error_m > 0.003:
        raise RuntimeError(
            f"Perfused tissue is not registered to the TCP: {registration_error_m} m"
        )

    joint_names = list(robot.joint_names)
    expected_joint_count = 12 if args.representation == "standalone" else 19
    if len(joint_names) != expected_joint_count:
        raise RuntimeError(
            f"Expected {expected_joint_count} joints, got {len(joint_names)}: "
            f"{joint_names}"
        )
    missing_tool = sorted(set(helper.TOOL_JOINTS.values()) - set(joint_names))
    if missing_tool:
        raise RuntimeError(f"Missing perfusion tool joints: {missing_tool}")
    if args.representation == "franka":
        missing_arm = sorted(
            {f"panda_joint{index}" for index in range(1, 8)} - set(joint_names)
        )
        if missing_arm:
            raise RuntimeError(f"Missing Franka arm joints: {missing_arm}")

    phases_checked = []
    steps_per_phase = max(1, args.steps // len(helper.TASK_PHASES))
    for phase in helper.TASK_PHASES:
        phase_contract = helper.phase_targets(phase)
        target = tensor_value(robot.data.default_joint_pos).clone()
        for name, value in phase_contract.items():
            target[:, joint_names.index(name)] = value
        robot.set_joint_position_target(target)
        for _ in range(steps_per_phase):
            robot.write_data_to_sim()
            sim.step(render=not args.headless)
            robot.update(sim.get_physics_dt())
        phases_checked.append(phase)

    camera_frames, depth_frame, rendered_sensor_evidence = (
        capture_registered_camera_frames(
            sim,
            camera_paths,
            width=args.render_width,
            height=args.render_height,
        )
    )
    arm_motion = {}
    if args.representation == "franka":
        arm_indices = [joint_names.index(f"panda_joint{index}") for index in range(1, 8)]
        arm_offsets = {
            "neutral": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "left_sweep": (0.18, -0.08, 0.04, 0.10, 0.0, -0.10, 0.08),
            "right_sweep": (-0.18, 0.06, -0.04, -0.08, 0.06, 0.10, -0.08),
        }
        for pose_name, offsets in arm_offsets.items():
            target = tensor_value(robot.data.default_joint_pos).clone()
            for index, offset in zip(arm_indices, offsets):
                target[:, index] += offset
            for name, value in helper.phase_targets("fuse").items():
                target[:, joint_names.index(name)] = value
            maximum_torque = 0.0
            minimum_tcp_z = float("inf")
            for _ in range(args.arm_pose_steps):
                robot.set_joint_position_target(target)
                robot.write_data_to_sim()
                sim.step(render=False)
                robot.update(sim.get_physics_dt())
                applied = getattr(robot.data, "applied_torque", None)
                if applied is not None:
                    torque = tensor_value(applied).detach().cpu().numpy()
                    maximum_torque = max(
                        maximum_torque,
                        float(np.max(np.abs(torque[:, arm_indices]))),
                    )
                minimum_tcp_z = min(
                    minimum_tcp_z,
                    float(world_translation(stage, tcp_path)[2]),
                )
            actual = tensor_value(robot.data.joint_pos).detach().cpu().numpy()
            expected = target.detach().cpu().numpy()
            maximum_error = float(
                np.max(np.abs(actual[:, arm_indices] - expected[:, arm_indices]))
            )
            if not np.isfinite(actual).all() or maximum_error > 0.35:
                raise RuntimeError(
                    f"Loaded Franka pose {pose_name} failed: "
                    f"maximum_error={maximum_error}"
                )
            if minimum_tcp_z <= 0.015:
                raise RuntimeError(
                    f"Loaded Franka pose {pose_name} violated ground clearance: "
                    f"{minimum_tcp_z} m"
                )
            arm_motion[pose_name] = {
                "steps": args.arm_pose_steps,
                "maximum_arm_joint_error_rad": maximum_error,
                "maximum_applied_arm_torque_nm": maximum_torque,
                "minimum_tcp_height_m": minimum_tcp_z,
            }

    joint_pos = tensor_value(robot.data.joint_pos).detach().cpu().numpy()
    if not np.isfinite(joint_pos).all():
        bad = np.argwhere(~np.isfinite(joint_pos))
        raise RuntimeError(f"Non-finite articulation state: {bad.tolist()}")

    flow_solver = helper.VascularFlowSolver()
    conservation = {}
    for condition in sorted(helper.VALID_CONDITIONS):
        flow = flow_solver.solve(condition)
        conservation[condition] = flow.conservation_error_ml_s
        if abs(flow.conservation_error_ml_s) > 1.0e-8:
            raise RuntimeError(
                f"Flow conservation failed for {condition}: "
                f"{flow.conservation_error_ml_s}"
            )

    closed_loop = {}
    verifier = helper.ClosedLoopPerfusionVerifier()
    expected_causes = {
        "arterial_occlusion": "arterial_inflow_obstruction",
        "venous_congestion": "venous_outflow_obstruction",
        "anastomotic_stenosis": "anastomotic_stenosis",
        "branch_leak": "active_branch_leak",
        "retraction_ischemia": "external_compression",
        "dressing_compression": "external_compression",
    }
    for condition in (
        "arterial_occlusion",
        "venous_congestion",
        "anastomotic_stenosis",
        "branch_leak",
        "retraction_ischemia",
        "dressing_compression",
    ):
        scan = verifier.scan_intervene_rescan(
            condition, duration_s=18.0, dt_s=0.15
        )
        before = scan["before"].assessment
        after = scan["after"].assessment
        arrays = (
            scan["before"].maps.flow_index,
            scan["before"].maps.icg_intensity,
            scan["before"].maps.icg_extravascular,
            scan["before"].maps.speckle_perfusion,
            scan["before"].maps.temperature_c,
            scan["before"].maps.oxygenation_fraction,
            scan["before"].maps.doppler_speed_m_s,
            scan["before"].maps.ultrasound_patency,
            scan["before"].maps.confidence,
        )
        if any(array.shape != (4, 6) or not np.isfinite(array).all() for array in arrays):
            raise RuntimeError(f"Invalid multimodal map for {condition}")
        if (
            before.abstained
            or before.likely_cause != expected_causes[condition]
            or scan["after_condition"] != "recovered"
            or not scan["intervention_completed"]
            or scan["recovery_fraction"] < 0.95
            or scan["viability_gain"] <= 0.0
        ):
            raise RuntimeError(
                f"Closed-loop recovery failed for {condition}: "
                f"cause={before.likely_cause}, "
                f"gain={scan['viability_gain']}, "
                f"recovery={scan['recovery_fraction']}"
            )
        closed_loop[condition] = {
            "cause": before.likely_cause,
            "action": scan["action"],
            "before_viability": before.global_viability_score,
            "after_viability": after.global_viability_score,
            "viability_gain": scan["viability_gain"],
            "nonperfused_fraction_reduction": scan[
                "nonperfused_fraction_reduction"
            ],
            "diagnostic_confidence": before.diagnostic_confidence,
            "abstained": before.abstained,
            "recovery_fraction": scan["recovery_fraction"],
            "intervention_evidence_source": scan["evidence_source"],
        }

    sensor_scan = verifier.scan("healthy", duration_s=6.0, dt_s=0.20)
    packet = helper.build_registered_sensor_packet(
        timestamp_s=sensor_scan.final_tracer.time_s,
        camera_frames=camera_frames,
        depth_frame=depth_frame,
        maps=sensor_scan.maps,
    )
    if not packet.valid:
        raise RuntimeError(
            "Registered rendered-sensor packet failed: "
            + "; ".join(packet.errors)
        )
    rendered_sensor_evidence["packet_timestamp_s"] = packet.timestamp_s
    rendered_sensor_evidence["registered_packet_shape_contract_valid"] = (
        packet.valid
    )

    probe_controller = helper.ProbeContactController()
    nominal_contact = probe_controller.update(
        measured_force_n=1.2, dt_s=1.0 / 120.0
    )
    overload_contact = probe_controller.update(
        measured_force_n=4.2, dt_s=1.0 / 120.0
    )
    if (
        not nominal_contact.coupled
        or nominal_contact.abort
        or not overload_contact.abort
        or overload_contact.target_extension_delta_m >= 0.0
    ):
        raise RuntimeError("Probe preload control contract failed")

    tissue_prim = stage.GetPrimAtPath(surface_path)
    attachment_prims = [
        stage.GetPrimAtPath(f"{tissue_root}/RuntimeAttachments/{side}Edge")
        for side in ("Left", "Right")
    ]
    if (
        not tissue_prim.IsValid()
        or not tissue_prim.HasAPI("PhysxSurfaceDeformableBodyAPI")
        or any(
            not prim.IsValid()
            or prim.GetTypeName() != "OmniPhysicsVtxXformAttachment"
            for prim in attachment_prims
        )
    ):
        raise RuntimeError("Perfused tissue deformable fixture contract failed")

    carb_logging.remove_logger(logger_handle)
    if engine_errors:
        raise RuntimeError("Isaac emitted engine errors:\n" + "\n".join(engine_errors))

    result = {
        "schema": "dr.anmar.perfusion-viability-cuda-smoke.v1",
        "status": "pass",
        "representation": args.representation,
        "steps": args.steps,
        "device": args.device,
        "isaaclab_distribution_version": distribution_version("isaaclab"),
        "isaacsim_distribution_version": distribution_version("isaacsim"),
        "joint_names": joint_names,
        "joint_count": len(joint_names),
        "finite_joint_state": True,
        "engine_error_count": 0,
        "phases_checked": phases_checked,
        "registered_camera_count": len(camera_paths),
        "tissue_registration_error_m": registration_error_m,
        "flow_conservation_error_ml_s": conservation,
        "closed_loop": closed_loop,
        "rendered_sensor_evidence": rendered_sensor_evidence,
        "loaded_arm_motion": arm_motion,
        "deformable_tissue": {
            "mesh_path": surface_path,
            "surface_deformable": True,
            "fixture_paths": fixture_paths,
            "fixture_attachment_count": len(attachment_prims),
            "attachment_types": attachment_types,
            "self_collision": False,
        },
        "probe_contact_control": {
            "nominal_coupled": nominal_contact.coupled,
            "nominal_force_error_n": nominal_contact.force_error_n,
            "overload_abort": overload_contact.abort,
            "overload_retract_delta_m": (
                overload_contact.target_extension_delta_m
            ),
        },
        "payload_mass_kg": 2.537,
        "research_only": True,
        "clinical_validation": False,
        "medical_device": False,
    }
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    try:
        code = main()
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
        raise SystemExit(code)
