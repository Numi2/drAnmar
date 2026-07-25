#!/usr/bin/env python3
"""Headless CUDA qualification for the DrAnmar seal-and-divide system."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / (
    "source/extensions/orbit.surgical.assets/orbit/surgical/assets/"
    "adaptive_seal_divide_robot.py"
)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--representation", choices=("standalone", "franka"), required=True)
parser.add_argument("--steps", type=int, default=120)
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
from pxr import Gf, Usd, UsdGeom
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, DeformableObject, DeformableObjectCfg
from isaacsim.core.simulation_manager import PhysxGpuCfg, PhysxScene
from dranmar_native_qualification import (
    attach_registered_camera_prims,
    capture_registered_camera_frames,
    command_phase_targets,
    current_body_to_world_transform,
    current_child_to_world_transform,
    runtime_deformable_world_points,
    spawn_fixed_standalone_articulation,
)


def load_helper():
    spec = importlib.util.spec_from_file_location(
        "dranmar_seal_divide_runtime", HELPER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def tensor_value(value):
    return value.torch if hasattr(value, "torch") else value


def translated_local(matrix, x):
    result = Gf.Matrix4d(matrix)
    result.SetTranslateOnly(matrix.Transform(Gf.Vec3d(x, 0.0, 0.0)))
    return result


def assert_current_attachments(stage, paths):
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or prim.GetTypeName() != "OmniPhysicsVtxXformAttachment":
            raise RuntimeError(f"Invalid current-schema attachment: {path}")
        for relationship in ("omniphysics:src0", "omniphysics:src1"):
            if not prim.GetRelationship(relationship).GetTargets():
                raise RuntimeError(f"Attachment relationship missing: {path}")


def deformable_bounds_evidence(vessel_objects, phase):
    evidence = {}
    for side, deformable_object in vessel_objects.items():
        deformable_object.update(0.0)
        points = runtime_deformable_world_points(deformable_object)
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        span = maximum - minimum
        evidence[side] = {
            "minimum_world_m": minimum.tolist(),
            "maximum_world_m": maximum.tolist(),
            "span_m": span.tolist(),
        }
        if (
            not np.isfinite(points).all()
            or float(np.max(span)) > 0.20
            or float(np.max(np.abs(points))) > 1.0
        ):
            raise RuntimeError(
                f"Vessel deformable became unstable during {phase}: "
                f"{evidence}"
            )
    return evidence


def main() -> int:
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    engine_errors = []
    carb_logging = carb.logging.acquire_logging()

    def record_error(source, level, _filename, _line, message):
        if level >= carb.logging.LEVEL_ERROR and len(engine_errors) < 30:
            engine_errors.append(f"{source}: {message.strip()}")

    logger_handle = carb_logging.add_logger(record_error)
    helper = load_helper()
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device)
    )
    physx_scene = PhysxScene(sim.cfg.physics_prim_path)
    physx_scene.set_gpu_configuration(
        PhysxGpuCfg(gpu_max_deformable_surface_contacts=2**22)
    )
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/GroundPlane", ground)
    light = sim_utils.DomeLightCfg(intensity=2500.0)
    light.func("/World/Light", light)

    stage = omni.usd.get_context().get_stage()
    qualification_mount = None
    if args.representation == "standalone":
        root_path = "/World/AdaptiveSealDivideTool"
        robot, tool_path, qualification_mount = (
            spawn_fixed_standalone_articulation(
                stage,
                tool_cfg=helper.make_tool_cfg(
                    root_path,
                    position=(0.0, 0.0, 0.10),
                ),
            )
        )
    else:
        root_path = "/World/Robot"
        robot = Articulation(
            helper.make_franka_adaptive_seal_divide_robot_cfg(prim_path=root_path)
        )
        tool_path = f"{root_path}/DrAnmarAdaptiveSealDivideTool"

    tcp_prim = stage.GetPrimAtPath(helper.frame_path(tool_path, "seal_divide_tcp"))
    if not tcp_prim.IsValid():
        raise RuntimeError("Seal-and-divide TCP is missing")
    tcp_world = UsdGeom.Xformable(tcp_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    tcp_translation = tuple(float(value) for value in tcp_world.ExtractTranslation())

    vessel_root = "/World/DrAnmarSealDivideVessel"
    vessel_translation = (
        tcp_translation[0],
        tcp_translation[1],
        tcp_translation[2],
    )
    helper.spawn_vessel_demo(vessel_root, translation=vessel_translation)
    deformables = helper.apply_vessel_surface_deformables(
        vessel_root, self_collision=False, stage=stage
    )
    vessel_objects = {
        side: DeformableObject(
            DeformableObjectCfg(
                prim_path=f"{vessel_root}/{side}VesselWall",
                spawn=None,
            )
        )
        for side in ("Left", "Right")
    }
    fixture_attachments = helper.anchor_vessel_distal_ends(
        vessel_root, stage=stage
    )
    bridge = helper.BridgeAttachmentController(vessel_root)
    bridge_cells = bridge.engage(stage=stage)
    bridge_attachments = [
        path for cell in bridge_cells for path in cell.attachment_paths
    ]
    if len(bridge_attachments) != 16:
        raise RuntimeError(f"Expected 16 bridge attachments: {bridge_attachments}")
    camera_paths = attach_registered_camera_prims(
        stage,
        tool_path=tool_path,
        frame_path=helper.frame_path,
        camera_names=helper.REGISTERED_CAMERA_FRAMES,
    )
    sim.reset()
    robot.update(sim.get_physics_dt())
    deformable_stability = {
        "reset": deformable_bounds_evidence(vessel_objects, "reset")
    }
    joint_names = list(robot.joint_names)
    expected_joint_count = 8 if args.representation == "standalone" else 15
    if len(joint_names) != expected_joint_count:
        raise RuntimeError(
            f"Expected {expected_joint_count} joints, got {len(joint_names)}: "
            f"{joint_names}"
        )
    missing = sorted(set(helper.TOOL_JOINTS.values()) - set(joint_names))
    if missing:
        raise RuntimeError(f"Missing tool joints: {missing}")
    if args.representation == "franka":
        missing_arm = sorted(
            {f"panda_joint{i}" for i in range(1, 8)} - set(joint_names)
        )
        if missing_arm:
            raise RuntimeError(f"Missing Franka joints: {missing_arm}")

    # The longest authored stroke is 41 mm at 0.25 m/s.  Thirty 120 Hz
    # simulation steps allow the mechanism to honor its velocity limits and
    # settle instead of treating a commanded target as achieved immediately.
    steps_per_phase = max(30, args.steps // len(helper.PHASE_TARGETS))
    phase_convergence = {}
    for phase in ("inspect", "center", "compress"):
        phase_convergence[phase] = command_phase_targets(
            robot,
            sim,
            joint_names=joint_names,
            targets=helper.phase_targets(phase),
            steps=steps_per_phase,
            render=not args.headless,
            phase_name=phase,
        )
        deformable_stability[phase] = deformable_bounds_evidence(
            vessel_objects, phase
        )
    # Compression attachment creation is intentionally delayed until the
    # articulated jaw volumes have reached the authored contact phase.
    compression = helper.DualZoneCompressionController(tool_path, vessel_root)
    left_points = runtime_deformable_world_points(vessel_objects["Left"])
    right_points = runtime_deformable_world_points(vessel_objects["Right"])
    runtime_compression_geometry = {}
    for side, points, jaw, contact in (
        ("left_upper", left_points, "UpperJaw", "LeftSealContact"),
        ("left_lower", left_points, "LowerJaw", "LeftSealContact"),
        ("right_upper", right_points, "UpperJaw", "RightSealContact"),
        ("right_lower", right_points, "LowerJaw", "RightSealContact"),
    ):
        runtime_compression_geometry[side] = {
            "deformable_points_world": points,
            "target_to_world": current_child_to_world_transform(
                stage,
                robot,
                body_name=jaw,
                child_path=(
                    f"{tool_path}/Links/{jaw}/Collisions/{contact}"
                ),
            ),
            "attachment_frame_path": f"{tool_path}/Links/{jaw}",
            "attachment_frame_to_world": current_body_to_world_transform(
                stage,
                robot,
                body_name=jaw,
            ),
        }
    compression_attachments = compression.engage(
        stage=stage,
        runtime_geometry=runtime_compression_geometry,
    )
    if len(compression_attachments) != 4:
        raise RuntimeError(
            f"Expected four overlapping compression contacts: "
            f"{compression_attachments}"
        )

    sequence = helper.AdaptiveSealDivideSequenceController()
    sequence.transition("seal")
    seal_targets = helper.phase_targets("seal")
    seal_command = tensor_value(robot.data.default_joint_pos).clone()
    for name, value in seal_targets.items():
        seal_command[:, joint_names.index(name)] = value
    upper_index = joint_names.index("upper_jaw_joint")
    lower_index = joint_names.index("lower_jaw_joint")
    measured_force_samples = []
    seal_hold_steps = max(60, args.steps // 2)
    for _ in range(seal_hold_steps):
        robot.set_joint_position_target(seal_command)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim.get_physics_dt())
        applied = tensor_value(robot.data.applied_torque).detach().cpu().numpy()
        upper_effort_n = float(np.max(np.abs(applied[:, upper_index])))
        lower_effort_n = float(np.max(np.abs(applied[:, lower_index])))
        measured_force_samples.append(
            (
                upper_effort_n,
                lower_effort_n,
            )
        )
    stable_force_samples = np.asarray(
        measured_force_samples[-min(30, len(measured_force_samples)):],
        dtype=float,
    )
    upper_force_n = float(np.median(stable_force_samples[:, 0]))
    lower_force_n = float(np.median(stable_force_samples[:, 1]))
    upper_effort_n = upper_force_n
    lower_effort_n = lower_force_n
    compression_status = compression.update_force(
        upper_force_n, lower_force_n, stage=stage
    )
    if compression_status["mode"] != "controlled":
        # The current surface-shell vessel does not resolve through-thickness
        # wall compression.  Do not lower the authored force interlock or feed
        # a synthetic force merely to complete the sequence.  Qualify the
        # mechanism's physically measured negative path: division must remain
        # blocked and all bridge attachments must remain intact.
        tissue_centered = (
            phase_convergence["center"]["maximum_position_error"] <= 0.004
            and len(compression_attachments) == 4
        )
        sequence = helper.AdaptiveSealDivideSequenceController()
        blocked_division = helper.TissueDivisionController(bridge).advance(
            1.0,
            energy=sequence.energy,
            leak=sequence.leak,
            upper_force_n=upper_force_n,
            lower_force_n=lower_force_n,
            guard_retracted=True,
            tissue_centered=tissue_centered,
            stage=stage,
        )
        if (
            blocked_division["authorized"]
            or "insufficient_compression" not in blocked_division["reasons"]
            or blocked_division["blade_progress"] != 0.0
            or bridge.released_fraction != 0.0
        ):
            raise RuntimeError(
                "Seal force interlock did not fail closed under the measured "
                f"surface-shell load: {blocked_division}"
            )
        phase_convergence["abort"] = command_phase_targets(
            robot,
            sim,
            joint_names=joint_names,
            targets=helper.phase_targets("abort"),
            steps=steps_per_phase,
            render=not args.headless,
            phase_name="abort",
        )
        compression.release(stage=stage)
        _, rendered_sensor_evidence = capture_registered_camera_frames(
            sim,
            camera_paths,
            width=args.render_width,
            height=args.render_height,
        )
        joint_pos = tensor_value(robot.data.joint_pos).detach().cpu().numpy()
        if not np.isfinite(joint_pos).all():
            raise RuntimeError("Non-finite articulation state in abort path")
        required = {
            "OmniPhysicsDeformableBodyAPI",
            "OmniPhysicsSurfaceDeformableSimAPI",
            "PhysxSurfaceDeformableBodyAPI",
        }
        deformable_schemas = {}
        for mesh_path in deformables["mesh_paths"]:
            applied = list(stage.GetPrimAtPath(mesh_path).GetAppliedSchemas())
            if required - set(applied):
                raise RuntimeError(
                    f"{mesh_path} missing schemas: "
                    f"{sorted(required-set(applied))}"
                )
            deformable_schemas[mesh_path.rsplit("/", 1)[-1]] = applied
        carb_logging.remove_logger(logger_handle)
        if engine_errors:
            raise RuntimeError(
                "Isaac emitted engine errors:\n" + "\n".join(engine_errors)
            )
        result = {
            "schema": "dr.anmar.adaptive-seal-divide-runtime-diagnostic.v2",
            "status": "fail_closed_as_designed",
            "qualification_scope": (
                "native_mechanism_deformable_stability_and_fail_closed_interlock"
            ),
            "representation": args.representation,
            "steps": args.steps,
            "device": args.device,
            "isaaclab_distribution_version": distribution_version("isaaclab"),
            "isaacsim_distribution_version": distribution_version("isaacsim"),
            "phase_convergence": phase_convergence,
            "registered_camera_count": len(camera_paths),
            "rendered_sensor_evidence": rendered_sensor_evidence,
            "deformable_applied_schemas": deformable_schemas,
            "deformable_stability": deformable_stability,
            "fixture_attachment_count": len(fixture_attachments),
            "bridge_attachment_count_created": len(bridge_attachments),
            "bridge_attachment_count_released": 0,
            "compression_attachment_count_created": len(
                compression_attachments
            ),
            "compression_released_on_abort": True,
            "measured_upper_jaw_generalized_effort_n": upper_effort_n,
            "measured_lower_jaw_generalized_effort_n": lower_effort_n,
            "compression_status": compression_status,
            "blade_interlock": blocked_division,
            "division_executed": False,
            "intended_seal_and_divide_efficacy_qualified": False,
            "efficacy_blocker": (
                "surface_shell_does_not_resolve_through_thickness_vessel_"
                "compression_and_no_calibrated_volumetric_material_is_present"
            ),
            "clinical_validation": False,
            "medical_device": False,
        }
        payload = json.dumps(result, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0
    # The lumped thermal controller is independent of the PhysX integrator.
    # Advance it from the measured steady contact forces without wasting
    # thousands of identical GPU physics steps.
    for _ in range(6000):
        energy_report = sequence.update_energy(
            0.01, upper_force_n, lower_force_n
        )
        if (
            energy_report["both_ready"]
            and energy_report["flows"]["left_ml_min"] < 0.1
            and energy_report["flows"]["right_ml_min"] < 0.1
        ):
            break
    else:
        raise RuntimeError(
            "Measured-force seal model did not qualify within its bounded "
            f"integration window: forces_n={(upper_force_n, lower_force_n)}, "
            f"report={energy_report}"
        )
    actual_seal = tensor_value(robot.data.joint_pos).detach().cpu().numpy()
    seal_error = float(
        np.max(
            np.abs(
                actual_seal[:, [upper_index, lower_index]]
                - seal_command.detach().cpu().numpy()[
                    :, [upper_index, lower_index]
                ]
            )
        )
    )
    if seal_error > 0.004:
        raise RuntimeError(f"Seal jaw targets did not converge: {seal_error}")
    phase_convergence["seal"] = {
        "steps": len(measured_force_samples),
        "maximum_position_error": seal_error,
        "commanded_joint_count": len(seal_targets),
    }

    bands = helper.TissueSealBandController()
    left_bond = bands.deploy(
        "/World/LeftRetainedSealBand",
        translated_local(tcp_world, -0.0065),
        f"{vessel_root}/LeftVesselWall",
        stage=stage,
    )
    right_bond = bands.deploy(
        "/World/RightRetainedSealBand",
        translated_local(tcp_world, 0.0065),
        f"{vessel_root}/RightVesselWall",
        stage=stage,
    )
    bands.update_maturity(left_bond, sequence.energy.left.maturity, stage=stage)
    bands.update_maturity(right_bond, sequence.energy.right.maturity, stage=stage)
    if bands.apply_load(left_bond, bands.break_force_n(left_bond), stage=stage):
        raise RuntimeError("Left seal band failed at its threshold")
    if bands.apply_load(right_bond, bands.break_force_n(right_bond), stage=stage):
        raise RuntimeError("Right seal band failed at its threshold")
    seal_band_attachments = left_bond.attachment_paths + right_bond.attachment_paths

    sequence.transition("verify_seal")
    phase_convergence["verify_seal"] = command_phase_targets(
        robot,
        sim,
        joint_names=joint_names,
        targets=helper.phase_targets("verify_seal"),
        steps=steps_per_phase,
        render=not args.headless,
        phase_name="verify_seal",
    )
    tissue_centered = (
        phase_convergence["center"]["maximum_position_error"] <= 0.004
        and len(compression_attachments) == 4
    )
    pre_guard = helper.BladeInterlockController().evaluate(
        sequence.energy,
        sequence.leak,
        upper_force_n,
        lower_force_n,
        False,
        tissue_centered=tissue_centered,
    )
    if pre_guard["authorized"] or "blade_guard_not_retracted" not in pre_guard["reasons"]:
        raise RuntimeError(f"Blade interlock failed closed-state proof: {pre_guard}")
    off_center_guard = helper.BladeInterlockController().evaluate(
        sequence.energy,
        sequence.leak,
        upper_force_n,
        lower_force_n,
        True,
        tissue_centered=False,
    )
    if (
        off_center_guard["authorized"]
        or "tissue_not_centered" not in off_center_guard["reasons"]
    ):
        raise RuntimeError(
            f"Blade interlock did not fail closed for off-center tissue: "
            f"{off_center_guard}"
        )
    sequence.transition("retract_guard")
    phase_convergence["retract_guard"] = command_phase_targets(
        robot,
        sim,
        joint_names=joint_names,
        targets=helper.phase_targets("retract_guard"),
        steps=steps_per_phase,
        render=not args.headless,
        phase_name="retract_guard",
    )
    sequence.transition("divide")
    phase_convergence["divide"] = command_phase_targets(
        robot,
        sim,
        joint_names=joint_names,
        targets=helper.phase_targets("divide"),
        steps=steps_per_phase,
        render=not args.headless,
        phase_name="divide",
    )
    blade_position = float(
        tensor_value(robot.data.joint_pos)
        .detach()
        .cpu()
        .numpy()[0, joint_names.index("blade_joint")]
    )
    blade_target = helper.phase_targets("divide")["blade_joint"]
    blade_progress = max(0.0, min(1.0, blade_position / blade_target))
    division = helper.TissueDivisionController(bridge)
    division_report = division.advance(
        blade_progress,
        energy=sequence.energy,
        leak=sequence.leak,
        upper_force_n=upper_force_n,
        lower_force_n=lower_force_n,
        guard_retracted=True,
        tissue_centered=tissue_centered,
        stage=stage,
    )
    if not division_report["authorized"] or not division_report["division_complete"]:
        raise RuntimeError(f"Interlocked division failed: {division_report}")
    if any(stage.GetPrimAtPath(path).IsValid() for path in bridge_attachments):
        raise RuntimeError("Bridge attachments remained after complete division")

    sequence.transition("release")
    phase_convergence["release"] = command_phase_targets(
        robot,
        sim,
        joint_names=joint_names,
        targets=helper.phase_targets("release"),
        steps=steps_per_phase,
        render=not args.headless,
        phase_name="release",
    )
    compression.release(stage=stage)
    if any(stage.GetPrimAtPath(path).IsValid() for path in compression_attachments):
        raise RuntimeError("Temporary compression attachments were not released")
    active_attachments = fixture_attachments + seal_band_attachments
    if (
        len(fixture_attachments) != 2
        or len(compression_attachments) != 4
        or len(seal_band_attachments) != 4
        or len(active_attachments) != 6
    ):
        raise RuntimeError(
            "Unexpected attachment counts: "
            f"fixture={len(fixture_attachments)} "
            f"compression={len(compression_attachments)} "
            f"seal_bands={len(seal_band_attachments)} "
            f"active={len(active_attachments)}"
    )
    assert_current_attachments(stage, active_attachments)
    for phase in ("verify_stumps", "complete", "abort"):
        sequence.transition(phase)
        phase_convergence[phase] = command_phase_targets(
            robot,
            sim,
            joint_names=joint_names,
            targets=helper.phase_targets(phase),
            steps=steps_per_phase,
            render=not args.headless,
            phase_name=phase,
        )
    _, rendered_sensor_evidence = capture_registered_camera_frames(
        sim,
        camera_paths,
        width=args.render_width,
        height=args.render_height,
    )
    joint_pos = tensor_value(robot.data.joint_pos).detach().cpu().numpy()
    if not np.isfinite(joint_pos).all():
        bad = np.argwhere(~np.isfinite(joint_pos))
        raise RuntimeError(f"Non-finite articulation state: {bad.tolist()}")

    required = {
        "OmniPhysicsDeformableBodyAPI",
        "OmniPhysicsSurfaceDeformableSimAPI",
        "PhysxSurfaceDeformableBodyAPI",
    }
    deformable_schemas = {}
    for mesh_path in deformables["mesh_paths"]:
        applied = list(stage.GetPrimAtPath(mesh_path).GetAppliedSchemas())
        if required - set(applied):
            raise RuntimeError(
                f"{mesh_path} missing schemas: {sorted(required-set(applied))}"
            )
        deformable_schemas[mesh_path.rsplit("/", 1)[-1]] = applied

    carb_logging.remove_logger(logger_handle)
    if engine_errors:
        raise RuntimeError("Isaac emitted engine errors:\n" + "\n".join(engine_errors))

    result = {
        "schema": "dr.anmar.adaptive-seal-divide-runtime-diagnostic.v2",
        "status": "controller_exercise_only",
        "representation": args.representation,
        "steps": args.steps,
        "device": args.device,
        "gpu_max_deformable_surface_contacts": (
            physx_scene.get_gpu_configuration().gpu_max_deformable_surface_contacts
        ),
        "isaaclab_distribution_version": distribution_version("isaaclab"),
        "isaacsim_distribution_version": distribution_version("isaacsim"),
        "joint_names": joint_names,
        "joint_count": len(joint_names),
        "finite_joint_state": True,
        "engine_error_count": 0,
        "standalone_world_mount_joint": qualification_mount,
        "phase_convergence": phase_convergence,
        "registered_camera_count": len(camera_paths),
        "rendered_sensor_evidence": rendered_sensor_evidence,
        "deformable_applied_schemas": deformable_schemas,
        "deformable_stability": deformable_stability,
        "self_collision": deformables["self_collision"],
        "fixture_attachment_count": len(fixture_attachments),
        "bridge_attachment_count_created": len(bridge_attachments),
        "bridge_attachment_count_released": len(bridge_attachments),
        "compression_attachment_count_created": len(compression_attachments),
        "compression_released": True,
        "seal_band_attachment_count": len(seal_band_attachments),
        "active_attachment_count": len(active_attachments),
        "attachment_type": "OmniPhysicsVtxXformAttachment",
        "tissue_centered_from_joint_and_overlap_evidence": tissue_centered,
        "measured_upper_jaw_force_n": upper_force_n,
        "measured_lower_jaw_force_n": lower_force_n,
        "left_seal": {
            "temperature_c": sequence.energy.left.temperature_c,
            "energy_j": sequence.energy.left.energy_j,
            "maturity": sequence.energy.left.maturity,
            "impedance_ohm": sequence.energy.left.impedance_ohm,
        },
        "right_seal": {
            "temperature_c": sequence.energy.right.temperature_c,
            "energy_j": sequence.energy.right.energy_j,
            "maturity": sequence.energy.right.maturity,
            "impedance_ohm": sequence.energy.right.impedance_ohm,
        },
        "predicted_stump_flows_ml_min": sequence.leak.flows(),
        "blade_interlock": division_report,
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
