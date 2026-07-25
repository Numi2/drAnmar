"""Shared native Isaac qualification helpers for DrAnmar research assets.

The helpers in this module do not estimate clinical state. They only command
authored articulation targets, measure convergence, and acquire registered RTX
camera outputs with a single live sensor pipeline at a time.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any


def tensor_value(value: Any):
    """Return the native tensor from Isaac 6 proxy tensor objects."""

    return value.torch if hasattr(value, "torch") else value


def current_body_to_world_transform(stage, robot, *, body_name: str):
    """Build a Gf transform from the live XYZW Isaac articulation pose."""

    from pxr import Gf

    try:
        body_index = list(robot.body_names).index(body_name)
    except ValueError as exc:
        raise RuntimeError(
            f"Articulation body is missing for runtime attachment: {body_name}"
        ) from exc
    body_path = None
    for prim in stage.Traverse():
        if prim.GetName() == body_name and prim.GetPath().pathString.endswith(
            f"/Links/{body_name}"
        ):
            body_path = prim.GetPath().pathString
            break
    if body_path is None:
        raise RuntimeError(f"Could not resolve articulation body prim: {body_name}")
    body_positions = _as_numpy(robot.data.body_pos_w)
    body_orientations = _as_numpy(robot.data.body_quat_w)
    position = body_positions[0, body_index]
    # Isaac Lab 6 tensor poses use XYZW quaternions.
    quaternion_xyzw = body_orientations[0, body_index]
    body_to_world = Gf.Matrix4d(1.0)
    body_to_world.SetRotate(
        Gf.Quatd(
            float(quaternion_xyzw[3]),
            Gf.Vec3d(
                float(quaternion_xyzw[0]),
                float(quaternion_xyzw[1]),
                float(quaternion_xyzw[2]),
            ),
        )
    )
    body_to_world.SetTranslateOnly(
        Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
    )
    return body_to_world


def current_child_to_world_transform(
    stage,
    robot,
    *,
    body_name: str,
    child_path: str,
):
    """Resolve a child prim transform from live PhysX articulation state.

    Dynamic body transforms live in Fabric while simulation is running, so a
    USD BBoxCache alone only sees the authored rest pose.  This composes the
    child's authored body-relative transform with the current tensor pose.
    """

    from pxr import Usd, UsdGeom

    body_path = None
    for prim in stage.Traverse():
        if prim.GetName() == body_name and prim.GetPath().pathString.endswith(
            f"/Links/{body_name}"
        ):
            body_path = prim.GetPath().pathString
            break
    if body_path is None:
        raise RuntimeError(f"Could not resolve articulation body prim: {body_name}")
    child = stage.GetPrimAtPath(child_path)
    body = stage.GetPrimAtPath(body_path)
    if not child.IsValid() or not body.IsValid():
        raise RuntimeError(
            f"Runtime attachment transform prim is missing: "
            f"body={body_path}, child={child_path}"
        )
    time_code = Usd.TimeCode.Default()
    authored_body_to_world = (
        UsdGeom.Xformable(body).ComputeLocalToWorldTransform(time_code)
    )
    authored_child_to_world = (
        UsdGeom.Xformable(child).ComputeLocalToWorldTransform(time_code)
    )
    child_to_body = authored_child_to_world * authored_body_to_world.GetInverse()
    body_to_world = current_body_to_world_transform(
        stage, robot, body_name=body_name
    )
    return child_to_body * body_to_world


def runtime_deformable_world_points(deformable_object):
    """Read current simulation-mesh nodes without copying through USD."""

    points = _as_numpy(deformable_object.data.nodal_pos_w)
    if points.ndim != 3 or points.shape[0] != 1 or points.shape[-1] != 3:
        raise RuntimeError(
            f"Unexpected deformable nodal-position shape: {points.shape}"
        )
    return points[0].copy()


def fix_standalone_mount_to_world(
    stage,
    *,
    tool_path: str,
    mount_relative_path: str = "Links/Mount",
) -> str:
    """Root a grouping-style standalone asset at its real rigid mount."""

    from omni.physx.scripts import utils as physx_utils
    from pxr import UsdPhysics

    mount_path = f"{tool_path.rstrip('/')}/{mount_relative_path.lstrip('/')}"
    mount = stage.GetPrimAtPath(mount_path)
    if not mount.IsValid():
        raise RuntimeError(f"Standalone rigid mount is missing: {mount_path}")
    if not mount.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"Standalone mount is not a rigid body: {mount_path}")

    tool_prim = stage.GetPrimAtPath(tool_path)
    source_api = UsdPhysics.ArticulationRootAPI(tool_prim)
    if not source_api:
        raise RuntimeError(
            f"Standalone tool lacks ArticulationRootAPI: {tool_path}"
        )

    def copy_articulation_properties(source_prim, target_prim) -> None:
        api = UsdPhysics.ArticulationRootAPI(source_prim)
        UsdPhysics.ArticulationRootAPI.Apply(target_prim)
        if "PhysxArticulationAPI" not in target_prim.GetAppliedSchemas():
            target_prim.AddAppliedSchema("PhysxArticulationAPI")
        for attribute_name in api.GetSchemaAttributeNames():
            source_attribute = source_prim.GetAttribute(attribute_name)
            target_attribute = target_prim.GetAttribute(attribute_name)
            if not target_attribute:
                target_attribute = target_prim.CreateAttribute(
                    attribute_name, source_attribute.GetTypeName()
                )
            target_attribute.Set(source_attribute.Get())
        for source_attribute in source_prim.GetAttributes():
            attribute_name = source_attribute.GetName()
            if not attribute_name.startswith("physxArticulation:"):
                continue
            target_attribute = target_prim.GetAttribute(attribute_name)
            if not target_attribute:
                target_attribute = target_prim.CreateAttribute(
                    attribute_name, source_attribute.GetTypeName()
                )
            target_attribute.Set(source_attribute.Get())

    # First identify the actual rigid root link. The assets deliberately keep
    # their data under a non-rigid grouping prim, which Isaac Lab's built-in
    # fix_root_link path cannot resolve.
    copy_articulation_properties(tool_prim, mount)
    tool_prim.RemoveAppliedSchema("PhysxArticulationAPI")
    tool_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)

    # Use the same joint authoring implementation as Isaac Lab itself.
    fixed_joint_prim = physx_utils.createJoint(
        stage=stage,
        joint_type="Fixed",
        from_prim=None,
        to_prim=mount,
    )
    if fixed_joint_prim is None or not fixed_joint_prim.IsValid():
        raise RuntimeError(f"Could not fix standalone mount: {mount_path}")

    # Match Isaac Lab's fixed-base parser workaround: the API lives on the
    # rigid root's parent while the joint connects that root to world.
    root_parent = mount.GetParent()
    copy_articulation_properties(mount, root_parent)
    mount.RemoveAppliedSchema("PhysxArticulationAPI")
    mount.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    if not UsdPhysics.ArticulationRootAPI(root_parent):
        raise RuntimeError(
            f"Qualification articulation root was not moved to {root_parent.GetPath()}"
        )
    return str(fixed_joint_prim.GetPath())


def spawn_fixed_standalone_articulation(
    stage,
    *,
    tool_cfg,
):
    """Spawn a grouping-root USD as a correctly fixed Isaac articulation."""

    from isaaclab.assets import Articulation

    if tool_cfg.spawn is None:
        raise ValueError("Standalone tool configuration has no spawn settings")
    tool_path = tool_cfg.prim_path
    robot = Articulation(tool_cfg)
    fixed_joint_path = fix_standalone_mount_to_world(
        stage, tool_path=tool_path
    )
    return robot, tool_path, fixed_joint_path


def attach_registered_camera_prims(
    stage,
    *,
    tool_path: str,
    frame_path: Callable[[str, str], str],
    camera_names: Iterable[str],
) -> dict[str, str]:
    """Author uncalibrated USD cameras at registered interaction frames."""

    from pxr import Gf, Sdf, UsdGeom

    created: dict[str, str] = {}
    for name in camera_names:
        parent_path = frame_path(tool_path, name)
        parent = stage.GetPrimAtPath(parent_path)
        if not parent.IsValid():
            raise RuntimeError(f"Registered camera frame is missing: {parent_path}")
        camera_path = f"{parent_path}/Camera"
        camera = UsdGeom.Camera.Define(stage, camera_path)
        xform = UsdGeom.Xformable(camera)
        xform.ClearXformOpOrder()
        # DrAnmar sensor frames use local +Z as the tissue-facing optical axis.
        # USD cameras observe along local -Z.
        xform.AddOrientOp().Set(Gf.Quatf(0.0, 0.0, 1.0, 0.0))
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.005, 2.0))
        prim = camera.GetPrim()
        prim.CreateAttribute("drAnmar:modality", Sdf.ValueTypeNames.String).Set(name)
        prim.CreateAttribute(
            "drAnmar:calibrationStatus", Sdf.ValueTypeNames.String
        ).Set("uncalibrated_research_camera")
        created[name] = camera_path
    return created


def _as_numpy(value):
    import numpy as np

    value = tensor_value(value)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _create_camera_sensor(camera_path: str, *, width: int, height: int):
    import omni.usd
    import isaaclab.sim as sim_utils
    from isaaclab.sensors.camera import Camera, CameraCfg
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    camera_prim = UsdGeom.Camera.Get(stage, camera_path)
    if not camera_prim:
        raise RuntimeError(f"USD camera is missing: {camera_path}")
    clipping = camera_prim.GetClippingRangeAttr().Get()
    return Camera(
        CameraCfg(
            prim_path=camera_path,
            update_period=0.0,
            height=height,
            width=width,
            data_types=["rgb", "distance_to_camera"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=float(camera_prim.GetFocalLengthAttr().Get()),
                focus_distance=float(
                    camera_prim.GetFocusDistanceAttr().Get() or 0.4
                ),
                horizontal_aperture=float(
                    camera_prim.GetHorizontalApertureAttr().Get()
                ),
                clipping_range=(float(clipping[0]), float(clipping[1])),
            ),
        )
    )


def capture_registered_camera_frames(
    sim,
    camera_paths: Mapping[str, str],
    *,
    width: int,
    height: int,
    render_steps_per_camera: int = 4,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Capture RGB and depth serially with at most one live RTX camera."""

    import numpy as np

    if width < 32 or height < 32:
        raise ValueError("camera dimensions must be at least 32 pixels")
    if render_steps_per_camera < 1:
        raise ValueError("render_steps_per_camera must be positive")
    frames: dict[str, dict[str, Any]] = {}
    evidence: dict[str, Any] = {}
    acquisition_order: list[str] = []
    simulation_time_s = 0.0

    for name, camera_path in camera_paths.items():
        started = time.perf_counter()
        sensor = _create_camera_sensor(camera_path, width=width, height=height)
        try:
            if not sensor.is_initialized:
                sensor._initialize_callback(None)
            for _ in range(render_steps_per_camera):
                sim.step(render=True)
                sensor.update(sim.get_physics_dt(), force_recompute=True)
                simulation_time_s += sim.get_physics_dt()
            rgb = _as_numpy(sensor.data.output["rgb"])[0].copy()
            depth = _as_numpy(sensor.data.output["distance_to_camera"])[0].copy()
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            frames[name] = {"rgb": rgb, "depth": depth}
        finally:
            try:
                sensor.__del__()
            finally:
                for attribute in ("_renderer", "_render_data"):
                    if hasattr(sensor, attribute):
                        setattr(sensor, attribute, None)
                del sensor
                gc.collect()

        rgb_values = frames[name]["rgb"][..., :3].astype(np.float32)
        depth_values = frames[name]["depth"]
        if rgb_values.shape[:2] != (height, width):
            raise RuntimeError(
                f"Rendered camera {name} has RGB shape {rgb_values.shape}"
            )
        if depth_values.shape != (height, width):
            raise RuntimeError(
                f"Rendered camera {name} has depth shape {depth_values.shape}"
            )
        finite_depth = depth_values[np.isfinite(depth_values)]
        rgb_variation = float(np.std(rgb_values))
        if not np.isfinite(rgb_values).all() or rgb_variation <= 0.25:
            raise RuntimeError(
                f"Rendered camera {name} lacks finite scene variation: "
                f"rgb_std={rgb_variation}"
            )
        if finite_depth.size == 0:
            raise RuntimeError(f"Rendered camera {name} has no finite depth")
        acquisition_order.append(name)
        evidence[name] = {
            "rgb_shape": list(frames[name]["rgb"].shape),
            "rgb_mean": float(np.mean(rgb_values)),
            "rgb_standard_deviation": rgb_variation,
            "depth_shape": list(depth_values.shape),
            "depth_finite_fraction": float(
                finite_depth.size / depth_values.size
            ),
            "depth_minimum_m": float(np.min(finite_depth)),
            "depth_maximum_m": float(np.max(finite_depth)),
            "simulation_timestamp_s": simulation_time_s,
            "wall_latency_ms": (time.perf_counter() - started) * 1000.0,
        }

    evidence["acquisition_policy"] = {
        "mode": "serialized_one_camera_at_a_time",
        "maximum_concurrent_camera_pipelines": 1,
        "render_steps_per_camera": render_steps_per_camera,
        "order": acquisition_order,
        "simulation_span_s": simulation_time_s,
        "fusion_requirement": (
            "buffer or interpolate timestamped frames to a common fusion time"
        ),
        "calibration_status": "uncalibrated_research_camera",
    }
    return frames, evidence


def command_phase_targets(
    robot,
    sim,
    *,
    joint_names: list[str],
    targets: Mapping[str, float],
    velocity_targets: Mapping[str, float] | None = None,
    steps: int,
    render: bool,
    maximum_position_error: float = 0.004,
    maximum_velocity_error: float = 5.0,
    phase_name: str | None = None,
) -> dict[str, Any]:
    """Command one authored phase and measure articulation convergence.

    Position and velocity targets are kept separate so a rotary cutter is
    never accidentally treated as a position servo.  This mirrors Isaac Lab's
    explicit actuator-target APIs and makes the qualification evidence state
    which control mode was exercised for every joint.
    """

    import numpy as np

    if steps < 1:
        raise ValueError("steps must be positive")
    velocity_targets = dict(velocity_targets or {})
    command = tensor_value(robot.data.default_joint_pos).clone()
    velocity_command = tensor_value(robot.data.default_joint_vel).clone()
    duplicate = sorted(set(targets).intersection(velocity_targets))
    if duplicate:
        raise RuntimeError(
            f"Phase targets command joints in two control modes: {duplicate}"
        )
    missing = sorted(
        (set(targets) | set(velocity_targets)).difference(joint_names)
    )
    if missing:
        raise RuntimeError(f"Phase target references missing joints: {missing}")
    commanded_indices = []
    for name, value in targets.items():
        index = joint_names.index(name)
        command[:, index] = float(value)
        commanded_indices.append(index)
    commanded_velocity_indices = []
    for name, value in velocity_targets.items():
        index = joint_names.index(name)
        velocity_command[:, index] = float(value)
        commanded_velocity_indices.append(index)
    for _ in range(steps):
        robot.set_joint_position_target(command)
        if velocity_targets:
            robot.set_joint_velocity_target(velocity_command)
        robot.write_data_to_sim()
        sim.step(render=render)
        robot.update(sim.get_physics_dt())
    actual = tensor_value(robot.data.joint_pos).detach().cpu().numpy()
    actual_velocity = tensor_value(robot.data.joint_vel).detach().cpu().numpy()
    desired = command.detach().cpu().numpy()
    desired_velocity = velocity_command.detach().cpu().numpy()
    error = np.abs(actual[:, commanded_indices] - desired[:, commanded_indices])
    maximum_error = float(np.max(error)) if error.size else 0.0
    per_joint = {
        name: {
            "commanded": float(desired[0, joint_names.index(name)]),
            "actual": float(actual[0, joint_names.index(name)]),
            "absolute_error": float(
                abs(
                    actual[0, joint_names.index(name)]
                    - desired[0, joint_names.index(name)]
                )
            ),
        }
        for name in targets
    }
    velocity_error = np.abs(
        actual_velocity[:, commanded_velocity_indices]
        - desired_velocity[:, commanded_velocity_indices]
    )
    maximum_measured_velocity_error = (
        float(np.max(velocity_error)) if velocity_error.size else 0.0
    )
    per_velocity_joint = {
        name: {
            "commanded_rad_s": float(
                desired_velocity[0, joint_names.index(name)]
            ),
            "actual_rad_s": float(
                actual_velocity[0, joint_names.index(name)]
            ),
            "absolute_error_rad_s": float(
                abs(
                    actual_velocity[0, joint_names.index(name)]
                    - desired_velocity[0, joint_names.index(name)]
                )
            ),
        }
        for name in velocity_targets
    }
    if not np.isfinite(actual).all() or not np.isfinite(actual_velocity).all():
        raise RuntimeError("Articulation produced a non-finite joint state")
    if (
        maximum_error > maximum_position_error
        or maximum_measured_velocity_error > maximum_velocity_error
    ):
        body_positions = _as_numpy(robot.data.body_pos_w)
        body_orientations = _as_numpy(robot.data.body_quat_w)
        body_evidence = {
            name: {
                "position": [
                    float(value) for value in body_positions[0, index]
                ],
                "orientation_wxyz": [
                    float(value) for value in body_orientations[0, index]
                ],
            }
            for index, name in enumerate(robot.body_names)
        }
        raise RuntimeError(
            "Articulation did not converge to the authored phase targets: "
            f"phase={phase_name!r}, maximum_error={maximum_error}, "
            f"position_limit={maximum_position_error}, "
            f"maximum_velocity_error={maximum_measured_velocity_error}, "
            f"velocity_limit={maximum_velocity_error}, "
            f"position_joints={per_joint}, velocity_joints={per_velocity_joint}, "
            f"body_positions_world={body_evidence}"
        )
    return {
        "phase": phase_name,
        "steps": steps,
        "maximum_position_error": maximum_error,
        "maximum_velocity_error_rad_s": maximum_measured_velocity_error,
        "commanded_joint_count": len(commanded_indices),
        "commanded_velocity_joint_count": len(commanded_velocity_indices),
        "joint_evidence": per_joint,
        "velocity_joint_evidence": per_velocity_joint,
    }
