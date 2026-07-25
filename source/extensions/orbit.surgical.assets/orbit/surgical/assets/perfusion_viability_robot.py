# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Isaac integration and physiological models for the DrAnmar perfusion robot.

A shared graph-based vascular state drives all synthetic modalities.  The
implementation is manufacturer-neutral and intended for simulation training. Values are
provisional and must not be interpreted as clinical thresholds or patient-care
settings.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CATALOG_SUBPATH = "Props/SurgicalAssessment/PerfusionViabilityRobot"
ASSET_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
ASSET_ROOT = ASSET_DATA_ROOT / CATALOG_SUBPATH
TOOL_PAYLOAD_USD = ASSET_ROOT / "dranmar_perfusion_viability_tool_payload.usda"
TOOL_STANDALONE_USD = ASSET_ROOT / "dranmar_perfusion_viability_tool_standalone.usda"
TOOL_RIGID_PROXY_USD = ASSET_ROOT / "dranmar_perfusion_viability_tool_rigid_proxy.usda"
TISSUE_DEMO_USD = ASSET_ROOT / "dranmar_perfused_tissue_demo.usda"
TRACER_PARTICLE_USD = ASSET_ROOT / "dranmar_icg_tracer_particle.usda"
COUPLING_PAD_USD = ASSET_ROOT / "dranmar_ultrasound_coupling_pad.usda"
FLOW_OCCLUDER_USD = ASSET_ROOT / "dranmar_flow_occluder.usda"
PERFUSION_GRAPH_PATH = ASSET_ROOT / "perfusion_network.json"

VALID_CONTRAST_STATES = frozenset({"full", "empty"})
VALID_GEL_STATES = frozenset({"full", "empty"})
VALID_SENSOR_STATES = frozenset({"ready", "degraded", "fault"})
VALID_SENSOR_MODALITIES = frozenset({
    "stereo_rgb",
    "nir_icg",
    "laser_speckle",
    "thermal",
    "surface_oxygenation",
    "depth",
    "doppler",
    "ultrasound",
})
VALID_CONDITIONS = frozenset({
    "healthy", "arterial_occlusion", "venous_congestion", "anastomotic_stenosis",
    "branch_leak", "retraction_ischemia", "dressing_compression", "recovered",
})
TASK_PHASES = (
    "inspect",
    "rgb",
    "icg",
    "speckle",
    "thermal",
    "oxygenation",
    "doppler",
    "ultrasound",
    "fuse",
    "diagnose",
    "intervene",
    "rescan",
    "verify",
)

TOOL_JOINTS = {
    "sensor_turret": "sensor_turret_joint",
    "filter_wheel": "filter_wheel_joint",
    "optical_focus": "optical_focus_joint",
    "speckle_scan_x": "speckle_scan_x_joint",
    "speckle_scan_y": "speckle_scan_y_joint",
    "ultrasound_extension": "ultrasound_extension_joint",
    "ultrasound_pitch": "ultrasound_pitch_joint",
    "ultrasound_compliance": "ultrasound_compliance_joint",
    "gel_valve": "gel_valve_joint",
    "doppler_extension": "doppler_extension_joint",
    "doppler_pitch": "doppler_pitch_joint",
    "contact_guard": "contact_guard_joint",
}

TOOL_FRAME_PATHS = {
    "panda_link8_mount": "Links/Mount/Frames/panda_link8_mount",
    "perfusion_tcp": "Links/Mount/Frames/perfusion_tcp",
    "roi_center": "Links/Mount/Frames/roi_center",
    "contact_guard_reference": "Links/Mount/Frames/contact_guard_reference",
    "count_reference": "Links/Mount/Frames/count_reference",
    "handover_reference": "Links/Mount/Frames/handover_reference",
    "rgb_left_camera": "Links/SensorTurret/Frames/rgb_left_camera",
    "rgb_right_camera": "Links/SensorTurret/Frames/rgb_right_camera",
    "nir_fluorescence_camera": "Links/SensorTurret/Frames/nir_fluorescence_camera",
    "speckle_camera": "Links/SensorTurret/Frames/speckle_camera",
    "thermal_camera": "Links/SensorTurret/Frames/thermal_camera",
    "multispectral_camera": "Links/SensorTurret/Frames/multispectral_camera",
    "optical_scan_reference": "Links/SensorTurret/Frames/optical_scan_reference",
    "structured_light_projector": "Links/OpticalFocus/Frames/structured_light_projector",
    "depth_reference": "Links/OpticalFocus/Frames/depth_reference",
    "speckle_projection_center": "Links/SpeckleMirrorY/Frames/speckle_projection_center",
    "ultrasound_probe_face": "Links/UltrasoundCompliance/Frames/ultrasound_probe_face",
    "ultrasound_probe_axis": "Links/UltrasoundCompliance/Frames/ultrasound_probe_axis",
    "ultrasound_force_reference": "Links/UltrasoundCompliance/Frames/ultrasound_force_reference",
    "gel_dispense_exit": "Links/UltrasoundCarriage/Frames/gel_dispense_exit",
    "doppler_probe_tip": "Links/DopplerGimbal/Frames/doppler_probe_tip",
    "doppler_beam_axis": "Links/DopplerGimbal/Frames/doppler_beam_axis",
    "contact_guard_force": "Links/ContactGuard/Frames/contact_guard_force",
}


def frame_path(tool_path: str, name: str) -> str:
    try:
        suffix = TOOL_FRAME_PATHS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown perfusion-robot frame {name!r}") from exc
    return f"{tool_path.rstrip('/')}/{suffix}"


def tensor_value(value: Any):
    """Support Isaac tensor proxy objects and native tensors."""
    return value.torch if hasattr(value, "torch") else value


def _check(value: str, allowed: frozenset[str], label: str) -> str:
    if value not in allowed:
        raise ValueError(f"Unsupported {label}={value!r}; expected one of {sorted(allowed)}")
    return value


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


@dataclass
class SensorConsumableLedger:
    """Conserve contrast and coupling gel across repeated acquisitions."""

    initial_contrast_ml: float = 6.0
    initial_gel_ml: float = 12.0
    contrast_remaining_ml: float | None = None
    gel_remaining_ml: float | None = None
    contrast_consumed_ml: float = 0.0
    gel_consumed_ml: float = 0.0

    def __post_init__(self) -> None:
        self.initial_contrast_ml = _finite(
            self.initial_contrast_ml, "initial_contrast_ml"
        )
        self.initial_gel_ml = _finite(self.initial_gel_ml, "initial_gel_ml")
        if self.initial_contrast_ml < 0.0 or self.initial_gel_ml < 0.0:
            raise ValueError("initial consumable volumes must be non-negative")
        if self.contrast_remaining_ml is None:
            self.contrast_remaining_ml = self.initial_contrast_ml
        if self.gel_remaining_ml is None:
            self.gel_remaining_ml = self.initial_gel_ml
        self.contrast_remaining_ml = _finite(
            self.contrast_remaining_ml, "contrast_remaining_ml"
        )
        self.gel_remaining_ml = _finite(
            self.gel_remaining_ml, "gel_remaining_ml"
        )
        if not 0.0 <= self.contrast_remaining_ml <= self.initial_contrast_ml:
            raise ValueError("contrast_remaining_ml is outside the ledger")
        if not 0.0 <= self.gel_remaining_ml <= self.initial_gel_ml:
            raise ValueError("gel_remaining_ml is outside the ledger")

    def consume(self, *, contrast_ml: float = 0.0, gel_ml: float = 0.0) -> dict[str, float]:
        contrast_request = _finite(contrast_ml, "contrast_ml")
        gel_request = _finite(gel_ml, "gel_ml")
        if contrast_request < 0.0 or gel_request < 0.0:
            raise ValueError("consumable requests must be non-negative")
        contrast_used = min(contrast_request, self.contrast_remaining_ml)
        gel_used = min(gel_request, self.gel_remaining_ml)
        self.contrast_remaining_ml -= contrast_used
        self.gel_remaining_ml -= gel_used
        self.contrast_consumed_ml += contrast_used
        self.gel_consumed_ml += gel_used
        return {
            "contrast_requested_ml": contrast_request,
            "contrast_used_ml": contrast_used,
            "gel_requested_ml": gel_request,
            "gel_used_ml": gel_used,
        }

    @property
    def conservation_error_ml(self) -> float:
        contrast_error = (
            self.initial_contrast_ml
            - self.contrast_remaining_ml
            - self.contrast_consumed_ml
        )
        gel_error = (
            self.initial_gel_ml - self.gel_remaining_ml - self.gel_consumed_ml
        )
        return float(contrast_error + gel_error)


@dataclass(frozen=True)
class SensorOperatingState:
    """Runtime health and registration state supplied by the sensor host."""

    sensor_state: str = "ready"
    failed_modalities: frozenset[str] = frozenset()
    registration_error_m: float = 0.0
    timestamp_skew_s: float = 0.0
    ultrasound_preload_n: float = 1.2
    doppler_preload_n: float = 0.35

    def __post_init__(self) -> None:
        _check(self.sensor_state, VALID_SENSOR_STATES, "sensor_state")
        unknown = set(self.failed_modalities).difference(VALID_SENSOR_MODALITIES)
        if unknown:
            raise ValueError(f"Unknown failed modalities: {sorted(unknown)}")
        for label in (
            "registration_error_m",
            "timestamp_skew_s",
            "ultrasound_preload_n",
            "doppler_preload_n",
        ):
            value = _finite(getattr(self, label), label)
            if value < 0.0:
                raise ValueError(f"{label} must be non-negative")


@dataclass(frozen=True)
class ProbeContactOutput:
    target_extension_delta_m: float
    coupled: bool
    overload: bool
    abort: bool
    force_error_n: float


@dataclass
class ProbeContactController:
    """Bounded outer loop for host-reported ultrasound or Doppler preload."""

    target_preload_n: float = 1.2
    minimum_coupled_force_n: float = 0.25
    soft_force_limit_n: float = 2.5
    hard_force_limit_n: float = 4.0
    position_gain_m_per_n: float = 0.0008
    maximum_step_m: float = 0.0015

    def update(self, *, measured_force_n: float, dt_s: float) -> ProbeContactOutput:
        force = _finite(measured_force_n, "measured_force_n")
        dt = _finite(dt_s, "dt_s")
        if force < 0.0:
            raise ValueError("measured_force_n must be non-negative")
        if dt <= 0.0:
            raise ValueError("dt_s must be positive")
        error = self.target_preload_n - force
        overload = force >= self.soft_force_limit_n
        abort = force >= self.hard_force_limit_n
        if abort:
            delta = -self.maximum_step_m
        else:
            delta = _clamp(
                self.position_gain_m_per_n * error * min(dt * 120.0, 1.0),
                -self.maximum_step_m,
                self.maximum_step_m,
            )
        return ProbeContactOutput(
            target_extension_delta_m=float(delta),
            coupled=bool(
                self.minimum_coupled_force_n
                <= force
                < self.soft_force_limit_n
            ),
            overload=overload,
            abort=abort,
            force_error_n=float(error),
        )


def load_perfusion_graph(path: Path = PERFUSION_GRAPH_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def make_tool_cfg(
    prim_path: str = "/World/DrAnmarPerfusionViabilityTool",
    *, contrast_state: str = "full", gel_state: str = "full", sensor_state: str = "ready",
    position=(0.0, 0.0, 0.35), orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    _check(contrast_state, VALID_CONTRAST_STATES, "contrast_state")
    _check(gel_state, VALID_GEL_STATES, "gel_state")
    _check(sensor_state, VALID_SENSOR_STATES, "sensor_state")
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(TOOL_STANDALONE_USD),
            variants={"contrast_state": contrast_state, "gel_state": gel_state, "sensor_state": sensor_state},
            # The research controller consumes host-reported preload at the
            # authored force frames; it does not require Isaac Lab to attach a
            # ContactSensor to every rigid body during composition.
            activate_contact_sensors=False,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=24,
                solver_velocity_iteration_count=8,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=position,
            rot=orientation_wxyz,
            joint_pos={name: 0.0 for name in TOOL_JOINTS.values()},
        ),
        actuators={
            "turret_filter": ImplicitActuatorCfg(
                joint_names_expr=["sensor_turret_joint", "filter_wheel_joint"],
                effort_limit_sim=35.0, velocity_limit_sim=2.0, stiffness=580.0, damping=42.0,
            ),
            "optical": ImplicitActuatorCfg(
                joint_names_expr=["optical_focus_joint", "speckle_scan_.*_joint"],
                effort_limit_sim=28.0, velocity_limit_sim=1.3, stiffness=1700.0, damping=66.0,
            ),
            "ultrasound": ImplicitActuatorCfg(
                joint_names_expr=["ultrasound_.*_joint"],
                effort_limit_sim=95.0, velocity_limit_sim=1.0, stiffness=3600.0, damping=130.0,
            ),
            "doppler": ImplicitActuatorCfg(
                joint_names_expr=["doppler_.*_joint"],
                effort_limit_sim=72.0, velocity_limit_sim=1.0, stiffness=3100.0, damping=112.0,
            ),
            "gel_guard": ImplicitActuatorCfg(
                joint_names_expr=["gel_valve_joint", "contact_guard_joint"],
                effort_limit_sim=25.0, velocity_limit_sim=0.25, stiffness=1150.0, damping=48.0,
            ),
        },
    )


def make_rigid_proxy_cfg(
    prim_path: str = "/World/DrAnmarPerfusionViabilityProxy",
    *, position=(0.0, 0.0, 0.35), orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(usd_path=str(TOOL_RIGID_PROXY_USD), activate_contact_sensors=True),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=orientation_wxyz),
    )


def _spawn_single_franka_with_tool(prim_path: str, cfg: Any, translation=None, orientation=None, **kwargs):
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    from isaaclab.sim.utils import create_prim, get_current_stage, select_usd_variants
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    robot = spawn_from_usd(prim_path, cfg, translation, orientation)
    stage = get_current_stage()
    robot_path = Sdf.Path(prim_path)
    robot_prim = stage.GetPrimAtPath(robot_path)
    descendants = list(Usd.PrimRange(robot_prim))
    named = {prim.GetName(): prim for prim in descendants}
    disabled = {
        "panda_hand_joint", "panda_hand", "panda_finger_joint1", "panda_finger_joint2",
        "panda_leftfinger", "panda_rightfinger",
    }
    missing = sorted(
        name for name in disabled | {"panda_link7"}
        if name not in named
    )
    if missing:
        raise RuntimeError(
            "Composable Franka asset changed; missing mount prims: "
            + ", ".join(missing)
        )
    stock_joint = UsdPhysics.Joint(named["panda_hand_joint"])
    stock_body_paths = stock_joint.GetBody0Rel().GetTargets()
    if (
        len(stock_body_paths) != 1
        or stock_body_paths[0] != named["panda_link7"].GetPath()
    ):
        raise RuntimeError(f"Unexpected stock Franka hand parent: {stock_body_paths}")
    stock_position = stock_joint.GetLocalPos0Attr().Get() or Gf.Vec3f(0, 0, 0)

    # NVIDIA's composable Franka omits the URDF terminal rigid body and
    # collapses panda_link8 into the panda_link7-to-hand joint. Reconstruct the
    # fixed terminal body so the payload really mounts to panda_link8.
    link8 = named.get("panda_link8")
    if link8 is None or not link8.IsValid():
        link8_path = robot_path.AppendChild("panda_link8")
        link8 = stage.DefinePrim(link8_path, "Xform")
        cache = UsdGeom.XformCache()
        robot_world = cache.GetLocalToWorldTransform(robot_prim)
        link7_world = cache.GetLocalToWorldTransform(named["panda_link7"])
        link8_world = Gf.Matrix4d(link7_world)
        link8_world.SetTranslate(link7_world.Transform(Gf.Vec3d(stock_position)))
        link8_local = link8_world * robot_world.GetInverse()
        link8_xform = UsdGeom.Xformable(link8)
        link8_xform.ClearXformOpOrder()
        link8_xform.AddTransformOp().Set(link8_local)
    if link8.IsInstanceProxy():
        raise RuntimeError("panda_link8 is instanceable and cannot host a payload")
    UsdPhysics.RigidBodyAPI.Apply(link8).CreateRigidBodyEnabledAttr().Set(True)
    link8_mass = UsdPhysics.MassAPI.Apply(link8)
    link8_mass.CreateMassAttr().Set(0.0001)
    link8_mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(0, 0, 0))
    link8_mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(1.0e-8, 1.0e-8, 1.0e-8))
    link8_mass.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0, 0, 0))
    joint_scope = stage.DefinePrim(
        robot_path.AppendChild("DrAnmarPerfusionJoints"), "Scope"
    )
    link8_joint = UsdPhysics.FixedJoint.Define(
        stage, joint_scope.GetPath().AppendChild("panda_link7_to_link8")
    )
    link8_joint.CreateBody0Rel().SetTargets([named["panda_link7"].GetPath()])
    link8_joint.CreateBody1Rel().SetTargets([link8.GetPath()])
    link8_joint.CreateLocalPos0Attr().Set(stock_position)
    link8_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    link8_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0, 0, 0))
    link8_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0, 0, 0))
    link8_joint.CreateCollisionEnabledAttr().Set(False)
    mount_body_paths = [link8.GetPath()]
    angle = math.radians(-45.0) / 2.0
    mount_local_rot0 = Gf.Quatf(math.cos(angle), 0, 0, math.sin(angle))

    candidate_paths = [
        prim.GetPath()
        for prim in descendants
        if prim.GetPath().HasPrefix(robot_path) and prim.GetName() in disabled
    ]
    paths_to_disable = []
    for path in sorted(candidate_paths, key=lambda item: str(item).count("/")):
        if not any(path.HasPrefix(parent) for parent in paths_to_disable):
            paths_to_disable.append(path)
    for path in paths_to_disable:
        stage.OverridePrim(path).SetActive(False)
    tool_path = f"{prim_path}/DrAnmarPerfusionViabilityTool"
    create_prim(tool_path, usd_path=str(TOOL_PAYLOAD_USD), stage=stage)
    select_usd_variants(
        tool_path,
        {
            "contrast_state": cfg.contrast_state,
            "gel_state": cfg.gel_state,
            "sensor_state": cfg.sensor_state,
        },
    )
    cache = UsdGeom.XformCache()
    robot_world = cache.GetLocalToWorldTransform(robot_prim)
    link8_world = cache.GetLocalToWorldTransform(link8)
    link8_local = link8_world * robot_world.GetInverse()
    mount_rotation = Gf.Matrix4d(1.0)
    mount_rotation.SetRotate(Gf.Quatd(math.cos(angle), 0, 0, math.sin(angle)))
    payload_local = mount_rotation * link8_local
    payload_xform = UsdGeom.Xformable(stage.GetPrimAtPath(tool_path))
    payload_xform.ClearXformOpOrder()
    payload_xform.AddTransformOp().Set(payload_local)
    joint = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/dranmar_perfusion_mount_joint")
    joint.CreateBody0Rel().SetTargets(mount_body_paths)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{tool_path}/Links/Mount")])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot0Attr().Set(mount_local_rot0)
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    return robot


def spawn_franka_with_tool(prim_path: str, cfg: Any, translation=None, orientation=None, **kwargs):
    from isaaclab.sim.utils import clone
    return clone(_spawn_single_franka_with_tool)(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)


def make_franka_perfusion_viability_robot_cfg(
    *, prim_path="/World/Robot", contrast_state="full", gel_state="full", sensor_state="ready",
):
    _check(contrast_state, VALID_CONTRAST_STATES, "contrast_state")
    _check(gel_state, VALID_GEL_STATES, "gel_state")
    _check(sensor_state, VALID_SENSOR_STATES, "sensor_state")
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

    @configclass
    class FrankaPerfusionUsdCfg(sim_utils.UsdFileCfg):
        contrast_state: str = "full"
        gel_state: str = "full"
        sensor_state: str = "ready"
        func = spawn_franka_with_tool

    cfg = FRANKA_PANDA_CFG.copy()
    cfg.prim_path = prim_path
    cfg.spawn = FrankaPerfusionUsdCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        variants={"Gripper": "Default", "Mesh": "Performance"},
        contrast_state=contrast_state,
        gel_state=gel_state,
        sensor_state=sensor_state,
        activate_contact_sensors=False,
        rigid_props=FRANKA_PANDA_CFG.spawn.rigid_props,
        articulation_props=FRANKA_PANDA_CFG.spawn.articulation_props,
    )
    cfg.init_state.joint_pos = {
        key: value
        for key, value in cfg.init_state.joint_pos.items()
        if "finger" not in key
    }
    cfg.init_state.joint_pos.update({name: 0.0 for name in TOOL_JOINTS.values()})
    cfg.actuators = {
        key: value for key, value in cfg.actuators.items() if key != "panda_hand"
    }
    cfg.actuators.update(
        {
            "perfusion_turret": ImplicitActuatorCfg(joint_names_expr=["sensor_turret_joint", "filter_wheel_joint"], effort_limit_sim=35.0, velocity_limit_sim=2.0, stiffness=580.0, damping=42.0),
            "perfusion_optical": ImplicitActuatorCfg(joint_names_expr=["optical_focus_joint", "speckle_scan_.*_joint"], effort_limit_sim=28.0, velocity_limit_sim=1.3, stiffness=1700.0, damping=66.0),
            "perfusion_ultrasound": ImplicitActuatorCfg(joint_names_expr=["ultrasound_.*_joint"], effort_limit_sim=95.0, velocity_limit_sim=1.0, stiffness=3600.0, damping=130.0),
            "perfusion_doppler": ImplicitActuatorCfg(joint_names_expr=["doppler_.*_joint"], effort_limit_sim=72.0, velocity_limit_sim=1.0, stiffness=3100.0, damping=112.0),
            "perfusion_aux": ImplicitActuatorCfg(joint_names_expr=["gel_valve_joint", "contact_guard_joint"], effort_limit_sim=25.0, velocity_limit_sim=0.25, stiffness=1150.0, damping=48.0),
        }
    )
    return cfg


def spawn_tissue_demo(
    prim_path: str = "/World/PerfusedTissue",
    *, condition: str = "healthy", translation=(0.0, 0.0, 0.0), orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
):
    _check(condition, VALID_CONDITIONS, "condition")
    import isaaclab.sim as sim_utils
    cfg = sim_utils.UsdFileCfg(usd_path=str(TISSUE_DEMO_USD), variants={"condition": condition})
    return cfg.func(prim_path, cfg, translation=translation, orientation=orientation_wxyz)


def apply_perfused_tissue_surface_deformable(
    root_path: str,
    *,
    self_collision: bool = False,
    youngs_modulus_pa: float = 125_000.0,
    poissons_ratio: float = 0.38,
    thickness_m: float = 0.008,
    stage=None,
) -> dict[str, Any]:
    """Cook and bind the tissue surface using current PhysX surface schemas."""

    from omni.physx.scripts import deformableUtils
    from pxr import Sdf, UsdShade

    if stage is None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
    mesh_path = f"{root_path.rstrip('/')}/Geometry/TissueSurface"
    mesh = stage.GetPrimAtPath(mesh_path)
    if not mesh or not mesh.IsValid():
        raise ValueError(f"No perfused tissue surface at {mesh_path}")
    success = deformableUtils.set_physics_surface_deformable_body(
        stage, mesh.GetPath()
    )
    if success is False:
        raise RuntimeError(f"Failed to cook surface deformable at {mesh_path}")
    mesh.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
    if mesh.HasAPI("PhysxSurfaceDeformableBodyAPI"):
        mesh.GetAttribute("physxDeformableBody:selfCollision").Set(
            bool(self_collision)
        )
    material_path = f"{root_path.rstrip('/')}/RuntimeMaterials/PerfusedTissue"
    material = UsdShade.Material.Define(stage, material_path)
    material_prim = material.GetPrim()
    youngs_modulus = _finite(youngs_modulus_pa, "youngs_modulus_pa")
    poisson = _finite(poissons_ratio, "poissons_ratio")
    thickness = _finite(thickness_m, "thickness_m")
    if youngs_modulus <= 0.0 or thickness <= 0.0:
        raise ValueError("surface modulus and thickness must be positive")
    if not -1.0 < poisson < 0.5:
        raise ValueError("poissons_ratio must be between -1 and 0.5")
    for schema in (
        "OmniPhysicsSurfaceDeformableMaterialAPI",
        "PhysxSurfaceDeformableMaterialAPI",
    ):
        try:
            material_prim.ApplyAPI(schema)
        except Exception:
            pass
    attributes = {
        "omniphysics:dynamicFriction": 0.48,
        "omniphysics:density": 1060.0,
        "omniphysics:youngsModulus": youngs_modulus,
        "omniphysics:poissonsRatio": poisson,
        "omniphysics:surfaceThickness": thickness,
        "omniphysics:surfaceBendStiffness": 0.0,
        "physxDeformableMaterial:elasticityDamping": 0.16,
        "physxDeformableMaterial:bendDamping": 0.18,
    }
    for name, value in attributes.items():
        attribute = material_prim.GetAttribute(name)
        if not attribute:
            attribute = material_prim.CreateAttribute(
                name, Sdf.ValueTypeNames.Float
            )
        attribute.Set(value)
    UsdShade.MaterialBindingAPI.Apply(mesh).Bind(
        material, UsdShade.Tokens.weakerThanDescendants, "physics"
    )
    return {
        "root_path": root_path,
        "mesh_path": mesh_path,
        "material_path": material_path,
        "self_collision": bool(self_collision),
    }


def create_perfused_tissue_fixture_attachment(
    deformable_path: str,
    target_path: str,
    attachment_path: str,
    *,
    maximum_vertices: int = 16,
    stage=None,
) -> str:
    """Attach the nearest populated surface vertices to a fixture target."""

    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    if stage is None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(attachment_path).IsValid():
        stage.RemovePrim(attachment_path)
    definition = Usd.SchemaRegistry().FindConcretePrimDefinition(
        "OmniPhysicsVtxXformAttachment"
    )
    if not definition:
        raise RuntimeError(
            "OmniPhysicsVtxXformAttachment is unavailable in this runtime"
        )
    deformable = stage.GetPrimAtPath(deformable_path)
    target = stage.GetPrimAtPath(target_path)
    mesh = UsdGeom.Mesh(deformable)
    points = list(mesh.GetPointsAttr().Get() or [])
    if not deformable.IsValid() or not mesh or not points:
        raise ValueError(f"Attachment source is not a mesh: {deformable_path}")
    if not target.IsValid() or not UsdGeom.Xformable(target):
        raise ValueError(f"Attachment target is not xformable: {target_path}")
    mesh_to_world = UsdGeom.Xformable(deformable).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    target_to_world = UsdGeom.Xformable(target).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    world_to_target = target_to_world.GetInverse()
    bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
    ).ComputeWorldBound(target).ComputeAlignedRange()
    minimum, maximum = bounds.GetMin(), bounds.GetMax()
    center = (minimum + maximum) * 0.5
    ranked = []
    for index, point in enumerate(points):
        world = mesh_to_world.Transform(Gf.Vec3d(point))
        delta = world - center
        overlaps = all(
            minimum[axis] - 0.003 <= world[axis] <= maximum[axis] + 0.003
            for axis in range(3)
        )
        ranked.append((float(Gf.Dot(delta, delta)), index, world, overlaps))
    ranked.sort(key=lambda item: item[0])
    selected = [item for item in ranked if item[3]][:maximum_vertices]
    if len(selected) < 4:
        selected = ranked[: min(maximum_vertices, len(ranked))]
    if not selected:
        raise RuntimeError(f"No fixture vertices available for {attachment_path}")
    attachment = stage.DefinePrim(
        attachment_path, "OmniPhysicsVtxXformAttachment"
    )
    attachment.CreateRelationship("omniphysics:src0").SetTargets(
        [Sdf.Path(deformable_path)]
    )
    attachment.CreateRelationship("omniphysics:src1").SetTargets(
        [Sdf.Path(target_path)]
    )
    attachment.CreateAttribute(
        "omniphysics:vtxIndicesSrc0", Sdf.ValueTypeNames.IntArray
    ).Set(Vt.IntArray([item[1] for item in selected]))
    attachment.CreateAttribute(
        "omniphysics:localPositionsSrc1", Sdf.ValueTypeNames.Point3fArray
    ).Set(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(world_to_target.Transform(item[2]))
                for item in selected
            ]
        )
    )
    attachment.CreateAttribute(
        "omniphysics:attachmentEnabled", Sdf.ValueTypeNames.Bool
    ).Set(True)
    return "OmniPhysicsVtxXformAttachment"


def spawn_coupling_pad(prim_path: str, *, translation=(0,0,0), orientation_wxyz=(1,0,0,0)):
    import isaaclab.sim as sim_utils
    cfg = sim_utils.UsdFileCfg(usd_path=str(COUPLING_PAD_USD))
    return cfg.func(prim_path, cfg, translation=translation, orientation=orientation_wxyz)


def spawn_flow_occluder(prim_path: str, *, state="partial", translation=(0,0,0), orientation_wxyz=(1,0,0,0)):
    if state not in {"open", "partial", "closed"}:
        raise ValueError(state)
    import isaaclab.sim as sim_utils
    cfg = sim_utils.UsdFileCfg(usd_path=str(FLOW_OCCLUDER_USD), variants={"state": state})
    return cfg.func(prim_path, cfg, translation=translation, orientation=orientation_wxyz)


def attach_camera_prims(stage, tool_path: str) -> dict[str, str]:
    """Create standard USD cameras at the authored modality frames.

    RTX rendering and annotators are deliberately configured by the host scene;
    this function only authors portable camera prims and optical parameters.
    """
    from pxr import Gf, Sdf, UsdGeom
    specs = {
        "rgb_left_camera": (24.0, 20.955, 0.01, 2.0),
        "rgb_right_camera": (24.0, 20.955, 0.01, 2.0),
        "nir_fluorescence_camera": (35.0, 20.955, 0.01, 2.0),
        "speckle_camera": (45.0, 20.955, 0.01, 1.0),
        "thermal_camera": (25.0, 20.955, 0.01, 2.0),
        "multispectral_camera": (35.0, 20.955, 0.01, 2.0),
    }
    created = {}
    for name, (focal, aperture, near, far) in specs.items():
        path = f"{frame_path(tool_path, name)}/Camera"
        camera = UsdGeom.Camera.Define(stage, path)
        camera_xform = UsdGeom.Xformable(camera)
        camera_xform.ClearXformOpOrder()
        # USD cameras observe along local -Z. The authored modality frames use
        # local +Z as the tissue-facing optical axis, so bridge the conventions
        # explicitly instead of leaving some sensors aimed into the turret.
        camera_xform.AddOrientOp().Set(Gf.Quatf(0.0, 0.0, 1.0, 0.0))
        camera.CreateFocalLengthAttr(float(focal))
        camera.CreateHorizontalApertureAttr(float(aperture))
        camera.CreateClippingRangeAttr(Gf.Vec2f(float(near), float(far)))
        camera.GetPrim().CreateAttribute("drAnmar:modality", Sdf.ValueTypeNames.String).Set(name)
        created[name] = path
    return created


def sensor_runtime_contract(tool_path: str) -> dict[str, Any]:
    """Return version-neutral contracts for the host sensor runtime."""
    return {
        "dynamic_scene_camera_route": "usd_rtx_camera",
        "camera_frames": {name: frame_path(tool_path, name) for name in (
            "rgb_left_camera", "rgb_right_camera", "nir_fluorescence_camera",
            "speckle_camera", "thermal_camera", "multispectral_camera",
        )},
        "ultrasound": {
            "probe_pose_frame": frame_path(tool_path, "ultrasound_probe_face"),
            "beam_axis_frame": frame_path(tool_path, "ultrasound_probe_axis"),
            "recommended_bridge": "i4h_robotic_ultrasound_raytracing_application",
            "output": "b_mode_frame",
        },
        "doppler": {
            "probe_tip_frame": frame_path(tool_path, "doppler_probe_tip"),
            "beam_axis_frame": frame_path(tool_path, "doppler_beam_axis"),
            "output": "signed_projected_velocity_and_direction",
        },
        "shared_timestamp_required": True,
        "shared_state": "vascular_flow_tracer_compression_leak_temperature_oxygenation",
    }


@dataclass(frozen=True)
class VascularFlowResult:
    condition: str
    recovery_fraction: float
    node_pressures_kpa: dict[str, float]
    edge_flows_ml_s: dict[str, float]
    edge_velocities_m_s: dict[str, float]
    region_flows_ml_s: dict[int, float]
    leak_flows_ml_s: dict[str, float]
    leak_fractions: dict[str, float]
    total_inflow_ml_s: float
    total_outflow_ml_s: float
    conservation_error_ml_s: float


class VascularFlowSolver:
    """Linear resistive network with obstruction, compression, and leak sinks."""
    def __init__(self, graph: Mapping[str, Any] | None = None):
        self.graph = dict(graph or load_perfusion_graph())
        self.nodes = dict(self.graph["nodes"])
        self.edges = list(self.graph["edges"])
        self.edge_by_id = {e["id"]: e for e in self.edges}
        self.boundary = dict(self.graph["boundary_nodes"])

    def _edge_multiplier(self, edge: Mapping[str, Any], condition: Mapping[str, Any]) -> float:
        multiplier = float(condition.get("edge_multipliers", {}).get(edge["id"], 1.0))
        region = edge.get("region")
        if region is not None:
            multiplier *= float(condition.get("region_compression", {}).get(str(region), 1.0))
        return max(multiplier, 1.0e-6)

    def _blended_condition(
        self, condition: str, recovery_fraction: float
    ) -> dict[str, Any]:
        base = self.graph["conditions"][condition]
        recovered = self.graph["conditions"]["recovered"]
        progress = _clamp(recovery_fraction)

        def blend_map(key: str) -> dict[str, float]:
            base_values = base.get(key, {})
            recovered_values = recovered.get(key, {})
            names = set(base_values) | set(recovered_values)
            default = 0.0 if key == "leak_edges" else 1.0
            return {
                name: (
                    (1.0 - progress) * float(base_values.get(name, default))
                    + progress * float(recovered_values.get(name, default))
                )
                for name in names
            }

        return {
            "edge_multipliers": blend_map("edge_multipliers"),
            "region_compression": blend_map("region_compression"),
            "leak_edges": blend_map("leak_edges"),
        }

    def solve(
        self,
        condition: str = "healthy",
        *,
        recovery_fraction: float = 0.0,
        arterial_pressure_kpa: float | None = None,
        venous_pressure_kpa: float | None = None,
    ) -> VascularFlowResult:
        import numpy as np
        _check(condition, VALID_CONDITIONS, "condition")
        recovery = _finite(recovery_fraction, "recovery_fraction")
        if not 0.0 <= recovery <= 1.0:
            raise ValueError("recovery_fraction must be between zero and one")
        condition_cfg = self._blended_condition(condition, recovery)
        p_ref = self.graph["reference_pressures_kpa"]
        arterial_pressure = _finite(
            p_ref["arterial"] if arterial_pressure_kpa is None else arterial_pressure_kpa,
            "arterial_pressure_kpa",
        )
        venous_pressure = _finite(
            p_ref["venous"] if venous_pressure_kpa is None else venous_pressure_kpa,
            "venous_pressure_kpa",
        )
        if arterial_pressure <= venous_pressure:
            raise ValueError("arterial_pressure_kpa must exceed venous_pressure_kpa")
        boundary_pressures = {
            self.boundary["arterial_inlet"]: arterial_pressure,
            self.boundary["venous_outlet"]: venous_pressure,
        }
        internal = [n for n in self.nodes if n not in boundary_pressures]
        index = {n: i for i, n in enumerate(internal)}
        A = np.zeros((len(internal), len(internal)), dtype=float)
        b = np.zeros(len(internal), dtype=float)
        conductance: dict[str, float] = {}
        for edge in self.edges:
            g = self._edge_multiplier(edge, condition_cfg) / max(float(edge["resistance_kpa_s_ml"]), 1.0e-12)
            conductance[edge["id"]] = g
            a, c = edge["from"], edge["to"]
            for u, v in ((a, c), (c, a)):
                if u in index:
                    A[index[u], index[u]] += g
                    if v in index:
                        A[index[u], index[v]] -= g
                    else:
                        b[index[u]] += g * boundary_pressures[v]
        leak_conductance: dict[str, tuple[str, float]] = {}
        for edge_id, fraction in condition_cfg.get("leak_edges", {}).items():
            edge = self.edge_by_id[edge_id]
            node = edge["to"]
            g = float(fraction) / max(float(edge["resistance_kpa_s_ml"]), 1.0e-12)
            leak_conductance[edge_id] = (node, g)
            if node in index:
                A[index[node], index[node]] += g
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(A, b, rcond=None)[0]
        pressure = dict(boundary_pressures)
        pressure.update({node: float(x[index[node]]) for node in internal})
        edge_flows = {}
        edge_velocities = {}
        region_flows = {}
        for edge in self.edges:
            q = conductance[edge["id"]] * (pressure[edge["from"]] - pressure[edge["to"]])
            edge_flows[edge["id"]] = float(q)
            radius = float(edge["radius_m"])
            edge_velocities[edge["id"]] = float((q * 1.0e-6) / max(math.pi * radius * radius, 1.0e-18))
            if edge.get("region") is not None:
                region_flows[int(edge["region"])] = abs(float(q))
        leak_flows = {edge_id: max(0.0, g * pressure[node]) for edge_id, (node, g) in leak_conductance.items()}
        inlet = self.boundary["arterial_inlet"]
        outlet = self.boundary["venous_outlet"]
        total_in = sum(q for edge_id, q in edge_flows.items() if self.edge_by_id[edge_id]["from"] == inlet) - sum(q for edge_id, q in edge_flows.items() if self.edge_by_id[edge_id]["to"] == inlet)
        total_out = sum(q for edge_id, q in edge_flows.items() if self.edge_by_id[edge_id]["to"] == outlet) - sum(q for edge_id, q in edge_flows.items() if self.edge_by_id[edge_id]["from"] == outlet)
        conservation = total_in - total_out - sum(leak_flows.values())
        return VascularFlowResult(
            condition=condition,
            recovery_fraction=recovery,
            node_pressures_kpa=pressure,
            edge_flows_ml_s=edge_flows,
            edge_velocities_m_s=edge_velocities,
            region_flows_ml_s=region_flows,
            leak_flows_ml_s=leak_flows,
            leak_fractions={
                edge_id: float(value)
                for edge_id, value in condition_cfg.get("leak_edges", {}).items()
                if float(value) > 0.0
            },
            total_inflow_ml_s=float(total_in),
            total_outflow_ml_s=float(total_out),
            conservation_error_ml_s=float(conservation),
        )


@dataclass
class TracerFrame:
    time_s: float
    arterial_input: float
    edge_concentration: dict[str, float]
    region_concentration: dict[int, float]
    extravascular_concentration: dict[int, float]


class ICGTracerTransport:
    """Stable edge-compartment transport driven by the solved vascular flows."""
    def __init__(
        self,
        graph: Mapping[str, Any] | None = None,
        *,
        injection_time_s: float = 1.0,
        peak_input_time_s: float = 4.5,
        input_scale: float = 1.0,
    ):
        self.graph = dict(graph or load_perfusion_graph())
        self.nodes = dict(self.graph["nodes"])
        self.edges = list(self.graph["edges"])
        self.edge_by_id = {e["id"]: e for e in self.edges}
        self.injection_time_s = _finite(injection_time_s, "injection_time_s")
        self.peak_input_time_s = _finite(peak_input_time_s, "peak_input_time_s")
        self.input_scale = _finite(input_scale, "input_scale")
        if self.injection_time_s < 0.0:
            raise ValueError("injection_time_s must be non-negative")
        if self.peak_input_time_s <= 0.0:
            raise ValueError("peak_input_time_s must be positive")
        if not 0.0 <= self.input_scale <= 1.0:
            raise ValueError("input_scale must be between zero and one")
        self.time_s = 0.0
        self.edge_c = {e["id"]: 0.0 for e in self.edges}
        self.region_c = {int(r["index"]): 0.0 for r in self.graph["regions"]}
        self.extra_c = {int(r["index"]): 0.0 for r in self.graph["regions"]}
        self._edge_volume_ml = {}
        for edge in self.edges:
            p0 = self.nodes[edge["from"]]["position_m"]
            p1 = self.nodes[edge["to"]]["position_m"]
            length = math.sqrt(sum((float(a)-float(b))**2 for a,b in zip(p0,p1)))
            radius = float(edge["radius_m"])
            self._edge_volume_ml[edge["id"]] = max(math.pi * radius * radius * length * 1.0e6, 1.0e-6)

    def reset(self) -> None:
        self.time_s = 0.0
        for key in self.edge_c: self.edge_c[key] = 0.0
        for key in self.region_c: self.region_c[key] = 0.0
        for key in self.extra_c: self.extra_c[key] = 0.0

    def arterial_input(self, time_s: float | None = None) -> float:
        t = self.time_s if time_s is None else float(time_s)
        x = max(0.0, t - self.injection_time_s)
        if x <= 0.0:
            return 0.0
        alpha = 3.4
        tp = self.peak_input_time_s
        ratio = x / tp
        return float(
            self.input_scale
            * (ratio**alpha)
            * math.exp(alpha * (1.0 - ratio))
        )

    def step(self, flow: VascularFlowResult, dt_s: float) -> TracerFrame:
        dt = _finite(dt_s, "dt_s")
        if dt <= 0:
            raise ValueError("dt_s must be positive")
        self.time_s += dt
        inlet = self.graph["boundary_nodes"]["arterial_inlet"]
        incoming: dict[str, list[tuple[float, float]]] = {n: [] for n in self.nodes}
        directed = {}
        for edge in self.edges:
            q = float(flow.edge_flows_ml_s[edge["id"]])
            if q >= 0:
                src, dst = edge["from"], edge["to"]
            else:
                src, dst = edge["to"], edge["from"]
            directed[edge["id"]] = (src, dst, abs(q))
            incoming[dst].append((abs(q), self.edge_c[edge["id"]]))
        node_c = {}
        for node in self.nodes:
            if node == inlet:
                node_c[node] = self.arterial_input()
            elif incoming[node]:
                total = sum(q for q,_ in incoming[node])
                node_c[node] = sum(q*c for q,c in incoming[node]) / max(total, 1.0e-12)
            else:
                node_c[node] = 0.0
        new_edge = {}
        for edge in self.edges:
            src, _dst, q = directed[edge["id"]]
            volume = self._edge_volume_ml[edge["id"]]
            tau = volume / max(q, 1.0e-8)
            retention = math.exp(-dt / max(tau, 1.0e-5))
            clearance = math.exp(-0.012 * dt)
            new_edge[edge["id"]] = max(0.0, (node_c[src] + (self.edge_c[edge["id"]] - node_c[src]) * retention) * clearance)
        self.edge_c = new_edge
        leak_cfg = flow.leak_fractions
        for region in self.graph["regions"]:
            idx = int(region["index"])
            cap = self.edge_c[region["capillary_edge"]]
            region_flow = float(flow.region_flows_ml_s.get(idx, 0.0))
            uptake = min(2.5, 0.16 + 8.0 * region_flow)
            washout = 0.10 + 2.0 * region_flow
            target = 0.76 * cap
            self.region_c[idx] += dt * (uptake * (target - self.region_c[idx]) - 0.018 * self.region_c[idx])
            self.region_c[idx] = max(0.0, self.region_c[idx])
            leak_fraction = float(leak_cfg.get(region["capillary_edge"], 0.0))
            self.extra_c[idx] += dt * (leak_fraction * cap * 0.7 - 0.035 * self.extra_c[idx])
            self.extra_c[idx] = max(0.0, self.extra_c[idx])
        return TracerFrame(self.time_s, self.arterial_input(), dict(self.edge_c), dict(self.region_c), dict(self.extra_c))


@dataclass(frozen=True)
class RegionICGMetrics:
    arrival_time_s: float | None
    wash_in_slope_per_s: float
    time_to_peak_s: float | None
    peak_intensity: float
    washout_slope_per_s: float
    area_under_curve: float


class PerfusionTimeSeries:
    def __init__(self, region_count: int = 24):
        self.times: list[float] = []
        self.values: dict[int, list[float]] = {i: [] for i in range(region_count)}
        self.extravascular: dict[int, list[float]] = {i: [] for i in range(region_count)}

    def append(self, frame: TracerFrame) -> None:
        self.times.append(float(frame.time_s))
        for i in self.values:
            self.values[i].append(float(frame.region_concentration.get(i, 0.0)))
            self.extravascular[i].append(float(frame.extravascular_concentration.get(i, 0.0)))

    def metrics(self, region: int) -> RegionICGMetrics:
        import numpy as np
        if not self.times:
            return RegionICGMetrics(None, 0.0, None, 0.0, 0.0, 0.0)
        t = np.asarray(self.times, dtype=float)
        y = np.asarray(self.values[region], dtype=float)
        peak = float(y.max())
        peak_i = int(y.argmax())
        threshold = 0.10 * peak
        arrival_indices = np.where(y >= threshold)[0]
        arrival = float(t[arrival_indices[0]]) if peak > 0 and len(arrival_indices) else None
        slopes = np.diff(y) / np.maximum(np.diff(t), 1.0e-9)
        wash_in = float(slopes[:max(1, peak_i)].max(initial=0.0))
        post = slopes[peak_i:] if peak_i < len(slopes) else np.asarray([], dtype=float)
        washout = float(post.min(initial=0.0))
        integrate = getattr(np, "trapezoid", None)
        if integrate is None:
            # Isaac Sim 6 currently ships a NumPy 1.x environment, while
            # ``numpy.trapezoid`` was introduced in NumPy 2.0.
            integrate = np.trapz
        auc = float(integrate(y, t))
        return RegionICGMetrics(arrival, wash_in, float(t[peak_i]), peak, washout, auc)


@dataclass(frozen=True)
class MultimodalMaps:
    # ``flow_index`` is retained as simulator truth for evaluation and plotting.
    # The diagnostic estimator must never consume it as a sensor observation.
    flow_index: Any
    icg_intensity: Any
    icg_extravascular: Any
    speckle_perfusion: Any
    temperature_c: Any
    oxygenation_fraction: Any
    doppler_speed_m_s: Any
    ultrasound_patency: Any
    confidence: Any
    modality_validity: Mapping[str, bool]
    registration_error_m: float
    timestamp_skew_s: float
    sensor_state: str
    faults: tuple[str, ...]


class MultimodalSensorModel:
    """Generate registered modality maps from the same flow and tracer state."""
    def __init__(self, graph: Mapping[str, Any] | None = None, *, seed: int = 20260725):
        import numpy as np
        self.graph = dict(graph or load_perfusion_graph())
        self.rng = np.random.default_rng(seed)
        self.rows = 4
        self.cols = 6
        self.edge_by_id = {
            edge["id"]: edge for edge in self.graph["edges"]
        }

    def _grid(self, values: Mapping[int, float]):
        import numpy as np
        out = np.zeros((self.rows, self.cols), dtype=float)
        for region in self.graph["regions"]:
            idx = int(region["index"]); r,c = region["grid"]
            out[int(r), int(c)] = float(values.get(idx, 0.0))
        return out

    def observe(
        self,
        flow: VascularFlowResult,
        tracer: TracerFrame,
        *,
        healthy_reference: VascularFlowResult | None = None,
        operating_state: SensorOperatingState | None = None,
        contrast_available: bool = True,
        gel_available: bool = True,
    ) -> MultimodalMaps:
        import numpy as np
        state = operating_state or SensorOperatingState()
        if healthy_reference is None:
            healthy_reference = VascularFlowSolver(self.graph).solve("healthy")
        flow_norm = {i: flow.region_flows_ml_s.get(i,0.0) / max(healthy_reference.region_flows_ml_s.get(i,1.0e-9),1.0e-9) for i in range(24)}
        flow_grid = np.clip(self._grid(flow_norm), 0.0, 1.8)
        validity = {name: True for name in VALID_SENSOR_MODALITIES}
        faults = set(state.failed_modalities)
        if not contrast_available:
            faults.add("nir_icg")
        if not gel_available or not 0.25 <= state.ultrasound_preload_n < 2.5:
            faults.add("ultrasound")
        if not 0.05 <= state.doppler_preload_n < 2.5:
            faults.add("doppler")
        if state.sensor_state == "fault":
            faults.update(VALID_SENSOR_MODALITIES)
        for name in faults:
            validity[name] = False

        icg = self._grid(tracer.region_concentration)
        extra = self._grid(tracer.extravascular_concentration)
        if not validity["nir_icg"]:
            icg.fill(0.0)
            extra.fill(0.0)

        noise_scale = 0.025 if state.sensor_state == "ready" else 0.10
        speckle = np.clip(
            flow_grid + self.rng.normal(0.0, noise_scale, flow_grid.shape),
            0.0,
            1.8,
        )
        # Perfusion warms tissue toward blood temperature; low flow cools gradually.
        temperature = 31.8 + 4.5 * (flow_grid / (0.28 + flow_grid))
        venous_pressure = np.asarray(
            [
                flow.node_pressures_kpa[
                    self.edge_by_id[region["venule_edge"]]["from"]
                ]
                for region in self.graph["regions"]
            ],
            dtype=float,
        ).reshape(self.rows, self.cols)
        venous_reference = float(
            self.graph["reference_pressures_kpa"]["venous"]
        )
        congestion = np.clip(
            (venous_pressure - venous_reference) / 20.0, 0.0, 0.18
        )
        oxygenation = np.clip(
            0.34
            + 0.61 * (flow_grid / (0.22 + flow_grid))
            - congestion,
            0.20,
            0.98,
        )
        doppler = {}
        patency = {}
        for region in self.graph["regions"]:
            idx=int(region["index"])
            edge=region["arteriole_edge"]
            doppler[idx]=abs(flow.edge_velocities_m_s[edge])
            patency[idx]=min(1.0,max(0.0,flow_norm[idx]))
        doppler_grid=self._grid(doppler)
        patency_grid=self._grid(patency)
        if not validity["laser_speckle"]:
            speckle.fill(0.0)
        if not validity["thermal"]:
            temperature.fill(0.0)
        if not validity["surface_oxygenation"]:
            oxygenation.fill(0.0)
        if not validity["doppler"]:
            doppler_grid.fill(0.0)
        if not validity["ultrasound"]:
            patency_grid.fill(0.0)
        degradation_penalty = 0.28 if state.sensor_state == "degraded" else 0.0
        fault_fraction = sum(not value for value in validity.values()) / len(validity)
        confidence=np.clip(
            0.92
            - 0.10*np.abs(flow_grid-speckle)
            - 0.08*(extra>0.05)
            - degradation_penalty
            - 0.55*fault_fraction
            - min(0.45, state.registration_error_m * 80.0)
            - min(0.35, state.timestamp_skew_s * 4.0),
            0.02,
            0.99,
        )
        return MultimodalMaps(
            flow_index=flow_grid,
            icg_intensity=icg,
            icg_extravascular=extra,
            speckle_perfusion=speckle,
            temperature_c=temperature,
            oxygenation_fraction=oxygenation,
            doppler_speed_m_s=doppler_grid,
            ultrasound_patency=patency_grid,
            confidence=confidence,
            modality_validity=validity,
            registration_error_m=float(state.registration_error_m),
            timestamp_skew_s=float(state.timestamp_skew_s),
            sensor_state=state.sensor_state,
            faults=tuple(sorted(faults)),
        )

    def synthetic_bmode(self, flow: VascularFlowResult, *, width_px: int=256, depth_px: int=192, include_color_doppler: bool=True):
        import numpy as np
        rng=np.random.default_rng(4451)
        depth=np.linspace(0,1,depth_px)[:,None]
        base=rng.rayleigh(scale=0.34,size=(depth_px,width_px))*np.exp(-1.5*depth)
        # layered interfaces
        for z,amp in ((20,0.55),(62,0.32),(145,0.20)):
            if z<depth_px: base[max(0,z-1):min(depth_px,z+2),:]+=amp
        color=np.zeros((depth_px,width_px),dtype=float)
        # six representative vascular lumens
        for i,edge_id in enumerate(("AT0","AT2","AB08","AB15","VT1","VB20")):
            x=int(width_px*(0.15+0.14*i)); z=int(depth_px*(0.36+0.08*(i%3))); radius=5+(i%2)*2
            yy,xx=np.ogrid[:depth_px,:width_px]
            mask=(xx-x)**2+(yy-z)**2<=radius**2
            base[mask]*=0.10
            if include_color_doppler:
                velocity=flow.edge_velocities_m_s.get(edge_id,0.0)
                color[mask]=velocity
        base=np.clip(base,0,1)
        return {"b_mode":base,"color_doppler":color}

    def doppler_measure(self, flow: VascularFlowResult, edge_id: str, beam_direction=(0.0,0.0,1.0)) -> dict[str,float]:
        import numpy as np
        edge = next(e for e in self.graph["edges"] if e["id"]==edge_id)
        p0=np.asarray(self.graph["nodes"][edge["from"]]["position_m"],dtype=float)
        p1=np.asarray(self.graph["nodes"][edge["to"]]["position_m"],dtype=float)
        tangent=(p1-p0)/max(np.linalg.norm(p1-p0),1.0e-12)
        beam=np.asarray(beam_direction,dtype=float)
        if beam.shape != (3,) or not np.all(np.isfinite(beam)):
            raise ValueError("beam_direction must contain three finite values")
        beam_norm=float(np.linalg.norm(beam))
        if beam_norm <= 1.0e-12:
            raise ValueError("beam_direction must be non-zero")
        beam/=beam_norm
        v=float(flow.edge_velocities_m_s[edge_id])
        projected=v*float(np.dot(tangent,beam))
        return {"edge_id":edge_id,"axial_velocity_m_s":projected,"speed_m_s":abs(v),"direction_sign":1.0 if projected>=0 else -1.0}


@dataclass(frozen=True)
class RegionAssessment:
    region: int
    viability_score: float
    confidence: float
    disagreement: float
    status: str
    likely_cause: str


@dataclass(frozen=True)
class PerfusionAssessment:
    condition: str | None
    global_viability_score: float
    nonperfused_fraction: float
    asymmetry: float
    sensor_disagreement: float
    likely_cause: str
    recommended_action: str
    diagnostic_confidence: float
    abstained: bool
    usable_modalities: tuple[str, ...]
    regions: tuple[RegionAssessment, ...]


class MultimodalPerfusionEstimator:
    """Fuse registered observations without access to simulator ground truth."""
    WEIGHTS = {
        "icg": 0.22,
        "speckle": 0.22,
        "thermal": 0.12,
        "oxygenation": 0.16,
        "doppler": 0.14,
        "ultrasound": 0.14,
    }

    @staticmethod
    def _metric_grid(
        metrics: Mapping[int, RegionICGMetrics],
        getter,
        *,
        default: float = 0.0,
    ):
        import numpy as np

        return np.asarray(
            [
                float(default if getter(metrics[index]) is None else getter(metrics[index]))
                for index in range(24)
            ],
            dtype=float,
        ).reshape(4, 6)

    def _icg_observation(
        self,
        maps: MultimodalMaps,
        metrics: Mapping[int, RegionICGMetrics] | None,
    ):
        import numpy as np

        final = np.asarray(maps.icg_intensity, dtype=float)
        final_norm = final / max(float(np.max(final)), 1.0e-6)
        if not metrics:
            return np.clip(final_norm, 0.0, 1.0)
        peak = self._metric_grid(metrics, lambda item: item.peak_intensity)
        arrival = self._metric_grid(
            metrics, lambda item: item.arrival_time_s, default=30.0
        )
        wash_in = self._metric_grid(
            metrics, lambda item: item.wash_in_slope_per_s
        )
        washout = np.abs(
            self._metric_grid(metrics, lambda item: item.washout_slope_per_s)
        )
        peak_norm = peak / max(float(np.percentile(peak, 95)), 1.0e-6)
        arrival_score = np.exp(-np.maximum(arrival - 2.5, 0.0) / 3.0)
        wash_in_norm = wash_in / max(
            float(np.percentile(wash_in, 95)), 1.0e-6
        )
        washout_norm = washout / max(
            float(np.percentile(washout, 95)), 1.0e-6
        )
        return np.clip(
            0.30 * final_norm
            + 0.30 * peak_norm
            + 0.20 * arrival_score
            + 0.12 * wash_in_norm
            + 0.08 * washout_norm,
            0.0,
            1.0,
        )

    def estimate(
        self,
        observations: MultimodalMaps | str,
        maps: MultimodalMaps | None = None,
        *,
        icg_metrics: Mapping[int, RegionICGMetrics] | None = None,
        scenario_label: str | None = None,
    ) -> PerfusionAssessment:
        import numpy as np

        # Backward-compatible argument parsing deliberately discards the old
        # leading condition string. It may be retained only as an evaluation
        # label and can never influence fusion or classification.
        if isinstance(observations, str):
            if maps is None:
                raise TypeError("maps are required after a legacy scenario label")
            if scenario_label is None:
                scenario_label = observations
            observed = maps
        else:
            if maps is not None:
                raise TypeError("maps must not be supplied twice")
            observed = observations

        icg = self._icg_observation(observed, icg_metrics)
        doppler_scale = max(
            float(np.percentile(observed.doppler_speed_m_s, 90)), 1.0e-6
        )
        arrays = {
            "icg": icg,
            "speckle": np.clip(observed.speckle_perfusion, 0.0, 1.0),
            "thermal": np.clip((observed.temperature_c - 31.5) / 4.2, 0.0, 1.0),
            "oxygenation": np.clip(
                (observed.oxygenation_fraction - 0.25) / 0.65, 0.0, 1.0
            ),
            "doppler": np.clip(
                observed.doppler_speed_m_s / doppler_scale, 0.0, 1.0
            ),
            "ultrasound": np.clip(observed.ultrasound_patency, 0.0, 1.0),
        }
        validity_names = {
            "icg": "nir_icg",
            "speckle": "laser_speckle",
            "thermal": "thermal",
            "oxygenation": "surface_oxygenation",
            "doppler": "doppler",
            "ultrasound": "ultrasound",
        }
        usable = tuple(
            name
            for name in self.WEIGHTS
            if observed.modality_validity.get(validity_names[name], False)
        )
        usable_weight = sum(self.WEIGHTS[name] for name in usable)
        fused = np.zeros_like(observed.flow_index, dtype=float)
        stack = []
        for name in usable:
            normalized_weight = self.WEIGHTS[name] / max(usable_weight, 1.0e-9)
            fused += normalized_weight * arrays[name]
            stack.append(arrays[name])
        # Extravascular tracer is a hazard signal rather than evidence of useful
        # tissue perfusion. Penalize the fused score locally so an active leak
        # cannot receive a better global viability result than an intact state.
        leak_penalty = (
            0.22 * np.clip(observed.icg_extravascular / 0.20, 0.0, 1.0)
            if observed.modality_validity.get("nir_icg", False)
            else np.zeros_like(fused)
        )
        fused = np.clip(fused - leak_penalty, 0.0, 1.0)
        disagreement = (
            np.std(np.stack(stack, axis=0), axis=0)
            if len(stack) >= 2
            else np.ones_like(fused)
        )
        regions = []
        cause, diagnostic_confidence = self._classify(
            observed, icg, usable
        )
        mean_confidence = float(np.mean(observed.confidence))
        abstained = bool(
            usable_weight < 0.50
            or observed.registration_error_m > 0.003
            or observed.timestamp_skew_s > 0.050
            or mean_confidence < 0.30
            or diagnostic_confidence < 0.48
        )
        if abstained:
            cause = "mixed_or_uncertain"
        action={
            "arterial_inflow_obstruction":"remove_or_reposition_occluder_or_clip",
            "venous_outflow_obstruction":"release_venous_compression_or_revise_outflow",
            "anastomotic_stenosis":"revise_anastomosis",
            "active_branch_leak":"control_branch_leak",
            "external_compression":"release_retraction_or_reduce_dressing_pressure",
            "normal_perfusion":"no_action",
            "mixed_or_uncertain":"repeat_scan_and_inspect_sensor_registration",
        }[cause]
        flat=fused.reshape(-1)
        conf=np.asarray(observed.confidence).reshape(-1)
        dis=disagreement.reshape(-1)
        for i,score in enumerate(flat):
            status="viable" if score>=0.68 else "borderline" if score>=0.45 else "nonperfused"
            regions.append(RegionAssessment(i,float(score),float(conf[i]),float(dis[i]),status,cause))
        global_score=float(np.mean(flat)); nonperf=float(np.mean(flat<0.45))
        left=float(np.mean(fused[:,:3])); right=float(np.mean(fused[:,3:])); asym=abs(left-right)
        return PerfusionAssessment(
            condition=scenario_label,
            global_viability_score=global_score,
            nonperfused_fraction=nonperf,
            asymmetry=asym,
            sensor_disagreement=float(np.mean(disagreement)),
            likely_cause=cause,
            recommended_action=action,
            diagnostic_confidence=float(diagnostic_confidence),
            abstained=abstained,
            usable_modalities=tuple(validity_names[name] for name in usable),
            regions=tuple(regions),
        )

    def _classify(
        self,
        maps: MultimodalMaps,
        icg_observation,
        usable: Sequence[str],
    ) -> tuple[str, float]:
        import numpy as np
        if (
            "icg" in usable
            and float(np.max(maps.icg_extravascular)) > 0.05
        ):
            return "active_branch_leak", 0.96
        proxies = []
        if "speckle" in usable:
            proxies.append(np.clip(maps.speckle_perfusion, 0.0, 1.5))
        if "ultrasound" in usable:
            proxies.append(np.clip(maps.ultrasound_patency, 0.0, 1.5))
        if "icg" in usable:
            proxies.append(np.clip(icg_observation, 0.0, 1.5))
        if not proxies:
            return "mixed_or_uncertain", 0.0
        perfusion = np.mean(np.stack(proxies, axis=0), axis=0)
        left = float(np.mean(perfusion[:, :3]))
        right = float(np.mean(perfusion[:, 3:]))
        mean = float(np.mean(perfusion))
        minimum = float(np.min(perfusion))
        ratio = right / max(left, 1.0e-6)
        localized = float(np.mean(perfusion < 0.45))
        oxygenation = (
            float(np.mean(maps.oxygenation_fraction))
            if "oxygenation" in usable
            else 0.0
        )
        if minimum < 0.55 and mean > 0.80 and (
            localized > 0.0 or ratio < 0.92 or ratio > 1.08
        ):
            return "external_compression", 0.84
        if ratio < 0.42 and mean < 0.68 and minimum < 0.36:
            return "arterial_inflow_obstruction", 0.92
        if mean < 0.72 and 0.42 <= ratio < 0.75:
            return "venous_outflow_obstruction", 0.86
        if ratio < 0.72 and right < 0.72 and mean > 0.68:
            return "anastomotic_stenosis", 0.78
        if mean > 0.80 and (
            "oxygenation" not in usable or oxygenation > 0.68
        ):
            return "normal_perfusion", 0.86
        return "mixed_or_uncertain", 0.35


@dataclass(frozen=True)
class InterventionEvidence:
    """Host-reported mechanical evidence for one intervention update."""

    action: str
    elapsed_s: float
    displacement_m: float = 0.0
    contact_force_n: float = 0.0
    lumen_gain_fraction: float = 0.0
    seal_fraction: float = 0.0

    def __post_init__(self) -> None:
        for label in (
            "elapsed_s",
            "displacement_m",
            "contact_force_n",
            "lumen_gain_fraction",
            "seal_fraction",
        ):
            value = _finite(getattr(self, label), label)
            if value < 0.0:
                raise ValueError(f"{label} must be non-negative")
        if self.lumen_gain_fraction > 1.0 or self.seal_fraction > 1.0:
            raise ValueError("fractional intervention evidence cannot exceed one")


@dataclass(frozen=True)
class InterventionUpdate:
    action: str
    recovery_fraction: float
    completed: bool
    accepted: bool
    reason: str


class PerfusionConditionController:
    """Convert measured intervention mechanics into continuous recovery."""

    EXPECTED_ACTION = {
        "arterial_occlusion": "remove_or_reposition_occluder_or_clip",
        "venous_congestion": "release_venous_compression_or_revise_outflow",
        "anastomotic_stenosis": "revise_anastomosis",
        "branch_leak": "control_branch_leak",
        "retraction_ischemia": "release_retraction_or_reduce_dressing_pressure",
        "dressing_compression": "release_retraction_or_reduce_dressing_pressure",
        "healthy": "no_action",
        "recovered": "no_action",
    }

    def __init__(self, condition: str = "healthy"):
        self.condition = _check(condition, VALID_CONDITIONS, "condition")
        self.recovery_fraction = 1.0 if condition == "recovered" else 0.0
        self.history: list[InterventionUpdate] = []

    def update(self, evidence: InterventionEvidence) -> InterventionUpdate:
        expected = self.EXPECTED_ACTION[self.condition]
        if evidence.action in {
            "repeat_scan_and_inspect_sensor_registration",
            "no_action",
        }:
            accepted = evidence.action == expected
            progress = self.recovery_fraction
            reason = "no_mechanical_change"
        elif evidence.action != expected:
            accepted = False
            progress = self.recovery_fraction
            reason = f"expected_{expected}"
        else:
            accepted = True
            if evidence.action == "remove_or_reposition_occluder_or_clip":
                progress = evidence.displacement_m / 0.006
                reason = "occluder_retraction"
            elif evidence.action == "release_venous_compression_or_revise_outflow":
                progress = evidence.displacement_m / 0.005
                reason = "venous_release_travel"
            elif evidence.action == "revise_anastomosis":
                progress = evidence.lumen_gain_fraction
                reason = "measured_lumen_gain"
            elif evidence.action == "control_branch_leak":
                force_window = 0.4 <= evidence.contact_force_n <= 4.0
                dwell = _clamp(evidence.elapsed_s / 2.0)
                progress = min(evidence.seal_fraction, dwell) if force_window else 0.0
                reason = (
                    "seal_contact_and_dwell"
                    if force_window
                    else "seal_contact_force_outside_window"
                )
            else:
                progress = evidence.displacement_m / 0.008
                reason = "external_compression_release_travel"
            progress = max(self.recovery_fraction, _clamp(progress))
        if accepted:
            self.recovery_fraction = progress
        update = InterventionUpdate(
            action=evidence.action,
            recovery_fraction=float(self.recovery_fraction),
            completed=bool(self.recovery_fraction >= 0.95),
            accepted=accepted,
            reason=reason,
        )
        self.history.append(update)
        return update

    def apply(
        self,
        action: str,
        evidence: InterventionEvidence | None = None,
    ) -> InterventionUpdate:
        """Compatibility entry point that refuses evidence-free recovery."""

        if evidence is None:
            raise ValueError(
                "physical intervention evidence is required; "
                "use update(InterventionEvidence(...))"
            )
        if evidence.action != action:
            raise ValueError("action and evidence.action differ")
        return self.update(evidence)


@dataclass(frozen=True)
class ScanResult:
    flow: VascularFlowResult
    time_series: PerfusionTimeSeries
    final_tracer: TracerFrame
    maps: MultimodalMaps
    assessment: PerfusionAssessment
    icg_metrics: Mapping[int, RegionICGMetrics]
    consumable_usage: Mapping[str, float]
    consumable_conservation_error_ml: float


@dataclass(frozen=True)
class RegisteredSensorPacket:
    timestamp_s: float
    camera_frames: Mapping[str, Any]
    depth_frame: Any
    maps: MultimodalMaps
    valid: bool
    errors: tuple[str, ...]


def build_registered_sensor_packet(
    *,
    timestamp_s: float,
    camera_frames: Mapping[str, Any],
    depth_frame: Any,
    maps: MultimodalMaps,
) -> RegisteredSensorPacket:
    """Validate host-rendered frames against the registered model packet."""

    import numpy as np

    timestamp = _finite(timestamp_s, "timestamp_s")
    errors = []
    expected_cameras = {
        "rgb_left_camera",
        "rgb_right_camera",
        "nir_fluorescence_camera",
        "speckle_camera",
        "thermal_camera",
        "multispectral_camera",
    }
    missing = sorted(expected_cameras.difference(camera_frames))
    if missing:
        errors.append("missing_camera_frames:" + ",".join(missing))
    for name, value in camera_frames.items():
        frame = np.asarray(value)
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            errors.append(f"{name}:invalid_shape:{tuple(frame.shape)}")
        elif frame.size == 0 or not np.isfinite(frame).all():
            errors.append(f"{name}:nonfinite_or_empty")
    depth = np.asarray(depth_frame)
    if depth.ndim != 2 or depth.size == 0:
        errors.append(f"depth:invalid_shape:{tuple(depth.shape)}")
    elif not np.isfinite(depth).any():
        errors.append("depth:no_finite_samples")
    if maps.registration_error_m > 0.003:
        errors.append("registration_error_exceeds_3mm")
    if maps.timestamp_skew_s > 0.050:
        errors.append("timestamp_skew_exceeds_50ms")
    return RegisteredSensorPacket(
        timestamp_s=timestamp,
        camera_frames=dict(camera_frames),
        depth_frame=depth_frame,
        maps=maps,
        valid=not errors,
        errors=tuple(errors),
    )


class ClosedLoopPerfusionVerifier:
    def __init__(self, graph: Mapping[str,Any] | None=None):
        self.graph=dict(graph or load_perfusion_graph())
        self.flow_solver=VascularFlowSolver(self.graph)
        self.sensor_model=MultimodalSensorModel(self.graph)
        self.estimator=MultimodalPerfusionEstimator()
        self.healthy_reference=self.flow_solver.solve("healthy")

    def scan(
        self,
        condition: str,
        *,
        duration_s: float = 24.0,
        dt_s: float = 0.10,
        recovery_fraction: float = 0.0,
        consumables: SensorConsumableLedger | None = None,
        operating_state: SensorOperatingState | None = None,
        contrast_per_scan_ml: float = 0.35,
        gel_per_scan_ml: float = 0.50,
    ) -> ScanResult:
        duration = _finite(duration_s, "duration_s")
        dt = _finite(dt_s, "dt_s")
        if duration <= 0.0:
            raise ValueError("duration_s must be positive")
        if dt <= 0.0:
            raise ValueError("dt_s must be positive")
        state = operating_state or SensorOperatingState()
        ledger = consumables or SensorConsumableLedger()
        if state.sensor_state == "fault":
            usage = ledger.consume()
        else:
            usage = ledger.consume(
                contrast_ml=contrast_per_scan_ml,
                gel_ml=gel_per_scan_ml,
            )
        contrast_available = (
            usage["contrast_used_ml"] >= usage["contrast_requested_ml"] > 0.0
        )
        gel_available = usage["gel_used_ml"] >= usage["gel_requested_ml"] > 0.0
        flow=self.flow_solver.solve(
            condition, recovery_fraction=recovery_fraction
        )
        tracer=ICGTracerTransport(
            self.graph, input_scale=1.0 if contrast_available else 0.0
        )
        history=PerfusionTimeSeries(len(self.graph["regions"]))
        frame=TracerFrame(0.0,0.0,{}, {i:0.0 for i in range(24)}, {i:0.0 for i in range(24)})
        steps=max(1,int(math.ceil(duration/dt)))
        for _ in range(steps):
            frame=tracer.step(flow,dt); history.append(frame)
        maps=self.sensor_model.observe(
            flow,
            frame,
            healthy_reference=self.healthy_reference,
            operating_state=state,
            contrast_available=contrast_available,
            gel_available=gel_available,
        )
        metrics={i:history.metrics(i) for i in range(24)}
        assessment=self.estimator.estimate(
            maps, icg_metrics=metrics, scenario_label=condition
        )
        return ScanResult(
            flow=flow,
            time_series=history,
            final_tracer=frame,
            maps=maps,
            assessment=assessment,
            icg_metrics=metrics,
            consumable_usage=usage,
            consumable_conservation_error_ml=ledger.conservation_error_ml,
        )

    @staticmethod
    def _research_evidence_profile(action: str) -> tuple[InterventionEvidence, ...]:
        samples = []
        for index in range(1, 6):
            progress = index / 5.0
            common = {"action": action, "elapsed_s": 0.5 * index}
            if action == "remove_or_reposition_occluder_or_clip":
                common["displacement_m"] = 0.006 * progress
            elif action == "release_venous_compression_or_revise_outflow":
                common["displacement_m"] = 0.005 * progress
            elif action == "revise_anastomosis":
                common["lumen_gain_fraction"] = progress
            elif action == "control_branch_leak":
                common["contact_force_n"] = 1.4
                common["seal_fraction"] = progress
            elif action == "release_retraction_or_reduce_dressing_pressure":
                common["displacement_m"] = 0.008 * progress
            samples.append(InterventionEvidence(**common))
        return tuple(samples)

    def scan_intervene_rescan(
        self,
        condition: str,
        *,
        duration_s: float = 24.0,
        dt_s: float = 0.10,
        intervention_evidence: Sequence[InterventionEvidence] | None = None,
        consumables: SensorConsumableLedger | None = None,
        operating_state: SensorOperatingState | None = None,
    ) -> dict[str,Any]:
        ledger = consumables or SensorConsumableLedger()
        before=self.scan(
            condition,
            duration_s=duration_s,
            dt_s=dt_s,
            consumables=ledger,
            operating_state=operating_state,
        )
        controller=PerfusionConditionController(condition)
        action = before.assessment.recommended_action
        evidence = tuple(
            intervention_evidence
            if intervention_evidence is not None
            else self._research_evidence_profile(action)
        )
        updates = [controller.update(item) for item in evidence]
        after=self.scan(
            condition,
            duration_s=duration_s,
            dt_s=dt_s,
            recovery_fraction=controller.recovery_fraction,
            consumables=ledger,
            operating_state=operating_state,
        )
        return {
            "before":before,
            "action":action,
            "after_condition":(
                "recovered"
                if controller.recovery_fraction >= 0.95
                else condition
            ),
            "after":after,
            "intervention_updates":tuple(updates),
            "recovery_fraction":controller.recovery_fraction,
            "intervention_completed":controller.recovery_fraction >= 0.95,
            "evidence_source":(
                "caller"
                if intervention_evidence is not None
                else "deterministic_research_fixture"
            ),
            "viability_gain":after.assessment.global_viability_score-before.assessment.global_viability_score,
            "nonperfused_fraction_reduction":before.assessment.nonperfused_fraction-after.assessment.nonperfused_fraction,
        }


class PerfusionScanPlanner:
    """Generate registered optical raster and contact-probe waypoints."""
    def optical_raster(self, *, center=(0.0,0.0,0.205), width_m=0.160, depth_m=0.100, rows=5, cols=7, standoff_m=0.090):
        center_values = tuple(float(value) for value in center)
        if len(center_values) != 3 or not all(
            math.isfinite(value) for value in center_values
        ):
            raise ValueError("center must contain three finite values")
        width = _finite(width_m, "width_m")
        depth = _finite(depth_m, "depth_m")
        standoff = _finite(standoff_m, "standoff_m")
        if width <= 0.0 or depth <= 0.0 or standoff <= 0.0:
            raise ValueError("scan dimensions and standoff must be positive")
        if not isinstance(rows, int) or not isinstance(cols, int) or rows < 1 or cols < 1:
            raise ValueError("rows and cols must be positive integers")
        waypoints=[]
        for r in range(rows):
            y=center_values[1]-depth/2+depth*r/max(rows-1,1)
            xs=[
                center_values[0]-width/2+width*c/max(cols-1,1)
                for c in range(cols)
            ]
            if r%2: xs.reverse()
            for x in xs:
                waypoints.append(
                    {
                        "position_m":[x,y,center_values[2]-standoff],
                        "target_m":list(center_values),
                        "mode":"optical",
                    }
                )
        return waypoints

    def contact_probe_waypoints(self, *, modality: str, region_centers: Sequence[Sequence[float]] | None=None, preload_n: float=1.2):
        if modality not in {"ultrasound","doppler"}:
            raise ValueError(modality)
        preload = _finite(preload_n, "preload_n")
        if preload < 0.0:
            raise ValueError("preload_n must be non-negative")
        if region_centers is None:
            graph=load_perfusion_graph(); region_centers=[r["center_m"] for r in graph["regions"]]
        points = [list(map(float, point)) for point in region_centers]
        if any(
            len(point) != 3 or not all(math.isfinite(value) for value in point)
            for point in points
        ):
            raise ValueError("region_centers must contain finite three-dimensional points")
        return [
            {
                "position_m": point,
                "modality": modality,
                "target_preload_n": preload,
            }
            for point in points
        ]


def _joint_targets(**overrides: float) -> dict[str, float]:
    targets = {name: 0.0 for name in TOOL_JOINTS.values()}
    unknown = set(overrides).difference(targets)
    if unknown:
        raise KeyError(f"Unknown perfusion joint targets: {sorted(unknown)}")
    targets.update(
        {name: _finite(value, f"{name}_target") for name, value in overrides.items()}
    )
    return targets


_FUSED_TARGETS = _joint_targets(
    optical_focus_joint=0.010,
    ultrasound_extension_joint=0.035,
    doppler_extension_joint=0.030,
    contact_guard_joint=0.003,
)

PHASE_TARGETS: dict[str, dict[str, float]] = {
    "inspect": _joint_targets(),
    "rgb": _joint_targets(optical_focus_joint=0.008),
    "icg": _joint_targets(
        sensor_turret_joint=-math.radians(22),
        filter_wheel_joint=math.radians(90),
        optical_focus_joint=0.012,
    ),
    "speckle": _joint_targets(
        filter_wheel_joint=math.radians(180),
        speckle_scan_x_joint=math.radians(7),
        speckle_scan_y_joint=-math.radians(6),
    ),
    "thermal": _joint_targets(
        sensor_turret_joint=math.radians(24),
        optical_focus_joint=0.006,
    ),
    "oxygenation": _joint_targets(
        sensor_turret_joint=math.radians(48),
        filter_wheel_joint=math.radians(270),
        optical_focus_joint=0.010,
    ),
    "doppler": _joint_targets(
        doppler_extension_joint=0.046,
        doppler_pitch_joint=math.radians(18),
        contact_guard_joint=0.003,
    ),
    "ultrasound": _joint_targets(
        ultrasound_extension_joint=0.054,
        ultrasound_pitch_joint=-math.radians(12),
        ultrasound_compliance_joint=0.004,
        gel_valve_joint=0.005,
        contact_guard_joint=0.004,
    ),
    "fuse": dict(_FUSED_TARGETS),
    "diagnose": dict(_FUSED_TARGETS),
    "intervene": _joint_targets(contact_guard_joint=0.004),
    "rescan": dict(_FUSED_TARGETS),
    "verify": dict(_FUSED_TARGETS),
}


def phase_targets(phase: str) -> dict[str, float]:
    """Return a complete joint target for every canonical task phase.

    ``fused`` remains an alias for the inspection-export variant name; the task
    contract's canonical controller phase is ``fuse``.
    """

    canonical = "fuse" if phase == "fused" else phase
    try:
        return dict(PHASE_TARGETS[canonical])
    except KeyError as exc:
        raise ValueError(f"Unknown phase {phase!r}") from exc
