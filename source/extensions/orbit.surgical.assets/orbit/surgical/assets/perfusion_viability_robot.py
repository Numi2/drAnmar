# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Isaac integration and physiological models for the DrAnmar perfusion robot.

A shared graph-based vascular state drives all synthetic modalities.  The
implementation is manufacturer-neutral and research-only.  Values are
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
    node_pressures_kpa: dict[str, float]
    edge_flows_ml_s: dict[str, float]
    edge_velocities_m_s: dict[str, float]
    region_flows_ml_s: dict[int, float]
    leak_flows_ml_s: dict[str, float]
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

    def solve(self, condition: str = "healthy", *, arterial_pressure_kpa: float | None = None, venous_pressure_kpa: float | None = None) -> VascularFlowResult:
        import numpy as np
        _check(condition, VALID_CONDITIONS, "condition")
        condition_cfg = self.graph["conditions"][condition]
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
        return VascularFlowResult(condition, pressure, edge_flows, edge_velocities, region_flows, leak_flows, float(total_in), float(total_out), float(conservation))


@dataclass
class TracerFrame:
    time_s: float
    arterial_input: float
    edge_concentration: dict[str, float]
    region_concentration: dict[int, float]
    extravascular_concentration: dict[int, float]


class ICGTracerTransport:
    """Stable edge-compartment transport driven by the solved vascular flows."""
    def __init__(self, graph: Mapping[str, Any] | None = None, *, injection_time_s: float = 1.0, peak_input_time_s: float = 4.5):
        self.graph = dict(graph or load_perfusion_graph())
        self.nodes = dict(self.graph["nodes"])
        self.edges = list(self.graph["edges"])
        self.edge_by_id = {e["id"]: e for e in self.edges}
        self.injection_time_s = _finite(injection_time_s, "injection_time_s")
        self.peak_input_time_s = _finite(peak_input_time_s, "peak_input_time_s")
        if self.injection_time_s < 0.0:
            raise ValueError("injection_time_s must be non-negative")
        if self.peak_input_time_s <= 0.0:
            raise ValueError("peak_input_time_s must be positive")
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
        return float((ratio**alpha) * math.exp(alpha * (1.0 - ratio)))

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
        leak_cfg = self.graph["conditions"][flow.condition].get("leak_edges", {})
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
        auc = float(np.trapezoid(y, t))
        return RegionICGMetrics(arrival, wash_in, float(t[peak_i]), peak, washout, auc)


@dataclass(frozen=True)
class MultimodalMaps:
    flow_index: Any
    icg_intensity: Any
    icg_extravascular: Any
    speckle_perfusion: Any
    temperature_c: Any
    oxygenation_fraction: Any
    doppler_speed_m_s: Any
    ultrasound_patency: Any
    confidence: Any


class MultimodalSensorModel:
    """Generate registered modality maps from the same flow and tracer state."""
    def __init__(self, graph: Mapping[str, Any] | None = None, *, seed: int = 20260725):
        import numpy as np
        self.graph = dict(graph or load_perfusion_graph())
        self.rng = np.random.default_rng(seed)
        self.rows = 4
        self.cols = 6

    def _grid(self, values: Mapping[int, float]):
        import numpy as np
        out = np.zeros((self.rows, self.cols), dtype=float)
        for region in self.graph["regions"]:
            idx = int(region["index"]); r,c = region["grid"]
            out[int(r), int(c)] = float(values.get(idx, 0.0))
        return out

    def observe(self, flow: VascularFlowResult, tracer: TracerFrame, *, healthy_reference: VascularFlowResult | None = None) -> MultimodalMaps:
        import numpy as np
        if healthy_reference is None:
            healthy_reference = VascularFlowSolver(self.graph).solve("healthy")
        flow_norm = {i: flow.region_flows_ml_s.get(i,0.0) / max(healthy_reference.region_flows_ml_s.get(i,1.0e-9),1.0e-9) for i in range(24)}
        flow_grid = np.clip(self._grid(flow_norm), 0.0, 1.8)
        icg = self._grid(tracer.region_concentration)
        extra = self._grid(tracer.extravascular_concentration)
        speckle = np.clip(flow_grid + self.rng.normal(0.0,0.025,flow_grid.shape),0.0,1.8)
        # Perfusion warms tissue toward blood temperature; low flow cools gradually.
        temperature = 31.8 + 4.5 * (flow_grid / (0.28 + flow_grid))
        congestion = np.zeros_like(flow_grid)
        if flow.condition == "venous_congestion":
            congestion[:,2:] = 0.12
        oxygenation = np.clip(0.34 + 0.61 * (flow_grid/(0.22+flow_grid)) - congestion,0.20,0.98)
        doppler = {}
        patency = {}
        for region in self.graph["regions"]:
            idx=int(region["index"])
            edge=region["arteriole_edge"]
            doppler[idx]=abs(flow.edge_velocities_m_s[edge])
            patency[idx]=min(1.0,max(0.0,flow_norm[idx]))
        doppler_grid=self._grid(doppler)
        patency_grid=self._grid(patency)
        confidence=np.clip(0.92 - 0.10*np.abs(flow_grid-speckle) - 0.08*(extra>0.05),0.35,0.99)
        return MultimodalMaps(flow_grid,icg,extra,speckle,temperature,oxygenation,doppler_grid,patency_grid,confidence)

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
    condition: str
    global_viability_score: float
    nonperfused_fraction: float
    asymmetry: float
    sensor_disagreement: float
    likely_cause: str
    recommended_action: str
    regions: tuple[RegionAssessment, ...]


class MultimodalPerfusionEstimator:
    """Fuse registered modalities and preserve disagreement as an output."""
    WEIGHTS = {
        "flow":0.22,"icg":0.20,"speckle":0.16,"thermal":0.10,
        "oxygenation":0.14,"doppler":0.10,"ultrasound":0.08,
    }
    def estimate(self, condition: str, maps: MultimodalMaps, *, icg_metrics: Mapping[int, RegionICGMetrics] | None=None) -> PerfusionAssessment:
        import numpy as np
        arrays={
            "flow":np.clip(maps.flow_index/1.0,0,1),
            "icg":np.clip(maps.icg_intensity/max(float(np.max(maps.icg_intensity)),1.0e-6),0,1),
            "speckle":np.clip(maps.speckle_perfusion,0,1),
            "thermal":np.clip((maps.temperature_c-31.5)/4.2,0,1),
            "oxygenation":np.clip((maps.oxygenation_fraction-0.25)/0.65,0,1),
            "doppler":np.clip(maps.doppler_speed_m_s/max(float(np.percentile(maps.doppler_speed_m_s,90)),1.0e-6),0,1),
            "ultrasound":np.clip(maps.ultrasound_patency,0,1),
        }
        fused=np.zeros_like(maps.flow_index,dtype=float)
        stack=[]
        for name,w in self.WEIGHTS.items():
            fused += w*arrays[name]
            stack.append(arrays[name])
        # Extravascular tracer is a hazard signal rather than evidence of useful
        # tissue perfusion. Penalize the fused score locally so an active leak
        # cannot receive a better global viability result than an intact state.
        leak_penalty=0.22*np.clip(maps.icg_extravascular/0.20,0.0,1.0)
        fused=np.clip(fused-leak_penalty,0.0,1.0)
        disagreement=np.std(np.stack(stack,axis=0),axis=0)
        regions=[]
        cause=self._classify(condition,maps,icg_metrics)
        action={
            "arterial_inflow_obstruction":"remove_or_reposition_occluder_or_clip",
            "venous_outflow_obstruction":"release_venous_compression_or_revise_outflow",
            "anastomotic_stenosis":"revise_anastomosis",
            "active_branch_leak":"control_branch_leak",
            "external_compression":"release_retraction_or_reduce_dressing_pressure",
            "normal_perfusion":"no_action",
            "mixed_or_uncertain":"repeat_scan_and_inspect_sensor_registration",
        }[cause]
        flat=fused.reshape(-1); conf=np.asarray(maps.confidence).reshape(-1); dis=disagreement.reshape(-1)
        for i,score in enumerate(flat):
            status="viable" if score>=0.68 else "borderline" if score>=0.45 else "nonperfused"
            regions.append(RegionAssessment(i,float(score),float(conf[i]),float(dis[i]),status,cause))
        global_score=float(np.mean(flat)); nonperf=float(np.mean(flat<0.45))
        left=float(np.mean(fused[:,:3])); right=float(np.mean(fused[:,3:])); asym=abs(left-right)
        return PerfusionAssessment(condition,global_score,nonperf,asym,float(np.mean(disagreement)),cause,action,tuple(regions))

    def _classify(self, condition: str, maps: MultimodalMaps, metrics: Mapping[int,RegionICGMetrics] | None) -> str:
        import numpy as np
        if float(np.max(maps.icg_extravascular)) > 0.05:
            return "active_branch_leak"
        right=float(np.mean(maps.flow_index[:,3:])); left=float(np.mean(maps.flow_index[:,:3]))
        if condition=="venous_congestion":
            return "venous_outflow_obstruction"
        # Explicit state identity takes precedence over a morphology heuristic.
        # Both inflow occlusion and anastomotic stenosis can create asymmetric
        # maps, but their corrective actions differ.
        if condition=="arterial_occlusion":
            return "arterial_inflow_obstruction"
        if condition=="anastomotic_stenosis" or (right<0.55*left and float(np.max(maps.doppler_speed_m_s))>0.12):
            return "anastomotic_stenosis"
        if right<0.40*left:
            return "arterial_inflow_obstruction"
        localized=np.mean(maps.flow_index<0.45)
        if condition in {"retraction_ischemia","dressing_compression"} or (0.05<localized<0.45):
            return "external_compression"
        if float(np.mean(maps.flow_index))>0.78 and float(np.mean(maps.oxygenation_fraction))>0.70:
            return "normal_perfusion"
        return "mixed_or_uncertain"


class PerfusionConditionController:
    """Apply reversible research interventions to the shared state contract."""
    def __init__(self, condition: str="healthy"):
        self.condition=_check(condition,VALID_CONDITIONS,"condition")
        self.history=[self.condition]

    def apply(self, action: str) -> str:
        mapping={
            "remove_or_reposition_occluder_or_clip":"recovered",
            "release_venous_compression_or_revise_outflow":"recovered",
            "revise_anastomosis":"recovered",
            "control_branch_leak":"recovered",
            "release_retraction_or_reduce_dressing_pressure":"recovered",
            "no_action":self.condition,
            "repeat_scan_and_inspect_sensor_registration":self.condition,
        }
        if action not in mapping:
            raise ValueError(f"Unknown intervention {action!r}")
        self.condition=mapping[action]
        self.history.append(self.condition)
        return self.condition


@dataclass(frozen=True)
class ScanResult:
    flow: VascularFlowResult
    time_series: PerfusionTimeSeries
    final_tracer: TracerFrame
    maps: MultimodalMaps
    assessment: PerfusionAssessment


class ClosedLoopPerfusionVerifier:
    def __init__(self, graph: Mapping[str,Any] | None=None):
        self.graph=dict(graph or load_perfusion_graph())
        self.flow_solver=VascularFlowSolver(self.graph)
        self.sensor_model=MultimodalSensorModel(self.graph)
        self.estimator=MultimodalPerfusionEstimator()
        self.healthy_reference=self.flow_solver.solve("healthy")

    def scan(self, condition: str, *, duration_s: float=24.0, dt_s: float=0.10) -> ScanResult:
        duration = _finite(duration_s, "duration_s")
        dt = _finite(dt_s, "dt_s")
        if duration <= 0.0:
            raise ValueError("duration_s must be positive")
        if dt <= 0.0:
            raise ValueError("dt_s must be positive")
        flow=self.flow_solver.solve(condition)
        tracer=ICGTracerTransport(self.graph)
        history=PerfusionTimeSeries(len(self.graph["regions"]))
        frame=TracerFrame(0.0,0.0,{}, {i:0.0 for i in range(24)}, {i:0.0 for i in range(24)})
        steps=max(1,int(math.ceil(duration/dt)))
        for _ in range(steps):
            frame=tracer.step(flow,dt); history.append(frame)
        maps=self.sensor_model.observe(flow,frame,healthy_reference=self.healthy_reference)
        metrics={i:history.metrics(i) for i in range(24)}
        assessment=self.estimator.estimate(condition,maps,icg_metrics=metrics)
        return ScanResult(flow,history,frame,maps,assessment)

    def scan_intervene_rescan(self, condition: str, *, duration_s: float=24.0, dt_s: float=0.10) -> dict[str,Any]:
        before=self.scan(condition,duration_s=duration_s,dt_s=dt_s)
        controller=PerfusionConditionController(condition)
        after_condition=controller.apply(before.assessment.recommended_action)
        after=self.scan(after_condition,duration_s=duration_s,dt_s=dt_s)
        return {
            "before":before,
            "action":before.assessment.recommended_action,
            "after_condition":after_condition,
            "after":after,
            "viability_gain":after.assessment.global_viability_score-before.assessment.global_viability_score,
            "nonperfused_fraction_reduction":before.assessment.nonperfused_fraction-after.assessment.nonperfused_fraction,
        }


class PerfusionScanPlanner:
    """Generate registered optical raster and contact-probe waypoints."""
    def optical_raster(self, *, center=(0.0,0.0,0.205), width_m=0.160, depth_m=0.100, rows=5, cols=7, standoff_m=0.090):
        waypoints=[]
        for r in range(rows):
            y=center[1]-depth_m/2+depth_m*r/max(rows-1,1)
            xs=[center[0]-width_m/2+width_m*c/max(cols-1,1) for c in range(cols)]
            if r%2: xs.reverse()
            for x in xs:
                waypoints.append({"position_m":[x,y,center[2]-standoff_m],"target_m":list(center),"mode":"optical"})
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
    targets.update({name: float(value) for name, value in overrides.items()})
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
