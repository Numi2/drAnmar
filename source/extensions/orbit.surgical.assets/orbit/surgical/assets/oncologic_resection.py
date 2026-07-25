# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""DrAnmar surgical-oncology assets and executable research mechanics.

The OpenUSD layers provide the articulated tool, liver/tumor substrate,
specimen system, semantic frames, and a three-station workcell.  This module
adds the part that a visual asset package cannot provide by itself:

* Isaac Lab configuration factories for standalone and Franka-mounted use;
* registered multimodal fusion with disagreement and stale-data abstention;
* discrete resection bonds with a fail-closed pedicle-seal interlock;
* volumetric tumor, healthy-tissue, and margin accounting;
* specimen containment and orientation state;
* a deterministic procedure controller and RL-facing observations/reward.

All tissue, sensing, energy, and task thresholds are provisional research
parameters.  Native simulation qualification is not clinical validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Mapping, Sequence


CATALOG_SUBPATH = "Props/SurgicalOncology/OncoSurgeryCell"
ASSET_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
ROOT = ASSET_DATA_ROOT / CATALOG_SUBPATH

TOOL_STANDALONE_USD = ROOT / "dranmar_oncosurgery_tool.usda"
TOOL_PAYLOAD_USD = ROOT / "dranmar_tumor_resection_tool_payload.usda"
TOOL_RIGID_PROXY_USD = ROOT / "dranmar_tumor_resection_tool_rigid_proxy.usda"
LIVER_DEMO_USD = ROOT / "dranmar_oncology_liver.usda"
SPECIMEN_BAG_USD = ROOT / "dranmar_specimen_bag.usda"
WORKCELL_USD = ROOT / "dranmar_oncosurgery_workcell.usda"
RESECTION_CELL_USD = ROOT / "dranmar_resection_cell.usda"
MARGIN_MARKER_USD = ROOT / "dranmar_margin_marker.usda"
MARGIN_INK_PARTICLE_USD = ROOT / "dranmar_margin_ink_particle.usda"
TUMOR_TRACER_PARTICLE_USD = ROOT / "dranmar_tumor_tracer_particle.usda"
TUMOR_FIELD_JSON = ROOT / "tumor_field.json"
RESECTION_TOPOLOGY_JSON = ROOT / "resection_topology.json"
TASK_CONTRACT_JSON = ROOT / "oncologic_resection_task_contract.json"
SENSOR_MODALITIES_JSON = ROOT / "sensor_modalities.json"
INTERACTION_FRAMES_JSON = ROOT / "interaction_frames.json"
DYNAMIC_PATIENT_ROOT = (
    ASSET_DATA_ROOT / "Props/Patients/DynamicAbdominalPatient"
)
DYNAMIC_PATIENT_USD = (
    DYNAMIC_PATIENT_ROOT / "dranmar_dynamic_abdominal_patient.usda"
)
DYNAMIC_PATIENT_LIVER_USD = (
    DYNAMIC_PATIENT_ROOT / "anatomy/dranmar_liver.usda"
)
DYNAMIC_PATIENT_RUNTIME = Path(__file__).with_name(
    "dynamic_abdominal_patient.py"
)

VALID_INSTRUMENT_STATES = frozenset({"ready", "spent"})
VALID_BAG_STATES = frozenset({"loaded", "deployed", "closed", "empty"})
VALID_TRACER_STATES = frozenset({"full", "partial", "empty"})
VALID_COLLECTION_STATES = frozenset({"empty", "partial", "full"})
VALID_LIVER_STATES = frozenset(
    {"initial", "mapped", "planned", "resected", "corrected"}
)
VALID_PATHOLOGY_STATES = frozenset({"solitary", "multifocal"})

TOOL_JOINTS = {
    "sensor_turret": "sensor_turret_joint",
    "hsi_filter_wheel": "hsi_filter_wheel_joint",
    "oct_scan_x": "oct_scan_x_joint",
    "oct_scan_y": "oct_scan_y_joint",
    "ultrasound_extension": "ultrasound_extension_joint",
    "ultrasound_pitch": "ultrasound_pitch_joint",
    "ultrasound_compliance": "ultrasound_compliance_joint",
    "raman_extension": "raman_extension_joint",
    "left_traction": "left_traction_joint",
    "right_traction": "right_traction_joint",
    "aspirator_extension": "aspirator_extension_joint",
    "aspirator_vibration": "aspirator_vibration_joint",
    "bipolar_left_jaw": "bipolar_left_jaw_joint",
    "bipolar_right_jaw": "bipolar_right_jaw_joint",
    "scissor_extension": "scissor_extension_joint",
    "scissor_guard": "scissor_guard_joint",
    "scissor_blade": "scissor_blade_joint",
    "bag_deployment": "bag_deployment_joint",
    "bag_closure": "bag_closure_joint",
    "margin_marker": "margin_marker_joint",
    "suction_valve": "suction_valve_joint",
    "irrigation_valve": "irrigation_valve_joint",
}

TOOL_FRAME_PATHS = {
    "franka_mount": "Links/Mount/Frames/franka_mount",
    "tool_center": "Links/Mount/Frames/tool_center",
    "rgb_camera_left": "Links/SensorTurret/Frames/rgb_camera_left",
    "rgb_camera_right": "Links/SensorTurret/Frames/rgb_camera_right",
    "nir_fluorescence_camera": (
        "Links/SensorTurret/Frames/nir_fluorescence_camera"
    ),
    "hsi_camera": "Links/SensorTurret/Frames/hsi_camera",
    "tumor_mapping_tcp": "Links/SensorTurret/Frames/tumor_mapping_tcp",
    "oct_beam_origin": "Links/OCTScannerY/Frames/oct_beam_origin",
    "ultrasound_probe_face": (
        "Links/UltrasoundCompliance/Frames/ultrasound_probe_face"
    ),
    "ultrasound_acoustic_axis": (
        "Links/UltrasoundCompliance/Frames/ultrasound_acoustic_axis"
    ),
    "raman_contact": "Links/RamanProbe/Frames/raman_contact",
    "left_traction_contact": (
        "Links/LeftTraction/Frames/left_traction_contact"
    ),
    "right_traction_contact": (
        "Links/RightTraction/Frames/right_traction_contact"
    ),
    "aspirator_tip": "Links/AspiratorRotor/Frames/aspirator_tip",
    "resection_tcp": "Links/AspiratorRotor/Frames/resection_tcp",
    "suction_center": "Links/AspiratorRotor/Frames/suction_center",
    "bipolar_left_contact": (
        "Links/BipolarLeftJaw/Frames/bipolar_left_contact"
    ),
    "bipolar_right_contact": (
        "Links/BipolarRightJaw/Frames/bipolar_right_contact"
    ),
    "bipolar_seal_center": (
        "Links/BipolarRightJaw/Frames/bipolar_seal_center"
    ),
    "scissor_guard_tip": "Links/ScissorGuard/Frames/scissor_guard_tip",
    "scissor_cut_plane": "Links/ScissorBlade/Frames/scissor_cut_plane",
    "specimen_bag_center": (
        "Links/BagDeployer/Frames/specimen_bag_center"
    ),
    "bag_closure_reference": (
        "Links/BagClosure/Frames/bag_closure_reference"
    ),
    "margin_marker_tip": "Links/MarginMarker/Frames/margin_marker_tip",
    "irrigation_center": "Links/IrrigationValve/Frames/irrigation_center",
    "cavity_scan_reference": (
        "Links/IrrigationValve/Frames/cavity_scan_reference"
    ),
}
REGISTERED_CAMERA_FRAMES = (
    "rgb_camera_left",
    "rgb_camera_right",
    "nir_fluorescence_camera",
    "hsi_camera",
)

TASK_PHASES = (
    "inspect",
    "register",
    "map",
    "plan",
    "capture",
    "resect_parenchyma",
    "manage_pedicles",
    "complete_boundary",
    "deploy_bag",
    "capture_specimen",
    "close_bag",
    "mark_orientation",
    "cavity_scan",
    "corrective_resection",
    "hemostasis_and_bile_check",
    "final_margin_report",
)


def _phase_targets(**updates: float) -> dict[str, float]:
    targets = {joint: 0.0 for joint in TOOL_JOINTS.values()}
    targets.update(updates)
    return targets


PHASE_TARGETS = {
    "inspect": _phase_targets(),
    "register": _phase_targets(
        sensor_turret_joint=math.radians(-20.0),
        ultrasound_extension_joint=0.025,
    ),
    "map": _phase_targets(
        sensor_turret_joint=math.radians(20.0),
        hsi_filter_wheel_joint=math.radians(90.0),
        oct_scan_x_joint=0.010,
        oct_scan_y_joint=-0.010,
        ultrasound_extension_joint=0.045,
        ultrasound_compliance_joint=0.004,
        raman_extension_joint=0.025,
    ),
    "plan": _phase_targets(
        sensor_turret_joint=0.0,
        ultrasound_extension_joint=0.035,
    ),
    "capture": _phase_targets(
        left_traction_joint=-0.025,
        right_traction_joint=0.025,
    ),
    "resect_parenchyma": _phase_targets(
        left_traction_joint=-0.020,
        right_traction_joint=0.020,
        aspirator_extension_joint=0.060,
        suction_valve_joint=0.007,
        irrigation_valve_joint=0.003,
    ),
    "manage_pedicles": _phase_targets(
        left_traction_joint=-0.018,
        right_traction_joint=0.018,
        aspirator_extension_joint=0.050,
        bipolar_left_jaw_joint=math.radians(24.0),
        bipolar_right_jaw_joint=math.radians(-24.0),
    ),
    "complete_boundary": _phase_targets(
        scissor_extension_joint=0.050,
        scissor_guard_joint=0.010,
        scissor_blade_joint=math.radians(28.0),
        suction_valve_joint=0.006,
    ),
    "deploy_bag": _phase_targets(bag_deployment_joint=0.090),
    "capture_specimen": _phase_targets(bag_deployment_joint=0.090),
    "close_bag": _phase_targets(
        bag_deployment_joint=0.090,
        bag_closure_joint=0.023,
    ),
    "mark_orientation": _phase_targets(
        bag_deployment_joint=0.070,
        bag_closure_joint=0.023,
        margin_marker_joint=0.050,
    ),
    "cavity_scan": _phase_targets(
        sensor_turret_joint=math.radians(15.0),
        oct_scan_x_joint=-0.010,
        oct_scan_y_joint=0.010,
        ultrasound_extension_joint=0.040,
        irrigation_valve_joint=0.004,
    ),
    "corrective_resection": _phase_targets(
        aspirator_extension_joint=0.055,
        suction_valve_joint=0.007,
    ),
    "hemostasis_and_bile_check": _phase_targets(
        sensor_turret_joint=0.0,
        ultrasound_extension_joint=0.025,
        irrigation_valve_joint=0.006,
    ),
    "final_margin_report": _phase_targets(),
}


def phase_targets(phase: str) -> dict[str, float]:
    """Return a copy of the complete 22-joint target vector for ``phase``."""

    try:
        return dict(PHASE_TARGETS[phase])
    except KeyError as exc:
        raise KeyError(f"Unknown oncology phase {phase!r}") from exc


def frame_path(tool_path: str, name: str) -> str:
    try:
        suffix = TOOL_FRAME_PATHS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown oncology frame {name!r}") from exc
    return f"{tool_path.rstrip('/')}/{suffix}"


def tensor_value(value: Any) -> Any:
    candidate = value.torch if hasattr(value, "torch") else value
    return candidate() if callable(candidate) else candidate


def _check_state(value: str, allowed: frozenset[str], label: str) -> str:
    if value not in allowed:
        raise ValueError(
            f"Unsupported {label}={value!r}; expected one of {sorted(allowed)}"
        )
    return value


def _wxyz_quaternion(
    orientation_wxyz: Sequence[float],
) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in orientation_wxyz)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("orientation_wxyz must contain four finite values")
    norm = math.sqrt(sum(value * value for value in values))
    if abs(norm - 1.0) > 1.0e-4:
        raise ValueError("orientation_wxyz must be a unit quaternion")
    return values


def _load_dynamic_patient_runtime() -> Any:
    """Load the sibling patient runtime without relying on extension order."""

    module_name = "dranmar_oncology_dynamic_patient_runtime"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(
        module_name, DYNAMIC_PATIENT_RUNTIME
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Unable to load Dynamic Patient runtime from "
            f"{DYNAMIC_PATIENT_RUNTIME}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _tool_variants(
    instrument_state: str,
    bag_state: str,
    tracer_state: str,
    collection_state: str,
) -> dict[str, str]:
    return {
        "instrument_state": _check_state(
            instrument_state, VALID_INSTRUMENT_STATES, "instrument_state"
        ),
        "bag_state": _check_state(bag_state, VALID_BAG_STATES, "bag_state"),
        "tracer_state": _check_state(
            tracer_state, VALID_TRACER_STATES, "tracer_state"
        ),
        "collection_state": _check_state(
            collection_state, VALID_COLLECTION_STATES, "collection_state"
        ),
    }


def make_tool_cfg(
    prim_path: str = "/World/DrAnmarTumorResectionTool",
    *,
    instrument_state: str = "ready",
    bag_state: str = "loaded",
    tracer_state: str = "full",
    collection_state: str = "empty",
    position: Sequence[float] = (0.0, 0.0, 0.35),
    orientation_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
):
    """Create a standalone Isaac Lab articulation configuration."""

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    variants = _tool_variants(
        instrument_state, bag_state, tracer_state, collection_state
    )
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(TOOL_STANDALONE_USD),
            variants=variants,
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=20,
                solver_velocity_iteration_count=6,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(float(value) for value in position),
            rot=_wxyz_quaternion(orientation_wxyz),
            joint_pos=phase_targets("inspect"),
        ),
        actuators={
            "oncology_imaging": ImplicitActuatorCfg(
                joint_names_expr=[
                    "sensor_turret_joint",
                    "hsi_filter_wheel_joint",
                    "oct_scan_.*_joint",
                    "ultrasound_.*_joint",
                    "raman_extension_joint",
                ],
                effort_limit_sim=95.0,
                velocity_limit_sim=1.5,
                stiffness=3600.0,
                damping=120.0,
            ),
            "oncology_traction": ImplicitActuatorCfg(
                joint_names_expr=[".*_traction_joint"],
                effort_limit_sim=120.0,
                velocity_limit_sim=0.18,
                stiffness=4800.0,
                damping=165.0,
            ),
            "oncology_resection": ImplicitActuatorCfg(
                joint_names_expr=[
                    "aspirator_.*_joint",
                    "bipolar_.*_jaw_joint",
                    "scissor_.*_joint",
                ],
                effort_limit_sim=140.0,
                velocity_limit_sim=1.5,
                stiffness=6000.0,
                damping=180.0,
            ),
            "oncology_specimen": ImplicitActuatorCfg(
                joint_names_expr=[
                    "bag_.*_joint",
                    "margin_marker_joint",
                ],
                effort_limit_sim=70.0,
                velocity_limit_sim=0.30,
                stiffness=3600.0,
                damping=110.0,
            ),
            "oncology_valves": ImplicitActuatorCfg(
                joint_names_expr=[".*_valve_joint"],
                effort_limit_sim=30.0,
                velocity_limit_sim=0.25,
                stiffness=1800.0,
                damping=55.0,
            ),
        },
    )


def make_rigid_proxy_cfg(
    prim_path: str = "/World/DrAnmarTumorResectionToolProxy",
    *,
    position: Sequence[float] = (0.0, 0.0, 0.35),
    orientation_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
):
    """Create the lower-cost rigid perception/contact representation."""

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(TOOL_RIGID_PROXY_USD),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=tuple(float(value) for value in position),
            rot=_wxyz_quaternion(orientation_wxyz),
        ),
    )


def make_liver_demo_cfg(
    prim_path: str = "/World/DrAnmarOncologyLiver",
    *,
    procedure_state: str = "initial",
    pathology_state: str = "multifocal",
    position: Sequence[float] = (0.0, 0.0, 0.0),
):
    """Create the task-state liver substrate.

    The authored liver is a registered topology/visual substrate, not a
    validated volumetric constitutive model.  Use the dynamic-patient binding
    contract below when a deformable whole-patient scene is required.
    """

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    variants = {
        "procedure_state": _check_state(
            procedure_state, VALID_LIVER_STATES, "procedure_state"
        ),
        "pathology_state": _check_state(
            pathology_state, VALID_PATHOLOGY_STATES, "pathology_state"
        ),
    }
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(LIVER_DEMO_USD),
            variants=variants,
            activate_contact_sensors=True,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=tuple(float(value) for value in position)
        ),
    )


def make_specimen_bag_cfg(
    prim_path: str = "/World/DrAnmarSpecimenBag",
    *,
    state: str = "open",
    position: Sequence[float] = (0.0, 0.0, 0.10),
):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    if state not in {"open", "closed"}:
        raise ValueError("state must be 'open' or 'closed'")
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(SPECIMEN_BAG_USD),
            variants={"state": state},
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=tuple(float(value) for value in position)
        ),
    )


def make_workcell_cfg(
    prim_path: str = "/World/DrAnmarOncoSurgeryCell",
    *,
    position: Sequence[float] = (0.0, 0.0, 0.0),
):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(usd_path=str(WORKCELL_USD)),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=tuple(float(value) for value in position)
        ),
    )


def _spawn_single_franka_with_tool(
    prim_path: str,
    cfg: Any,
    translation: Sequence[float] | None = None,
    orientation: Sequence[float] | None = None,
    **_: Any,
):
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    from isaaclab.sim.utils import create_prim, get_current_stage, select_usd_variants
    from pxr import Gf, Sdf, UsdPhysics

    robot = spawn_from_usd(prim_path, cfg, translation, orientation)
    stage = get_current_stage()
    robot_path = Sdf.Path(prim_path)
    stock_names = {
        "panda_hand_joint",
        "panda_hand",
        "panda_finger_joint1",
        "panda_finger_joint2",
        "panda_leftfinger",
        "panda_rightfinger",
    }
    hand_joints = [
        prim
        for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(robot_path)
        and prim.GetName() == "panda_hand_joint"
    ]
    if len(hand_joints) == 1:
        stock_joint = UsdPhysics.Joint(hand_joints[0])
        body0 = stock_joint.GetBody0Rel().GetTargets()
        local_pos = stock_joint.GetLocalPos0Attr().Get() or Gf.Vec3f(0, 0, 0)
        local_rot = stock_joint.GetLocalRot0Attr().Get() or Gf.Quatf(1, 0, 0, 0)
    else:
        body0 = [
            prim.GetPath()
            for prim in stage.Traverse()
            if prim.GetPath().HasPrefix(robot_path)
            and prim.GetName() == "panda_link8"
        ]
        local_pos = Gf.Vec3f(0, 0, 0)
        half_angle = math.radians(-45.0) / 2.0
        local_rot = Gf.Quatf(
            math.cos(half_angle), 0, 0, math.sin(half_angle)
        )
    if len(body0) != 1 or not stage.GetPrimAtPath(body0[0]).IsValid():
        raise RuntimeError(f"Could not resolve one Franka link8 mount: {body0}")

    candidates = [
        prim.GetPath()
        for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(robot_path)
        and prim.GetName() in stock_names
    ]
    inactive: list[Any] = []
    for path in sorted(candidates, key=lambda item: str(item).count("/")):
        if not any(path.HasPrefix(parent) for parent in inactive):
            inactive.append(path)
    for path in inactive:
        stage.OverridePrim(path).SetActive(False)

    tool_path = f"{prim_path}/DrAnmarTumorResectionTool"
    create_prim(tool_path, usd_path=str(TOOL_PAYLOAD_USD), stage=stage)
    select_usd_variants(
        tool_path,
        _tool_variants(
            cfg.instrument_state,
            cfg.bag_state,
            cfg.tracer_state,
            cfg.collection_state,
        ),
    )
    joint = UsdPhysics.FixedJoint.Define(
        stage, f"{prim_path}/dranmar_oncology_mount_joint"
    )
    joint.CreateBody0Rel().SetTargets(body0)
    joint.CreateBody1Rel().SetTargets(
        [Sdf.Path(f"{tool_path}/Links/Mount")]
    )
    joint.CreateLocalPos0Attr().Set(local_pos)
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot0Attr().Set(local_rot)
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    return robot


def spawn_franka_with_oncology_tool(
    prim_path: str,
    cfg: Any,
    translation: Sequence[float] | None = None,
    orientation: Sequence[float] | None = None,
    **kwargs: Any,
):
    from isaaclab.sim.utils import clone

    return clone(_spawn_single_franka_with_tool)(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )


def make_franka_oncology_robot_cfg(
    *,
    prim_path: str = "/World/Robot",
    instrument_state: str = "ready",
    bag_state: str = "loaded",
    tracer_state: str = "full",
    collection_state: str = "empty",
):
    """Create a Franka configuration with the stock hand replaced by the tool."""

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

    variants = _tool_variants(
        instrument_state, bag_state, tracer_state, collection_state
    )

    @configclass
    class FrankaOncologyUsdCfg(sim_utils.UsdFileCfg):
        instrument_state: str = "ready"
        bag_state: str = "loaded"
        tracer_state: str = "full"
        collection_state: str = "empty"
        func = spawn_franka_with_oncology_tool

    cfg = FRANKA_PANDA_CFG.copy()
    cfg.prim_path = prim_path
    cfg.spawn = FrankaOncologyUsdCfg(
        usd_path=(
            f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/"
            "FrankaPanda/franka.usd"
        ),
        variants={"Gripper": "Default", "Mesh": "Performance"},
        instrument_state=variants["instrument_state"],
        bag_state=variants["bag_state"],
        tracer_state=variants["tracer_state"],
        collection_state=variants["collection_state"],
        activate_contact_sensors=True,
        rigid_props=FRANKA_PANDA_CFG.spawn.rigid_props,
        articulation_props=FRANKA_PANDA_CFG.spawn.articulation_props,
    )
    cfg.init_state.joint_pos = {
        key: value
        for key, value in cfg.init_state.joint_pos.items()
        if "finger" not in key
    }
    cfg.init_state.joint_pos.update(phase_targets("inspect"))
    cfg.actuators = {
        key: value
        for key, value in cfg.actuators.items()
        if key != "panda_hand"
    }
    cfg.actuators.update(
        {
            "oncology_imaging": ImplicitActuatorCfg(
                joint_names_expr=[
                    "sensor_turret_joint",
                    "hsi_filter_wheel_joint",
                    "oct_scan_.*_joint",
                    "ultrasound_.*_joint",
                    "raman_extension_joint",
                ],
                effort_limit_sim=95.0,
                velocity_limit_sim=1.5,
                stiffness=3600.0,
                damping=120.0,
            ),
            "oncology_traction": ImplicitActuatorCfg(
                joint_names_expr=[".*_traction_joint"],
                effort_limit_sim=120.0,
                velocity_limit_sim=0.18,
                stiffness=4800.0,
                damping=165.0,
            ),
            "oncology_resection": ImplicitActuatorCfg(
                joint_names_expr=[
                    "aspirator_.*_joint",
                    "bipolar_.*_jaw_joint",
                    "scissor_.*_joint",
                ],
                effort_limit_sim=140.0,
                velocity_limit_sim=1.5,
                stiffness=6000.0,
                damping=180.0,
            ),
            "oncology_specimen": ImplicitActuatorCfg(
                joint_names_expr=["bag_.*_joint", "margin_marker_joint"],
                effort_limit_sim=70.0,
                velocity_limit_sim=0.30,
                stiffness=3600.0,
                damping=110.0,
            ),
            "oncology_valves": ImplicitActuatorCfg(
                joint_names_expr=[".*_valve_joint"],
                effort_limit_sim=30.0,
                velocity_limit_sim=0.25,
                stiffness=1800.0,
                damping=55.0,
            ),
        }
    )
    return cfg


def dynamic_patient_oncology_binding(
    patient_prim_path: str = "/World/DrAnmarDynamicPatient",
) -> dict[str, Any]:
    """Describe the explicit whole-patient integration boundary.

    The returned contract prevents silently stacking the demo liver over the
    dynamic patient's liver.  A compositor must deactivate the demo substrate,
    bind oncology state to the patient's liver/tumor prims, and preserve the
    patient's global blood and bile ledgers.
    """

    base = patient_prim_path.rstrip("/")
    return {
        "schema": "dr.anmar.dynamic-patient-oncology-binding.v2",
        "patient_root": base,
        "liver_prim": f"{base}/Anatomy/liver",
        "tumor_prim": f"{base}/Anatomy/liver_tumor",
        "major_vessels_prim": f"{base}/Anatomy/major_vessels",
        "gallbladder_prim": f"{base}/Anatomy/gallbladder",
        "demo_liver_active": False,
        "native_deformable_component": "liver",
        "deformable_representation": "gpu_volume_tetmesh",
        "irreversible_topology_representation": (
            "registered_discrete_resection_graph"
        ),
        "maximum_active_deformable_components": 1,
        "shared_ledgers": ("blood", "bile"),
        "required_registration_frames": (
            "tumor_mapping_tcp",
            "resection_tcp",
            "cavity_scan_reference",
        ),
        "clinical_validation": False,
    }


def _require_native_volume_route(result: Mapping[str, Any]) -> dict[str, Any]:
    """Reject silent proxy fallback when oncology requests native liver FEM."""

    route = str(result.get("route", ""))
    native_volume = (
        "volume" in route
        and route != "not_applied"
        and not route.startswith("host_controlled")
    )
    if not native_volume:
        detail = result.get("error", "no native volume route was reported")
        raise RuntimeError(
            "Oncology liver requires a native GPU volume deformable; "
            f"received route={route!r}: {detail}"
        )
    required = ("body_prim_path", "simulation_mesh_path")
    missing = [key for key in required if not result.get(key)]
    if missing:
        raise RuntimeError(
            f"Native liver deformable omitted runtime paths: {missing}"
        )
    return dict(result)


def spawn_oncology_volume_liver(
    prim_path: str = "/World/DrAnmarOncologyVolumeLiver",
    *,
    position: Sequence[float] = (0.0, 0.0, 0.0),
    orientation_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    stage: Any | None = None,
    patient_runtime: Any | None = None,
) -> dict[str, Any]:
    """Spawn the Dynamic Patient liver as a native GPU volume deformable.

    Its explicit TetMesh supplies continuous deformation. The registered
    resection graph remains authoritative for irreversible cutting and
    detachment because PhysX does not mutate deformable topology at runtime.
    """

    if not str(prim_path).startswith("/"):
        raise ValueError("prim_path must be an absolute USD prim path")
    translation = tuple(float(value) for value in position)
    if len(translation) != 3 or not all(
        math.isfinite(value) for value in translation
    ):
        raise ValueError("position must contain three finite values")

    import isaaclab.sim as sim_utils
    from pxr import UsdGeom

    if stage is None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
    cfg = sim_utils.UsdFileCfg(usd_path=str(DYNAMIC_PATIENT_LIVER_USD))
    cfg.func(
        prim_path,
        cfg,
        translation=translation,
        orientation=_wxyz_quaternion(orientation_wxyz),
    )
    runtime = patient_runtime or _load_dynamic_patient_runtime()
    component = next(
        (
            entry
            for entry in runtime.load_anatomy_manifest()["components"]
            if entry["id"] == "liver"
        ),
        None,
    )
    if component is None:
        raise RuntimeError("Dynamic Patient manifest has no liver component")
    result = _require_native_volume_route(
        runtime.apply_component_deformable(
            stage,
            prim_path,
            component,
            material_path=f"{prim_path}/Physics/LiverMaterial",
        )
    )
    result.update(
        {
            "oncology_binding": "standalone_dynamic_patient_liver",
            "continuous_mechanics": "native_gpu_volume_deformable",
            "irreversible_topology": "registered_discrete_resection_graph",
            "constitutive_validation": False,
        }
    )
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid() or not UsdGeom.Xformable(root):
        raise RuntimeError(f"Spawned liver root is invalid: {prim_path}")
    return result


def activate_dynamic_patient_oncology_liver(
    patient_prim_path: str = "/World/DrAnmarDynamicPatient",
    *,
    demo_liver_prim_path: str | None = None,
    stage: Any | None = None,
    patient_runtime: Any | None = None,
) -> dict[str, Any]:
    """Activate the one-liver whole-patient GPU deformable route.

    This fails closed if the native volume route is unavailable and
    deactivates a supplied demo liver to prevent overlapping anatomy.
    """

    if not str(patient_prim_path).startswith("/"):
        raise ValueError("patient_prim_path must be an absolute USD prim path")
    if stage is None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
    binding = dynamic_patient_oncology_binding(patient_prim_path)
    patient = stage.GetPrimAtPath(binding["patient_root"])
    liver = stage.GetPrimAtPath(binding["liver_prim"])
    if not patient.IsValid() or not liver.IsValid():
        raise RuntimeError(
            "Dynamic Patient and its liver must exist before oncology "
            f"activation: patient={binding['patient_root']}, "
            f"liver={binding['liver_prim']}"
        )
    if demo_liver_prim_path is not None:
        if demo_liver_prim_path == binding["liver_prim"]:
            raise ValueError("demo liver path cannot be the patient liver path")
        demo = stage.GetPrimAtPath(demo_liver_prim_path)
        if not demo.IsValid():
            raise RuntimeError(
                f"Requested demo liver does not exist: {demo_liver_prim_path}"
            )
        demo.SetActive(False)

    runtime = patient_runtime or _load_dynamic_patient_runtime()
    routes = runtime.apply_patient_deformables(
        binding["patient_root"],
        include=("liver",),
        stage=stage,
    )
    result = _require_native_volume_route(routes["liver"])
    result.update(
        {
            "oncology_binding": binding,
            "continuous_mechanics": "native_gpu_volume_deformable",
            "irreversible_topology": "registered_discrete_resection_graph",
            "constitutive_validation": False,
        }
    )
    return result


def attach_camera_prims(stage: Any, tool_path: str) -> dict[str, str]:
    """Author portable USD cameras at the registered optical frames.

    Isaac Lab ``Camera`` objects and annotators remain host-scene concerns. This
    helper authors only the USD camera interface so the same tool can be used
    by interactive, manager-based, and headless qualification scenes.
    """

    from pxr import Gf, Sdf, UsdGeom

    specs = {
        "rgb_camera_left": (24.0, 20.955, 0.01, 2.0),
        "rgb_camera_right": (24.0, 20.955, 0.01, 2.0),
        "nir_fluorescence_camera": (35.0, 20.955, 0.01, 1.5),
        "hsi_camera": (35.0, 20.955, 0.01, 1.0),
    }
    created: dict[str, str] = {}
    for name, (focal, aperture, near, far) in specs.items():
        path = f"{frame_path(tool_path, name)}/Camera"
        camera = UsdGeom.Camera.Define(stage, path)
        xform = UsdGeom.Xformable(camera)
        xform.ClearXformOpOrder()
        # USD cameras look along local -Z; authored oncology optical frames use
        # +Z as the tissue-facing axis.
        xform.AddOrientOp().Set(Gf.Quatf(0.0, 0.0, 1.0, 0.0))
        camera.CreateFocalLengthAttr(float(focal))
        camera.CreateHorizontalApertureAttr(float(aperture))
        camera.CreateClippingRangeAttr(Gf.Vec2f(float(near), float(far)))
        camera.GetPrim().CreateAttribute(
            "drAnmar:modality", Sdf.ValueTypeNames.String
        ).Set(name)
        created[name] = path
    return created


def sensor_runtime_contract(tool_path: str) -> dict[str, Any]:
    """Return the version-neutral host contract for oncology sensors."""

    return {
        "schema": "dr.anmar.oncology-sensor-runtime.v1",
        "rtx_cameras": {
            name: {
                "frame": frame_path(tool_path, name),
                "update_period_s": period,
                "timestamp_required": True,
            }
            for name, period in {
                "rgb_camera_left": 1.0 / 30.0,
                "rgb_camera_right": 1.0 / 30.0,
                "nir_fluorescence_camera": 1.0 / 20.0,
                "hsi_camera": 1.0 / 10.0,
            }.items()
        },
        "oct": {
            "origin_frame": frame_path(tool_path, "oct_beam_origin"),
            "output": "registered_surface_depth_and_margin_probability",
            "calibration": "proxy_pending_physical_calibration",
        },
        "ultrasound": {
            "probe_frame": frame_path(tool_path, "ultrasound_probe_face"),
            "beam_axis_frame": frame_path(
                tool_path, "ultrasound_acoustic_axis"
            ),
            "recommended_bridge": (
                "isaac_for_healthcare_robotic_ultrasound_or_validated_proxy"
            ),
            "output": "registered_b_mode_and_protected_structure_probability",
        },
        "raman": {
            "contact_frame": frame_path(tool_path, "raman_contact"),
            "output": "registered_contact_margin_probability",
            "calibration": "proxy_pending_physical_calibration",
        },
        "fusion": {
            "minimum_modalities": 2,
            "maximum_timestamp_skew_s": 0.050,
            "maximum_sample_age_s": 0.250,
            "maximum_registration_error_m": 0.003,
            "buffering": "timestamped_common-time_interpolation",
            "failure_policy": "abstain",
        },
    }


class SafetyInterlockError(RuntimeError):
    """Raised when a protected oncologic action is rejected fail-closed."""


def _probability(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be finite and inside [0, 1]")
    return result


@dataclass(frozen=True)
class SensorReading:
    modality: str
    tumor_probability: float
    margin_probability: float
    protected_structure_probability: float
    confidence: float
    timestamp_s: float
    registration_error_m: float
    valid: bool = True

    def __post_init__(self) -> None:
        for name in (
            "tumor_probability",
            "margin_probability",
            "protected_structure_probability",
            "confidence",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        if not math.isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if (
            not math.isfinite(self.registration_error_m)
            or self.registration_error_m < 0.0
        ):
            raise ValueError("registration_error_m must be finite and non-negative")


@dataclass(frozen=True)
class FusionResult:
    tumor_probability: float
    margin_probability: float
    protected_structure_probability: float
    sensor_disagreement: float
    confidence: float
    modalities: tuple[str, ...]
    actionable: bool
    abstention_reason: str | None = None


@dataclass
class MultimodalOncologyFusion:
    """Registration-aware fusion that abstains on disagreement or stale data."""

    modality_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "rgb_depth": 0.45,
            "nir_fluorescence": 1.00,
            "hyperspectral": 0.85,
            "ultrasound": 1.00,
            "oct": 0.95,
            "raman": 1.10,
        }
    )
    minimum_modalities: int = 2
    maximum_age_s: float = 0.250
    maximum_skew_s: float = 0.050
    maximum_registration_error_m: float = 0.003
    maximum_disagreement: float = 0.25
    minimum_confidence: float = 0.55

    def fuse(
        self,
        readings: Iterable[SensorReading],
        *,
        reference_time_s: float | None = None,
    ) -> FusionResult:
        samples = tuple(readings)
        if not samples:
            return self._abstain("no_sensor_samples")
        if reference_time_s is None:
            reference_time_s = max(sample.timestamp_s for sample in samples)
        if not math.isfinite(reference_time_s):
            raise ValueError("reference_time_s must be finite")

        accepted = tuple(
            sample
            for sample in samples
            if sample.valid
            and sample.modality in self.modality_weights
            and reference_time_s - sample.timestamp_s <= self.maximum_age_s
            and sample.timestamp_s <= reference_time_s + 1.0e-9
            and sample.registration_error_m
            <= self.maximum_registration_error_m
        )
        modalities = {sample.modality for sample in accepted}
        if len(modalities) < self.minimum_modalities:
            return self._abstain("insufficient_registered_modalities")
        skew = max(sample.timestamp_s for sample in accepted) - min(
            sample.timestamp_s for sample in accepted
        )
        if skew > self.maximum_skew_s:
            return self._abstain("sensor_timestamp_skew")

        weights = tuple(
            self.modality_weights[sample.modality] * sample.confidence
            for sample in accepted
        )
        total = sum(weights)
        if total <= 1.0e-9:
            return self._abstain("zero_effective_confidence")

        def weighted(attribute: str) -> float:
            return sum(
                weight * getattr(sample, attribute)
                for sample, weight in zip(accepted, weights)
            ) / total

        ranges = []
        for attribute in (
            "tumor_probability",
            "margin_probability",
            "protected_structure_probability",
        ):
            values = [getattr(sample, attribute) for sample in accepted]
            ranges.append(max(values) - min(values))
        disagreement = max(ranges)
        confidence = (
            sum(sample.confidence * weight for sample, weight in zip(accepted, weights))
            / total
        ) * (1.0 - disagreement)
        reason = None
        if disagreement > self.maximum_disagreement:
            reason = "sensor_disagreement"
        elif confidence < self.minimum_confidence:
            reason = "low_fused_confidence"
        return FusionResult(
            tumor_probability=weighted("tumor_probability"),
            margin_probability=weighted("margin_probability"),
            protected_structure_probability=weighted(
                "protected_structure_probability"
            ),
            sensor_disagreement=disagreement,
            confidence=confidence,
            modalities=tuple(sorted(modalities)),
            actionable=reason is None,
            abstention_reason=reason,
        )

    @staticmethod
    def _abstain(reason: str) -> FusionResult:
        return FusionResult(
            tumor_probability=0.0,
            margin_probability=0.0,
            protected_structure_probability=0.0,
            sensor_disagreement=1.0,
            confidence=0.0,
            modalities=(),
            actionable=False,
            abstention_reason=reason,
        )


@dataclass
class TumorCell:
    id: str
    center_m: tuple[float, float, float]
    tissue_class: str
    tumor_probability: float
    planned_resection: bool
    protected_structure: str | None
    vessel_clearance_m: float
    duct_clearance_m: float
    sensor_uncertainty: float
    removed: bool = False
    ablated: bool = False


@dataclass
class TumorFieldModel:
    """Mutable episode state backed by the package's 3-D tumor field."""

    cells: dict[str, TumorCell]
    spacing_m: tuple[float, float, float]
    planned_margin_m: float
    _margin_cache: tuple[float, ...] | None = field(
        default=None, init=False, repr=False
    )

    @classmethod
    def from_json(
        cls, path: str | Path = TUMOR_FIELD_JSON
    ) -> "TumorFieldModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        raw_cells = payload["cells"]
        cells = {
            item["id"]: TumorCell(
                id=item["id"],
                center_m=tuple(float(value) for value in item["center_m"]),
                tissue_class=item["tissue_class"],
                tumor_probability=_probability(
                    item["tumor_probability"], "tumor_probability"
                ),
                planned_resection=bool(item["planned_resection"]),
                protected_structure=item.get("protected_structure"),
                vessel_clearance_m=float(item["vessel_clearance_m"]),
                duct_clearance_m=float(item["duct_clearance_m"]),
                sensor_uncertainty=_probability(
                    item["sensor_uncertainty"], "sensor_uncertainty"
                ),
                removed=bool(item.get("removed", False)),
                ablated=bool(item.get("ablated", False)),
            )
            for item in raw_cells
        }
        if len(cells) != len(raw_cells):
            raise ValueError("tumor_field.json contains duplicate cell ids")
        spacing = tuple(float(value) for value in payload["cell_spacing_m"])
        if len(spacing) != 3 or any(value <= 0.0 for value in spacing):
            raise ValueError("cell_spacing_m must contain three positive values")
        return cls(
            cells=cells,
            spacing_m=spacing,
            planned_margin_m=float(payload["planned_margin_m"]),
        )

    @property
    def cell_volume_mm3(self) -> float:
        return math.prod(self.spacing_m) * 1.0e9

    @property
    def planned_cell_ids(self) -> tuple[str, ...]:
        return tuple(
            cell.id for cell in self.cells.values() if cell.planned_resection
        )

    @property
    def tumor_cell_ids(self) -> tuple[str, ...]:
        return tuple(
            cell.id
            for cell in self.cells.values()
            if cell.tissue_class != "healthy_parenchyma"
        )

    def remove(self, cell_ids: Iterable[str]) -> int:
        changed = 0
        for cell_id in dict.fromkeys(cell_ids):
            try:
                cell = self.cells[cell_id]
            except KeyError as exc:
                raise KeyError(f"Unknown tumor-field cell {cell_id!r}") from exc
            if not cell.removed:
                cell.removed = True
                changed += 1
        if changed:
            self._margin_cache = None
        return changed

    def ablate(self, cell_ids: Iterable[str]) -> int:
        changed = 0
        for cell_id in dict.fromkeys(cell_ids):
            try:
                cell = self.cells[cell_id]
            except KeyError as exc:
                raise KeyError(f"Unknown tumor-field cell {cell_id!r}") from exc
            if not cell.ablated:
                cell.ablated = True
                changed += 1
        if changed:
            self._margin_cache = None
        return changed

    @property
    def residual_tumor_volume_mm3(self) -> float:
        count = sum(
            not cell.removed
            and not cell.ablated
            and cell.tissue_class != "healthy_parenchyma"
            for cell in self.cells.values()
        )
        return count * self.cell_volume_mm3

    @property
    def tumor_removed_volume_mm3(self) -> float:
        count = sum(
            (cell.removed or cell.ablated)
            and cell.tissue_class != "healthy_parenchyma"
            for cell in self.cells.values()
        )
        return count * self.cell_volume_mm3

    @property
    def healthy_removed_volume_mm3(self) -> float:
        count = sum(
            cell.removed and cell.tissue_class == "healthy_parenchyma"
            for cell in self.cells.values()
        )
        return count * self.cell_volume_mm3

    @property
    def resected_volume_mm3(self) -> float:
        return (
            sum(cell.removed for cell in self.cells.values())
            * self.cell_volume_mm3
        )

    def _margin_distances_m(self) -> tuple[float, ...]:
        if self._margin_cache is not None:
            return self._margin_cache
        residual = [
            cell
            for cell in self.cells.values()
            if cell.tissue_class != "healthy_parenchyma"
            and not cell.removed
            and not cell.ablated
        ]
        if residual:
            self._margin_cache = (0.0,)
            return self._margin_cache
        treated_tumor = [
            cell
            for cell in self.cells.values()
            if cell.tissue_class != "healthy_parenchyma"
            and (cell.removed or cell.ablated)
        ]
        remaining_healthy = [
            cell
            for cell in self.cells.values()
            if cell.tissue_class == "healthy_parenchyma" and not cell.removed
        ]
        if not treated_tumor or not remaining_healthy:
            self._margin_cache = (0.0,)
            return self._margin_cache
        half_diagonal = 0.5 * math.sqrt(
            sum(value * value for value in self.spacing_m)
        )
        distances = []
        for tumor in treated_tumor:
            nearest = min(
                math.dist(tumor.center_m, healthy.center_m)
                for healthy in remaining_healthy
            )
            distances.append(max(0.0, nearest - half_diagonal))
        self._margin_cache = tuple(distances)
        return self._margin_cache

    @property
    def minimum_margin_m(self) -> float:
        return min(self._margin_distances_m())

    @property
    def margin_percentile_05_m(self) -> float:
        values = sorted(self._margin_distances_m())
        index = max(0, math.ceil(0.05 * len(values)) - 1)
        return values[index]

    def metrics(self) -> dict[str, float]:
        return {
            "planned_volume_mm3": (
                len(self.planned_cell_ids) * self.cell_volume_mm3
            ),
            "resected_volume_mm3": self.resected_volume_mm3,
            "healthy_tissue_removed_mm3": self.healthy_removed_volume_mm3,
            "tumor_removed_mm3": self.tumor_removed_volume_mm3,
            "residual_tumor_volume_mm3": self.residual_tumor_volume_mm3,
            "minimum_margin_m": self.minimum_margin_m,
            "margin_percentile_05_m": self.margin_percentile_05_m,
        }


@dataclass
class ResectionBond:
    id: str
    kind: str
    center_m: tuple[float, float, float]
    normal: tuple[float, float, float]
    recommended_modality: str
    mechanical_work_threshold_j: float
    aspiration_energy_threshold_j: float
    clearance_m: float
    nearest_protected_structure: str | None
    seal_required: bool
    released: bool = False
    sealed: bool = False


@dataclass
class ResectionTopologyModel:
    """Discrete topology with a hard interlock on unsealed pedicles."""

    bonds: dict[str, ResectionBond]
    unsafe_attempts: int = 0
    vessel_injury_count: int = 0
    duct_injury_count: int = 0

    @classmethod
    def from_json(
        cls, path: str | Path = RESECTION_TOPOLOGY_JSON
    ) -> "ResectionTopologyModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        bonds = {
            item["id"]: ResectionBond(
                id=item["id"],
                kind=item["kind"],
                center_m=tuple(float(value) for value in item["center_m"]),
                normal=tuple(float(value) for value in item["normal"]),
                recommended_modality=item["recommended_modality"],
                mechanical_work_threshold_j=float(
                    item["mechanical_work_threshold_j"]
                ),
                aspiration_energy_threshold_j=float(
                    item["aspiration_energy_threshold_j"]
                ),
                clearance_m=float(item["clearance_m"]),
                nearest_protected_structure=item.get(
                    "nearest_protected_structure"
                ),
                seal_required=bool(item["seal_required"]),
                released=bool(item.get("released", False)),
                sealed=bool(item.get("sealed", False)),
            )
            for item in payload["bonds"]
        }
        if len(bonds) != int(payload["bond_count"]):
            raise ValueError("Resection topology bond count is inconsistent")
        return cls(bonds=bonds)

    def _bond(self, bond_id: str) -> ResectionBond:
        try:
            return self.bonds[bond_id]
        except KeyError as exc:
            raise KeyError(f"Unknown resection bond {bond_id!r}") from exc

    def seal(
        self,
        bond_id: str,
        *,
        compression_force_n: float,
        energy_j: float,
    ) -> bool:
        bond = self._bond(bond_id)
        if not bond.seal_required:
            raise ValueError(f"{bond_id} does not require energy sealing")
        force = float(compression_force_n)
        energy = float(energy_j)
        if not math.isfinite(force) or not math.isfinite(energy):
            raise ValueError("compression_force_n and energy_j must be finite")
        if force < 4.0 or force > 40.0:
            raise SafetyInterlockError(
                "Pedicle seal rejected: compression must remain within the "
                "provisional 4-40 N research window"
            )
        minimum_energy = max(
            0.05, min(0.14, 0.60 * bond.mechanical_work_threshold_j)
        )
        if energy < minimum_energy:
            raise SafetyInterlockError(
                f"Pedicle seal rejected: {energy:.4f} J is below the "
                f"{minimum_energy:.4f} J research threshold"
            )
        bond.sealed = True
        return True

    def release(
        self,
        bond_id: str,
        *,
        modality: str,
        mechanical_work_j: float = 0.0,
        aspiration_energy_j: float = 0.0,
    ) -> bool:
        bond = self._bond(bond_id)
        if bond.released:
            return False
        if bond.seal_required and not bond.sealed:
            self.unsafe_attempts += 1
            raise SafetyInterlockError(
                f"{bond.id} is a protected {bond.kind}; seal confirmation is "
                "required before division"
            )
        if modality != bond.recommended_modality:
            raise SafetyInterlockError(
                f"{bond.id} requires {bond.recommended_modality}, not {modality}"
            )
        if modality == "selective_aspiration":
            supplied = float(aspiration_energy_j)
            required = bond.aspiration_energy_threshold_j
        else:
            supplied = float(mechanical_work_j)
            required = bond.mechanical_work_threshold_j
        if not math.isfinite(supplied) or supplied < 0.0:
            raise ValueError("supplied release energy/work must be non-negative")
        if supplied < required:
            raise SafetyInterlockError(
                f"{bond.id} release rejected: {supplied:.5f} J is below "
                f"{required:.5f} J"
            )
        bond.released = True
        return True

    def record_protected_structure_injury(self, structure: str) -> None:
        if structure == "vessel":
            self.vessel_injury_count += 1
        elif structure == "duct":
            self.duct_injury_count += 1
        else:
            raise ValueError("structure must be 'vessel' or 'duct'")

    @property
    def released_fraction(self) -> float:
        return sum(bond.released for bond in self.bonds.values()) / len(self.bonds)

    @property
    def sealed_fraction(self) -> float:
        required = [bond for bond in self.bonds.values() if bond.seal_required]
        return sum(bond.sealed for bond in required) / len(required)

    @property
    def specimen_detached(self) -> bool:
        return all(bond.released for bond in self.bonds.values()) and all(
            not bond.seal_required or bond.sealed
            for bond in self.bonds.values()
        )


ORIENTATION_MARKERS = frozenset(
    {"superior", "inferior", "medial", "lateral", "anterior", "posterior"}
)


@dataclass
class SpecimenWorkflow:
    deployed: bool = False
    captured: bool = False
    closed: bool = False
    orientation_markers: set[str] = field(default_factory=set)

    def deploy(self) -> None:
        self.deployed = True

    def capture(self, *, specimen_detached: bool) -> None:
        if not self.deployed:
            raise SafetyInterlockError("Specimen capture requires a deployed bag")
        if not specimen_detached:
            raise SafetyInterlockError(
                "Specimen capture requires confirmed complete detachment"
            )
        self.captured = True

    def close(self) -> None:
        if not self.captured:
            raise SafetyInterlockError("Bag closure requires contained specimen")
        self.closed = True

    def mark_orientation(self, marker: str) -> None:
        if not self.closed:
            raise SafetyInterlockError(
                "Orientation marking requires a closed specimen bag"
            )
        if marker not in ORIENTATION_MARKERS:
            raise ValueError(
                f"Unknown marker {marker!r}; expected {sorted(ORIENTATION_MARKERS)}"
            )
        self.orientation_markers.add(marker)

    @property
    def orientation_complete(self) -> bool:
        return self.orientation_markers == ORIENTATION_MARKERS


@dataclass(frozen=True)
class OncologyDomainParameters:
    registration_bias_m: tuple[float, float, float]
    tissue_stiffness_scale: float
    tool_friction_scale: float
    tumor_probability_bias: float
    blood_loss_scale: float
    bile_loss_scale: float
    dropped_modality: str | None


def sample_domain_parameters(
    seed: int | None = None,
) -> OncologyDomainParameters:
    """Sample bounded research-domain randomization for Isaac Lab episodes."""

    generator = random.Random(seed)
    modalities = (
        None,
        None,
        "rgb_depth",
        "nir_fluorescence",
        "hyperspectral",
        "ultrasound",
        "oct",
        "raman",
    )
    return OncologyDomainParameters(
        registration_bias_m=tuple(
            generator.uniform(-0.002, 0.002) for _ in range(3)
        ),
        tissue_stiffness_scale=generator.uniform(0.75, 1.30),
        tool_friction_scale=generator.uniform(0.80, 1.20),
        tumor_probability_bias=generator.uniform(-0.08, 0.08),
        blood_loss_scale=generator.uniform(0.75, 1.35),
        bile_loss_scale=generator.uniform(0.75, 1.35),
        dropped_modality=generator.choice(modalities),
    )


@dataclass
class OncologicResectionEpisode:
    """Deterministic oncology task state suitable for scripted or RL control."""

    tumor_field: TumorFieldModel = field(default_factory=TumorFieldModel.from_json)
    topology: ResectionTopologyModel = field(
        default_factory=ResectionTopologyModel.from_json
    )
    fusion: MultimodalOncologyFusion = field(
        default_factory=MultimodalOncologyFusion
    )
    specimen: SpecimenWorkflow = field(default_factory=SpecimenWorkflow)
    phase_index: int = 0
    registration_error_m: float | None = None
    latest_fusion: FusionResult | None = None
    planned_cell_ids: set[str] = field(default_factory=set)
    traction_captured: bool = False
    cavity_scanned: bool = False
    correction_count: int = 0
    blood_loss_ml: float = 0.0
    bile_loss_ml: float = 0.0
    hemostasis_checked: bool = False
    finalized: bool = False

    @property
    def phase(self) -> str:
        return TASK_PHASES[self.phase_index]

    def register(self, error_m: float) -> None:
        value = float(error_m)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("registration error must be finite and non-negative")
        self.registration_error_m = value

    def map_sensors(
        self,
        readings: Iterable[SensorReading],
        *,
        reference_time_s: float | None = None,
    ) -> FusionResult:
        self.latest_fusion = self.fusion.fuse(
            readings, reference_time_s=reference_time_s
        )
        return self.latest_fusion

    def set_plan(self, cell_ids: Iterable[str] | None = None) -> None:
        selected = (
            self.tumor_field.planned_cell_ids
            if cell_ids is None
            else tuple(dict.fromkeys(cell_ids))
        )
        unknown = set(selected) - set(self.tumor_field.cells)
        if unknown:
            raise KeyError(f"Unknown planned cells: {sorted(unknown)[:5]}")
        if not selected:
            raise ValueError("A resection plan must contain at least one cell")
        self.planned_cell_ids = set(selected)

    def confirm_traction_capture(self) -> None:
        self.traction_captured = True

    def resect_cells(
        self, cell_ids: Iterable[str], *, corrective: bool = False
    ) -> int:
        selected = tuple(dict.fromkeys(cell_ids))
        if not corrective:
            outside = set(selected) - self.planned_cell_ids
            if outside:
                raise SafetyInterlockError(
                    "Primary resection rejected outside the accepted plan"
                )
        changed = self.tumor_field.remove(selected)
        if corrective and changed:
            self.correction_count += 1
        return changed

    def record_cavity_scan(
        self,
        readings: Iterable[SensorReading],
        *,
        reference_time_s: float | None = None,
    ) -> FusionResult:
        result = self.map_sensors(
            readings, reference_time_s=reference_time_s
        )
        if not result.actionable:
            raise SafetyInterlockError(
                f"Cavity scan abstained: {result.abstention_reason}"
            )
        self.cavity_scanned = True
        return result

    def record_losses(self, *, blood_ml: float, bile_ml: float) -> None:
        blood = float(blood_ml)
        bile = float(bile_ml)
        if (
            not math.isfinite(blood)
            or not math.isfinite(bile)
            or blood < 0.0
            or bile < 0.0
        ):
            raise ValueError("blood_ml and bile_ml must be finite and non-negative")
        self.blood_loss_ml += blood
        self.bile_loss_ml += bile

    def confirm_hemostasis_and_bile_check(self) -> None:
        self.hemostasis_checked = True

    def _current_gate(self) -> tuple[bool, str | None]:
        phase = self.phase
        if phase == "inspect":
            return True, None
        if phase == "register":
            ok = (
                self.registration_error_m is not None
                and self.registration_error_m
                <= self.fusion.maximum_registration_error_m
            )
            return ok, "registration_not_within_3_mm"
        if phase == "map":
            ok = self.latest_fusion is not None and self.latest_fusion.actionable
            return ok, "multimodal_map_not_actionable"
        if phase == "plan":
            return bool(self.planned_cell_ids), "resection_plan_missing"
        if phase == "capture":
            return self.traction_captured, "traction_capture_unconfirmed"
        if phase == "resect_parenchyma":
            non_pedicle = [
                bond
                for bond in self.topology.bonds.values()
                if not bond.seal_required
            ]
            return all(bond.released for bond in non_pedicle), (
                "parenchymal_boundary_incomplete"
            )
        if phase == "manage_pedicles":
            pedicles = [
                bond
                for bond in self.topology.bonds.values()
                if bond.seal_required
            ]
            return all(bond.released and bond.sealed for bond in pedicles), (
                "protected_pedicles_not_safely_divided"
            )
        if phase == "complete_boundary":
            return self.topology.specimen_detached, "specimen_not_detached"
        if phase == "deploy_bag":
            return self.specimen.deployed, "specimen_bag_not_deployed"
        if phase == "capture_specimen":
            return self.specimen.captured, "specimen_not_contained"
        if phase == "close_bag":
            return self.specimen.closed, "specimen_bag_not_closed"
        if phase == "mark_orientation":
            return self.specimen.orientation_complete, (
                "orientation_markers_incomplete"
            )
        if phase == "cavity_scan":
            return self.cavity_scanned, "cavity_scan_missing"
        if phase == "corrective_resection":
            return self.tumor_field.residual_tumor_volume_mm3 == 0.0, (
                "residual_tumor_present"
            )
        if phase == "hemostasis_and_bile_check":
            return self.hemostasis_checked, "hemostasis_or_bile_check_missing"
        return self.finalized, "final_report_not_recorded"

    def advance(self) -> str:
        if self.phase_index >= len(TASK_PHASES) - 1:
            raise RuntimeError("Episode is already at the final phase")
        allowed, reason = self._current_gate()
        if not allowed:
            raise SafetyInterlockError(
                f"Cannot leave phase {self.phase!r}: {reason}"
            )
        self.phase_index += 1
        return self.phase

    def finalize(self) -> dict[str, Any]:
        if self.phase != "final_margin_report":
            raise SafetyInterlockError(
                "Final report is only available in final_margin_report phase"
            )
        self.finalized = True
        return self.report()

    def report(self) -> dict[str, Any]:
        metrics: dict[str, Any] = self.tumor_field.metrics()
        metrics.update(
            {
                "vessel_injury_count": self.topology.vessel_injury_count,
                "duct_injury_count": self.topology.duct_injury_count,
                "protected_structure_injury_count": (
                    self.topology.vessel_injury_count
                    + self.topology.duct_injury_count
                ),
                "blood_loss_ml": self.blood_loss_ml,
                "bile_loss_ml": self.bile_loss_ml,
                "bag_capture_success": self.specimen.captured,
                "bag_closed": self.specimen.closed,
                "orientation_marker_completeness": (
                    len(self.specimen.orientation_markers)
                    / len(ORIENTATION_MARKERS)
                ),
                "sensor_disagreement": (
                    self.latest_fusion.sensor_disagreement
                    if self.latest_fusion is not None
                    else 1.0
                ),
                "correction_count": self.correction_count,
                "specimen_detached": self.topology.specimen_detached,
                "unsafe_action_attempts": self.topology.unsafe_attempts,
            }
        )
        success_checks = {
            "residual_tumor_clear": (
                metrics["residual_tumor_volume_mm3"] <= 0.0
            ),
            "minimum_margin": metrics["minimum_margin_m"] >= 0.010,
            "protected_structures_intact": (
                metrics["protected_structure_injury_count"] <= 0
            ),
            "blood_loss_bounded": self.blood_loss_ml <= 5.0,
            "bile_loss_bounded": self.bile_loss_ml <= 0.2,
            "specimen_detached": self.topology.specimen_detached,
            "specimen_contained": self.specimen.captured and self.specimen.closed,
            "orientation_complete": self.specimen.orientation_complete,
            "cavity_verified": self.cavity_scanned,
            "hemostasis_checked": self.hemostasis_checked,
        }
        return {
            "schema": "dr.anmar.oncologic-resection-result.v1",
            "phase": self.phase,
            "metrics": metrics,
            "success_checks": success_checks,
            "success": all(success_checks.values()),
            "clinical_validation": False,
        }

    def observation(self) -> tuple[float, ...]:
        initial_tumor = max(
            1.0,
            len(self.tumor_field.tumor_cell_ids)
            * self.tumor_field.cell_volume_mm3,
        )
        planned_volume = max(
            1.0,
            len(self.tumor_field.planned_cell_ids)
            * self.tumor_field.cell_volume_mm3,
        )
        latest = self.latest_fusion
        return (
            self.phase_index / (len(TASK_PHASES) - 1),
            self.topology.released_fraction,
            self.topology.sealed_fraction,
            self.tumor_field.residual_tumor_volume_mm3 / initial_tumor,
            self.tumor_field.healthy_removed_volume_mm3 / planned_volume,
            min(self.tumor_field.minimum_margin_m / 0.010, 2.0),
            min(self.blood_loss_ml / 5.0, 2.0),
            min(self.bile_loss_ml / 0.2, 2.0),
            latest.sensor_disagreement if latest else 1.0,
            latest.confidence if latest else 0.0,
            float(self.specimen.closed),
            len(self.specimen.orientation_markers) / len(ORIENTATION_MARKERS),
        )

    def reward(self) -> float:
        """Dense bounded task reward; success remains report-gated."""

        observation = self.observation()
        reward = (
            1.5 * observation[0]
            + 1.0 * observation[1]
            + 0.5 * observation[2]
            + 1.5 * (1.0 - min(observation[3], 1.0))
            + 0.5 * observation[10]
            + 0.5 * observation[11]
            - 1.0 * min(observation[4], 1.0)
            - 1.0 * min(observation[6], 1.0)
            - 1.0 * min(observation[7], 1.0)
            - 0.5 * observation[8]
            - 0.5 * self.topology.unsafe_attempts
        )
        if self.finalized and self.report()["success"]:
            reward += 5.0
        return reward
