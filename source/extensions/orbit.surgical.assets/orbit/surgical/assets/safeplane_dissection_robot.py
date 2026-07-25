# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Isaac Lab integration for the DrAnmar SafePlane Dissection Robot.

The module provides a Franka hand-replacement spawner, layered tissue
substrate integration, distributed traction, four dissection modalities,
protected-structure safety state, particle-fluid helpers, and a topology-
based completion verifier. All parameters are provisional research values.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CATALOG_SUBPATH = "Props/SurgicalDissection/SafePlaneDissectionRobot"
ASSET_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
ASSET_ROOT = ASSET_DATA_ROOT / CATALOG_SUBPATH
TOOL_PAYLOAD_USD = ASSET_ROOT / "dranmar_safeplane_dissection_tool_payload.usda"
TOOL_STANDALONE_USD = ASSET_ROOT / "dranmar_safeplane_dissection_tool_standalone.usda"
TOOL_RIGID_PROXY_USD = ASSET_ROOT / "dranmar_safeplane_dissection_tool_rigid_proxy.usda"
TISSUE_DEMO_USD = ASSET_ROOT / "dranmar_safeplane_tissue_demo.usda"
ADHESION_BRIDGE_USD = ASSET_ROOT / "dranmar_adhesion_bridge.usda"
PROTECTED_VESSEL_USD = ASSET_ROOT / "dranmar_protected_vessel_branch.usda"
PROTECTED_NERVE_USD = ASSET_ROOT / "dranmar_protected_nerve_branch.usda"
PROTECTED_DUCT_USD = ASSET_ROOT / "dranmar_protected_duct_branch.usda"
MICRO_SCISSORS_USD = ASSET_ROOT / "dranmar_micro_scissors_cartridge.usda"
DISSECTION_TOPOLOGY_PATH = ASSET_ROOT / "dissection_topology.json"

VALID_SCISSORS_STATES = frozenset({"fresh", "spent"})
VALID_FLUID_STATES = frozenset({"full", "empty"})
VALID_COLLECTION_STATES = frozenset({"empty", "partial", "full"})
VALID_ENERGY_STATES = frozenset({"ready", "fault"})
PROTECTED_STRUCTURES = ("vessel", "nerve", "duct")

TOOL_JOINTS = {
    "left_traction": "left_traction_joint",
    "right_traction": "right_traction_joint",
    "left_pad_pitch": "left_pad_pitch_joint",
    "right_pad_pitch": "right_pad_pitch_joint",
    "left_pad_compliance": "left_pad_compliance_joint",
    "right_pad_compliance": "right_pad_compliance_joint",
    "left_spreader": "left_spreader_joint",
    "right_spreader": "right_spreader_joint",
    "hydro_pitch": "hydro_pitch_joint",
    "hydro_extension": "hydro_extension_joint",
    "scissor_extension": "scissor_extension_joint",
    "scissor_guard": "scissor_guard_joint",
    "scissor_blade": "scissor_blade_joint",
    "energy_tip_extension": "energy_tip_extension_joint",
    "suction_valve": "suction_valve_joint",
    "hydro_valve": "hydro_valve_joint",
    "irrigation_valve": "irrigation_valve_joint",
}

TOOL_FRAME_PATHS = {
    "panda_link8_mount": "Links/Mount/Frames/panda_link8_mount",
    "safeplane_tcp": "Links/Mount/Frames/safeplane_tcp",
    "safe_plane_reference": "Links/Mount/Frames/safe_plane_reference",
    "roi_center": "Links/Mount/Frames/roi_center",
    "suction_center": "Links/Mount/Frames/suction_center",
    "irrigation_center": "Links/Mount/Frames/irrigation_center",
    "stereo_left": "Links/Mount/Frames/stereo_left",
    "stereo_right": "Links/Mount/Frames/stereo_right",
    "depth_camera": "Links/Mount/Frames/depth_camera",
    "fluorescence_camera": "Links/Mount/Frames/fluorescence_camera",
    "thermal_camera": "Links/Mount/Frames/thermal_camera",
    "protected_structure_probe": "Links/Mount/Frames/protected_structure_probe",
    "hydro_nozzle_tip": "Links/HydroNozzle/Frames/hydro_nozzle_tip",
    "hydro_axis": "Links/HydroNozzle/Frames/hydro_axis",
    "scissor_cut_plane": "Links/ScissorCarriage/Frames/scissor_cut_plane",
    "scissor_guard_tip": "Links/ScissorGuard/Frames/scissor_guard_tip",
    "energy_tip": "Links/EnergyTip/Frames/energy_tip",
    "energy_axis": "Links/EnergyTip/Frames/energy_axis",
    "left_spreader_tip": "Links/LeftSpreader/Frames/left_spreader_tip",
    "right_spreader_tip": "Links/RightSpreader/Frames/right_spreader_tip",
    "left_traction_pad": "Links/LeftTractionPad/Frames/left_traction_pad",
    "right_traction_pad": "Links/RightTractionPad/Frames/right_traction_pad",
    "count_reference": "Links/Mount/Frames/count_reference",
    "disposal_reference": "Links/Mount/Frames/disposal_reference",
}
REGISTERED_CAMERA_FRAMES = (
    "stereo_left", "stereo_right", "depth_camera",
    "fluorescence_camera", "thermal_camera",
)
for _side in ("left", "right"):
    for _index in range(4):
        TOOL_FRAME_PATHS[f"{_side}_capture_cell_{_index}"] = f"Links/{_side.title()}TractionPad/Frames/{_side}_capture_cell_{_index}"

PARTICLE_RADIUS_M = 0.00072
PARTICLE_VOLUME_ML = 4.0 / 3.0 * math.pi * PARTICLE_RADIUS_M**3 * 1.0e6


def frame_path(tool_path: str, name: str) -> str:
    try:
        suffix = TOOL_FRAME_PATHS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown SafePlane frame {name!r}") from exc
    return f"{tool_path.rstrip('/')}/{suffix}"


def tensor_value(value: Any):
    return value.torch if hasattr(value, "torch") else value


def _xyzw_from_wxyz(orientation_wxyz) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in orientation_wxyz)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("orientation_wxyz must contain four finite values")
    if abs(math.sqrt(sum(value * value for value in values)) - 1.0) > 1.0e-4:
        raise ValueError("orientation_wxyz must be a unit quaternion")
    w, x, y, z = values
    return x, y, z, w


def _check(value: str, allowed: frozenset[str], label: str) -> str:
    if value not in allowed:
        raise ValueError(f"Unsupported {label}={value!r}; expected one of {sorted(allowed)}")
    return value


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _finite_nonnegative(value: float, label: str) -> float:
    result = _finite(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def make_tool_cfg(
    prim_path: str = "/World/DrAnmarSafePlaneDissectionTool",
    *,
    scissors_state: str = "fresh",
    hydro_state: str = "full",
    irrigation_state: str = "full",
    collection_state: str = "empty",
    energy_state: str = "ready",
    position=(0.0, 0.0, 0.35),
    orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    _check(scissors_state, VALID_SCISSORS_STATES, "scissors_state")
    _check(hydro_state, VALID_FLUID_STATES, "hydro_state")
    _check(irrigation_state, VALID_FLUID_STATES, "irrigation_state")
    _check(collection_state, VALID_COLLECTION_STATES, "collection_state")
    _check(energy_state, VALID_ENERGY_STATES, "energy_state")
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(TOOL_STANDALONE_USD),
            variants={
                "scissors_state": scissors_state,
                "hydro_state": hydro_state,
                "irrigation_state": irrigation_state,
                "collection_state": collection_state,
                "energy_state": energy_state,
            },
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=24,
                solver_velocity_iteration_count=8,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=position,
            rot=_xyzw_from_wxyz(orientation_wxyz),
            joint_pos={name: 0.0 for name in TOOL_JOINTS.values()},
        ),
        actuators={
            "traction": ImplicitActuatorCfg(
                joint_names_expr=[".*traction_joint"], effort_limit_sim=120.0,
                velocity_limit_sim=0.18, stiffness=4400.0, damping=150.0,
            ),
            "traction_pad": ImplicitActuatorCfg(
                joint_names_expr=[".*pad_pitch_joint", ".*pad_compliance_joint"],
                effort_limit_sim=48.0, velocity_limit_sim=1.2,
                stiffness=1100.0, damping=72.0,
            ),
            "spreader": ImplicitActuatorCfg(
                joint_names_expr=[".*spreader_joint"], effort_limit_sim=85.0,
                velocity_limit_sim=0.14, stiffness=3200.0, damping=125.0,
            ),
            "hydro": ImplicitActuatorCfg(
                joint_names_expr=["hydro_pitch_joint", "hydro_extension_joint"],
                effort_limit_sim=80.0, velocity_limit_sim=1.0,
                stiffness=3600.0, damping=115.0,
            ),
            "scissors": ImplicitActuatorCfg(
                joint_names_expr=["scissor_extension_joint", "scissor_guard_joint", "scissor_blade_joint"],
                effort_limit_sim=120.0, velocity_limit_sim=1.4,
                stiffness=5200.0, damping=160.0,
            ),
            "energy_tip": ImplicitActuatorCfg(
                joint_names_expr=["energy_tip_extension_joint"], effort_limit_sim=70.0,
                velocity_limit_sim=0.18, stiffness=3500.0, damping=115.0,
            ),
            "valves": ImplicitActuatorCfg(
                joint_names_expr=[".*_valve_joint"], effort_limit_sim=25.0,
                velocity_limit_sim=0.25, stiffness=1600.0, damping=48.0,
            ),
        },
    )


def make_rigid_proxy_cfg(
    prim_path: str = "/World/DrAnmarSafePlaneDissectionProxy",
    *, position=(0.0, 0.0, 0.35), orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(TOOL_RIGID_PROXY_USD), activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=position, rot=_xyzw_from_wxyz(orientation_wxyz)
        ),
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
    if len(stock_body_paths) != 1 or stock_body_paths[0] != named["panda_link7"].GetPath():
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
        link8_world.SetTranslate(
            link7_world.Transform(Gf.Vec3d(stock_position))
        )
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
        robot_path.AppendChild("DrAnmarSafePlaneJoints"), "Scope"
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
    mount_local_pos0 = Gf.Vec3f(0, 0, 0)
    angle = math.radians(-45.0) / 2.0
    mount_local_rot0 = Gf.Quatf(math.cos(angle), 0, 0, math.sin(angle))

    candidate_paths = [
        prim.GetPath() for prim in descendants
        if prim.GetPath().HasPrefix(robot_path) and prim.GetName() in disabled
    ]
    paths_to_disable = []
    for path in sorted(candidate_paths, key=lambda item: str(item).count("/")):
        if not any(path.HasPrefix(parent) for parent in paths_to_disable):
            paths_to_disable.append(path)
    for path in paths_to_disable:
        stage.OverridePrim(path).SetActive(False)
    tool_path = f"{prim_path}/DrAnmarSafePlaneDissectionTool"
    create_prim(tool_path, usd_path=str(TOOL_PAYLOAD_USD), stage=stage)
    select_usd_variants(
        tool_path,
        {
            "scissors_state": cfg.scissors_state,
            "hydro_state": cfg.hydro_state,
            "irrigation_state": cfg.irrigation_state,
            "collection_state": cfg.collection_state,
            "energy_state": cfg.energy_state,
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
    joint = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/dranmar_safeplane_mount_joint")
    joint.CreateBody0Rel().SetTargets(mount_body_paths)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{tool_path}/Links/Mount")])
    joint.CreateLocalPos0Attr().Set(mount_local_pos0)
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot0Attr().Set(mount_local_rot0)
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    return robot


def spawn_franka_with_tool(prim_path: str, cfg: Any, translation=None, orientation=None, **kwargs):
    from isaaclab.sim.utils import clone
    return clone(_spawn_single_franka_with_tool)(
        prim_path, cfg, translation=translation, orientation=orientation, **kwargs
    )


def make_franka_safeplane_dissection_robot_cfg(
    *, prim_path="/World/Robot", scissors_state="fresh", hydro_state="full",
    irrigation_state="full", collection_state="empty", energy_state="ready",
):
    _check(scissors_state, VALID_SCISSORS_STATES, "scissors_state")
    _check(hydro_state, VALID_FLUID_STATES, "hydro_state")
    _check(irrigation_state, VALID_FLUID_STATES, "irrigation_state")
    _check(collection_state, VALID_COLLECTION_STATES, "collection_state")
    _check(energy_state, VALID_ENERGY_STATES, "energy_state")
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

    @configclass
    class FrankaSafePlaneUsdCfg(sim_utils.UsdFileCfg):
        scissors_state: str = "fresh"
        hydro_state: str = "full"
        irrigation_state: str = "full"
        collection_state: str = "empty"
        energy_state: str = "ready"
        func = spawn_franka_with_tool

    cfg = FRANKA_PANDA_CFG.copy()
    cfg.prim_path = prim_path
    cfg.spawn = FrankaSafePlaneUsdCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        variants={"Gripper": "Default", "Mesh": "Performance"},
        scissors_state=scissors_state,
        hydro_state=hydro_state,
        irrigation_state=irrigation_state,
        collection_state=collection_state,
        energy_state=energy_state,
        activate_contact_sensors=True,
        rigid_props=FRANKA_PANDA_CFG.spawn.rigid_props,
        articulation_props=FRANKA_PANDA_CFG.spawn.articulation_props,
    )
    cfg.init_state.joint_pos = {key: value for key, value in cfg.init_state.joint_pos.items() if "finger" not in key}
    cfg.init_state.joint_pos.update({name: 0.0 for name in TOOL_JOINTS.values()})
    cfg.actuators = {key: value for key, value in cfg.actuators.items() if key != "panda_hand"}
    cfg.actuators.update(
        {
            "safeplane_traction": ImplicitActuatorCfg(joint_names_expr=[".*traction_joint"], effort_limit_sim=120.0, velocity_limit_sim=0.18, stiffness=4400.0, damping=150.0),
            "safeplane_pads": ImplicitActuatorCfg(joint_names_expr=[".*pad_pitch_joint", ".*pad_compliance_joint"], effort_limit_sim=48.0, velocity_limit_sim=1.2, stiffness=1100.0, damping=72.0),
            "safeplane_spreader": ImplicitActuatorCfg(joint_names_expr=[".*spreader_joint"], effort_limit_sim=85.0, velocity_limit_sim=0.14, stiffness=3200.0, damping=125.0),
            "safeplane_hydro": ImplicitActuatorCfg(joint_names_expr=["hydro_pitch_joint", "hydro_extension_joint"], effort_limit_sim=80.0, velocity_limit_sim=1.0, stiffness=3600.0, damping=115.0),
            "safeplane_scissors": ImplicitActuatorCfg(joint_names_expr=["scissor_extension_joint", "scissor_guard_joint", "scissor_blade_joint"], effort_limit_sim=120.0, velocity_limit_sim=1.4, stiffness=5200.0, damping=160.0),
            "safeplane_energy": ImplicitActuatorCfg(joint_names_expr=["energy_tip_extension_joint"], effort_limit_sim=70.0, velocity_limit_sim=0.18, stiffness=3500.0, damping=115.0),
            "safeplane_valves": ImplicitActuatorCfg(joint_names_expr=[".*_valve_joint"], effort_limit_sim=25.0, velocity_limit_sim=0.25, stiffness=1600.0, damping=48.0),
        }
    )
    return cfg


def _current_stage(stage=None):
    if stage is not None:
        return stage
    import omni.usd
    return omni.usd.get_context().get_stage()


def _world_transform(stage, path: str):
    from pxr import Usd, UsdGeom
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise ValueError(f"No valid prim at {path}")
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def spawn_tissue_demo(
    prim_path: str = "/World/DrAnmarSafePlaneTissue",
    *, translation=(0.0, 0.0, 0.0), orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
):
    import isaaclab.sim as sim_utils
    cfg = sim_utils.UsdFileCfg(usd_path=str(TISSUE_DEMO_USD))
    return cfg.func(
        prim_path, cfg, translation=translation,
        orientation=_xyzw_from_wxyz(orientation_wxyz),
    )


def _create_surface_material(
    stage, material_path: str, *, youngs_modulus_pa: float,
    poissons_ratio: float, thickness_m: float, dynamic_friction: float,
):
    from pxr import UsdShade
    material = UsdShade.Material.Define(stage, material_path)
    prim = material.GetPrim()
    for schema in (
        "OmniPhysicsBaseMaterialAPI", "OmniPhysicsDeformableMaterialAPI",
        "OmniPhysicsSurfaceDeformableMaterialAPI",
        "PhysxSurfaceDeformableMaterialAPI",
    ):
        try:
            prim.ApplyAPI(schema)
        except Exception:
            pass
    for name, value in {
        "omniphysics:dynamicFriction": dynamic_friction,
        "omniphysics:density": 1060.0,
        "omniphysics:youngsModulus": youngs_modulus_pa,
        "omniphysics:poissonsRatio": poissons_ratio,
        "omniphysics:surfaceThickness": thickness_m,
        "omniphysics:surfaceBendStiffness": 0.0,
        "physxDeformableMaterial:elasticityDamping": 0.16,
        "physxDeformableMaterial:bendDamping": 0.18,
    }.items():
        attribute = prim.GetAttribute(name)
        if attribute:
            attribute.Set(value)
    return material


def apply_tissue_surface_deformables(root_path: str, *, self_collision=False, stage=None):
    stage = _current_stage(stage)
    from omni.physx.scripts import deformableUtils
    from pxr import UsdShade
    paths = []
    specifications = {
        "SuperficialFlap": (95000.0, 0.36, 0.0045, 0.48),
        "TargetBed": (145000.0, 0.38, 0.0055, 0.52),
    }
    for child, values in specifications.items():
        mesh_path = f"{root_path.rstrip('/')}/Anatomy/{child}"
        mesh = stage.GetPrimAtPath(mesh_path)
        if not mesh or not mesh.IsValid():
            raise ValueError(f"No tissue surface at {mesh_path}")
        success = deformableUtils.set_physics_surface_deformable_body(stage, mesh.GetPath())
        if success is False:
            raise RuntimeError(f"Failed to cook surface deformable at {mesh_path}")
        mesh.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
        if mesh.HasAPI("PhysxSurfaceDeformableBodyAPI"):
            mesh.GetAttribute("physxDeformableBody:selfCollision").Set(bool(self_collision))
        material = _create_surface_material(
            stage, f"{root_path}/RuntimeMaterials/{child}", youngs_modulus_pa=values[0],
            poissons_ratio=values[1], thickness_m=values[2],
            dynamic_friction=values[3],
        )
        UsdShade.MaterialBindingAPI.Apply(mesh).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics"
        )
        paths.append(mesh_path)
    return {"root_path": root_path, "mesh_paths": paths, "self_collision": bool(self_collision)}


def create_deformable_attachment(deformable_path: str, target_path: str, attachment_path: str, *, stage=None):
    """Create and verify an overlap-prioritized attachment across Isaac versions."""
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt
    stage = _current_stage(stage)
    if stage.GetPrimAtPath(attachment_path).IsValid():
        stage.RemovePrim(attachment_path)
    definition = Usd.SchemaRegistry().FindConcretePrimDefinition(
        "OmniPhysicsVtxXformAttachment"
    )
    if definition:
        deformable = stage.GetPrimAtPath(deformable_path)
        target = stage.GetPrimAtPath(target_path)
        mesh = UsdGeom.Mesh(deformable)
        points = list(mesh.GetPointsAttr().Get() or [])
        if not deformable.IsValid() or not mesh or not points:
            raise ValueError(f"Attachment source is not a populated mesh: {deformable_path}")
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
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide],
        ).ComputeWorldBound(target).ComputeAlignedRange()
        minimum, maximum = bounds.GetMin(), bounds.GetMax()
        center = (minimum + maximum) * 0.5
        ranked = []
        for index, point in enumerate(points):
            world = mesh_to_world.Transform(Gf.Vec3d(point))
            delta = world - center
            overlaps = all(
                minimum[axis] - 0.0025 <= world[axis] <= maximum[axis] + 0.0025
                for axis in range(3)
            )
            ranked.append((float(Gf.Dot(delta, delta)), index, world, overlaps))
        ranked.sort(key=lambda item: item[0])
        selected = [item for item in ranked if item[3]][:12]
        if len(selected) < 4:
            raise RuntimeError(
                f"Attachment capture volume does not overlap enough deformable "
                f"vertices for {attachment_path}: source={deformable_path}, "
                f"target={target_path}, overlapping={len(selected)}, "
                "required=4, overlap_margin_m=0.0025"
            )
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
        ).Set(Vt.Vec3fArray([
            Gf.Vec3f(world_to_target.Transform(item[2])) for item in selected
        ]))
        attachment.CreateAttribute(
            "omniphysics:attachmentEnabled", Sdf.ValueTypeNames.Bool
        ).Set(True)
        if (
            not attachment.IsValid()
            or attachment.GetTypeName() != "OmniPhysicsVtxXformAttachment"
            or not attachment.GetRelationship("omniphysics:src0").GetTargets()
            or not attachment.GetRelationship("omniphysics:src1").GetTargets()
        ):
            raise RuntimeError(f"Could not author attachment {attachment_path}")
        return "OmniPhysicsVtxXformAttachment"

    import omni.kit.commands
    def execute_and_verify(command: str, **kwargs) -> str:
        omni.kit.commands.execute(command, **kwargs)
        if not stage.GetPrimAtPath(attachment_path).IsValid():
            raise RuntimeError(f"{command} did not author {attachment_path}")
        return command
    try:
        return execute_and_verify(
            "CreateAutoDeformableAttachment",
            target_attachment_path=Sdf.Path(attachment_path),
            attachable0_path=Sdf.Path(deformable_path),
            attachable1_path=Sdf.Path(target_path),
        )
    except Exception as current_error:
        if stage.GetPrimAtPath(attachment_path).IsValid():
            stage.RemovePrim(attachment_path)
        try:
            return execute_and_verify(
                "CreatePhysicsAttachment",
                target_attachment_path=Sdf.Path(attachment_path),
                actor0_path=Sdf.Path(deformable_path),
                actor1_path=Sdf.Path(target_path),
            )
        except Exception as legacy_error:
            raise RuntimeError(
                f"Could not create {attachment_path}: "
                f"current={current_error!r}; legacy={legacy_error!r}"
            ) from legacy_error


def remove_prims(paths: Iterable[str], *, stage=None):
    stage = _current_stage(stage)
    for path in paths:
        if stage.GetPrimAtPath(path).IsValid():
            stage.RemovePrim(path)


def remove_or_deactivate_prim(path: str, *, stage=None) -> bool:
    """Disable physics authored through either a local spec or a reference."""
    stage = _current_stage(stage)
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid() or not prim.IsActive():
        return False
    stage.RemovePrim(path)
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid() and prim.IsActive():
        stage.OverridePrim(path).SetActive(False)
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid() and prim.IsActive():
        raise RuntimeError(f"Could not remove or deactivate {path}")
    return True


def anchor_target_bed(root_path: str, *, stage=None) -> list[str]:
    """Ground the cooked target bed through two explicit kinematic fixtures."""
    from pxr import UsdPhysics
    stage = _current_stage(stage)
    root_path = root_path.rstrip("/")
    scope = f"{root_path}/RuntimeFixtureAttachments"
    stage.DefinePrim(scope, "Scope")
    created = []
    try:
        for side in ("Left", "Right"):
            target_path = f"{root_path}/TargetBedFixture{side}"
            target = stage.GetPrimAtPath(target_path)
            if not target.IsValid():
                raise ValueError(f"Target-bed fixture is missing: {target_path}")
            rigid = UsdPhysics.RigidBodyAPI.Apply(target)
            rigid.CreateRigidBodyEnabledAttr(True)
            rigid.CreateKinematicEnabledAttr(True)
            attachment = f"{scope}/{side.lower()}"
            create_deformable_attachment(
                f"{root_path}/Anatomy/TargetBed", target_path, attachment,
                stage=stage,
            )
            created.append(attachment)
    except Exception:
        remove_prims(created, stage=stage)
        raise
    return created


def load_dissection_topology() -> dict[str, Any]:
    return json.loads(DISSECTION_TOPOLOGY_PATH.read_text(encoding="utf-8"))


@dataclass
class TractionCell:
    side: str
    index: int
    attachment_path: str
    released: bool = False


@dataclass
class BilateralTractionController:
    tool_path: str
    tissue_root: str
    cells: list[TractionCell] = field(default_factory=list)
    nominal_force_n: float = 1.4
    soft_force_limit_n: float = 3.0
    hard_force_limit_n: float = 5.0

    def capture(self, *, stage=None):
        stage = _current_stage(stage)
        superficial = f"{self.tissue_root.rstrip('/')}/Anatomy/SuperficialFlap"
        scope = f"{self.tissue_root.rstrip('/')}/RuntimeTractionAttachments"
        stage.DefinePrim(scope, "Scope")
        created: list[str] = []
        cells: list[TractionCell] = []
        try:
            for side in ("left", "right"):
                link = side.title()
                for index in range(4):
                    target = f"{self.tool_path.rstrip('/')}/Links/{link}TractionPad/Collisions/CaptureCell_{index:02d}"
                    attachment = f"{scope}/{side}_{index:02d}"
                    create_deformable_attachment(superficial, target, attachment, stage=stage)
                    created.append(attachment)
                    cells.append(TractionCell(side, index, attachment))
        except Exception:
            remove_prims(created, stage=stage)
            raise
        self.cells = cells
        return list(cells)

    def release_side(self, side: str, *, stage=None):
        stage = _current_stage(stage)
        released = []
        for cell in self.cells:
            if cell.side == side and not cell.released:
                remove_prims([cell.attachment_path], stage=stage)
                cell.released = True
                released.append(cell.index)
        return released

    def update_force(self, left_force_n: float, right_force_n: float, *, stage=None):
        stage = _current_stage(stage)
        events = []
        for side, force in (
            ("left", _finite_nonnegative(left_force_n, "left_force_n")),
            ("right", _finite_nonnegative(right_force_n, "right_force_n")),
        ):
            active = [cell for cell in self.cells if cell.side == side and not cell.released]
            if force >= self.hard_force_limit_n:
                for cell in active:
                    remove_prims([cell.attachment_path], stage=stage)
                    cell.released = True
                events.append({"side": side, "event": "hard_release", "force_n": force})
            elif force >= self.soft_force_limit_n and active:
                cell = sorted(active, key=lambda value: abs(value.index - 1.5), reverse=True)[0]
                remove_prims([cell.attachment_path], stage=stage)
                cell.released = True
                events.append({"side": side, "event": "peripheral_cell_release", "cell": cell.index, "force_n": force})
        return events

    def snapshot(self):
        return {
            side: {"active_cells": [cell.index for cell in self.cells if cell.side == side and not cell.released]}
            for side in ("left", "right")
        }


@dataclass
class BridgeRuntimeState:
    index: int
    bridge_class: str
    position: tuple[float, float, float]
    recommended_mode: str
    mechanical_threshold_j: float
    hydro_threshold_ml: float
    energy_threshold_j: float
    nearest_structure: str | None
    clearance_m: float
    attachment_paths: list[str] = field(default_factory=list)
    mechanical_work_j: float = 0.0
    hydro_volume_ml: float = 0.0
    energy_dose_j: float = 0.0
    released: bool = False
    release_mode: str | None = None


@dataclass
class AdhesionBridgeController:
    tissue_root: str
    states: dict[int, BridgeRuntimeState] = field(default_factory=dict)

    def __post_init__(self):
        topology = load_dissection_topology()
        self.states = {
            int(item["index"]): BridgeRuntimeState(
                index=int(item["index"]),
                bridge_class=str(item["bridge_class"]),
                position=tuple(float(value) for value in item["position_m"]),
                recommended_mode=str(item["recommended_mode"]),
                mechanical_threshold_j=float(item["thresholds"]["mechanical_work_j"]),
                hydro_threshold_ml=float(item["thresholds"]["hydro_volume_ml"]),
                energy_threshold_j=float(item["thresholds"]["energy_dose_j"]),
                nearest_structure=item.get("nearest_structure"),
                clearance_m=float(item.get("clearance_m", math.inf)),
            )
            for item in topology["adhesion_bridges"]
        }

    def engage(self, *, stage=None):
        stage = _current_stage(stage)
        superficial = f"{self.tissue_root.rstrip('/')}/Anatomy/SuperficialFlap"
        target = f"{self.tissue_root.rstrip('/')}/Anatomy/TargetBed"
        scope = f"{self.tissue_root.rstrip('/')}/RuntimeBridgeAttachments"
        stage.DefinePrim(scope, "Scope")
        created = []
        try:
            for index, state in self.states.items():
                base = f"{self.tissue_root.rstrip('/')}/AdhesionBridges/Bridge_{index:02d}"
                paths = []
                for label, actor, anchor in (
                    ("upper", superficial, f"{base}/UpperAnchor"),
                    ("lower", target, f"{base}/LowerAnchor"),
                ):
                    attachment = f"{scope}/bridge_{index:02d}_{label}"
                    create_deformable_attachment(actor, anchor, attachment, stage=stage)
                    paths.append(attachment)
                    created.append(attachment)
                state.attachment_paths = paths
        except Exception:
            remove_prims(created, stage=stage)
            raise
        return self.snapshot()

    def release(self, index: int, mode: str, *, stage=None):
        state = self.states[int(index)]
        if state.released:
            return False
        stage = _current_stage(stage)
        joint_path = f"{self.tissue_root.rstrip('/')}/AdhesionBridges/Bridge_{index:02d}/ContinuityJoint"
        remove_or_deactivate_prim(joint_path, stage=stage)
        state.released = True
        state.release_mode = str(mode)
        return True

    @staticmethod
    def _weight(point: Sequence[float], center: Sequence[float], radius_m: float) -> float:
        values = tuple(_finite(v, "point") for v in point)
        center_values = tuple(_finite(v, "center") for v in center)
        radius = _finite_nonnegative(radius_m, "radius_m")
        distance = math.dist(values, center_values)
        return max(0.0, 1.0 - distance / max(radius, 1.0e-9))

    def apply_blunt_work(self, local_position: Sequence[float], work_j: float, *, radius_m: float = 0.018, stage=None):
        released = []
        for state in self.states.values():
            if state.released:
                continue
            weight = self._weight(local_position, state.position, radius_m)
            if weight <= 0:
                continue
            class_scale = 1.0 if state.bridge_class == "loose_connective_fibre" else 0.55 if state.bridge_class == "vascularized_adhesion" else 0.28
            state.mechanical_work_j += _finite_nonnegative(work_j, "work_j") * weight * class_scale
            hydration_scale = max(0.28, 1.0 - 0.72 * state.hydro_volume_ml / max(state.hydro_threshold_ml, 1.0e-9))
            if state.mechanical_work_j >= state.mechanical_threshold_j * hydration_scale:
                if self.release(state.index, "blunt_spreading", stage=stage):
                    released.append(state.index)
        return released

    def apply_hydro_volume(self, local_position: Sequence[float], volume_ml: float, *, radius_m: float = 0.024, stage=None):
        released = []
        for state in self.states.values():
            if state.released:
                continue
            weight = self._weight(local_position, state.position, radius_m)
            if weight <= 0:
                continue
            class_scale = 1.0 if state.bridge_class == "loose_connective_fibre" else 0.62 if state.bridge_class == "vascularized_adhesion" else 0.22
            state.hydro_volume_ml += _finite_nonnegative(volume_ml, "volume_ml") * weight * class_scale
            if state.hydro_volume_ml >= state.hydro_threshold_ml and state.bridge_class != "dense_fibrous_band":
                if self.release(state.index, "hydrodissection", stage=stage):
                    released.append(state.index)
        return released

    def apply_energy(self, local_position: Sequence[float], energy_j: float, *, radius_m: float = 0.012, stage=None):
        released = []
        for state in self.states.values():
            if state.released:
                continue
            weight = self._weight(local_position, state.position, radius_m)
            if weight <= 0:
                continue
            state.energy_dose_j += _finite_nonnegative(energy_j, "energy_j") * weight
            if state.energy_dose_j >= state.energy_threshold_j:
                if self.release(state.index, "low_energy_dissection", stage=stage):
                    released.append(state.index)
        return released

    def nearest_unreleased(self, local_position: Sequence[float], radius_m: float = 0.009):
        local_position = tuple(_finite(value, "local_position") for value in local_position)
        radius_m = _finite_nonnegative(radius_m, "radius_m")
        candidates = [
            (math.dist(tuple(local_position), state.position), state)
            for state in self.states.values()
            if not state.released
        ]
        if not candidates:
            return None
        distance, state = min(candidates, key=lambda item: item[0])
        return state if distance <= radius_m else None

    def cut_nearest(self, local_position: Sequence[float], *, guard_retracted: bool, blade_closed: bool, stage=None):
        if not guard_retracted:
            return {"released": False, "reason": "scissor_guard_not_retracted"}
        if not blade_closed:
            return {"released": False, "reason": "scissor_blade_not_closed"}
        state = self.nearest_unreleased(local_position)
        if state is None:
            return {"released": False, "reason": "no_bridge_in_cut_volume"}
        released = self.release(state.index, "guarded_scissors", stage=stage)
        return {"released": released, "bridge_index": state.index, "bridge_class": state.bridge_class}

    @property
    def release_fraction(self):
        if not self.states:
            return 0.0
        return sum(state.released for state in self.states.values()) / len(self.states)

    def snapshot(self):
        return {
            "release_fraction": self.release_fraction,
            "released_count": sum(state.released for state in self.states.values()),
            "total_count": len(self.states),
            "bridges": {
                index: {
                    "class": state.bridge_class,
                    "released": state.released,
                    "release_mode": state.release_mode,
                    "mechanical_work_j": state.mechanical_work_j,
                    "hydro_volume_ml": state.hydro_volume_ml,
                    "energy_dose_j": state.energy_dose_j,
                }
                for index, state in self.states.items()
            },
        }


def _distance_point_segment(point, a, b) -> float:
    p = tuple(float(v) for v in point)
    a = tuple(float(v) for v in a)
    b = tuple(float(v) for v in b)
    ab = tuple(b[i] - a[i] for i in range(3))
    ap = tuple(p[i] - a[i] for i in range(3))
    denom = sum(value * value for value in ab)
    amount = 0.0 if denom <= 1.0e-18 else max(0.0, min(1.0, sum(ap[i] * ab[i] for i in range(3)) / denom))
    closest = tuple(a[i] + amount * ab[i] for i in range(3))
    return math.dist(p, closest)


@dataclass
class ProtectedStructureState:
    name: str
    intact: bool = True
    injury_mechanism: str | None = None
    blood_loss_ml: float = 0.0
    duct_leak_ml: float = 0.0
    nerve_conduction_fraction: float = 1.0


@dataclass
class ProtectedStructureController:
    tissue_root: str
    states: dict[str, ProtectedStructureState] = field(
        default_factory=lambda: {name: ProtectedStructureState(name) for name in PROTECTED_STRUCTURES}
    )
    attachments: list[str] = field(default_factory=list)

    def topology(self):
        return load_dissection_topology()["protected_structures"]

    def attach_to_target_bed(self, *, stage=None):
        stage = _current_stage(stage)
        target = f"{self.tissue_root.rstrip('/')}/Anatomy/TargetBed"
        scope = f"{self.tissue_root.rstrip('/')}/RuntimeProtectedStructureAttachments"
        stage.DefinePrim(scope, "Scope")
        created = []
        try:
            for name in PROTECTED_STRUCTURES:
                root = f"{self.tissue_root.rstrip('/')}/ProtectedStructures/{name.title()}"
                for segment in ("ProximalSegment", "DistalSegment"):
                    attachment = f"{scope}/{name}_{segment.lower()}"
                    create_deformable_attachment(target, f"{root}/Links/{segment}", attachment, stage=stage)
                    created.append(attachment)
        except Exception:
            remove_prims(created, stage=stage)
            raise
        self.attachments = created
        return list(created)

    def distance_to(self, local_position: Sequence[float], structure: str) -> float:
        if structure not in PROTECTED_STRUCTURES:
            raise KeyError(structure)
        local_position = tuple(_finite(value, "local_position") for value in local_position)
        points = self.topology()[structure]["centerline_m"]
        return min(_distance_point_segment(local_position, a, b) for a, b in zip(points[:-1], points[1:]))

    def nearest(self, local_position: Sequence[float]):
        values = {name: self.distance_to(local_position, name) for name in PROTECTED_STRUCTURES}
        name = min(values, key=values.get)
        return name, values[name], values

    def evaluate_action(self, local_position: Sequence[float], modality: str):
        clearances = {
            "blunt": 0.0025,
            "hydro": 0.0030,
            "scissors": 0.0050,
            "energy": 0.0070,
        }
        if modality not in clearances:
            raise ValueError(f"Unknown modality {modality!r}")
        minimum = clearances[modality]
        name, distance, all_distances = self.nearest(local_position)
        reasons = []
        if distance < minimum:
            reasons.append(f"{name}_clearance_below_{minimum:.4f}_m")
        if not self.states[name].intact:
            reasons.append(f"{name}_already_injured")
        return {
            "authorized": not reasons,
            "nearest_structure": name,
            "distance_m": distance,
            "minimum_clearance_m": minimum,
            "all_distances_m": all_distances,
            "reasons": reasons,
        }

    def injure(self, structure: str, mechanism: str, *, stage=None):
        if structure not in self.states:
            raise KeyError(structure)
        state = self.states[structure]
        if not state.intact:
            return False
        stage = _current_stage(stage)
        root = f"{self.tissue_root.rstrip('/')}/ProtectedStructures/{structure.title()}"
        joint = f"{root}/Joints/ContinuityJoint"
        remove_or_deactivate_prim(joint, stage=stage)
        prim = stage.GetPrimAtPath(root)
        if prim and prim.IsValid():
            variants = prim.GetVariantSets().GetVariantSet("integrity")
            if variants:
                variants.SetVariantSelection("injured")
        state.intact = False
        state.injury_mechanism = str(mechanism)
        if structure == "nerve":
            state.nerve_conduction_fraction = 0.0
        return True

    def update_complication(self, dt: float, *, pressure_pa: float = 12000.0, duct_pressure_pa: float = 900.0):
        dt = _finite_nonnegative(dt, "dt")
        pressure_pa = _finite_nonnegative(pressure_pa, "pressure_pa")
        duct_pressure_pa = _finite_nonnegative(duct_pressure_pa, "duct_pressure_pa")
        vessel = self.states["vessel"]
        duct = self.states["duct"]
        if not vessel.intact:
            vessel.blood_loss_ml += 0.45 * math.sqrt(max(pressure_pa, 0.0) / 12000.0) * dt
        if not duct.intact:
            duct.duct_leak_ml += 0.08 * math.sqrt(max(duct_pressure_pa, 0.0) / 900.0) * dt
        return self.snapshot()

    def snapshot(self):
        return {
            name: {
                "intact": state.intact,
                "injury_mechanism": state.injury_mechanism,
                "blood_loss_ml": state.blood_loss_ml,
                "duct_leak_ml": state.duct_leak_ml,
                "nerve_conduction_fraction": state.nerve_conduction_fraction,
            }
            for name, state in self.states.items()
        }


@dataclass
class FluidLedger:
    reservoir_capacity_ml: float = 35.0
    reservoir_ml: float = 35.0
    collection_capacity_ml: float = 55.0
    emitted_ml: float = 0.0
    aspirated_ml: float = 0.0
    absorbed_ml: float = 0.0
    spilled_ml: float = 0.0
    active_particle_ml: float = 0.0

    def __post_init__(self):
        for name in (
            "reservoir_capacity_ml", "reservoir_ml", "collection_capacity_ml",
            "emitted_ml", "aspirated_ml", "absorbed_ml", "spilled_ml",
            "active_particle_ml",
        ):
            setattr(self, name, _finite_nonnegative(getattr(self, name), name))
        if self.reservoir_ml > self.reservoir_capacity_ml:
            raise ValueError("reservoir_ml cannot exceed reservoir_capacity_ml")

    def emit(self, requested_ml: float) -> float:
        amount = min(_finite_nonnegative(requested_ml, "requested_ml"), self.reservoir_ml)
        self.reservoir_ml -= amount
        self.emitted_ml += amount
        self.active_particle_ml += amount
        return amount

    def aspirate(self, requested_ml: float) -> float:
        capacity = max(0.0, self.collection_capacity_ml - self.aspirated_ml)
        amount = min(_finite_nonnegative(requested_ml, "requested_ml"), self.active_particle_ml, capacity)
        self.active_particle_ml -= amount
        self.aspirated_ml += amount
        return amount

    def absorb(self, requested_ml: float) -> float:
        amount = min(_finite_nonnegative(requested_ml, "requested_ml"), self.active_particle_ml)
        self.active_particle_ml -= amount
        self.absorbed_ml += amount
        return amount

    def spill(self, requested_ml: float) -> float:
        amount = min(_finite_nonnegative(requested_ml, "requested_ml"), self.active_particle_ml)
        self.active_particle_ml -= amount
        self.spilled_ml += amount
        return amount

    @property
    def balance_error_ml(self):
        return self.reservoir_capacity_ml - (
            self.reservoir_ml + self.active_particle_ml + self.aspirated_ml + self.absorbed_ml + self.spilled_ml
        )

    def snapshot(self):
        return {
            "reservoir_ml": self.reservoir_ml,
            "emitted_ml": self.emitted_ml,
            "active_particle_ml": self.active_particle_ml,
            "aspirated_ml": self.aspirated_ml,
            "absorbed_ml": self.absorbed_ml,
            "spilled_ml": self.spilled_ml,
            "balance_error_ml": self.balance_error_ml,
        }


def ensure_dissection_particle_system(
    *, stage=None, physics_scene_path="/World/physicsScene",
    root_path="/World/DrAnmarDissectionFluid", particle_radius_m=PARTICLE_RADIUS_M,
):
    particle_radius_m = _finite_nonnegative(particle_radius_m, "particle_radius_m")
    if particle_radius_m <= 0.0:
        raise ValueError("particle_radius_m must be positive")
    stage = _current_stage(stage)
    from omni.physx.scripts import particleUtils, physicsUtils
    from pxr import Sdf, UsdGeom, UsdPhysics

    stage.DefinePrim(root_path, "Scope")
    if not stage.GetPrimAtPath(physics_scene_path).IsValid():
        UsdPhysics.Scene.Define(stage, physics_scene_path)
    system_path = f"{root_path}/ParticleSystem"
    set_path = f"{root_path}/Particles"
    material_path = f"{root_path}/PBDMaterial"
    if not stage.GetPrimAtPath(material_path).IsValid():
        particleUtils.add_pbd_particle_material(
            stage, Sdf.Path(material_path), cohesion=0.0015, viscosity=0.0015,
            surface_tension=0.0035, friction=0.04,
        )
    if not stage.GetPrimAtPath(system_path).IsValid():
        particleUtils.add_physx_particle_system(
            stage=stage, particle_system_path=Sdf.Path(system_path),
            simulation_owner=Sdf.Path(physics_scene_path),
            particle_contact_offset=particle_radius_m * 1.15,
            rest_offset=particle_radius_m * 0.90,
            solid_rest_offset=particle_radius_m * 1.80,
            fluid_rest_offset=particle_radius_m * 0.92,
        )
        physicsUtils.add_physics_material_to_prim(
            stage, stage.GetPrimAtPath(system_path), Sdf.Path(material_path)
        )
    if not stage.GetPrimAtPath(set_path).IsValid():
        particleUtils.add_physx_particleset_points(
            stage, Sdf.Path(set_path), [], [], [], Sdf.Path(system_path),
            True, True, 0, 1.0, particle_radius_m * 2.0,
        )
        UsdGeom.Points(stage.GetPrimAtPath(set_path)).GetWidthsAttr().Set([])
    return {"root_path": root_path, "particle_system_path": system_path, "particle_set_path": set_path}


def emit_hydro_burst(
    tool_path: str, ledger: FluidLedger, *, requested_ml=0.22, jet_speed_m_s=1.35,
    stage=None, particle_set_path="/World/DrAnmarDissectionFluid/Particles",
):
    stage = _current_stage(stage)
    from pxr import Gf, UsdGeom, Vt
    points = UsdGeom.Points(stage.GetPrimAtPath(particle_set_path))
    if not points:
        raise ValueError(f"No particle set at {particle_set_path}")
    jet_speed_m_s = _finite_nonnegative(jet_speed_m_s, "jet_speed_m_s")
    available = ledger.emit(requested_ml)
    count = int(available / PARTICLE_VOLUME_ML)
    count -= count % 7
    actual_ml = count * PARTICLE_VOLUME_ML
    remainder = available - actual_ml
    ledger.reservoir_ml += remainder
    ledger.emitted_ml -= remainder
    ledger.active_particle_ml -= remainder
    if count <= 0:
        return {"particle_count": 0, "emitted_ml": 0.0}
    transform = _world_transform(stage, frame_path(tool_path, "hydro_nozzle_tip"))
    current_positions = list(points.GetPointsAttr().Get() or [])
    current_velocities = list(points.GetVelocitiesAttr().Get() or [])
    current_widths = list(points.GetWidthsAttr().Get() or [])
    per_jet = count // 7
    for jet in range(7):
        angle = 2.0 * math.pi * jet / 7.0
        origin_local = Gf.Vec3d(0.0012 * math.cos(angle), 0.0012 * math.sin(angle), 0.0)
        direction_local = Gf.Vec3d(0.10 * math.cos(angle), 0.10 * math.sin(angle), 1.0).GetNormalized()
        origin_world = transform.Transform(origin_local)
        direction_world = transform.TransformDir(direction_local).GetNormalized()
        for index in range(per_jet):
            jitter = (index % 5 - 2) * 0.00010
            position = origin_world + direction_world * jitter
            current_positions.append(Gf.Vec3f(position))
            current_velocities.append(Gf.Vec3f(direction_world * jet_speed_m_s))
            current_widths.append(PARTICLE_RADIUS_M * 2.0)
    points.GetPointsAttr().Set(Vt.Vec3fArray(current_positions))
    points.GetVelocitiesAttr().Set(Vt.Vec3fArray(current_velocities))
    points.GetWidthsAttr().Set(current_widths)
    return {"particle_count": count, "emitted_ml": actual_ml, "particle_set_path": particle_set_path}


@dataclass
class SuctionFieldController:
    capture_radius_m: float = 0.031
    capture_depth_m: float = 0.040
    throat_radius_m: float = 0.007
    max_acceleration_m_s2: float = 20.0

    def update_particles(
        self, tool_path: str, ledger: FluidLedger, *, dt: float, opening=1.0,
        stage=None, particle_set_path="/World/DrAnmarDissectionFluid/Particles",
    ):
        stage = _current_stage(stage)
        from pxr import Gf, UsdGeom, Vt
        points = UsdGeom.Points(stage.GetPrimAtPath(particle_set_path))
        positions = list(points.GetPointsAttr().Get() or [])
        velocities = list(points.GetVelocitiesAttr().Get() or [])
        widths = list(points.GetWidthsAttr().Get() or [PARTICLE_RADIUS_M * 2.0] * len(positions))
        if not positions:
            return {"active": 0, "captured": 0, "aspirated_ml": 0.0}
        dt = _finite_nonnegative(dt, "dt")
        opening = max(0.0, min(1.0, _finite(opening, "opening")))
        capture_transform = _world_transform(stage, frame_path(tool_path, "suction_center"))
        inverse = capture_transform.GetInverse()
        throat = capture_transform.ExtractTranslation()
        remaining_collection_ml = max(
            0.0, ledger.collection_capacity_ml - ledger.aspirated_ml
        )
        removable_particle_count = int(
            math.floor(
                (
                    min(ledger.active_particle_ml, remaining_collection_ml)
                    + 1.0e-12
                )
                / PARTICLE_VOLUME_ML
            )
        )
        kept_positions, kept_velocities, kept_widths = [], [], []
        captured = 0
        capacity_blocked = 0
        for position, velocity, width in zip(positions, velocities, widths):
            world = Gf.Vec3d(position)
            local = inverse.Transform(world)
            radial = math.hypot(local[0], local[1])
            to_throat = throat - world
            distance = max(float(to_throat.GetLength()), 1.0e-8)
            if opening > 0 and distance <= self.throat_radius_m:
                if captured < removable_particle_count:
                    captured += 1
                    continue
                capacity_blocked += 1
            new_velocity = Gf.Vec3d(velocity)
            if radial <= self.capture_radius_m and abs(local[2]) <= self.capture_depth_m and opening > 0:
                new_velocity += to_throat / distance * (opening * self.max_acceleration_m_s2 * max(0.0, dt))
            kept_positions.append(Gf.Vec3f(world))
            kept_velocities.append(Gf.Vec3f(new_velocity))
            kept_widths.append(float(width))
        points.GetPointsAttr().Set(Vt.Vec3fArray(kept_positions))
        points.GetVelocitiesAttr().Set(Vt.Vec3fArray(kept_velocities))
        points.GetWidthsAttr().Set(kept_widths)
        aspirated = ledger.aspirate(captured * PARTICLE_VOLUME_ML)
        expected_aspirated = captured * PARTICLE_VOLUME_ML
        if not math.isclose(aspirated, expected_aspirated, rel_tol=0.0, abs_tol=1.0e-12):
            raise RuntimeError(
                "Particle removal and collection ledger diverged: "
                f"removed_ml={expected_aspirated}, accounted_ml={aspirated}"
            )
        return {
            "active": len(kept_positions),
            "captured": captured,
            "aspirated_ml": aspirated,
            "capacity_blocked": capacity_blocked,
            "remaining_collection_ml": max(
                0.0, ledger.collection_capacity_ml - ledger.aspirated_ml
            ),
        }


@dataclass
class EnergyDissectionState:
    temperature_c: float = 37.0
    delivered_energy_j: float = 0.0
    smoke_generated_ml: float = 0.0
    overtemperature: bool = False


@dataclass
class LowEnergyDissectionController:
    target_temperature_c: float = 72.0
    maximum_temperature_c: float = 95.0
    maximum_power_w: float = 22.0
    heat_capacity_j_k: float = 1.1
    heat_loss_w_k: float = 0.18
    state: EnergyDissectionState = field(default_factory=EnergyDissectionState)

    def update(self, dt: float, contact_force_n: float, requested_power_w: float | None = None):
        dt = _finite_nonnegative(dt, "dt")
        force_scale = max(
            0.0, min(1.0, _finite_nonnegative(contact_force_n, "contact_force_n") / 1.5)
        )
        if requested_power_w is None:
            requested_power_w = max(0.0, min(self.maximum_power_w, (self.target_temperature_c - self.state.temperature_c) * 0.8))
        power = 0.0 if self.state.overtemperature else max(
            0.0, min(self.maximum_power_w, _finite_nonnegative(requested_power_w, "requested_power_w"))
        )
        absorbed = power * (0.30 + 0.70 * force_scale)
        loss = self.heat_loss_w_k * max(0.0, self.state.temperature_c - 37.0)
        self.state.temperature_c += (absorbed - loss) * dt / max(self.heat_capacity_j_k, 1.0e-9)
        energy = absorbed * dt
        self.state.delivered_energy_j += energy
        self.state.smoke_generated_ml += max(0.0, self.state.temperature_c - 58.0) * energy * 0.0004
        self.state.overtemperature = self.state.temperature_c > self.maximum_temperature_c
        return {"energy_j": energy, "state": self.state}


@dataclass
class ScissorsInterlockController:
    minimum_guard_retraction_m: float = 0.009
    minimum_structure_clearance_m: float = 0.005
    violations: int = 0

    def evaluate(self, local_position: Sequence[float], guard_retraction_m: float, protected: ProtectedStructureController):
        safety = protected.evaluate_action(local_position, "scissors")
        reasons = list(safety["reasons"])
        if _finite_nonnegative(guard_retraction_m, "guard_retraction_m") < self.minimum_guard_retraction_m:
            reasons.append("guard_not_fully_retracted")
        return {**safety, "authorized": not reasons, "reasons": reasons}

    def request_cut(
        self, local_position: Sequence[float], guard_retraction_m: float,
        bridges: AdhesionBridgeController, protected: ProtectedStructureController,
        *, override=False, stage=None,
    ):
        result = self.evaluate(local_position, guard_retraction_m, protected)
        if not result["authorized"] and not override:
            self.violations += 1
            return {**result, "released": False}
        if not result["authorized"] and override:
            protected.injure(result["nearest_structure"], "scissor_override", stage=stage)
        cut = bridges.cut_nearest(local_position, guard_retracted=True, blade_closed=True, stage=stage)
        return {**result, **cut, "override": bool(override)}


@dataclass
class DissectionCompletionVerifier:
    bridges: AdhesionBridgeController
    protected: ProtectedStructureController

    def evaluate(self, *, visibility_fraction: float, traction_stable: bool):
        visibility_fraction = _finite(visibility_fraction, "visibility_fraction")
        if not 0.0 <= visibility_fraction <= 1.0:
            raise ValueError("visibility_fraction must be between 0 and 1")
        bridge_snapshot = self.bridges.snapshot()
        structure_snapshot = self.protected.snapshot()
        protected_intact = all(value["intact"] for value in structure_snapshot.values())
        residual = bridge_snapshot["total_count"] - bridge_snapshot["released_count"]
        complete = (
            residual == 0
            and protected_intact
            and visibility_fraction >= 0.90
            and bool(traction_stable)
        )
        return {
            "complete": complete,
            "released_bridge_fraction": bridge_snapshot["release_fraction"],
            "residual_bridge_count": residual,
            "protected_structures_intact": protected_intact,
            "visibility_fraction": visibility_fraction,
            "traction_stable": bool(traction_stable),
            "complications": [name for name, value in structure_snapshot.items() if not value["intact"]],
        }


PHASE_TARGETS = {
    "inspect": {name: 0.0 for name in TOOL_JOINTS.values()},
    "capture": {
        **{name: 0.0 for name in TOOL_JOINTS.values()},
        "left_traction_joint": -0.006, "right_traction_joint": 0.006,
        "left_pad_pitch_joint": math.radians(-8), "right_pad_pitch_joint": math.radians(8),
        "left_pad_compliance_joint": 0.003, "right_pad_compliance_joint": 0.003,
        "suction_valve_joint": 0.002,
    },
    "traction": {
        **{name: 0.0 for name in TOOL_JOINTS.values()},
        "left_traction_joint": -0.026, "right_traction_joint": 0.026,
        "left_pad_pitch_joint": math.radians(-14), "right_pad_pitch_joint": math.radians(14),
        "left_pad_compliance_joint": 0.004, "right_pad_compliance_joint": 0.004,
        "suction_valve_joint": 0.003,
    },
    "blunt": {
        **{name: 0.0 for name in TOOL_JOINTS.values()},
        "left_traction_joint": -0.028, "right_traction_joint": 0.028,
        "left_pad_pitch_joint": math.radians(-16), "right_pad_pitch_joint": math.radians(16),
        "left_pad_compliance_joint": 0.004, "right_pad_compliance_joint": 0.004,
        "left_spreader_joint": -0.018, "right_spreader_joint": 0.018,
        "suction_valve_joint": 0.004, "irrigation_valve_joint": 0.002,
    },
    "hydro": {
        **{name: 0.0 for name in TOOL_JOINTS.values()},
        "left_traction_joint": -0.030, "right_traction_joint": 0.030,
        "left_pad_pitch_joint": math.radians(-17), "right_pad_pitch_joint": math.radians(17),
        "left_pad_compliance_joint": 0.004, "right_pad_compliance_joint": 0.004,
        "left_spreader_joint": -0.016, "right_spreader_joint": 0.016,
        "hydro_pitch_joint": math.radians(12), "hydro_extension_joint": 0.042,
        "hydro_valve_joint": 0.007, "suction_valve_joint": 0.005, "irrigation_valve_joint": 0.002,
    },
    "scissors": {
        **{name: 0.0 for name in TOOL_JOINTS.values()},
        "left_traction_joint": -0.031, "right_traction_joint": 0.031,
        "left_pad_pitch_joint": math.radians(-18), "right_pad_pitch_joint": math.radians(18),
        "left_pad_compliance_joint": 0.004, "right_pad_compliance_joint": 0.004,
        "left_spreader_joint": -0.014, "right_spreader_joint": 0.014,
        "scissor_extension_joint": 0.046, "scissor_guard_joint": -0.010,
        "scissor_blade_joint": math.radians(30), "suction_valve_joint": 0.006,
    },
    "energy": {
        **{name: 0.0 for name in TOOL_JOINTS.values()},
        "left_traction_joint": -0.031, "right_traction_joint": 0.031,
        "left_pad_pitch_joint": math.radians(-18), "right_pad_pitch_joint": math.radians(18),
        "left_pad_compliance_joint": 0.004, "right_pad_compliance_joint": 0.004,
        "energy_tip_extension_joint": 0.042, "suction_valve_joint": 0.006,
    },
    "verify": {
        **{name: 0.0 for name in TOOL_JOINTS.values()},
        "left_traction_joint": -0.032, "right_traction_joint": 0.032,
        "left_pad_pitch_joint": math.radians(-18), "right_pad_pitch_joint": math.radians(18),
        "left_pad_compliance_joint": 0.003, "right_pad_compliance_joint": 0.003,
        "suction_valve_joint": 0.003,
    },
    "complete": {name: 0.0 for name in TOOL_JOINTS.values()},
    "abort": {
        **{name: 0.0 for name in TOOL_JOINTS.values()},
        "suction_valve_joint": 0.008, "irrigation_valve_joint": 0.005,
    },
}


def phase_targets(phase: str):
    try:
        return dict(PHASE_TARGETS[phase])
    except KeyError as exc:
        raise KeyError(f"Unknown SafePlane phase {phase!r}") from exc


@dataclass
class SafePlaneDissectionSequenceController:
    tissue_root: str
    tool_path: str
    phase: str = "inspect"
    history: list[str] = field(default_factory=list)
    traction: BilateralTractionController = field(init=False)
    bridges: AdhesionBridgeController = field(init=False)
    protected: ProtectedStructureController = field(init=False)
    hydro_ledger: FluidLedger = field(default_factory=FluidLedger)
    suction: SuctionFieldController = field(default_factory=SuctionFieldController)
    energy: LowEnergyDissectionController = field(default_factory=LowEnergyDissectionController)
    scissors: ScissorsInterlockController = field(default_factory=ScissorsInterlockController)
    verifier: DissectionCompletionVerifier = field(init=False)

    def __post_init__(self):
        self.traction = BilateralTractionController(self.tool_path, self.tissue_root)
        self.bridges = AdhesionBridgeController(self.tissue_root)
        self.protected = ProtectedStructureController(self.tissue_root)
        self.verifier = DissectionCompletionVerifier(self.bridges, self.protected)

    def transition(self, phase: str):
        targets = phase_targets(phase)
        self.phase = phase
        self.history.append(phase)
        return targets

    def initialize_physical_connections(self, *, stage=None):
        return {
            "target_bed_fixtures": anchor_target_bed(
                self.tissue_root, stage=stage
            ),
            "traction": self.traction.capture(stage=stage),
            "bridges": self.bridges.engage(stage=stage),
            "protected_structures": self.protected.attach_to_target_bed(stage=stage),
        }

    def blunt_action(self, local_position: Sequence[float], work_j: float, *, override=False, stage=None):
        safety = self.protected.evaluate_action(local_position, "blunt")
        if not safety["authorized"] and not override:
            return {"safety": safety, "released_bridges": [], "blocked": True}
        if not safety["authorized"] and override:
            self.protected.injure(safety["nearest_structure"], "blunt_override", stage=stage)
        released = self.bridges.apply_blunt_work(local_position, work_j, stage=stage)
        return {"safety": safety, "released_bridges": released, "override": bool(override)}

    def hydro_action(self, local_position: Sequence[float], volume_ml: float, *, override=False, stage=None):
        safety = self.protected.evaluate_action(local_position, "hydro")
        if not safety["authorized"] and not override:
            return {"safety": safety, "released_bridges": [], "blocked": True}
        if not safety["authorized"] and override:
            self.protected.injure(safety["nearest_structure"], "hydro_override", stage=stage)
        released = self.bridges.apply_hydro_volume(local_position, volume_ml, stage=stage)
        return {"safety": safety, "released_bridges": released, "override": bool(override)}

    def energy_action(self, local_position: Sequence[float], *, dt: float, contact_force_n: float, requested_power_w: float | None = None, override=False, stage=None):
        safety = self.protected.evaluate_action(local_position, "energy")
        if not safety["authorized"] and not override:
            return {"safety": safety, "released_bridges": [], "blocked": True}
        if not safety["authorized"] and override:
            self.protected.injure(safety["nearest_structure"], "energy_override", stage=stage)
        energy_result = self.energy.update(dt, contact_force_n, requested_power_w)
        released = self.bridges.apply_energy(local_position, energy_result["energy_j"], stage=stage)
        return {"safety": safety, "released_bridges": released, "energy": energy_result, "override": bool(override)}

    def scissors_action(self, local_position: Sequence[float], guard_retraction_m: float, *, override=False, stage=None):
        return self.scissors.request_cut(
            local_position, guard_retraction_m, self.bridges, self.protected,
            override=override, stage=stage,
        )

    def verify(self, *, visibility_fraction: float, traction_stable: bool):
        return self.verifier.evaluate(visibility_fraction=visibility_fraction, traction_stable=traction_stable)
