# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Isaac integration for the DrAnmar Adaptive Seal-and-Divide Robot.

The payload replaces the Panda hand at ``panda_link8``. Runtime helpers
center and compress a vascular pedicle, create two physical stump seal
bands, estimate seal maturity from a lumped energy/impedance model, enforce
a blade interlock, release the pre-authored mechanical tissue bridge during
division, and estimate residual stump flow. All values are provisional
research parameters and are not patient-care settings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import math

CATALOG_SUBPATH = "Props/SurgicalDivision/AdaptiveSealDivideRobot"
ASSET_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
ROOT = ASSET_DATA_ROOT / CATALOG_SUBPATH
TOOL_PAYLOAD_USD = ROOT / "dranmar_adaptive_seal_divide_tool_payload.usda"
TOOL_STANDALONE_USD = ROOT / "dranmar_adaptive_seal_divide_tool_standalone.usda"
TOOL_RIGID_PROXY_USD = ROOT / "dranmar_adaptive_seal_divide_tool_rigid_proxy.usda"
VESSEL_USD = ROOT / "dranmar_seal_divide_vessel_demo.usda"
SEAL_BAND_USD = ROOT / "dranmar_tissue_seal_band.usda"
BLADE_USD = ROOT / "dranmar_division_blade_cartridge.usda"
VAPOR_USD = ROOT / "dranmar_seal_vapor_particle.usda"

# Isotropic small-strain research baseline for the demo vessel shell.  The
# modulus is the reported fresh ex-vivo porcine aorta mechanical-test mean
# (202.4 kPa), Poisson ratio 0.35 is the midpoint of the measured 0.3-0.4
# in-plane porcine arterial-wall range, and density 1060 kg/m^3 follows a
# published isotropic arterial-wall structural model.  This is not a
# calibrated patient, vessel-type, or electrosurgical tissue model.
VESSEL_SURFACE_MATERIAL = {
    "density_kg_m3":1060.0,
    "youngs_modulus_pa":202_400.0,
    "poissons_ratio":0.35,
    "surface_thickness_m":0.00068,
    "dynamic_friction":0.40,
}

VALID_CARTRIDGE_STATES = frozenset({"fresh", "spent"})
VALID_SALINE_STATES = frozenset({"full", "empty"})
VALID_COLLECTION_STATES = frozenset({"empty", "partial", "full"})
VALID_ENERGY_STATES = frozenset({"ready", "fault"})
BRIDGE_PIN_COUNT = 8

TOOL_JOINTS = {
    "left_centering":"left_centering_joint",
    "right_centering":"right_centering_joint",
    "upper_jaw":"upper_jaw_joint",
    "lower_jaw":"lower_jaw_joint",
    "blade_guard":"blade_guard_joint",
    "blade":"blade_joint",
    "suction_valve":"suction_valve_joint",
    "irrigation_valve":"irrigation_valve_joint",
}
TOOL_FRAME_PATHS = {
    "panda_link8_mount":"Links/Mount/Frames/panda_link8_mount",
    "seal_divide_tcp":"Links/Mount/Frames/seal_divide_tcp",
    "tissue_center_reference":"Links/Mount/Frames/tissue_center_reference",
    "left_seal_zone":"Links/Mount/Frames/left_seal_zone",
    "right_seal_zone":"Links/Mount/Frames/right_seal_zone",
    "cut_plane":"Links/Mount/Frames/cut_plane",
    "suction_center":"Links/Mount/Frames/suction_center",
    "irrigation_center":"Links/Mount/Frames/irrigation_center",
    "thermal_camera":"Links/Mount/Frames/thermal_camera",
    "impedance_probe":"Links/Mount/Frames/impedance_probe",
    "seal_verification_probe":"Links/Mount/Frames/seal_verification_probe",
    "left_centering_contact":"Links/LeftCentering/Frames/left_centering_contact",
    "right_centering_contact":"Links/RightCentering/Frames/right_centering_contact",
    "upper_jaw_contact":"Links/UpperJaw/Frames/upper_jaw_contact",
    "lower_jaw_contact":"Links/LowerJaw/Frames/lower_jaw_contact",
    "blade_tip":"Links/BladeCarriage/Frames/blade_tip",
    "blade_guard_reference":"Links/BladeGuard/Frames/blade_guard_reference",
    "count_reference":"Links/Mount/Frames/count_reference",
    "disposal_reference":"Links/Mount/Frames/disposal_reference",
}
REGISTERED_CAMERA_FRAMES = ("thermal_camera",)

def frame_path(tool_path: str, name: str) -> str:
    try:
        suffix = TOOL_FRAME_PATHS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown seal/divide frame {name!r}") from exc
    return f"{tool_path.rstrip('/')}/{suffix}"

def tensor_value(value: Any):
    return value.torch if hasattr(value, "torch") else value


def _xyzw_from_wxyz(orientation_wxyz) -> tuple[float, float, float, float]:
    values=tuple(float(value) for value in orientation_wxyz)
    if len(values)!=4 or not all(math.isfinite(value) for value in values):raise ValueError("orientation_wxyz must contain four finite values")
    if abs(math.sqrt(sum(value*value for value in values))-1.0)>1.0e-4:raise ValueError("orientation_wxyz must be a unit quaternion")
    w,x,y,z=values
    return x,y,z,w


def _check(value: str, allowed: frozenset[str], label: str) -> str:
    if value not in allowed:
        raise ValueError(f"Unsupported {label}={value!r}; expected one of {sorted(allowed)}")
    return value


def _finite(value: float,label: str) -> float:
    result=float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _finite_nonnegative(value: float,label: str) -> float:
    result=_finite(value,label)
    if result<0.0:
        raise ValueError(f"{label} must be nonnegative")
    return result

def make_tool_cfg(
    prim_path: str = "/World/DrAnmarAdaptiveSealDivideTool",
    *,
    cartridge_state: str = "fresh",
    saline_state: str = "full",
    collection_state: str = "empty",
    energy_state: str = "ready",
    position=(0.0, 0.0, 0.35),
    orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    _check(cartridge_state, VALID_CARTRIDGE_STATES, "cartridge_state")
    _check(saline_state, VALID_SALINE_STATES, "saline_state")
    _check(collection_state, VALID_COLLECTION_STATES, "collection_state")
    _check(energy_state, VALID_ENERGY_STATES, "energy_state")
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(TOOL_STANDALONE_USD),
            variants={"cartridge_state":cartridge_state,"saline_state":saline_state,"collection_state":collection_state,"energy_state":energy_state},
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False,solver_position_iteration_count=24,solver_velocity_iteration_count=8),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=position,rot=_xyzw_from_wxyz(orientation_wxyz),joint_pos={name:0.0 for name in TOOL_JOINTS.values()}),
        actuators={
            "centering":ImplicitActuatorCfg(joint_names_expr=[".*centering_joint"],effort_limit_sim=90.0,velocity_limit_sim=0.16,stiffness=4200.0,damping=145.0),
            "seal_jaws":ImplicitActuatorCfg(joint_names_expr=[".*jaw_joint"],effort_limit_sim=360.0,velocity_limit_sim=0.10,stiffness=18000.0,damping=420.0),
            "blade_system":ImplicitActuatorCfg(joint_names_expr=["blade_guard_joint","blade_joint"],effort_limit_sim=280.0,velocity_limit_sim=0.25,stiffness=16000.0,damping=360.0),
            "valves":ImplicitActuatorCfg(joint_names_expr=[".*_valve_joint"],effort_limit_sim=30.0,velocity_limit_sim=0.25,stiffness=1800.0,damping=55.0),
        },
    )

def make_rigid_proxy_cfg(prim_path="/World/DrAnmarAdaptiveSealDivideProxy", *, position=(0,0,0.35), orientation_wxyz=(1,0,0,0)):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    return RigidObjectCfg(prim_path=prim_path,spawn=sim_utils.UsdFileCfg(usd_path=str(TOOL_RIGID_PROXY_USD),activate_contact_sensors=True),init_state=RigidObjectCfg.InitialStateCfg(pos=position,rot=_xyzw_from_wxyz(orientation_wxyz)))

def _spawn_single_franka_with_tool(prim_path: str, cfg: Any, translation=None, orientation=None, **kwargs):
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    from isaaclab.sim.utils import create_prim, get_current_stage, select_usd_variants
    from pxr import Gf, Sdf, UsdPhysics
    robot=spawn_from_usd(prim_path,cfg,translation,orientation)
    stage=get_current_stage()
    robot_path=Sdf.Path(prim_path)
    names_to_disable={
        "panda_hand_joint","panda_hand","panda_finger_joint1",
        "panda_finger_joint2","panda_leftfinger","panda_rightfinger",
    }
    hand_joints=[
        prim for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(robot_path) and prim.GetName()=="panda_hand_joint"
    ]
    if len(hand_joints)==1:
        stock_joint=UsdPhysics.Joint(hand_joints[0])
        mount_body_paths=stock_joint.GetBody0Rel().GetTargets()
        mount_local_pos0=stock_joint.GetLocalPos0Attr().Get() or Gf.Vec3f(0,0,0)
        mount_local_rot0=stock_joint.GetLocalRot0Attr().Get() or Gf.Quatf(1,0,0,0)
    else:
        link8_paths=[
            prim.GetPath() for prim in stage.Traverse()
            if prim.GetPath().HasPrefix(robot_path) and prim.GetName()=="panda_link8"
        ]
        if len(link8_paths)!=1:
            raise RuntimeError("Could not resolve the Franka hand mount")
        mount_body_paths=link8_paths
        mount_local_pos0=Gf.Vec3f(0,0,0)
        half_angle=math.radians(-45.0)/2.0
        mount_local_rot0=Gf.Quatf(
            math.cos(half_angle),0,0,math.sin(half_angle)
        )
    if len(mount_body_paths)!=1 or not stage.GetPrimAtPath(mount_body_paths[0]).IsValid():
        raise RuntimeError(f"Invalid Franka hand mount target: {mount_body_paths}")

    candidate_paths=[
        prim.GetPath() for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(robot_path) and prim.GetName() in names_to_disable
    ]
    paths_to_disable=[]
    for path in sorted(candidate_paths,key=lambda item:str(item).count("/")):
        if not any(path.HasPrefix(parent) for parent in paths_to_disable):
            paths_to_disable.append(path)
    for path in paths_to_disable:
        stage.OverridePrim(path).SetActive(False)
    tool_path=f"{prim_path}/DrAnmarAdaptiveSealDivideTool"
    create_prim(tool_path,usd_path=str(TOOL_PAYLOAD_USD),stage=stage)
    select_usd_variants(tool_path,{"cartridge_state":cfg.cartridge_state,"saline_state":cfg.saline_state,"collection_state":cfg.collection_state,"energy_state":cfg.energy_state})
    joint=UsdPhysics.FixedJoint.Define(stage,f"{prim_path}/dranmar_seal_divide_mount_joint")
    joint.CreateBody0Rel().SetTargets(mount_body_paths)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{tool_path}/Links/Mount")])
    joint.CreateLocalPos0Attr().Set(mount_local_pos0)
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0,0,0))
    joint.CreateLocalRot0Attr().Set(mount_local_rot0)
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1,0,0,0))
    return robot

def spawn_franka_with_tool(prim_path: str,cfg: Any,translation=None,orientation=None,**kwargs):
    from isaaclab.sim.utils import clone
    return clone(_spawn_single_franka_with_tool)(prim_path,cfg,translation=translation,orientation=orientation,**kwargs)

def make_franka_adaptive_seal_divide_robot_cfg(*, prim_path="/World/Robot", cartridge_state="fresh", saline_state="full", collection_state="empty", energy_state="ready"):
    _check(cartridge_state,VALID_CARTRIDGE_STATES,"cartridge_state");_check(saline_state,VALID_SALINE_STATES,"saline_state");_check(collection_state,VALID_COLLECTION_STATES,"collection_state");_check(energy_state,VALID_ENERGY_STATES,"energy_state")
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
    @configclass
    class FrankaSealDivideUsdCfg(sim_utils.UsdFileCfg):
        cartridge_state: str="fresh";saline_state: str="full";collection_state: str="empty";energy_state: str="ready";func=spawn_franka_with_tool
    cfg=FRANKA_PANDA_CFG.copy();cfg.prim_path=prim_path
    cfg.spawn=FrankaSealDivideUsdCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd",variants={"Gripper":"Default","Mesh":"Performance"},cartridge_state=cartridge_state,saline_state=saline_state,collection_state=collection_state,energy_state=energy_state,activate_contact_sensors=True,rigid_props=FRANKA_PANDA_CFG.spawn.rigid_props,articulation_props=FRANKA_PANDA_CFG.spawn.articulation_props)
    cfg.init_state.joint_pos={k:v for k,v in cfg.init_state.joint_pos.items() if "finger" not in k};cfg.init_state.joint_pos.update({name:0.0 for name in TOOL_JOINTS.values()})
    cfg.actuators={k:v for k,v in cfg.actuators.items() if k!="panda_hand"}
    cfg.actuators.update({
        "seal_divide_centering":ImplicitActuatorCfg(joint_names_expr=[".*centering_joint"],effort_limit_sim=90.0,velocity_limit_sim=0.16,stiffness=4200.0,damping=145.0),
        "seal_divide_jaws":ImplicitActuatorCfg(joint_names_expr=[".*jaw_joint"],effort_limit_sim=360.0,velocity_limit_sim=0.10,stiffness=18000.0,damping=420.0),
        "seal_divide_blade":ImplicitActuatorCfg(joint_names_expr=["blade_guard_joint","blade_joint"],effort_limit_sim=280.0,velocity_limit_sim=0.25,stiffness=16000.0,damping=360.0),
        "seal_divide_valves":ImplicitActuatorCfg(joint_names_expr=[".*_valve_joint"],effort_limit_sim=30.0,velocity_limit_sim=0.25,stiffness=1800.0,damping=55.0),
    })
    return cfg

def _current_stage(stage=None):
    if stage is not None:return stage
    import omni.usd
    return omni.usd.get_context().get_stage()

def spawn_vessel_demo(prim_path="/World/DrAnmarSealDivideVessel", *, translation=(0,0,0), orientation_wxyz=(1,0,0,0)):
    import isaaclab.sim as sim_utils
    cfg=sim_utils.UsdFileCfg(usd_path=str(VESSEL_USD));return cfg.func(prim_path,cfg,translation=translation,orientation=_xyzw_from_wxyz(orientation_wxyz))

def apply_vessel_surface_deformables(root_path: str, *, self_collision=False, stage=None):
    stage=_current_stage(stage);results=[]
    from omni.physx.scripts import deformableUtils
    from pxr import Sdf,UsdPhysics,UsdShade
    root_path=root_path.rstrip("/")
    material_path=f"{root_path}/RuntimeMaterials/VesselWallSurface"
    material=UsdShade.Material.Define(stage,material_path)
    material_prim=material.GetPrim()
    for schema in (
        "OmniPhysicsBaseMaterialAPI",
        "OmniPhysicsDeformableMaterialAPI",
        "OmniPhysicsSurfaceDeformableMaterialAPI",
        "PhysxDeformableMaterialAPI",
        "PhysxSurfaceDeformableMaterialAPI",
    ):
        if schema not in material_prim.GetAppliedSchemas():
            material_prim.AddAppliedSchema(schema)
    values=VESSEL_SURFACE_MATERIAL
    material_prim.CreateAttribute(
        "omniphysics:density",Sdf.ValueTypeNames.Float
    ).Set(values["density_kg_m3"])
    material_prim.CreateAttribute(
        "omniphysics:dynamicFriction",Sdf.ValueTypeNames.Float
    ).Set(values["dynamic_friction"])
    material_prim.CreateAttribute(
        "omniphysics:youngsModulus",Sdf.ValueTypeNames.Float
    ).Set(values["youngs_modulus_pa"])
    material_prim.CreateAttribute(
        "omniphysics:poissonsRatio",Sdf.ValueTypeNames.Float
    ).Set(values["poissons_ratio"])
    material_prim.CreateAttribute(
        "omniphysics:surfaceThickness",Sdf.ValueTypeNames.Float
    ).Set(values["surface_thickness_m"])
    material_prim.CreateAttribute(
        "omniphysics:surfaceBendStiffness",Sdf.ValueTypeNames.Float
    ).Set(
        values["youngs_modulus_pa"]
        /(12.0*(1.0-values["poissons_ratio"]**2))
    )
    for child in ("LeftVesselWall","RightVesselWall"):
        mesh_path=f"{root_path}/{child}";mesh=stage.GetPrimAtPath(mesh_path)
        if not mesh or not mesh.IsValid():raise ValueError(f"No vessel wall at {mesh_path}")
        UsdShade.MaterialBindingAPI.Apply(mesh).Bind(
            material,UsdShade.Tokens.weakerThanDescendants,"physics"
        )
        ok=deformableUtils.set_physics_surface_deformable_body(stage,mesh.GetPath())
        if ok is False:raise RuntimeError(f"Failed to cook vessel surface deformable at {mesh_path}")
        mesh.CreateAttribute(
            "omniphysics:restBendAnglesDefault",Sdf.ValueTypeNames.Token
        ).Set("restShapeDefault")
        mesh.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
        if mesh.HasAPI("PhysxSurfaceDeformableBodyAPI"):
            mesh.GetAttribute("physxDeformableBody:selfCollision").Set(
                bool(self_collision)
            )
            # The two vessel halves are only 0.4 mm apart.  Keep the contact
            # envelope below half that authored clearance so the solver does
            # not begin by depenetrating otherwise separated stump surfaces.
            mesh.CreateAttribute(
                "physxCollision:contactOffset",Sdf.ValueTypeNames.Float
            ).Set(0.0001)
            mesh.CreateAttribute(
                "physxCollision:restOffset",Sdf.ValueTypeNames.Float
            ).Set(0.0)
        results.append(mesh_path)
    # NVIDIA documents that element-level collision filtering is not
    # supported between two surface deformables.  These halves meet at an
    # attached seam, so use the supported prim-pair filter and let their
    # explicit bridge attachments carry the cross-seam load.
    left_prim=stage.GetPrimAtPath(f"{root_path}/LeftVesselWall")
    UsdPhysics.FilteredPairsAPI.Apply(left_prim).CreateFilteredPairsRel().AddTarget(
        f"{root_path}/RightVesselWall"
    )
    return {"root_path":root_path,"mesh_paths":results,"self_collision":bool(self_collision)}

def create_deformable_attachment(
    deformable_path: str,
    target_path: str,
    attachment_path: str,
    *,
    stage=None,
    deformable_points_world=None,
    target_to_world=None,
    attachment_frame_path=None,
    attachment_frame_to_world=None,
    excluded_vertex_indices=None,
    maximum_vertices=12,
    selected_vertex_indices_out=None,
):
    """Create and verify an overlap-prioritized attachment across Isaac versions."""
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics,Vt
    stage=_current_stage(stage)
    if stage.GetPrimAtPath(attachment_path).IsValid():
        stage.RemovePrim(attachment_path)
    definition=Usd.SchemaRegistry().FindConcretePrimDefinition(
        "OmniPhysicsVtxXformAttachment"
    )
    if definition:
        deformable=stage.GetPrimAtPath(deformable_path)
        target=stage.GetPrimAtPath(target_path)
        mesh=UsdGeom.Mesh(deformable)
        points=list(mesh.GetPointsAttr().Get() or [])
        if not deformable.IsValid() or not mesh or not points:
            raise ValueError(f"Attachment source is not a populated mesh: {deformable_path}")
        if not target.IsValid() or not UsdGeom.Xformable(target):
            raise ValueError(f"Attachment target is not xformable: {target_path}")
        runtime_geometry=(
            deformable_points_world is not None
            or target_to_world is not None
            or attachment_frame_to_world is not None
        )
        if runtime_geometry and (
            deformable_points_world is None
            or target_to_world is None
            or attachment_frame_to_world is None
        ):
            raise ValueError(
                "Runtime attachment selection requires current deformable "
                "world points, target-to-world, and attachment-frame-to-world"
            )
        if attachment_frame_path is None:
            frame=target
            while frame.IsValid() and not frame.HasAPI(UsdPhysics.RigidBodyAPI):
                frame=frame.GetParent()
            if not frame.IsValid():
                raise RuntimeError(
                    f"Attachment target has no rigid-body frame: {target_path}"
                )
            attachment_frame_path=str(frame.GetPath())
        attachment_frame=stage.GetPrimAtPath(attachment_frame_path)
        if not attachment_frame.IsValid():
            raise RuntimeError(
                f"Attachment frame is missing: {attachment_frame_path}"
            )
        mesh_to_world=UsdGeom.Xformable(deformable).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        if target_to_world is None:
            target_to_world=UsdGeom.Xformable(target).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            bounds=UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_,UsdGeom.Tokens.guide],
            ).ComputeWorldBound(target).ComputeAlignedRange()
        else:
            untransformed=UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_,UsdGeom.Tokens.guide],
            ).ComputeUntransformedBound(target)
            untransformed.Transform(target_to_world)
            bounds=untransformed.ComputeAlignedRange()
        if attachment_frame_to_world is None:
            attachment_frame_to_world=UsdGeom.Xformable(
                attachment_frame
            ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        world_to_attachment_frame=attachment_frame_to_world.GetInverse()
        minimum,maximum=bounds.GetMin(),bounds.GetMax()
        center=(minimum+maximum)*0.5
        if deformable_points_world is not None:
            if len(deformable_points_world)!=len(points):
                raise RuntimeError(
                    f"Runtime deformable topology changed for {deformable_path}: "
                    f"usd_vertices={len(points)}, simulation_vertices="
                    f"{len(deformable_points_world)}"
                )
            world_points=[
                Gf.Vec3d(float(point[0]),float(point[1]),float(point[2]))
                for point in deformable_points_world
            ]
        else:
            world_points=[
                mesh_to_world.Transform(Gf.Vec3d(point)) for point in points
            ]
        if maximum_vertices<4:
            raise ValueError("maximum_vertices must be at least four")
        excluded=set(excluded_vertex_indices or ())
        ranked=[]
        for index,world in enumerate(world_points):
            delta=world-center
            overlaps=all(
                minimum[axis]-0.0025<=world[axis]<=maximum[axis]+0.0025
                for axis in range(3)
            )
            ranked.append((float(Gf.Dot(delta,delta)),index,world,overlaps))
        ranked.sort(key=lambda item:item[0])
        selected=[
            item for item in ranked if item[3] and item[1] not in excluded
        ][:maximum_vertices]
        if len(selected)<4:
            nearest_center_distance_m=(
                math.sqrt(ranked[0][0]) if ranked else math.inf
            )
            source_min=tuple(
                min(point[axis] for point in world_points) for axis in range(3)
            )
            source_max=tuple(
                max(point[axis] for point in world_points) for axis in range(3)
            )
            nearest_world=(
                tuple(ranked[0][2]) if ranked else None
            )
            raise RuntimeError(
                f"Attachment capture volume does not overlap enough deformable "
                f"vertices for {attachment_path}: source={deformable_path}, "
                f"target={target_path}, overlapping={len(selected)}, "
                f"required=4, overlap_margin_m=0.0025, "
                f"target_bounds_world=({tuple(minimum)}, {tuple(maximum)}), "
                f"source_bounds_world=({source_min}, {source_max}), "
                f"nearest_vertex_world={nearest_world}, "
                f"nearest_vertex_to_target_center_m={nearest_center_distance_m}"
            )
        attachment=stage.DefinePrim(
            attachment_path,"OmniPhysicsVtxXformAttachment"
        )
        attachment.CreateRelationship("omniphysics:src0").SetTargets(
            [Sdf.Path(deformable_path)]
        )
        attachment.CreateRelationship("omniphysics:src1").SetTargets(
            [Sdf.Path(attachment_frame_path)]
        )
        attachment.CreateAttribute(
            "omniphysics:vtxIndicesSrc0",Sdf.ValueTypeNames.IntArray
        ).Set(Vt.IntArray([item[1] for item in selected]))
        attachment.CreateAttribute(
            "omniphysics:localPositionsSrc1",Sdf.ValueTypeNames.Point3fArray
        ).Set(Vt.Vec3fArray([
            Gf.Vec3f(world_to_attachment_frame.Transform(item[2]))
            for item in selected
        ]))
        if selected_vertex_indices_out is not None:
            selected_vertex_indices_out.extend(item[1] for item in selected)
        attachment.CreateAttribute(
            "omniphysics:attachmentEnabled",Sdf.ValueTypeNames.Bool
        ).Set(True)
        if (
            not attachment.IsValid()
            or attachment.GetTypeName()!="OmniPhysicsVtxXformAttachment"
            or not attachment.GetRelationship("omniphysics:src0").GetTargets()
            or not attachment.GetRelationship("omniphysics:src1").GetTargets()
        ):
            raise RuntimeError(f"Could not author attachment {attachment_path}")
        return "OmniPhysicsVtxXformAttachment"

    import omni.kit.commands
    def execute_and_verify(command: str,**kwargs) -> str:
        omni.kit.commands.execute(command,**kwargs)
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
        if stage.GetPrimAtPath(attachment_path).IsValid():stage.RemovePrim(attachment_path)
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
    stage=_current_stage(stage)
    for path in paths:
        if stage.GetPrimAtPath(path).IsValid():stage.RemovePrim(path)


def anchor_vessel_distal_ends(root_path: str,*,stage=None) -> list[str]:
    """Attach both cooked vessel halves to explicit kinematic fixtures."""
    from pxr import Gf,Usd,UsdGeom,UsdPhysics
    stage=_current_stage(stage);root_path=root_path.rstrip("/")
    attachments_root=f"{root_path}/RuntimeFixtureAttachments"
    frames_root=f"{root_path}/RuntimeFixtureFrames"
    stage.DefinePrim(attachments_root,"Scope")
    stage.DefinePrim(frames_root,"Scope");created=[]
    try:
        for label,vessel,target in (
            ("left","LeftVesselWall","LeftFixtureAnchor"),
            ("right","RightVesselWall","RightFixtureAnchor"),
        ):
            target_path=f"{root_path}/{target}"
            target_prim=stage.GetPrimAtPath(target_path)
            if not target_prim.IsValid():
                raise ValueError(f"Vessel fixture anchor is missing: {target_path}")
            frame_path=f"{frames_root}/{label}"
            frame=UsdGeom.Xform.Define(stage,frame_path)
            root_to_world=UsdGeom.Xformable(
                stage.GetPrimAtPath(root_path)
            ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            target_to_world=UsdGeom.Xformable(
                target_prim
            ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            frame_to_world=Gf.Matrix4d(1.0)
            frame_to_world.SetRotate(target_to_world.ExtractRotationMatrix())
            frame_to_world.SetTranslateOnly(target_to_world.ExtractTranslation())
            frame.MakeMatrixXform().Set(
                frame_to_world*root_to_world.GetInverse()
            )
            rigid=UsdPhysics.RigidBodyAPI.Apply(frame.GetPrim())
            rigid.CreateRigidBodyEnabledAttr(True)
            rigid.CreateKinematicEnabledAttr(True)
            attachment_path=f"{attachments_root}/{label}"
            create_deformable_attachment(
                f"{root_path}/{vessel}",
                target_path,
                attachment_path,
                stage=stage,
                attachment_frame_path=frame_path,
            )
            created.append(attachment_path)
    except Exception:
        remove_prims(created,stage=stage);raise
    return created


@dataclass
class DualZoneCompressionController:
    tool_root: str
    vessel_root: str
    minimum_total_force_n: float=8.0
    target_total_force_n: float=18.0
    soft_force_limit_n: float=32.0
    hard_release_limit_n: float=45.0
    attachment_paths: list[str]=field(default_factory=list)
    engaged: bool=False
    def engage(self,*,stage=None,runtime_geometry=None):
        stage=_current_stage(stage)
        parent=f"{self.vessel_root}/RuntimeJawCompressionAttachments"
        stage.DefinePrim(parent,"Scope");created=[]
        try:
            for side,vessel,contact in (
                ("left_upper","LeftVesselWall","UpperJaw/Collisions/LeftSealContact"),
                ("left_lower","LeftVesselWall","LowerJaw/Collisions/LeftSealContact"),
                ("right_upper","RightVesselWall","UpperJaw/Collisions/RightSealContact"),
                ("right_lower","RightVesselWall","LowerJaw/Collisions/RightSealContact"),
            ):
                attachment=f"{parent}/{side}"
                geometry=(
                    {} if runtime_geometry is None
                    else runtime_geometry.get(side,{})
                )
                create_deformable_attachment(
                    f"{self.vessel_root}/{vessel}",
                    f"{self.tool_root}/Links/{contact}",
                    attachment,stage=stage,**geometry,
                )
                created.append(attachment)
        except Exception:
            remove_prims(created,stage=stage);raise
        self.attachment_paths=created;self.engaged=True
        return list(created)
    def release(self,*,stage=None):
        remove_prims(self.attachment_paths,stage=stage)
        self.engaged=False
    def update_force(self,upper_force_n: float,lower_force_n: float,*,stage=None):
        upper=_finite_nonnegative(upper_force_n,"upper_force_n")
        lower=_finite_nonnegative(lower_force_n,"lower_force_n")
        total=upper+lower
        if total>self.hard_release_limit_n and self.engaged:self.release(stage=stage)
        return {
            "mode":(
                "hard_release" if total>self.hard_release_limit_n
                else "soft_limit" if total>self.soft_force_limit_n
                else "insufficient" if total<self.minimum_total_force_n
                else "controlled"
            ),
            "upper_force_n":upper,
            "lower_force_n":lower,
            "total_force_n":total,
            "target_error_n":total-self.target_total_force_n,
            "engaged":self.engaged,
        }

@dataclass
class BridgeCell:
    index: int
    pin_path: str
    attachment_paths: list[str]
    released: bool=False

@dataclass
class BridgeAttachmentController:
    vessel_root: str
    cells: list[BridgeCell]=field(default_factory=list)
    release_order: tuple[int,...]=(2,1,3,0,4,7,5,6)
    def engage(self, *, stage=None):
        if tuple(sorted(self.release_order))!=tuple(range(BRIDGE_PIN_COUNT)):
            raise ValueError("release_order must contain each bridge index exactly once")
        stage=_current_stage(stage);stage.DefinePrim(f"{self.vessel_root}/RuntimeBridgeAttachments","Scope");created=[];cells=[]
        left=f"{self.vessel_root}/LeftVesselWall";right=f"{self.vessel_root}/RightVesselWall"
        used_vertices={"left":set(),"right":set()}
        try:
            for i in range(BRIDGE_PIN_COUNT):
                pin=f"{self.vessel_root}/BridgePins/BridgePin_{i:02d}/Capture";paths=[]
                for side,actor in (("left",left),("right",right)):
                    ap=f"{self.vessel_root}/RuntimeBridgeAttachments/pin_{i:02d}_{side}"
                    selected=[]
                    create_deformable_attachment(
                        actor,pin,ap,stage=stage,
                        excluded_vertex_indices=used_vertices[side],
                        maximum_vertices=4,
                        selected_vertex_indices_out=selected,
                    )
                    used_vertices[side].update(selected)
                    paths.append(ap);created.append(ap)
                cells.append(BridgeCell(i,pin,paths))
        except Exception:
            remove_prims(created,stage=stage);raise
        self.cells=cells;return cells
    @property
    def released_fraction(self):return 0.0 if not self.cells else sum(c.released for c in self.cells)/len(self.cells)
    def set_cut_progress(self,progress: float,*,stage=None):
        progress=max(0.0,min(1.0,_finite(progress,"progress")));target=int(math.floor(progress*len(self.release_order)+1e-9));stage=_current_stage(stage)
        by_index={c.index:c for c in self.cells}
        for idx in self.release_order[:target]:
            cell=by_index.get(idx)
            if cell is not None and not cell.released:
                remove_prims(cell.attachment_paths,stage=stage);cell.released=True
        if progress>=1.0:
            bridge=stage.GetPrimAtPath(f"{self.vessel_root}/BridgeVisual")
            if bridge and bridge.IsValid():
                from pxr import UsdGeom
                UsdGeom.Imageable(bridge).MakeInvisible()
        return self.released_fraction
    @property
    def complete(self):return bool(self.cells) and all(c.released for c in self.cells)

@dataclass
class SealZoneState:
    name: str
    temperature_c: float=37.0
    impedance_ohm: float=120.0
    energy_j: float=0.0
    thermal_dose: float=0.0
    maturity: float=0.0
    compression_force_n: float=0.0
    overtemperature: bool=False
    impedance_fault: bool=False

@dataclass
class AdaptiveSealEnergyController:
    target_temperature_c: float=78.0
    maximum_temperature_c: float=105.0
    maximum_power_w: float=45.0
    heat_capacity_j_k: float=1.8
    heat_loss_w_k: float=0.22
    minimum_compression_force_n: float=8.0
    maximum_compression_force_n: float=32.0
    left: SealZoneState=field(default_factory=lambda:SealZoneState("left"))
    right: SealZoneState=field(default_factory=lambda:SealZoneState("right"))
    def recommended_power_w(self,zone: SealZoneState):
        if zone.overtemperature or zone.impedance_fault:return 0.0
        compression_scale=max(0.0,min(1.0,(zone.compression_force_n-self.minimum_compression_force_n)/max(self.maximum_compression_force_n-self.minimum_compression_force_n,1e-9)))
        return max(0.0,min(self.maximum_power_w,(self.target_temperature_c-zone.temperature_c)*1.25*(0.55+0.45*compression_scale)))
    def update_zone(self,zone: SealZoneState,dt: float,compression_force_n: float,commanded_power_w: float|None=None):
        dt=_finite_nonnegative(dt,"dt")
        zone.compression_force_n=_finite_nonnegative(
            compression_force_n,"compression_force_n"
        )
        if commanded_power_w is None:commanded_power_w=self.recommended_power_w(zone)
        power=max(0.0,min(
            _finite_nonnegative(self.maximum_power_w,"maximum_power_w"),
            _finite_nonnegative(commanded_power_w,"commanded_power_w"),
        ))
        compression_eff=max(0.0,min(1.0,zone.compression_force_n/max(self.minimum_compression_force_n,1e-9)))
        absorbed=power*(0.35+0.65*compression_eff);loss=self.heat_loss_w_k*max(0.0,zone.temperature_c-37.0)
        zone.temperature_c += (absorbed-loss)*dt/max(self.heat_capacity_j_k,1e-9);zone.energy_j += absorbed*dt
        dose_rate=0.0 if zone.temperature_c<45.0 else math.exp((zone.temperature_c-62.0)/8.0)
        zone.thermal_dose += dose_rate*dt;zone.maturity=1.0-math.exp(-zone.thermal_dose/8.0)
        zone.impedance_ohm=max(18.0,120.0*(1.0-0.0045*max(0.0,zone.temperature_c-37.0))*(1.0+1.6*zone.maturity))
        zone.overtemperature=zone.temperature_c>self.maximum_temperature_c
        zone.impedance_fault=not math.isfinite(zone.impedance_ohm) or zone.impedance_ohm<10.0 or zone.impedance_ohm>500.0
        return zone
    def update(self,dt: float,left_force_n: float,right_force_n: float,left_power_w: float|None=None,right_power_w: float|None=None):
        return self.update_zone(self.left,dt,left_force_n,left_power_w),self.update_zone(self.right,dt,right_force_n,right_power_w)
    def zone_ready(self,zone: SealZoneState):return zone.maturity>=0.90 and not zone.overtemperature and not zone.impedance_fault and self.minimum_compression_force_n<=zone.compression_force_n<=self.maximum_compression_force_n
    @property
    def both_ready(self):return self.zone_ready(self.left) and self.zone_ready(self.right)

def _spawn_reference_at_transform(stage,prim_path: str,usd_path: Path,world_transform: Any,variants: dict[str,str]|None=None):
    from pxr import Gf,UsdGeom
    prim=stage.DefinePrim(prim_path,"Xform");prim.GetReferences().AddReference(str(usd_path))
    UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d(world_transform))
    if variants:
        for name,value in variants.items():prim.GetVariantSets().GetVariantSet(name).SetVariantSelection(value)
    return prim

@dataclass
class SealBandBond:
    band_path: str
    vessel_path: str
    attachment_paths: list[str]
    maturity: float=0.0
    failed: bool=False

@dataclass
class TissueSealBandController:
    initial_break_force_n: float=0.6
    mature_break_force_n: float=7.5
    bonds: list[SealBandBond]=field(default_factory=list)
    def deploy(self,prim_path: str,world_transform: Any,vessel_path: str,*,stage=None):
        stage=_current_stage(stage);_spawn_reference_at_transform(stage,prim_path,SEAL_BAND_USD,world_transform,{"state":"fresh"});stage.DefinePrim(f"{prim_path}/Attachments","Scope");created=[]
        try:
            for name in ("UpperBondVolume","LowerBondVolume"):
                ap=f"{prim_path}/Attachments/{name}";create_deformable_attachment(vessel_path,f"{prim_path}/Collisions/{name}",ap,stage=stage);created.append(ap)
        except Exception:
            remove_prims(created+[prim_path],stage=stage);raise
        bond=SealBandBond(prim_path,vessel_path,created);self.bonds.append(bond);return bond
    def update_maturity(self,bond: SealBandBond,maturity: float,*,stage=None):
        bond.maturity=max(0.0,min(1.0,_finite(maturity,"maturity")));stage=_current_stage(stage);prim=stage.GetPrimAtPath(bond.band_path)
        if prim and prim.IsValid() and not bond.failed:
            prim.GetVariantSets().GetVariantSet("state").SetVariantSelection("mature" if bond.maturity>=0.90 else "fresh")
        return bond.maturity
    def break_force_n(self,bond: SealBandBond):
        return self.initial_break_force_n+(self.mature_break_force_n-self.initial_break_force_n)*bond.maturity
    def apply_load(self,bond: SealBandBond,load_n: float,*,stage=None):
        load=_finite(load_n,"load_n")
        if bond.failed or abs(load)<=self.break_force_n(bond):return False
        remove_prims(bond.attachment_paths,stage=stage);bond.failed=True;stage=_current_stage(stage);prim=stage.GetPrimAtPath(bond.band_path)
        if prim and prim.IsValid():prim.GetVariantSets().GetVariantSet("state").SetVariantSelection("failed")
        return True

@dataclass
class StumpSealState:
    maturity: float=0.0
    damage: float=0.0
    residual_gap_fraction: float=1.0

@dataclass
class DualStumpLeakModel:
    reference_area_m2: float=5.54e-5
    pressure_pa: float=16000.0
    density_kg_m3: float=1060.0
    discharge_coefficient: float=0.62
    left: StumpSealState=field(default_factory=StumpSealState)
    right: StumpSealState=field(default_factory=StumpSealState)
    def effective_area_m2(self,state: StumpSealState):
        seal=max(0.0,min(1.0,_finite(state.maturity,"maturity")))
        gap=max(0.0,min(1.0,_finite(state.residual_gap_fraction,"residual_gap_fraction")))
        damage=_finite_nonnegative(state.damage,"damage")
        return self.reference_area_m2*(gap**2.2)*((1.0-seal)**3.0)*(1.0+damage)
    def flow_ml_min(self,state: StumpSealState):
        area=self.effective_area_m2(state);q=self.discharge_coefficient*area*math.sqrt(2.0*max(0.0,self.pressure_pa)/max(self.density_kg_m3,1e-9));return q*60.0*1e6
    def flows(self):return {"left_ml_min":self.flow_ml_min(self.left),"right_ml_min":self.flow_ml_min(self.right)}

@dataclass
class BladeInterlockController:
    minimum_jaw_force_n: float=8.0
    maximum_jaw_force_n: float=32.0
    maximum_stump_flow_ml_min: float=0.1
    def evaluate(self,energy: AdaptiveSealEnergyController,leak: DualStumpLeakModel,upper_force_n: float,lower_force_n: float,guard_retracted: bool,*,tissue_centered: bool):
        reasons=[]
        if tissue_centered is not True:reasons.append("tissue_not_centered")
        total=_finite_nonnegative(upper_force_n,"upper_force_n")+_finite_nonnegative(lower_force_n,"lower_force_n")
        if total<self.minimum_jaw_force_n:reasons.append("insufficient_compression")
        if total>self.maximum_jaw_force_n:reasons.append("excess_compression")
        if not energy.zone_ready(energy.left):reasons.append("left_seal_not_ready")
        if not energy.zone_ready(energy.right):reasons.append("right_seal_not_ready")
        flows=leak.flows()
        if flows["left_ml_min"]>self.maximum_stump_flow_ml_min:reasons.append("left_predicted_leak")
        if flows["right_ml_min"]>self.maximum_stump_flow_ml_min:reasons.append("right_predicted_leak")
        if not guard_retracted:reasons.append("blade_guard_not_retracted")
        return {"authorized":not reasons,"reasons":reasons,"predicted_flows":flows,"tissue_centered":tissue_centered is True}

@dataclass
class TissueDivisionController:
    bridge: BridgeAttachmentController
    interlock: BladeInterlockController=field(default_factory=BladeInterlockController)
    blade_progress: float=0.0
    violations: int=0
    def advance(self,progress: float,*,energy: AdaptiveSealEnergyController,leak: DualStumpLeakModel,upper_force_n: float,lower_force_n: float,guard_retracted: bool,tissue_centered: bool,stage=None):
        result=self.interlock.evaluate(energy,leak,upper_force_n,lower_force_n,guard_retracted,tissue_centered=tissue_centered)
        requested=max(0.0,min(1.0,_finite(progress,"progress")))
        if requested>self.blade_progress and not result["authorized"]:
            self.violations+=1;return {**result,"blade_progress":self.blade_progress,"bridge_release_fraction":self.bridge.released_fraction}
        self.blade_progress=max(self.blade_progress,requested);released=self.bridge.set_cut_progress(self.blade_progress,stage=stage)
        return {**result,"blade_progress":self.blade_progress,"bridge_release_fraction":released,"division_complete":self.bridge.complete}

PHASE_TARGETS={
    "inspect":{"left_centering_joint":0.0,"right_centering_joint":0.0,"upper_jaw_joint":0.0,"lower_jaw_joint":0.0,"blade_guard_joint":0.0,"blade_joint":0.0,"suction_valve_joint":0.0,"irrigation_valve_joint":0.0},
    "center":{"left_centering_joint":0.022,"right_centering_joint":-0.022,"upper_jaw_joint":0.0,"lower_jaw_joint":0.0,"blade_guard_joint":0.0,"blade_joint":0.0,"suction_valve_joint":0.002,"irrigation_valve_joint":0.0},
    "compress":{"left_centering_joint":0.022,"right_centering_joint":-0.022,"upper_jaw_joint":0.010,"lower_jaw_joint":-0.010,"blade_guard_joint":0.0,"blade_joint":0.0,"suction_valve_joint":0.004,"irrigation_valve_joint":0.0},
    # The inward electrode faces meet at +/-10 mm jaw travel.  Holding that
    # geometry through seal and division avoids the former 3 mm per-jaw
    # crossover, which pulled captured tissue apart instead of compressing it.
    "seal":{"left_centering_joint":0.022,"right_centering_joint":-0.022,"upper_jaw_joint":0.010,"lower_jaw_joint":-0.010,"blade_guard_joint":0.0,"blade_joint":0.0,"suction_valve_joint":0.005,"irrigation_valve_joint":0.0},
    "verify_seal":{"left_centering_joint":0.022,"right_centering_joint":-0.022,"upper_jaw_joint":0.010,"lower_jaw_joint":-0.010,"blade_guard_joint":0.0,"blade_joint":0.0,"suction_valve_joint":0.003,"irrigation_valve_joint":0.0},
    "retract_guard":{"left_centering_joint":0.022,"right_centering_joint":-0.022,"upper_jaw_joint":0.010,"lower_jaw_joint":-0.010,"blade_guard_joint":-0.011,"blade_joint":0.0,"suction_valve_joint":0.004,"irrigation_valve_joint":0.0},
    "divide":{"left_centering_joint":0.022,"right_centering_joint":-0.022,"upper_jaw_joint":0.010,"lower_jaw_joint":-0.010,"blade_guard_joint":-0.011,"blade_joint":0.041,"suction_valve_joint":0.006,"irrigation_valve_joint":0.0},
    "release":{"left_centering_joint":0.0,"right_centering_joint":0.0,"upper_jaw_joint":0.0,"lower_jaw_joint":0.0,"blade_guard_joint":0.0,"blade_joint":0.0,"suction_valve_joint":0.003,"irrigation_valve_joint":0.003},
    "verify_stumps":{"left_centering_joint":0.0,"right_centering_joint":0.0,"upper_jaw_joint":0.0,"lower_jaw_joint":0.0,"blade_guard_joint":0.0,"blade_joint":0.0,"suction_valve_joint":0.002,"irrigation_valve_joint":0.0},
    "complete":{"left_centering_joint":0.0,"right_centering_joint":0.0,"upper_jaw_joint":0.0,"lower_jaw_joint":0.0,"blade_guard_joint":0.0,"blade_joint":0.0,"suction_valve_joint":0.0,"irrigation_valve_joint":0.0},
    "abort":{"left_centering_joint":0.0,"right_centering_joint":0.0,"upper_jaw_joint":0.0,"lower_jaw_joint":0.0,"blade_guard_joint":0.0,"blade_joint":0.0,"suction_valve_joint":0.008,"irrigation_valve_joint":0.004},
}

def phase_targets(phase: str):
    try:return dict(PHASE_TARGETS[phase])
    except KeyError as exc:raise KeyError(f"Unknown seal/divide phase {phase!r}") from exc

@dataclass
class AdaptiveSealDivideSequenceController:
    phase: str="inspect"
    energy: AdaptiveSealEnergyController=field(default_factory=AdaptiveSealEnergyController)
    leak: DualStumpLeakModel=field(default_factory=DualStumpLeakModel)
    history: list[str]=field(default_factory=list)
    def transition(self,phase: str):phase_targets(phase);self.phase=phase;self.history.append(phase);return phase_targets(phase)
    def update_energy(self,dt: float,left_force_n: float,right_force_n: float,left_power_w: float|None=None,right_power_w: float|None=None):
        left,right=self.energy.update(dt,left_force_n,right_force_n,left_power_w,right_power_w);self.leak.left.maturity=left.maturity;self.leak.right.maturity=right.maturity;self.leak.left.residual_gap_fraction=max(0.0,1.0-left_force_n/24.0);self.leak.right.residual_gap_fraction=max(0.0,1.0-right_force_n/24.0);return {"left":left,"right":right,"flows":self.leak.flows(),"both_ready":self.energy.both_ready}
