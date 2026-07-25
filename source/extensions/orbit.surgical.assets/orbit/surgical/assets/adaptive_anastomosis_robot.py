# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Isaac Lab integration for the DrAnmar Adaptive Anastomosis Robot.

The payload replaces the Panda hand at ``panda_link8``. Runtime helpers provide
bilateral hollow-tissue capture, coaxial approximation, surface-deformable
attachment management, circumferential retained-staple deployment,
reinforcement-collar bonding, lumen-patency metrics, and pressure-decay leak
verification. All physical values are provisional research parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
import math

CATALOG_SUBPATH = "Props/SurgicalReconstruction/AdaptiveAnastomosisRobot"
ASSET_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
ROOT = ASSET_DATA_ROOT / CATALOG_SUBPATH
TOOL_PAYLOAD_USD = ROOT / "dranmar_adaptive_anastomosis_tool_payload.usda"
TOOL_STANDALONE_USD = ROOT / "dranmar_adaptive_anastomosis_tool_standalone.usda"
TOOL_RIGID_PROXY_USD = ROOT / "dranmar_adaptive_anastomosis_tool_rigid_proxy.usda"
TISSUE_USD = ROOT / "dranmar_hollow_tissue_demo.usda"
STAPLE_USD = ROOT / "dranmar_anastomosis_staple.usda"
COLLAR_USD = ROOT / "dranmar_reinforcement_collar.usda"
COLLAR_PROXY_USD = ROOT / "dranmar_reinforcement_collar_rigid_proxy.usda"
DROPLET_USD = ROOT / "dranmar_leak_test_droplet.usda"

VALID_BINARY_STATES = frozenset({"loaded", "empty"})
VALID_TEST_MEDIUM_STATES = frozenset({"full", "empty"})
VALID_COLLECTION_STATES = frozenset({"empty", "partial", "full"})
VALID_TISSUE_STATES = frozenset({"initial", "aligned", "completed"})
STAPLE_COUNT = 16
CAPTURE_CELL_COUNT_PER_SIDE = 6
COLLAR_SECTOR_COUNT = 16
STAPLE_RING_RADIUS_M = 0.0112
LEAK_PARTICLE_RADIUS_M = 0.0009
LEAK_PARTICLE_VOLUME_ML = 4.0 / 3.0 * math.pi * LEAK_PARTICLE_RADIUS_M**3 * 1.0e6

TOOL_JOINTS = {
    "left_approximation":"left_approximation_joint",
    "right_approximation":"right_approximation_joint",
    "left_capture":"left_capture_joint",
    "right_capture":"right_capture_joint",
    "left_eversion":"left_eversion_joint",
    "right_eversion":"right_eversion_joint",
    "mandrel_extension":"mandrel_extension_joint",
    "mandrel_expansion":"mandrel_expansion_joint",
    "staple_driver":"staple_driver_joint",
    "collar_carousel":"collar_carousel_joint",
    "collar_applicator":"collar_applicator_joint",
    "left_occluder_valve":"left_occluder_valve_joint",
    "right_occluder_valve":"right_occluder_valve_joint",
    "pressure_valve":"pressure_valve_joint",
}
TOOL_FRAME_PATHS = {
    "panda_link8_mount":"Links/Mount/Frames/panda_link8_mount",
    "anastomosis_tcp":"Links/Mount/Frames/anastomosis_tcp",
    "seam_reference":"Links/Mount/Frames/seam_reference",
    "lumen_axis_reference":"Links/Mount/Frames/lumen_axis_reference",
    "camera_left":"Links/Mount/Frames/camera_left",
    "camera_right":"Links/Mount/Frames/camera_right",
    "pressure_sensor":"Links/Mount/Frames/pressure_sensor",
    "leak_observation":"Links/Mount/Frames/leak_observation",
    "count_reference":"Links/Mount/Frames/count_reference",
    "disposal_reference":"Links/Mount/Frames/disposal_reference",
    "left_capture_center":"Links/LeftCaptureSleeve/Frames/left_capture_center",
    "left_tissue_edge_reference":"Links/LeftCaptureSleeve/Frames/left_tissue_edge_reference",
    "right_capture_center":"Links/RightCaptureSleeve/Frames/right_capture_center",
    "right_tissue_edge_reference":"Links/RightCaptureSleeve/Frames/right_tissue_edge_reference",
    "left_eversion_contact":"Links/LeftEversionRing/Frames/left_eversion_contact",
    "right_eversion_contact":"Links/RightEversionRing/Frames/right_eversion_contact",
    "mandrel_tip":"Links/Mandrel/Frames/mandrel_tip",
    "pressure_inlet":"Links/Mandrel/Frames/pressure_inlet",
    "patency_reference":"Links/MandrelExpander/Frames/patency_reference",
    "staple_anvil_reference":"Links/StapleAnvil/Frames/staple_anvil_reference",
    "staple_crown_reference":"Links/StapleDriver/Frames/staple_crown_reference",
    "collar_application":"Links/CollarApplicator/Frames/collar_application",
}
REGISTERED_CAMERA_FRAMES = ("camera_left", "camera_right")


def frame_path(tool_path: str, name: str) -> str:
    try: suffix=TOOL_FRAME_PATHS[name]
    except KeyError as exc: raise KeyError(f"Unknown anastomosis frame {name!r}") from exc
    return f"{tool_path.rstrip('/')}/{suffix}"


def tensor_value(value: Any):
    return value.torch if hasattr(value,"torch") else value


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


def make_tool_cfg(
    prim_path: str="/World/DrAnmarAdaptiveAnastomosisTool",
    *,
    staple_state: str="loaded",
    collar_state: str="loaded",
    test_medium_state: str="full",
    collection_state: str="empty",
    position=(0,0,0.35),
    orientation_wxyz=(1,0,0,0),
):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    _check(staple_state,VALID_BINARY_STATES,"staple_state")
    _check(collar_state,VALID_BINARY_STATES,"collar_state")
    _check(test_medium_state,VALID_TEST_MEDIUM_STATES,"test_medium_state")
    _check(collection_state,VALID_COLLECTION_STATES,"collection_state")
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(TOOL_STANDALONE_USD),
            variants={"staple_state":staple_state,"collar_state":collar_state,"test_medium_state":test_medium_state,"collection_state":collection_state},
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False,solver_position_iteration_count=24,solver_velocity_iteration_count=8),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=position,rot=_xyzw_from_wxyz(orientation_wxyz),joint_pos={name:0.0 for name in TOOL_JOINTS.values()}),
        actuators={
            "approximation":ImplicitActuatorCfg(joint_names_expr=[".*approximation_joint"],effort_limit_sim=180.0,velocity_limit_sim=0.16,stiffness=9000.0,damping=260.0),
            "capture_eversion":ImplicitActuatorCfg(joint_names_expr=[".*capture_joint",".*eversion_joint"],effort_limit_sim=130.0,velocity_limit_sim=0.14,stiffness=6800.0,damping=205.0),
            "mandrel":ImplicitActuatorCfg(joint_names_expr=["mandrel_.*_joint"],effort_limit_sim=95.0,velocity_limit_sim=0.20,stiffness=5200.0,damping=155.0),
            "staple":ImplicitActuatorCfg(joint_names_expr=["staple_driver_joint"],effort_limit_sim=340.0,velocity_limit_sim=0.28,stiffness=17000.0,damping=360.0),
            "collar":ImplicitActuatorCfg(joint_names_expr=["collar_.*_joint"],effort_limit_sim=125.0,velocity_limit_sim=1.4,stiffness=7000.0,damping=190.0),
            "valves":ImplicitActuatorCfg(joint_names_expr=[".*_valve_joint"],effort_limit_sim=32.0,velocity_limit_sim=0.25,stiffness=1750.0,damping=55.0),
        },
    )


def make_rigid_proxy_cfg(prim_path="/World/DrAnmarAdaptiveAnastomosisProxy", *, position=(0,0,0.35), orientation_wxyz=(1,0,0,0)):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(usd_path=str(TOOL_RIGID_PROXY_USD),activate_contact_sensors=True),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position,rot=_xyzw_from_wxyz(orientation_wxyz)),
    )


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
        mount_local_rot0=Gf.Quatf(math.cos(half_angle),0,0,math.sin(half_angle))
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
    tool_path=f"{prim_path}/DrAnmarAdaptiveAnastomosisTool"
    create_prim(tool_path,usd_path=str(TOOL_PAYLOAD_USD),stage=stage)
    select_usd_variants(tool_path,{"staple_state":cfg.staple_state,"collar_state":cfg.collar_state,"test_medium_state":cfg.test_medium_state,"collection_state":cfg.collection_state})
    joint=UsdPhysics.FixedJoint.Define(stage,f"{prim_path}/dranmar_anastomosis_mount_joint")
    joint.CreateBody0Rel().SetTargets(mount_body_paths)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{tool_path}/Links/Mount")])
    joint.CreateLocalPos0Attr().Set(mount_local_pos0);joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0,0,0))
    joint.CreateLocalRot0Attr().Set(mount_local_rot0)
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1,0,0,0))
    return robot


def spawn_franka_with_tool(prim_path: str,cfg: Any,translation=None,orientation=None,**kwargs):
    from isaaclab.sim.utils import clone
    return clone(_spawn_single_franka_with_tool)(prim_path,cfg,translation=translation,orientation=orientation,**kwargs)


def make_franka_adaptive_anastomosis_robot_cfg(
    *,
    prim_path="/World/Robot",
    staple_state="loaded",
    collar_state="loaded",
    test_medium_state="full",
    collection_state="empty",
):
    _check(staple_state,VALID_BINARY_STATES,"staple_state")
    _check(collar_state,VALID_BINARY_STATES,"collar_state")
    _check(test_medium_state,VALID_TEST_MEDIUM_STATES,"test_medium_state")
    _check(collection_state,VALID_COLLECTION_STATES,"collection_state")
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
    @configclass
    class FrankaAnastomosisUsdCfg(sim_utils.UsdFileCfg):
        staple_state: str="loaded"
        collar_state: str="loaded"
        test_medium_state: str="full"
        collection_state: str="empty"
        func=spawn_franka_with_tool
    cfg=FRANKA_PANDA_CFG.copy();cfg.prim_path=prim_path
    cfg.spawn=FrankaAnastomosisUsdCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        variants={"Gripper":"Default","Mesh":"Performance"},
        staple_state=staple_state,collar_state=collar_state,test_medium_state=test_medium_state,collection_state=collection_state,
        activate_contact_sensors=True,rigid_props=FRANKA_PANDA_CFG.spawn.rigid_props,articulation_props=FRANKA_PANDA_CFG.spawn.articulation_props,
    )
    cfg.init_state.joint_pos={k:v for k,v in cfg.init_state.joint_pos.items() if "finger" not in k}
    cfg.init_state.joint_pos.update({name:0.0 for name in TOOL_JOINTS.values()})
    cfg.actuators={k:v for k,v in cfg.actuators.items() if k!="panda_hand"}
    cfg.actuators.update({
        "anastomosis_approximation":ImplicitActuatorCfg(joint_names_expr=[".*approximation_joint"],effort_limit_sim=180.0,velocity_limit_sim=0.16,stiffness=9000.0,damping=260.0),
        "anastomosis_capture_eversion":ImplicitActuatorCfg(joint_names_expr=[".*capture_joint",".*eversion_joint"],effort_limit_sim=130.0,velocity_limit_sim=0.14,stiffness=6800.0,damping=205.0),
        "anastomosis_mandrel":ImplicitActuatorCfg(joint_names_expr=["mandrel_.*_joint"],effort_limit_sim=95.0,velocity_limit_sim=0.20,stiffness=5200.0,damping=155.0),
        "anastomosis_staple":ImplicitActuatorCfg(joint_names_expr=["staple_driver_joint"],effort_limit_sim=340.0,velocity_limit_sim=0.28,stiffness=17000.0,damping=360.0),
        "anastomosis_collar":ImplicitActuatorCfg(joint_names_expr=["collar_.*_joint"],effort_limit_sim=125.0,velocity_limit_sim=1.4,stiffness=7000.0,damping=190.0),
        "anastomosis_valves":ImplicitActuatorCfg(joint_names_expr=[".*_valve_joint"],effort_limit_sim=32.0,velocity_limit_sim=0.25,stiffness=1750.0,damping=55.0),
    })
    return cfg


def _current_stage(stage=None):
    if stage is not None:return stage
    import omni.usd
    return omni.usd.get_context().get_stage()


def spawn_hollow_tissue_demo(prim_path="/World/DrAnmarHollowTissue", *, state="initial", translation=(0,0,0), orientation_wxyz=(1,0,0,0)):
    _check(state,VALID_TISSUE_STATES,"state")
    import isaaclab.sim as sim_utils
    cfg=sim_utils.UsdFileCfg(usd_path=str(TISSUE_USD),variants={"state":state})
    return cfg.func(prim_path,cfg,translation=translation,orientation=_xyzw_from_wxyz(orientation_wxyz))


def _create_surface_material(stage, material_path: str):
    from pxr import UsdShade
    material=UsdShade.Material.Define(stage,material_path);prim=material.GetPrim()
    for schema in ("OmniPhysicsBaseMaterialAPI","OmniPhysicsDeformableMaterialAPI","OmniPhysicsSurfaceDeformableMaterialAPI","PhysxSurfaceDeformableMaterialAPI"):
        try:prim.ApplyAPI(schema)
        except Exception:pass
    for name,value in {
        "omniphysics:dynamicFriction":0.38,"omniphysics:density":1060.0,"omniphysics:youngsModulus":180000.0,
        "omniphysics:poissonsRatio":0.47,"omniphysics:surfaceThickness":0.0024,"omniphysics:surfaceBendStiffness":0.0,
        "physxDeformableMaterial:elasticityDamping":0.16,"physxDeformableMaterial:bendDamping":0.18,
    }.items():
        attr=prim.GetAttribute(name)
        if attr:attr.Set(value)
    return material


def apply_hollow_tissue_surface_deformables(root_path: str, *, self_collision=False, stage=None):
    from omni.physx.scripts import deformableUtils
    from pxr import UsdShade
    stage=_current_stage(stage);out={}
    material=_create_surface_material(stage,f"{root_path}/RuntimeMaterials/TissueSurface")
    for side in ("LeftTissue","RightTissue"):
        mesh_path=f"{root_path}/{side}/SimulationMesh";mesh=stage.GetPrimAtPath(mesh_path)
        if not mesh or not mesh.IsValid():raise ValueError(f"Missing {mesh_path}")
        result=deformableUtils.set_physics_surface_deformable_body(stage,mesh.GetPath())
        if result is False:raise RuntimeError(f"Failed to cook {mesh_path}")
        try:mesh.ApplyAPI("PhysxSurfaceDeformableBodyAPI");mesh.GetAttribute("physxDeformableBody:selfCollision").Set(bool(self_collision))
        except Exception:pass
        binding=UsdShade.MaterialBindingAPI.Apply(mesh);binding.Bind(material,UsdShade.Tokens.weakerThanDescendants,"physics")
        out[side]=mesh_path
    return out


def create_deformable_attachment(deformable_path: str,target_path: str,attachment_path: str,*,stage=None):
    """Create an overlap-prioritized attachment across Isaac generations."""
    from pxr import Gf,Sdf,Usd,UsdGeom,Vt
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
        mesh_to_world=UsdGeom.Xformable(deformable).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        target_to_world=UsdGeom.Xformable(target).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        world_to_target=target_to_world.GetInverse()
        bounds=UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_,UsdGeom.Tokens.guide],
        ).ComputeWorldBound(target).ComputeAlignedRange()
        minimum,maximum=bounds.GetMin(),bounds.GetMax()
        center=(minimum+maximum)*0.5
        ranked=[]
        for index,point in enumerate(points):
            world=mesh_to_world.Transform(Gf.Vec3d(point))
            delta=world-center
            overlaps=all(
                minimum[axis]-0.0025<=world[axis]<=maximum[axis]+0.0025
                for axis in range(3)
            )
            ranked.append((float(Gf.Dot(delta,delta)),index,world,overlaps))
        ranked.sort(key=lambda item:item[0])
        selected=[item for item in ranked if item[3]][:12]
        if len(selected)<4:
            raise RuntimeError(
                f"Attachment capture volume does not overlap enough deformable "
                f"vertices for {attachment_path}: source={deformable_path}, "
                f"target={target_path}, overlapping={len(selected)}, "
                "required=4, overlap_margin_m=0.0025"
            )
        attachment=stage.DefinePrim(
            attachment_path,"OmniPhysicsVtxXformAttachment"
        )
        attachment.CreateRelationship("omniphysics:src0").SetTargets(
            [Sdf.Path(deformable_path)]
        )
        attachment.CreateRelationship("omniphysics:src1").SetTargets(
            [Sdf.Path(target_path)]
        )
        attachment.CreateAttribute(
            "omniphysics:vtxIndicesSrc0",Sdf.ValueTypeNames.IntArray
        ).Set(Vt.IntArray([item[1] for item in selected]))
        attachment.CreateAttribute(
            "omniphysics:localPositionsSrc1",Sdf.ValueTypeNames.Point3fArray
        ).Set(Vt.Vec3fArray([
            Gf.Vec3f(world_to_target.Transform(item[2])) for item in selected
        ]))
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


def anchor_hollow_tissue_distal_ends(root_path: str,*,stage=None) -> list[str]:
    """Attach both cooked tissue surfaces to explicit kinematic fixtures."""
    from pxr import UsdPhysics
    stage=_current_stage(stage)
    root_path=root_path.rstrip("/")
    attachments_root=f"{root_path}/RuntimeFixtureAttachments"
    stage.DefinePrim(attachments_root,"Scope")
    created=[]
    try:
        for label,tissue,target in (
            ("left","LeftTissue/SimulationMesh","LeftFixtureAnchor"),
            ("right","RightTissue/SimulationMesh","RightFixtureAnchor"),
        ):
            target_path=f"{root_path}/{target}"
            target_prim=stage.GetPrimAtPath(target_path)
            if not target_prim.IsValid():
                raise ValueError(f"Tissue fixture anchor is missing: {target_path}")
            rigid_body=UsdPhysics.RigidBodyAPI.Apply(target_prim)
            rigid_body.CreateRigidBodyEnabledAttr(True)
            rigid_body.CreateKinematicEnabledAttr(True)
            attachment_path=f"{attachments_root}/{label}"
            create_deformable_attachment(
                f"{root_path}/{tissue}",target_path,attachment_path,stage=stage
            )
            created.append(attachment_path)
    except Exception:
        remove_prims(created,stage=stage)
        raise
    return created


def remove_prims(paths: Iterable[str],*,stage=None):
    stage=_current_stage(stage)
    for path in paths:
        if stage.GetPrimAtPath(path).IsValid():stage.RemovePrim(path)


def _nonnegative_finite(value: float,label: str) -> float:
    amount=float(value)
    if not math.isfinite(amount) or amount<0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return amount


def _fraction(value: float,label: str) -> float:
    amount=float(value)
    if not math.isfinite(amount):
        raise ValueError(f"{label} must be finite")
    return max(0.0,min(1.0,amount))


@dataclass
class SideCapture:
    side: str
    tissue_path: str
    attachment_paths: list[str]=field(default_factory=list)
    engaged: bool=False


@dataclass
class BilateralTissueCaptureController:
    tool_path: str
    left_tissue_path: str
    right_tissue_path: str
    cells_per_side: int=CAPTURE_CELL_COUNT_PER_SIDE
    target_force_per_side_n: float=1.6
    soft_force_limit_n: float=3.5
    hard_release_limit_n: float=6.0
    left: SideCapture=field(init=False)
    right: SideCapture=field(init=False)
    def __post_init__(self):
        if int(self.cells_per_side)!=self.cells_per_side or self.cells_per_side<=0:
            raise ValueError("cells_per_side must be a positive integer")
        self.cells_per_side=int(self.cells_per_side)
        self.target_force_per_side_n=_nonnegative_finite(
            self.target_force_per_side_n,"target_force_per_side_n"
        )
        self.soft_force_limit_n=_nonnegative_finite(
            self.soft_force_limit_n,"soft_force_limit_n"
        )
        self.hard_release_limit_n=_nonnegative_finite(
            self.hard_release_limit_n,"hard_release_limit_n"
        )
        if not (
            self.target_force_per_side_n
            <=self.soft_force_limit_n
            <=self.hard_release_limit_n
        ):
            raise ValueError("capture limits must be target <= soft <= hard")
        self.left=SideCapture("left",self.left_tissue_path);self.right=SideCapture("right",self.right_tissue_path)
    def engage_side(self,capture: SideCapture,*,stage=None):
        if capture.engaged:return list(capture.attachment_paths)
        stage=_current_stage(stage);stage.DefinePrim(f"{self.tool_path}/RuntimeAttachments","Scope");created=[]
        link="LeftCaptureSleeve" if capture.side=="left" else "RightCaptureSleeve"
        try:
            for i in range(self.cells_per_side):
                ap=f"{self.tool_path}/RuntimeAttachments/{capture.side}_capture_{i:02d}"
                target=f"{self.tool_path}/Links/{link}/Collisions/CaptureCell_{i:02d}"
                create_deformable_attachment(capture.tissue_path,target,ap,stage=stage);created.append(ap)
        except Exception:
            remove_prims(created,stage=stage);raise
        capture.attachment_paths=created;capture.engaged=True;return list(created)
    def engage(self,*,stage=None):
        return {"left":self.engage_side(self.left,stage=stage),"right":self.engage_side(self.right,stage=stage)}
    def release_side(self,capture: SideCapture,*,stage=None):
        remove_prims(capture.attachment_paths,stage=stage);capture.attachment_paths.clear();capture.engaged=False
    def release(self,*,stage=None):self.release_side(self.left,stage=stage);self.release_side(self.right,stage=stage)
    def update_loads(self,left_force_n: float,right_force_n: float,*,stage=None):
        left=_nonnegative_finite(abs(float(left_force_n)),"left_force_n")
        right=_nonnegative_finite(abs(float(right_force_n)),"right_force_n")
        peak=max(left,right)
        hard=peak>self.hard_release_limit_n
        soft=peak>self.soft_force_limit_n
        if hard and (self.left.engaged or self.right.engaged):
            self.release(stage=stage)
        return {
            "mode":"hard_release" if hard else "soft_limit" if soft else "controlled",
            "left_force_n":left,
            "right_force_n":right,
            "left_target_error_n":left-self.target_force_per_side_n,
            "right_target_error_n":right-self.target_force_per_side_n,
            "engaged":self.left.engaged and self.right.engaged,
        }


def _spawn_reference_at_transform(stage,prim_path: str,usd_path: Path,world_transform: Any,variants: dict[str,str]|None=None):
    from pxr import Gf,UsdGeom
    prim=stage.DefinePrim(prim_path,"Xform");prim.GetReferences().AddReference(str(usd_path))
    UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d(world_transform))
    if variants:
        for name,value in variants.items():prim.GetVariantSets().GetVariantSet(name).SetVariantSelection(value)
    return prim


def _ring_local_matrix(angle_rad: float,radius_m: float):
    from pxr import Gf
    rotation=Gf.Matrix4d(1.0);rotation.SetRotate(Gf.Rotation(Gf.Vec3d(1,0,0),math.degrees(angle_rad)))
    translation=Gf.Matrix4d(1.0);translation.SetTranslate(Gf.Vec3d(0,radius_m,0))
    return translation*rotation


def deploy_staple_ring(
    parent_path: str,
    crown_world_transform: Any,
    left_tissue_path: str,
    right_tissue_path: str,
    *,
    staple_count: int=STAPLE_COUNT,
    radius_m: float=STAPLE_RING_RADIUS_M,
    stage=None,
):
    if int(staple_count)!=staple_count or not 1<=staple_count<=STAPLE_COUNT:
        raise ValueError(f"staple_count must be an integer in [1, {STAPLE_COUNT}]")
    staple_count=int(staple_count)
    radius_m=_nonnegative_finite(radius_m,"radius_m")
    if radius_m==0.0:
        raise ValueError("radius_m must be positive")
    stage=_current_stage(stage);stage.DefinePrim(parent_path,"Scope");deployments=[]
    current_path=None
    try:
        for i in range(staple_count):
            angle=2*math.pi*i/staple_count
            path=f"{parent_path}/Staple_{i:02d}"
            current_path=path
            world=_ring_local_matrix(angle,radius_m)*crown_world_transform
            _spawn_reference_at_transform(stage,path,STAPLE_USD,world,{"state":"formed"})
            stage.DefinePrim(f"{path}/Attachments","Scope")
            left_ap=f"{path}/Attachments/left";right_ap=f"{path}/Attachments/right"
            create_deformable_attachment(left_tissue_path,f"{path}/Collisions/LeftLegAttachment",left_ap,stage=stage)
            create_deformable_attachment(right_tissue_path,f"{path}/Collisions/RightLegAttachment",right_ap,stage=stage)
            deployments.append({"staple_path":path,"angle_rad":angle,"attachment_paths":[left_ap,right_ap],"retained":True})
            current_path=None
    except Exception:
        paths=[d["staple_path"] for d in deployments]
        if current_path is not None:
            paths.append(current_path)
        remove_prims(paths,stage=stage);raise
    return deployments


@dataclass
class StapleRingRetentionController:
    pullout_force_per_staple_n: float=1.4
    deployments: list[dict[str,Any]]=field(default_factory=list)
    def __post_init__(self):
        self.pullout_force_per_staple_n=_nonnegative_finite(
            self.pullout_force_per_staple_n,"pullout_force_per_staple_n"
        )
    def register(self,deployments):
        additions=list(deployments)
        for deployment in additions:
            if "staple_path" not in deployment or len(deployment.get("attachment_paths",[]))!=2:
                raise ValueError("each staple deployment needs one path and two attachments")
        self.deployments.extend(additions);return additions
    @property
    def retained_fraction(self):
        return 0.0 if not self.deployments else sum(bool(d.get("retained",False)) for d in self.deployments)/len(self.deployments)
    def apply_loads(self,loads_n: Sequence[float],*,stage=None):
        loads=list(loads_n)
        if len(loads)!=len(self.deployments):
            raise ValueError("loads_n must match the registered staple count")
        released=[]
        for deployment,load in zip(self.deployments,loads):
            magnitude=_nonnegative_finite(abs(float(load)),"staple_load_n")
            if deployment.get("retained",False) and magnitude>self.pullout_force_per_staple_n:
                remove_prims(deployment["attachment_paths"],stage=stage);deployment["retained"]=False;released.append(deployment["staple_path"])
        return released


@dataclass
class CollarBond:
    collar_path: str
    attachment_paths: list[str]
    cure_fraction: float=0.0
    broken_sectors: set[int]=field(default_factory=set)


@dataclass
class ReinforcementCollarBondController:
    cure_time_s: float=45.0
    initial_sector_tack_force_n: float=0.18
    final_sector_break_force_n: float=2.2
    bonds: list[CollarBond]=field(default_factory=list)
    def __post_init__(self):
        self.cure_time_s=_nonnegative_finite(self.cure_time_s,"cure_time_s")
        self.initial_sector_tack_force_n=_nonnegative_finite(
            self.initial_sector_tack_force_n,"initial_sector_tack_force_n"
        )
        self.final_sector_break_force_n=_nonnegative_finite(
            self.final_sector_break_force_n,"final_sector_break_force_n"
        )
        if self.cure_time_s==0.0:
            raise ValueError("cure_time_s must be positive")
        if self.initial_sector_tack_force_n>self.final_sector_break_force_n:
            raise ValueError("initial tack cannot exceed cured break force")
    def deploy(self,prim_path: str,world_transform: Any,left_tissue_path: str,right_tissue_path: str,*,stage=None):
        stage=_current_stage(stage);_spawn_reference_at_transform(stage,prim_path,COLLAR_PROXY_USD,world_transform,{"state":"fresh"});stage.DefinePrim(f"{prim_path}/Attachments","Scope");created=[]
        try:
            for i in range(COLLAR_SECTOR_COUNT):
                for side,tissue in (("Left",left_tissue_path),("Right",right_tissue_path)):
                    ap=f"{prim_path}/Attachments/{side.lower()}_{i:02d}"
                    create_deformable_attachment(tissue,f"{prim_path}/Collisions/{side}BondCell_{i:02d}",ap,stage=stage);created.append(ap)
        except Exception:
            remove_prims(created+[prim_path],stage=stage);raise
        bond=CollarBond(prim_path,created);self.bonds.append(bond);return bond
    def update(self,dt: float):
        elapsed=_nonnegative_finite(dt,"dt")
        for bond in self.bonds:
            bond.cure_fraction=min(1.0,bond.cure_fraction+elapsed/self.cure_time_s)
    def apply_sector_load(self,bond: CollarBond,sector: int,load_n: float,*,stage=None):
        if int(sector)!=sector or not 0<=sector<COLLAR_SECTOR_COUNT:
            raise ValueError(f"sector must be an integer in [0, {COLLAR_SECTOR_COUNT-1}]")
        sector=int(sector)
        if sector in bond.broken_sectors:return False
        cure=_fraction(bond.cure_fraction,"cure_fraction")
        threshold=self.initial_sector_tack_force_n+(
            self.final_sector_break_force_n-self.initial_sector_tack_force_n
        )*cure
        if _nonnegative_finite(abs(float(load_n)),"load_n")<=threshold:return False
        paths=[f"{bond.collar_path}/Attachments/left_{sector:02d}",f"{bond.collar_path}/Attachments/right_{sector:02d}"]
        remove_prims(paths,stage=stage);bond.broken_sectors.add(int(sector));return True
    def bonded_fraction(self,bond: CollarBond):return max(0.0,1.0-len(bond.broken_sectors)/COLLAR_SECTOR_COUNT)


@dataclass
class PatencyReport:
    minimum_radius_m: float
    mean_radius_m: float
    area_fraction: float
    centerline_offset_m: float
    axis_error_deg: float
    passed: bool


@dataclass
class LumenPatencyController:
    reference_radius_m: float=0.0095
    minimum_accepted_radius_m: float=0.0085
    maximum_centerline_offset_m: float=0.0025
    maximum_axis_error_deg: float=7.0
    def __post_init__(self):
        self.reference_radius_m=_nonnegative_finite(
            self.reference_radius_m,"reference_radius_m"
        )
        self.minimum_accepted_radius_m=_nonnegative_finite(
            self.minimum_accepted_radius_m,"minimum_accepted_radius_m"
        )
        self.maximum_centerline_offset_m=_nonnegative_finite(
            self.maximum_centerline_offset_m,"maximum_centerline_offset_m"
        )
        self.maximum_axis_error_deg=_nonnegative_finite(
            self.maximum_axis_error_deg,"maximum_axis_error_deg"
        )
        if self.reference_radius_m==0.0:
            raise ValueError("reference_radius_m must be positive")
    def evaluate(self,radial_samples_m: Sequence[float],*,centerline_offset_m: float=0.0,axis_error_deg: float=0.0):
        values=[_nonnegative_finite(x,"radial_sample_m") for x in radial_samples_m]
        if not values:raise ValueError("radial_samples_m must not be empty")
        offset=float(centerline_offset_m);axis_error=float(axis_error_deg)
        if not math.isfinite(offset) or not math.isfinite(axis_error):
            raise ValueError("centerline offset and axis error must be finite")
        minimum=min(values);mean=sum(values)/len(values)
        area_fraction=min(1.0,(minimum/self.reference_radius_m)**2)
        passed=minimum>=self.minimum_accepted_radius_m and abs(offset)<=self.maximum_centerline_offset_m and abs(axis_error)<=self.maximum_axis_error_deg
        return PatencyReport(minimum,mean,area_fraction,offset,axis_error,passed)


def measure_lumen_seam_geometry(
    left_points_world: Sequence[Sequence[float]],
    right_points_world: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Measure lumen radii, coaxiality, and edge gap from live tissue nodes.

    The calculation is geometric: principal axes and seam rings come directly
    from the supplied world-space simulation nodes. It does not infer a
    successful seam from staple or collar state.
    """

    import numpy as np

    left=np.asarray(tensor_value(left_points_world),dtype=float)
    right=np.asarray(tensor_value(right_points_world),dtype=float)
    if (
        left.ndim!=2 or right.ndim!=2 or left.shape[1:]!=(3,)
        or right.shape[1:]!=(3,) or len(left)<16 or len(right)<16
    ):
        raise ValueError("left and right tissue nodes must be populated Nx3 arrays")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("tissue nodes must be finite")

    def principal_axis(points):
        centered=points-points.mean(axis=0)
        values,vectors=np.linalg.eigh(centered.T@centered/max(1,len(points)-1))
        return vectors[:,int(np.argmax(values))]

    left_axis=principal_axis(left)
    right_axis=principal_axis(right)
    if float(np.dot(left_axis,right_axis))<0.0:right_axis=-right_axis
    axis=left_axis+right_axis
    axis_norm=float(np.linalg.norm(axis))
    if axis_norm<=1.0e-12:
        axis=principal_axis(np.concatenate((left,right),axis=0))
    else:
        axis=axis/axis_norm
    center_delta=right.mean(axis=0)-left.mean(axis=0)
    if float(np.dot(axis,center_delta))<0.0:axis=-axis
    if float(np.dot(left_axis,axis))<0.0:left_axis=-left_axis
    if float(np.dot(right_axis,axis))<0.0:right_axis=-right_axis

    left_projection=left@axis
    right_projection=right@axis
    left_tolerance=max(1.0e-5,0.01*float(np.ptp(left_projection)))
    right_tolerance=max(1.0e-5,0.01*float(np.ptp(right_projection)))
    left_seam=left[left_projection>=left_projection.max()-left_tolerance]
    right_seam=right[right_projection<=right_projection.min()+right_tolerance]
    if len(left_seam)<8 or len(right_seam)<8:
        raise RuntimeError(
            "Could not resolve populated seam rings from tissue nodes: "
            f"left={len(left_seam)}, right={len(right_seam)}"
        )

    left_center=left_seam.mean(axis=0)
    right_center=right_seam.mean(axis=0)

    def inner_radius_samples(points,center):
        radial=points-center
        radial-=np.outer(radial@axis,axis)
        radii=np.linalg.norm(radial,axis=1)
        # The hollow wall contributes equal inner and outer seam rings.
        # Select the lower radial cluster without using an authored radius.
        ordered=np.sort(radii)
        return ordered[:max(4,len(ordered)//2)]

    left_radii=inner_radius_samples(left_seam,left_center)
    right_radii=inner_radius_samples(right_seam,right_center)
    radial_samples=np.concatenate((left_radii,right_radii))
    seam_delta=right_center-left_center
    axial_gap=max(0.0,float(np.dot(seam_delta,axis)))
    perpendicular=seam_delta-np.dot(seam_delta,axis)*axis
    centerline_offset=float(np.linalg.norm(perpendicular))
    cosine=float(np.clip(np.dot(left_axis,right_axis),-1.0,1.0))
    axis_error_deg=float(math.degrees(math.acos(cosine)))
    return {
        "radial_samples_m":[float(value) for value in radial_samples],
        "minimum_radius_m":float(radial_samples.min()),
        "mean_radius_m":float(radial_samples.mean()),
        "centerline_offset_m":centerline_offset,
        "axis_error_deg":axis_error_deg,
        "edge_gap_m":axial_gap,
        "axis_world":[float(value) for value in axis],
        "left_seam_node_count":int(len(left_seam)),
        "right_seam_node_count":int(len(right_seam)),
        "source":"live_world_space_simulation_nodes",
    }


@dataclass
class LeakTestLedger:
    initial_reservoir_ml: float=60.0
    reservoir_ml: float|None=None
    injected_ml: float=0.0
    chamber_ml: float=0.0
    leaked_ml: float=0.0
    collected_ml: float=0.0
    spilled_ml: float=0.0
    discarded_ml: float=0.0
    def __post_init__(self):
        self.initial_reservoir_ml=_nonnegative_finite(
            self.initial_reservoir_ml,"initial_reservoir_ml"
        )
        if self.reservoir_ml is None:
            self.reservoir_ml=self.initial_reservoir_ml
        for name in (
            "reservoir_ml","injected_ml","chamber_ml",
            "leaked_ml","collected_ml","spilled_ml","discarded_ml",
        ):
            setattr(self,name,_nonnegative_finite(getattr(self,name),name))
        if self.reservoir_ml>self.initial_reservoir_ml:
            raise ValueError("reservoir_ml cannot exceed initial_reservoir_ml")
        if self.accounted_leak_ml>self.leaked_ml+1e-12:
            raise ValueError("collected, spilled, and discarded leak exceeds leaked volume")
    def inject(self,volume_ml: float):
        v=min(_nonnegative_finite(volume_ml,"volume_ml"),self.reservoir_ml)
        self.reservoir_ml-=v;self.injected_ml+=v;self.chamber_ml+=v;return v
    def leak(self,volume_ml: float):
        v=min(_nonnegative_finite(volume_ml,"volume_ml"),self.chamber_ml)
        self.chamber_ml-=v;self.leaked_ml+=v;return v
    @property
    def accounted_leak_ml(self):
        return self.collected_ml+self.spilled_ml+self.discarded_ml
    @property
    def active_leak_ml(self):
        return max(0.0,self.leaked_ml-self.accounted_leak_ml)
    def collect(self,volume_ml: float):
        v=min(_nonnegative_finite(volume_ml,"volume_ml"),self.active_leak_ml)
        self.collected_ml+=v;return v
    def spill(self,volume_ml: float):
        v=min(_nonnegative_finite(volume_ml,"volume_ml"),self.active_leak_ml)
        self.spilled_ml+=v;return v
    def discard(self,volume_ml: float):
        v=min(_nonnegative_finite(volume_ml,"volume_ml"),self.active_leak_ml)
        self.discarded_ml+=v;return v
    @property
    def conservation_error_ml(self):
        return self.initial_reservoir_ml-(
            self.reservoir_ml+self.chamber_ml+self.active_leak_ml+
            self.collected_ml+self.spilled_ml+self.discarded_ml
        )
    def snapshot(self):
        return {
            "initial_reservoir_ml":self.initial_reservoir_ml,
            "reservoir_ml":self.reservoir_ml,
            "injected_ml":self.injected_ml,
            "chamber_ml":self.chamber_ml,
            "leaked_ml":self.leaked_ml,
            "active_leak_ml":self.active_leak_ml,
            "collected_ml":self.collected_ml,
            "spilled_ml":self.spilled_ml,
            "discarded_ml":self.discarded_ml,
            "conservation_error_ml":self.conservation_error_ml,
        }


@dataclass
class PressureDecayLeakController:
    target_pressure_pa: float=8000.0
    chamber_compliance_m3_pa: float=1.8e-11
    fluid_density_kg_m3: float=1000.0
    discharge_coefficient: float=0.62
    observation_window_s: float=8.0
    maximum_residual_leak_ml_min: float=2.0
    pressure_pa: float=0.0
    elapsed_s: float=0.0
    integrated_leak_ml: float=0.0
    peak_leak_ml_min: float=0.0
    history: list[dict[str,float]]=field(default_factory=list)
    def __post_init__(self):
        for name in (
            "target_pressure_pa","chamber_compliance_m3_pa",
            "fluid_density_kg_m3","discharge_coefficient",
            "observation_window_s","maximum_residual_leak_ml_min",
        ):
            setattr(self,name,_nonnegative_finite(getattr(self,name),name))
        if self.target_pressure_pa==0.0:
            raise ValueError("target_pressure_pa must be positive")
        if self.chamber_compliance_m3_pa==0.0:
            raise ValueError("chamber_compliance_m3_pa must be positive")
        if self.fluid_density_kg_m3==0.0:
            raise ValueError("fluid_density_kg_m3 must be positive")
        if self.observation_window_s==0.0:
            raise ValueError("observation_window_s must be positive")
    def reset(self):self.pressure_pa=0.0;self.elapsed_s=0.0;self.integrated_leak_ml=0.0;self.peak_leak_ml_min=0.0;self.history.clear()
    def begin_observation(self):
        self.elapsed_s=0.0;self.integrated_leak_ml=0.0
        self.peak_leak_ml_min=0.0;self.history.clear()
    def effective_leak_area_m2(self,*,edge_gap_m: float,retained_staple_fraction: float,collar_bond_fraction: float):
        circumference=2*math.pi*0.0108
        gap=_nonnegative_finite(edge_gap_m,"edge_gap_m")
        staple_scale=1.0-_fraction(retained_staple_fraction,"retained_staple_fraction")
        collar_scale=max(0.04,1.0-0.94*_fraction(collar_bond_fraction,"collar_bond_fraction"))
        return max(2.0e-10,circumference*gap*(0.06+0.94*staple_scale)*collar_scale)
    def update(self,dt: float,*,pump_flow_ml_s: float=0.0,edge_gap_m: float=0.0,retained_staple_fraction: float=1.0,collar_bond_fraction: float=1.0):
        dt=_nonnegative_finite(dt,"dt");area=self.effective_leak_area_m2(edge_gap_m=edge_gap_m,retained_staple_fraction=retained_staple_fraction,collar_bond_fraction=collar_bond_fraction)
        q_out_m3_s=self.discharge_coefficient*area*math.sqrt(max(0.0,2.0*self.pressure_pa/self.fluid_density_kg_m3))
        q_in_m3_s=_nonnegative_finite(pump_flow_ml_s,"pump_flow_ml_s")*1.0e-6
        next_pressure=max(0.0,self.pressure_pa+(q_in_m3_s-q_out_m3_s)*dt/self.chamber_compliance_m3_pa)
        if q_in_m3_s>0.0:
            next_pressure=min(self.target_pressure_pa,next_pressure)
        self.pressure_pa=next_pressure
        leak_ml_min=q_out_m3_s*1.0e6*60.0;leak_ml=q_out_m3_s*1.0e6*dt
        self.elapsed_s+=dt;self.integrated_leak_ml+=leak_ml;self.peak_leak_ml_min=max(self.peak_leak_ml_min,leak_ml_min)
        sample={"time_s":self.elapsed_s,"pressure_pa":self.pressure_pa,"leak_ml_min":leak_ml_min,"effective_leak_area_m2":area};self.history.append(sample);return sample
    @property
    def average_leak_ml_min(self):return 0.0 if self.elapsed_s<=0 else self.integrated_leak_ml*60.0/self.elapsed_s
    @property
    def complete(self):return self.elapsed_s>=self.observation_window_s
    @property
    def passed(self):return self.complete and self.average_leak_ml_min<=self.maximum_residual_leak_ml_min and self.pressure_pa>=0.70*self.target_pressure_pa


def ensure_leak_particle_system(*,physics_scene_path="/physicsScene",root_path="/World/DrAnmarLeakTest",system_path=None,particles_path=None,material_path=None,stage=None):
    stage=_current_stage(stage)
    from omni.physx.scripts import particleUtils,physicsUtils
    from pxr import Sdf,UsdGeom,UsdPhysics
    stage.DefinePrim(root_path,"Scope")
    system_path=system_path or f"{root_path}/ParticleSystem"
    particles_path=particles_path or f"{root_path}/Particles"
    material_path=material_path or f"{root_path}/PBDMaterial"
    scene_path=Sdf.Path(physics_scene_path)
    if not stage.GetPrimAtPath(scene_path).IsValid():UsdPhysics.Scene.Define(stage,scene_path)
    if not stage.GetPrimAtPath(material_path).IsValid():
        particleUtils.add_pbd_particle_material(
            stage,Sdf.Path(material_path),friction=0.08,viscosity=0.0035,
            cohesion=0.01,surface_tension=0.02
        )
    if not stage.GetPrimAtPath(system_path).IsValid():
        particleUtils.add_physx_particle_system(
            stage=stage,particle_system_path=Sdf.Path(system_path),
            simulation_owner=scene_path,
            particle_contact_offset=LEAK_PARTICLE_RADIUS_M*1.15,
            rest_offset=LEAK_PARTICLE_RADIUS_M*0.9,
            solid_rest_offset=LEAK_PARTICLE_RADIUS_M*1.8,
            fluid_rest_offset=LEAK_PARTICLE_RADIUS_M*0.92,
        )
    physicsUtils.add_physics_material_to_prim(
        stage,stage.GetPrimAtPath(system_path),Sdf.Path(material_path)
    )
    if not stage.GetPrimAtPath(particles_path).IsValid():
        particleUtils.add_physx_particleset_points(
            stage,Sdf.Path(particles_path),[],[],[],Sdf.Path(system_path),
            True,True,0,1.0,LEAK_PARTICLE_RADIUS_M*2.0
        )
        UsdGeom.Points(stage.GetPrimAtPath(particles_path)).GetWidthsAttr().Set([])
    return {
        "root_path":root_path,
        "particle_system_path":system_path,
        "particles_path":particles_path,
        "material_path":material_path,
    }


def emit_leak_particles(positions: Sequence[Sequence[float]],velocities: Sequence[Sequence[float]],*,particle_volume_ml=LEAK_PARTICLE_VOLUME_ML,system_path="/World/DrAnmarLeakTest/ParticleSystem",particles_path="/World/DrAnmarLeakTest/Particles",ledger: LeakTestLedger|None=None,stage=None):
    stage=_current_stage(stage)
    ensure_leak_particle_system(
        system_path=system_path,particles_path=particles_path,stage=stage
    )
    from pxr import Gf,UsdGeom
    particle_volume_ml=_nonnegative_finite(particle_volume_ml,"particle_volume_ml")
    if particle_volume_ml==0.0:
        raise ValueError("particle_volume_ml must be positive")
    positions=list(positions);velocities=list(velocities)
    if len(positions)!=len(velocities):
        raise ValueError("positions and velocities must have equal lengths")
    converted_positions=[];converted_velocities=[]
    for position,velocity in zip(positions,velocities):
        if len(position)!=3 or len(velocity)!=3:
            raise ValueError("particle vectors must be length three")
        values=[float(value) for value in (*position,*velocity)]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("particle vectors must be finite")
        converted_positions.append(Gf.Vec3f(*values[:3]))
        converted_velocities.append(Gf.Vec3f(*values[3:]))
    allowed=len(converted_positions);remainder=0.0
    if ledger is not None:
        debited=ledger.leak(allowed*particle_volume_ml)
        allowed=int(debited/particle_volume_ml)
        actual=allowed*particle_volume_ml
        remainder=debited-actual
        ledger.chamber_ml+=remainder
        ledger.leaked_ml-=remainder
    converted_positions=converted_positions[:allowed]
    converted_velocities=converted_velocities[:allowed]
    if not converted_positions:
        return {
            "particle_count":0,"emitted_ml":0.0,
            "quantization_remainder_ml":remainder,
        }
    points=UsdGeom.Points(stage.GetPrimAtPath(particles_path))
    current_positions=list(points.GetPointsAttr().Get() or [])
    current_velocities=list(points.GetVelocitiesAttr().Get() or [])
    current_widths=list(points.GetWidthsAttr().Get() or [])
    current_positions.extend(converted_positions)
    current_velocities.extend(converted_velocities)
    current_widths.extend([LEAK_PARTICLE_RADIUS_M*2.0]*allowed)
    points.GetPointsAttr().Set(current_positions)
    points.GetVelocitiesAttr().Set(current_velocities)
    points.GetWidthsAttr().Set(current_widths)
    return {
        "particle_count":allowed,
        "emitted_ml":allowed*particle_volume_ml,
        "quantization_remainder_ml":remainder,
        "total_particle_count":len(current_positions),
    }


PHASE_TARGETS={
    "inspect":{"left_approximation_joint":0.0,"right_approximation_joint":0.0,"left_capture_joint":0.0,"right_capture_joint":0.0,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":-0.055,"mandrel_expansion_joint":0.0,"staple_driver_joint":0.0,"collar_carousel_joint":0.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
    "capture":{"left_approximation_joint":0.004,"right_approximation_joint":-0.004,"left_capture_joint":0.008,"right_capture_joint":-0.008,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":-0.025,"mandrel_expansion_joint":0.0,"staple_driver_joint":0.0,"collar_carousel_joint":0.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
    "align":{"left_approximation_joint":0.012,"right_approximation_joint":-0.012,"left_capture_joint":0.008,"right_capture_joint":-0.008,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":0.0,"mandrel_expansion_joint":0.004,"staple_driver_joint":0.0,"collar_carousel_joint":0.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
    "mandrel":{"left_approximation_joint":0.012,"right_approximation_joint":-0.012,"left_capture_joint":0.008,"right_capture_joint":-0.008,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":0.0,"mandrel_expansion_joint":0.006,"staple_driver_joint":0.0,"collar_carousel_joint":0.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
    "approximate":{"left_approximation_joint":0.030,"right_approximation_joint":-0.030,"left_capture_joint":0.008,"right_capture_joint":-0.008,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":0.0,"mandrel_expansion_joint":0.004,"staple_driver_joint":0.0,"collar_carousel_joint":0.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
    "evert":{"left_approximation_joint":0.030,"right_approximation_joint":-0.030,"left_capture_joint":0.008,"right_capture_joint":-0.008,"left_eversion_joint":0.006,"right_eversion_joint":-0.006,"mandrel_extension_joint":0.0,"mandrel_expansion_joint":0.004,"staple_driver_joint":0.0,"collar_carousel_joint":0.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
    "staple":{"left_approximation_joint":0.030,"right_approximation_joint":-0.030,"left_capture_joint":0.008,"right_capture_joint":-0.008,"left_eversion_joint":0.006,"right_eversion_joint":-0.006,"mandrel_extension_joint":0.0,"mandrel_expansion_joint":0.004,"staple_driver_joint":-0.027,"collar_carousel_joint":0.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
    "release_capture":{"left_approximation_joint":0.014,"right_approximation_joint":-0.014,"left_capture_joint":0.0,"right_capture_joint":0.0,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":0.0,"mandrel_expansion_joint":0.002,"staple_driver_joint":0.0,"collar_carousel_joint":0.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
    "reinforce":{"left_approximation_joint":0.014,"right_approximation_joint":-0.014,"left_capture_joint":0.0,"right_capture_joint":0.0,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":0.0,"mandrel_expansion_joint":0.002,"staple_driver_joint":0.0,"collar_carousel_joint":120.0,"collar_applicator_joint":0.046,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
    "occlude":{"left_approximation_joint":0.0,"right_approximation_joint":0.0,"left_capture_joint":0.0,"right_capture_joint":0.0,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":0.0,"mandrel_expansion_joint":0.002,"staple_driver_joint":0.0,"collar_carousel_joint":120.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.008,"right_occluder_valve_joint":0.008,"pressure_valve_joint":0.0},
    "pressurize":{"left_approximation_joint":0.0,"right_approximation_joint":0.0,"left_capture_joint":0.0,"right_capture_joint":0.0,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":0.0,"mandrel_expansion_joint":0.002,"staple_driver_joint":0.0,"collar_carousel_joint":120.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.008,"right_occluder_valve_joint":0.008,"pressure_valve_joint":0.008},
    "verify":{"left_approximation_joint":0.0,"right_approximation_joint":0.0,"left_capture_joint":0.0,"right_capture_joint":0.0,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":0.0,"mandrel_expansion_joint":0.002,"staple_driver_joint":0.0,"collar_carousel_joint":120.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.008,"right_occluder_valve_joint":0.008,"pressure_valve_joint":0.002},
    "complete":{"left_approximation_joint":0.0,"right_approximation_joint":0.0,"left_capture_joint":0.0,"right_capture_joint":0.0,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":-0.055,"mandrel_expansion_joint":0.0,"staple_driver_joint":0.0,"collar_carousel_joint":0.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
    "abort":{"left_approximation_joint":0.0,"right_approximation_joint":0.0,"left_capture_joint":0.0,"right_capture_joint":0.0,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":-0.055,"mandrel_expansion_joint":0.0,"staple_driver_joint":0.0,"collar_carousel_joint":0.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
}


def phase_targets(phase: str):
    try:return dict(PHASE_TARGETS[phase])
    except KeyError as exc:raise KeyError(f"Unknown anastomosis phase {phase!r}") from exc


@dataclass
class AdaptiveAnastomosisSequenceController:
    phase: str="inspect"
    capture: BilateralTissueCaptureController|None=None
    staples: StapleRingRetentionController=field(default_factory=StapleRingRetentionController)
    collar: ReinforcementCollarBondController=field(default_factory=ReinforcementCollarBondController)
    patency: LumenPatencyController=field(default_factory=LumenPatencyController)
    leak_test: PressureDecayLeakController=field(default_factory=PressureDecayLeakController)
    history: list[str]=field(default_factory=list)
    def __post_init__(self):
        phase_targets(self.phase)
    def transition(self,phase: str):
        targets=phase_targets(phase);self.phase=phase;self.history.append(phase)
        if phase=="pressurize":self.leak_test.reset()
        elif phase=="verify":self.leak_test.begin_observation()
        return targets
