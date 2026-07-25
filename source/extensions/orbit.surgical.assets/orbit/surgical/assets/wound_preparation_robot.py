# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Isaac Lab integration for the DrAnmar wound-preparation robot.

The payload replaces the Panda hand at ``panda_link8`` and provides a concentric
irrigation / aspiration head, compliant contact guard, rotary debridement
cartridge, multimodal sensor frames, a particle-volume ledger, and task-level
wound-preparation controllers.

All numerical values are provisional engineering seeds. This package is not a
medical device and is not clinically validated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import math
import random

CATALOG_SUBPATH = "Props/SurgicalPreparation/WoundPreparationRobot"
ASSET_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
ROOT = ASSET_DATA_ROOT / CATALOG_SUBPATH
TOOL_PAYLOAD_USD = ROOT / "dranmar_wound_preparation_tool_payload.usda"
TOOL_STANDALONE_USD = ROOT / "dranmar_wound_preparation_tool_standalone.usda"
TOOL_RIGID_PROXY_USD = ROOT / "dranmar_wound_preparation_tool_rigid_proxy.usda"
DROPLET_USD = ROOT / "dranmar_irrigation_droplet.usda"
DEBRIS_USD = ROOT / "dranmar_debridement_fragment.usda"
WOUND_BED_USD = ROOT / "dranmar_wound_bed_demo.usda"
BRUSH_CARTRIDGE_USD = ROOT / "dranmar_debridement_brush_cartridge.usda"
CURETTE_CARTRIDGE_USD = ROOT / "dranmar_debridement_curette_cartridge.usda"
PAD_CARTRIDGE_USD = ROOT / "dranmar_debridement_pad_cartridge.usda"

VALID_IRRIGATION_STATES = frozenset({"loaded", "low", "empty"})
VALID_COLLECTION_STATES = frozenset({"empty", "partial", "full"})
IRRIGATION_NOZZLE_COUNT = 10
PARTICLE_RADIUS_M = 0.00065
PARTICLE_VOLUME_ML = 4.0 / 3.0 * math.pi * PARTICLE_RADIUS_M ** 3 * 1.0e6

TOOL_JOINTS = {
    "contact_guard": "contact_guard_joint",
    "debridement_extension": "debridement_extension_joint",
    "debridement_rotor": "debridement_rotor_joint",
    "irrigation_valve": "irrigation_valve_joint",
    "suction_valve": "suction_valve_joint",
}

TOOL_FRAME_PATHS = {
    "panda_link8_mount": "Links/Mount/Frames/panda_link8_mount",
    "wound_preparation_tcp": "Links/Mount/Frames/wound_preparation_tcp",
    "contact_guard_center": "Links/ContactGuard/Frames/contact_guard_center",
    "debridement_contact": "Links/DebridementRotor/Frames/debridement_contact",
    "rotor_axis": "Links/DebridementRotor/Frames/rotor_axis",
    "irrigation_jet_origin": "Links/Mount/Frames/irrigation_jet_origin",
    "suction_capture_center": "Links/Mount/Frames/suction_capture_center",
    "suction_throat": "Links/Mount/Frames/suction_throat",
    "camera_left": "Links/Mount/Frames/camera_left",
    "camera_right": "Links/Mount/Frames/camera_right",
    "depth_camera": "Links/Mount/Frames/depth_camera",
    "fluorescence_camera": "Links/Mount/Frames/fluorescence_camera",
    "illumination_ring": "Links/Mount/Frames/illumination_ring",
    "irrigation_reservoir_port": "Links/IrrigationReservoir/Frames/irrigation_reservoir_port",
    "waste_canister_port": "Links/WasteCanister/Frames/waste_canister_port",
    "cartridge_mount": "Links/DebridementRotor/Frames/cartridge_mount",
    "service_reference": "Links/Mount/Frames/service_reference",
    "count_reference": "Links/Mount/Frames/count_reference",
}
REGISTERED_CAMERA_FRAMES = (
    "camera_left", "camera_right", "depth_camera", "fluorescence_camera",
)


def frame_path(tool_path: str, name: str) -> str:
    try:
        suffix = TOOL_FRAME_PATHS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown wound-preparation frame {name!r}") from exc
    return f"{tool_path.rstrip('/')}/{suffix}"


def tensor_value(value: Any):
    """Return the underlying torch tensor for Isaac 6 proxy tensors."""
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


def make_tool_cfg(
    prim_path: str = "/World/DrAnmarWoundPreparationTool",
    *,
    irrigation_state: str = "loaded",
    collection_state: str = "empty",
    position=(0.0, 0.0, 0.35),
    orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
):
    """Return the standalone wound-preparation tool articulation."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    _check(irrigation_state, VALID_IRRIGATION_STATES, "irrigation_state")
    _check(collection_state, VALID_COLLECTION_STATES, "collection_state")
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(TOOL_STANDALONE_USD),
            variants={"irrigation_state": irrigation_state, "collection_state": collection_state},
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=position,
            rot=_xyzw_from_wxyz(orientation_wxyz),
            joint_pos={
                "contact_guard_joint": 0.0,
                "debridement_extension_joint": 0.0,
                "debridement_rotor_joint": 0.0,
                "irrigation_valve_joint": 0.0,
                "suction_valve_joint": 0.0,
            },
        ),
        actuators={
            "contact_guard": ImplicitActuatorCfg(
                joint_names_expr=["contact_guard_joint"], effort_limit_sim=32.0,
                velocity_limit_sim=0.08, stiffness=1100.0, damping=30.0,
            ),
            "debridement_extension": ImplicitActuatorCfg(
                joint_names_expr=["debridement_extension_joint"], effort_limit_sim=45.0,
                velocity_limit_sim=0.10, stiffness=4200.0, damping=115.0,
            ),
            "debridement_rotor": ImplicitActuatorCfg(
                joint_names_expr=["debridement_rotor_joint"], effort_limit_sim=0.32,
                velocity_limit_sim=90.0, stiffness=0.0, damping=0.018,
            ),
            "irrigation_valve": ImplicitActuatorCfg(
                joint_names_expr=["irrigation_valve_joint"], effort_limit_sim=25.0,
                velocity_limit_sim=2.5, stiffness=1800.0, damping=45.0,
            ),
            "suction_valve": ImplicitActuatorCfg(
                joint_names_expr=["suction_valve_joint"], effort_limit_sim=2.0,
                velocity_limit_sim=2.5, stiffness=8.0, damping=0.45,
            ),
        },
    )


def make_rigid_proxy_cfg(
    prim_path: str = "/World/DrAnmarWoundPreparationToolProxy",
    *, position=(0.0, 0.0, 0.35), orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(usd_path=str(TOOL_RIGID_PROXY_USD), activate_contact_sensors=True),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=position, rot=_xyzw_from_wxyz(orientation_wxyz)
        ),
    )


def _spawn_single_franka_with_wound_preparation_tool(prim_path: str, cfg: Any, translation=None, orientation=None, **kwargs):
    """Spawn Franka, deactivate the Panda hand, and mount the DrAnmar payload."""
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    from isaaclab.sim.utils import create_prim, get_current_stage, select_usd_variants
    from pxr import Gf, Sdf, UsdPhysics

    robot = spawn_from_usd(prim_path, cfg, translation, orientation)
    stage = get_current_stage()
    names_to_disable = {
        "panda_hand_joint", "panda_hand", "panda_finger_joint1", "panda_finger_joint2",
        "panda_leftfinger", "panda_rightfinger",
    }
    robot_path = Sdf.Path(prim_path)
    hand_joint_prims = [
        prim
        for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(robot_path) and prim.GetName() == "panda_hand_joint"
    ]
    if len(hand_joint_prims) == 1:
        stock_hand_joint = UsdPhysics.Joint(hand_joint_prims[0])
        mount_body_paths = stock_hand_joint.GetBody0Rel().GetTargets()
        mount_local_pos0 = stock_hand_joint.GetLocalPos0Attr().Get() or Gf.Vec3f(0, 0, 0)
        mount_local_rot0 = stock_hand_joint.GetLocalRot0Attr().Get() or Gf.Quatf(1, 0, 0, 0)
    else:
        link8_paths = [
            prim.GetPath()
            for prim in stage.Traverse()
            if prim.GetPath().HasPrefix(robot_path) and prim.GetName() == "panda_link8"
        ]
        if len(link8_paths) != 1:
            raise RuntimeError(
                "Could not resolve the Franka hand mount from panda_hand_joint or panda_link8"
            )
        mount_body_paths = link8_paths
        mount_local_pos0 = Gf.Vec3f(0, 0, 0)
        half_angle = math.radians(-45.0) / 2.0
        mount_local_rot0 = Gf.Quatf(
            math.cos(half_angle), 0, 0, math.sin(half_angle)
        )
    if len(mount_body_paths) != 1 or not stage.GetPrimAtPath(mount_body_paths[0]).IsValid():
        raise RuntimeError(f"Invalid Franka hand mount target: {mount_body_paths}")

    candidate_paths = [
        prim.GetPath()
        for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(robot_path) and prim.GetName() in names_to_disable
    ]
    paths_to_disable = []
    for path in sorted(candidate_paths, key=lambda item: str(item).count("/")):
        if not any(path.HasPrefix(parent) for parent in paths_to_disable):
            paths_to_disable.append(path)
    for path in paths_to_disable:
        stage.OverridePrim(path).SetActive(False)

    tool_path = f"{prim_path}/DrAnmarWoundPreparationTool"
    create_prim(tool_path, usd_path=str(TOOL_PAYLOAD_USD), stage=stage)
    select_usd_variants(
        tool_path,
        {"irrigation_state": cfg.irrigation_state, "collection_state": cfg.collection_state},
    )

    joint = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/dranmar_wound_preparation_mount_joint")
    joint.CreateBody0Rel().SetTargets(mount_body_paths)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{tool_path}/Links/Mount")])
    joint.CreateLocalPos0Attr().Set(mount_local_pos0)
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot0Attr().Set(mount_local_rot0)
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    return robot


def spawn_franka_with_wound_preparation_tool(prim_path: str, cfg: Any, translation=None, orientation=None, **kwargs):
    from isaaclab.sim.utils import clone
    return clone(_spawn_single_franka_with_wound_preparation_tool)(
        prim_path, cfg, translation=translation, orientation=orientation, **kwargs
    )


def make_franka_wound_preparation_robot_cfg(
    *, prim_path: str = "/World/Robot", irrigation_state: str = "loaded", collection_state: str = "empty"
):
    """Return the Isaac Lab Franka with its stock hand replaced by this tool."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

    _check(irrigation_state, VALID_IRRIGATION_STATES, "irrigation_state")
    _check(collection_state, VALID_COLLECTION_STATES, "collection_state")

    @configclass
    class FrankaWoundPreparationUsdCfg(sim_utils.UsdFileCfg):
        irrigation_state: str = "loaded"
        collection_state: str = "empty"
        func = spawn_franka_with_wound_preparation_tool

    cfg = FRANKA_PANDA_CFG.copy()
    cfg.prim_path = prim_path
    cfg.spawn = FrankaWoundPreparationUsdCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        variants={"Gripper": "Default", "Mesh": "Performance"},
        irrigation_state=irrigation_state,
        collection_state=collection_state,
        activate_contact_sensors=True,
        rigid_props=FRANKA_PANDA_CFG.spawn.rigid_props,
        articulation_props=FRANKA_PANDA_CFG.spawn.articulation_props,
    )
    cfg.init_state.joint_pos = {key: value for key, value in cfg.init_state.joint_pos.items() if "finger" not in key}
    cfg.init_state.joint_pos.update({
        "contact_guard_joint": 0.0,
        "debridement_extension_joint": 0.0,
        "debridement_rotor_joint": 0.0,
        "irrigation_valve_joint": 0.0,
        "suction_valve_joint": 0.0,
    })
    cfg.actuators.pop("panda_hand", None)
    cfg.actuators.update({
        "wound_prep_guard": ImplicitActuatorCfg(
            joint_names_expr=["contact_guard_joint"], effort_limit_sim=32.0,
            velocity_limit_sim=0.08, stiffness=1100.0, damping=30.0,
        ),
        "wound_prep_extension": ImplicitActuatorCfg(
            joint_names_expr=["debridement_extension_joint"], effort_limit_sim=45.0,
            velocity_limit_sim=0.10, stiffness=4200.0, damping=115.0,
        ),
        "wound_prep_rotor": ImplicitActuatorCfg(
            joint_names_expr=["debridement_rotor_joint"], effort_limit_sim=0.32,
            velocity_limit_sim=90.0, stiffness=0.0, damping=0.018,
        ),
        "wound_prep_valves": ImplicitActuatorCfg(
            joint_names_expr=[".*_valve_joint"], effort_limit_sim=25.0,
            velocity_limit_sim=2.5, stiffness=1800.0, damping=45.0,
        ),
    })
    return cfg


def spawn_wound_bed_demo(
    prim_path: str = "/World/DrAnmarWoundBed",
    *, translation=(0.0, 0.0, 0.0), orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
):
    import isaaclab.sim as sim_utils
    cfg = sim_utils.UsdFileCfg(usd_path=str(WOUND_BED_USD), activate_contact_sensors=True)
    return cfg.func(
        prim_path, cfg, translation=translation,
        orientation=_xyzw_from_wxyz(orientation_wxyz),
    )


def _current_stage(stage=None):
    if stage is not None:
        return stage
    import omni.usd
    return omni.usd.get_context().get_stage()


def apply_wound_surface_deformable(
    wound_root_path: str = "/World/DrAnmarWoundBed",
    *, stage=None, material_path: str = "/World/Materials/DrAnmarWoundSurface",
    youngs_modulus_pa: float = 55_000.0, poissons_ratio: float = 0.45,
    surface_thickness_m: float = 0.007, density_kg_m3: float = 1_050.0,
    dynamic_friction: float = 0.58, elasticity_damping: float = 0.16,
    bend_damping: float = 0.14, self_collision: bool = True,
) -> dict[str, Any]:
    """Cook the portable wound mesh through the current surface-deformable route."""
    stage = _current_stage(stage)
    from omni.physx.scripts import deformableUtils
    from pxr import UsdShade

    mesh_path = f"{wound_root_path.rstrip('/')}/TissueSurface/SimulationMesh"
    mesh_prim = stage.GetPrimAtPath(mesh_path)
    if not mesh_prim or not mesh_prim.IsValid():
        raise ValueError(f"No wound simulation mesh at {mesh_path}")

    material = UsdShade.Material.Define(stage, material_path)
    prim = material.GetPrim()
    prim.ApplyAPI("OmniPhysicsBaseMaterialAPI")
    prim.GetAttribute("omniphysics:dynamicFriction").Set(float(dynamic_friction))
    prim.GetAttribute("omniphysics:density").Set(float(density_kg_m3))
    prim.ApplyAPI("OmniPhysicsDeformableMaterialAPI")
    prim.GetAttribute("omniphysics:youngsModulus").Set(float(youngs_modulus_pa))
    prim.GetAttribute("omniphysics:poissonsRatio").Set(float(poissons_ratio))
    prim.ApplyAPI("OmniPhysicsSurfaceDeformableMaterialAPI")
    prim.GetAttribute("omniphysics:surfaceThickness").Set(float(surface_thickness_m))
    prim.GetAttribute("omniphysics:surfaceBendStiffness").Set(0.0)
    prim.ApplyAPI("PhysxSurfaceDeformableMaterialAPI")
    prim.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(float(elasticity_damping))
    prim.GetAttribute("physxDeformableMaterial:bendDamping").Set(float(bend_damping))

    success = deformableUtils.set_physics_surface_deformable_body(stage, mesh_prim.GetPath())
    if success is False:
        raise RuntimeError(f"PhysX could not create a surface deformable at {mesh_path}")
    mesh_prim.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
    if mesh_prim.HasAPI("PhysxSurfaceDeformableBodyAPI"):
        mesh_prim.GetAttribute("physxDeformableBody:selfCollision").Set(bool(self_collision))
    binding = UsdShade.MaterialBindingAPI.Apply(mesh_prim)
    binding.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
    return {
        "mesh_path": mesh_path, "material_path": material_path,
        "parameters": {
            "youngs_modulus_pa": youngs_modulus_pa, "poissons_ratio": poissons_ratio,
            "surface_thickness_m": surface_thickness_m, "density_kg_m3": density_kg_m3,
            "dynamic_friction": dynamic_friction, "elasticity_damping": elasticity_damping,
            "bend_damping": bend_damping, "self_collision": self_collision,
            "status": "provisional_engineering_seed",
        },
    }


def create_deformable_attachment(
    deformable_prim_path: str, rigid_prim_path: str, attachment_path: str, *, stage=None
) -> str:
    """Create an overlap-generated rigid/deformable attachment across Isaac generations."""
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt
    stage = _current_stage(stage)
    if stage.GetPrimAtPath(attachment_path).IsValid():
        stage.RemovePrim(attachment_path)

    # Isaac Sim 6 replaced the command-authored PhysxPhysicsAttachment with
    # explicit OmniPhysics vertex attachments. Author the current schema
    # directly so headless runtimes do not depend on an optional UI command.
    prim_definition = Usd.SchemaRegistry().FindConcretePrimDefinition(
        "OmniPhysicsVtxXformAttachment"
    )
    if prim_definition:
        deformable_prim = stage.GetPrimAtPath(deformable_prim_path)
        rigid_prim = stage.GetPrimAtPath(rigid_prim_path)
        mesh = UsdGeom.Mesh(deformable_prim)
        points = list(mesh.GetPointsAttr().Get() or [])
        if not deformable_prim.IsValid() or not mesh or not points:
            raise ValueError(f"Attachment source is not a populated mesh: {deformable_prim_path}")
        if not rigid_prim.IsValid() or not UsdGeom.Xformable(rigid_prim):
            raise ValueError(f"Attachment target is not xformable: {rigid_prim_path}")

        mesh_to_world = UsdGeom.Xformable(deformable_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        rigid_to_world = UsdGeom.Xformable(rigid_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        world_to_rigid = rigid_to_world.GetInverse()
        bounds = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide],
        ).ComputeWorldBound(rigid_prim).ComputeAlignedRange()
        minimum, maximum = bounds.GetMin(), bounds.GetMax()
        center = (minimum + maximum) * 0.5
        margin = 0.0025
        ranked: list[tuple[float, int, Gf.Vec3d, bool]] = []
        for index, point in enumerate(points):
            world = mesh_to_world.Transform(Gf.Vec3d(point))
            delta = world - center
            distance_sq = float(Gf.Dot(delta, delta))
            overlaps = all(
                minimum[axis] - margin <= world[axis] <= maximum[axis] + margin
                for axis in range(3)
            )
            ranked.append((distance_sq, index, world, overlaps))
        ranked.sort(key=lambda item: item[0])
        selected = [item for item in ranked if item[3]][:12]
        if len(selected) < 4:
            raise RuntimeError(
                f"Attachment capture volume does not overlap enough deformable "
                f"vertices for {attachment_path}: source={deformable_prim_path}, "
                f"target={rigid_prim_path}, overlapping={len(selected)}, "
                "required=4, overlap_margin_m=0.0025"
            )

        attachment = stage.DefinePrim(attachment_path, "OmniPhysicsVtxXformAttachment")
        attachment.CreateRelationship("omniphysics:src0").SetTargets(
            [Sdf.Path(deformable_prim_path)]
        )
        attachment.CreateRelationship("omniphysics:src1").SetTargets(
            [Sdf.Path(rigid_prim_path)]
        )
        attachment.CreateAttribute(
            "omniphysics:vtxIndicesSrc0", Sdf.ValueTypeNames.IntArray
        ).Set(Vt.IntArray([item[1] for item in selected]))
        attachment.CreateAttribute(
            "omniphysics:localPositionsSrc1", Sdf.ValueTypeNames.Point3fArray
        ).Set(
            Vt.Vec3fArray(
                [Gf.Vec3f(world_to_rigid.Transform(item[2])) for item in selected]
            )
        )
        attachment.CreateAttribute(
            "omniphysics:attachmentEnabled", Sdf.ValueTypeNames.Bool
        ).Set(True)
        if (
            not attachment.IsValid()
            or attachment.GetTypeName() != "OmniPhysicsVtxXformAttachment"
            or not attachment.GetRelationship("omniphysics:src0").GetTargets()
            or not attachment.GetRelationship("omniphysics:src1").GetTargets()
        ):
            raise RuntimeError(f"Could not author current attachment schema at {attachment_path}")
        return "OmniPhysicsVtxXformAttachment"

    import omni.kit.commands

    def execute_and_verify(command: str, **kwargs) -> str:
        omni.kit.commands.execute(command, **kwargs)
        attachment = stage.GetPrimAtPath(attachment_path)
        if not attachment.IsValid():
            raise RuntimeError(f"{command} did not author {attachment_path}")
        return command

    try:
        return execute_and_verify(
            "CreateAutoDeformableAttachment",
            target_attachment_path=Sdf.Path(attachment_path),
            attachable0_path=Sdf.Path(deformable_prim_path),
            attachable1_path=Sdf.Path(rigid_prim_path),
        )
    except Exception as current_error:
        if stage.GetPrimAtPath(attachment_path).IsValid():
            stage.RemovePrim(attachment_path)
        try:
            return execute_and_verify(
                "CreatePhysicsAttachment",
                target_attachment_path=Sdf.Path(attachment_path),
                actor0_path=Sdf.Path(deformable_prim_path),
                actor1_path=Sdf.Path(rigid_prim_path),
            )
        except Exception as legacy_error:
            raise RuntimeError(
                f"Could not create attachment {attachment_path}: current={current_error!r}; legacy={legacy_error!r}"
            ) from legacy_error


def attach_demo_debris(
    wound_root_path: str = "/World/DrAnmarWoundBed", *, stage=None
) -> dict[str, str]:
    """Attach every demo fragment to the wound mesh until debridement releases it."""
    stage = _current_stage(stage)
    tissue_path = f"{wound_root_path.rstrip('/')}/TissueSurface/SimulationMesh"
    attachments_root = f"{wound_root_path.rstrip('/')}/RuntimeDebrisAttachments"
    stage.DefinePrim(attachments_root, "Scope")
    result: dict[str, str] = {}
    debris_scope = stage.GetPrimAtPath(f"{wound_root_path.rstrip('/')}/Debris")
    if not debris_scope or not debris_scope.IsValid():
        raise ValueError("Demo debris scope is missing")
    for fragment in debris_scope.GetChildren():
        collider = f"{fragment.GetPath()}/Collisions/AdhesionPatch"
        attachment = f"{attachments_root}/{fragment.GetName()}"
        create_deformable_attachment(tissue_path, collider, attachment, stage=stage)
        result[str(fragment.GetPath())] = attachment
    return result


def _world_transform(stage, prim_path: str):
    from pxr import Usd, UsdGeom
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"Invalid frame prim {prim_path}")
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def _nonnegative_finite(value: float, label: str) -> float:
    amount = float(value)
    if not math.isfinite(amount) or amount < 0.0:
        raise ValueError(f"{label} must be a finite non-negative value")
    return amount


@dataclass
class FluidLedger:
    """Conservative volume bookkeeping around the particle approximation."""
    reservoir_capacity_ml: float = 45.0
    reservoir_ml: float = 45.0
    collection_capacity_ml: float = 55.0
    emitted_ml: float = 0.0
    aspirated_ml: float = 0.0
    spilled_ml: float = 0.0
    discarded_ml: float = 0.0
    active_particle_ml: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "reservoir_capacity_ml", "reservoir_ml", "collection_capacity_ml",
            "emitted_ml", "aspirated_ml", "spilled_ml", "discarded_ml",
            "active_particle_ml",
        ):
            setattr(self, name, _nonnegative_finite(getattr(self, name), name))
        if self.reservoir_ml > self.reservoir_capacity_ml:
            raise ValueError("reservoir_ml cannot exceed reservoir_capacity_ml")
        if self.aspirated_ml > self.collection_capacity_ml:
            raise ValueError("aspirated_ml cannot exceed collection_capacity_ml")

    def emit(self, requested_ml: float) -> float:
        amount = min(_nonnegative_finite(requested_ml, "requested_ml"), self.reservoir_ml)
        self.reservoir_ml -= amount
        self.emitted_ml += amount
        self.active_particle_ml += amount
        return amount

    def aspirate(self, amount_ml: float) -> float:
        amount = min(
            _nonnegative_finite(amount_ml, "amount_ml"),
            self.active_particle_ml,
            self.collection_remaining_ml,
        )
        self.active_particle_ml -= amount
        self.aspirated_ml += amount
        return amount

    def mark_spilled(self, amount_ml: float) -> float:
        amount = min(_nonnegative_finite(amount_ml, "amount_ml"), self.active_particle_ml)
        self.active_particle_ml -= amount
        self.spilled_ml += amount
        return amount

    def discard(self, amount_ml: float) -> float:
        """Account for particles intentionally culled for numerical maintenance."""
        amount = min(_nonnegative_finite(amount_ml, "amount_ml"), self.active_particle_ml)
        self.active_particle_ml -= amount
        self.discarded_ml += amount
        return amount

    @property
    def collection_remaining_ml(self) -> float:
        return max(0.0, self.collection_capacity_ml - self.aspirated_ml)

    @property
    def balance_error_ml(self) -> float:
        accounted = self.reservoir_ml + self.active_particle_ml + self.aspirated_ml + self.spilled_ml + self.discarded_ml
        return self.reservoir_capacity_ml - accounted

    def snapshot(self) -> dict[str, float]:
        return {
            "reservoir_capacity_ml": self.reservoir_capacity_ml,
            "reservoir_ml": self.reservoir_ml,
            "emitted_ml": self.emitted_ml,
            "active_particle_ml": self.active_particle_ml,
            "aspirated_ml": self.aspirated_ml,
            "spilled_ml": self.spilled_ml,
            "discarded_ml": self.discarded_ml,
            "collection_capacity_ml": self.collection_capacity_ml,
            "collection_remaining_ml": self.collection_remaining_ml,
            "balance_error_ml": self.balance_error_ml,
        }


def ensure_irrigation_particle_system(
    *, stage=None, physics_scene_path: str = "/physicsScene",
    root_path: str = "/World/DrAnmarIrrigationParticles",
    particle_system_path: str | None = None, particle_set_path: str | None = None,
    particle_radius_m: float = PARTICLE_RADIUS_M,
) -> dict[str, str]:
    """Create a PhysX PBD liquid system and an initially empty particle set."""
    stage = _current_stage(stage)
    from omni.physx.scripts import particleUtils, physicsUtils
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    stage.DefinePrim(root_path, "Scope")
    if not stage.GetPrimAtPath(physics_scene_path).IsValid():
        UsdPhysics.Scene.Define(stage, physics_scene_path)
    particle_system_path = particle_system_path or f"{root_path}/ParticleSystem"
    particle_set_path = particle_set_path or f"{root_path}/Particles"
    material_path = f"{root_path}/PBDMaterial"

    if not stage.GetPrimAtPath(material_path).IsValid():
        particleUtils.add_pbd_particle_material(
            stage, Sdf.Path(material_path), cohesion=0.002, viscosity=0.002,
            surface_tension=0.004, friction=0.05,
        )
    if not stage.GetPrimAtPath(particle_system_path).IsValid():
        particleUtils.add_physx_particle_system(
            stage=stage, particle_system_path=Sdf.Path(particle_system_path),
            simulation_owner=Sdf.Path(physics_scene_path),
            particle_contact_offset=particle_radius_m * 1.15,
            rest_offset=particle_radius_m * 0.90,
            solid_rest_offset=particle_radius_m * 1.80,
            fluid_rest_offset=particle_radius_m * 0.92,
        )
        physicsUtils.add_physics_material_to_prim(
            stage, stage.GetPrimAtPath(particle_system_path), Sdf.Path(material_path)
        )
    if not stage.GetPrimAtPath(particle_set_path).IsValid():
        particleUtils.add_physx_particleset_points(
            stage, Sdf.Path(particle_set_path), [], [], [], Sdf.Path(particle_system_path),
            True, True, 0, 1.0, particle_radius_m * 2.0,
        )
        points = UsdGeom.Points(stage.GetPrimAtPath(particle_set_path))
        points.GetWidthsAttr().Set([])
    return {
        "root_path": root_path, "material_path": material_path,
        "particle_system_path": particle_system_path, "particle_set_path": particle_set_path,
    }


def emit_irrigation_burst(
    tool_path: str, ledger: FluidLedger, *, requested_ml: float = 0.25,
    jet_speed_m_s: float = 1.20, launch_spread_deg: float = 1.5,
    random_seed: int | None = 0, stage=None,
    particle_set_path: str = "/World/DrAnmarIrrigationParticles/Particles",
) -> dict[str, Any]:
    """Append a multi-nozzle PBD particle burst and debit its exact particle volume."""
    stage = _current_stage(stage)
    from pxr import Gf, UsdGeom, Vt

    points = UsdGeom.Points(stage.GetPrimAtPath(particle_set_path))
    if not points:
        raise ValueError(f"No irrigation particle set at {particle_set_path}")
    jet_speed_m_s = _nonnegative_finite(jet_speed_m_s, "jet_speed_m_s")
    launch_spread_deg = _nonnegative_finite(launch_spread_deg, "launch_spread_deg")
    available = ledger.emit(requested_ml)
    requested_particles = int(available / PARTICLE_VOLUME_ML)
    count = max(0, requested_particles - requested_particles % IRRIGATION_NOZZLE_COUNT)
    actual_ml = count * PARTICLE_VOLUME_ML
    # Return non-emitted quantization remainder to the reservoir.
    remainder = available - actual_ml
    ledger.reservoir_ml += remainder
    ledger.emitted_ml -= remainder
    ledger.active_particle_ml -= remainder
    if count == 0:
        return {"particle_count": 0, "emitted_ml": 0.0, "quantization_remainder_ml": remainder}

    transform = _world_transform(stage, frame_path(tool_path, "irrigation_jet_origin"))
    current_positions = list(points.GetPointsAttr().Get() or [])
    current_velocities = list(points.GetVelocitiesAttr().Get() or [])
    current_widths = list(points.GetWidthsAttr().Get() or [])
    if len(current_velocities) < len(current_positions):
        current_velocities.extend(
            Gf.Vec3f(0.0) for _ in range(len(current_positions) - len(current_velocities))
        )
    if len(current_widths) < len(current_positions):
        current_widths.extend(
            PARTICLE_RADIUS_M * 2.0
            for _ in range(len(current_positions) - len(current_widths))
        )
    current_velocities = current_velocities[:len(current_positions)]
    current_widths = current_widths[:len(current_positions)]
    rng = random.Random(random_seed)
    spread = math.tan(math.radians(launch_spread_deg))
    per_nozzle = count // IRRIGATION_NOZZLE_COUNT
    for nozzle in range(IRRIGATION_NOZZLE_COUNT):
        angle = 2.0 * math.pi * nozzle / IRRIGATION_NOZZLE_COUNT
        local_origin = Gf.Vec3d(0.0062 * math.cos(angle), 0.0062 * math.sin(angle), 0.0030)
        base_direction = Gf.Vec3d(
            -0.0044 * math.cos(angle), -0.0044 * math.sin(angle), 0.0140
        ).GetNormalized()
        tangent = Gf.Vec3d(-math.sin(angle), math.cos(angle), 0.0)
        radial = Gf.Vec3d(math.cos(angle), math.sin(angle), 0.0)
        world_origin = transform.Transform(local_origin)
        for index in range(per_nozzle):
            perturbed = (
                base_direction
                + tangent * (rng.uniform(-1.0, 1.0) * spread)
                + radial * (rng.uniform(-1.0, 1.0) * spread)
            ).GetNormalized()
            world_direction = transform.TransformDir(perturbed).GetNormalized()
            axial_jitter = (index % 5 - 2) * 0.00012
            position = world_origin + world_direction * axial_jitter
            current_positions.append(Gf.Vec3f(position))
            current_velocities.append(Gf.Vec3f(world_direction * jet_speed_m_s))
            current_widths.append(PARTICLE_RADIUS_M * 2.0)
    points.GetPointsAttr().Set(Vt.Vec3fArray(current_positions))
    points.GetVelocitiesAttr().Set(Vt.Vec3fArray(current_velocities))
    points.GetWidthsAttr().Set(current_widths)
    return {
        "particle_count": count, "emitted_ml": actual_ml,
        "particle_volume_ml": PARTICLE_VOLUME_ML,
        "launch_spread_deg": launch_spread_deg,
        "random_seed": random_seed,
        "quantization_remainder_ml": remainder,
        "particle_set_path": particle_set_path,
    }


@dataclass
class SuctionFieldController:
    capture_radius_m: float = 0.023
    capture_depth_m: float = 0.030
    throat_radius_m: float = 0.0065
    max_acceleration_m_s2: float = 18.0
    swirl_gain: float = 0.25

    def update_particles(
        self, tool_path: str, ledger: FluidLedger, *, dt: float, opening: float = 1.0,
        stage=None, particle_set_path: str = "/World/DrAnmarIrrigationParticles/Particles",
    ) -> dict[str, Any]:
        """Apply a converging suction field and remove particles entering the throat."""
        stage = _current_stage(stage)
        from pxr import Gf, UsdGeom, Vt
        dt = _nonnegative_finite(dt, "dt")
        points = UsdGeom.Points(stage.GetPrimAtPath(particle_set_path))
        positions = list(points.GetPointsAttr().Get() or [])
        velocities = list(points.GetVelocitiesAttr().Get() or [])
        if not positions:
            return {"active": 0, "captured": 0, "aspirated_ml": 0.0}
        opening = max(0.0, min(1.0, float(opening)))
        capture_T = _world_transform(stage, frame_path(tool_path, "suction_capture_center"))
        throat_T = _world_transform(stage, frame_path(tool_path, "suction_throat"))
        inverse_capture = capture_T.GetInverse()
        throat_world = throat_T.ExtractTranslation()
        kept_positions, kept_velocities, kept_widths = [], [], []
        widths = list(points.GetWidthsAttr().Get() or [PARTICLE_RADIUS_M * 2] * len(positions))
        if len(velocities) < len(positions):
            velocities.extend(Gf.Vec3f(0.0) for _ in range(len(positions) - len(velocities)))
        if len(widths) < len(positions):
            widths.extend(PARTICLE_RADIUS_M * 2.0 for _ in range(len(positions) - len(widths)))
        velocities = velocities[:len(positions)]
        widths = widths[:len(positions)]
        capture_budget = min(
            int((ledger.collection_remaining_ml + 1.0e-12) / PARTICLE_VOLUME_ML),
            int((ledger.active_particle_ml + 1.0e-12) / PARTICLE_VOLUME_ML),
        )
        captured = 0
        capture_blocked = 0
        for position, velocity, width in zip(positions, velocities, widths):
            world = Gf.Vec3d(position)
            local = inverse_capture.Transform(world)
            radial = math.hypot(local[0], local[1])
            in_capture = radial <= self.capture_radius_m and abs(local[2]) <= self.capture_depth_m * 0.75
            to_throat = throat_world - world
            distance = max(float(to_throat.GetLength()), 1.0e-8)
            if opening > 0 and distance <= self.throat_radius_m:
                if captured < capture_budget:
                    captured += 1
                    continue
                capture_blocked += 1
            new_velocity = Gf.Vec3d(velocity)
            if in_capture and opening > 0:
                direction = to_throat / distance
                swirl = Gf.Vec3d(-local[1], local[0], 0)
                if swirl.GetLength() > 1.0e-9:
                    swirl = capture_T.TransformDir(swirl.GetNormalized())
                gain = opening * self.max_acceleration_m_s2 * max(0.15, 1.0 - radial / self.capture_radius_m)
                new_velocity += (direction + swirl * self.swirl_gain) * gain * dt
            kept_positions.append(Gf.Vec3f(world))
            kept_velocities.append(Gf.Vec3f(new_velocity))
            kept_widths.append(float(width))
        points.GetPointsAttr().Set(Vt.Vec3fArray(kept_positions))
        points.GetVelocitiesAttr().Set(Vt.Vec3fArray(kept_velocities))
        points.GetWidthsAttr().Set(kept_widths)
        aspirated_ml = ledger.aspirate(captured * PARTICLE_VOLUME_ML)
        expected_ml = captured * PARTICLE_VOLUME_ML
        if not math.isclose(aspirated_ml, expected_ml, rel_tol=1.0e-9, abs_tol=1.0e-12):
            raise RuntimeError("particle capture and fluid ledger diverged")
        return {
            "active": len(kept_positions),
            "captured": captured,
            "capture_blocked": capture_blocked,
            "aspirated_ml": aspirated_ml,
        }

    def update_rigid_debris(
        self, tool_path: str, debris_paths: Iterable[str], *, dt: float,
        opening: float = 1.0, stage=None,
    ) -> dict[str, list[str]]:
        """Steer released rigid fragments toward the throat and remove captured ones."""
        stage = _current_stage(stage)
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        dt = _nonnegative_finite(dt, "dt")
        opening = max(0.0, min(1.0, _nonnegative_finite(opening, "opening")))
        capture_T = _world_transform(stage, frame_path(tool_path, "suction_capture_center"))
        inverse_capture = capture_T.GetInverse()
        throat_world = _world_transform(stage, frame_path(tool_path, "suction_throat")).ExtractTranslation()
        removed, accelerated = [], []
        for path in debris_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                continue
            position = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
            local = inverse_capture.Transform(position)
            radial = math.hypot(local[0], local[1])
            to_throat = throat_world - position
            distance = max(float(to_throat.GetLength()), 1.0e-8)
            if opening > 0 and distance <= self.throat_radius_m:
                stage.RemovePrim(path)
                removed.append(path)
                continue
            if radial <= self.capture_radius_m and abs(local[2]) <= self.capture_depth_m and opening > 0:
                velocity_attr = UsdPhysics.RigidBodyAPI(prim).GetVelocityAttr()
                current = Gf.Vec3f(velocity_attr.Get() or Gf.Vec3f(0))
                direction = to_throat / distance
                velocity_attr.Set(current + Gf.Vec3f(direction * (opening * self.max_acceleration_m_s2 * dt)))
                accelerated.append(path)
        return {"removed": removed, "accelerated": accelerated}


@dataclass
class DebrisBond:
    debris_path: str
    attachment_path: str
    threshold_j: float
    accumulated_work_j: float = 0.0
    released: bool = False


@dataclass
class DebridementReleaseController:
    """Release attached debris after cumulative brush/curette contact work."""
    bonds: dict[str, DebrisBond] = field(default_factory=dict)

    def register_demo(self, attachments: Mapping[str, str], *, stage=None) -> None:
        stage = _current_stage(stage)
        for debris_path, attachment_path in attachments.items():
            prim = stage.GetPrimAtPath(debris_path)
            if not prim or not prim.IsValid():
                raise ValueError(f"Invalid debris prim {debris_path}")
            threshold_attr = prim.GetAttribute("drAnmar:adhesionWorkThresholdJ")
            threshold_value = threshold_attr.Get()
            threshold = _nonnegative_finite(
                0.006 if threshold_value is None else threshold_value,
                "adhesion_work_threshold_j",
            )
            if threshold <= 0.0:
                raise ValueError("adhesion work threshold must be greater than zero")
            self.bonds[debris_path] = DebrisBond(debris_path, attachment_path, threshold)

    def update(
        self, contact_forces_n: Mapping[str, float], tangential_speeds_m_s: Mapping[str, float],
        *, dt: float, stage=None,
    ) -> list[str]:
        stage = _current_stage(stage)
        dt = _nonnegative_finite(dt, "dt")
        released: list[str] = []
        for path, bond in self.bonds.items():
            if bond.released:
                continue
            force = _nonnegative_finite(contact_forces_n.get(path, 0.0), "contact_force_n")
            speed = _nonnegative_finite(tangential_speeds_m_s.get(path, 0.0), "tangential_speed_m_s")
            bond.accumulated_work_j += force * speed * dt
            if bond.accumulated_work_j >= bond.threshold_j:
                if stage.GetPrimAtPath(bond.attachment_path).IsValid():
                    stage.RemovePrim(bond.attachment_path)
                bond.released = True
                released.append(path)
        return released

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            path: {
                "threshold_j": bond.threshold_j,
                "accumulated_work_j": bond.accumulated_work_j,
                "released": bond.released,
                "attachment_path": bond.attachment_path,
            }
            for path, bond in self.bonds.items()
        }


def phase_targets(phase: str) -> dict[str, float]:
    """Return joint targets for the canonical wound-preparation sequence."""
    phases = {
        "inspect": {
            "contact_guard_joint": 0.0, "debridement_extension_joint": 0.0,
            "debridement_rotor_joint_velocity": 0.0, "irrigation_valve_joint": 0.0,
            "suction_valve_joint": 0.0,
        },
        "contact": {
            "contact_guard_joint": 0.006, "debridement_extension_joint": 0.0,
            "debridement_rotor_joint_velocity": 0.0, "irrigation_valve_joint": 0.0,
            "suction_valve_joint": 0.0,
        },
        "pre_rinse": {
            "contact_guard_joint": 0.006, "debridement_extension_joint": 0.0,
            "debridement_rotor_joint_velocity": 0.0, "irrigation_valve_joint": 0.006,
            "suction_valve_joint": math.radians(45.0),
        },
        "aspirate": {
            "contact_guard_joint": 0.006, "debridement_extension_joint": 0.0,
            "debridement_rotor_joint_velocity": 0.0, "irrigation_valve_joint": 0.0,
            "suction_valve_joint": math.radians(80.0),
        },
        "debride": {
            "contact_guard_joint": 0.006, "debridement_extension_joint": 0.018,
            "debridement_rotor_joint_velocity": math.radians(2520.0),
            "irrigation_valve_joint": 0.002, "suction_valve_joint": math.radians(65.0),
        },
        "post_rinse": {
            "contact_guard_joint": 0.006, "debridement_extension_joint": 0.0,
            "debridement_rotor_joint_velocity": 0.0, "irrigation_valve_joint": 0.006,
            "suction_valve_joint": math.radians(75.0),
        },
        "dry": {
            "contact_guard_joint": 0.006, "debridement_extension_joint": 0.0,
            "debridement_rotor_joint_velocity": 0.0, "irrigation_valve_joint": 0.0,
            "suction_valve_joint": math.radians(85.0),
        },
        "verify": {
            "contact_guard_joint": 0.0, "debridement_extension_joint": 0.0,
            "debridement_rotor_joint_velocity": 0.0, "irrigation_valve_joint": 0.0,
            "suction_valve_joint": 0.0,
        },
    }
    try:
        return dict(phases[phase])
    except KeyError as exc:
        raise KeyError(f"Unknown wound-preparation phase {phase!r}; expected one of {sorted(phases)}") from exc


@dataclass
class WoundPreparationSequenceController:
    tool_path: str
    wound_root_path: str
    ledger: FluidLedger = field(default_factory=FluidLedger)
    suction: SuctionFieldController = field(default_factory=SuctionFieldController)
    debridement: DebridementReleaseController = field(default_factory=DebridementReleaseController)
    phase: str = "inspect"
    history: list[dict[str, Any]] = field(default_factory=list)

    def transition(self, phase: str) -> dict[str, float]:
        targets = phase_targets(phase)
        self.phase = phase
        self.history.append({"phase": phase, "targets": dict(targets), "fluid": self.ledger.snapshot()})
        return targets

    def snapshot(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "tool_path": self.tool_path,
            "wound_root_path": self.wound_root_path,
            "fluid": self.ledger.snapshot(),
            "debridement": self.debridement.snapshot(),
            "history": list(self.history),
            "status": "simulation_training_workcell",
        }
