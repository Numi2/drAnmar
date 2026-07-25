# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Isaac Lab integration for the DrAnmar atraumatic surgical exposure robot.

The tool replaces the Panda hand at its verified stock joint frame. Its bilateral
compliant pads use independent, overlap-prioritized vertex attachments. The control helpers
maintain ROI exposure while enforcing provisional pad-force limits. This module
is an engineering research interface, not clinical control software.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import math

CATALOG_SUBPATH = "Props/SurgicalExposure/AtraumaticExposureRobot"
ASSET_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
ROOT = ASSET_DATA_ROOT / CATALOG_SUBPATH
TOOL_PAYLOAD_USD = ROOT / "dranmar_atraumatic_exposure_tool_payload.usda"
TOOL_STANDALONE_USD = ROOT / "dranmar_atraumatic_exposure_tool_standalone.usda"
TOOL_RIGID_PROXY_USD = ROOT / "dranmar_atraumatic_exposure_tool_rigid_proxy.usda"
FENESTRATED_PAD_USD = ROOT / "dranmar_fenestrated_retraction_pad.usda"
MICROCUP_PAD_USD = ROOT / "dranmar_microcup_retraction_pad.usda"
TISSUE_DEMO_USD = ROOT / "dranmar_exposure_tissue_demo.usda"

VALID_PAD_TYPES = frozenset({"fenestrated", "microcup"})
CAPTURE_CELL_COUNT = 6

TOOL_JOINTS = {
    "left_carriage": "left_carriage_joint",
    "right_carriage": "right_carriage_joint",
    "left_lift": "left_lift_joint",
    "right_lift": "right_lift_joint",
    "left_pitch": "left_pitch_joint",
    "right_pitch": "right_pitch_joint",
    "left_compliance": "left_compliance_joint",
    "right_compliance": "right_compliance_joint",
}

TOOL_FRAME_PATHS = {
    "panda_link8_mount": "Links/Mount/Frames/panda_link8_mount",
    "exposure_tcp": "Links/Mount/Frames/exposure_tcp",
    "roi_camera": "Links/Mount/Frames/roi_camera",
    "illumination_center": "Links/Mount/Frames/illumination_center",
    "exposure_center": "Links/Mount/Frames/exposure_center",
    "count_reference": "Links/Mount/Frames/count_reference",
    "left_pad_center": "Links/LeftPad/Frames/left_pad_center",
    "right_pad_center": "Links/RightPad/Frames/right_pad_center",
    "left_pad_normal": "Links/LeftPad/Frames/left_pad_normal",
    "right_pad_normal": "Links/RightPad/Frames/right_pad_normal",
    "left_force_sensor": "Links/LeftPad/Frames/left_force_sensor",
    "right_force_sensor": "Links/RightPad/Frames/right_force_sensor",
}
REGISTERED_CAMERA_FRAMES = ("roi_camera",)
for _side in ("left", "right"):
    for _index in range(CAPTURE_CELL_COUNT):
        TOOL_FRAME_PATHS[f"{_side}_capture_{_index:02d}"] = (
            f"Links/{_side.capitalize()}Pad/Frames/{_side}_capture_{_index:02d}"
        )


def frame_path(tool_path: str, name: str) -> str:
    try:
        suffix = TOOL_FRAME_PATHS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown exposure-tool frame {name!r}") from exc
    return f"{tool_path.rstrip('/')}/{suffix}"


def tensor_value(value: Any):
    """Return a native tensor from Isaac 6 tensor proxy objects when required."""
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


def _nonnegative_finite(value: float, label: str) -> float:
    result = _finite(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def make_tool_cfg(
    prim_path: str = "/World/DrAnmarAtraumaticExposureTool",
    *,
    pad_type: str = "fenestrated",
    position=(0.0, 0.0, 0.35),
    orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
):
    """Return a standalone Isaac Lab articulation configuration."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    _check(pad_type, VALID_PAD_TYPES, "pad_type")
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(TOOL_STANDALONE_USD),
            variants={"pad_type": pad_type},
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=20,
                solver_velocity_iteration_count=6,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=position,
            rot=_xyzw_from_wxyz(orientation_wxyz),
            joint_pos={
                "left_carriage_joint": 0.0,
                "right_carriage_joint": 0.0,
                "left_lift_joint": -0.012,
                "right_lift_joint": -0.012,
                "left_pitch_joint": math.radians(58.0),
                "right_pitch_joint": math.radians(-58.0),
                "left_compliance_joint": 0.0,
                "right_compliance_joint": 0.0,
            },
        ),
        actuators={
            "lateral_retraction": ImplicitActuatorCfg(
                joint_names_expr=[".*_carriage_joint"],
                effort_limit_sim=95.0,
                velocity_limit_sim=0.12,
                stiffness=5200.0,
                damping=190.0,
            ),
            "independent_lift": ImplicitActuatorCfg(
                joint_names_expr=[".*_lift_joint"],
                effort_limit_sim=110.0,
                velocity_limit_sim=0.10,
                stiffness=6200.0,
                damping=210.0,
            ),
            "pad_pitch": ImplicitActuatorCfg(
                joint_names_expr=[".*_pitch_joint"],
                effort_limit_sim=7.0,
                velocity_limit_sim=2.0,
                stiffness=52.0,
                damping=2.5,
            ),
            "pad_compliance": ImplicitActuatorCfg(
                joint_names_expr=[".*_compliance_joint"],
                effort_limit_sim=16.0,
                velocity_limit_sim=0.08,
                stiffness=1250.0,
                damping=38.0,
            ),
        },
    )


def make_rigid_proxy_cfg(
    prim_path: str = "/World/DrAnmarAtraumaticExposureProxy",
    *,
    position=(0.0, 0.0, 0.35),
    orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(TOOL_RIGID_PROXY_USD),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=position, rot=_xyzw_from_wxyz(orientation_wxyz)
        ),
    )


def _spawn_single_franka_with_exposure_tool(
    prim_path: str,
    cfg: Any,
    translation=None,
    orientation=None,
    **kwargs,
):
    """Spawn stock Franka, remove Panda hand, and mount the exposure payload."""
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

    tool_path = f"{prim_path}/DrAnmarAtraumaticExposureTool"
    create_prim(tool_path, usd_path=str(TOOL_PAYLOAD_USD), stage=stage)
    select_usd_variants(tool_path, {"pad_type": cfg.pad_type})

    mount_joint = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/dranmar_exposure_mount_joint")
    mount_joint.CreateBody0Rel().SetTargets(mount_body_paths)
    mount_joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{tool_path}/Links/Mount")])
    mount_joint.CreateLocalPos0Attr().Set(mount_local_pos0)
    mount_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    mount_joint.CreateLocalRot0Attr().Set(mount_local_rot0)
    mount_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    return robot


def spawn_franka_with_exposure_tool(
    prim_path: str, cfg: Any, translation=None, orientation=None, **kwargs
):
    from isaaclab.sim.utils import clone
    return clone(_spawn_single_franka_with_exposure_tool)(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )


def make_franka_exposure_robot_cfg(
    *,
    prim_path: str = "/World/Robot",
    pad_type: str = "fenestrated",
):
    """Return the stock Isaac Lab Franka with the Panda hand replaced."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

    _check(pad_type, VALID_PAD_TYPES, "pad_type")

    @configclass
    class FrankaExposureUsdCfg(sim_utils.UsdFileCfg):
        pad_type: str = "fenestrated"
        func = spawn_franka_with_exposure_tool

    cfg = FRANKA_PANDA_CFG.copy()
    cfg.prim_path = prim_path
    cfg.spawn = FrankaExposureUsdCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        variants={"Gripper": "Default", "Mesh": "Performance"},
        pad_type=pad_type,
        activate_contact_sensors=True,
        rigid_props=FRANKA_PANDA_CFG.spawn.rigid_props,
        articulation_props=FRANKA_PANDA_CFG.spawn.articulation_props,
    )
    cfg.init_state.joint_pos = {
        key: value for key, value in cfg.init_state.joint_pos.items() if "finger" not in key
    }
    cfg.init_state.joint_pos.update({
        "left_carriage_joint": 0.0,
        "right_carriage_joint": 0.0,
        "left_lift_joint": -0.012,
        "right_lift_joint": -0.012,
        "left_pitch_joint": math.radians(58.0),
        "right_pitch_joint": math.radians(-58.0),
        "left_compliance_joint": 0.0,
        "right_compliance_joint": 0.0,
    })
    cfg.actuators.pop("panda_hand", None)
    cfg.actuators.update({
        "exposure_lateral": ImplicitActuatorCfg(
            joint_names_expr=[".*_carriage_joint"], effort_limit_sim=95.0,
            velocity_limit_sim=0.12, stiffness=5200.0, damping=190.0,
        ),
        "exposure_lift": ImplicitActuatorCfg(
            joint_names_expr=[".*_lift_joint"], effort_limit_sim=110.0,
            velocity_limit_sim=0.10, stiffness=6200.0, damping=210.0,
        ),
        "exposure_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_pitch_joint"], effort_limit_sim=7.0,
            velocity_limit_sim=2.0, stiffness=52.0, damping=2.5,
        ),
        "exposure_compliance": ImplicitActuatorCfg(
            joint_names_expr=[".*_compliance_joint"], effort_limit_sim=16.0,
            velocity_limit_sim=0.08, stiffness=1250.0, damping=38.0,
        ),
    })
    return cfg


def spawn_exposure_tissue_demo(
    prim_path: str = "/World/DrAnmarExposureTissue",
    *,
    translation=(0.0, 0.0, 0.0),
    orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
):
    import isaaclab.sim as sim_utils
    cfg = sim_utils.UsdFileCfg(usd_path=str(TISSUE_DEMO_USD))
    return cfg.func(
        prim_path, cfg, translation=translation,
        orientation=_xyzw_from_wxyz(orientation_wxyz),
    )


def _current_stage(stage=None):
    if stage is not None:
        return stage
    import omni.usd
    return omni.usd.get_context().get_stage()


def apply_exposure_tissue_surface_deformables(
    tissue_root_path: str = "/World/DrAnmarExposureTissue",
    *,
    stage=None,
    material_path: str = "/World/Materials/DrAnmarExposureTissueSurface",
    youngs_modulus_pa: float = 60_000.0,
    poissons_ratio: float = 0.45,
    surface_thickness_m: float = 0.006,
    density_kg_m3: float = 1_050.0,
    dynamic_friction: float = 0.58,
    elasticity_damping: float = 0.16,
    bend_damping: float = 0.14,
    self_collision: bool = True,
) -> dict[str, Any]:
    """Cook both portable flap meshes with the current surface-deformable API."""
    stage = _current_stage(stage)
    from omni.physx.scripts import deformableUtils
    from pxr import UsdShade

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

    result = {"root_path": tissue_root_path, "material_path": material_path, "flaps": {}}
    for side in ("LeftFlap", "RightFlap"):
        actor_path = f"{tissue_root_path.rstrip('/')}/{side}"
        mesh_path = f"{actor_path}/SimulationMesh"
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        if not mesh_prim or not mesh_prim.IsValid():
            raise ValueError(f"No exposure tissue mesh at {mesh_path}")
        success = deformableUtils.set_physics_surface_deformable_body(stage, mesh_prim.GetPath())
        if success is False:
            raise RuntimeError(f"PhysX could not create a surface deformable at {mesh_path}")
        mesh_prim.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
        if mesh_prim.HasAPI("PhysxSurfaceDeformableBodyAPI"):
            mesh_prim.GetAttribute("physxDeformableBody:selfCollision").Set(bool(self_collision))
        UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics"
        )
        result["flaps"][side] = {"actor_path": actor_path, "mesh_path": mesh_path}
    result["parameters"] = {
        "youngs_modulus_pa": youngs_modulus_pa,
        "poissons_ratio": poissons_ratio,
        "surface_thickness_m": surface_thickness_m,
        "density_kg_m3": density_kg_m3,
        "dynamic_friction": dynamic_friction,
        "elasticity_damping": elasticity_damping,
        "bend_damping": bend_damping,
        "self_collision": self_collision,
        "status": "provisional_engineering_seed",
    }
    return result


def create_deformable_attachment(
    deformable_prim_path: str,
    rigid_prim_path: str,
    attachment_path: str,
    *,
    stage=None,
) -> str:
    """Create a verified rigid/deformable attachment across Isaac generations."""
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt
    stage = _current_stage(stage)
    if stage.GetPrimAtPath(attachment_path).IsValid():
        stage.RemovePrim(attachment_path)

    prim_definition = Usd.SchemaRegistry().FindConcretePrimDefinition(
        "OmniPhysicsVtxXformAttachment"
    )
    if prim_definition:
        deformable_prim = stage.GetPrimAtPath(deformable_prim_path)
        if deformable_prim.IsValid() and not deformable_prim.IsA(UsdGeom.Mesh):
            candidate = stage.GetPrimAtPath(
                f"{deformable_prim_path.rstrip('/')}/SimulationMesh"
            )
            if candidate.IsValid():
                deformable_prim = candidate
                deformable_prim_path = str(candidate.GetPath())
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
                f"Could not create attachment {attachment_path}: "
                f"current={current_error!r}; legacy={legacy_error!r}"
            ) from legacy_error


def remove_prims(paths: Iterable[str], *, stage=None) -> None:
    stage = _current_stage(stage)
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            stage.RemovePrim(path)


def anchor_tissue_outer_bands(
    tissue_root_path: str = "/World/DrAnmarExposureTissue",
    *,
    stage=None,
) -> list[str]:
    """Attach the outer flap bands to the static fixture anchors."""
    stage = _current_stage(stage)
    attachments = []
    for side in ("Left", "Right"):
        path = f"{tissue_root_path}/Attachments/{side}OuterAnchor"
        create_deformable_attachment(
            f"{tissue_root_path}/{side}Flap",
            f"{tissue_root_path}/Fixture/{side}Anchor",
            path,
            stage=stage,
        )
        attachments.append(path)
    return attachments


@dataclass
class CaptureCell:
    side: str
    index: int
    attachment_path: str
    rigid_cell_path: str
    active: bool = True
    released_reason: str | None = None


@dataclass
class DistributedPadCaptureController:
    """Manage six independent tissue bonds per pad.

    Multiple small attachments distribute pad traction over the contact area.
    Overload handling progressively releases the outermost cells before the
    controller releases an entire pad. This is a research proxy for local loss
    of contact, not a calibrated tissue-injury or vacuum model.
    """

    tool_path: str
    left_tissue_path: str
    right_tissue_path: str
    stage: Any = None
    soft_cell_release_force_n: float = 0.75
    hard_pad_release_force_n: float = 4.0
    cells: list[CaptureCell] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.stage = _current_stage(self.stage)

    def capture(self) -> list[CaptureCell]:
        self.release_all("recapture")
        for side, tissue_path in (("left", self.left_tissue_path), ("right", self.right_tissue_path)):
            side_title = side.capitalize()
            for index in range(CAPTURE_CELL_COUNT):
                rigid_path = f"{self.tool_path}/Links/{side_title}Pad/Collisions/TissueCaptureCell_{index:02d}"
                attachment_path = f"{self.tool_path}/RuntimeAttachments/{side_title}Capture_{index:02d}"
                method = create_deformable_attachment(
                    tissue_path, rigid_path, attachment_path, stage=self.stage
                )
                self.cells.append(CaptureCell(side, index, attachment_path, rigid_path))
                self.events.append({"event": "capture", "side": side, "index": index, "method": method})
        return list(self.cells)

    def active_cells(self, side: str | None = None) -> list[CaptureCell]:
        return [cell for cell in self.cells if cell.active and (side is None or cell.side == side)]

    def release_cell(self, side: str, index: int, reason: str) -> bool:
        for cell in self.cells:
            if cell.side == side and cell.index == index and cell.active:
                remove_prims([cell.attachment_path], stage=self.stage)
                cell.active = False
                cell.released_reason = reason
                self.events.append({"event": "release_cell", "side": side, "index": index, "reason": reason})
                return True
        return False

    def release_all(self, reason: str = "commanded_release") -> None:
        active = self.active_cells()
        remove_prims([cell.attachment_path for cell in active], stage=self.stage)
        for cell in active:
            cell.active = False
            cell.released_reason = reason
        if active:
            self.events.append({"event": "release_all", "reason": reason, "count": len(active)})

    def update_loads(
        self,
        *,
        left_total_force_n: float,
        right_total_force_n: float,
        left_cell_forces_n: Sequence[float] | None = None,
        right_cell_forces_n: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        result = {"released": [], "hard_release": []}
        for side, total, values in (
            ("left", left_total_force_n, left_cell_forces_n),
            ("right", right_total_force_n, right_cell_forces_n),
        ):
            total = _nonnegative_finite(total, f"{side}_total_force_n")
            if total >= self.hard_pad_release_force_n:
                for cell in self.active_cells(side):
                    self.release_cell(side, cell.index, "hard_pad_overload")
                    result["hard_release"].append((side, cell.index))
                continue
            if values is not None:
                for index, force in enumerate(values[:CAPTURE_CELL_COUNT]):
                    force = _nonnegative_finite(force, f"{side}_cell_force_n[{index}]")
                    if force >= self.soft_cell_release_force_n:
                        if self.release_cell(side, index, "local_cell_overload"):
                            result["released"].append((side, index))
            elif total > 0:
                active = self.active_cells(side)
                estimate = total / max(1, len(active))
                if estimate >= self.soft_cell_release_force_n and active:
                    # Release the furthest longitudinal cell first.
                    order = [0, 3, 2, 5, 1, 4]
                    chosen = next((i for i in order if any(c.index == i for c in active)), active[0].index)
                    if self.release_cell(side, chosen, "estimated_distributed_overload"):
                        result["released"].append((side, chosen))
        result["active_left"] = len(self.active_cells("left"))
        result["active_right"] = len(self.active_cells("right"))
        return result


def estimate_pad_force_n(
    compression_m: float,
    compression_velocity_m_s: float = 0.0,
    *,
    stiffness_n_m: float = 1_250.0,
    damping_n_s_m: float = 38.0,
) -> float:
    """Estimate normal pad load from the authored compliant-axis deflection."""
    compression = max(0.0, -_finite(compression_m, "compression_m"))
    closing_velocity = max(
        0.0, -_finite(compression_velocity_m_s, "compression_velocity_m_s")
    )
    stiffness = _nonnegative_finite(stiffness_n_m, "stiffness_n_m")
    damping = _nonnegative_finite(damping_n_s_m, "damping_n_s_m")
    return stiffness * compression + damping * closing_velocity


@dataclass
class ForceControlOutput:
    joint_targets: dict[str, float]
    force_error_n: dict[str, float]
    exposure_error: float
    overload: dict[str, bool]
    mode: str


@dataclass
class ForceControlledRetractionController:
    """Outer-loop ROI controller with independent force-limited pad motion."""

    target_visible_fraction: float = 0.88
    target_force_per_pad_n: float = 1.25
    soft_force_limit_n: float = 2.5
    hard_force_limit_n: float = 4.0
    max_force_asymmetry_n: float = 1.0
    lateral_gain_m_per_fraction: float = 0.010
    lift_gain_m_per_fraction: float = 0.006
    force_gain_m_per_n: float = 0.0018
    integral_gain_m_per_fraction_s: float = 0.0012
    max_integral_m: float = 0.008
    nominal_update_hz: float = 120.0
    left_carriage_m: float = 0.006
    right_carriage_m: float = -0.006
    left_lift_m: float = 0.017
    right_lift_m: float = 0.017
    integral_error: float = 0.0

    def reset(self) -> None:
        self.left_carriage_m = 0.006
        self.right_carriage_m = -0.006
        self.left_lift_m = 0.017
        self.right_lift_m = 0.017
        self.integral_error = 0.0

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def update(
        self,
        *,
        dt: float,
        visible_fraction: float,
        left_force_n: float,
        right_force_n: float,
    ) -> ForceControlOutput:
        dt = _finite(dt, "dt")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        update_scale = dt * _nonnegative_finite(
            self.nominal_update_hz, "nominal_update_hz"
        )
        if update_scale <= 0.0:
            raise ValueError("nominal_update_hz must be positive")
        visible = self._clamp(_finite(visible_fraction, "visible_fraction"), 0.0, 1.0)
        left_force = _nonnegative_finite(left_force_n, "left_force_n")
        right_force = _nonnegative_finite(right_force_n, "right_force_n")
        exposure_error = self.target_visible_fraction - visible
        self.integral_error = self._clamp(
            self.integral_error + exposure_error * dt,
            -self.max_integral_m / max(self.integral_gain_m_per_fraction_s, 1e-9),
            self.max_integral_m / max(self.integral_gain_m_per_fraction_s, 1e-9),
        )

        left_over = left_force >= self.soft_force_limit_n
        right_over = right_force >= self.soft_force_limit_n
        hard_left = left_force >= self.hard_force_limit_n
        hard_right = right_force >= self.hard_force_limit_n

        if hard_left or hard_right:
            # Immediate commanded unloading; capture controller handles bond release.
            relief_step = 0.004 * update_scale
            self.left_carriage_m = max(0.0, self.left_carriage_m - relief_step)
            self.right_carriage_m = min(0.0, self.right_carriage_m + relief_step)
            self.left_lift_m = min(0.030, self.left_lift_m + relief_step)
            self.right_lift_m = min(0.030, self.right_lift_m + relief_step)
            mode = "hard_overload_relief"
        else:
            visibility_step = (
                self.lateral_gain_m_per_fraction * exposure_error
                + self.integral_gain_m_per_fraction_s * self.integral_error
            )
            left_force_error = self.target_force_per_pad_n - left_force
            right_force_error = self.target_force_per_pad_n - right_force
            left_step = (
                visibility_step + self.force_gain_m_per_n * left_force_error
            ) * update_scale
            right_step = (
                visibility_step + self.force_gain_m_per_n * right_force_error
            ) * update_scale
            if left_over:
                left_step = min(
                    left_step,
                    -self.force_gain_m_per_n
                    * (left_force - self.soft_force_limit_n)
                    * update_scale,
                )
            if right_over:
                right_step = min(
                    right_step,
                    -self.force_gain_m_per_n
                    * (right_force - self.soft_force_limit_n)
                    * update_scale,
                )
            self.left_carriage_m = self._clamp(self.left_carriage_m + left_step, 0.0, 0.040)
            self.right_carriage_m = self._clamp(self.right_carriage_m - right_step, -0.040, 0.0)

            # Lift assists exposure but unloads a pad that is already force limited.
            lift_step = (
                self.lift_gain_m_per_fraction * exposure_error * update_scale
            )
            overload_lift_step = 0.002 * update_scale
            self.left_lift_m = self._clamp(
                self.left_lift_m
                - lift_step
                + (overload_lift_step if left_over else 0.0),
                -0.025,
                0.030,
            )
            self.right_lift_m = self._clamp(
                self.right_lift_m
                - lift_step
                + (overload_lift_step if right_over else 0.0),
                -0.025,
                0.030,
            )
            mode = "force_limited_exposure_control"

        # Differential correction reduces excessive bilateral force asymmetry.
        asymmetry = left_force - right_force
        if abs(asymmetry) > self.max_force_asymmetry_n:
            correction = min(0.0025, 0.0012 * abs(asymmetry)) * update_scale
            if asymmetry > 0:
                self.left_carriage_m = max(0.0, self.left_carriage_m - correction)
            else:
                self.right_carriage_m = min(0.0, self.right_carriage_m + correction)

        return ForceControlOutput(
            joint_targets={
                "left_carriage_joint": self.left_carriage_m,
                "right_carriage_joint": self.right_carriage_m,
                "left_lift_joint": self.left_lift_m,
                "right_lift_joint": self.right_lift_m,
                "left_pitch_joint": math.radians(-16.0),
                "right_pitch_joint": math.radians(16.0),
                "left_compliance_joint": 0.0,
                "right_compliance_joint": 0.0,
            },
            force_error_n={
                "left": self.target_force_per_pad_n - left_force,
                "right": self.target_force_per_pad_n - right_force,
                "asymmetry": asymmetry,
            },
            exposure_error=exposure_error,
            overload={"left_soft": left_over, "right_soft": right_over, "left_hard": hard_left, "right_hard": hard_right},
            mode=mode,
        )


class ROIExposureEstimator:
    """Visibility metrics usable with segmentation masks or geometric flap edges."""

    @staticmethod
    def from_masks(roi_mask: Any, occluder_mask: Any) -> float:
        import numpy as np
        roi = np.asarray(tensor_value(roi_mask), dtype=bool)
        occluder = np.asarray(tensor_value(occluder_mask), dtype=bool)
        if roi.shape != occluder.shape:
            raise ValueError(f"mask shape mismatch: roi={roi.shape}, occluder={occluder.shape}")
        total = int(roi.sum())
        if total == 0:
            return 0.0
        visible = np.logical_and(roi, np.logical_not(occluder)).sum()
        return float(visible / total)

    @staticmethod
    def from_edge_gap(gap_width_m: float, target_width_m: float = 0.044) -> float:
        gap = _finite(gap_width_m, "gap_width_m")
        target = _finite(target_width_m, "target_width_m")
        if target <= 0:
            raise ValueError("target_width_m must be positive")
        return max(0.0, min(1.0, gap / target))

    @staticmethod
    def bilateral_balance(left_visible_fraction: float, right_visible_fraction: float) -> float:
        left = max(
            0.0,
            min(1.0, _finite(left_visible_fraction, "left_visible_fraction")),
        )
        right = max(
            0.0,
            min(1.0, _finite(right_visible_fraction, "right_visible_fraction")),
        )
        return 1.0 - abs(left - right)


PHASE_TARGETS = {
    "stowed": {
        "left_carriage_joint": 0.0, "right_carriage_joint": 0.0,
        "left_lift_joint": -0.012, "right_lift_joint": -0.012,
        "left_pitch_joint": math.radians(58.0), "right_pitch_joint": math.radians(-58.0),
        "left_compliance_joint": 0.0, "right_compliance_joint": 0.0,
    },
    "approach": {
        "left_carriage_joint": 0.002, "right_carriage_joint": -0.002,
        "left_lift_joint": 0.004, "right_lift_joint": 0.004,
        "left_pitch_joint": math.radians(35.0), "right_pitch_joint": math.radians(-35.0),
        "left_compliance_joint": 0.0, "right_compliance_joint": 0.0,
    },
    "deploy": {
        "left_carriage_joint": 0.006, "right_carriage_joint": -0.006,
        "left_lift_joint": 0.014, "right_lift_joint": 0.014,
        "left_pitch_joint": math.radians(12.0), "right_pitch_joint": math.radians(-12.0),
        "left_compliance_joint": 0.0, "right_compliance_joint": 0.0,
    },
    "contact": {
        "left_carriage_joint": 0.006, "right_carriage_joint": -0.006,
        "left_lift_joint": 0.017, "right_lift_joint": 0.017,
        "left_pitch_joint": math.radians(2.0), "right_pitch_joint": math.radians(-2.0),
        "left_compliance_joint": -0.002, "right_compliance_joint": -0.002,
    },
    "capture": {
        "left_carriage_joint": 0.006, "right_carriage_joint": -0.006,
        "left_lift_joint": 0.017, "right_lift_joint": 0.017,
        "left_pitch_joint": math.radians(2.0), "right_pitch_joint": math.radians(-2.0),
        "left_compliance_joint": -0.002, "right_compliance_joint": -0.002,
    },
    "retract": {
        "left_carriage_joint": 0.032, "right_carriage_joint": -0.032,
        "left_lift_joint": -0.012, "right_lift_joint": -0.012,
        "left_pitch_joint": math.radians(-16.0), "right_pitch_joint": math.radians(16.0),
        "left_compliance_joint": 0.0, "right_compliance_joint": 0.0,
    },
    "hold": {
        "left_carriage_joint": 0.032, "right_carriage_joint": -0.032,
        "left_lift_joint": -0.012, "right_lift_joint": -0.012,
        "left_pitch_joint": math.radians(-16.0), "right_pitch_joint": math.radians(16.0),
        "left_compliance_joint": 0.0, "right_compliance_joint": 0.0,
    },
    "overload_relief": {
        "left_carriage_joint": 0.020, "right_carriage_joint": -0.020,
        "left_lift_joint": -0.004, "right_lift_joint": -0.004,
        "left_pitch_joint": math.radians(-5.0), "right_pitch_joint": math.radians(5.0),
        "left_compliance_joint": 0.0, "right_compliance_joint": 0.0,
    },
    "release": {
        "left_carriage_joint": 0.010, "right_carriage_joint": -0.010,
        "left_lift_joint": 0.010, "right_lift_joint": 0.010,
        "left_pitch_joint": math.radians(20.0), "right_pitch_joint": math.radians(-20.0),
        "left_compliance_joint": 0.0, "right_compliance_joint": 0.0,
    },
}


def phase_targets(phase: str) -> dict[str, float]:
    try:
        return dict(PHASE_TARGETS[phase])
    except KeyError as exc:
        raise KeyError(f"Unknown exposure phase {phase!r}; expected one of {sorted(PHASE_TARGETS)}") from exc


@dataclass
class ExposureSequenceController:
    """Discrete workflow coordinator around capture and force-aware hold control."""

    tool_path: str
    left_tissue_path: str
    right_tissue_path: str
    stage: Any = None
    phase: str = "stowed"
    capture: DistributedPadCaptureController = field(init=False)
    force_controller: ForceControlledRetractionController = field(default_factory=ForceControlledRetractionController)
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.stage = _current_stage(self.stage)
        self.capture = DistributedPadCaptureController(
            tool_path=self.tool_path,
            left_tissue_path=self.left_tissue_path,
            right_tissue_path=self.right_tissue_path,
            stage=self.stage,
        )

    def set_phase(self, phase: str) -> dict[str, float]:
        targets = phase_targets(phase)
        if phase == "capture":
            self.capture.capture()
        elif phase == "release":
            self.capture.release_all("sequence_release")
        elif phase == "stowed":
            self.force_controller.reset()
        self.phase = phase
        self.history.append({"event": "phase", "phase": phase, "targets": targets})
        return targets

    def hold_update(
        self,
        *,
        dt: float,
        visible_fraction: float,
        left_compression_m: float,
        right_compression_m: float,
        left_compression_velocity_m_s: float = 0.0,
        right_compression_velocity_m_s: float = 0.0,
        left_cell_forces_n: Sequence[float] | None = None,
        right_cell_forces_n: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        left_force = estimate_pad_force_n(left_compression_m, left_compression_velocity_m_s)
        right_force = estimate_pad_force_n(right_compression_m, right_compression_velocity_m_s)
        control = self.force_controller.update(
            dt=dt,
            visible_fraction=visible_fraction,
            left_force_n=left_force,
            right_force_n=right_force,
        )
        release = self.capture.update_loads(
            left_total_force_n=left_force,
            right_total_force_n=right_force,
            left_cell_forces_n=left_cell_forces_n,
            right_cell_forces_n=right_cell_forces_n,
        )
        if control.overload["left_hard"] or control.overload["right_hard"]:
            self.phase = "overload_relief"
        else:
            self.phase = "hold"
        result = {
            "phase": self.phase,
            "joint_targets": control.joint_targets,
            "visible_fraction": float(visible_fraction),
            "left_force_n": left_force,
            "right_force_n": right_force,
            "force_error_n": control.force_error_n,
            "exposure_error": control.exposure_error,
            "overload": control.overload,
            "capture_release": release,
            "mode": control.mode,
        }
        self.history.append({"event": "hold_update", **result})
        return result
