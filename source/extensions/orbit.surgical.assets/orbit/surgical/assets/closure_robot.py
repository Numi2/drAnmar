"""Isaac Lab integration for the DrAnmar Approximate-Staple-Seal robot.

This module intentionally keeps the closure mechanics in the simulator:

* the payload is fixed directly to ``panda_link8`` before the articulation
  view is initialized;
* tissue capture, retained staple legs, and adhesive bonds are native PhysX
  deformable attachments;
* tissue transforms and nodal positions are never rewritten to imitate a
  closure.

This is an executable simulation-training mechanism with disclosed engineering
parameters. Nothing in this module models staple penetration, metal plasticity,
adhesive chemistry, biological healing, or clinically calibrated failure
strength; real-world and clinical evidence are not established.
"""

from __future__ import annotations

import math
import os
from functools import partial
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping


APP_VERSION = "0.1.0"
CATALOG_SUBPATH = Path("Props/SurgicalClosure/ClosureRobot")
EXTENSION_ROOT = Path(__file__).resolve().parents[3]
ASSET_DATA_ROOT = EXTENSION_ROOT / "data"
ASSET_ROOT_ENV = "DRANMAR_CLOSURE_ROBOT_ASSET_ROOT"

TOOL_ROOT_NAME = "DrAnmarClosureTool"
MOUNT_ROTATION_WXYZ = (
    0.9238795325112867,
    0.0,
    0.0,
    -0.3826834323650898,
)
FRANKA_REMOVED_PRIM_NAMES = frozenset(
    {
        "panda_hand_joint",
        "panda_hand",
        "panda_finger_joint1",
        "panda_finger_joint2",
        "panda_leftfinger",
        "panda_rightfinger",
    }
)

CLOSURE_JOINT_NAMES = (
    "left_approximation_joint",
    "right_approximation_joint",
    "left_clamp_joint",
    "right_clamp_joint",
    "staple_driver_joint",
    "adhesive_deploy_joint",
    "adhesive_meter_joint",
)

OPEN_TARGETS = {
    "left_approximation_joint": 0.0,
    "right_approximation_joint": 0.0,
    "left_clamp_joint": math.radians(28.0),
    "right_clamp_joint": math.radians(-28.0),
    "staple_driver_joint": 0.0,
    "adhesive_deploy_joint": 0.0,
    "adhesive_meter_joint": 0.0,
}


class ClosurePhase(str, Enum):
    READY = "ready"
    CAPTURE = "capture"
    APPROXIMATE = "approximate"
    STAPLE = "staple"
    RELEASE = "release"
    ADHESIVE = "adhesive"
    CURE_LEADING = "cure_leading"
    CURE_TRAILING = "cure_trailing"
    COMPLETE = "complete"


def _coerce_phase(value: str | ClosurePhase) -> ClosurePhase:
    if isinstance(value, ClosurePhase):
        return value
    try:
        return ClosurePhase(str(value).strip().lower())
    except ValueError as exc:
        allowed = ", ".join(item.value for item in ClosurePhase)
        raise ValueError(f"Unsupported closure phase {value!r}; expected: {allowed}") from exc


def asset_root(*, require: bool = True) -> Path:
    """Resolve the repository catalog directory or an explicit extraction."""

    candidates: list[Path] = []
    if override := os.environ.get(ASSET_ROOT_ENV):
        candidates.append(Path(override).expanduser())
    candidates.append(ASSET_DATA_ROOT / CATALOG_SUBPATH)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    if require:
        rendered = "\n".join(f"- {candidate}" for candidate in candidates)
        raise FileNotFoundError(
            "DrAnmar closure-robot assets were not found. Checked:\n"
            f"{rendered}\nSet {ASSET_ROOT_ENV} to a catalog extraction."
        )
    return candidates[0]


def _asset(name: str) -> Path:
    path = asset_root() / name
    if not path.is_file():
        raise FileNotFoundError(f"Closure-robot asset is missing: {path}")
    return path


def payload_usd() -> Path:
    return _asset("dranmar_closure_tool_payload.usda")


def standalone_tool_usd() -> Path:
    return _asset("dranmar_closure_tool_standalone.usda")


def rigid_proxy_usd() -> Path:
    return _asset("dranmar_closure_tool_rigid_proxy.usda")


def tissue_demo_usd() -> Path:
    return _asset("dranmar_closure_tissue_demo.usda")


def formed_staple_usd() -> Path:
    return _asset("dranmar_closure_staple.usda")


def adhesive_bead_usd() -> Path:
    return _asset("dranmar_closure_adhesive_bead.usda")


def closure_phase_targets(
    phase: str | ClosurePhase,
    *,
    approximation_m: float = 0.022,
    adhesive_meter_fraction: float = 1.0,
) -> dict[str, float]:
    """Return bounded mechanism targets for one ordered closure phase."""

    selected = _coerce_phase(phase)
    approximation = min(max(float(approximation_m), 0.0), 0.026)
    meter_fraction = min(max(float(adhesive_meter_fraction), 0.0), 1.0)
    targets = dict(OPEN_TARGETS)
    if selected in {ClosurePhase.READY, ClosurePhase.COMPLETE}:
        return targets

    targets["left_clamp_joint"] = 0.0
    targets["right_clamp_joint"] = 0.0
    if selected is ClosurePhase.CAPTURE:
        return targets

    targets["left_approximation_joint"] = approximation
    targets["right_approximation_joint"] = -approximation
    if selected is ClosurePhase.APPROXIMATE:
        return targets

    if selected is ClosurePhase.STAPLE:
        targets["staple_driver_joint"] = 0.014
        return targets

    if selected is ClosurePhase.RELEASE:
        targets["left_approximation_joint"] = 0.0
        targets["right_approximation_joint"] = 0.0
        targets["left_clamp_joint"] = math.radians(28.0)
        targets["right_clamp_joint"] = math.radians(-28.0)
        return targets

    targets["left_approximation_joint"] = 0.0
    targets["right_approximation_joint"] = 0.0
    targets["left_clamp_joint"] = math.radians(28.0)
    targets["right_clamp_joint"] = math.radians(-28.0)
    targets["adhesive_deploy_joint"] = -0.030
    targets["adhesive_meter_joint"] = 0.010 * meter_fraction
    return targets


def tensor_value(value: Any) -> Any:
    """Return a Torch-compatible tensor from Isaac/Warp proxy values."""

    if hasattr(value, "torch"):
        candidate = value.torch
        return candidate() if callable(candidate) else candidate
    try:
        import warp as wp  # type: ignore

        if isinstance(value, wp.array):
            return wp.to_torch(value)
    except (ImportError, ModuleNotFoundError, TypeError):
        pass
    return value


def set_joint_targets(
    articulation: Any,
    targets: Mapping[str, float],
) -> dict[str, float]:
    """Apply named closure-mechanism targets to a live articulation."""

    selected = {str(name): float(value) for name, value in targets.items()}
    if not selected or not all(math.isfinite(value) for value in selected.values()):
        raise ValueError("Closure joint targets must be non-empty and finite")
    joint_names = list(articulation.joint_names)
    try:
        joint_ids = [joint_names.index(name) for name in selected]
    except ValueError as exc:
        raise RuntimeError(
            "The Franka closure articulation is missing one or more payload joints"
        ) from exc

    import torch

    positions = tensor_value(articulation.data.joint_pos)
    target_tensor = torch.tensor(
        [[selected[name] for name in selected]],
        dtype=positions.dtype,
        device=positions.device,
    )
    articulation.set_joint_position_target(target_tensor, joint_ids=joint_ids)
    return selected


def set_closure_phase_target(
    articulation: Any,
    phase: str | ClosurePhase,
    **kwargs: float,
) -> dict[str, float]:
    targets = closure_phase_targets(phase, **kwargs)
    return set_joint_targets(articulation, targets)


def _closure_actuators() -> dict[str, Any]:
    from isaaclab.actuators import ImplicitActuatorCfg  # type: ignore

    return {
        "closure_approximation": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_approximation_joint",
                "right_approximation_joint",
            ],
            effort_limit_sim=80.0,
            velocity_limit_sim=0.06,
            stiffness=5000.0,
            damping=180.0,
            armature=0.02,
        ),
        "closure_clamps": ImplicitActuatorCfg(
            joint_names_expr=["left_clamp_joint", "right_clamp_joint"],
            effort_limit_sim=4.0,
            velocity_limit_sim=1.2,
            stiffness=35.0,
            damping=1.8,
            armature=0.006,
        ),
        "closure_staple_driver": ImplicitActuatorCfg(
            joint_names_expr=["staple_driver_joint"],
            effort_limit_sim=180.0,
            velocity_limit_sim=0.035,
            stiffness=14000.0,
            damping=260.0,
            armature=0.04,
        ),
        "closure_adhesive_deploy": ImplicitActuatorCfg(
            joint_names_expr=["adhesive_deploy_joint"],
            effort_limit_sim=35.0,
            velocity_limit_sim=0.04,
            stiffness=3500.0,
            damping=120.0,
            armature=0.02,
        ),
        "closure_adhesive_meter": ImplicitActuatorCfg(
            joint_names_expr=["adhesive_meter_joint"],
            effort_limit_sim=55.0,
            velocity_limit_sim=0.04,
            stiffness=7000.0,
            damping=180.0,
            armature=0.02,
        ),
    }


def make_tool_cfg(
    *,
    prim_path: str = "{ENV_REGEX_NS}/DrAnmarClosureTool",
    staple_state: str = "loaded",
    adhesive_state: str = "full",
    usd_path: str | Path | None = None,
    fix_root_link: bool = True,
):
    """Build the independently spawnable articulated closure tool."""

    import isaaclab.sim as sim_utils  # type: ignore
    from isaaclab.assets import ArticulationCfg  # type: ignore

    selected_path = standalone_tool_usd() if usd_path is None else Path(usd_path)
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(selected_path.expanduser().resolve()),
            variants={
                "staple_state": str(staple_state),
                "adhesive_state": str(adhesive_state),
            },
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=fix_root_link,
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos=dict(OPEN_TARGETS),
            joint_vel={".*": 0.0},
        ),
        actuators=_closure_actuators(),
    )


def _spawn_franka_payload(
    prim_path: str,
    cfg: Any,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    *,
    source_spawn: Any,
    staple_state: str,
    adhesive_state: str,
    **kwargs: Any,
) -> Any:
    """Spawn Franka, remove the stock hand, then compose the payload and joint."""

    root_prim = source_spawn(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )

    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root_path = str(root_prim.GetPath())
    descendants = list(Usd.PrimRange(root_prim))
    named = {prim.GetName(): prim for prim in descendants}
    missing = sorted(FRANKA_REMOVED_PRIM_NAMES - named.keys())
    if missing:
        raise RuntimeError(
            "The composable Franka asset changed; missing hand prims: "
            + ", ".join(missing)
        )

    link8 = named.get("panda_link8")
    if link8 is None or not link8.IsValid():
        # Isaac Sim 5.1's composable Franka omits the URDF's terminal
        # panda_link8 and joins panda_link7 directly to panda_hand. Preserve
        # the supplied mount contract by replacing that hand with a physical,
        # fixed panda_link8 at the exact same authored frame.
        link7 = named.get("panda_link7")
        hand = named.get("panda_hand")
        hand_joint_prim = named.get("panda_hand_joint")
        if (
            link7 is None
            or hand is None
            or hand_joint_prim is None
            or not link7.IsValid()
            or not hand.IsValid()
            or not hand_joint_prim.IsValid()
        ):
            raise RuntimeError(
                "Composable Franka asset has neither panda_link8 nor the "
                "panda_link7-to-hand frame needed to reconstruct it"
            )
        hand_joint = UsdPhysics.FixedJoint(hand_joint_prim)
        link8_path = Sdf.Path(root_path).AppendChild("panda_link8")
        link8 = stage.DefinePrim(link8_path, "Xform")
        link8_xform = UsdGeom.Xformable(link8)
        link8_xform.ClearXformOpOrder()
        link8_xform.AddTransformOp().Set(
            UsdGeom.Xformable(hand).GetLocalTransformation()
        )
        UsdPhysics.RigidBodyAPI.Apply(
            link8
        ).CreateRigidBodyEnabledAttr().Set(True)
        link8_mass = UsdPhysics.MassAPI.Apply(link8)
        link8_mass.CreateMassAttr().Set(0.0001)
        link8_mass.CreateCenterOfMassAttr().Set(
            Gf.Vec3f(0.0, 0.0, 0.0)
        )
        link8_mass.CreateDiagonalInertiaAttr().Set(
            Gf.Vec3f(1.0e-8, 1.0e-8, 1.0e-8)
        )
        link8_mass.CreatePrincipalAxesAttr().Set(
            Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
        )
        compatibility_scope = stage.DefinePrim(
            Sdf.Path(root_path).AppendChild("DrAnmarClosureJoints"),
            "Scope",
        )
        link8_joint = UsdPhysics.FixedJoint.Define(
            stage,
            compatibility_scope.GetPath().AppendChild(
                "panda_link7_to_link8"
            ),
        )
        link8_joint.CreateBody0Rel().SetTargets([link7.GetPath()])
        link8_joint.CreateBody1Rel().SetTargets([link8.GetPath()])
        link8_joint.CreateLocalPos0Attr().Set(
            hand_joint.GetLocalPos0Attr().Get()
        )
        link8_joint.CreateLocalPos1Attr().Set(
            hand_joint.GetLocalPos1Attr().Get()
        )
        link8_joint.CreateLocalRot0Attr().Set(
            hand_joint.GetLocalRot0Attr().Get()
        )
        link8_joint.CreateLocalRot1Attr().Set(
            hand_joint.GetLocalRot1Attr().Get()
        )
        link8_joint.CreateCollisionEnabledAttr().Set(False)
        named["panda_link8"] = link8

    for name in (
        "panda_finger_joint1",
        "panda_finger_joint2",
        "panda_leftfinger",
        "panda_rightfinger",
        "panda_hand_joint",
        "panda_hand",
    ):
        prim = named[name]
        if prim.IsInstanceProxy():
            raise RuntimeError(
                "The selected Franka USD is instanceable; the DrAnmar hand "
                "replacement requires the composable non-instanceable asset"
            )
        prim.SetActive(False)

    payload_path = Sdf.Path(root_path).AppendChild(TOOL_ROOT_NAME)
    payload_root = stage.DefinePrim(payload_path, "Xform")
    payload_root.GetReferences().AddReference(str(payload_usd()))
    payload_root.GetVariantSets().GetVariantSet("staple_state").SetVariantSelection(
        str(staple_state)
    )
    payload_root.GetVariantSets().GetVariantSet("adhesive_state").SetVariantSelection(
        str(adhesive_state)
    )

    # Seed the payload at the authored link8 pose.  The fixed joint remains
    # authoritative; this matching rest pose prevents an initialization shock.
    cache = UsdGeom.XformCache()
    root_world = cache.GetLocalToWorldTransform(root_prim)
    link8_world = cache.GetLocalToWorldTransform(link8)
    link8_local = link8_world * root_world.GetInverse()
    mount_rotation = Gf.Matrix4d(1.0)
    mount_rotation.SetRotate(
        Gf.Quatd(
            MOUNT_ROTATION_WXYZ[0],
            Gf.Vec3d(*MOUNT_ROTATION_WXYZ[1:]),
        )
    )
    payload_local = mount_rotation * link8_local
    xformable = UsdGeom.Xformable(payload_root)
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(payload_local)

    mount_link_path = payload_path.AppendPath("Links/Mount")
    joint_scope = stage.DefinePrim(
        Sdf.Path(root_path).AppendChild("DrAnmarClosureJoints"),
        "Scope",
    )
    fixed_joint = UsdPhysics.FixedJoint.Define(
        stage,
        joint_scope.GetPath().AppendChild("panda_link8_to_closure_mount"),
    )
    fixed_joint.CreateBody0Rel().SetTargets([link8.GetPath()])
    fixed_joint.CreateBody1Rel().SetTargets([mount_link_path])
    fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed_joint.CreateLocalRot0Attr().Set(
        Gf.Quatf(
            MOUNT_ROTATION_WXYZ[0],
            Gf.Vec3f(*MOUNT_ROTATION_WXYZ[1:]),
        )
    )
    fixed_joint.CreateLocalRot1Attr().Set(
        Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
    )
    fixed_joint.CreateCollisionEnabledAttr().Set(False)
    return root_prim


def make_franka_closure_robot_cfg(
    *,
    prim_path: str = "{ENV_REGEX_NS}/ClosureRobot",
    staple_state: str = "loaded",
    adhesive_state: str = "full",
    franka_usd_path: str | Path | None = None,
):
    """Build one Franka-plus-closure articulation with no Panda fingers."""

    import isaaclab.sim as sim_utils  # type: ignore
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # type: ignore

    cfg = FRANKA_PANDA_CFG.replace(prim_path=prim_path)
    if franka_usd_path is None:
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # type: ignore

        franka_path = (
            f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/"
            "FrankaPanda/franka.usd"
        )
    else:
        franka_path = str(franka_usd_path)
    spawn = sim_utils.UsdFileCfg(
        usd_path=franka_path,
        variants={"Gripper": "Default", "Mesh": "Performance"},
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # The station controller performs the same gravity compensation
            # expected of a powered surgical arm; contact and joint reaction
            # forces remain fully dynamic.
            disable_gravity=True,
            max_depenetration_velocity=1.0,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
        activate_contact_sensors=True,
    )
    spawn.func = partial(
        _spawn_franka_payload,
        source_spawn=spawn.func,
        staple_state=str(staple_state),
        adhesive_state=str(adhesive_state),
    )
    cfg.spawn = spawn

    cfg.init_state.joint_pos = {
        name: value
        for name, value in cfg.init_state.joint_pos.items()
        if "finger" not in name and "hand" not in name
    }
    cfg.init_state.joint_pos.update(OPEN_TARGETS)
    cfg.init_state.joint_vel = {".*": 0.0}
    cfg.actuators = {
        name: actuator
        for name, actuator in cfg.actuators.items()
        if "hand" not in name.lower() and "finger" not in name.lower()
    }
    # The stock training configuration's 12 Nm forearm limit is intended for
    # a lightweight Panda hand. The closure payload and two deformable-tissue
    # attachments need a gravity-compensated station hold with enough
    # physical drive authority to resist their reaction loads.
    cfg.actuators["panda_shoulder"].effort_limit_sim = 87.0
    cfg.actuators["panda_shoulder"].stiffness = 400.0
    cfg.actuators["panda_shoulder"].damping = 80.0
    cfg.actuators["panda_forearm"].effort_limit_sim = 40.0
    cfg.actuators["panda_forearm"].stiffness = 400.0
    cfg.actuators["panda_forearm"].damping = 80.0
    cfg.actuators.update(_closure_actuators())
    return cfg


def spawn_tissue_demo(
    prim_path: str = "/World/ClosureTissue",
    *,
    usd_path: str | Path | None = None,
) -> Any:
    """Reference the supplied two-sided wound tissue into the current stage."""

    import omni.usd
    from pxr import Sdf

    stage = omni.usd.get_context().get_stage()
    root = stage.DefinePrim(Sdf.Path(prim_path), "Xform")
    selected = tissue_demo_usd() if usd_path is None else Path(usd_path)
    root.GetReferences().AddReference(str(selected.expanduser().resolve()))
    return root


def apply_tissue_demo_surface_deformables(
    root_prim_path: str,
    *,
    youngs_modulus_pa: float = 80_000.0,
    poissons_ratio: float = 0.45,
    density_kg_m3: float = 1050.0,
    dynamic_friction: float = 0.55,
    surface_thickness_m: float = 0.0015,
) -> dict[str, Any]:
    """Cook both tissue meshes through PhysX's real surface-deformable API."""

    import omni.usd
    from omni.physx.scripts import deformableUtils
    from pxr import UsdShade

    stage = omni.usd.get_context().get_stage()
    root = root_prim_path.rstrip("/")
    mesh_paths = {
        "left": f"{root}/LeftTissue/SimulationMesh",
        "right": f"{root}/RightTissue/SimulationMesh",
    }
    material_path = f"{root}/ClosureTissuePhysicsMaterial"
    material = UsdShade.Material.Define(stage, material_path)
    material_prim = material.GetPrim()
    material_prim.ApplyAPI("OmniPhysicsBaseMaterialAPI")
    material_prim.GetAttribute("omniphysics:dynamicFriction").Set(
        float(dynamic_friction)
    )
    material_prim.GetAttribute("omniphysics:density").Set(float(density_kg_m3))
    material_prim.ApplyAPI("OmniPhysicsDeformableMaterialAPI")
    material_prim.GetAttribute("omniphysics:youngsModulus").Set(
        float(youngs_modulus_pa)
    )
    material_prim.GetAttribute("omniphysics:poissonsRatio").Set(
        float(poissons_ratio)
    )
    material_prim.ApplyAPI("OmniPhysicsSurfaceDeformableMaterialAPI")
    material_prim.GetAttribute("omniphysics:surfaceThickness").Set(
        float(surface_thickness_m)
    )
    material_prim.GetAttribute("omniphysics:surfaceBendStiffness").Set(0.0)
    material_prim.ApplyAPI("PhysxSurfaceDeformableMaterialAPI")
    material_prim.GetAttribute(
        "physxDeformableMaterial:elasticityDamping"
    ).Set(0.005)
    material_prim.GetAttribute("physxDeformableMaterial:bendDamping").Set(0.01)

    for mesh_path in mesh_paths.values():
        mesh = stage.GetPrimAtPath(mesh_path)
        if not mesh or not mesh.IsValid():
            raise ValueError(f"Closure tissue has no mesh at {mesh_path}")
        success = deformableUtils.set_physics_surface_deformable_body(
            stage,
            mesh.GetPath(),
        )
        if success is False:
            raise RuntimeError(
                f"PhysX failed to create a surface deformable at {mesh_path}"
            )
        mesh.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
        mesh.GetAttribute("physxDeformableBody:selfCollision").Set(False)
        UsdShade.MaterialBindingAPI.Apply(mesh).Bind(
            material,
            UsdShade.Tokens.weakerThanDescendants,
            "physics",
        )
    return {
        "root_prim_path": root,
        "left_tissue_path": mesh_paths["left"],
        "right_tissue_path": mesh_paths["right"],
        "material_path": material_path,
        "backend": "physx_surface_deformable",
        "transform_writes": False,
    }


def _attachment(
    stage: Any,
    path: str,
    actor0_path: str,
    actor1_path: str,
    *,
    overlap_offset_m: float,
) -> str:
    from pxr import PhysxSchema, Sdf

    attachment = PhysxSchema.PhysxPhysicsAttachment.Define(
        stage,
        Sdf.Path(path),
    )
    attachment.GetActor0Rel().SetTargets([Sdf.Path(actor0_path)])
    attachment.GetActor1Rel().SetTargets([Sdf.Path(actor1_path)])
    auto = PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment.GetPrim())
    auto.CreateDeformableVertexOverlapOffsetAttr(float(overlap_offset_m))
    auto.CreateCollisionFilteringOffsetAttr(float(overlap_offset_m))
    return path


def _remove_paths(stage: Any, paths: Iterable[str]) -> None:
    from pxr import Sdf

    for path in tuple(paths):
        if stage.GetPrimAtPath(path).IsValid():
            stage.RemovePrim(Sdf.Path(path))


def _attachment_evidence(
    stage: Any,
    paths: Iterable[str],
) -> dict[str, int]:
    """Read back the attachment schemas that the live stage exposes."""

    from pxr import PhysxSchema

    prim_count = 0
    enabled_count = 0
    actor_pair_count = 0
    auto_overlap_count = 0
    explicit_point_count = 0
    for path in tuple(paths):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        prim_count += 1
        attachment = PhysxSchema.PhysxPhysicsAttachment(prim)
        enabled = attachment.GetAttachmentEnabledAttr().Get()
        enabled_count += int(enabled is not False)
        auto_overlap_count += int(
            prim.HasAPI(PhysxSchema.PhysxAutoAttachmentAPI)
        )
        actor0_targets = attachment.GetActor0Rel().GetTargets()
        actor1_targets = attachment.GetActor1Rel().GetTargets()
        actor_pair_count += int(
            len(actor0_targets) == 1
            and len(actor1_targets) == 1
            and stage.GetPrimAtPath(actor0_targets[0]).IsValid()
            and stage.GetPrimAtPath(actor1_targets[0]).IsValid()
        )
        points0 = attachment.GetPoints0Attr().Get() or ()
        points1 = attachment.GetPoints1Attr().Get() or ()
        explicit_point_count += min(len(points0), len(points1))
    return {
        "attachment_prim_count": prim_count,
        "attachment_enabled_count": enabled_count,
        "attachment_actor_pair_count": actor_pair_count,
        "attachment_auto_overlap_count": auto_overlap_count,
        "attachment_explicit_point_count": explicit_point_count,
    }


def capture_tissue_edges(
    stage: Any,
    *,
    tool_path: str,
    left_tissue_path: str,
    right_tissue_path: str,
    attachment_root: str | None = None,
) -> tuple[str, str]:
    """Create temporary clamp-to-tissue attachments in the overlap volumes."""

    root = attachment_root or f"{tool_path.rstrip('/')}/RuntimeAttachments"
    stage.DefinePrim(root, "Scope")
    left_path = f"{root}/LeftClampCapture"
    right_path = f"{root}/RightClampCapture"
    return (
        _attachment(
            stage,
            left_path,
            left_tissue_path,
            f"{tool_path}/Links/LeftClamp/Collisions/TissueCaptureVolume",
            overlap_offset_m=0.0015,
        ),
        _attachment(
            stage,
            right_path,
            right_tissue_path,
            f"{tool_path}/Links/RightClamp/Collisions/TissueCaptureVolume",
            overlap_offset_m=0.0015,
        ),
    )


def anchor_tissue_outer_edges(
    stage: Any,
    *,
    tissue_root_path: str,
    left_tissue_path: str,
    right_tissue_path: str,
) -> tuple[str, str]:
    """Attach the two outer tissue margins to static physical anchor bars."""

    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    root = tissue_root_path.rstrip("/")
    anchor_root = f"{root}/PhysicalOuterAnchors"
    attachment_root = f"{root}/WorldAttachments"
    stage.DefinePrim(anchor_root, "Scope")
    stage.DefinePrim(attachment_root, "Scope")
    attachment_paths: list[str] = []
    for side, x_m, tissue_path in (
        ("Left", -0.050, left_tissue_path),
        ("Right", 0.050, right_tissue_path),
    ):
        anchor_path = f"{anchor_root}/{side}Anchor"
        cube = UsdGeom.Cube.Define(stage, Sdf.Path(anchor_path))
        cube.CreateSizeAttr(1.0)
        cube.CreateVisibilityAttr().Set("invisible")
        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(x_m, 0.0, 0.001))
        xform.AddScaleOp().Set(Gf.Vec3d(0.006, 0.110, 0.004))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr().Set(
            True
        )
        attachment_paths.append(
            _attachment(
                stage,
                f"{attachment_root}/{side}WorldAnchor",
                tissue_path,
                anchor_path,
                overlap_offset_m=0.0015,
            )
        )
    return tuple(attachment_paths)


def release_tissue_capture(stage: Any, attachment_paths: Iterable[str]) -> None:
    """Remove only temporary jaw attachments; retained closure bonds remain."""

    _remove_paths(stage, attachment_paths)


def _spawn_reference(
    stage: Any,
    prim_path: str,
    usd_path: Path,
    *,
    translation_m: tuple[float, float, float],
    orientation_wxyz: tuple[float, float, float, float],
    state: str,
) -> Any:
    from pxr import Gf, Sdf, UsdGeom

    root = stage.DefinePrim(Sdf.Path(prim_path), "Xform")
    root.GetReferences().AddReference(str(usd_path))
    root.GetVariantSets().GetVariantSet("state").SetVariantSelection(state)
    xform = UsdGeom.Xformable(root)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*translation_m))
    xform.AddOrientOp().Set(
        Gf.Quatf(orientation_wxyz[0], Gf.Vec3f(*orientation_wxyz[1:]))
    )
    return root


def deploy_formed_staple(
    stage: Any,
    *,
    prim_path: str,
    left_tissue_path: str,
    right_tissue_path: str,
    translation_m: tuple[float, float, float],
    orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> dict[str, Any]:
    """Spawn one dynamic formed staple and attach each leg independently."""

    root = _spawn_reference(
        stage,
        prim_path,
        formed_staple_usd(),
        translation_m=translation_m,
        orientation_wxyz=orientation_wxyz,
        state="formed",
    )
    attachment_root = f"{prim_path}/RuntimeAttachments"
    stage.DefinePrim(attachment_root, "Scope")
    left = _attachment(
        stage,
        f"{attachment_root}/LeftLegRetention",
        left_tissue_path,
        f"{prim_path}/Collisions/LeftLegAttachment",
        overlap_offset_m=0.0015,
    )
    right = _attachment(
        stage,
        f"{attachment_root}/RightLegRetention",
        right_tissue_path,
        f"{prim_path}/Collisions/RightLegAttachment",
        overlap_offset_m=0.0015,
    )
    return {
        "prim_path": str(root.GetPath()),
        "attachment_paths": (left, right),
        "attachment_count": 2,
        "dynamic_rigid_body": True,
        "kinematic": False,
    }


@dataclass
class StapleRetentionController:
    stage: Any
    provisional_pullout_threshold_n: float = 18.0
    deployments: list[dict[str, Any]] = field(default_factory=list)

    def deploy(self, **kwargs: Any) -> dict[str, Any]:
        deployment = deploy_formed_staple(self.stage, **kwargs)
        self.deployments.append(deployment)
        return deployment

    def report_load(self, index: int, resultant_load_n: float) -> bool:
        load = float(resultant_load_n)
        if not math.isfinite(load) or load < 0.0:
            raise ValueError("resultant_load_n must be finite and non-negative")
        deployment = self.deployments[index]
        if load <= self.provisional_pullout_threshold_n:
            return False
        _remove_paths(self.stage, deployment["attachment_paths"])
        deployment["pulled_out"] = True
        deployment["pullout_load_n"] = load
        return True

    def reset(self) -> None:
        for item in self.deployments:
            path = item.get("prim_path")
            if path and self.stage.GetPrimAtPath(path).IsValid():
                self.stage.RemovePrim(path)
        self.deployments.clear()


@dataclass
class AdhesiveBondController:
    stage: Any
    left_tissue_path: str
    right_tissue_path: str
    provisional_fresh_failure_n: float = 2.0
    provisional_cured_failure_n: float = 12.0
    beads: list[dict[str, Any]] = field(default_factory=list)

    def deposit(
        self,
        *,
        prim_path: str,
        translation_m: tuple[float, float, float],
        orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ) -> dict[str, Any]:
        root = _spawn_reference(
            self.stage,
            prim_path,
            adhesive_bead_usd(),
            translation_m=translation_m,
            orientation_wxyz=orientation_wxyz,
            state="fresh",
        )
        attachment_root = f"{prim_path}/RuntimeAttachments"
        self.stage.DefinePrim(attachment_root, "Scope")
        paths = [
            _attachment(
                self.stage,
                f"{attachment_root}/LeftInitialTack",
                self.left_tissue_path,
                f"{prim_path}/Collisions/LeftBondVolume",
                overlap_offset_m=0.0015,
            ),
            _attachment(
                self.stage,
                f"{attachment_root}/RightInitialTack",
                self.right_tissue_path,
                f"{prim_path}/Collisions/RightBondVolume",
                overlap_offset_m=0.0015,
            ),
        ]
        bead = {
            "prim_path": str(root.GetPath()),
            "cure_fraction": 0.0,
            "stage": "fresh",
            "attachment_paths": paths,
        }
        self.beads.append(bead)
        return bead

    def set_cure_fraction(self, bead_index: int, cure_fraction: float) -> dict[str, Any]:
        fraction = min(max(float(cure_fraction), 0.0), 1.0)
        bead = self.beads[bead_index]
        root = bead["prim_path"]
        attachment_root = f"{root}/RuntimeAttachments"
        if fraction >= 0.5 and bead["stage"] == "fresh":
            for side, tissue in (
                ("Left", self.left_tissue_path),
                ("Right", self.right_tissue_path),
            ):
                bead["attachment_paths"].append(
                    _attachment(
                        self.stage,
                        f"{attachment_root}/{side}LeadingCure",
                        tissue,
                        f"{root}/Collisions/{side}BondCureLeading",
                        overlap_offset_m=0.0015,
                    )
                )
            bead["stage"] = "leading_cure"
        if fraction >= 1.0 and bead["stage"] != "cured":
            for side, tissue in (
                ("Left", self.left_tissue_path),
                ("Right", self.right_tissue_path),
            ):
                bead["attachment_paths"].append(
                    _attachment(
                        self.stage,
                        f"{attachment_root}/{side}TrailingCure",
                        tissue,
                        f"{root}/Collisions/{side}BondCureTrailing",
                        overlap_offset_m=0.0015,
                    )
                )
            prim = self.stage.GetPrimAtPath(root)
            prim.GetVariantSets().GetVariantSet("state").SetVariantSelection("cured")
            bead["stage"] = "cured"
        bead["cure_fraction"] = fraction
        bead["attachment_count"] = len(bead["attachment_paths"])
        return bead

    def report_load(self, bead_index: int, resultant_load_n: float) -> bool:
        load = float(resultant_load_n)
        if not math.isfinite(load) or load < 0.0:
            raise ValueError("resultant_load_n must be finite and non-negative")
        bead = self.beads[bead_index]
        threshold = (
            self.provisional_fresh_failure_n
            + bead["cure_fraction"]
            * (
                self.provisional_cured_failure_n
                - self.provisional_fresh_failure_n
            )
        )
        if load <= threshold:
            return False
        _remove_paths(self.stage, bead["attachment_paths"])
        bead["failed"] = True
        bead["failure_load_n"] = load
        return True

    def reset(self) -> None:
        for bead in self.beads:
            path = bead.get("prim_path")
            if path and self.stage.GetPrimAtPath(path).IsValid():
                self.stage.RemovePrim(path)
        self.beads.clear()


@dataclass
class ClosureSequenceController:
    """Own the discrete PhysX events around articulation joint targets."""

    stage: Any
    tool_path: str
    left_tissue_path: str
    right_tissue_path: str
    phase: ClosurePhase = ClosurePhase.READY
    capture_attachments: tuple[str, ...] = ()
    staple_retention: StapleRetentionController = field(init=False)
    adhesive_bonds: AdhesiveBondController = field(init=False)

    def __post_init__(self) -> None:
        self.staple_retention = StapleRetentionController(self.stage)
        self.adhesive_bonds = AdhesiveBondController(
            self.stage,
            self.left_tissue_path,
            self.right_tissue_path,
        )

    def targets(self) -> dict[str, float]:
        return closure_phase_targets(self.phase)

    def capture(self) -> tuple[str, ...]:
        if not self.capture_attachments:
            self.capture_attachments = capture_tissue_edges(
                self.stage,
                tool_path=self.tool_path,
                left_tissue_path=self.left_tissue_path,
                right_tissue_path=self.right_tissue_path,
            )
        self.phase = ClosurePhase.CAPTURE
        return self.capture_attachments

    def release_capture(self) -> None:
        release_tissue_capture(self.stage, self.capture_attachments)
        self.capture_attachments = ()
        self.phase = ClosurePhase.RELEASE

    def reset(self) -> None:
        release_tissue_capture(self.stage, self.capture_attachments)
        self.capture_attachments = ()
        self.staple_retention.reset()
        self.adhesive_bonds.reset()
        self.phase = ClosurePhase.READY

    def snapshot(self) -> dict[str, Any]:
        attachment_paths = [
            *self.capture_attachments,
            *(
                path
                for deployment in self.staple_retention.deployments
                for path in deployment.get("attachment_paths", ())
            ),
            *(
                path
                for bead in self.adhesive_bonds.beads
                for path in bead.get("attachment_paths", ())
            ),
        ]
        adhesive_attachments = sum(
            len(bead.get("attachment_paths", ()))
            for bead in self.adhesive_bonds.beads
        )
        return {
            "schema": "dr.anmar.closure-robot-runtime.v1",
            "phase": self.phase.value,
            "capture_attachment_count": len(self.capture_attachments),
            "formed_staple_count": len(self.staple_retention.deployments),
            "staple_attachment_count": sum(
                len(item.get("attachment_paths", ()))
                for item in self.staple_retention.deployments
            ),
            "adhesive_bead_count": len(self.adhesive_bonds.beads),
            "adhesive_bond_attachment_count": adhesive_attachments,
            **_attachment_evidence(self.stage, attachment_paths),
            "transform_writes": False,
            "clinical_validation": False,
        }


def frame_path(root_prim_path: str, name: str) -> str:
    frames = {
        "closure_tcp": "Links/Mount/Frames/closure_tcp",
        "wound_center": "Links/Mount/Frames/wound_center",
        "left_tissue_capture": "Links/LeftClamp/Frames/left_tissue_capture",
        "right_tissue_capture": "Links/RightClamp/Frames/right_tissue_capture",
        "staple_exit": "Links/Mount/Frames/staple_exit",
        "adhesive_tip": "Links/AdhesiveCarriage/Frames/adhesive_tip",
    }
    try:
        suffix = frames[name]
    except KeyError as exc:
        raise ValueError(f"Unknown closure-robot frame: {name}") from exc
    return f"{root_prim_path.rstrip('/')}/{suffix}"


class ClosureRobotAssets:
    TOOL_PAYLOAD = (
        "Props/SurgicalClosure/ClosureRobot/"
        "dranmar_closure_tool_payload.usda"
    )
    TOOL_STANDALONE = (
        "Props/SurgicalClosure/ClosureRobot/"
        "dranmar_closure_tool_standalone.usda"
    )
    TOOL_RIGID_PROXY = (
        "Props/SurgicalClosure/ClosureRobot/"
        "dranmar_closure_tool_rigid_proxy.usda"
    )
    TISSUE_DEMO = (
        "Props/SurgicalClosure/ClosureRobot/"
        "dranmar_closure_tissue_demo.usda"
    )
    FORMED_STAPLE = (
        "Props/SurgicalClosure/ClosureRobot/"
        "dranmar_closure_staple.usda"
    )
    ADHESIVE_BEAD = (
        "Props/SurgicalClosure/ClosureRobot/"
        "dranmar_closure_adhesive_bead.usda"
    )


__all__ = [
    "APP_VERSION",
    "ASSET_ROOT_ENV",
    "AdhesiveBondController",
    "CATALOG_SUBPATH",
    "CLOSURE_JOINT_NAMES",
    "ClosurePhase",
    "ClosureRobotAssets",
    "ClosureSequenceController",
    "FRANKA_REMOVED_PRIM_NAMES",
    "MOUNT_ROTATION_WXYZ",
    "OPEN_TARGETS",
    "StapleRetentionController",
    "adhesive_bead_usd",
    "anchor_tissue_outer_edges",
    "apply_tissue_demo_surface_deformables",
    "asset_root",
    "capture_tissue_edges",
    "closure_phase_targets",
    "deploy_formed_staple",
    "formed_staple_usd",
    "frame_path",
    "make_franka_closure_robot_cfg",
    "make_tool_cfg",
    "payload_usd",
    "release_tissue_capture",
    "rigid_proxy_usd",
    "set_closure_phase_target",
    "set_joint_targets",
    "spawn_tissue_demo",
    "standalone_tool_usd",
    "tensor_value",
    "tissue_demo_usd",
]
