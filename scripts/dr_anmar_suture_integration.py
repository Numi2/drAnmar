"""Shared installation contract for the Dr.Anmar needle-suture instrument."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from dr_anmar_needle_model import derive_needle, load_needle_profile, sample_episode_parameters

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DR_ANMAR_NEEDLE_NAME = "DrAnmar Needle"
DR_ANMAR_NEEDLE_ASSET_ID = "dr-anmar-needle"
DR_ANMAR_NEEDLE_ASSET_VERSION = "1.16.0"
DR_ANMAR_NEEDLE_ROOT_PRIM = "DrAnmarNeedle"
DR_ANMAR_ASSET_ROOT = REPOSITORY_ROOT / "assets/dr_anmar"
SUTURE_ASSET_PATH = DR_ANMAR_ASSET_ROOT / "suture/DrAnmarSuture4_0.usda"
SUTURE_BASE_ASSET_PATH = DR_ANMAR_ASSET_ROOT / "suture/DrAnmarSuture4_0_base.usda"
SUTURE_GEOMETRY_ASSET_PATH = DR_ANMAR_ASSET_ROOT / "suture/DrAnmarSuture4_0_geometry.usd"
SUTURE_MATERIALS_ASSET_PATH = DR_ANMAR_ASSET_ROOT / "suture/DrAnmarSuture4_0_materials.usda"
SUTURE_NORMAL_ROUGHNESS_TEXTURE_PATH = (
    DR_ANMAR_ASSET_ROOT / "suture/textures/DrAnmarSuture4_0_braid_normal_roughness.png"
)
SUTURE_PHYSICS_ASSET_PATH = DR_ANMAR_ASSET_ROOT / "suture/DrAnmarSuture4_0_physics.usda"
SUTURE_PHYSX_ASSET_PATH = DR_ANMAR_ASSET_ROOT / "suture/DrAnmarSuture4_0_physx.usda"
DR_ANMAR_NEEDLE_ASSET_PATH = DR_ANMAR_ASSET_ROOT / "needle/DrAnmarNeedle.usda"
DR_ANMAR_NEEDLE_BASE_ASSET_PATH = DR_ANMAR_ASSET_ROOT / "needle/DrAnmarNeedle_base.usda"
DR_ANMAR_NEEDLE_GEOMETRY_ASSET_PATH = DR_ANMAR_ASSET_ROOT / "needle/DrAnmarNeedle_geometry.usd"
DR_ANMAR_NEEDLE_MATERIALS_ASSET_PATH = DR_ANMAR_ASSET_ROOT / "needle/DrAnmarNeedle_materials.usda"
DR_ANMAR_NEEDLE_PHYSICS_ASSET_PATH = DR_ANMAR_ASSET_ROOT / "needle/DrAnmarNeedle_physics.usda"
DR_ANMAR_NEEDLE_PHYSX_ASSET_PATH = DR_ANMAR_ASSET_ROOT / "needle/DrAnmarNeedle_physx.usda"
SUTURE_NEEDLE_INTERFACE_CENTER_M = (-0.00025, 0.0, 0.0)

# A 90-degree yaw keeps the 180 mm strand inside the shared PSM workspace.
# The assembly is deliberately an additional sterile-table instrument: it
# never replaces the room's existing task object or task-specific thread.
SUTURE_LANDING_POSITION_M = (-0.080, -0.105, 0.0030)
SUTURE_LANDING_ROTATION_WXYZ = (
    0.7071067811865476,
    0.0,
    0.0,
    0.7071067811865475,
)


@dataclass(frozen=True)
class NeedleEpisodeMassProperties:
    mass_kg: float
    mass_scale: float
    center_of_mass_m: tuple[float, float, float]
    diagonal_inertia_kg_m2: tuple[float, float, float]
    principal_axes_wxyz: tuple[float, float, float, float]


def validate_source_assets() -> None:
    """Fail early if the composed instrument cannot be trusted or loaded."""

    missing = [
        str(path)
        for path in (
            SUTURE_ASSET_PATH,
            SUTURE_BASE_ASSET_PATH,
            SUTURE_GEOMETRY_ASSET_PATH,
            SUTURE_MATERIALS_ASSET_PATH,
            SUTURE_NORMAL_ROUGHNESS_TEXTURE_PATH,
            SUTURE_PHYSICS_ASSET_PATH,
            SUTURE_PHYSX_ASSET_PATH,
            DR_ANMAR_NEEDLE_ASSET_PATH,
            DR_ANMAR_NEEDLE_BASE_ASSET_PATH,
            DR_ANMAR_NEEDLE_GEOMETRY_ASSET_PATH,
            DR_ANMAR_NEEDLE_MATERIALS_ASSET_PATH,
            DR_ANMAR_NEEDLE_PHYSICS_ASSET_PATH,
            DR_ANMAR_NEEDLE_PHYSX_ASSET_PATH,
        )
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError("The Dr.Anmar needle-suture instrument is incomplete. Missing " + ", ".join(missing))


def configure_dr_anmar_needle(
    scene_cfg: Any,
    *,
    asset_base_cfg_type: Any,
    usd_file_cfg_type: Any,
) -> dict[str, Any]:
    """Add the composed instrument without replacing any task-owned entity."""

    validate_source_assets()
    if getattr(scene_cfg, "dr_anmar_needle", None) is not None:
        raise RuntimeError(f"{DR_ANMAR_NEEDLE_NAME} was configured twice")
    scene_cfg.dr_anmar_needle = asset_base_cfg_type(
        prim_path=f"{{ENV_REGEX_NS}}/{DR_ANMAR_NEEDLE_ROOT_PRIM}",
        init_state=asset_base_cfg_type.InitialStateCfg(
            pos=SUTURE_LANDING_POSITION_M,
            rot=SUTURE_LANDING_ROTATION_WXYZ,
        ),
        spawn=usd_file_cfg_type(usd_path=str(DR_ANMAR_NEEDLE_ASSET_PATH)),
    )
    return {
        "name": DR_ANMAR_NEEDLE_NAME,
        "asset_id": DR_ANMAR_NEEDLE_ASSET_ID,
        "asset_version": DR_ANMAR_NEEDLE_ASSET_VERSION,
        "asset": str(DR_ANMAR_NEEDLE_ASSET_PATH),
        "suture_entry_layer": str(SUTURE_ASSET_PATH),
        "suture_base_layer": str(SUTURE_BASE_ASSET_PATH),
        "suture_geometry_layer": str(SUTURE_GEOMETRY_ASSET_PATH),
        "suture_materials_layer": str(SUTURE_MATERIALS_ASSET_PATH),
        "suture_normal_roughness_texture": str(SUTURE_NORMAL_ROUGHNESS_TEXTURE_PATH),
        "suture_physics_layer": str(SUTURE_PHYSICS_ASSET_PATH),
        "suture_physx_layer": str(SUTURE_PHYSX_ASSET_PATH),
        "geometry_layer": str(DR_ANMAR_NEEDLE_GEOMETRY_ASSET_PATH),
        "base_layer": str(DR_ANMAR_NEEDLE_BASE_ASSET_PATH),
        "materials_layer": str(DR_ANMAR_NEEDLE_MATERIALS_ASSET_PATH),
        "physics_layer": str(DR_ANMAR_NEEDLE_PHYSICS_ASSET_PATH),
        "physx_layer": str(DR_ANMAR_NEEDLE_PHYSX_ASSET_PATH),
        "physics_variant_set": "Physics",
        "physics_variant_choices": ["none", "physics", "physx"],
        "default_physics_variant": "physx",
        "prim_path": f"/World/envs/env_0/{DR_ANMAR_NEEDLE_ROOT_PRIM}",
        "landing_position_m": list(SUTURE_LANDING_POSITION_M),
        "landing_rotation_wxyz": list(SUTURE_LANDING_ROTATION_WXYZ),
        "needle_geometry_provenance": "independent_DrAnmar_parametric_geometry",
        "suture_physics_provenance": "independent_DrAnmar_4_0_model",
        "task_object_replaced": False,
        "current_thread_replaced": False,
    }


def apply_dr_anmar_needle_episode_domain(
    stage: Any,
    *,
    seed: int,
    root_path: str = "/World/envs/env_0/DrAnmarNeedle",
) -> dict[str, Any]:
    """Apply replayable, bounded sim-to-real randomization to the live asset."""

    from pxr import Gf, UsdPhysics

    profile = load_needle_profile()
    parameters = sample_episode_parameters(profile, seed)
    mass_properties = needle_mass_properties_for_mass(
        profile,
        parameters.mass_kg,
    )
    needle_prim = stage.GetPrimAtPath(f"{root_path}/Needle")
    material_prim = stage.GetPrimAtPath(f"{root_path}/Looks/NeedleSteelPhysics")
    shader_prim = stage.GetPrimAtPath(f"{root_path}/Looks/NeedleSteelVisual/PreviewSurface")
    if not needle_prim.IsValid() or not material_prim.IsValid() or not shader_prim.IsValid():
        raise RuntimeError(f"{DR_ANMAR_NEEDLE_NAME} episode domain is missing")
    mass_api = UsdPhysics.MassAPI(needle_prim)
    mass_api.GetMassAttr().Set(parameters.mass_kg)
    mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(*mass_properties.center_of_mass_m))
    mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*mass_properties.diagonal_inertia_kg_m2))
    principal_axes = mass_properties.principal_axes_wxyz
    mass_api.GetPrincipalAxesAttr().Set(
        Gf.Quatf(
            principal_axes[0],
            Gf.Vec3f(*principal_axes[1:]),
        )
    )
    material_api = UsdPhysics.MaterialAPI(material_prim)
    material_api.GetStaticFrictionAttr().Set(parameters.static_friction)
    material_api.GetDynamicFrictionAttr().Set(parameters.dynamic_friction)
    material_api.GetRestitutionAttr().Set(parameters.restitution)
    roughness = shader_prim.GetAttribute("inputs:roughness")
    if roughness.IsValid():
        roughness.Set(parameters.surface_roughness)
    needle_prim.SetCustomDataByKey("drAnmarEpisodeSeed", parameters.seed)
    payload: dict[str, Any] = parameters.payload()
    payload.update(
        {
            "center_of_mass_m": list(mass_properties.center_of_mass_m),
            "diagonal_inertia_kg_m2": list(mass_properties.diagonal_inertia_kg_m2),
            "principal_axes_wxyz": list(mass_properties.principal_axes_wxyz),
        }
    )
    return payload


def needle_mass_properties_for_mass(
    profile: dict[str, Any],
    target_mass_kg: float,
) -> NeedleEpisodeMassProperties:
    """Scale density-dependent inertia without changing needle geometry."""

    derived = derive_needle(profile)
    target_mass = float(target_mass_kg)
    if not math.isfinite(target_mass) or target_mass <= 0.0:
        raise ValueError("needle target mass must be positive")
    mass_scale = target_mass / derived.mass_kg
    baseline_inertia = derived.mass_properties.diagonal_inertia_kg_m2
    return NeedleEpisodeMassProperties(
        mass_kg=target_mass,
        mass_scale=mass_scale,
        center_of_mass_m=derived.mass_properties.center_of_mass_m,
        diagonal_inertia_kg_m2=(
            baseline_inertia[0] * mass_scale,
            baseline_inertia[1] * mass_scale,
            baseline_inertia[2] * mass_scale,
        ),
        principal_axes_wxyz=derived.mass_properties.principal_axes_wxyz,
    )


def local_room_ids(rooms: Iterable[dict[str, Any]]) -> tuple[str, ...]:
    """Return rooms whose Isaac scene is constructed by this repository."""

    return tuple(str(room["id"]) for room in rooms if not room.get("external_provider"))
