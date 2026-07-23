#!/usr/bin/env python3
"""Shared installation contract for the Dr.Anmar needle-suture instrument."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from dr_anmar_needle_model import load_needle_profile, sample_episode_parameters


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DR_ANMAR_NEEDLE_NAME = "DrAnmar Needle"
DR_ANMAR_NEEDLE_ASSET_ID = "dr-anmar-needle"
DR_ANMAR_NEEDLE_ASSET_VERSION = "1.0.0"
DR_ANMAR_NEEDLE_ROOT_PRIM = "DrAnmarNeedle"
DR_ANMAR_ASSET_ROOT = REPOSITORY_ROOT / "assets/dr_anmar"
SUTURE_ASSET_PATH = DR_ANMAR_ASSET_ROOT / "suture/DrAnmarSuture4_0.usda"
DR_ANMAR_NEEDLE_ASSET_PATH = (
    DR_ANMAR_ASSET_ROOT / "needle/DrAnmarNeedle.usda"
)
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


def validate_source_assets() -> None:
    """Fail early if the composed instrument cannot be trusted or loaded."""

    missing = [
        str(path)
        for path in (
            SUTURE_ASSET_PATH,
            DR_ANMAR_NEEDLE_ASSET_PATH,
        )
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            "The Dr.Anmar needle-suture instrument is incomplete. Missing "
            + ", ".join(missing)
        )


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
) -> dict[str, float | int]:
    """Apply replayable, bounded sim-to-real randomization to the live asset."""

    from pxr import UsdPhysics

    parameters = sample_episode_parameters(load_needle_profile(), seed)
    needle_prim = stage.GetPrimAtPath(f"{root_path}/Needle")
    material_prim = stage.GetPrimAtPath(
        f"{root_path}/Materials/NeedleSteel"
    )
    shader_prim = stage.GetPrimAtPath(
        f"{root_path}/Materials/NeedleSteel/PreviewSurface"
    )
    if (
        not needle_prim.IsValid()
        or not material_prim.IsValid()
        or not shader_prim.IsValid()
    ):
        raise RuntimeError(f"{DR_ANMAR_NEEDLE_NAME} episode domain is missing")
    UsdPhysics.MassAPI(needle_prim).GetMassAttr().Set(parameters.mass_kg)
    material_api = UsdPhysics.MaterialAPI(material_prim)
    material_api.GetStaticFrictionAttr().Set(parameters.static_friction)
    material_api.GetDynamicFrictionAttr().Set(parameters.dynamic_friction)
    material_api.GetRestitutionAttr().Set(parameters.restitution)
    roughness = shader_prim.GetAttribute("inputs:roughness")
    if roughness.IsValid():
        roughness.Set(parameters.surface_roughness)
    needle_prim.SetCustomDataByKey("drAnmarEpisodeSeed", parameters.seed)
    return parameters.payload()


def local_room_ids(rooms: Iterable[dict[str, Any]]) -> tuple[str, ...]:
    """Return rooms whose Isaac scene is constructed by this repository."""

    return tuple(
        str(room["id"])
        for room in rooms
        if not room.get("external_provider")
    )
