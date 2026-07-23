#!/usr/bin/env python3
"""Shared installation contract for the Dr.Anmar needle-suture instrument."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SUTURE_ASSET_PATH = (
    REPOSITORY_ROOT
    / "source/extensions/orbit.surgical.assets/data/Props/DrAnmarSuture/DrAnmarSuture4_0.usda"
)
SUTURE_ASSEMBLY_PATH = SUTURE_ASSET_PATH.with_name("DrAnmarNeedleSuture4_0.usda")
NEEDLE_ASSET_PATH = (
    REPOSITORY_ROOT
    / "source/extensions/orbit.surgical.assets/data/Props/Surgical_needle/needle_sdf.usd"
)
NEEDLE_ASSET_SHA256 = "2b317a61f93631a7192e7ed2839ef20f7a75c05aa5f84a3905696134a64f36d7"
NEEDLE_UNIFORM_SCALE = 0.4
NEEDLE_SWAGE_ANCHOR_M = (0.0478657183, 0.0491908647, 0.0009574010)
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_assets() -> None:
    """Fail early if the composed instrument cannot be trusted or loaded."""

    missing = [
        str(path)
        for path in (SUTURE_ASSET_PATH, SUTURE_ASSEMBLY_PATH, NEEDLE_ASSET_PATH)
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            "The Dr.Anmar needle-suture instrument is incomplete. Missing "
            + ", ".join(missing)
        )
    needle_digest = sha256(NEEDLE_ASSET_PATH)
    if needle_digest != NEEDLE_ASSET_SHA256:
        raise RuntimeError(
            "The pinned ORBIT needle mesh changed; re-derive and validate its "
            "factory-swage anchor before use"
        )


def configure_suture_instrument(
    scene_cfg: Any,
    *,
    asset_base_cfg_type: Any,
    usd_file_cfg_type: Any,
) -> dict[str, Any]:
    """Add the composed instrument without replacing any task-owned entity."""

    validate_source_assets()
    if getattr(scene_cfg, "dr_anmar_suture", None) is not None:
        raise RuntimeError("The Dr.Anmar needle-suture instrument was configured twice")
    scene_cfg.dr_anmar_suture = asset_base_cfg_type(
        prim_path="{ENV_REGEX_NS}/DrAnmarNeedleSuture4_0",
        init_state=asset_base_cfg_type.InitialStateCfg(
            pos=SUTURE_LANDING_POSITION_M,
            rot=SUTURE_LANDING_ROTATION_WXYZ,
        ),
        spawn=usd_file_cfg_type(usd_path=str(SUTURE_ASSEMBLY_PATH)),
    )
    return {
        "asset": str(SUTURE_ASSEMBLY_PATH),
        "prim_path": "/World/envs/env_0/DrAnmarNeedleSuture4_0",
        "landing_position_m": list(SUTURE_LANDING_POSITION_M),
        "landing_rotation_wxyz": list(SUTURE_LANDING_ROTATION_WXYZ),
        "task_object_replaced": False,
        "current_thread_replaced": False,
    }


def local_room_ids(rooms: Iterable[dict[str, Any]]) -> tuple[str, ...]:
    """Return rooms whose Isaac scene is constructed by this repository."""

    return tuple(
        str(room["id"])
        for room in rooms
        if not room.get("external_provider")
    )
