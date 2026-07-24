"""Dr.Anmar needle and attached-thread catalog helpers.

The segmented needle-thread assemblies use maximal-coordinate D6 joints, so
they are spawned as raw OpenUSD assets instead of reduced-coordinate Isaac Lab
articulations.  The rigid proxy remains available for perception, handover and
dataset workflows that need one tensor-managed rigid body.
"""

from __future__ import annotations

from pathlib import Path

from . import ORBITSURGICAL_ASSETS_DATA_DIR


CATALOG_ROOT = Path(ORBITSURGICAL_ASSETS_DATA_DIR) / "Props/SurgicalClosure"
NEEDLE_ROOT = CATALOG_ROOT / "Needle"
NEEDLE_THREAD_ROOT = CATALOG_ROOT / "NeedleThread"


class DrAnmarNeedleAssets:
    """I4H-compatible relative paths for the Dr.Anmar needle system."""

    NEEDLE = "Props/SurgicalClosure/Needle/dranmar_needle.usda"
    NEEDLE_THREAD_COILED = (
        "Props/SurgicalClosure/NeedleThread/dranmar_needle_thread.usda"
    )
    NEEDLE_THREAD_EXTENDED = (
        "Props/SurgicalClosure/NeedleThread/dranmar_needle_thread_extended.usda"
    )
    NEEDLE_THREAD_RIGID_PROXY = (
        "Props/SurgicalClosure/NeedleThread/"
        "dranmar_needle_thread_rigid_proxy.usda"
    )


def needle_usd() -> Path:
    return NEEDLE_ROOT / "dranmar_needle.usda"


def needle_thread_usd(configuration: str = "coiled") -> Path:
    normalized = str(configuration).strip().lower()
    filenames = {
        "coiled": "dranmar_needle_thread.usda",
        "extended": "dranmar_needle_thread_extended.usda",
        "rigid_proxy": "dranmar_needle_thread_rigid_proxy.usda",
    }
    try:
        return NEEDLE_THREAD_ROOT / filenames[normalized]
    except KeyError as exc:
        raise ValueError(
            "configuration must be coiled, extended, or rigid_proxy"
        ) from exc


def frame_path(frame_name: str, *, assembly: bool = True) -> str:
    """Return the authored frame prim path for a composed asset."""

    name = str(frame_name).strip()
    if not name or "/" in name:
        raise ValueError("frame_name must be one authored frame name")
    root = "DrAnmarNeedleThread" if assembly else "DrAnmarNeedle"
    return f"/{root}/Frames/{name}"


def make_needle_cfg(
    *,
    prim_path: str = "{ENV_REGEX_NS}/DrAnmarNeedleV030",
    usd_path: str | Path | None = None,
):
    """Create a tensor-managed rigid standalone needle configuration."""

    import isaaclab.sim as sim_utils  # type: ignore
    from isaaclab.assets import RigidObjectCfg  # type: ignore

    selected = needle_usd() if usd_path is None else Path(usd_path).expanduser()
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(selected.resolve()),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.195, 0.015, 0.0012),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def make_needle_thread_rigid_proxy_cfg(
    *,
    prim_path: str = "{ENV_REGEX_NS}/DrAnmarNeedleThreadProxy",
    usd_path: str | Path | None = None,
):
    """Create the stable rigid needle-thread perception proxy."""

    import isaaclab.sim as sim_utils  # type: ignore
    from isaaclab.assets import RigidObjectCfg  # type: ignore

    selected = (
        needle_thread_usd("rigid_proxy")
        if usd_path is None
        else Path(usd_path).expanduser()
    )
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(selected.resolve()),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.060, -0.120, 0.0012),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def make_segmented_needle_thread_cfg(
    *,
    configuration: str = "coiled",
    prim_path: str = "{ENV_REGEX_NS}/DrAnmarNeedleThread",
    usd_path: str | Path | None = None,
):
    """Create a raw maximal-coordinate needle-thread assembly configuration."""

    normalized = str(configuration).strip().lower()
    if normalized not in {"coiled", "extended"}:
        raise ValueError("segmented configuration must be coiled or extended")

    import isaaclab.sim as sim_utils  # type: ignore
    from isaaclab.assets import AssetBaseCfg  # type: ignore

    selected = (
        needle_thread_usd(normalized)
        if usd_path is None
        else Path(usd_path).expanduser()
    )
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(usd_path=str(selected.resolve())),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                (-0.055, -0.120, 0.0012)
                if normalized == "coiled"
                else (0.075, -0.180, 0.0012)
            ),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


spawn_segmented_needle_thread = make_segmented_needle_thread_cfg


__all__ = [
    "CATALOG_ROOT",
    "DrAnmarNeedleAssets",
    "NEEDLE_ROOT",
    "NEEDLE_THREAD_ROOT",
    "frame_path",
    "make_needle_cfg",
    "make_needle_thread_rigid_proxy_cfg",
    "make_segmented_needle_thread_cfg",
    "needle_thread_usd",
    "needle_usd",
    "spawn_segmented_needle_thread",
]
