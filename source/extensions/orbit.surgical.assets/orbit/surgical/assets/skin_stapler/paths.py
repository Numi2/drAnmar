"""Portable paths for the repository-local DrAnmar skin stapler."""
from __future__ import annotations

import os
from pathlib import Path

CATALOG_SUBPATH = Path("Props/SurgicalClosure/SkinStapler")
EXTENSION_ROOT = Path(__file__).resolve().parents[4]
ASSET_DATA_ROOT = EXTENSION_ROOT / "data"
_ENV = "DRANMAR_SKIN_STAPLER_ASSET_ROOT"


def project_root() -> Path:
    """Return the ``orbit.surgical.assets`` extension root."""

    return EXTENSION_ROOT


def asset_root(*, require: bool = True) -> Path:
    override = os.environ.get(_ENV)
    candidates = []
    if override:
        candidates.append(Path(override).expanduser())
    candidates.extend(
        [
            ASSET_DATA_ROOT / CATALOG_SUBPATH,
        ]
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    if require:
        rendered = "\n".join(f"- {candidate}" for candidate in candidates)
        raise FileNotFoundError(
            f"DrAnmar skin stapler asset root was not found. Checked:\n{rendered}\n"
            f"Set {_ENV} to a catalog extraction if using the helper-only wheel."
        )
    return candidates[0]


def articulated_usd() -> Path:
    return asset_root() / "skin_stapler_articulated.usda"


def rigid_proxy_usd() -> Path:
    return asset_root() / "skin_stapler_rigid_proxy.usda"


def staple_usd() -> Path:
    return asset_root() / "skin_staple.usda"


def physics_profile() -> Path:
    return asset_root() / "physics_profile.json"


def interaction_frames() -> Path:
    return asset_root() / "interaction_frames.json"


try:
    from i4h_asset_helper import BaseI4HAssets as _BaseI4HAssets
except (ImportError, ModuleNotFoundError):
    _BaseI4HAssets = object


class SurgicalClosureAssets(_BaseI4HAssets):
    """I4H-compatible relative paths for DrAnmar closure assets."""

    SKIN_STAPLER_ARTICULATED = (
        "Props/SurgicalClosure/SkinStapler/skin_stapler_articulated.usda"
    )
    SKIN_STAPLER_RIGID_PROXY = (
        "Props/SurgicalClosure/SkinStapler/skin_stapler_rigid_proxy.usda"
    )
    SKIN_STAPLE = "Props/SurgicalClosure/SkinStapler/skin_staple.usda"
