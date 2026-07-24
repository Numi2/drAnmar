# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Reachable default landings for Dr.Anmar operating-room assets."""

from __future__ import annotations


# Both PSMs share this compact table region after their roots are narrowed.
# Asset-specific Z values preserve the authored table contact height.
OPERATIVE_CENTER_M = (-0.040, -0.030, 0.0)
MAX_HORIZONTAL_OFFSET_M = 0.075

_LANDINGS_M: dict[str, tuple[float, float, float]] = {
    "skin_stapler": (0.015, 0.020, 0.0140),
    "skin_adhesive_applicator": (-0.090, 0.020, 0.0180),
    "skin_adhesive_cap": (-0.010, 0.020, 0.0180),
    "skin_adhesive_bead": (0.010, -0.035, 0.0010),
    "dr_anmar_needle": (-0.005, 0.015, 0.0012),
    "dr_anmar_needle_v030": (-0.070, 0.015, 0.0012),
    "dr_anmar_needle_thread_coiled": (-0.075, -0.055, 0.0012),
    "dr_anmar_needle_thread_extended": (0.005, -0.070, 0.0012),
    "dr_anmar_needle_thread_proxy": (-0.025, -0.055, 0.0012),
    "dr_anmar_needle_suture": (-0.060, -0.045, 0.0030),
    "dr_anmar_tissue": (-0.040, -0.055, 0.0040),
    "vascular_clip": (-0.080, -0.050, 0.0003),
    "laparotomy_sponge": (0.005, -0.055, 0.0141),
}


def asset_landing(asset_id: str) -> tuple[float, float, float]:
    """Return a table landing guaranteed to remain near the operative center."""

    try:
        landing = _LANDINGS_M[asset_id]
    except KeyError as error:
        raise ValueError(f"Unknown Dr.Anmar bench asset landing: {asset_id}") from error
    horizontal_offset_squared = (
        (landing[0] - OPERATIVE_CENTER_M[0]) ** 2
        + (landing[1] - OPERATIVE_CENTER_M[1]) ** 2
    )
    if horizontal_offset_squared > MAX_HORIZONTAL_OFFSET_M**2:
        raise RuntimeError(
            f"Dr.Anmar bench asset {asset_id} is outside the shared PSM workspace"
        )
    return landing


def landing_manifest() -> dict[str, tuple[float, float, float]]:
    """Expose a copy for diagnostics without allowing runtime mutation."""

    return dict(_LANDINGS_M)
