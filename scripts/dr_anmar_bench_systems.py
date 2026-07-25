# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Selectable Dr.Anmar robot-system stations for the NVIDIA surgical bench."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

FEATURED_ROBOT_EXCLUSIVE_GROUP = "featured_robot_system"
# The standalone mechanisms are authored upward from their mount interfaces.
# This side station keeps the largest system inside the NVIDIA table footprint
# and out of the two PSMs' neutral shared handoff corridor.
FEATURED_ROBOT_POSITION_M = (0.08, 0.08, 0.015)
FEATURED_SUBSTRATE_POSITION_M = (0.08, 0.08, 0.001)


BENCH_ROBOT_SYSTEM_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "id": "wound_preparation_robot",
        "title": "Wound preparation robot",
        "description": (
            "Articulated inspection, irrigation, aspiration and debridement "
            "system with its wound-bed procedure substrate."
        ),
        "path": (
            "Props/SurgicalPreparation/WoundPreparationRobot/"
            "dranmar_wound_preparation_tool_standalone.usda"
        ),
        "payload_path": (
            "Props/SurgicalPreparation/WoundPreparationRobot/"
            "dranmar_wound_preparation_tool_payload.usda"
        ),
        "rigid_proxy_path": (
            "Props/SurgicalPreparation/WoundPreparationRobot/"
            "dranmar_wound_preparation_tool_rigid_proxy.usda"
        ),
        "auxiliary_path": (
            "Props/SurgicalPreparation/WoundPreparationRobot/"
            "dranmar_wound_bed_demo.usda"
        ),
        "provider": "dr_anmar",
        "catalog_section": "robot_systems",
        "bench_kind": "robot_system",
        "exclusive_group": FEATURED_ROBOT_EXCLUSIVE_GROUP,
        "representation": "standalone_articulation_with_task_substrate",
        "state": "inspection_ready",
        "default": False,
    },
    {
        "id": "atraumatic_exposure_robot",
        "title": "Atraumatic exposure robot",
        "description": (
            "Articulated bilateral tissue-capture and force-controlled "
            "retraction system with its exposure substrate."
        ),
        "path": (
            "Props/SurgicalExposure/AtraumaticExposureRobot/"
            "dranmar_atraumatic_exposure_tool_standalone.usda"
        ),
        "payload_path": (
            "Props/SurgicalExposure/AtraumaticExposureRobot/"
            "dranmar_atraumatic_exposure_tool_payload.usda"
        ),
        "rigid_proxy_path": (
            "Props/SurgicalExposure/AtraumaticExposureRobot/"
            "dranmar_atraumatic_exposure_tool_rigid_proxy.usda"
        ),
        "auxiliary_path": (
            "Props/SurgicalExposure/AtraumaticExposureRobot/"
            "dranmar_exposure_tissue_demo.usda"
        ),
        "provider": "dr_anmar",
        "catalog_section": "robot_systems",
        "bench_kind": "robot_system",
        "exclusive_group": FEATURED_ROBOT_EXCLUSIVE_GROUP,
        "representation": "standalone_articulation_with_task_substrate",
        "state": "inspection_ready",
        "default": False,
    },
    {
        "id": "adaptive_hemostasis_robot",
        "title": "Adaptive hemostasis robot",
        "description": (
            "Articulated field-clearing, temporary-control, clip, patch and "
            "seal-verification system with its bleeding-vessel substrate."
        ),
        "path": (
            "Props/SurgicalHemostasis/AdaptiveHemostasisRobot/"
            "dranmar_adaptive_hemostasis_tool_standalone.usda"
        ),
        "payload_path": (
            "Props/SurgicalHemostasis/AdaptiveHemostasisRobot/"
            "dranmar_adaptive_hemostasis_tool_payload.usda"
        ),
        "rigid_proxy_path": (
            "Props/SurgicalHemostasis/AdaptiveHemostasisRobot/"
            "dranmar_adaptive_hemostasis_tool_rigid_proxy.usda"
        ),
        "auxiliary_path": (
            "Props/SurgicalHemostasis/AdaptiveHemostasisRobot/"
            "dranmar_bleeding_vessel_demo.usda"
        ),
        "provider": "dr_anmar",
        "catalog_section": "robot_systems",
        "bench_kind": "robot_system",
        "exclusive_group": FEATURED_ROBOT_EXCLUSIVE_GROUP,
        "representation": "standalone_articulation_with_task_substrate",
        "state": "inspection_ready",
        "default": False,
    },
    {
        "id": "adaptive_anastomosis_robot",
        "title": "Adaptive anastomosis robot",
        "description": (
            "Articulated capture, alignment, staple-ring, reinforcement and "
            "leak-test system with its hollow-tissue substrate."
        ),
        "path": (
            "Props/SurgicalReconstruction/AdaptiveAnastomosisRobot/"
            "dranmar_adaptive_anastomosis_tool_standalone.usda"
        ),
        "payload_path": (
            "Props/SurgicalReconstruction/AdaptiveAnastomosisRobot/"
            "dranmar_adaptive_anastomosis_tool_payload.usda"
        ),
        "rigid_proxy_path": (
            "Props/SurgicalReconstruction/AdaptiveAnastomosisRobot/"
            "dranmar_adaptive_anastomosis_tool_rigid_proxy.usda"
        ),
        "auxiliary_path": (
            "Props/SurgicalReconstruction/AdaptiveAnastomosisRobot/"
            "dranmar_hollow_tissue_demo.usda"
        ),
        "provider": "dr_anmar",
        "catalog_section": "robot_systems",
        "bench_kind": "robot_system",
        "exclusive_group": FEATURED_ROBOT_EXCLUSIVE_GROUP,
        "representation": "standalone_articulation_with_task_substrate",
        "state": "inspection_ready",
        "default": False,
    },
    {
        "id": "adaptive_seal_divide_robot",
        "title": "Adaptive seal-and-divide robot",
        "description": (
            "Articulated centering, compression-sealing, guarded division and "
            "stump-verification system with its vessel substrate."
        ),
        "path": (
            "Props/SurgicalDivision/AdaptiveSealDivideRobot/"
            "dranmar_adaptive_seal_divide_tool_standalone.usda"
        ),
        "payload_path": (
            "Props/SurgicalDivision/AdaptiveSealDivideRobot/"
            "dranmar_adaptive_seal_divide_tool_payload.usda"
        ),
        "rigid_proxy_path": (
            "Props/SurgicalDivision/AdaptiveSealDivideRobot/"
            "dranmar_adaptive_seal_divide_tool_rigid_proxy.usda"
        ),
        "auxiliary_path": (
            "Props/SurgicalDivision/AdaptiveSealDivideRobot/"
            "dranmar_seal_divide_vessel_demo.usda"
        ),
        "provider": "dr_anmar",
        "catalog_section": "robot_systems",
        "bench_kind": "robot_system",
        "exclusive_group": FEATURED_ROBOT_EXCLUSIVE_GROUP,
        "representation": "standalone_articulation_with_task_substrate",
        "state": "inspection_ready",
        "default": False,
    },
    {
        "id": "safeplane_dissection_robot",
        "title": "SafePlane dissection robot",
        "description": (
            "Articulated traction, blunt, hydro, guarded-scissor and energy "
            "dissection system with its protected-structure substrate."
        ),
        "path": (
            "Props/SurgicalDissection/SafePlaneDissectionRobot/"
            "dranmar_safeplane_dissection_tool_standalone.usda"
        ),
        "payload_path": (
            "Props/SurgicalDissection/SafePlaneDissectionRobot/"
            "dranmar_safeplane_dissection_tool_payload.usda"
        ),
        "rigid_proxy_path": (
            "Props/SurgicalDissection/SafePlaneDissectionRobot/"
            "dranmar_safeplane_dissection_tool_rigid_proxy.usda"
        ),
        "auxiliary_path": (
            "Props/SurgicalDissection/SafePlaneDissectionRobot/"
            "dranmar_safeplane_tissue_demo.usda"
        ),
        "provider": "dr_anmar",
        "catalog_section": "robot_systems",
        "bench_kind": "robot_system",
        "exclusive_group": FEATURED_ROBOT_EXCLUSIVE_GROUP,
        "representation": "standalone_articulation_with_task_substrate",
        "state": "inspection_ready",
        "default": False,
    },
)

BENCH_ROBOT_SYSTEMS_BY_ID = {
    str(item["id"]): item for item in BENCH_ROBOT_SYSTEM_CATALOG
}


def related_asset_paths(item: dict[str, Any]) -> tuple[str, ...]:
    """Return the payload, planning proxy and procedure substrate paths."""

    return tuple(
        str(item[key])
        for key in ("payload_path", "rigid_proxy_path", "auxiliary_path")
        if item.get(key)
    )


def resolve_featured_robot_system(selected: Iterable[str]) -> str | None:
    """Resolve the single large robot station selected for the shared bench."""

    selected_ids = set(selected)
    matches = tuple(
        asset_id for asset_id in BENCH_ROBOT_SYSTEMS_BY_ID if asset_id in selected_ids
    )
    if len(matches) > 1:
        raise ValueError(
            "Choose one featured Dr.Anmar robot system at a time: " + ", ".join(matches)
        )
    return matches[0] if matches else None
