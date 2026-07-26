# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Contact-calibrated grasp frames for lift policies."""

import math

# Selected by a parallel 1,200-environment Isaac Lab sweep against physics-owned
# bilateral jaw contact, first-outcome strict success, object height, and
# angular velocity. A mesh centroid is not a valid substitute because the PSM
# tool-tip command frame is offset from the jaw collision center.
BLOCK_CONTACT_CALIBRATED_GRASP_OFFSET_M = (0.0, 0.0, -0.0014)
BLOCK_CONTACT_CALIBRATED_GRASP_OFFSET_SOURCE = (
    "isaac_lab_parallel_1200_env_first_outcome_contact_sweep"
)

# Geometry measured from the composed needle_sdf.usd layer after the task's
# 0.4 scale. Fractions run from the blunt/swage end toward the sharp end.
# Isaac Lab decides which fraction survives the physics-owned grasp and lift
# gates. Arc fraction 0.40 is contact-qualified for needle pickup by two
# complete 1,200-environment populations (seeds 17 and 2361): 2,228 of 2,400
# sustained pickups with zero hard failures.
NEEDLE_ARC_CENTER_XY_M = (0.01896937, -0.00036503)
NEEDLE_ARC_RADIUS_M = 0.01918304
NEEDLE_ARC_START_RAD = math.radians(89.4488)
NEEDLE_ARC_EXTENT_RAD = math.radians(181.0808)
NEEDLE_CENTERLINE_Z_M = 0.00038296
NEEDLE_TOOL_TIP_TO_JAW_COLLISION_Z_M = BLOCK_CONTACT_CALIBRATED_GRASP_OFFSET_M[2]
NEEDLE_PROVISIONAL_ARC_FRACTION = 0.40
NEEDLE_PROVISIONAL_GRASP_Z_OFFSET_M = 0.0006
NEEDLE_GEOMETRY_GRASP_OFFSET_SOURCE = (
    "composed_openusd_scaled_arc_fit_plus_psm_tool_tip_to_jaw_collision_offset"
)


def needle_geometry_grasp_offset_m(
    arc_fraction: float,
) -> tuple[float, float, float]:
    """Return a geometry-derived candidate grasp offset along the needle arc."""

    if not 0.0 <= arc_fraction <= 1.0:
        raise ValueError("needle arc fraction must be between 0.0 and 1.0")
    angle = NEEDLE_ARC_START_RAD + arc_fraction * NEEDLE_ARC_EXTENT_RAD
    return (
        NEEDLE_ARC_CENTER_XY_M[0] + NEEDLE_ARC_RADIUS_M * math.cos(angle),
        NEEDLE_ARC_CENTER_XY_M[1] + NEEDLE_ARC_RADIUS_M * math.sin(angle),
        NEEDLE_CENTERLINE_Z_M + NEEDLE_TOOL_TIP_TO_JAW_COLLISION_Z_M,
    )


_NEEDLE_PROVISIONAL_GEOMETRY_OFFSET_M = needle_geometry_grasp_offset_m(
    NEEDLE_PROVISIONAL_ARC_FRACTION
)
NEEDLE_PROVISIONAL_GRASP_OFFSET_M = (
    _NEEDLE_PROVISIONAL_GEOMETRY_OFFSET_M[0],
    _NEEDLE_PROVISIONAL_GEOMETRY_OFFSET_M[1],
    NEEDLE_PROVISIONAL_GRASP_Z_OFFSET_M,
)
