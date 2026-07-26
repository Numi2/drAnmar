# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Contact-calibrated grasp frames for lift policies."""

# Selected by a parallel 1,200-environment Isaac Lab sweep against physics-owned
# bilateral jaw contact, first-outcome strict success, object height, and
# angular velocity. A mesh centroid is not a valid substitute because the PSM
# tool-tip command frame is offset from the jaw collision center.
BLOCK_CONTACT_CALIBRATED_GRASP_OFFSET_M = (0.0, 0.0, -0.0014)
BLOCK_CONTACT_CALIBRATED_GRASP_OFFSET_SOURCE = (
    "isaac_lab_parallel_1200_env_first_outcome_contact_sweep"
)
