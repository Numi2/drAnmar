# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Contact-calibrated grasp frames for lift policies."""

# Selected by a bounded 1,200-environment Isaac Lab sweep against physics-owned
# bilateral jaw contact, completed lifts, object height, and angular velocity.
# A mesh centroid is not a valid substitute because the PSM tool-tip command
# frame is offset from the jaw collision center.
BLOCK_CONTACT_CALIBRATED_GRASP_OFFSET_M = (0.0, 0.0, -0.002)
BLOCK_CONTACT_CALIBRATED_GRASP_OFFSET_SOURCE = (
    "isaac_lab_1200_env_contact_sweep"
)
