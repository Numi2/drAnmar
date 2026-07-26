# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Asset-derived physical grasp frames for lift policies."""

# Volume centroid of the closed, composed block mesh after the task's 0.011
# runtime scale. The USD root is near an upper side edge, so targeting the root
# creates a large closure moment arm and an avoidable angular impulse.
BLOCK_PHYSICAL_GRASP_OFFSET_M = (0.0055, 0.0, -0.0060057354)
BLOCK_PHYSICAL_GRASP_OFFSET_SOURCE = "closed_composed_mesh_volume_centroid"
