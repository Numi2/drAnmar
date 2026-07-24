# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""One-dimensional proportional PSM jaw action on Isaac Lab's native path."""

from __future__ import annotations

import torch
from isaaclab.envs.mdp.actions import BinaryJointPositionAction


class ProportionalJointPositionAction(BinaryJointPositionAction):
    """Interpolate the configured close/open targets from one -1..+1 value.

    The action dimension and endpoint convention stay identical to NVIDIA's
    binary PSM action: -1 is exactly closed and +1 is exactly open.
    """

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        normalized = torch.clamp(actions.to(dtype=self._open_command.dtype), -1.0, 1.0)
        aperture = (normalized + 1.0) * 0.5
        self._processed_actions = (
            self._close_command.unsqueeze(0)
            + aperture * (self._open_command - self._close_command).unsqueeze(0)
        )
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )
