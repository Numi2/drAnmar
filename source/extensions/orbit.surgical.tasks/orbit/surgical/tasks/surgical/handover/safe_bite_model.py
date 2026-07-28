# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Analytic retained-needle transport with a tightly bounded learned residual."""

from __future__ import annotations

import torch

from .end_to_end_model import EndToEndHandoverMLPModel
from .residual_model import HandoverAnalyticController

SAFE_BITE_HANDOVER_COMPLETE = 116
SAFE_BITE_ENTRY_ARMED = 117
SAFE_BITE_TISSUE_CONTACT = 118


class SafeBiteAnalyticController(HandoverAnalyticController):
    """Execute the proven handover, then servo only the receiver to T1."""

    def __init__(self) -> None:
        super().__init__()
        self.safe_bite_position_action_limit = 0.12
        self.safe_bite_orientation_action_limit = 0.12
        self.safe_bite_translation_orientation_gate = 0.35
        self.safe_bite_lateral_descent_gate_m = 0.008
        self.post_arm_inward_action_limit = 0.03

    def forward(
        self,
        raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_action, giver_residual_mask, _ = super().forward(raw)
        if raw.shape[1] < 122:
            return (
                base_action,
                giver_residual_mask,
                torch.zeros_like(base_action, dtype=torch.bool),
            )
        handover_complete = raw[:, SAFE_BITE_HANDOVER_COMPLETE] > 0.5
        entry_armed = raw[:, SAFE_BITE_ENTRY_ARMED] > 0.5
        tissue_contact = raw[:, SAFE_BITE_TISSUE_CONTACT] > 0.5
        active = handover_complete & ~entry_armed & ~tissue_contact
        contact_transition_active = (
            handover_complete & entry_armed & ~tissue_contact
        )

        position_error_m = raw[:, 107:110] * 0.02
        translation = (
            position_error_m / self.position_scale
        ).clamp(
            -self.safe_bite_position_action_limit,
            self.safe_bite_position_action_limit,
        )
        rotation_error = (
            raw[:, 110:113]
            + raw[:, 113:116]
        )
        orientation = (
            rotation_error / self.orientation_scale
        ).clamp(
            -self.safe_bite_orientation_action_limit,
            self.safe_bite_orientation_action_limit,
        )
        orientation_magnitude = torch.linalg.vector_norm(
            rotation_error, dim=-1
        )
        lateral_error = torch.linalg.vector_norm(
            position_error_m[:, :2], dim=-1
        )
        # Do not descend while far laterally or badly oriented.  This keeps
        # the deterministic base controller outside the tissue footprint
        # until its approach vector is credible.
        gate_descent = (
            lateral_error > self.safe_bite_lateral_descent_gate_m
        ) | (
            orientation_magnitude
            > self.safe_bite_translation_orientation_gate
        )
        translation[:, 2] = torch.where(
            gate_descent & (translation[:, 2] < 0.0),
            torch.zeros_like(translation[:, 2]),
            translation[:, 2],
        )
        safe_receiver_action = torch.cat(
            (
                translation,
                orientation,
                -torch.ones(
                    (raw.shape[0], 1),
                    dtype=raw.dtype,
                    device=raw.device,
                ),
            ),
            dim=-1,
        )
        inward_receiver_action = torch.cat(
            (
                (
                    raw[:, 119:122]
                    * self.post_arm_inward_action_limit
                ),
                torch.zeros(
                    (raw.shape[0], 3),
                    dtype=raw.dtype,
                    device=raw.device,
                ),
                -torch.ones(
                    (raw.shape[0], 1),
                    dtype=raw.dtype,
                    device=raw.device,
                ),
            ),
            dim=-1,
        )
        successor_receiver_action = torch.where(
            contact_transition_active.unsqueeze(-1),
            inward_receiver_action,
            safe_receiver_action,
        )
        giver_is_robot_1 = raw[:, 82] > 0.5
        robot_1_action = torch.where(
            giver_is_robot_1.unsqueeze(-1),
            base_action[:, :7],
            successor_receiver_action,
        )
        robot_2_action = torch.where(
            giver_is_robot_1.unsqueeze(-1),
            successor_receiver_action,
            base_action[:, 7:14],
        )
        safe_action = torch.cat((robot_1_action, robot_2_action), dim=-1)
        base_action = torch.where(
            (active | contact_transition_active).unsqueeze(-1),
            safe_action,
            base_action,
        ).clamp(-1.0, 1.0)

        receiver_role_mask = torch.zeros_like(
            safe_receiver_action, dtype=torch.bool
        )
        receiver_role_mask[:, :6] = active.unsqueeze(-1)
        robot_1_receiver_mask = torch.where(
            giver_is_robot_1.unsqueeze(-1),
            torch.zeros_like(receiver_role_mask),
            receiver_role_mask,
        )
        robot_2_receiver_mask = torch.where(
            giver_is_robot_1.unsqueeze(-1),
            receiver_role_mask,
            torch.zeros_like(receiver_role_mask),
        )
        receiver_residual_mask = torch.cat(
            (robot_1_receiver_mask, robot_2_receiver_mask), dim=-1
        )
        return base_action, giver_residual_mask, receiver_residual_mask


class SafeBiteEndToEndMLPModel(EndToEndHandoverMLPModel):
    """Train one receiver residual head; keep prerequisite actions analytic."""

    def __init__(
        self,
        *args,
        residual_scale: float = 0.003,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            residual_scale=residual_scale,
            **kwargs,
        )
        self.controller = SafeBiteAnalyticController()
        self.configure_safe_bite_adaptation()

    def configure_safe_bite_adaptation(self) -> None:
        """Learn receiver pose corrections only after retained handover."""

        if getattr(self, "_safe_bite_adaptation_configured", False):
            return
        self.receiver_adaptation_enabled = True
        for parameter in self.phase_network.parameters():
            parameter.requires_grad_(False)
        safe_bite_head = self.phase_network.heads[3]
        safe_bite_head.weight.requires_grad_(True)
        safe_bite_head.bias.requires_grad_(True)
        receiver_pose_rows = torch.zeros(
            14,
            dtype=safe_bite_head.weight.dtype,
            device=safe_bite_head.weight.device,
        )
        receiver_pose_rows[7:13] = 1.0
        safe_bite_head.weight.register_hook(
            lambda gradient: gradient * receiver_pose_rows.unsqueeze(-1)
        )
        safe_bite_head.bias.register_hook(
            lambda gradient: gradient * receiver_pose_rows
        )
        if self.distribution is not None:
            for parameter_name in ("std_param", "log_std_param"):
                parameter = getattr(
                    self.distribution,
                    parameter_name,
                    None,
                )
                if parameter is not None:
                    parameter.requires_grad_(False)
        self._safe_bite_adaptation_configured = True

    def configure_receiver_adaptation(self) -> None:
        """Keep generic training launchers on the T1-only authority boundary."""

        self.configure_safe_bite_adaptation()
