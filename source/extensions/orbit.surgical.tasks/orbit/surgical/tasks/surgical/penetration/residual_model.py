# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Analytical tissue-entry controller with a bounded recurrent residual."""

from __future__ import annotations

import torch
from rsl_rl.models import RNNModel
from torch import nn

from isaaclab.utils.math import (
    compute_pose_error,
    quat_apply,
    quat_conjugate,
    quat_from_matrix,
    quat_mul,
)


class PenetrationAnalyticController(nn.Module):
    """Approach, align, indent, puncture and stabilize at measured depth."""

    def __init__(self) -> None:
        super().__init__()
        self.translation_scale_m = 0.00025
        self.rotation_scale_rad = 0.00872664626
        self.normal_advance_limit = 0.4  # 0.1 mm per 20 ms
        self.normalized_contact_threshold = 0.02
        self.normalized_force_limit = 1.25

    def forward(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        needle_position = raw[..., 23:26]
        needle_quaternion = raw[..., 26:30]
        end_effector_position = raw[..., 16:19]
        end_effector_quaternion = raw[..., 19:23]
        entry_position = raw[..., 36:39]
        surface_normal = torch.nn.functional.normalize(raw[..., 43:46], dim=-1)
        contacts = raw[..., 48:50]
        normalized_wrench = raw[..., 50:56]
        phase = torch.argmax(raw[..., 58:63], dim=-1)

        approach_offset = torch.where(
            (phase == 0).unsqueeze(-1),
            torch.full_like(surface_normal, 0.008),
            torch.where(
                (phase == 1).unsqueeze(-1),
                torch.full_like(surface_normal, -0.0001),
                torch.where(
                    (phase == 2).unsqueeze(-1),
                    torch.full_like(surface_normal, -0.0015),
                    torch.full_like(surface_normal, -0.0020),
                ),
            ),
        )
        target_position = entry_position + surface_normal * approach_offset
        translation = ((target_position - needle_position) / self.translation_scale_m).clamp(
            -1.0, 1.0
        )
        normal_component = torch.sum(
            translation * surface_normal, dim=-1, keepdim=True
        )
        tangential_component = translation - surface_normal * normal_component
        translation = tangential_component + surface_normal * normal_component.clamp(
            -self.normal_advance_limit, self.normal_advance_limit
        )

        # Align the needle's local tangent +X with the inward normal and its
        # plane normal +Z with the wound tangent.  The IK command acts on the
        # tool, so compose the desired needle orientation through the authored
        # mid-grasp transform before computing Isaac's world-frame pose error.
        reference_x = torch.tensor(
            (1.0, 0.0, 0.0), device=raw.device, dtype=raw.dtype
        ).expand_as(surface_normal)
        reference_minus_z = torch.tensor(
            (0.0, 0.0, -1.0), device=raw.device, dtype=raw.dtype
        ).expand_as(surface_normal)
        reference_axis = torch.where(
            (torch.abs(surface_normal[..., :1]) > 0.9), reference_minus_z, reference_x
        )
        wound_tangent = torch.nn.functional.normalize(
            torch.linalg.cross(surface_normal, reference_axis), dim=-1
        )
        current_plane_normal = quat_apply(
            needle_quaternion,
            torch.tensor((0.0, 0.0, 1.0), device=raw.device, dtype=raw.dtype).expand_as(
                surface_normal
            ),
        )
        current_tangent = quat_apply(
            needle_quaternion,
            torch.tensor((1.0, 0.0, 0.0), device=raw.device, dtype=raw.dtype).expand_as(
                surface_normal
            ),
        )
        plane_sign = torch.sign(torch.sum(current_plane_normal * wound_tangent, dim=-1, keepdim=True))
        plane_sign = torch.where(plane_sign == 0.0, torch.ones_like(plane_sign), plane_sign)
        wound_tangent = wound_tangent * plane_sign
        desired_tangent = -surface_normal
        desired_second_axis = torch.linalg.cross(wound_tangent, desired_tangent)
        desired_needle_matrix = torch.stack(
            (desired_tangent, desired_second_axis, wound_tangent), dim=-1
        )
        desired_needle_quaternion = quat_from_matrix(desired_needle_matrix)
        grasp_quaternion = quat_mul(
            quat_conjugate(needle_quaternion), end_effector_quaternion
        )
        desired_tool_quaternion = quat_mul(desired_needle_quaternion, grasp_quaternion)
        _, orientation_error = compute_pose_error(
            end_effector_position,
            end_effector_quaternion,
            end_effector_position,
            desired_tool_quaternion,
        )
        rotation = (orientation_error / self.rotation_scale_rad).clamp(-0.1, 0.1)
        # Use a 5-degree controller margin inside the authoritative 10-degree
        # environment gate so translation has room for IK coupling without
        # numerical chatter at the phase boundary.
        tangent_aligned = torch.sum(current_tangent * desired_tangent, dim=-1) >= 0.996194698
        plane_aligned = torch.abs(torch.sum(current_plane_normal * wound_tangent, dim=-1)) >= 0.996194698
        rotation = torch.where(
            (tangent_aligned & plane_aligned).unsqueeze(-1), torch.zeros_like(rotation), rotation
        )
        aligning = (phase == 1) & ~(tangent_aligned & plane_aligned)
        translation = torch.where(
            aligning.unsqueeze(-1), torch.zeros_like(translation), translation
        )

        contact_phase = phase >= 2
        normal_command = torch.sum(translation * surface_normal, dim=-1, keepdim=True)
        normal_command = normal_command.clamp(-self.normal_advance_limit, self.normal_advance_limit)
        translation = torch.where(
            contact_phase.unsqueeze(-1), surface_normal * normal_command, translation
        )
        rotation = torch.where(contact_phase.unsqueeze(-1), torch.zeros_like(rotation), rotation)
        base = torch.cat((translation, rotation), dim=-1)

        bilateral = torch.all(contacts > self.normalized_contact_threshold, dim=-1)
        normalized_force = torch.linalg.vector_norm(normalized_wrench[..., :3], dim=-1)
        grasp_loss = ~bilateral
        force_unsafe = (phase <= 2) & (normalized_force > self.normalized_force_limit)
        unsafe = grasp_loss | force_unsafe
        retreat = torch.cat(
            (surface_normal * self.normal_advance_limit, torch.zeros_like(rotation)), dim=-1
        )
        base = torch.where(force_unsafe.unsqueeze(-1), retreat, base)
        base = torch.where(grasp_loss.unsqueeze(-1), torch.zeros_like(base), base)
        return base, phase, unsafe


class PenetrationResidualGRUModel(RNNModel):
    """GRU-128 actor whose learned output cannot bypass the safety base."""

    def __init__(self, *args, residual_scale: float = 0.25, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.residual_scale = float(residual_scale)
        self.controller = PenetrationAnalyticController()
        final_linear = next(
            module for module in reversed(self.mlp) if isinstance(module, nn.Linear)
        )
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def forward(
        self,
        obs,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        raw = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        residual = super().forward(obs, masks, hidden_state, stochastic_output)
        base, phase, unsafe = self.controller(raw)
        surface_normal = torch.nn.functional.normalize(raw[..., 43:46], dim=-1)
        contact_phase = phase >= 2
        normal_residual = torch.sum(residual[..., :3] * surface_normal, dim=-1, keepdim=True)
        translation_residual = torch.where(
            contact_phase.unsqueeze(-1),
            surface_normal * normal_residual,
            residual[..., :3],
        )
        rotation_residual = torch.where(
            contact_phase.unsqueeze(-1),
            torch.zeros_like(residual[..., 3:]),
            residual[..., 3:],
        )
        safe_residual = self.residual_scale * torch.cat(
            (translation_residual, rotation_residual), dim=-1
        )
        safe_residual = torch.where(
            unsafe.unsqueeze(-1), torch.zeros_like(safe_residual), safe_residual
        )
        return (base + safe_residual).clamp(-1.0, 1.0)
