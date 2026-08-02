# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Analytical tissue-entry controller with a bounded recurrent residual."""

from __future__ import annotations

import torch
from rsl_rl.models import RNNModel
from rsl_rl.utils import unpad_trajectories
from torch import nn

from isaaclab.utils.math import (
    axis_angle_from_quat,
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
        end_effector_quaternion = raw[..., 19:23]
        entry_position = raw[..., 36:39]
        surface_normal = torch.nn.functional.normalize(raw[..., 43:46], dim=-1)
        indentation = raw[..., 46]
        contacts = raw[..., 48:50]
        normalized_wrench = raw[..., 50:56]
        phase = torch.argmax(raw[..., 58:63], dim=-1)

        approach_offset = torch.where(
            (phase == 0).unsqueeze(-1),
            torch.full_like(surface_normal, 0.003),
            torch.where(
                (phase == 1).unsqueeze(-1),
                torch.full_like(surface_normal, 0.0010),
                torch.where(
                    (phase == 2).unsqueeze(-1),
                    torch.full_like(surface_normal, -0.0015),
                    torch.full_like(surface_normal, -0.0020),
                ),
            ),
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
            torch.tensor((-1.0, 0.0, 0.0), device=raw.device, dtype=raw.dtype).expand_as(
                surface_normal
            ),
        )
        plane_sign = torch.sign(torch.sum(current_plane_normal * wound_tangent, dim=-1, keepdim=True))
        plane_sign = torch.where(plane_sign == 0.0, torch.ones_like(plane_sign), plane_sign)
        wound_tangent = wound_tangent * plane_sign
        desired_tangent = -surface_normal
        desired_local_x = -desired_tangent
        desired_second_axis = torch.linalg.cross(wound_tangent, desired_local_x)
        desired_needle_matrix = torch.stack(
            (desired_local_x, desired_second_axis, wound_tangent), dim=-1
        )
        desired_needle_quaternion = quat_from_matrix(desired_needle_matrix)
        grasp_quaternion = quat_mul(
            quat_conjugate(needle_quaternion), end_effector_quaternion
        )
        desired_tool_quaternion = quat_mul(desired_needle_quaternion, grasp_quaternion)

        # Solve orientation through the measured settled grasp. Translation is
        # applied only while rotation is paused below, so the measured tip
        # displacement maps one-for-one to rigid tool translation without an
        # idealized grasp-position offset.
        desired_tip_position = entry_position + surface_normal * approach_offset
        translation_delta_m = desired_tip_position - needle_position
        normal_component = torch.sum(
            translation_delta_m * surface_normal, dim=-1, keepdim=True
        )
        tangential_component = translation_delta_m - surface_normal * normal_component
        tangential_error = torch.linalg.vector_norm(
            tangential_component, dim=-1, keepdim=True
        )
        tangential_component = torch.where(
            ((phase >= 2).unsqueeze(-1) & (tangential_error <= 0.0009)),
            torch.zeros_like(tangential_component),
            tangential_component,
        )
        tangential_limit_m = 0.00005
        tangential_component = tangential_component * torch.clamp(
            tangential_limit_m / tangential_error.clamp_min(1.0e-9), max=1.0
        )
        normal_limit_m = self.normal_advance_limit * self.translation_scale_m
        bounded_delta_m = tangential_component + surface_normal * normal_component.clamp(
            -normal_limit_m, normal_limit_m
        )
        # Clamp the vector norm only after the normal/tangential decomposition.
        # Per-axis clipping first rotates a saturated diagonal command, which
        # produced millimetres of lateral drift during the stand-off approach.
        delta_norm = torch.linalg.vector_norm(bounded_delta_m, dim=-1, keepdim=True)
        bounded_delta_m = bounded_delta_m * torch.clamp(
            self.translation_scale_m / delta_norm.clamp_min(1.0e-9), max=1.0
        )
        translation = bounded_delta_m / self.translation_scale_m
        orientation_error = axis_angle_from_quat(
            quat_mul(
                desired_tool_quaternion,
                quat_conjugate(end_effector_quaternion),
            )
        )
        rotation = (orientation_error / self.rotation_scale_rad).clamp(-1.0, 1.0)
        # Arbitrate the six-dimensional DLS command instead of letting the
        # larger rotation residual starve PSM translation. Use the same
        # ten-degree boundary as the authoritative gate; the phase transition
        # itself latches alignment once the measured entry region is valid.
        tangent_aligned = torch.sum(current_tangent * desired_tangent, dim=-1) >= 0.984807753
        plane_aligned = torch.abs(torch.sum(current_plane_normal * wound_tangent, dim=-1)) >= 0.984807753
        rotation = torch.where(
            (tangent_aligned & plane_aligned).unsqueeze(-1), torch.zeros_like(rotation), rotation
        )
        aligning = (phase <= 1) & ~(tangent_aligned & plane_aligned)
        translation = torch.where(
            aligning.unsqueeze(-1), torch.zeros_like(translation), translation
        )
        indent_phase = phase >= 2
        contact_phase = indent_phase & (indentation > 0.0)
        normal_command = torch.sum(translation * surface_normal, dim=-1, keepdim=True)
        normal_command = normal_command.clamp(-self.normal_advance_limit, self.normal_advance_limit)
        translation = torch.where(
            contact_phase.unsqueeze(-1), surface_normal * normal_command, translation
        )
        rotation = torch.where(indent_phase.unsqueeze(-1), torch.zeros_like(rotation), rotation)
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
        # Isaac Lab's compatibility config still serializes pre-rsl-rl-5
        # stochastic policy fields. The installed RNNModel accepts the new
        # distribution_cfg contract only, so consume the deprecated keys at
        # this custom-model boundary instead of leaking them to RSL-RL.
        for deprecated_key in (
            "stochastic",
            "init_noise_std",
            "noise_std_type",
            "state_dependent_std",
        ):
            kwargs.pop(deprecated_key, None)
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
        latent = self.get_latent(obs, masks, hidden_state)
        residual = torch.tanh(self.mlp(latent))
        if masks is not None:
            raw = unpad_trajectories(raw, masks)
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
        action_mean = (base + safe_residual).clamp(-1.0, 1.0)
        # PPO must evaluate the exact composed action sent to the environment.
        # Centering the distribution on the raw residual while returning
        # base+residual makes stored actions and log probabilities disagree.
        if self.distribution is not None:
            self.distribution.update(action_mean)
            if stochastic_output:
                return self.distribution.sample()
            return self.distribution.deterministic_output(action_mean)
        return action_mean


class ThroughPunctureAnalyticController(nn.Module):
    """Drive a top-to-top curved bite until 20% of the arc re-emerges."""

    def __init__(self) -> None:
        super().__init__()
        self.entry_controller = PenetrationAnalyticController()
        self.translation_scale_m = 0.00025
        self.rotation_scale_rad = 0.00872664626
        self.curvature_radius_m = 0.0070028174960433945
        self.tissue_thickness_m = 0.006
        self.target_exposed_fraction = 0.22
        self.drive_rotation_command = 0.5

    def forward(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phase = torch.argmax(raw[..., 58:65], dim=-1)
        entry_phase = torch.nn.functional.one_hot(phase.clamp(max=4), num_classes=5).to(
            dtype=raw.dtype
        )
        entry_raw = torch.cat((raw[..., :58], entry_phase, raw[..., 65:71]), dim=-1)
        entry_base, _, unsafe = self.entry_controller(entry_raw)
        needle_position = raw[..., 23:26]
        needle_quaternion = raw[..., 26:30]
        entry_position = raw[..., 36:39]
        surface_normal = torch.nn.functional.normalize(raw[..., 43:46], dim=-1)
        indentation = raw[..., 46].clamp_min(0.0)
        exposed_fraction = raw[..., 72].clamp_min(0.0)

        reference_x = torch.tensor(
            (1.0, 0.0, 0.0), device=raw.device, dtype=raw.dtype
        ).expand_as(surface_normal)
        reference_minus_z = torch.tensor(
            (0.0, 0.0, -1.0), device=raw.device, dtype=raw.dtype
        ).expand_as(surface_normal)
        reference_axis = torch.where(
            torch.abs(surface_normal[..., :1]) > 0.9, reference_minus_z, reference_x
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
        plane_sign = torch.sign(
            torch.sum(current_plane_normal * wound_tangent, dim=-1, keepdim=True)
        )
        wound_tangent = wound_tangent * torch.where(
            plane_sign == 0.0, torch.ones_like(plane_sign), plane_sign
        )
        start_tangent = -surface_normal
        current_tangent = quat_apply(
            needle_quaternion,
            torch.tensor((-1.0, 0.0, 0.0), device=raw.device, dtype=raw.dtype).expand_as(
                surface_normal
            ),
        )
        sine = torch.sum(
            torch.linalg.cross(current_tangent, start_tangent) * wound_tangent,
            dim=-1,
        )
        cosine = torch.sum(start_tangent * current_tangent, dim=-1).clamp(-1.0, 1.0)
        orientation_angle = torch.atan2(sine, cosine).clamp_min(0.0)
        geometric_angle = torch.asin(
            (indentation / self.curvature_radius_m).clamp(0.0, 1.0)
        )
        trajectory_angle = torch.maximum(orientation_angle, geometric_angle)
        drive_direction = torch.linalg.cross(start_tangent, wound_tangent)
        desired_tip_position = (
            entry_position
            + drive_direction
            * (self.curvature_radius_m * (1.0 - torch.cos(trajectory_angle))).unsqueeze(-1)
            + start_tangent
            * (self.curvature_radius_m * torch.sin(trajectory_angle)).unsqueeze(-1)
        )
        translation_delta = desired_tip_position - needle_position
        delta_norm = torch.linalg.vector_norm(translation_delta, dim=-1, keepdim=True)
        translation = translation_delta * torch.clamp(
            self.translation_scale_m / delta_norm.clamp_min(1.0e-9), max=1.0
        ) / self.translation_scale_m

        # The sharp tip re-emerges through the top surface after a half turn.
        exit_angle = torch.tensor(torch.pi, device=raw.device, dtype=raw.dtype)
        target_angle = exit_angle + self.target_exposed_fraction * torch.pi
        rotate_active = (orientation_angle < target_angle) & (
            exposed_fraction < self.target_exposed_fraction
        ) & (phase <= 5)
        rotation = (
            -wound_tangent
            * rotate_active.unsqueeze(-1).to(raw.dtype)
            * self.drive_rotation_command
        )
        through_base = torch.cat((translation, rotation), dim=-1).clamp(-1.0, 1.0)
        # Keep following the same circular tip trajectory after top re-emergence.
        # Switching to a static historical exit-point correction
        # rotates about the jaws, lifts the sharp tip back into the tract, and
        # reverses exposure. The curvature controller already produced the
        # qualified exit intersection and remains authoritative to presentation.
        through_phase = phase >= 3
        base = torch.where(through_phase.unsqueeze(-1), through_base, entry_base)
        base = torch.where(unsafe.unsqueeze(-1), torch.zeros_like(base), base)
        return base, phase, unsafe


class ThroughPunctureResidualGRUModel(PenetrationResidualGRUModel):
    """Bounded recurrent correction around the curvature-following controller."""

    def __init__(self, *args, residual_scale: float = 0.20, **kwargs) -> None:
        super().__init__(*args, residual_scale=residual_scale, **kwargs)
        self.controller = ThroughPunctureAnalyticController()

    def forward(
        self,
        obs,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        raw = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        latent = self.get_latent(obs, masks, hidden_state)
        residual = torch.tanh(self.mlp(latent))
        if masks is not None:
            raw = unpad_trajectories(raw, masks)
        base, phase, unsafe = self.controller(raw)
        surface_normal = torch.nn.functional.normalize(raw[..., 43:46], dim=-1)
        needle_quaternion = raw[..., 26:30]
        wound_tangent = quat_apply(
            needle_quaternion,
            torch.tensor((0.0, 0.0, 1.0), device=raw.device, dtype=raw.dtype).expand_as(
                surface_normal
            ),
        )
        drive_direction = torch.nn.functional.normalize(
            torch.linalg.cross(-surface_normal, wound_tangent), dim=-1
        )
        contact_phase = phase >= 2
        normal_component = torch.sum(
            residual[..., :3] * surface_normal, dim=-1, keepdim=True
        )
        drive_component = torch.sum(
            residual[..., :3] * drive_direction, dim=-1, keepdim=True
        )
        constrained_translation = (
            surface_normal * normal_component + drive_direction * drive_component
        )
        rotation_component = torch.sum(
            residual[..., 3:] * wound_tangent, dim=-1, keepdim=True
        )
        constrained_rotation = wound_tangent * rotation_component
        safe_residual = self.residual_scale * torch.cat(
            (
                torch.where(
                    contact_phase.unsqueeze(-1), constrained_translation, residual[..., :3]
                ),
                torch.where(
                    contact_phase.unsqueeze(-1), constrained_rotation, residual[..., 3:]
                ),
            ),
            dim=-1,
        )
        safe_residual = torch.where(
            unsafe.unsqueeze(-1), torch.zeros_like(safe_residual), safe_residual
        )
        action_mean = (base + safe_residual).clamp(-1.0, 1.0)
        if self.distribution is not None:
            self.distribution.update(action_mean)
            if stochastic_output:
                return self.distribution.sample()
            return self.distribution.deterministic_output(action_mean)
        return action_mean


class PulloutAnalyticController(nn.Module):
    """Compose giver passage with receiver acquisition, transfer, and pullout."""

    def __init__(self) -> None:
        super().__init__()
        self.through_controller = ThroughPunctureAnalyticController()

    def forward(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Pullout expands the previous-action term from 6D to 14D. Rebuild the
        # exact 77D through-puncture view consumed by the qualified controller.
        through_raw = torch.cat(
            (raw[..., :65], raw[..., 65:71], raw[..., 79:85]), dim=-1
        )
        # Keep the giver at stand-off until lateral error is inside the entry
        # tolerance; otherwise the curved shaft can reach tissue before the tip
        # is valid. The distal PSM must also lift clear before alignment
        # rotation, rather than sweeping its jaws through the rigid surface.
        entry_delta = raw[..., 23:26] - raw[..., 36:39]
        entry_surface_normal = torch.nn.functional.normalize(
            raw[..., 43:46], dim=-1
        )
        entry_lateral_delta = entry_delta - entry_surface_normal * torch.sum(
            entry_delta * entry_surface_normal, dim=-1, keepdim=True
        )
        entry_lateral_error = torch.linalg.vector_norm(
            entry_lateral_delta, dim=-1
        )
        reported_phase = torch.argmax(through_raw[..., 58:65], dim=-1)
        hold_standoff = (reported_phase <= 2) & (entry_lateral_error > 0.0009)
        standoff_phase = torch.nn.functional.one_hot(
            torch.zeros_like(reported_phase), num_classes=7
        ).to(raw.dtype)
        through_raw = through_raw.clone()
        through_raw[..., 58:65] = torch.where(
            hold_standoff.unsqueeze(-1), standoff_phase, through_raw[..., 58:65]
        )
        calibrated_entry = through_raw[..., 36:39] - 0.0007 * through_raw[..., 43:46]
        apply_surface_calibration = (reported_phase == 2) & ~hold_standoff
        through_raw[..., 36:39] = torch.where(
            apply_surface_calibration.unsqueeze(-1),
            calibrated_entry,
            through_raw[..., 36:39],
        )
        giver_body, _, unsafe = self.through_controller(through_raw)
        phase = torch.argmax(raw[..., 116:128], dim=-1)
        # Giver contact loss is unsafe before transfer, but intentional after
        # receiver custody. The environment separately hard-fails any receiver
        # custody loss during pull/clear.
        unsafe = unsafe & (phase < 9)
        receiver_guidance = raw[..., 110:116].clamp(-1.0, 1.0)
        receiver_active = phase >= 7
        receiver_body = torch.where(
            receiver_active.unsqueeze(-1),
            receiver_guidance,
            torch.zeros_like(receiver_guidance),
        )
        receiver_body = torch.where(
            (phase >= 11).unsqueeze(-1), torch.zeros_like(receiver_body), receiver_body
        )
        receiver_body = torch.where(
            (phase == 9).unsqueeze(-1), torch.zeros_like(receiver_body), receiver_body
        )
        giver_body = torch.where(
            (phase >= 7).unsqueeze(-1), torch.zeros_like(giver_body), giver_body
        )
        giver_retreat = torch.cat(
            (
                torch.nn.functional.normalize(raw[..., 43:46], dim=-1),
                torch.zeros_like(raw[..., 43:46]),
            ),
            dim=-1,
        )
        giver_retreat_active = (phase >= 9) & (phase < 11)
        giver_body = torch.where(
            giver_retreat_active.unsqueeze(-1), giver_retreat, giver_body
        )
        giver_gripper = torch.where(
            phase >= 10,
            torch.ones_like(phase, dtype=raw.dtype),
            -torch.ones_like(phase, dtype=raw.dtype),
        ).unsqueeze(-1)
        receiver_gripper = torch.where(
            phase >= 8,
            -torch.ones_like(phase, dtype=raw.dtype),
            torch.ones_like(phase, dtype=raw.dtype),
        ).unsqueeze(-1)
        action = torch.cat(
            (giver_body, giver_gripper, receiver_body, receiver_gripper), dim=-1
        )
        action = torch.where(unsafe.unsqueeze(-1), torch.zeros_like(action), action)
        return action.clamp(-1.0, 1.0), phase, unsafe


class PulloutResidualGRUModel(PenetrationResidualGRUModel):
    """GRU residual whose gripper sequencing remains analytically owned."""

    def __init__(self, *args, residual_scale: float = 0.15, **kwargs) -> None:
        super().__init__(*args, residual_scale=residual_scale, **kwargs)
        self.controller = PulloutAnalyticController()

    def forward(
        self,
        obs,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        raw = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        latent = self.get_latent(obs, masks, hidden_state)
        residual = torch.tanh(self.mlp(latent))
        if masks is not None:
            raw = unpad_trajectories(raw, masks)
        base, _, unsafe = self.controller(raw)
        body_mask = torch.tensor(
            (1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0),
            device=raw.device,
            dtype=raw.dtype,
        )
        safe_residual = self.residual_scale * residual * body_mask
        safe_residual = torch.where(
            unsafe.unsqueeze(-1), torch.zeros_like(safe_residual), safe_residual
        )
        action_mean = (base + safe_residual).clamp(-1.0, 1.0)
        if self.distribution is not None:
            self.distribution.update(action_mean)
            if stochastic_output:
                return self.distribution.sample()
            return self.distribution.deterministic_output(action_mean)
        return action_mean
