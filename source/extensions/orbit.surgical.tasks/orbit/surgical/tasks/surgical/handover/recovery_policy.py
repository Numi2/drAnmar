# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Attempt-level learned recovery around the frozen handover policy.

The qualified 98-observation, 14-action handover actor remains the authority
for the first pickup attempt and for every post-custody phase.  This module
only supplies one bounded needle-frame grasp correction after a failed pickup
has completed a deterministic full-open reset.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch import nn

from orbit.surgical.tasks.surgical.lift.grasp_frames import (
    NEEDLE_PROVISIONAL_GRASP_OFFSET_M,
    needle_geometry_grasp_offset_m,
)

_RECEIVER_GRASP_OFFSET = needle_geometry_grasp_offset_m(0.65)


def _quat_xyzw_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert normalized XYZW quaternions to rotation matrices."""

    quaternion = quaternion / torch.linalg.vector_norm(
        quaternion,
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0e-8)
    x, y, z, w = quaternion.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack(
        (
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


def _quat_conjugate_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    result = quaternion.clone()
    result[:, :3] = -result[:, :3]
    return result


def _quat_multiply_xyzw(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    left_xyz, left_w = left[:, :3], left[:, 3:]
    right_xyz, right_w = right[:, :3], right[:, 3:]
    xyz = (
        left_w * right_xyz
        + right_w * left_xyz
        + torch.cross(left_xyz, right_xyz, dim=-1)
    )
    w = left_w * right_w - (left_xyz * right_xyz).sum(
        dim=-1,
        keepdim=True,
    )
    return torch.cat((xyz, w), dim=-1)


def _axis_angle_to_quat_xyzw(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.vector_norm(
        axis_angle,
        dim=-1,
        keepdim=True,
    )
    half_angle = 0.5 * angle
    scale = torch.where(
        angle > 1.0e-8,
        torch.sin(half_angle) / angle,
        0.5 - angle.square() / 48.0,
    )
    return torch.cat((axis_angle * scale, torch.cos(half_angle)), dim=-1)


def _quat_xyzw_to_axis_angle(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / torch.linalg.vector_norm(
        quaternion,
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0e-8)
    quaternion = torch.where(
        (quaternion[:, 3:] < 0.0),
        -quaternion,
        quaternion,
    )
    vector = quaternion[:, :3]
    vector_norm = torch.linalg.vector_norm(
        vector,
        dim=-1,
        keepdim=True,
    )
    angle = 2.0 * torch.atan2(
        vector_norm,
        quaternion[:, 3:].clamp_min(1.0e-8),
    )
    scale = torch.where(
        vector_norm > 1.0e-8,
        angle / vector_norm,
        2.0 + vector_norm.square() / 3.0,
    )
    return vector * scale


def _project_vector(
    value: torch.Tensor,
    maximum_norm: float,
) -> torch.Tensor:
    norm = torch.linalg.vector_norm(
        value,
        dim=-1,
        keepdim=True,
    )
    scale = torch.clamp(maximum_norm / norm.clamp_min(1.0e-8), max=1.0)
    return value * scale


class PickupRecoveryHead(nn.Module):
    """One-shot retry correction head with no gripper output."""

    input_dim: int = 29
    output_dim: int = 6

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, self.output_dim),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.network(context))


class HandoverPickupRecoveryPolicy(nn.Module):
    """Frozen handover actor plus a deterministic retry coordinator."""

    def __init__(
        self,
        base_policy: nn.Module,
        recovery_head: PickupRecoveryHead | None = None,
        *,
        position_cap_m: float = 0.005,
        orientation_cap_rad: float = math.radians(5.0),
        episode_frames: int = 2000,
    ) -> None:
        super().__init__()
        if not 0.0 < position_cap_m <= 0.005:
            raise ValueError("pickup recovery position cap must be in (0, 0.005]")
        if not 0.0 < orientation_cap_rad <= math.radians(5.0):
            raise ValueError(
                "pickup recovery orientation cap must be in (0, 5 degrees]"
            )
        self.base_policy = base_policy
        self.recovery_head = recovery_head or PickupRecoveryHead()
        self.register_buffer(
            "canonical_grasp_offset",
            torch.tensor(NEEDLE_PROVISIONAL_GRASP_OFFSET_M),
        )
        self.state_canonical = 0
        self.state_failed = 1
        self.state_reopening = 2
        self.state_open_settle = 3
        self.state_learned_retry = 4
        self.state_secure = 5
        self.context_dim = 29
        self.position_cap_m = float(position_cap_m)
        self.orientation_cap_rad = float(orientation_cap_rad)
        self.episode_frames = int(episode_frames)

        for parameter in self.base_policy.parameters():
            parameter.requires_grad_(False)

        self.position_scale = 0.01
        self.orientation_scale = 0.05
        self.approach_height = 0.02
        self.lateral_alignment_threshold = 0.005
        self.close_distance = 0.005
        self.slow_approach_radius = 0.02
        self.slow_approach_action_limit = 0.1
        self.orientation_action_limit = 0.6
        self.orientation_tolerance = 0.035
        self.normalized_contact_threshold = 0.002
        self.close_dwell_steps = 15
        self.custody_loss_steps = 3
        self.open_settle_steps = 3
        self.open_joint_displacement_tolerance_rad = 0.0215
        self.closed_joint_displacement_rad = 0.4085

        self._batch_size = 0
        self._fixed_correction: torch.Tensor | None = None
        self._fixed_correction_after_first_retry: torch.Tensor | None = None
        self._fixed_correction_delta: torch.Tensor | None = None
        self._correction_candidates: torch.Tensor | None = None
        self.retry_state = torch.empty(0, dtype=torch.long)
        self.retry_count = torch.empty(0, dtype=torch.long)
        self.episode_step = torch.empty(0, dtype=torch.long)
        self.close_dwell = torch.empty(0, dtype=torch.long)
        self.custody_loss_dwell = torch.empty(0, dtype=torch.long)
        self.open_settle_dwell = torch.empty(0, dtype=torch.long)
        self.ever_bilateral = torch.empty(0, dtype=torch.bool)
        self.bilateral_contact_history = torch.empty(
            (0, 5),
            dtype=torch.bool,
        )
        self.failure_forces = torch.empty((0, 2))
        self.failure_loss_flags = torch.empty((0, 2))
        self.correction = torch.empty((0, 6))
        self.first_attempt_failed = torch.empty(0, dtype=torch.bool)
        self.recovered_custody = torch.empty(0, dtype=torch.bool)
        self.activation_count = torch.empty(0, dtype=torch.long)
        self.last_context = torch.empty((0, self.context_dim))
        self.last_activation_mask = torch.empty(0, dtype=torch.bool)

    def _initialize_state(
        self,
        raw: torch.Tensor,
    ) -> None:
        batch_size = raw.shape[0]
        device, dtype = raw.device, raw.dtype
        self._batch_size = batch_size
        self.retry_state = torch.full(
            (batch_size,),
            self.state_canonical,
            dtype=torch.long,
            device=device,
        )
        self.retry_count = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=device,
        )
        self.episode_step = torch.zeros_like(self.retry_count)
        self.close_dwell = torch.zeros_like(self.retry_count)
        self.custody_loss_dwell = torch.zeros_like(self.retry_count)
        self.open_settle_dwell = torch.zeros_like(self.retry_count)
        self.ever_bilateral = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=device,
        )
        self.bilateral_contact_history = torch.zeros(
            (batch_size, 5),
            dtype=torch.bool,
            device=device,
        )
        self.failure_forces = torch.zeros(
            (batch_size, 2),
            dtype=dtype,
            device=device,
        )
        self.failure_loss_flags = torch.zeros_like(self.failure_forces)
        self.correction = torch.zeros(
            (batch_size, 6),
            dtype=dtype,
            device=device,
        )
        self.first_attempt_failed = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=device,
        )
        self.recovered_custody = torch.zeros_like(self.first_attempt_failed)
        self.activation_count = torch.zeros_like(self.retry_count)
        self.last_context = torch.zeros(
            (batch_size, self.context_dim),
            dtype=dtype,
            device=device,
        )
        self.last_activation_mask = torch.zeros_like(
            self.first_attempt_failed
        )

    def set_fixed_correction(
        self,
        correction: torch.Tensor | None,
    ) -> None:
        """Use a fixed physical 6D correction for controlled sweeps."""

        if correction is None:
            self._fixed_correction = None
            self._fixed_correction_after_first_retry = None
            return
        if correction.ndim not in {1, 2} or correction.shape[-1] != 6:
            raise ValueError("fixed pickup recovery correction must end in 6")
        self._fixed_correction = correction.detach().clone()
        self._fixed_correction_after_first_retry = None
        self._fixed_correction_delta = None

    def set_fixed_correction_after_first_retry(
        self,
        correction: torch.Tensor | None,
    ) -> None:
        """Escalate to a second fixed correction on retry two and later."""

        if correction is None:
            self._fixed_correction_after_first_retry = None
            return
        if self._fixed_correction is None:
            raise ValueError(
                "later pickup correction requires a first-retry correction"
            )
        if correction.ndim not in {1, 2} or correction.shape[-1] != 6:
            raise ValueError("later pickup recovery correction must end in 6")
        self._fixed_correction_after_first_retry = (
            correction.detach().clone()
        )

    def set_fixed_correction_delta(
        self,
        correction_delta: torch.Tensor | None,
    ) -> None:
        """Add a controlled local DAgger offset to the head prediction."""

        if correction_delta is None:
            self._fixed_correction_delta = None
            return
        if (
            correction_delta.ndim not in {1, 2}
            or correction_delta.shape[-1] != 6
        ):
            raise ValueError("fixed pickup correction delta must end in 6")
        self._fixed_correction_delta = correction_delta.detach().clone()
        self._fixed_correction = None
        self._fixed_correction_after_first_retry = None

    def set_correction_candidates(
        self,
        candidates: torch.Tensor | None,
    ) -> None:
        """Snap learned cumulative corrections to proven physical choices."""

        if candidates is None:
            self._correction_candidates = None
            return
        if (
            candidates.ndim != 2
            or candidates.shape[0] < 2
            or candidates.shape[1] != 6
        ):
            raise ValueError(
                "pickup correction candidates must have shape [N>=2, 6]"
            )
        if bool(
            (
                candidates[:, :3].norm(dim=-1)
                > self.position_cap_m + 1.0e-8
            ).any()
        ):
            raise ValueError("pickup correction candidate exceeds position cap")
        if bool(
            (
                candidates[:, 3:].norm(dim=-1)
                > self.orientation_cap_rad + 1.0e-8
            ).any()
        ):
            raise ValueError(
                "pickup correction candidate exceeds orientation cap"
            )
        self._correction_candidates = candidates.detach().clone()

    def _select_role(
        self,
        robot_1_value: torch.Tensor,
        robot_2_value: torch.Tensor,
        giver_is_robot_1: torch.Tensor,
    ) -> torch.Tensor:
        return torch.where(
            giver_is_robot_1.unsqueeze(-1),
            robot_1_value,
            robot_2_value,
        )

    def _recovery_context(
        self,
        raw: torch.Tensor,
        giver_is_robot_1: torch.Tensor,
    ) -> torch.Tensor:
        giver_ee = self._select_role(
            raw[:, 32:35],
            raw[:, 39:42],
            giver_is_robot_1,
        )
        giver_orientation = self._select_role(
            raw[:, 35:39],
            raw[:, 42:46],
            giver_is_robot_1,
        )
        object_pose = self._select_role(
            raw[:, 46:53],
            raw[:, 53:60],
            giver_is_robot_1,
        )
        object_position = object_pose[:, :3]
        object_rotation = _quat_xyzw_to_matrix(object_pose[:, 3:7])
        rotation_6d = torch.cat(
            (object_rotation[:, :, 0], object_rotation[:, :, 1]),
            dim=-1,
        )
        base_offset = self.canonical_grasp_offset.to(
            dtype=raw.dtype,
            device=raw.device,
        )
        # The qualified controller defines its canonical grasp offset directly
        # in the giver root frame.  Only the learned residual is expressed in
        # needle coordinates; this makes a zero correction exactly canonical.
        canonical_position = (
            object_position + base_offset.expand(raw.shape[0], -1)
        )
        position_error_root = canonical_position - giver_ee
        position_error_needle = torch.matmul(
            object_rotation.transpose(-1, -2),
            position_error_root.unsqueeze(-1),
        ).squeeze(-1)
        identity = torch.zeros_like(giver_orientation)
        identity[:, 3] = 1.0
        orientation_error_root = _quat_xyzw_to_axis_angle(
            _quat_multiply_xyzw(
                identity,
                _quat_conjugate_xyzw(giver_orientation),
            )
        )
        orientation_error_needle = torch.matmul(
            object_rotation.transpose(-1, -2),
            orientation_error_root.unsqueeze(-1),
        ).squeeze(-1)
        tool_error = torch.cat(
            (position_error_needle, orientation_error_needle),
            dim=-1,
        )
        force_imbalance = (
            self.failure_forces[:, 1:] - self.failure_forces[:, :1]
        )
        normalized_retry = (
            self.retry_count.to(raw.dtype).clamp(max=5.0) / 5.0
        ).unsqueeze(-1)
        remaining_time = (
            1.0
            - self.episode_step.to(raw.dtype).unsqueeze(-1)
            / float(self.episode_frames)
        ).clamp(0.0, 1.0)
        semantic = torch.cat(
            (
                object_position,
                rotation_6d,
                tool_error,
                self.failure_forces,
                force_imbalance,
                self.failure_loss_flags,
                self.ever_bilateral.to(raw.dtype).unsqueeze(-1),
                normalized_retry,
                remaining_time,
            ),
            dim=-1,
        )
        previous = torch.cat(
            (
                self.correction[:, :3] / self.position_cap_m,
                self.correction[:, 3:] / self.orientation_cap_rad,
            ),
            dim=-1,
        )
        context = torch.cat((semantic, previous), dim=-1)
        if context.shape[-1] != self.context_dim:
            raise RuntimeError(
                f"pickup recovery context drifted to {context.shape[-1]}"
            )
        return context

    def _activate_recovery(
        self,
        raw: torch.Tensor,
        giver_is_robot_1: torch.Tensor,
        activation: torch.Tensor,
    ) -> None:
        if not bool(activation.any()):
            return
        context = self._recovery_context(raw, giver_is_robot_1)
        proposed_normalized = self.recovery_head(context)
        proposed_delta = torch.cat(
            (
                proposed_normalized[:, :3] * self.position_cap_m,
                proposed_normalized[:, 3:] * self.orientation_cap_rad,
            ),
            dim=-1,
        )
        proposed = self.correction + proposed_delta
        if self._fixed_correction is not None:
            fixed = self._fixed_correction.to(
                device=raw.device,
                dtype=raw.dtype,
            )
            if fixed.ndim == 1:
                fixed = fixed.expand(raw.shape[0], -1)
            if fixed.shape[0] != raw.shape[0]:
                raise ValueError(
                    "fixed correction batch does not match policy batch"
                )
            if self._fixed_correction_after_first_retry is not None:
                later = self._fixed_correction_after_first_retry.to(
                    device=raw.device,
                    dtype=raw.dtype,
                )
                if later.ndim == 1:
                    later = later.expand(raw.shape[0], -1)
                if later.shape[0] != raw.shape[0]:
                    raise ValueError(
                        "later correction batch does not match policy batch"
                    )
                fixed = torch.where(
                    (self.retry_count > 1).unsqueeze(-1),
                    later,
                    fixed,
                )
            proposed = fixed
        elif self._fixed_correction_delta is not None:
            fixed_delta = self._fixed_correction_delta.to(
                device=raw.device,
                dtype=raw.dtype,
            )
            if fixed_delta.ndim == 1:
                fixed_delta = fixed_delta.expand(raw.shape[0], -1)
            if fixed_delta.shape[0] != raw.shape[0]:
                raise ValueError(
                    "fixed correction delta batch does not match policy batch"
                )
            proposed = proposed + fixed_delta
        if self._correction_candidates is not None:
            candidates = self._correction_candidates.to(
                device=raw.device,
                dtype=raw.dtype,
            )
            normalized_proposed = torch.cat(
                (
                    proposed[:, :3] / self.position_cap_m,
                    proposed[:, 3:] / self.orientation_cap_rad,
                ),
                dim=-1,
            )
            normalized_candidates = torch.cat(
                (
                    candidates[:, :3] / self.position_cap_m,
                    candidates[:, 3:] / self.orientation_cap_rad,
                ),
                dim=-1,
            )
            nearest = torch.argmin(
                (
                    normalized_proposed.unsqueeze(1)
                    - normalized_candidates.unsqueeze(0)
                )
                .square()
                .sum(dim=-1),
                dim=-1,
            )
            proposed = candidates[nearest]
        projected = torch.cat(
            (
                _project_vector(proposed[:, :3], self.position_cap_m),
                _project_vector(
                    proposed[:, 3:],
                    self.orientation_cap_rad,
                ),
            ),
            dim=-1,
        )
        self.correction[activation] = projected[activation].detach()
        self.last_context = context.detach()
        self.last_activation_mask = activation.detach().clone()
        self.activation_count[activation] += 1

    def _corrected_giver_action(
        self,
        raw: torch.Tensor,
        giver_is_robot_1: torch.Tensor,
    ) -> torch.Tensor:
        giver_ee = self._select_role(
            raw[:, 32:35],
            raw[:, 39:42],
            giver_is_robot_1,
        )
        giver_orientation = self._select_role(
            raw[:, 35:39],
            raw[:, 42:46],
            giver_is_robot_1,
        )
        object_pose = self._select_role(
            raw[:, 46:53],
            raw[:, 53:60],
            giver_is_robot_1,
        )
        object_position = object_pose[:, :3]
        object_rotation = _quat_xyzw_to_matrix(object_pose[:, 3:7])
        base_offset = self.canonical_grasp_offset.to(
            dtype=raw.dtype,
            device=raw.device,
        ).expand(raw.shape[0], -1)
        correction_in_giver = torch.matmul(
            object_rotation,
            self.correction[:, :3].unsqueeze(-1),
        ).squeeze(-1)
        grasp_position = object_position + base_offset + correction_in_giver
        delta = grasp_position - giver_ee
        lateral_distance = torch.linalg.vector_norm(
            delta[:, :2],
            dim=-1,
        )
        above = grasp_position.clone()
        above[:, 2] += self.approach_height
        target = torch.where(
            (
                lateral_distance > self.lateral_alignment_threshold
            ).unsqueeze(-1),
            above,
            grasp_position,
        )
        distance = torch.linalg.vector_norm(delta, dim=-1)
        translation = ((target - giver_ee) / self.position_scale).clamp(
            -1.0,
            1.0,
        )
        translation = torch.where(
            (distance < self.slow_approach_radius).unsqueeze(-1),
            translation.clamp(
                -self.slow_approach_action_limit,
                self.slow_approach_action_limit,
            ),
            translation,
        )

        orientation_delta_root = torch.matmul(
            object_rotation,
            self.correction[:, 3:].unsqueeze(-1),
        ).squeeze(-1)
        target_orientation = _axis_angle_to_quat_xyzw(
            orientation_delta_root
        )
        orientation_error = _quat_xyzw_to_axis_angle(
            _quat_multiply_xyzw(
                target_orientation,
                _quat_conjugate_xyzw(giver_orientation),
            )
        )
        orientation = (
            orientation_error / self.orientation_scale
        ).clamp(
            -self.orientation_action_limit,
            self.orientation_action_limit,
        )
        orientation_ready = (
            torch.linalg.vector_norm(orientation_error, dim=-1)
            < self.orientation_tolerance
        )
        pregrasp_position = grasp_position.clone()
        pregrasp_position[:, 2] += self.approach_height
        orientation_wait_translation = (
            (pregrasp_position - giver_ee) / self.position_scale
        ).clamp(-1.0, 1.0)
        translation = torch.where(
            (~orientation_ready).unsqueeze(-1),
            orientation_wait_translation,
            translation,
        )
        giver_contacts = self._select_role(
            raw[:, 66:68],
            raw[:, 68:70],
            giver_is_robot_1,
        )
        any_contact = torch.any(
            giver_contacts > self.normalized_contact_threshold,
            dim=-1,
        )
        gripper = torch.where(
            (
                ((distance < self.close_distance) & orientation_ready)
                | any_contact
            ),
            -torch.ones_like(distance),
            torch.ones_like(distance),
        ).unsqueeze(-1)
        return torch.cat((translation, orientation, gripper), dim=-1)

    def _replace_giver_action(
        self,
        base_action: torch.Tensor,
        giver_action: torch.Tensor,
        giver_is_robot_1: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        robot_1 = base_action[:, :7]
        robot_2 = base_action[:, 7:14]
        robot_1 = torch.where(
            (active & giver_is_robot_1).unsqueeze(-1),
            giver_action,
            robot_1,
        )
        robot_2 = torch.where(
            (active & ~giver_is_robot_1).unsqueeze(-1),
            giver_action,
            robot_2,
        )
        return torch.cat((robot_1, robot_2), dim=-1)

    def forward(
        self,
        obs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        raw = obs["policy"]
        if self._batch_size != raw.shape[0]:
            self._initialize_state(raw)
        base_action = self.base_policy(obs)
        self.episode_step += 1
        self.last_activation_mask = torch.zeros(
            raw.shape[0],
            dtype=torch.bool,
            device=raw.device,
        )

        giver_is_robot_1 = raw[:, 82] > 0.5
        giver_contacts = self._select_role(
            raw[:, 66:68],
            raw[:, 68:70],
            giver_is_robot_1,
        )
        bilateral_live = torch.all(
            giver_contacts > self.normalized_contact_threshold,
            dim=-1,
        )
        self.bilateral_contact_history = torch.roll(
            self.bilateral_contact_history,
            shifts=-1,
            dims=-1,
        )
        self.bilateral_contact_history[:, -1] = bilateral_live
        bilateral_qualified = (
            self.bilateral_contact_history.sum(dim=-1) >= 3
        )
        any_contact = torch.any(
            giver_contacts > self.normalized_contact_threshold,
            dim=-1,
        )
        phase = torch.argmax(raw[:, 77:82], dim=-1)
        self.ever_bilateral |= bilateral_qualified | (phase >= 1)

        giver_joint_displacement = self._select_role(
            raw[:, 6:8],
            raw[:, 22:24],
            giver_is_robot_1,
        ).abs()
        previous_giver_gripper_action = torch.where(
            giver_is_robot_1,
            raw[:, 90],
            raw[:, 97],
        )
        pickup_not_lifted = phase <= 1
        secure_now = (
            pickup_not_lifted
            & (phase >= 1)
            & bilateral_qualified
        )
        recovered_now = secure_now & (self.retry_count > 0)
        self.recovered_custody |= recovered_now
        self.retry_state[secure_now] = self.state_secure
        self.close_dwell[secure_now] = 0
        self.custody_loss_dwell[:] = torch.where(
            (phase == 1) & ~bilateral_live,
            self.custody_loss_dwell + 1,
            torch.zeros_like(self.custody_loss_dwell),
        )

        first_or_retry_approach = (
            (self.retry_state == self.state_canonical)
            | (self.retry_state == self.state_learned_retry)
        )
        closing = (
            pickup_not_lifted
            & first_or_retry_approach
            & (previous_giver_gripper_action < 0.0)
        )
        corrected_action = self._corrected_giver_action(
            raw,
            giver_is_robot_1,
        )
        self.close_dwell[:] = torch.where(
            closing,
            self.close_dwell + 1,
            torch.zeros_like(self.close_dwell),
        )
        # Starting this timer when the close command begins would preempt the
        # qualified 0.50 -> 0.07 rad jaw motion: at the configured 1 rad/s
        # limit, the jaws cannot physically finish that travel in 15 control
        # steps.  Require both the 15-step dwell and the existing full-close
        # declaration, which preserves every valid incumbent first attempt.
        missed_after_dwell = (
            pickup_not_lifted
            & first_or_retry_approach
            & ~bilateral_qualified
            & (self.close_dwell >= self.close_dwell_steps)
            & torch.all(
                giver_joint_displacement
                >= self.closed_joint_displacement_rad,
                dim=-1,
            )
        )
        lost_after_custody = (
            (phase == 1)
            & (self.retry_state == self.state_secure)
            & self.ever_bilateral
            & (self.custody_loss_dwell >= self.custody_loss_steps)
        )
        failure = (
            missed_after_dwell
            | lost_after_custody
        ) & (self.retry_state != self.state_failed) & (
            self.retry_state != self.state_reopening
        ) & (
            self.retry_state != self.state_open_settle
        )
        if bool(failure.any()):
            self.failure_forces[failure] = giver_contacts[
                failure
            ].clamp(0.0, 1.0)
            self.failure_loss_flags[failure] = (
                giver_contacts[failure] <= self.normalized_contact_threshold
            ).to(raw.dtype)
            self.first_attempt_failed |= failure & (self.retry_count == 0)
            self.retry_state[failure] = self.state_failed
            self.open_settle_dwell[failure] = 0
            self.close_dwell[failure] = 0
            self.custody_loss_dwell[failure] = 0

        # A three-step loss declares failure.  Keep the closed jaws stationary
        # until the five-step bilateral filter is empty, so a late contact
        # sample cannot turn the required reopening into a premature release.
        failed_grasp = self.retry_state == self.state_failed
        ready_to_reopen = failed_grasp & ~torch.any(
            self.bilateral_contact_history,
            dim=-1,
        )
        self.retry_state[ready_to_reopen] = self.state_reopening
        failed_grasp = self.retry_state == self.state_failed
        resetting = (
            (self.retry_state == self.state_reopening)
            | (self.retry_state == self.state_open_settle)
        )
        fully_open = torch.all(
            giver_joint_displacement
            <= self.open_joint_displacement_tolerance_rad,
            dim=-1,
        )
        force_free = ~any_contact
        ready_to_settle = resetting & fully_open & force_free
        self.retry_state[
            ready_to_settle & (self.retry_state == self.state_reopening)
        ] = self.state_open_settle
        self.open_settle_dwell[:] = torch.where(
            ready_to_settle,
            self.open_settle_dwell + 1,
            torch.zeros_like(self.open_settle_dwell),
        )
        activation = (
            (self.retry_state == self.state_open_settle)
            & (self.open_settle_dwell >= self.open_settle_steps)
        )
        if bool(activation.any()):
            self.retry_count[activation] += 1
            self.retry_state[activation] = self.state_learned_retry
            self.open_settle_dwell[activation] = 0
            self._activate_recovery(raw, giver_is_robot_1, activation)

        hold_closed_action = torch.zeros(
            (raw.shape[0], 7),
            dtype=raw.dtype,
            device=raw.device,
        )
        hold_closed_action[:, 6] = -1.0
        result = self._replace_giver_action(
            base_action,
            hold_closed_action,
            giver_is_robot_1,
            failed_grasp,
        )
        open_action = torch.zeros_like(hold_closed_action)
        open_action[:, 6] = 1.0
        result = self._replace_giver_action(
            result,
            open_action,
            giver_is_robot_1,
            resetting,
        )
        learned_retry = (
            (self.retry_state == self.state_learned_retry) & (phase <= 1)
        )
        result = self._replace_giver_action(
            result,
            corrected_action,
            giver_is_robot_1,
            learned_retry,
        )
        return result.clamp(-1.0, 1.0)

    def _clear_state(self, mask: torch.Tensor) -> None:
        self.retry_state[mask] = self.state_canonical
        self.retry_count[mask] = 0
        self.episode_step[mask] = 0
        self.close_dwell[mask] = 0
        self.custody_loss_dwell[mask] = 0
        self.open_settle_dwell[mask] = 0
        self.ever_bilateral[mask] = False
        self.bilateral_contact_history[mask] = False
        self.failure_forces[mask] = 0.0
        self.failure_loss_flags[mask] = 0.0
        self.correction[mask] = 0.0
        self.first_attempt_failed[mask] = False
        self.recovered_custody[mask] = False
        self.activation_count[mask] = 0

    def reset(
        self,
        dones: torch.Tensor | None = None,
        hidden_state: Any = None,
    ) -> None:
        reset_base = getattr(self.base_policy, "reset", None)
        if reset_base is not None:
            reset_base(dones, hidden_state)
        if self._batch_size == 0:
            return
        if dones is None:
            mask = torch.ones_like(self.retry_count, dtype=torch.bool)
        else:
            mask = dones.to(device=self.retry_count.device, dtype=torch.bool)
        self._clear_state(mask)

    @torch.jit.export
    def reset_export(self, dones: torch.Tensor) -> None:
        self.base_policy.reset_export(dones)
        if self._batch_size != 0:
            self._clear_state(
                dones.to(
                    device=self.retry_count.device,
                    dtype=torch.bool,
                )
            )

    def as_jit(self) -> nn.Module:
        return _RecoveryTensorExport(self)


class ReceiverRecoveryHead(PickupRecoveryHead):
    """Separate one-shot receiver retry head with no gripper authority."""


class ReceiverRetryGate(nn.Module):
    """Predict whether the canonical receiver approach needs an early retry."""

    input_dim = 105

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.network(context).squeeze(-1)


class ReceiverCandidateValue(nn.Module):
    """Score a receiver correction candidate for one failed-grasp state."""

    input_dim = 35

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class ReceiverContextCandidateSelector(nn.Module):
    """Select one of the frozen 16 receiver corrections from context."""

    input_dim = 29
    candidate_count = 16

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, self.candidate_count),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.network(context)


class ReceiverAttemptActorCritic(nn.Module):
    """One-decision residual actor-critic for receiver acquisition."""

    input_dim = 36
    action_dim = 6

    def __init__(self, initial_std: float = 0.25) -> None:
        super().__init__()
        if not 0.05 <= initial_std <= 0.5:
            raise ValueError(
                "receiver attempt initial std must be in [0.05, 0.5]"
            )
        self.actor = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, self.action_dim),
        )
        actor_final = self.actor[-1]
        assert isinstance(actor_final, nn.Linear)
        nn.init.zeros_(actor_final.weight)
        nn.init.zeros_(actor_final.bias)
        self.critic = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), math.log(initial_std))
        )

    def bounded_log_std(self) -> torch.Tensor:
        return self.log_std.clamp(math.log(0.05), math.log(0.5))

    def value(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.critic(features).squeeze(-1))

    def action_statistics(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor(features)
        log_std = self.bounded_log_std().to(
            device=features.device,
            dtype=features.dtype,
        )
        return mean, log_std, self.value(features)

    @staticmethod
    def _log_probability(
        pre_tanh: torch.Tensor,
        action: torch.Tensor,
        mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> torch.Tensor:
        inverse_variance = torch.exp(-2.0 * log_std)
        gaussian = (
            -0.5 * (pre_tanh - mean).square() * inverse_variance
            - log_std
            - 0.5 * math.log(2.0 * math.pi)
        ).sum(dim=-1)
        squash = torch.log(
            (1.0 - action.square()).clamp_min(1.0e-6)
        ).sum(dim=-1)
        return gaussian - squash

    def act(
        self,
        features: torch.Tensor,
        *,
        stochastic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, value = self.action_statistics(features)
        if stochastic:
            pre_tanh = mean + torch.exp(log_std) * torch.randn_like(mean)
        else:
            pre_tanh = mean
        action = torch.tanh(pre_tanh)
        return (
            action,
            self._log_probability(
                pre_tanh,
                action,
                mean,
                log_std,
            ),
            value,
        )

    def evaluate_actions(
        self,
        features: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clipped = action.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
        pre_tanh = torch.atanh(clipped)
        mean, log_std, value = self.action_statistics(features)
        log_probability = self._log_probability(
            pre_tanh,
            clipped,
            mean,
            log_std,
        )
        entropy = (
            0.5
            + 0.5 * math.log(2.0 * math.pi)
            + log_std
        ).sum().expand_as(log_probability)
        return log_probability, entropy, value


class HandoverReceiverRecoveryPolicy(nn.Module):
    """Frozen pickup composite plus isolated receiver-acquisition retries."""

    def __init__(
        self,
        base_policy: nn.Module,
        recovery_head: ReceiverRecoveryHead | None = None,
        *,
        retry_gate: ReceiverRetryGate | None = None,
        gate_feature_mean: torch.Tensor | None = None,
        gate_feature_std: torch.Tensor | None = None,
        gate_step: int = 100,
        gate_threshold: float = 0.8,
        gate_group_replicas: int = 1,
        gate_group_control_replica: bool = False,
        stabilization_gate: ReceiverRetryGate | None = None,
        stabilization_gate_feature_mean: torch.Tensor | None = None,
        stabilization_gate_feature_std: torch.Tensor | None = None,
        stabilization_gate_step: int = 100,
        stabilization_gate_threshold: float = 0.8,
        candidate_value: ReceiverCandidateValue | None = None,
        candidate_value_feature_mean: torch.Tensor | None = None,
        candidate_value_feature_std: torch.Tensor | None = None,
        candidate_corrections: torch.Tensor | None = None,
        candidate_local_offsets: torch.Tensor | None = None,
        candidate_min_logit_advantage: float = 0.0,
        candidate_selector: ReceiverContextCandidateSelector | None = None,
        candidate_selector_feature_mean: torch.Tensor | None = None,
        candidate_selector_feature_std: torch.Tensor | None = None,
        retry_candidate_portfolios: torch.Tensor | None = None,
        retry_force_imbalance_threshold: float = 0.005,
        retry_candidate_sweep_replicas: int = 1,
        retry_candidate_index: int | None = None,
        attempt_actor_critic: ReceiverAttemptActorCritic | None = None,
        attempt_feature_mean: torch.Tensor | None = None,
        attempt_feature_std: torch.Tensor | None = None,
        attempt_stochastic: bool = False,
        attempt_position_cap_m: float = 0.001,
        attempt_orientation_cap_rad: float = math.radians(1.0),
        candidate_first_attempt: bool = False,
        enable_retries: bool = True,
        stabilize_giver_during_acquisition: bool = False,
        giver_stabilization_start_step: int = 0,
        receiver_secure_settle_steps: int = 0,
        receiver_custody_confirmation_steps: int = 0,
        retry_clearance_retreat: bool = False,
        receiver_retention_contact_centering: bool = False,
        receiver_retention_servo: bool = False,
        receiver_retention_servo_gain: float = 50.0,
        receiver_retention_servo_action_limit: float = 0.02,
        position_cap_m: float = 0.005,
        orientation_cap_rad: float = math.radians(5.0),
        episode_frames: int = 2000,
    ) -> None:
        super().__init__()
        if not 0.0 < position_cap_m <= 0.005:
            raise ValueError(
                "receiver recovery position cap must be in (0, 0.005]"
            )
        if not 0.0 < orientation_cap_rad <= math.radians(5.0):
            raise ValueError(
                "receiver recovery orientation cap must be in (0, 5 degrees]"
            )
        if gate_step <= 0:
            raise ValueError("receiver retry gate step must be positive")
        if not 0.0 < gate_threshold < 1.0:
            raise ValueError(
                "receiver retry gate threshold must be inside (0, 1)"
            )
        if gate_group_replicas <= 0:
            raise ValueError(
                "receiver retry gate group replicas must be positive"
            )
        if receiver_secure_settle_steps < 0:
            raise ValueError(
                "receiver secure settle steps must be non-negative"
            )
        if receiver_custody_confirmation_steps < 0:
            raise ValueError(
                "receiver custody confirmation steps must be non-negative"
            )
        if retry_force_imbalance_threshold < 0.0:
            raise ValueError(
                "receiver retry force-imbalance threshold must be "
                "non-negative"
            )
        if retry_candidate_sweep_replicas <= 0:
            raise ValueError(
                "receiver retry candidate sweep replicas must be positive"
            )
        if retry_candidate_index is not None and retry_candidate_index < 0:
            raise ValueError(
                "receiver retry candidate index must be non-negative"
            )
        if giver_stabilization_start_step < 0:
            raise ValueError(
                "giver stabilization start step must be non-negative"
            )
        if receiver_retention_servo_gain <= 0.0:
            raise ValueError(
                "receiver retention servo gain must be positive"
            )
        if not 0.0 < receiver_retention_servo_action_limit <= 0.1:
            raise ValueError(
                "receiver retention servo action limit must be in (0, 0.1]"
            )
        if candidate_min_logit_advantage < 0.0:
            raise ValueError(
                "receiver candidate logit advantage must be non-negative"
            )
        if not 0.0 < attempt_position_cap_m <= 0.001:
            raise ValueError(
                "receiver attempt position cap must be in (0, 0.001]"
            )
        if not (
            0.0
            < attempt_orientation_cap_rad
            <= math.radians(1.0)
        ):
            raise ValueError(
                "receiver attempt orientation cap must be in (0, 1 degree]"
            )
        if stabilization_gate_step <= 0:
            raise ValueError(
                "receiver stabilization gate step must be positive"
            )
        if not 0.0 < stabilization_gate_threshold < 1.0:
            raise ValueError(
                "receiver stabilization gate threshold must be "
                "inside (0, 1)"
            )
        self.base_policy = base_policy
        self.recovery_head = recovery_head or ReceiverRecoveryHead()
        self.retry_gate = retry_gate
        if retry_gate is None:
            if gate_feature_mean is not None or gate_feature_std is not None:
                raise ValueError(
                    "receiver retry gate normalization requires a gate"
                )
            gate_feature_mean = torch.empty(0)
            gate_feature_std = torch.empty(0)
        else:
            if gate_feature_mean is None or gate_feature_std is None:
                raise ValueError(
                    "receiver retry gate requires feature normalization"
                )
            if (
                gate_feature_mean.shape != (ReceiverRetryGate.input_dim,)
                or gate_feature_std.shape != (ReceiverRetryGate.input_dim,)
            ):
                raise ValueError(
                    "receiver retry gate normalization must have 105 values"
                )
            if bool(torch.any(gate_feature_std <= 0.0)):
                raise ValueError(
                    "receiver retry gate feature standard deviations "
                    "must be positive"
                )
            for parameter in retry_gate.parameters():
                parameter.requires_grad_(False)
        self.register_buffer(
            "gate_feature_mean",
            gate_feature_mean.detach().clone(),
        )
        self.register_buffer(
            "gate_feature_std",
            gate_feature_std.detach().clone(),
        )
        self.stabilization_gate = stabilization_gate
        if stabilization_gate is None:
            if (
                stabilization_gate_feature_mean is not None
                or stabilization_gate_feature_std is not None
            ):
                raise ValueError(
                    "receiver stabilization normalization requires a gate"
                )
            stabilization_gate_feature_mean = torch.empty(0)
            stabilization_gate_feature_std = torch.empty(0)
        else:
            if (
                stabilization_gate_feature_mean is None
                or stabilization_gate_feature_std is None
                or stabilization_gate_feature_mean.shape
                != (ReceiverRetryGate.input_dim,)
                or stabilization_gate_feature_std.shape
                != (ReceiverRetryGate.input_dim,)
            ):
                raise ValueError(
                    "receiver stabilization gate normalization must "
                    "have 105 values"
                )
            if bool(
                torch.any(stabilization_gate_feature_std <= 0.0)
            ):
                raise ValueError(
                    "receiver stabilization feature standard deviations "
                    "must be positive"
                )
            for parameter in stabilization_gate.parameters():
                parameter.requires_grad_(False)
        self.register_buffer(
            "stabilization_gate_feature_mean",
            stabilization_gate_feature_mean.detach().clone(),
        )
        self.register_buffer(
            "stabilization_gate_feature_std",
            stabilization_gate_feature_std.detach().clone(),
        )
        self.candidate_value = candidate_value
        if candidate_value is None:
            if (
                candidate_value_feature_mean is not None
                or candidate_value_feature_std is not None
                or candidate_corrections is not None
                or candidate_local_offsets is not None
            ):
                raise ValueError(
                    "receiver candidate tensors require a value model"
                )
            candidate_value_feature_mean = torch.empty(0)
            candidate_value_feature_std = torch.empty(0)
            candidate_corrections = torch.empty((0, 6))
            candidate_local_offsets = torch.empty((0, 6))
        else:
            if (
                candidate_value_feature_mean is None
                or candidate_value_feature_std is None
                or candidate_corrections is None
                or candidate_value_feature_mean.shape
                != (ReceiverCandidateValue.input_dim,)
                or candidate_value_feature_std.shape
                != (ReceiverCandidateValue.input_dim,)
                or candidate_corrections.ndim != 2
                or candidate_corrections.shape[0] < 2
                or candidate_corrections.shape[1] != 6
            ):
                raise ValueError(
                    "receiver candidate value checkpoint shape drifted"
                )
            if (
                candidate_local_offsets is not None
                and candidate_local_offsets.shape != (32, 6)
            ):
                raise ValueError(
                    "receiver local candidate refinement requires 32 offsets"
                )
            if candidate_local_offsets is None:
                candidate_local_offsets = torch.empty((0, 6))
            if bool(
                torch.any(candidate_value_feature_std <= 0.0)
            ):
                raise ValueError(
                    "receiver candidate feature standard deviations "
                    "must be positive"
                )
            for parameter in candidate_value.parameters():
                parameter.requires_grad_(False)
        self.register_buffer(
            "candidate_value_feature_mean",
            candidate_value_feature_mean.detach().clone(),
        )
        self.register_buffer(
            "candidate_value_feature_std",
            candidate_value_feature_std.detach().clone(),
        )
        self.register_buffer(
            "candidate_corrections",
            candidate_corrections.detach().clone(),
        )
        self.register_buffer(
            "candidate_local_offsets",
            candidate_local_offsets.detach().clone(),
        )
        if (
            retry_candidate_sweep_replicas > 1
            and (
                candidate_value is None
                or retry_candidate_sweep_replicas
                != candidate_corrections.shape[0]
            )
        ):
            raise ValueError(
                "receiver retry candidate sweep replicas must match the "
                "frozen candidate count"
            )
        if (
            retry_candidate_index is not None
            and (
                candidate_value is None
                or retry_candidate_index >= candidate_corrections.shape[0]
            )
        ):
            raise ValueError(
                "receiver retry candidate index must select a frozen "
                "candidate"
            )
        self.retry_candidate_sweep_replicas = int(
            retry_candidate_sweep_replicas
        )
        self.retry_candidate_index = retry_candidate_index
        self.candidate_selector = candidate_selector
        if candidate_selector is None:
            if (
                candidate_selector_feature_mean is not None
                or candidate_selector_feature_std is not None
            ):
                raise ValueError(
                    "receiver selector statistics require a selector"
                )
            candidate_selector_feature_mean = torch.empty(0)
            candidate_selector_feature_std = torch.empty(0)
        else:
            if (
                candidate_value is None
                or candidate_corrections.shape
                != (ReceiverContextCandidateSelector.candidate_count, 6)
            ):
                raise ValueError(
                    "receiver context selector requires the frozen common-16 "
                    "candidate scorer"
                )
            if (
                candidate_selector_feature_mean is None
                or candidate_selector_feature_std is None
                or candidate_selector_feature_mean.shape
                != (ReceiverContextCandidateSelector.input_dim,)
                or candidate_selector_feature_std.shape
                != (ReceiverContextCandidateSelector.input_dim,)
            ):
                raise ValueError(
                    "receiver selector feature statistics must have shape "
                    "(29,)"
                )
            if bool(torch.any(candidate_selector_feature_std <= 0.0)):
                raise ValueError(
                    "receiver selector feature standard deviations must be "
                    "positive"
                )
            for parameter in candidate_selector.parameters():
                parameter.requires_grad_(False)
        self.register_buffer(
            "candidate_selector_feature_mean",
            candidate_selector_feature_mean.detach().clone(),
        )
        self.register_buffer(
            "candidate_selector_feature_std",
            candidate_selector_feature_std.detach().clone(),
        )
        if retry_candidate_portfolios is None:
            retry_candidate_portfolios = torch.empty(
                (0, 0),
                dtype=torch.long,
            )
        else:
            if (
                candidate_value is None
                or retry_candidate_portfolios.ndim != 2
                or retry_candidate_portfolios.shape[0] != 3
                or retry_candidate_portfolios.shape[1] < 1
                or retry_candidate_portfolios.dtype != torch.long
                or bool(torch.any(retry_candidate_portfolios < 0))
                or bool(
                    torch.any(
                        retry_candidate_portfolios
                        >= candidate_corrections.shape[0]
                    )
                )
            ):
                raise ValueError(
                    "receiver retry portfolios must be a 3xN long tensor "
                    "of frozen candidate indices"
                )
            for portfolio in retry_candidate_portfolios:
                if torch.unique(portfolio).numel() != portfolio.numel():
                    raise ValueError(
                        "receiver retry portfolios cannot repeat a "
                        "candidate"
                    )
        self.register_buffer(
            "retry_candidate_portfolios",
            retry_candidate_portfolios.detach().clone(),
        )
        self.retry_force_imbalance_threshold = float(
            retry_force_imbalance_threshold
        )
        self.attempt_actor_critic = attempt_actor_critic
        if attempt_actor_critic is None:
            if attempt_feature_mean is not None or attempt_feature_std is not None:
                raise ValueError(
                    "receiver attempt statistics require an actor-critic"
                )
            attempt_feature_mean = torch.empty(0)
            attempt_feature_std = torch.empty(0)
        else:
            if candidate_value is None:
                raise ValueError(
                    "receiver attempt actor requires the promoted candidate "
                    "selector"
                )
            if (
                attempt_feature_mean is None
                or attempt_feature_std is None
                or attempt_feature_mean.shape
                != (ReceiverAttemptActorCritic.input_dim,)
                or attempt_feature_std.shape
                != (ReceiverAttemptActorCritic.input_dim,)
            ):
                raise ValueError(
                    "receiver attempt feature statistics must have shape (36,)"
                )
            if bool(torch.any(attempt_feature_std <= 0.0)):
                raise ValueError(
                    "receiver attempt feature standard deviations must be "
                    "positive"
                )
            for parameter in attempt_actor_critic.parameters():
                parameter.requires_grad_(False)
        self.register_buffer(
            "attempt_feature_mean",
            attempt_feature_mean.detach().clone(),
        )
        self.register_buffer(
            "attempt_feature_std",
            attempt_feature_std.detach().clone(),
        )
        self.attempt_stochastic = bool(attempt_stochastic)
        self.attempt_position_cap_m = float(attempt_position_cap_m)
        self.attempt_orientation_cap_rad = float(
            attempt_orientation_cap_rad
        )
        self.register_buffer(
            "canonical_grasp_offset",
            torch.tensor(
                (
                    float(_RECEIVER_GRASP_OFFSET[0]),
                    float(_RECEIVER_GRASP_OFFSET[1]),
                    -0.0018,
                )
            ),
        )
        self.state_canonical = 0
        self.state_failed = 1
        self.state_reopening = 2
        self.state_open_settle = 3
        self.state_learned_retry = 4
        self.state_secure = 5
        self.context_dim = 29
        self.gate_step = int(gate_step)
        self.gate_threshold = float(gate_threshold)
        self.gate_group_replicas = int(gate_group_replicas)
        self.gate_group_control_replica = bool(
            gate_group_control_replica
        )
        self.stabilization_gate_step = int(stabilization_gate_step)
        self.stabilization_gate_threshold = float(
            stabilization_gate_threshold
        )
        self.enable_retries = bool(enable_retries)
        self.candidate_first_attempt = bool(candidate_first_attempt)
        self.candidate_min_logit_advantage = float(
            candidate_min_logit_advantage
        )
        self.stabilize_giver_during_acquisition = bool(
            stabilize_giver_during_acquisition
        )
        self.giver_stabilization_start_step = int(
            giver_stabilization_start_step
        )
        self.receiver_secure_settle_steps = int(
            receiver_secure_settle_steps
        )
        self.receiver_custody_confirmation_steps = int(
            receiver_custody_confirmation_steps
        )
        self.retry_custody_confirmation_steps = max(
            self.receiver_custody_confirmation_steps,
            5,
        )
        self.retry_clearance_retreat = bool(retry_clearance_retreat)
        self.retry_clearance_retreat_steps = 10
        self.retry_clearance_retreat_action_limit = 0.01
        self.receiver_retention_contact_centering = bool(
            receiver_retention_contact_centering
        )
        self.receiver_retention_servo = bool(receiver_retention_servo)
        self.receiver_retention_servo_gain = float(
            receiver_retention_servo_gain
        )
        self.receiver_retention_servo_action_limit = float(
            receiver_retention_servo_action_limit
        )
        self.position_cap_m = float(position_cap_m)
        self.orientation_cap_rad = float(orientation_cap_rad)
        self.episode_frames = int(episode_frames)
        for parameter in self.base_policy.parameters():
            parameter.requires_grad_(False)

        self.position_scale = 0.01
        self.orientation_scale = 0.05
        self.approach_height = 0.02
        self.lateral_alignment_threshold = 0.005
        self.close_distance = 0.001
        self.slow_approach_radius = 0.02
        self.slow_approach_action_limit = 0.1
        self.orientation_action_limit = 0.6
        self.contact_centering_action_limit = 0.0025
        self.normalized_contact_threshold = 0.002
        # A single qualified contact frame is not enough to restart motion.
        # Require three consecutive bilateral giver frames at the task's
        # physical 0.01 N qualification threshold. The screen showed that a
        # doubled 0.02 N threshold prevented every genuine retry from starting.
        # Release separately uses five consecutive live bilateral frames.
        self.normalized_giver_restore_force_margin = (
            self.normalized_contact_threshold
        )
        self.giver_restore_steps = 3
        self.close_dwell_steps = 15
        self.acquisition_timeout_steps = 500
        self.open_settle_steps = 3
        self.open_joint_displacement_tolerance_rad = 0.0215
        self.closed_joint_displacement_rad = 0.4085

        self._batch_size = 0
        self._fixed_correction: torch.Tensor | None = None
        self._fixed_correction_delta: torch.Tensor | None = None
        self.retry_state = torch.empty(0, dtype=torch.long)
        self.retry_count = torch.empty(0, dtype=torch.long)
        self.episode_step = torch.empty(0, dtype=torch.long)
        self.close_dwell = torch.empty(0, dtype=torch.long)
        self.acquisition_dwell = torch.empty(0, dtype=torch.long)
        self.acquisition_started = torch.empty(0, dtype=torch.bool)
        self.custody_loss_dwell = torch.empty(0, dtype=torch.long)
        self.giver_degradation_dwell = torch.empty(
            0,
            dtype=torch.long,
        )
        self.giver_restore_dwell = torch.empty(0, dtype=torch.long)
        self.open_settle_dwell = torch.empty(0, dtype=torch.long)
        self.ever_bilateral = torch.empty(0, dtype=torch.bool)
        self.bilateral_contact_history = torch.empty(
            (0, 5),
            dtype=torch.bool,
        )
        self.failure_forces = torch.empty((0, 2))
        self.failure_loss_flags = torch.empty((0, 2))
        self.correction = torch.empty((0, 6))
        self.first_attempt_failed = torch.empty(0, dtype=torch.bool)
        self.recovered_acquisition = torch.empty(0, dtype=torch.bool)
        self.activation_count = torch.empty(0, dtype=torch.long)
        self.last_context = torch.empty((0, self.context_dim))
        self.last_activation_mask = torch.empty(0, dtype=torch.bool)
        self.last_activation_giver_bilateral = torch.empty(
            0,
            dtype=torch.bool,
        )
        self.gate_evaluated = torch.empty(0, dtype=torch.bool)
        self.gate_triggered = torch.empty(0, dtype=torch.bool)
        self.gate_probability = torch.empty(0)
        self.first_attempt_candidate_active = torch.empty(
            0,
            dtype=torch.bool,
        )
        self.receiver_secure_settle_dwell = torch.empty(
            0,
            dtype=torch.long,
        )
        self.stabilization_gate_evaluated = torch.empty(
            0,
            dtype=torch.bool,
        )
        self.stabilization_gate_selected = torch.empty(
            0,
            dtype=torch.bool,
        )
        self.stabilization_gate_probability = torch.empty(0)
        self.selected_candidate_index = torch.empty(
            0,
            dtype=torch.long,
        )
        self.selected_candidate_score = torch.empty(0)
        self.selected_candidate_advantage = torch.empty(0)
        self.selected_candidate_applied = torch.empty(
            0,
            dtype=torch.bool,
        )
        self.applied_candidate_index = torch.empty(
            0,
            dtype=torch.long,
        )
        self.used_candidate_mask = torch.empty(
            (0, 0),
            dtype=torch.bool,
        )
        self.selected_retry_portfolio = torch.empty(
            0,
            dtype=torch.long,
        )
        self.selected_retry_portfolio_rank = torch.empty(
            0,
            dtype=torch.long,
        )
        self.receiver_secure_live_dwell = torch.empty(
            0,
            dtype=torch.long,
        )
        self.receiver_release_authorized = torch.empty(
            0,
            dtype=torch.bool,
        )
        self.giver_release_completed = torch.empty(
            0,
            dtype=torch.bool,
        )
        self.first_failure_giver_any = torch.empty(
            0,
            dtype=torch.bool,
        )
        self.first_failure_giver_bilateral = torch.empty(
            0,
            dtype=torch.bool,
        )
        self.first_failure_close_miss = torch.empty(
            0,
            dtype=torch.bool,
        )
        self.first_failure_acquisition_stall = torch.empty(
            0,
            dtype=torch.bool,
        )
        self.first_failure_receiver_loss = torch.empty(
            0,
            dtype=torch.bool,
        )
        self.first_failure_giver_degradation = torch.empty(
            0,
            dtype=torch.bool,
        )
        self.reopen_started = torch.empty(0, dtype=torch.bool)
        self.giver_restore_qualified = torch.empty(
            0,
            dtype=torch.bool,
        )
        self.retry_release_authorized = torch.empty(
            0,
            dtype=torch.bool,
        )
        self.retry_release_aborted = torch.empty(
            0,
            dtype=torch.bool,
        )
        self.receiver_retention_offset = torch.empty((0, 3))
        self.receiver_retention_offset_latched = torch.empty(
            0,
            dtype=torch.bool,
        )
        self.last_attempt_features = torch.empty(
            (0, ReceiverAttemptActorCritic.input_dim)
        )
        self.last_attempt_action = torch.empty(
            (0, ReceiverAttemptActorCritic.action_dim)
        )
        self.last_attempt_log_probability = torch.empty(0)
        self.last_attempt_value = torch.empty(0)
        self.last_attempt_baseline_correction = torch.empty((0, 6))
        self.last_attempt_baseline_advantage = torch.empty(0)

    def _initialize_state(self, raw: torch.Tensor) -> None:
        batch_size = raw.shape[0]
        device, dtype = raw.device, raw.dtype
        self._batch_size = batch_size
        self.retry_state = torch.full(
            (batch_size,),
            self.state_canonical,
            dtype=torch.long,
            device=device,
        )
        self.retry_count = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=device,
        )
        self.episode_step = torch.zeros_like(self.retry_count)
        self.close_dwell = torch.zeros_like(self.retry_count)
        self.acquisition_dwell = torch.zeros_like(self.retry_count)
        self.acquisition_started = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=device,
        )
        self.custody_loss_dwell = torch.zeros_like(self.retry_count)
        self.giver_degradation_dwell = torch.zeros_like(self.retry_count)
        self.giver_restore_dwell = torch.zeros_like(self.retry_count)
        self.open_settle_dwell = torch.zeros_like(self.retry_count)
        self.ever_bilateral = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=device,
        )
        self.bilateral_contact_history = torch.zeros(
            (batch_size, 5),
            dtype=torch.bool,
            device=device,
        )
        self.failure_forces = torch.zeros(
            (batch_size, 2),
            dtype=dtype,
            device=device,
        )
        self.failure_loss_flags = torch.zeros_like(self.failure_forces)
        self.correction = torch.zeros(
            (batch_size, 6),
            dtype=dtype,
            device=device,
        )
        self.first_attempt_failed = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=device,
        )
        self.recovered_acquisition = torch.zeros_like(
            self.first_attempt_failed
        )
        self.activation_count = torch.zeros_like(self.retry_count)
        self.last_context = torch.zeros(
            (batch_size, self.context_dim),
            dtype=dtype,
            device=device,
        )
        self.last_activation_mask = torch.zeros_like(
            self.first_attempt_failed
        )
        self.last_activation_giver_bilateral = torch.zeros_like(
            self.first_attempt_failed
        )
        self.gate_evaluated = torch.zeros_like(
            self.first_attempt_failed
        )
        self.gate_triggered = torch.zeros_like(
            self.first_attempt_failed
        )
        self.gate_probability = torch.zeros(
            batch_size,
            dtype=dtype,
            device=device,
        )
        self.first_attempt_candidate_active = torch.zeros_like(
            self.first_attempt_failed
        )
        self.receiver_secure_settle_dwell = torch.zeros_like(
            self.retry_count
        )
        self.stabilization_gate_evaluated = torch.zeros_like(
            self.first_attempt_failed
        )
        self.stabilization_gate_selected = torch.zeros_like(
            self.first_attempt_failed
        )
        self.stabilization_gate_probability = torch.zeros(
            batch_size,
            dtype=dtype,
            device=device,
        )
        self.selected_candidate_index = torch.full(
            (batch_size,),
            -1,
            dtype=torch.long,
            device=device,
        )
        self.selected_candidate_score = torch.zeros(
            batch_size,
            dtype=dtype,
            device=device,
        )
        self.selected_candidate_advantage = torch.zeros(
            batch_size,
            dtype=dtype,
            device=device,
        )
        self.selected_candidate_applied = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=device,
        )
        self.applied_candidate_index = torch.full(
            (batch_size,),
            -1,
            dtype=torch.long,
            device=device,
        )
        self.used_candidate_mask = torch.zeros(
            (batch_size, self.candidate_corrections.shape[0]),
            dtype=torch.bool,
            device=device,
        )
        self.selected_retry_portfolio = torch.full(
            (batch_size,),
            -1,
            dtype=torch.long,
            device=device,
        )
        self.selected_retry_portfolio_rank = torch.full(
            (batch_size,),
            -1,
            dtype=torch.long,
            device=device,
        )
        self.receiver_secure_live_dwell = torch.zeros_like(
            self.retry_count
        )
        self.receiver_release_authorized = torch.zeros_like(
            self.first_attempt_failed
        )
        self.giver_release_completed = torch.zeros_like(
            self.first_attempt_failed
        )
        self.first_failure_giver_any = torch.zeros_like(
            self.first_attempt_failed
        )
        self.first_failure_giver_bilateral = torch.zeros_like(
            self.first_attempt_failed
        )
        self.first_failure_close_miss = torch.zeros_like(
            self.first_attempt_failed
        )
        self.first_failure_acquisition_stall = torch.zeros_like(
            self.first_attempt_failed
        )
        self.first_failure_receiver_loss = torch.zeros_like(
            self.first_attempt_failed
        )
        self.first_failure_giver_degradation = torch.zeros_like(
            self.first_attempt_failed
        )
        self.reopen_started = torch.zeros_like(
            self.first_attempt_failed
        )
        self.giver_restore_qualified = torch.zeros_like(
            self.first_attempt_failed
        )
        self.retry_release_authorized = torch.zeros_like(
            self.first_attempt_failed
        )
        self.retry_release_aborted = torch.zeros_like(
            self.first_attempt_failed
        )
        self.receiver_retention_offset = torch.zeros(
            (batch_size, 3),
            dtype=dtype,
            device=device,
        )
        self.receiver_retention_offset_latched = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=device,
        )
        self.last_attempt_features = torch.zeros(
            (batch_size, ReceiverAttemptActorCritic.input_dim),
            dtype=dtype,
            device=device,
        )
        self.last_attempt_action = torch.zeros(
            (batch_size, ReceiverAttemptActorCritic.action_dim),
            dtype=dtype,
            device=device,
        )
        self.last_attempt_log_probability = torch.zeros(
            batch_size,
            dtype=dtype,
            device=device,
        )
        self.last_attempt_value = torch.zeros(
            batch_size,
            dtype=dtype,
            device=device,
        )
        self.last_attempt_baseline_correction = torch.zeros(
            (batch_size, 6),
            dtype=dtype,
            device=device,
        )
        self.last_attempt_baseline_advantage = torch.zeros(
            batch_size,
            dtype=dtype,
            device=device,
        )

    def set_fixed_correction(
        self,
        correction: torch.Tensor | None,
    ) -> None:
        if correction is None:
            self._fixed_correction = None
            return
        if correction.ndim not in {1, 2} or correction.shape[-1] != 6:
            raise ValueError("fixed receiver recovery correction must end in 6")
        self._fixed_correction = correction.detach().clone()
        self._fixed_correction_delta = None

    def set_fixed_correction_delta(
        self,
        correction_delta: torch.Tensor | None,
    ) -> None:
        if correction_delta is None:
            self._fixed_correction_delta = None
            return
        if (
            correction_delta.ndim not in {1, 2}
            or correction_delta.shape[-1] != 6
        ):
            raise ValueError("fixed receiver correction delta must end in 6")
        self._fixed_correction_delta = correction_delta.detach().clone()
        self._fixed_correction = None

    def _select_role(
        self,
        robot_1_value: torch.Tensor,
        robot_2_value: torch.Tensor,
        use_robot_1: torch.Tensor,
    ) -> torch.Tensor:
        return torch.where(
            use_robot_1.unsqueeze(-1),
            robot_1_value,
            robot_2_value,
        )

    def _recovery_context(
        self,
        raw: torch.Tensor,
        giver_is_robot_1: torch.Tensor,
    ) -> torch.Tensor:
        receiver_is_robot_1 = ~giver_is_robot_1
        receiver_ee = self._select_role(
            raw[:, 32:35],
            raw[:, 39:42],
            receiver_is_robot_1,
        )
        receiver_orientation = self._select_role(
            raw[:, 35:39],
            raw[:, 42:46],
            receiver_is_robot_1,
        )
        object_pose = self._select_role(
            raw[:, 46:53],
            raw[:, 53:60],
            receiver_is_robot_1,
        )
        object_position = object_pose[:, :3]
        object_rotation = _quat_xyzw_to_matrix(object_pose[:, 3:7])
        rotation_6d = torch.cat(
            (object_rotation[:, :, 0], object_rotation[:, :, 1]),
            dim=-1,
        )
        base_offset = self.canonical_grasp_offset.to(
            dtype=raw.dtype,
            device=raw.device,
        )
        canonical_position = object_position + base_offset
        position_error_receiver = canonical_position - receiver_ee
        position_error_needle = torch.matmul(
            object_rotation.transpose(-1, -2),
            position_error_receiver.unsqueeze(-1),
        ).squeeze(-1)
        giver_orientation = self._select_role(
            raw[:, 35:39],
            raw[:, 42:46],
            giver_is_robot_1,
        )
        receiver_roll = torch.zeros_like(giver_orientation)
        receiver_roll[:, 2] = 1.0
        canonical_orientation = _quat_multiply_xyzw(
            receiver_roll,
            giver_orientation,
        )
        orientation_error_receiver = _quat_xyzw_to_axis_angle(
            _quat_multiply_xyzw(
                canonical_orientation,
                _quat_conjugate_xyzw(receiver_orientation),
            )
        )
        orientation_error_needle = torch.matmul(
            object_rotation.transpose(-1, -2),
            orientation_error_receiver.unsqueeze(-1),
        ).squeeze(-1)
        force_imbalance = (
            self.failure_forces[:, 1:] - self.failure_forces[:, :1]
        )
        giver_contacts = self._select_role(
            raw[:, 66:68],
            raw[:, 68:70],
            giver_is_robot_1,
        )
        giver_custody_quality = giver_contacts.amin(
            dim=-1,
            keepdim=True,
        )
        normalized_retry_and_active = (
            self.retry_count.to(raw.dtype).clamp(max=5.0) / 5.0
        ).unsqueeze(-1)
        remaining_time = (
            1.0
            - self.episode_step.to(raw.dtype).unsqueeze(-1)
            / float(self.episode_frames)
        ).clamp(0.0, 1.0)
        semantic = torch.cat(
            (
                object_position,
                rotation_6d,
                position_error_needle,
                orientation_error_needle,
                self.failure_forces,
                force_imbalance,
                self.failure_loss_flags,
                giver_custody_quality,
                normalized_retry_and_active,
                remaining_time,
            ),
            dim=-1,
        )
        previous = torch.cat(
            (
                self.correction[:, :3] / self.position_cap_m,
                self.correction[:, 3:] / self.orientation_cap_rad,
            ),
            dim=-1,
        )
        context = torch.cat((semantic, previous), dim=-1)
        if context.shape[-1] != self.context_dim:
            raise RuntimeError(
                f"receiver recovery context drifted to {context.shape[-1]}"
            )
        return context

    def _activate_recovery(
        self,
        raw: torch.Tensor,
        giver_is_robot_1: torch.Tensor,
        activation: torch.Tensor,
    ) -> None:
        if not bool(activation.any()):
            return
        context = self._recovery_context(raw, giver_is_robot_1)
        normalized_delta = self.recovery_head(context)
        giver_contacts = self._select_role(
            raw[:, 66:68],
            raw[:, 68:70],
            giver_is_robot_1,
        )
        self.last_activation_giver_bilateral[activation] = torch.all(
            giver_contacts[activation] > self.normalized_contact_threshold,
            dim=-1,
        )
        proposed = self.correction + torch.cat(
            (
                normalized_delta[:, :3] * self.position_cap_m,
                normalized_delta[:, 3:] * self.orientation_cap_rad,
            ),
            dim=-1,
        )
        if self._fixed_correction is not None:
            fixed = self._fixed_correction.to(
                device=raw.device,
                dtype=raw.dtype,
            )
            if fixed.ndim == 1:
                fixed = fixed.expand(raw.shape[0], -1)
            if fixed.shape[0] != raw.shape[0]:
                raise ValueError(
                    "fixed receiver correction batch does not match policy"
                )
            proposed = fixed
        else:
            if self.candidate_value is not None:
                active_context = context[activation]
                candidates = self.candidate_corrections.to(
                    device=raw.device,
                    dtype=raw.dtype,
                )
                candidate_count = candidates.shape[0]
                normalized_candidates = torch.cat(
                    (
                        candidates[:, :3] / self.position_cap_m,
                        candidates[:, 3:] / self.orientation_cap_rad,
                    ),
                    dim=-1,
                )
                candidate_features = torch.cat(
                    (
                        active_context.unsqueeze(1).expand(
                            -1,
                            candidate_count,
                            -1,
                        ),
                        normalized_candidates.unsqueeze(0).expand(
                            active_context.shape[0],
                            -1,
                            -1,
                        ),
                    ),
                    dim=-1,
                )
                normalized_features = (
                    candidate_features
                    - self.candidate_value_feature_mean.to(
                        device=raw.device,
                        dtype=raw.dtype,
                    )
                ) / self.candidate_value_feature_std.to(
                    device=raw.device,
                    dtype=raw.dtype,
                )
                candidate_scores = self.candidate_value(
                    normalized_features.reshape(-1, 35)
                ).reshape(active_context.shape[0], candidate_count)
                best_score, best_index = candidate_scores.max(dim=-1)
                active_indices = torch.nonzero(
                    activation,
                    as_tuple=False,
                ).squeeze(-1)
                portfolio_active = torch.zeros(
                    active_context.shape[0],
                    dtype=torch.bool,
                    device=raw.device,
                )
                retry_sweep_active = torch.zeros_like(portfolio_active)
                if self.candidate_selector is not None:
                    normalized_context = (
                        active_context
                        - self.candidate_selector_feature_mean.to(
                            device=raw.device,
                            dtype=raw.dtype,
                        )
                    ) / self.candidate_selector_feature_std.to(
                        device=raw.device,
                        dtype=raw.dtype,
                    )
                    selector_logits = self.candidate_selector(
                        normalized_context
                    )
                    best_index = selector_logits.argmax(dim=-1)
                if (
                    self.retry_candidate_sweep_replicas > 1
                    or self.retry_candidate_index is not None
                ):
                    retry_sweep_active = (
                        self.retry_count[active_indices] > 0
                    )
                    if bool(retry_sweep_active.any()):
                        best_index = best_index.clone()
                        if self.retry_candidate_index is None:
                            best_index[retry_sweep_active] = (
                                active_indices[retry_sweep_active]
                                % self.retry_candidate_sweep_replicas
                            )
                        else:
                            best_index[retry_sweep_active] = (
                                self.retry_candidate_index
                            )
                if self.retry_candidate_portfolios.numel() > 0:
                    portfolio_active = (
                        self.retry_count[active_indices] > 0
                    )
                    if bool(portfolio_active.any()):
                        portfolio_environment = active_indices[
                            portfolio_active
                        ]
                        loss_flags = self.failure_loss_flags[
                            portfolio_environment
                        ] > 0.5
                        force_imbalance = (
                            self.failure_forces[
                                portfolio_environment,
                                1,
                            ]
                            - self.failure_forces[
                                portfolio_environment,
                                0,
                            ]
                        )
                        portfolio_id = torch.zeros(
                            portfolio_environment.shape[0],
                            dtype=torch.long,
                            device=raw.device,
                        )
                        jaw_1_missing = (
                            loss_flags[:, 0] & ~loss_flags[:, 1]
                        )
                        jaw_2_missing = (
                            loss_flags[:, 1] & ~loss_flags[:, 0]
                        )
                        portfolio_id[jaw_1_missing] = 1
                        portfolio_id[jaw_2_missing] = 2
                        unresolved = ~(jaw_1_missing | jaw_2_missing)
                        portfolio_id[
                            unresolved
                            & (
                                force_imbalance
                                > self.retry_force_imbalance_threshold
                            )
                        ] = 1
                        portfolio_id[
                            unresolved
                            & (
                                force_imbalance
                                < -self.retry_force_imbalance_threshold
                            )
                        ] = 2
                        orders = self.retry_candidate_portfolios.to(
                            device=raw.device
                        )[portfolio_id]
                        already_used = self.used_candidate_mask[
                            portfolio_environment
                        ].gather(1, orders)
                        unused = ~already_used
                        rank_axis = torch.arange(
                            orders.shape[1],
                            device=raw.device,
                        ).unsqueeze(0).expand_as(orders)
                        rank = torch.where(
                            unused,
                            rank_axis,
                            torch.full_like(
                                rank_axis,
                                orders.shape[1],
                            ),
                        ).amin(dim=-1)
                        has_candidate = rank < orders.shape[1]
                        if bool((~has_candidate).any()):
                            exhausted_environment = (
                                portfolio_environment[~has_candidate]
                            )
                            self.used_candidate_mask[
                                exhausted_environment
                            ] = False
                            rank = rank.clone()
                            rank[~has_candidate] = 0
                        chosen = orders.gather(
                            1,
                            rank.unsqueeze(-1),
                        ).squeeze(-1)
                        best_index = best_index.clone()
                        best_index[portfolio_active] = chosen
                        self.selected_retry_portfolio[
                            portfolio_environment
                        ] = portfolio_id
                        self.selected_retry_portfolio_rank[
                            portfolio_environment
                        ] = rank
                forced_retry_candidate = (
                    portfolio_active | retry_sweep_active
                )
                best_score = candidate_scores.gather(
                    1,
                    best_index.unsqueeze(-1),
                ).squeeze(-1)
                zero_index = candidates.square().sum(dim=-1).argmin()
                zero_score = candidate_scores[:, zero_index]
                proposed = proposed.clone()
                self.selected_candidate_index[active_indices] = best_index
                selected_correction = candidates[best_index]
                selected_score = best_score
                if self.candidate_local_offsets.shape[0] > 0:
                    local_candidates = (
                        candidates[best_index].unsqueeze(1)
                        + self.candidate_local_offsets.to(
                            device=raw.device,
                            dtype=raw.dtype,
                        ).unsqueeze(0)
                    )
                    local_candidates = torch.cat(
                        (
                            _project_vector(
                                local_candidates[:, :, :3].reshape(-1, 3),
                                self.position_cap_m,
                            ).reshape(-1, 32, 3),
                            _project_vector(
                                local_candidates[:, :, 3:].reshape(-1, 3),
                                self.orientation_cap_rad,
                            ).reshape(-1, 32, 3),
                        ),
                        dim=-1,
                    )
                    normalized_local = torch.cat(
                        (
                            local_candidates[:, :, :3]
                            / self.position_cap_m,
                            local_candidates[:, :, 3:]
                            / self.orientation_cap_rad,
                        ),
                        dim=-1,
                    )
                    local_features = torch.cat(
                        (
                            active_context.unsqueeze(1).expand(
                                -1,
                                32,
                                -1,
                            ),
                            normalized_local,
                        ),
                        dim=-1,
                    )
                    normalized_local_features = (
                        local_features
                        - self.candidate_value_feature_mean.to(
                            device=raw.device,
                            dtype=raw.dtype,
                        )
                    ) / self.candidate_value_feature_std.to(
                        device=raw.device,
                        dtype=raw.dtype,
                    )
                    local_scores = self.candidate_value(
                        normalized_local_features.reshape(-1, 35)
                    ).reshape(active_context.shape[0], 32)
                    local_best_score, local_best_index = local_scores.max(
                        dim=-1
                    )
                    selected_correction = local_candidates[
                        torch.arange(
                            active_context.shape[0],
                            device=raw.device,
                        ),
                        local_best_index,
                    ]
                    selected_score = local_best_score
                    selected_correction = torch.where(
                        forced_retry_candidate.unsqueeze(-1),
                        candidates[best_index],
                        selected_correction,
                    )
                    selected_score = torch.where(
                        forced_retry_candidate,
                        best_score,
                        selected_score,
                    )
                advantage = selected_score - zero_score
                candidate_applied = (
                    advantage >= self.candidate_min_logit_advantage
                ) | forced_retry_candidate
                applied_index = torch.where(
                    candidate_applied,
                    best_index,
                    zero_index.expand_as(best_index),
                )
                proposed[activation] = torch.where(
                    candidate_applied.unsqueeze(-1),
                    selected_correction,
                    candidates[zero_index].unsqueeze(0),
                )
                self.applied_candidate_index[active_indices] = (
                    applied_index
                )
                self.used_candidate_mask[
                    active_indices,
                    applied_index,
                ] = True
                self.selected_candidate_score[active_indices] = (
                    selected_score
                )
                self.selected_candidate_advantage[active_indices] = advantage
                self.selected_candidate_applied[active_indices] = (
                    candidate_applied
                )
                if self.attempt_actor_critic is not None:
                    baseline_correction = proposed[activation].clone()
                    normalized_baseline = torch.cat(
                        (
                            baseline_correction[:, :3]
                            / self.position_cap_m,
                            baseline_correction[:, 3:]
                            / self.orientation_cap_rad,
                        ),
                        dim=-1,
                    )
                    attempt_features = torch.cat(
                        (
                            active_context,
                            normalized_baseline,
                            advantage.unsqueeze(-1),
                        ),
                        dim=-1,
                    )
                    if (
                        attempt_features.shape[-1]
                        != ReceiverAttemptActorCritic.input_dim
                    ):
                        raise RuntimeError(
                            "receiver attempt feature shape drifted"
                        )
                    normalized_attempt_features = (
                        attempt_features
                        - self.attempt_feature_mean.to(
                            device=raw.device,
                            dtype=raw.dtype,
                        )
                    ) / self.attempt_feature_std.to(
                        device=raw.device,
                        dtype=raw.dtype,
                    )
                    (
                        attempt_action,
                        attempt_log_probability,
                        attempt_value,
                    ) = self.attempt_actor_critic.act(
                        normalized_attempt_features,
                        stochastic=self.attempt_stochastic,
                    )
                    attempt_residual = torch.cat(
                        (
                            attempt_action[:, :3]
                            * self.attempt_position_cap_m,
                            attempt_action[:, 3:]
                            * self.attempt_orientation_cap_rad,
                        ),
                        dim=-1,
                    )
                    proposed[activation] = baseline_correction + attempt_residual
                    self.last_attempt_features[active_indices] = (
                        attempt_features.detach()
                    )
                    self.last_attempt_action[active_indices] = (
                        attempt_action.detach()
                    )
                    self.last_attempt_log_probability[active_indices] = (
                        attempt_log_probability.detach()
                    )
                    self.last_attempt_value[active_indices] = (
                        attempt_value.detach()
                    )
                    self.last_attempt_baseline_correction[active_indices] = (
                        baseline_correction.detach()
                    )
                    self.last_attempt_baseline_advantage[active_indices] = (
                        advantage.detach()
                    )
            if self._fixed_correction_delta is not None:
                fixed_delta = self._fixed_correction_delta.to(
                    device=raw.device,
                    dtype=raw.dtype,
                )
                if fixed_delta.ndim == 1:
                    fixed_delta = fixed_delta.expand(raw.shape[0], -1)
                if fixed_delta.shape[0] != raw.shape[0]:
                    raise ValueError(
                        "fixed receiver correction delta batch does not "
                        "match policy"
                    )
                proposed = proposed + fixed_delta
        projected = torch.cat(
            (
                _project_vector(proposed[:, :3], self.position_cap_m),
                _project_vector(
                    proposed[:, 3:],
                    self.orientation_cap_rad,
                ),
            ),
            dim=-1,
        )
        self.correction[activation] = projected[activation].detach()
        self.last_context = context.detach()
        self.last_activation_mask = activation.detach().clone()
        self.activation_count[activation] += 1

    def _corrected_receiver_action(
        self,
        raw: torch.Tensor,
        giver_is_robot_1: torch.Tensor,
    ) -> torch.Tensor:
        receiver_is_robot_1 = ~giver_is_robot_1
        receiver_ee = self._select_role(
            raw[:, 32:35],
            raw[:, 39:42],
            receiver_is_robot_1,
        )
        receiver_orientation = self._select_role(
            raw[:, 35:39],
            raw[:, 42:46],
            receiver_is_robot_1,
        )
        giver_orientation = self._select_role(
            raw[:, 35:39],
            raw[:, 42:46],
            giver_is_robot_1,
        )
        object_pose = self._select_role(
            raw[:, 46:53],
            raw[:, 53:60],
            receiver_is_robot_1,
        )
        object_position = object_pose[:, :3]
        object_rotation = _quat_xyzw_to_matrix(object_pose[:, 3:7])
        base_offset = self.canonical_grasp_offset.to(
            dtype=raw.dtype,
            device=raw.device,
        )
        correction_in_receiver = torch.matmul(
            object_rotation,
            self.correction[:, :3].unsqueeze(-1),
        ).squeeze(-1)
        grasp_position = (
            object_position + base_offset + correction_in_receiver
        )
        delta = grasp_position - receiver_ee
        lateral_distance = torch.linalg.vector_norm(
            delta[:, :2],
            dim=-1,
        )
        above = grasp_position.clone()
        above[:, 2] += self.approach_height
        target = torch.where(
            (
                lateral_distance > self.lateral_alignment_threshold
            ).unsqueeze(-1),
            above,
            grasp_position,
        )
        distance = torch.linalg.vector_norm(delta, dim=-1)
        translation = ((target - receiver_ee) / self.position_scale).clamp(
            -1.0,
            1.0,
        )
        translation = torch.where(
            (distance < self.slow_approach_radius).unsqueeze(-1),
            translation.clamp(
                -self.slow_approach_action_limit,
                self.slow_approach_action_limit,
            ),
            translation,
        )

        receiver_roll = torch.zeros_like(giver_orientation)
        receiver_roll[:, 2] = 1.0
        canonical_orientation = _quat_multiply_xyzw(
            receiver_roll,
            giver_orientation,
        )
        orientation_delta_receiver = torch.matmul(
            object_rotation,
            self.correction[:, 3:].unsqueeze(-1),
        ).squeeze(-1)
        target_orientation = _quat_multiply_xyzw(
            _axis_angle_to_quat_xyzw(orientation_delta_receiver),
            canonical_orientation,
        )
        orientation_error = _quat_xyzw_to_axis_angle(
            _quat_multiply_xyzw(
                target_orientation,
                _quat_conjugate_xyzw(receiver_orientation),
            )
        )
        orientation = (
            orientation_error / self.orientation_scale
        ).clamp(
            -self.orientation_action_limit,
            self.orientation_action_limit,
        )
        receiver_contacts = self._select_role(
            raw[:, 66:68],
            raw[:, 68:70],
            receiver_is_robot_1,
        )
        any_contact = torch.any(
            receiver_contacts > self.normalized_contact_threshold,
            dim=-1,
        )
        translation = torch.where(
            any_contact.unsqueeze(-1),
            torch.zeros_like(translation),
            translation,
        )
        contact_imbalance = (
            receiver_contacts[:, 1] - receiver_contacts[:, 0]
        )
        translation[:, 2] += torch.where(
            any_contact,
            torch.sign(contact_imbalance)
            * self.contact_centering_action_limit,
            torch.zeros_like(contact_imbalance),
        )
        orientation = torch.where(
            any_contact.unsqueeze(-1),
            torch.zeros_like(orientation),
            orientation,
        )
        gripper = torch.where(
            (distance < self.close_distance) | any_contact,
            -torch.ones_like(distance),
            torch.ones_like(distance),
        ).unsqueeze(-1)
        return torch.cat((translation, orientation, gripper), dim=-1)

    def _replace_role_action(
        self,
        base_action: torch.Tensor,
        role_action: torch.Tensor,
        role_is_robot_1: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        robot_1 = torch.where(
            (active & role_is_robot_1).unsqueeze(-1),
            role_action,
            base_action[:, :7],
        )
        robot_2 = torch.where(
            (active & ~role_is_robot_1).unsqueeze(-1),
            role_action,
            base_action[:, 7:14],
        )
        return torch.cat((robot_1, robot_2), dim=-1)

    def forward(
        self,
        obs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        raw = obs["policy"]
        if self._batch_size != raw.shape[0]:
            self._initialize_state(raw)
        base_action = self.base_policy(obs)
        self.episode_step += 1
        self.last_activation_mask = torch.zeros(
            raw.shape[0],
            dtype=torch.bool,
            device=raw.device,
        )
        self.last_activation_giver_bilateral[:] = False
        giver_is_robot_1 = raw[:, 82] > 0.5
        receiver_is_robot_1 = ~giver_is_robot_1
        phase = torch.argmax(raw[:, 77:82], dim=-1)
        receiver_contacts = self._select_role(
            raw[:, 66:68],
            raw[:, 68:70],
            receiver_is_robot_1,
        )
        giver_contacts = self._select_role(
            raw[:, 66:68],
            raw[:, 68:70],
            giver_is_robot_1,
        )
        giver_any_contact = torch.any(
            giver_contacts > self.normalized_contact_threshold,
            dim=-1,
        )
        giver_bilateral_live = torch.all(
            giver_contacts > self.normalized_contact_threshold,
            dim=-1,
        )
        giver_force_margin_live = torch.all(
            giver_contacts >= self.normalized_giver_restore_force_margin,
            dim=-1,
        )
        receiver_ee_position = self._select_role(
            raw[:, 32:35],
            raw[:, 39:42],
            receiver_is_robot_1,
        )
        object_position_receiver = self._select_role(
            raw[:, 46:49],
            raw[:, 53:56],
            receiver_is_robot_1,
        )
        receiver_relative_offset = (
            object_position_receiver - receiver_ee_position
        )
        retention_entry = (
            (phase == 3) & ~self.receiver_retention_offset_latched
        )
        self.receiver_retention_offset[retention_entry] = (
            receiver_relative_offset[retention_entry]
        )
        self.receiver_retention_offset_latched |= phase >= 3
        bilateral_live = torch.all(
            receiver_contacts > self.normalized_contact_threshold,
            dim=-1,
        )
        self.bilateral_contact_history = torch.roll(
            self.bilateral_contact_history,
            shifts=-1,
            dims=-1,
        )
        self.bilateral_contact_history[:, -1] = bilateral_live
        bilateral_qualified = (
            self.bilateral_contact_history.sum(dim=-1) >= 3
        )
        self.ever_bilateral |= bilateral_qualified
        any_contact = torch.any(
            receiver_contacts > self.normalized_contact_threshold,
            dim=-1,
        )
        secure_now = (
            ((phase == 2) | (phase == 3))
            & bilateral_qualified
            & bilateral_live
        )
        self.first_attempt_candidate_active[secure_now] = False
        self.recovered_acquisition |= (
            secure_now & (self.retry_count > 0)
        )
        self.retry_state[secure_now] = self.state_secure
        self.close_dwell[secure_now] = 0
        self.acquisition_dwell[secure_now] = 0
        custody_guard_active = (
            (
                (phase == 2)
                & (self.retry_state == self.state_secure)
            )
            | (
                (phase == 3)
                & giver_any_contact
                & ~self.giver_release_completed
            )
        )
        self.custody_loss_dwell[:] = torch.where(
            custody_guard_active & ~bilateral_live,
            self.custody_loss_dwell + 1,
            torch.zeros_like(self.custody_loss_dwell),
        )
        basic_receiver_custody = (
            (phase == 3)
            & (self.retry_state == self.state_secure)
            & bilateral_live
            & giver_any_contact
            & ~self.giver_release_completed
        )
        retry_transfer = self.retry_count > 0
        # Retry activation already proves three consecutive bilateral giver
        # frames. Once the receiver closes, load transfer can legitimately
        # unload either giver jaw. Requiring giver bilateral contact throughout
        # receiver confirmation deadlocked every observed retry release.
        confirming_receiver_custody = basic_receiver_custody
        self.receiver_secure_live_dwell[:] = torch.where(
            confirming_receiver_custody,
            self.receiver_secure_live_dwell + 1,
            torch.zeros_like(self.receiver_secure_live_dwell),
        )
        retry_confirmation_steps = torch.where(
            self.retry_count > 0,
            torch.full_like(
                self.retry_count,
                self.retry_custody_confirmation_steps,
            ),
            torch.full_like(
                self.retry_count,
                self.receiver_custody_confirmation_steps,
            ),
        )
        release_authorized_now = (
            confirming_receiver_custody
            & (
                self.receiver_secure_live_dwell
                >= retry_confirmation_steps
            )
        )
        self.receiver_release_authorized |= release_authorized_now
        self.retry_release_authorized |= (
            release_authorized_now & retry_transfer
        )
        contact_lost_before_release = (
            (phase == 3)
            & ~bilateral_live
            & ~self.giver_release_completed
        )
        release_aborted = contact_lost_before_release
        self.receiver_release_authorized[
            release_aborted
        ] = False
        self.retry_release_aborted |= (
            release_aborted & retry_transfer & giver_any_contact
        )
        self.giver_release_completed |= (
            (phase >= 3)
            & self.receiver_release_authorized
            & ~giver_any_contact
            & bilateral_live
        )
        receiver_retry_phase = (
            (phase == 2)
            | (
                (phase == 3)
                & giver_any_contact
                & ~self.giver_release_completed
            )
        )

        receiver_joint_displacement = self._select_role(
            raw[:, 6:8],
            raw[:, 22:24],
            receiver_is_robot_1,
        ).abs()
        previous_receiver_gripper_action = torch.where(
            receiver_is_robot_1,
            raw[:, 90],
            raw[:, 97],
        )
        first_or_retry = (
            (self.retry_state == self.state_canonical)
            | (self.retry_state == self.state_learned_retry)
        )
        receiver_base_action = self._select_role(
            base_action[:, :7],
            base_action[:, 7:14],
            receiver_is_robot_1,
        )
        canonical_approach_commanded = (
            (self.retry_state == self.state_canonical)
            & (phase == 2)
            & (
                torch.linalg.vector_norm(
                    receiver_base_action[:, :6],
                    dim=-1,
                )
                > 1.0e-6
            )
        )
        # Phase 2 begins while the giver may still be presenting the needle.
        # Start the failure clock only when the frozen policy actually commands
        # receiver motion, so transport time cannot trigger a false retry.
        self.acquisition_started |= canonical_approach_commanded
        self.acquisition_started |= (
            (self.retry_state == self.state_learned_retry)
            & receiver_retry_phase
        )
        self.acquisition_started &= receiver_retry_phase & first_or_retry
        acquisition_active = (
            receiver_retry_phase
            & first_or_retry
            & self.acquisition_started
        )
        self.acquisition_dwell[:] = torch.where(
            acquisition_active,
            self.acquisition_dwell + 1,
            torch.zeros_like(self.acquisition_dwell),
        )
        gate_retry = torch.zeros_like(acquisition_active)
        if self.candidate_first_attempt and self.retry_gate is None:
            candidate_activation = (
                acquisition_active
                & (self.retry_state == self.state_canonical)
                & (self.retry_count == 0)
                & ~bilateral_qualified
                & ~self.gate_evaluated
                & (self.acquisition_dwell >= self.gate_step)
            )
            if bool(candidate_activation.any()):
                self.gate_evaluated[candidate_activation] = True
                self.gate_probability[candidate_activation] = 1.0
                self.failure_forces[candidate_activation] = (
                    receiver_contacts[candidate_activation].clamp(
                        0.0,
                        1.0,
                    )
                )
                self.failure_loss_flags[candidate_activation] = (
                    receiver_contacts[candidate_activation]
                    <= self.normalized_contact_threshold
                ).to(raw.dtype)
                self._activate_recovery(
                    raw,
                    giver_is_robot_1,
                    candidate_activation,
                )
                applied = candidate_activation & (
                    self.selected_candidate_applied
                    if self.candidate_value is not None
                    else torch.ones_like(candidate_activation)
                )
                self.gate_triggered[applied] = True
                self.first_attempt_candidate_active[applied] = True
        if (
            (self.enable_retries or self.candidate_first_attempt)
            and self.retry_gate is not None
        ):
            gate_now = (
                acquisition_active
                & (self.retry_state == self.state_canonical)
                & (self.retry_count == 0)
                & ~bilateral_qualified
                & ~self.gate_evaluated
                & (self.acquisition_dwell >= self.gate_step)
            )
            if bool(gate_now.any()):
                gate_features = torch.cat(
                    (raw[gate_now], receiver_base_action[gate_now]),
                    dim=-1,
                )
                normalized_gate_features = (
                    gate_features
                    - self.gate_feature_mean.to(
                        dtype=raw.dtype,
                        device=raw.device,
                    )
                ) / self.gate_feature_std.to(
                    dtype=raw.dtype,
                    device=raw.device,
                )
                gate_probability = torch.sigmoid(
                    self.retry_gate(normalized_gate_features)
                )
                self.gate_evaluated[gate_now] = True
                self.gate_probability[gate_now] = gate_probability
                selected = gate_probability >= self.gate_threshold
                selected_indices = torch.nonzero(
                    gate_now,
                    as_tuple=False,
                ).squeeze(-1)[selected]
                if (
                    self.gate_group_replicas > 1
                    and selected_indices.numel() > 0
                ):
                    if raw.shape[0] % self.gate_group_replicas:
                        raise RuntimeError(
                            "receiver gate group replicas must divide "
                            "the policy batch"
                        )
                    group_index = (
                        torch.arange(raw.shape[0], device=raw.device)
                        // self.gate_group_replicas
                    )
                    selected_groups = torch.unique(
                        selected_indices // self.gate_group_replicas
                    )
                    grouped_selected = torch.isin(
                        group_index,
                        selected_groups,
                    )
                    if self.gate_group_control_replica:
                        grouped_selected &= (
                            torch.arange(
                                raw.shape[0],
                                device=raw.device,
                            )
                            % self.gate_group_replicas
                            != 0
                        )
                    selected_indices = torch.nonzero(
                        grouped_selected
                        & (phase == 2)
                        & (self.retry_state == self.state_canonical)
                        & (self.retry_count == 0)
                        & ~bilateral_qualified,
                        as_tuple=False,
                    ).squeeze(-1)
                    self.gate_evaluated[selected_indices] = True
                self.gate_triggered[selected_indices] = True
                if self.candidate_first_attempt:
                    candidate_activation = torch.zeros_like(gate_retry)
                    candidate_activation[selected_indices] = True
                    self.failure_forces[candidate_activation] = (
                        receiver_contacts[candidate_activation].clamp(
                            0.0,
                            1.0,
                        )
                    )
                    self.failure_loss_flags[candidate_activation] = (
                        receiver_contacts[candidate_activation]
                        <= self.normalized_contact_threshold
                    ).to(raw.dtype)
                    self._activate_recovery(
                        raw,
                        giver_is_robot_1,
                        candidate_activation,
                    )
                    self.first_attempt_candidate_active[
                        candidate_activation
                        & (
                            self.selected_candidate_applied
                            if self.candidate_value is not None
                            else torch.ones_like(candidate_activation)
                        )
                    ] = True
                else:
                    gate_retry[selected_indices] = True
        if self.stabilization_gate is not None:
            stabilization_gate_now = (
                acquisition_active
                & (self.retry_state == self.state_canonical)
                & (self.retry_count == 0)
                & ~bilateral_qualified
                & ~self.stabilization_gate_evaluated
                & (
                    self.acquisition_dwell
                    >= self.stabilization_gate_step
                )
            )
            if bool(stabilization_gate_now.any()):
                stabilization_features = torch.cat(
                    (
                        raw[stabilization_gate_now],
                        receiver_base_action[stabilization_gate_now],
                    ),
                    dim=-1,
                )
                normalized_stabilization_features = (
                    stabilization_features
                    - self.stabilization_gate_feature_mean.to(
                        dtype=raw.dtype,
                        device=raw.device,
                    )
                ) / self.stabilization_gate_feature_std.to(
                    dtype=raw.dtype,
                    device=raw.device,
                )
                stabilization_probability = torch.sigmoid(
                    self.stabilization_gate(
                        normalized_stabilization_features
                    )
                )
                self.stabilization_gate_evaluated[
                    stabilization_gate_now
                ] = True
                self.stabilization_gate_probability[
                    stabilization_gate_now
                ] = stabilization_probability
                selected = (
                    stabilization_probability
                    >= self.stabilization_gate_threshold
                )
                selected_indices = torch.nonzero(
                    stabilization_gate_now,
                    as_tuple=False,
                ).squeeze(-1)[selected]
                self.stabilization_gate_selected[
                    selected_indices
                ] = True
        closing = (
            receiver_retry_phase
            & first_or_retry
            & (previous_receiver_gripper_action < 0.0)
        )
        self.close_dwell[:] = torch.where(
            closing,
            self.close_dwell + 1,
            torch.zeros_like(self.close_dwell),
        )
        failed_close = (
            receiver_retry_phase
            & first_or_retry
            & ~bilateral_qualified
            & (self.close_dwell >= self.close_dwell_steps)
            & torch.all(
                receiver_joint_displacement
                >= self.closed_joint_displacement_rad,
                dim=-1,
            )
        )
        stalled_acquisition = (
            acquisition_active
            & ~bilateral_qualified
            & (
                self.acquisition_dwell
                >= self.acquisition_timeout_steps
            )
        )
        lost_after_acquisition = (
            ((phase == 2) | (phase == 3))
            & (self.retry_state == self.state_secure)
            & self.ever_bilateral
            & (self.custody_loss_dwell >= 3)
            & (
                (phase == 2)
                | (
                    giver_any_contact
                    & ~self.giver_release_completed
                )
            )
        )
        giver_degradation_observed = (
            acquisition_active
            & ~bilateral_qualified
            & giver_any_contact
            & ~giver_bilateral_live
        )
        self.giver_degradation_dwell[:] = torch.where(
            giver_degradation_observed,
            self.giver_degradation_dwell + 1,
            torch.zeros_like(self.giver_degradation_dwell),
        )
        # A one-frame giver contact flicker occurs in otherwise successful
        # transfers. Latch only persistent physical custody degradation.
        giver_custody_degrading = (
            self.giver_degradation_dwell >= 3
        )
        if self.enable_retries:
            failure = (
                failed_close
                | stalled_acquisition
                | lost_after_acquisition
                | giver_custody_degrading
                | gate_retry
            ) & (
                self.retry_state != self.state_failed
            ) & (
                self.retry_state != self.state_reopening
            ) & (
                self.retry_state != self.state_open_settle
            )
        else:
            failure = torch.zeros_like(failed_close)
        if bool(failure.any()):
            first_failure = (
                failure
                & (self.retry_count == 0)
                & ~self.first_attempt_failed
            )
            self.first_failure_giver_any[first_failure] = (
                giver_any_contact[first_failure]
            )
            self.first_failure_giver_bilateral[first_failure] = (
                giver_bilateral_live[first_failure]
            )
            self.first_failure_close_miss |= (
                first_failure & failed_close
            )
            self.first_failure_acquisition_stall |= (
                first_failure & stalled_acquisition
            )
            self.first_failure_receiver_loss |= (
                first_failure & lost_after_acquisition
            )
            self.first_failure_giver_degradation |= (
                first_failure & giver_custody_degrading
            )
            self.first_attempt_candidate_active[failure] = False
            self.failure_forces[failure] = receiver_contacts[
                failure
            ].clamp(0.0, 1.0)
            self.failure_loss_flags[failure] = (
                receiver_contacts[failure]
                <= self.normalized_contact_threshold
            ).to(raw.dtype)
            self.first_attempt_failed |= (
                failure & (self.retry_count == 0)
            )
            self.retry_state[failure] = self.state_failed
            self.open_settle_dwell[failure] = 0
            self.close_dwell[failure] = 0
            self.acquisition_dwell[failure] = 0
            self.acquisition_started[failure] = False
            self.custody_loss_dwell[failure] = 0
            self.giver_degradation_dwell[failure] = 0
            self.giver_restore_dwell[failure] = 0
            self.receiver_secure_live_dwell[failure] = 0
            self.receiver_release_authorized[failure] = False

        failed_grasp = self.retry_state == self.state_failed
        # A missed receiver grasp can unload one giver jaw before the needle
        # drops. Reopen immediately while commanding the giver closed, then
        # wait for restored bilateral giver custody before the next approach.
        ready_to_reopen = failed_grasp & giver_any_contact
        self.retry_state[ready_to_reopen] = self.state_reopening
        self.reopen_started |= ready_to_reopen
        failed_grasp = self.retry_state == self.state_failed
        resetting = (
            (self.retry_state == self.state_reopening)
            | (self.retry_state == self.state_open_settle)
        )
        fully_open = torch.all(
            receiver_joint_displacement
            <= self.open_joint_displacement_tolerance_rad,
            dim=-1,
        )
        force_free = ~any_contact
        ready_to_settle = resetting & fully_open & force_free
        self.retry_state[
            ready_to_settle & (self.retry_state == self.state_reopening)
        ] = self.state_open_settle
        self.open_settle_dwell[:] = torch.where(
            ready_to_settle,
            self.open_settle_dwell + 1,
            torch.zeros_like(self.open_settle_dwell),
        )
        giver_restored = (
            (self.retry_state == self.state_open_settle)
            & giver_bilateral_live
            & giver_force_margin_live
        )
        self.giver_restore_dwell[:] = torch.where(
            giver_restored,
            self.giver_restore_dwell + 1,
            torch.zeros_like(self.giver_restore_dwell),
        )
        activation = (
            (self.retry_state == self.state_open_settle)
            & (self.open_settle_dwell >= self.open_settle_steps)
            & (self.giver_restore_dwell >= self.giver_restore_steps)
        )
        if bool(activation.any()):
            self.giver_restore_qualified |= activation
            self.retry_count[activation] += 1
            self.retry_state[activation] = self.state_learned_retry
            self.open_settle_dwell[activation] = 0
            self.giver_restore_dwell[activation] = 0
            self.acquisition_dwell[activation] = 0
            self.acquisition_started[activation] = True
            self._activate_recovery(raw, giver_is_robot_1, activation)

        receiver_hold_closed = torch.zeros(
            (raw.shape[0], 7),
            dtype=raw.dtype,
            device=raw.device,
        )
        receiver_hold_closed[:, 6] = -1.0
        receiver_open = torch.zeros_like(receiver_hold_closed)
        receiver_open[:, 6] = 1.0
        giver_hold = torch.zeros_like(receiver_hold_closed)
        giver_hold[:, 6] = -1.0
        stabilize_selected = self.stabilization_gate_selected
        if self.stabilize_giver_during_acquisition:
            stabilize_selected = stabilize_selected | (
                self.acquisition_dwell
                >= self.giver_stabilization_start_step
            )
        stabilize_giver = (
            stabilize_selected
            & (phase == 2)
            & self.acquisition_started
        )
        receiver_settle_phase = phase == 3
        self.receiver_secure_settle_dwell[:] = torch.where(
            receiver_settle_phase,
            self.receiver_secure_settle_dwell + 1,
            torch.zeros_like(self.receiver_secure_settle_dwell),
        )
        receiver_secure_settling = (
            (phase == 3)
            & (
                self.receiver_secure_settle_dwell
                <= self.receiver_secure_settle_steps
            )
            & (self.receiver_secure_settle_steps > 0)
        )
        recovery_active = (
            failed_grasp
            | resetting
            | (
                (self.retry_state == self.state_learned_retry)
                & receiver_retry_phase
            )
        )
        pre_release_custody_hold = (
            (phase == 3)
            & giver_any_contact
            & ~self.receiver_release_authorized
            & ~self.giver_release_completed
            & ~failed_grasp
            & ~resetting
        )
        result = self._replace_role_action(
            base_action,
            giver_hold,
            giver_is_robot_1,
            (
                recovery_active
                | stabilize_giver
                | receiver_secure_settling
                | pre_release_custody_hold
            ),
        )
        result = self._replace_role_action(
            result,
            receiver_hold_closed,
            receiver_is_robot_1,
            failed_grasp,
        )
        result = self._replace_role_action(
            result,
            receiver_open,
            receiver_is_robot_1,
            resetting,
        )
        retry_clearance_retreat_active = (
            self.retry_clearance_retreat
            & resetting
            & fully_open
            & force_free
            & giver_any_contact
            & ~giver_bilateral_live
            & (
                self.open_settle_dwell
                <= self.retry_clearance_retreat_steps
            )
        )
        retreat_direction = -receiver_relative_offset
        retreat_direction = retreat_direction / torch.linalg.vector_norm(
            retreat_direction,
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0e-6)
        retry_clearance_retreat_action = receiver_open.clone()
        retry_clearance_retreat_action[:, :3] = (
            retreat_direction
            * self.retry_clearance_retreat_action_limit
        )
        result = self._replace_role_action(
            result,
            retry_clearance_retreat_action,
            receiver_is_robot_1,
            retry_clearance_retreat_active,
        )
        result = self._replace_role_action(
            result,
            receiver_hold_closed,
            receiver_is_robot_1,
            receiver_secure_settling | pre_release_custody_hold,
        )
        corrected_receiver_action = self._corrected_receiver_action(
            raw,
            giver_is_robot_1,
        )
        learned_retry = (
            (self.retry_state == self.state_learned_retry)
            & receiver_retry_phase
        )
        corrected_approach = (
            learned_retry
            | (
                self.first_attempt_candidate_active
                & (phase == 2)
            )
        )
        result = self._replace_role_action(
            result,
            corrected_receiver_action,
            receiver_is_robot_1,
            corrected_approach,
        )
        retention_servo_active = (
            self.receiver_retention_servo
            & (phase == 3)
            & self.receiver_retention_offset_latched
            & ~giver_any_contact
        )
        retention_servo_action = torch.zeros_like(receiver_hold_closed)
        retention_servo_action[:, :3] = (
            (
                receiver_relative_offset
                - self.receiver_retention_offset
            )
            * self.receiver_retention_servo_gain
        ).clamp(
            -self.receiver_retention_servo_action_limit,
            self.receiver_retention_servo_action_limit,
        )
        retention_servo_action[:, 6] = -1.0
        result = self._replace_role_action(
            result,
            retention_servo_action,
            receiver_is_robot_1,
            retention_servo_active,
        )
        retention_contact_centering_active = (
            self.receiver_retention_contact_centering
            & (phase == 3)
            & ~giver_any_contact
            & any_contact
        )
        retention_contact_centering_action = torch.zeros_like(
            receiver_hold_closed
        )
        retention_contact_centering_action[:, 2] = (
            torch.sign(receiver_contacts[:, 1] - receiver_contacts[:, 0])
            * self.contact_centering_action_limit
        )
        retention_contact_centering_action[:, 6] = -1.0
        result = self._replace_role_action(
            result,
            retention_contact_centering_action,
            receiver_is_robot_1,
            retention_contact_centering_active,
        )
        return result.clamp(-1.0, 1.0)

    def _clear_state(self, mask: torch.Tensor) -> None:
        self.retry_state[mask] = self.state_canonical
        self.retry_count[mask] = 0
        self.episode_step[mask] = 0
        self.close_dwell[mask] = 0
        self.acquisition_dwell[mask] = 0
        self.acquisition_started[mask] = False
        self.custody_loss_dwell[mask] = 0
        self.giver_degradation_dwell[mask] = 0
        self.giver_restore_dwell[mask] = 0
        self.open_settle_dwell[mask] = 0
        self.ever_bilateral[mask] = False
        self.bilateral_contact_history[mask] = False
        self.failure_forces[mask] = 0.0
        self.failure_loss_flags[mask] = 0.0
        self.correction[mask] = 0.0
        self.first_attempt_failed[mask] = False
        self.recovered_acquisition[mask] = False
        self.activation_count[mask] = 0
        self.last_activation_giver_bilateral[mask] = False
        self.gate_evaluated[mask] = False
        self.gate_triggered[mask] = False
        self.gate_probability[mask] = 0.0
        self.first_attempt_candidate_active[mask] = False
        self.receiver_secure_settle_dwell[mask] = 0
        self.stabilization_gate_evaluated[mask] = False
        self.stabilization_gate_selected[mask] = False
        self.stabilization_gate_probability[mask] = 0.0
        self.selected_candidate_index[mask] = -1
        self.selected_candidate_score[mask] = 0.0
        self.selected_candidate_advantage[mask] = 0.0
        self.selected_candidate_applied[mask] = False
        self.applied_candidate_index[mask] = -1
        self.used_candidate_mask[mask] = False
        self.selected_retry_portfolio[mask] = -1
        self.selected_retry_portfolio_rank[mask] = -1
        self.receiver_secure_live_dwell[mask] = 0
        self.receiver_release_authorized[mask] = False
        self.giver_release_completed[mask] = False
        self.first_failure_giver_any[mask] = False
        self.first_failure_giver_bilateral[mask] = False
        self.first_failure_close_miss[mask] = False
        self.first_failure_acquisition_stall[mask] = False
        self.first_failure_receiver_loss[mask] = False
        self.first_failure_giver_degradation[mask] = False
        self.reopen_started[mask] = False
        self.giver_restore_qualified[mask] = False
        self.retry_release_authorized[mask] = False
        self.retry_release_aborted[mask] = False
        self.receiver_retention_offset[mask] = 0.0
        self.receiver_retention_offset_latched[mask] = False
        self.last_attempt_features[mask] = 0.0
        self.last_attempt_action[mask] = 0.0
        self.last_attempt_log_probability[mask] = 0.0
        self.last_attempt_value[mask] = 0.0
        self.last_attempt_baseline_correction[mask] = 0.0
        self.last_attempt_baseline_advantage[mask] = 0.0

    def reset(
        self,
        dones: torch.Tensor | None = None,
        hidden_state: Any = None,
    ) -> None:
        reset_base = getattr(self.base_policy, "reset", None)
        if reset_base is not None:
            reset_base(dones, hidden_state)
        if self._batch_size == 0:
            return
        if dones is None:
            mask = torch.ones_like(self.retry_count, dtype=torch.bool)
        else:
            mask = dones.to(device=self.retry_count.device, dtype=torch.bool)
        self._clear_state(mask)

    @torch.jit.export
    def reset_export(self, dones: torch.Tensor) -> None:
        self.base_policy.reset_export(dones)
        if self._batch_size != 0:
            self._clear_state(
                dones.to(
                    device=self.retry_count.device,
                    dtype=torch.bool,
                )
            )

    def as_jit(self) -> nn.Module:
        return _RecoveryTensorExport(self)


class _MappingTensorPolicy(nn.Module):
    """Adapt a stateless tensor-export policy to the eager observation map."""

    def __init__(self, tensor_policy: nn.Module) -> None:
        super().__init__()
        self.tensor_policy = tensor_policy

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.tensor_policy(obs["policy"])

    @torch.jit.export
    def reset_export(self, dones: torch.Tensor) -> None:
        return None


def _as_mapping_export(policy: nn.Module) -> nn.Module:
    if isinstance(policy, HandoverReceiverRecoveryPolicy):
        exported = copy.deepcopy(policy)
        exported.base_policy = _as_mapping_export(policy.base_policy)
        return exported
    if isinstance(policy, HandoverPickupRecoveryPolicy):
        exported = copy.deepcopy(policy)
        exported.base_policy = _as_mapping_export(policy.base_policy)
        return exported
    as_jit = getattr(policy, "as_jit", None)
    if as_jit is None:
        raise TypeError("base recovery policy does not expose as_jit")
    return _MappingTensorPolicy(as_jit())


class _RecoveryTensorExport(nn.Module):
    """Tensor-only stateful export with per-environment reset support."""

    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.policy = _as_mapping_export(policy)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.policy({"policy": observation})

    @torch.jit.export
    def reset(self, dones: torch.Tensor) -> None:
        self.policy.reset_export(dones)
