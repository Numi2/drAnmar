# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Attempt-level learned recovery around the frozen handover policy.

The qualified 98-observation, 14-action handover actor remains the authority
for the first pickup attempt and for every post-custody phase.  This module
only supplies one bounded needle-frame grasp correction after a failed pickup
has completed a deterministic full-open reset.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from orbit.surgical.tasks.surgical.lift.grasp_frames import (
    NEEDLE_PROVISIONAL_GRASP_OFFSET_M,
)


_FIRST_ATTEMPT = 0
_REOPENING = 1
_OPEN_SETTLE = 2
_LEARNED_RETRY = 3
_SECURE_CUSTODY = 4


def _quat_xyzw_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert normalized XYZW quaternions to rotation matrices."""

    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(
        1.0e-8
    )
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
    angle = axis_angle.norm(dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    scale = torch.where(
        angle > 1.0e-8,
        torch.sin(half_angle) / angle,
        0.5 - angle.square() / 48.0,
    )
    return torch.cat((axis_angle * scale, torch.cos(half_angle)), dim=-1)


def _quat_xyzw_to_axis_angle(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / quaternion.norm(
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0e-8)
    quaternion = torch.where(
        (quaternion[:, 3:] < 0.0),
        -quaternion,
        quaternion,
    )
    vector = quaternion[:, :3]
    vector_norm = vector.norm(dim=-1, keepdim=True)
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
    norm = value.norm(dim=-1, keepdim=True)
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
        self.last_context: torch.Tensor | None = None
        self.last_activation_mask: torch.Tensor | None = None

    def _initialize_state(
        self,
        raw: torch.Tensor,
    ) -> None:
        batch_size = raw.shape[0]
        device, dtype = raw.device, raw.dtype
        self._batch_size = batch_size
        self.retry_state = torch.full(
            (batch_size,),
            _FIRST_ATTEMPT,
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
            (batch_size, PickupRecoveryHead.input_dim),
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
            return
        if correction.ndim not in {1, 2} or correction.shape[-1] != 6:
            raise ValueError("fixed pickup recovery correction must end in 6")
        self._fixed_correction = correction.detach().clone()

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
        base_offset = torch.as_tensor(
            NEEDLE_PROVISIONAL_GRASP_OFFSET_M,
            dtype=raw.dtype,
            device=raw.device,
        )
        canonical_position = object_position + torch.matmul(
            object_rotation,
            base_offset.expand(raw.shape[0], -1).unsqueeze(-1),
        ).squeeze(-1)
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
        if context.shape[-1] != PickupRecoveryHead.input_dim:
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
            proposed = fixed
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
        base_offset = torch.as_tensor(
            NEEDLE_PROVISIONAL_GRASP_OFFSET_M,
            dtype=raw.dtype,
            device=raw.device,
        ).expand(raw.shape[0], -1)
        local_offset = base_offset + self.correction[:, :3]
        grasp_position = object_position + torch.matmul(
            object_rotation,
            local_offset.unsqueeze(-1),
        ).squeeze(-1)
        delta = grasp_position - giver_ee
        lateral_distance = delta[:, :2].norm(dim=-1)
        above = grasp_position.clone()
        above[:, 2] += self.approach_height
        target = torch.where(
            (
                lateral_distance > self.lateral_alignment_threshold
            ).unsqueeze(-1),
            above,
            grasp_position,
        )
        distance = delta.norm(dim=-1)
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
            orientation_error.norm(dim=-1) < self.orientation_tolerance
        )
        gripper = torch.where(
            (distance < self.close_distance) & orientation_ready,
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
        obs: Mapping[str, torch.Tensor],
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
        any_contact = torch.any(
            giver_contacts > self.normalized_contact_threshold,
            dim=-1,
        )
        phase = torch.argmax(raw[:, 77:82], dim=-1)
        self.ever_bilateral |= bilateral_live | (phase >= 1)

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
        base_giver_gripper_action = torch.where(
            giver_is_robot_1,
            base_action[:, 6],
            base_action[:, 13],
        )

        pickup_not_lifted = phase <= 1
        secure_now = pickup_not_lifted & (phase >= 1) & bilateral_live
        recovered_now = secure_now & (self.retry_count > 0)
        self.recovered_custody |= recovered_now
        self.retry_state[secure_now] = _SECURE_CUSTODY
        self.close_dwell[secure_now] = 0
        self.custody_loss_dwell[:] = torch.where(
            (phase == 1) & ~bilateral_live,
            self.custody_loss_dwell + 1,
            torch.zeros_like(self.custody_loss_dwell),
        )

        first_or_retry_approach = (
            (self.retry_state == _FIRST_ATTEMPT)
            | (self.retry_state == _LEARNED_RETRY)
        )
        closing = (
            pickup_not_lifted
            & first_or_retry_approach
            & (base_giver_gripper_action < 0.0)
        )
        learned_retry = self.retry_state == _LEARNED_RETRY
        corrected_action = self._corrected_giver_action(
            raw,
            giver_is_robot_1,
        )
        corrected_gripper = corrected_action[:, 6]
        closing = torch.where(
            learned_retry,
            corrected_gripper < 0.0,
            closing,
        )
        self.close_dwell[:] = torch.where(
            closing,
            self.close_dwell + 1,
            torch.zeros_like(self.close_dwell),
        )
        missed_after_full_close = (
            pickup_not_lifted
            & first_or_retry_approach
            & ~any_contact
            & (previous_giver_gripper_action < 0.0)
            & torch.all(
                giver_joint_displacement
                >= self.closed_joint_displacement_rad,
                dim=-1,
            )
        )
        missed_after_dwell = (
            pickup_not_lifted
            & first_or_retry_approach
            & ~bilateral_live
            & (self.close_dwell >= self.close_dwell_steps)
        )
        lost_after_custody = (
            (phase == 1)
            & (self.custody_loss_dwell >= self.custody_loss_steps)
        )
        failure = (
            missed_after_full_close
            | missed_after_dwell
            | lost_after_custody
        ) & (self.retry_state != _REOPENING) & (
            self.retry_state != _OPEN_SETTLE
        )
        if bool(failure.any()):
            self.failure_forces[failure] = giver_contacts[
                failure
            ].clamp(0.0, 1.0)
            self.failure_loss_flags[failure] = (
                giver_contacts[failure] <= self.normalized_contact_threshold
            ).to(raw.dtype)
            self.first_attempt_failed |= failure & (self.retry_count == 0)
            self.retry_state[failure] = _REOPENING
            self.open_settle_dwell[failure] = 0
            self.close_dwell[failure] = 0
            self.custody_loss_dwell[failure] = 0

        resetting = (
            (self.retry_state == _REOPENING)
            | (self.retry_state == _OPEN_SETTLE)
        )
        fully_open = torch.all(
            giver_joint_displacement
            <= self.open_joint_displacement_tolerance_rad,
            dim=-1,
        )
        force_free = ~any_contact
        ready_to_settle = resetting & fully_open & force_free
        self.retry_state[
            ready_to_settle & (self.retry_state == _REOPENING)
        ] = _OPEN_SETTLE
        self.open_settle_dwell[:] = torch.where(
            ready_to_settle,
            self.open_settle_dwell + 1,
            torch.zeros_like(self.open_settle_dwell),
        )
        activation = (
            (self.retry_state == _OPEN_SETTLE)
            & (self.open_settle_dwell >= self.open_settle_steps)
        )
        if bool(activation.any()):
            self.retry_count[activation] += 1
            self.retry_state[activation] = _LEARNED_RETRY
            self.open_settle_dwell[activation] = 0
            self._activate_recovery(raw, giver_is_robot_1, activation)

        open_action = torch.zeros(
            (raw.shape[0], 7),
            dtype=raw.dtype,
            device=raw.device,
        )
        open_action[:, 6] = 1.0
        result = self._replace_giver_action(
            base_action,
            open_action,
            giver_is_robot_1,
            resetting,
        )
        learned_retry = (
            (self.retry_state == _LEARNED_RETRY) & (phase == 0)
        )
        result = self._replace_giver_action(
            result,
            corrected_action,
            giver_is_robot_1,
            learned_retry,
        )
        return result.clamp(-1.0, 1.0)

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
        self.retry_state[mask] = _FIRST_ATTEMPT
        self.retry_count[mask] = 0
        self.episode_step[mask] = 0
        self.close_dwell[mask] = 0
        self.custody_loss_dwell[mask] = 0
        self.open_settle_dwell[mask] = 0
        self.ever_bilateral[mask] = False
        self.failure_forces[mask] = 0.0
        self.failure_loss_flags[mask] = 0.0
        self.correction[mask] = 0.0
        self.first_attempt_failed[mask] = False
        self.recovered_custody[mask] = False
        self.activation_count[mask] = 0

