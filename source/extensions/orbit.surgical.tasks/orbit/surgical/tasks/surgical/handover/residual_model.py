# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Closest-arm physical handover controller with bounded learned residuals."""

from __future__ import annotations

import copy

import torch
from rsl_rl.models import MLPModel
from torch import nn

from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
)

from orbit.surgical.assets import PSM_GRIPPER_PROFILE
from orbit.surgical.tasks.surgical.lift.grasp_frames import (
    NEEDLE_PROVISIONAL_GRASP_OFFSET_M,
    needle_geometry_grasp_offset_m,
)

_RECEIVER_OFFSET = needle_geometry_grasp_offset_m(0.65)


class HandoverAnalyticController(nn.Module):
    """Exact ordered pickup, presentation, acquisition, and release base."""

    def __init__(self) -> None:
        super().__init__()
        self.position_scale = 0.01
        self.orientation_scale = 0.05
        self.approach_height = 0.02
        self.lateral_alignment_threshold = 0.005
        self.close_distance = 0.005
        self.receiver_close_distance = 0.001
        self.slow_approach_radius = 0.02
        self.slow_approach_action_limit = 0.1
        self.receiver_contact_centering_action_limit = 0.0025
        self.normalized_contact_threshold = 0.002
        self.contact_force_observation_scale = 0.2
        self.giver_lift_contact_force_threshold_n = 0.01
        self.giver_pre_lift_min_contact_jaws = 2
        self.presentation_fraction_from_giver = 0.35
        self.presentation_height_in_robot_frame = -0.13
        self.presentation_ready_tolerance = 0.005
        self.minimum_lift_height_in_robot_frame = -0.139
        self.carry_lateral_action_limit = 0.06
        self.carry_lateral_ramp_height = 0.01
        self.pickup_vertical_action_limit = 0.01
        self.pickup_initial_vertical_action_limit = 0.01
        self.pickup_deceleration_height = 0.01
        self.carry_vertical_action_limit = 0.015
        self.giver_lift_on_live_contact = True
        self.giver_pregrasp_orientation_action_limit = 0.6
        self.giver_pregrasp_orientation_tolerance = 0.035
        self.receiver_orientation_action_limit = 0.6
        gripper_travel_rad = abs(
            float(PSM_GRIPPER_PROFILE["open_rad"])
            - float(PSM_GRIPPER_PROFILE["close_rad"])
        )
        self.giver_retry_open_displacement_rad = 0.05 * gripper_travel_rad
        self.giver_retry_closed_displacement_rad = 0.95 * gripper_travel_rad
        self.giver_grasp_x = float(
            NEEDLE_PROVISIONAL_GRASP_OFFSET_M[0]
        )
        self.giver_grasp_y = float(
            NEEDLE_PROVISIONAL_GRASP_OFFSET_M[1]
        )
        self.giver_grasp_z = float(
            NEEDLE_PROVISIONAL_GRASP_OFFSET_M[2]
        )
        self.receiver_grasp_x = float(_RECEIVER_OFFSET[0])
        self.receiver_grasp_y = float(_RECEIVER_OFFSET[1])
        self.receiver_grasp_z = -0.0018

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

    def _approach_action(
        self,
        ee_position: torch.Tensor,
        object_position: torch.Tensor,
        grasp_x: float,
        grasp_y: float,
        grasp_z: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grasp_position = object_position.clone()
        grasp_position[:, 0] += grasp_x
        grasp_position[:, 1] += grasp_y
        grasp_position[:, 2] += grasp_z
        delta = grasp_position - ee_position
        lateral_distance = torch.linalg.vector_norm(
            delta[:, :2],
            dim=-1,
        )
        above = grasp_position.clone()
        above[:, 2] += self.approach_height
        target = torch.where(
            (
                lateral_distance
                > self.lateral_alignment_threshold
            ).unsqueeze(-1),
            above,
            grasp_position,
        )
        distance = torch.linalg.vector_norm(
            grasp_position - ee_position,
            dim=-1,
        )
        action = (
            (target - ee_position) / self.position_scale
        ).clamp(-1.0, 1.0)
        action = torch.where(
            (distance < self.slow_approach_radius).unsqueeze(-1),
            action.clamp(
                -self.slow_approach_action_limit,
                self.slow_approach_action_limit,
            ),
            action,
        )
        return action, distance

    def forward(
        self,
        raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        giver_is_robot_1 = raw[:, 82] > 0.5
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
        receiver_ee = self._select_role(
            raw[:, 32:35],
            raw[:, 39:42],
            ~giver_is_robot_1,
        )
        receiver_orientation = self._select_role(
            raw[:, 35:39],
            raw[:, 42:46],
            ~giver_is_robot_1,
        )
        object_pose_in_giver = self._select_role(
            raw[:, 46:53],
            raw[:, 53:60],
            giver_is_robot_1,
        )
        object_pose_in_receiver = self._select_role(
            raw[:, 46:53],
            raw[:, 53:60],
            ~giver_is_robot_1,
        )
        object_in_giver = object_pose_in_giver[:, :3]
        object_in_receiver = object_pose_in_receiver[:, :3]
        giver_contacts = self._select_role(
            raw[:, 66:68],
            raw[:, 68:70],
            giver_is_robot_1,
        )
        giver_gripper_joint_displacement = self._select_role(
            raw[:, 6:8],
            raw[:, 22:24],
            giver_is_robot_1,
        ).abs().mean(dim=-1)
        previous_giver_gripper_action = torch.where(
            giver_is_robot_1,
            raw[:, 90],
            raw[:, 97],
        )
        receiver_contacts = self._select_role(
            raw[:, 66:68],
            raw[:, 68:70],
            ~giver_is_robot_1,
        )
        phase = torch.argmax(raw[:, 77:82], dim=-1)
        giver_tool_target_orientation = torch.zeros_like(
            giver_orientation
        )
        giver_tool_target_orientation[:, 3] = 1.0
        giver_tool_orientation_error = axis_angle_from_quat(
            quat_mul(
                giver_tool_target_orientation,
                quat_conjugate(giver_orientation),
            )
        )
        giver_pregrasp_orientation_action = (
            giver_tool_orientation_error / self.orientation_scale
        ).clamp(
            -self.giver_pregrasp_orientation_action_limit,
            self.giver_pregrasp_orientation_action_limit,
        )
        giver_pregrasp_orientation_ready = (
            torch.linalg.vector_norm(
                giver_tool_orientation_error,
                dim=-1,
            )
            < self.giver_pregrasp_orientation_tolerance
        )

        giver_approach, giver_distance = self._approach_action(
            giver_ee,
            object_in_giver,
            self.giver_grasp_x,
            self.giver_grasp_y,
            self.giver_grasp_z,
        )
        giver_pregrasp_position = object_in_giver.clone()
        giver_pregrasp_position[:, 0] += self.giver_grasp_x
        giver_pregrasp_position[:, 1] += self.giver_grasp_y
        giver_pregrasp_position[:, 2] += (
            self.giver_grasp_z + self.approach_height
        )
        giver_orientation_wait_action = (
            (giver_pregrasp_position - giver_ee)
            / self.position_scale
        ).clamp(-1.0, 1.0)
        giver_approach = torch.where(
            (
                (phase == 0)
                & ~giver_pregrasp_orientation_ready
            ).unsqueeze(-1),
            giver_orientation_wait_action,
            giver_approach,
        )
        receiver_approach, receiver_distance = (
            self._approach_action(
                receiver_ee,
                object_in_receiver,
                self.receiver_grasp_x,
                self.receiver_grasp_y,
                self.receiver_grasp_z,
            )
        )
        root_2_in_giver = object_in_giver - object_in_receiver
        presentation_in_giver = (
            self.presentation_fraction_from_giver
            * root_2_in_giver
        )
        giver_target = presentation_in_giver.clone()
        giver_target[:, 2] = (
            self.presentation_height_in_robot_frame
        )
        vertical_only = (
            object_in_giver[:, 2]
            < self.minimum_lift_height_in_robot_frame
        )
        giver_target[:, :2] = torch.where(
            vertical_only.unsqueeze(-1),
            object_in_giver[:, :2],
            giver_target[:, :2],
        )
        giver_error = (
            giver_target - object_in_giver
        ) / self.position_scale
        pickup_progress = (
            (
                object_in_giver[:, 2]
                - (
                    self.minimum_lift_height_in_robot_frame
                    - self.pickup_deceleration_height
                )
            )
            / self.pickup_deceleration_height
        ).clamp(0.0, 1.0)
        pickup_deceleration_fraction = (
            pickup_progress
            * pickup_progress
            * (3.0 - 2.0 * pickup_progress)
        )
        pickup_vertical_limit = (
            self.pickup_initial_vertical_action_limit
            + (
                self.pickup_vertical_action_limit
                - self.pickup_initial_vertical_action_limit
            )
            * pickup_deceleration_fraction
        )
        giver_vertical_limit = torch.where(
            vertical_only,
            pickup_vertical_limit,
            torch.full_like(
                giver_error[:, 2],
                self.carry_vertical_action_limit,
            ),
        ).unsqueeze(-1)
        giver_vertical_action = torch.maximum(
            torch.minimum(
                giver_error[:, 2:],
                giver_vertical_limit,
            ),
            -giver_vertical_limit,
        )
        carry_ramp_fraction = (
            (
                object_in_giver[:, 2]
                - self.minimum_lift_height_in_robot_frame
            )
            / self.carry_lateral_ramp_height
        ).clamp(0.0, 1.0)
        carry_ramp_fraction = carry_ramp_fraction * carry_ramp_fraction * (
            3.0 - 2.0 * carry_ramp_fraction
        )
        carry_lateral_limit = (
            self.carry_lateral_action_limit * carry_ramp_fraction
        ).unsqueeze(-1)
        giver_lateral_action = torch.maximum(
            torch.minimum(
                giver_error[:, :2],
                carry_lateral_limit,
            ),
            -carry_lateral_limit,
        )
        giver_carry = torch.cat(
            (
                giver_lateral_action,
                giver_vertical_action,
            ),
            dim=-1,
        )
        giver_bilateral_contact = torch.all(
            giver_contacts > self.normalized_contact_threshold,
            dim=-1,
        )
        giver_lift_contact_qualified = torch.all(
            giver_contacts
            > (
                self.giver_lift_contact_force_threshold_n
                * self.contact_force_observation_scale
            ),
            dim=-1,
        )
        giver_any_contact = torch.any(
            giver_contacts > self.normalized_contact_threshold,
            dim=-1,
        )
        giver_pre_lift_contact = torch.where(
            torch.full_like(
                giver_any_contact,
                self.giver_pre_lift_min_contact_jaws == 1,
            ),
            giver_any_contact,
            giver_lift_contact_qualified,
        )
        receiver_any_contact = torch.any(
            receiver_contacts > self.normalized_contact_threshold,
            dim=-1,
        )
        receiver_bilateral_contact = torch.all(
            receiver_contacts > self.normalized_contact_threshold,
            dim=-1,
        )
        live_contact_lift_enabled = torch.full_like(
            phase,
            self.giver_lift_on_live_contact,
            dtype=torch.bool,
        )
        giver_carry_mode = (
            ((phase >= 1) & (phase <= 2))
            | ((phase == 0) & live_contact_lift_enabled)
        )
        giver_transport_active = giver_carry_mode & torch.where(
            phase <= 1,
            giver_pre_lift_contact,
            giver_bilateral_contact,
        )
        presentation_ready = (
            torch.linalg.vector_norm(
                giver_target - object_in_giver,
                dim=-1,
            )
            < self.presentation_ready_tolerance
        )
        receiver_approach_active = (
            (phase == 2)
            & presentation_ready
            & giver_bilateral_contact
            & ~receiver_any_contact
        )
        giver_retry_reopen_required = (
            (phase <= 2)
            & ~giver_any_contact
            & (previous_giver_gripper_action < 0.0)
            & (
                (
                    (phase >= 1)
                    & (giver_distance >= self.close_distance)
                )
                | (
                    (phase == 0)
                    & (
                        giver_gripper_joint_displacement
                        >= self.giver_retry_closed_displacement_rad
                    )
                )
            )
        )
        giver_retry_reopening = (
            (phase <= 2)
            & ~giver_any_contact
            & (previous_giver_gripper_action > 0.0)
            & (
                giver_gripper_joint_displacement
                > self.giver_retry_open_displacement_rad
            )
        )
        giver_retry_waiting_for_reapproach = (
            (phase <= 2)
            & ~giver_any_contact
            & (previous_giver_gripper_action > 0.0)
            & (
                giver_gripper_joint_displacement
                <= self.giver_retry_open_displacement_rad
            )
            & (giver_distance >= self.close_distance)
        )
        giver_retry_reset_active = (
            giver_retry_reopen_required
            | giver_retry_reopening
        )
        giver_retry_open_active = (
            giver_retry_reset_active
            | giver_retry_waiting_for_reapproach
        )

        giver_translation = torch.where(
            giver_transport_active.unsqueeze(-1),
            giver_carry,
            giver_approach,
        )
        # Do not begin the retry approach until the failed grasp has completed
        # its open reset. Once fully open, approach resumes and the jaws remain
        # open until the tool returns to grasp distance.
        giver_translation = torch.where(
            giver_retry_reset_active.unsqueeze(-1),
            torch.zeros_like(giver_translation),
            giver_translation,
        )
        giver_translation = torch.where(
            (
                (phase == 2)
                & giver_bilateral_contact
                & receiver_any_contact
            ).unsqueeze(-1),
            torch.zeros_like(giver_translation),
            giver_translation,
        )
        giver_retreat = torch.zeros_like(giver_translation)
        giver_retreat[:, 2] = self.carry_lateral_action_limit
        giver_release_translation = torch.where(
            ((phase == 3) & giver_any_contact).unsqueeze(-1),
            torch.zeros_like(giver_translation),
            giver_retreat,
        )
        giver_translation = torch.where(
            (phase >= 3).unsqueeze(-1),
            giver_release_translation,
            giver_translation,
        )

        receiver_translation = torch.where(
            receiver_approach_active.unsqueeze(-1),
            receiver_approach,
            torch.zeros_like(receiver_approach),
        )
        receiver_translation = torch.where(
            (phase >= 3).unsqueeze(-1),
            torch.zeros_like(receiver_translation),
            receiver_translation,
        )
        receiver_translation = torch.where(
            ((phase == 2) & receiver_any_contact).unsqueeze(-1),
            torch.zeros_like(receiver_translation),
            receiver_translation,
        )
        receiver_contact_imbalance = (
            receiver_contacts[:, 1] - receiver_contacts[:, 0]
        )
        receiver_contact_centering = (
            torch.sign(receiver_contact_imbalance)
            * self.receiver_contact_centering_action_limit
        )
        receiver_translation[:, 2] += torch.where(
            (
                (phase == 2)
                & giver_bilateral_contact
                & receiver_any_contact
            ),
            receiver_contact_centering,
            torch.zeros_like(receiver_contact_centering),
        )

        giver_closing = (
            (
                (
                    (giver_distance < self.close_distance)
                    & giver_pregrasp_orientation_ready
                )
                | giver_any_contact
                | ((phase >= 1) & (phase <= 2))
            )
            & (phase < 3)
        )
        giver_closing |= (
            (phase == 3) & ~receiver_bilateral_contact
        )
        # The previous binary action supplies hysteresis, so every miss opens
        # fully before proximity can initiate the next close without chattering.
        giver_closing &= ~giver_retry_open_active
        receiver_closing = (
            (phase >= 2)
            & (
                (
                    receiver_distance
                    < self.receiver_close_distance
                )
                | receiver_any_contact
                | (phase >= 3)
            )
        )
        giver_gripper = torch.where(
            giver_closing,
            -torch.ones_like(giver_distance),
            torch.ones_like(giver_distance),
        ).unsqueeze(-1)
        receiver_gripper = torch.where(
            receiver_closing,
            -torch.ones_like(receiver_distance),
            torch.ones_like(receiver_distance),
        ).unsqueeze(-1)

        giver_object_orientation = object_pose_in_giver[:, 3:7]
        giver_object_angular_velocity = raw[:, 63:66]
        giver_target_orientation = torch.zeros_like(
            giver_object_orientation
        )
        giver_target_orientation[:, 3] = 1.0
        giver_orientation_error = axis_angle_from_quat(
            quat_mul(
                giver_target_orientation,
                quat_conjugate(giver_object_orientation),
            )
        )
        giver_orientation_action = (
            (
                giver_orientation_error
                - 0.001 * giver_object_angular_velocity
            )
            / self.orientation_scale
        ).clamp(-0.035, 0.035)
        giver_orientation_action = torch.where(
            giver_transport_active.unsqueeze(-1),
            giver_orientation_action,
            torch.zeros_like(giver_orientation_action),
        )
        giver_orientation_action = torch.where(
            (
                (phase == 0)
                & ~giver_any_contact
            ).unsqueeze(-1),
            giver_pregrasp_orientation_action,
            giver_orientation_action,
        )

        receiver_roll = torch.zeros_like(giver_orientation)
        receiver_roll[:, 2] = 1.0
        receiver_target_orientation = quat_mul(
            receiver_roll,
            giver_orientation,
        )
        receiver_orientation_error = axis_angle_from_quat(
            quat_mul(
                receiver_target_orientation,
                quat_conjugate(receiver_orientation),
            )
        )
        receiver_orientation_action = (
            receiver_orientation_error / self.orientation_scale
        ).clamp(
            -self.receiver_orientation_action_limit,
            self.receiver_orientation_action_limit,
        )
        receiver_orientation_action = torch.where(
            receiver_approach_active.unsqueeze(-1),
            receiver_orientation_action,
            torch.zeros_like(receiver_orientation_action),
        )

        giver_action = torch.cat(
            (
                giver_translation,
                giver_orientation_action,
                giver_gripper,
            ),
            dim=-1,
        )
        receiver_action = torch.cat(
            (
                receiver_translation,
                receiver_orientation_action,
                receiver_gripper,
            ),
            dim=-1,
        )
        robot_1_action = torch.where(
            giver_is_robot_1.unsqueeze(-1),
            giver_action,
            receiver_action,
        )
        robot_2_action = torch.where(
            giver_is_robot_1.unsqueeze(-1),
            receiver_action,
            giver_action,
        )
        base_action = torch.cat(
            (robot_1_action, robot_2_action),
            dim=-1,
        ).clamp(-1.0, 1.0)

        giver_residual = torch.zeros_like(giver_action)
        giver_transport_residual = (
            (phase == 2)
            & giver_bilateral_contact
            & ~receiver_any_contact
        )
        giver_residual[:, :3] = giver_transport_residual.unsqueeze(-1)
        receiver_residual = torch.zeros_like(receiver_action)
        receiver_residual_enabled = receiver_approach_active
        receiver_residual[:, :3] = receiver_residual_enabled.unsqueeze(-1)
        no_giver_residual = torch.zeros_like(giver_residual)
        no_receiver_residual = torch.zeros_like(receiver_residual)
        robot_1_giver_residual = torch.where(
            giver_is_robot_1.unsqueeze(-1),
            giver_residual,
            no_giver_residual,
        )
        robot_2_giver_residual = torch.where(
            giver_is_robot_1.unsqueeze(-1),
            no_giver_residual,
            giver_residual,
        )
        robot_1_receiver_residual = torch.where(
            giver_is_robot_1.unsqueeze(-1),
            no_receiver_residual,
            receiver_residual,
        )
        robot_2_receiver_residual = torch.where(
            giver_is_robot_1.unsqueeze(-1),
            receiver_residual,
            no_receiver_residual,
        )
        giver_residual_mask = torch.cat(
            (robot_1_giver_residual, robot_2_giver_residual),
            dim=-1,
        ) > 0.5
        receiver_residual_mask = torch.cat(
            (
                robot_1_receiver_residual,
                robot_2_receiver_residual,
            ),
            dim=-1,
        ) > 0.5
        return (
            base_action,
            giver_residual_mask,
            receiver_residual_mask,
        )


class HandoverResidualMLPModel(MLPModel):
    """Learn only bounded translations around the exact physical sequence."""

    def __init__(
        self,
        *args,
        residual_scale: float = 0.03,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.controller = HandoverAnalyticController()
        self.residual_scale = residual_scale
        final_linear = next(
            module
            for module in reversed(self.mlp)
            if isinstance(module, nn.Linear)
        )
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)
        if self.distribution is not None:
            for parameter_name in ("std_param", "log_std_param"):
                parameter = getattr(
                    self.distribution,
                    parameter_name,
                    None,
                )
                if parameter is not None:
                    parameter.requires_grad_(False)

    def configure_giver_adaptation(self) -> None:
        """Freeze the qualified receiver policy and train giver-only rows."""
        final_linear = next(
            module
            for module in reversed(self.mlp)
            if isinstance(module, nn.Linear)
        )
        for parameter in self.mlp.parameters():
            parameter.requires_grad_(False)
        final_linear.weight.requires_grad_(True)
        final_linear.bias.requires_grad_(True)
        giver_row_mask = torch.zeros(
            final_linear.out_features,
            dtype=final_linear.weight.dtype,
            device=final_linear.weight.device,
        )
        giver_row_mask[3:6] = 1.0
        giver_row_mask[10:13] = 1.0
        final_linear.weight.register_hook(
            lambda gradient: gradient
            * giver_row_mask.unsqueeze(-1)
        )
        final_linear.bias.register_hook(
            lambda gradient: gradient * giver_row_mask
        )

    def forward(
        self,
        obs,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        latent = self.get_latent(obs, masks, hidden_state)
        raw = torch.cat(
            [obs[group] for group in self.obs_groups],
            dim=-1,
        )
        (
            base,
            giver_residual_mask,
            receiver_residual_mask,
        ) = self.controller(raw)
        network_output = torch.tanh(self.mlp(latent))
        giver_channel_output = torch.cat(
            (
                network_output[:, 3:6],
                torch.zeros_like(network_output[:, 3:7]),
                network_output[:, 10:13],
                torch.zeros_like(network_output[:, 10:14]),
            ),
            dim=-1,
        )
        residual = self.residual_scale * (
            giver_channel_output
            * giver_residual_mask.to(raw.dtype)
            + network_output
            * receiver_residual_mask.to(raw.dtype)
        )
        residual_mask = giver_residual_mask | receiver_residual_mask
        mean = (base + residual).clamp(-1.0, 1.0)
        if self.distribution is None:
            return mean
        if stochastic_output:
            self.distribution.update(mean)
            sampled = self.distribution.sample()
            return torch.where(residual_mask, sampled, mean)
        return self.distribution.deterministic_output(mean)

    def as_jit(self) -> nn.Module:
        return _HandoverResidualExport(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _HandoverResidualOnnxExport(self, verbose)


class _HandoverResidualExport(nn.Module):
    """TorchScript-compatible deterministic handover policy."""

    def __init__(self, model: HandoverResidualMLPModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.controller = copy.deepcopy(model.controller)
        self.residual_scale = model.residual_scale
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module()
            if model.distribution is not None
            else nn.Identity()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        (
            base,
            giver_residual_mask,
            receiver_residual_mask,
        ) = self.controller(obs)
        network_output = torch.tanh(
            self.mlp(self.obs_normalizer(obs))
        )
        giver_channel_output = torch.cat(
            (
                network_output[:, 3:6],
                torch.zeros_like(network_output[:, 3:7]),
                network_output[:, 10:13],
                torch.zeros_like(network_output[:, 10:14]),
            ),
            dim=-1,
        )
        residual = self.residual_scale * (
            giver_channel_output
            * giver_residual_mask.to(obs.dtype)
            + network_output
            * receiver_residual_mask.to(obs.dtype)
        )
        return self.deterministic_output(
            (base + residual).clamp(-1.0, 1.0)
        )

    @torch.jit.export
    def reset(self) -> None:
        pass


class _HandoverResidualOnnxExport(_HandoverResidualExport):
    """ONNX metadata for the deterministic residual handover policy."""

    is_recurrent: bool = False

    def __init__(
        self,
        model: HandoverResidualMLPModel,
        verbose: bool,
    ) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.input_size = model.obs_dim

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
