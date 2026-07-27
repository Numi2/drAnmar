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
        self.receiver_contact_centering_action_limit = 0.005
        self.normalized_contact_threshold = 0.002
        self.presentation_fraction_from_giver = 0.35
        self.presentation_height_in_robot_frame = -0.13
        self.presentation_ready_tolerance = 0.005
        self.minimum_lift_height_in_robot_frame = -0.139
        self.carry_lateral_action_limit = 0.06
        self.carry_vertical_action_limit = 0.10
        self.receiver_orientation_action_limit = 0.6
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        receiver_contacts = self._select_role(
            raw[:, 66:68],
            raw[:, 68:70],
            ~giver_is_robot_1,
        )
        phase = torch.argmax(raw[:, 77:82], dim=-1)

        giver_approach, giver_distance = self._approach_action(
            giver_ee,
            object_in_giver,
            self.giver_grasp_x,
            self.giver_grasp_y,
            self.giver_grasp_z,
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
        giver_carry = torch.cat(
            (
                giver_error[:, :2].clamp(
                    -self.carry_lateral_action_limit,
                    self.carry_lateral_action_limit,
                ),
                giver_error[:, 2:].clamp(
                    -self.carry_vertical_action_limit,
                    self.carry_vertical_action_limit,
                ),
            ),
            dim=-1,
        )
        giver_bilateral_contact = torch.all(
            giver_contacts > self.normalized_contact_threshold,
            dim=-1,
        )
        giver_any_contact = torch.any(
            giver_contacts > self.normalized_contact_threshold,
            dim=-1,
        )
        receiver_any_contact = torch.any(
            receiver_contacts > self.normalized_contact_threshold,
            dim=-1,
        )
        receiver_bilateral_contact = torch.all(
            receiver_contacts > self.normalized_contact_threshold,
            dim=-1,
        )
        giver_carry_mode = (phase >= 1) & (phase <= 2)
        presentation_ready = (
            torch.linalg.vector_norm(
                giver_target - object_in_giver,
                dim=-1,
            )
            < self.presentation_ready_tolerance
        )

        giver_translation = torch.where(
            giver_carry_mode.unsqueeze(-1),
            giver_carry,
            giver_approach,
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
            (
                (phase <= 1)
                | ((phase == 2) & ~presentation_ready)
            ).unsqueeze(-1),
            torch.zeros_like(receiver_approach),
            receiver_approach,
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
            (phase == 2) & receiver_any_contact,
            receiver_contact_centering,
            torch.zeros_like(receiver_contact_centering),
        )

        giver_closing = (
            (
                (giver_distance < self.close_distance)
                | giver_any_contact
                | ((phase >= 1) & (phase <= 2))
            )
            & (phase < 3)
        )
        giver_closing |= (
            (phase == 3) & ~receiver_bilateral_contact
        )
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
            giver_carry_mode.unsqueeze(-1),
            giver_orientation_action,
            torch.zeros_like(giver_orientation_action),
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
            (phase < 3).unsqueeze(-1),
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
        receiver_residual = torch.zeros_like(receiver_action)
        receiver_residual_enabled = (
            (phase == 2)
            & presentation_ready
            & ~receiver_any_contact
        )
        receiver_residual[:, :3] = receiver_residual_enabled.unsqueeze(-1)
        robot_1_residual = torch.where(
            giver_is_robot_1.unsqueeze(-1),
            giver_residual,
            receiver_residual,
        )
        robot_2_residual = torch.where(
            giver_is_robot_1.unsqueeze(-1),
            receiver_residual,
            giver_residual,
        )
        residual_mask = torch.cat(
            (robot_1_residual, robot_2_residual),
            dim=-1,
        ) > 0.5
        return base_action, residual_mask


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
        base, residual_mask = self.controller(raw)
        residual = (
            self.residual_scale
            * torch.tanh(self.mlp(latent))
            * residual_mask.to(raw.dtype)
        )
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
        base, residual_mask = self.controller(obs)
        residual = (
            self.residual_scale
            * torch.tanh(self.mlp(self.obs_normalizer(obs)))
            * residual_mask.to(obs.dtype)
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
