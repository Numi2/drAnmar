# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Contact-conditioned lift controller with a bounded learned residual."""

from __future__ import annotations

import copy

import torch
from rsl_rl.models import MLPModel
from torch import nn


class LiftResidualMLPModel(MLPModel):
    """Approach, grasp, and lift with physics contact as the phase transition."""

    def __init__(
        self,
        *args,
        end_effector_position_start: int = 16,
        object_position_start: int = 23,
        object_velocity_start: int = 30,
        target_position_start: int = 36,
        contact_force_start: int = 43,
        position_scale: float = 0.01,
        approach_height: float = 0.02,
        grasp_height: float = 0.0,
        lateral_alignment_threshold: float = 0.004,
        close_distance: float = 0.003,
        slow_approach_radius: float = 0.02,
        slow_approach_action_limit: float = 0.1,
        normalized_contact_threshold: float = 0.002,
        carry_angular_velocity_scale: float = 2.5,
        carry_action_limit: float = 0.1,
        residual_scale: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.end_effector_position_start = end_effector_position_start
        self.object_position_start = object_position_start
        self.object_velocity_start = object_velocity_start
        self.target_position_start = target_position_start
        self.contact_force_start = contact_force_start
        self.position_scale = position_scale
        self.approach_height = approach_height
        self.grasp_height = grasp_height
        self.lateral_alignment_threshold = lateral_alignment_threshold
        self.close_distance = close_distance
        self.slow_approach_radius = slow_approach_radius
        self.slow_approach_action_limit = slow_approach_action_limit
        self.normalized_contact_threshold = normalized_contact_threshold
        self.carry_angular_velocity_scale = carry_angular_velocity_scale
        self.carry_action_limit = carry_action_limit
        self.residual_scale = residual_scale

        final_linear = next(
            module for module in reversed(self.mlp) if isinstance(module, nn.Linear)
        )
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def _base_action_from_raw(self, raw: torch.Tensor) -> torch.Tensor:
        ee_position = raw[
            :,
            self.end_effector_position_start : self.end_effector_position_start
            + 3,
        ]
        object_position = raw[
            :,
            self.object_position_start : self.object_position_start + 3,
        ]
        target_position = raw[
            :,
            self.target_position_start : self.target_position_start + 3,
        ]
        contact_forces = raw[
            :,
            self.contact_force_start : self.contact_force_start + 2,
        ]
        object_angular_velocity = raw[
            :,
            self.object_velocity_start + 3 : self.object_velocity_start + 6,
        ]
        ee_to_object = object_position - ee_position
        lateral_distance = torch.linalg.vector_norm(ee_to_object[:, :2], dim=-1)
        above_object = object_position.clone()
        above_object[:, 2] += self.approach_height
        grasp_position = object_position.clone()
        grasp_position[:, 2] += self.grasp_height
        grasp_distance = torch.linalg.vector_norm(
            grasp_position - ee_position,
            dim=-1,
        )
        pregrasp = lateral_distance > self.lateral_alignment_threshold
        approach_position = torch.where(
            pregrasp.unsqueeze(-1),
            above_object,
            grasp_position,
        )

        bilateral_contact = torch.all(
            contact_forces > self.normalized_contact_threshold,
            dim=-1,
        )
        approach_action = (
            (approach_position - ee_position) / self.position_scale
        ).clamp(-1.0, 1.0)
        slow_approach_action = approach_action.clamp(
            -self.slow_approach_action_limit,
            self.slow_approach_action_limit,
        )
        approach_action = torch.where(
            (grasp_distance < self.slow_approach_radius).unsqueeze(-1),
            slow_approach_action,
            approach_action,
        )
        carry_action = (
            (target_position - object_position) / self.position_scale
        ).clamp(-self.carry_action_limit, self.carry_action_limit)
        translation_action = torch.where(
            bilateral_contact.unsqueeze(-1),
            carry_action,
            approach_action,
        )
        carry_orientation_action = (
            -object_angular_velocity / self.carry_angular_velocity_scale
        ).clamp(-1.0, 1.0)
        orientation_action = torch.where(
            bilateral_contact.unsqueeze(-1),
            carry_orientation_action,
            torch.zeros_like(carry_orientation_action),
        )
        body_action = torch.cat(
            (translation_action, orientation_action),
            dim=-1,
        ).clamp(-1.0, 1.0)

        closing = (
            grasp_distance < self.close_distance
        ) | torch.any(
            contact_forces > self.normalized_contact_threshold,
            dim=-1,
        )
        gripper_action = torch.where(
            closing,
            -torch.ones_like(grasp_distance),
            torch.ones_like(grasp_distance),
        ).unsqueeze(-1)
        return torch.cat((body_action, gripper_action), dim=-1)

    def _base_action(self, obs) -> torch.Tensor:
        raw = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        return self._base_action_from_raw(raw)

    def forward(
        self,
        obs,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        latent = self.get_latent(obs, masks, hidden_state)
        residual = self.mlp(latent)
        mean = (
            self._base_action(obs) + self.residual_scale * residual
        ).clamp(-1.0, 1.0)
        if self.distribution is None:
            return mean
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return self.distribution.deterministic_output(mean)

    def as_jit(self) -> nn.Module:
        return _LiftResidualExport(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _LiftResidualOnnxExport(self, verbose)


class _LiftResidualExport(nn.Module):
    """TorchScript-compatible deterministic lift policy."""

    def __init__(self, model: LiftResidualMLPModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module()
            if model.distribution is not None
            else nn.Identity()
        )
        self.end_effector_position_start = model.end_effector_position_start
        self.object_position_start = model.object_position_start
        self.object_velocity_start = model.object_velocity_start
        self.target_position_start = model.target_position_start
        self.contact_force_start = model.contact_force_start
        self.position_scale = model.position_scale
        self.approach_height = model.approach_height
        self.grasp_height = model.grasp_height
        self.lateral_alignment_threshold = model.lateral_alignment_threshold
        self.close_distance = model.close_distance
        self.slow_approach_radius = model.slow_approach_radius
        self.slow_approach_action_limit = model.slow_approach_action_limit
        self.normalized_contact_threshold = model.normalized_contact_threshold
        self.carry_angular_velocity_scale = model.carry_angular_velocity_scale
        self.carry_action_limit = model.carry_action_limit
        self.residual_scale = model.residual_scale

    def _base_action(self, obs: torch.Tensor) -> torch.Tensor:
        ee_position = obs[
            :,
            self.end_effector_position_start : self.end_effector_position_start
            + 3,
        ]
        object_position = obs[
            :,
            self.object_position_start : self.object_position_start + 3,
        ]
        target_position = obs[
            :,
            self.target_position_start : self.target_position_start + 3,
        ]
        contact_forces = obs[
            :,
            self.contact_force_start : self.contact_force_start + 2,
        ]
        object_angular_velocity = obs[
            :,
            self.object_velocity_start + 3 : self.object_velocity_start + 6,
        ]
        ee_to_object = object_position - ee_position
        lateral_distance = torch.linalg.vector_norm(ee_to_object[:, :2], dim=-1)
        above_object = object_position.clone()
        above_object[:, 2] += self.approach_height
        grasp_position = object_position.clone()
        grasp_position[:, 2] += self.grasp_height
        grasp_distance = torch.linalg.vector_norm(
            grasp_position - ee_position,
            dim=-1,
        )
        approach_position = torch.where(
            (lateral_distance > self.lateral_alignment_threshold).unsqueeze(-1),
            above_object,
            grasp_position,
        )
        bilateral_contact = torch.all(
            contact_forces > self.normalized_contact_threshold,
            dim=-1,
        )
        approach_action = (
            (approach_position - ee_position) / self.position_scale
        ).clamp(-1.0, 1.0)
        slow_approach_action = approach_action.clamp(
            -self.slow_approach_action_limit,
            self.slow_approach_action_limit,
        )
        approach_action = torch.where(
            (grasp_distance < self.slow_approach_radius).unsqueeze(-1),
            slow_approach_action,
            approach_action,
        )
        carry_action = (
            (target_position - object_position) / self.position_scale
        ).clamp(-self.carry_action_limit, self.carry_action_limit)
        translation_action = torch.where(
            bilateral_contact.unsqueeze(-1),
            carry_action,
            approach_action,
        )
        carry_orientation_action = (
            -object_angular_velocity / self.carry_angular_velocity_scale
        ).clamp(-1.0, 1.0)
        orientation_action = torch.where(
            bilateral_contact.unsqueeze(-1),
            carry_orientation_action,
            torch.zeros_like(carry_orientation_action),
        )
        body_action = torch.cat(
            (translation_action, orientation_action),
            dim=-1,
        ).clamp(-1.0, 1.0)
        closing = (
            grasp_distance < self.close_distance
        ) | torch.any(
            contact_forces > self.normalized_contact_threshold,
            dim=-1,
        )
        gripper_action = torch.where(
            closing,
            -torch.ones_like(grasp_distance),
            torch.ones_like(grasp_distance),
        ).unsqueeze(-1)
        return torch.cat((body_action, gripper_action), dim=-1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        base = self._base_action(obs)
        residual = self.mlp(self.obs_normalizer(obs))
        output = (base + self.residual_scale * residual).clamp(-1.0, 1.0)
        return self.deterministic_output(output)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _LiftResidualOnnxExport(_LiftResidualExport):
    """ONNX metadata for the deterministic lift policy."""

    is_recurrent: bool = False

    def __init__(self, model: LiftResidualMLPModel, verbose: bool) -> None:
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
