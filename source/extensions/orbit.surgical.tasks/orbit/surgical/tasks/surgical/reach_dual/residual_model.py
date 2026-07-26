# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Dual-arm residual actor anchored by two qualified relative-IK controllers."""

from __future__ import annotations

import copy

import torch
from torch import nn

from ..reach.residual_model import ReachResidualMLPModel


class DualReachResidualMLPModel(ReachResidualMLPModel):
    """Learn bounded coordination residuals around two analytic PSM controllers."""

    def __init__(
        self,
        *args,
        arm_1_position_error_start: int = 46,
        arm_1_orientation_error_start: int = 49,
        arm_2_position_error_start: int = 52,
        arm_2_orientation_error_start: int = 55,
        **kwargs,
    ) -> None:
        kwargs.pop("position_error_start", None)
        kwargs.pop("orientation_error_start", None)
        super().__init__(
            *args,
            position_error_start=arm_1_position_error_start,
            orientation_error_start=arm_1_orientation_error_start,
            **kwargs,
        )
        self.arm_2_position_error_start = arm_2_position_error_start
        self.arm_2_orientation_error_start = arm_2_orientation_error_start

    def _arm_action(
        self,
        raw: torch.Tensor,
        position_start: int,
        orientation_start: int,
    ) -> torch.Tensor:
        position = raw[:, position_start : position_start + 3]
        orientation = raw[:, orientation_start : orientation_start + 3]
        return torch.cat(
            (
                position / self.position_scale,
                orientation / self.orientation_scale,
            ),
            dim=-1,
        ).clamp(-1.0, 1.0)

    def _base_action(self, obs) -> torch.Tensor:
        raw = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        return torch.cat(
            (
                self._arm_action(
                    raw,
                    self.position_error_start,
                    self.orientation_error_start,
                ),
                self._arm_action(
                    raw,
                    self.arm_2_position_error_start,
                    self.arm_2_orientation_error_start,
                ),
            ),
            dim=-1,
        )

    def as_jit(self) -> nn.Module:
        return _DualReachResidualExport(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _DualReachResidualOnnxExport(self, verbose)


class _DualReachResidualExport(nn.Module):
    """TorchScript-compatible deterministic dual residual policy."""

    def __init__(self, model: DualReachResidualMLPModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module()
            if model.distribution is not None
            else nn.Identity()
        )
        self.arm_1_position_error_start = model.position_error_start
        self.arm_1_orientation_error_start = model.orientation_error_start
        self.arm_2_position_error_start = model.arm_2_position_error_start
        self.arm_2_orientation_error_start = model.arm_2_orientation_error_start
        self.position_scale = model.position_scale
        self.orientation_scale = model.orientation_scale
        self.residual_scale = model.residual_scale

    def _arm_action(
        self,
        obs: torch.Tensor,
        position_start: int,
        orientation_start: int,
    ) -> torch.Tensor:
        position = obs[:, position_start : position_start + 3]
        orientation = obs[:, orientation_start : orientation_start + 3]
        return torch.cat(
            (
                position / self.position_scale,
                orientation / self.orientation_scale,
            ),
            dim=-1,
        ).clamp(-1.0, 1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        base = torch.cat(
            (
                self._arm_action(
                    obs,
                    self.arm_1_position_error_start,
                    self.arm_1_orientation_error_start,
                ),
                self._arm_action(
                    obs,
                    self.arm_2_position_error_start,
                    self.arm_2_orientation_error_start,
                ),
            ),
            dim=-1,
        )
        residual = self.mlp(self.obs_normalizer(obs))
        output = (base + self.residual_scale * residual).clamp(-1.0, 1.0)
        return self.deterministic_output(output)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _DualReachResidualOnnxExport(_DualReachResidualExport):
    """ONNX metadata for the deterministic dual residual policy."""

    is_recurrent: bool = False

    def __init__(self, model: DualReachResidualMLPModel, verbose: bool) -> None:
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
