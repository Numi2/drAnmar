# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Residual RSL-RL actor with the qualified relative-IK controller as its base."""

from __future__ import annotations

import copy

import torch
from rsl_rl.models import MLPModel
from torch import nn


class ReachResidualMLPModel(MLPModel):
    """Learn a bounded residual around the analytic Cartesian reach controller."""

    def __init__(
        self,
        *args,
        position_error_start: int = 23,
        orientation_error_start: int = 26,
        position_scale: float = 0.01,
        orientation_scale: float = 0.05,
        residual_scale: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.position_error_start = position_error_start
        self.orientation_error_start = orientation_error_start
        self.position_scale = position_scale
        self.orientation_scale = orientation_scale
        self.residual_scale = residual_scale

        final_linear = next(
            module for module in reversed(self.mlp) if isinstance(module, nn.Linear)
        )
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def _base_action(self, obs) -> torch.Tensor:
        raw = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        position = raw[
            :,
            self.position_error_start : self.position_error_start + 3,
        ]
        orientation = raw[
            :,
            self.orientation_error_start : self.orientation_error_start + 3,
        ]
        return torch.cat(
            (
                position / self.position_scale,
                orientation / self.orientation_scale,
            ),
            dim=-1,
        ).clamp(-1.0, 1.0)

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
        return _ReachResidualExport(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _ReachResidualOnnxExport(self, verbose)


class _ReachResidualExport(nn.Module):
    """TorchScript-compatible deterministic residual policy."""

    def __init__(self, model: ReachResidualMLPModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module()
            if model.distribution is not None
            else nn.Identity()
        )
        self.position_error_start = model.position_error_start
        self.orientation_error_start = model.orientation_error_start
        self.position_scale = model.position_scale
        self.orientation_scale = model.orientation_scale
        self.residual_scale = model.residual_scale

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        position = obs[
            :,
            self.position_error_start : self.position_error_start + 3,
        ]
        orientation = obs[
            :,
            self.orientation_error_start : self.orientation_error_start + 3,
        ]
        base = torch.cat(
            (
                position / self.position_scale,
                orientation / self.orientation_scale,
            ),
            dim=-1,
        ).clamp(-1.0, 1.0)
        residual = self.mlp(self.obs_normalizer(obs))
        output = (base + self.residual_scale * residual).clamp(-1.0, 1.0)
        return self.deterministic_output(output)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _ReachResidualOnnxExport(_ReachResidualExport):
    """ONNX metadata for the deterministic residual policy."""

    is_recurrent: bool = False

    def __init__(self, model: ReachResidualMLPModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.input_size = model.obs_dim

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
