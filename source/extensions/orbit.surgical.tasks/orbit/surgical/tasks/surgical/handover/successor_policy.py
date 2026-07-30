# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Compact full-action policy for the learned handover successor."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn


SUCCESSOR_CHECKPOINT_SCHEMA = "dranmar-handover-successor-policy-1.0"
HANDOVER_OBSERVATION_DIM = 98
HANDOVER_ACTION_DIM = 14
HANDOVER_PHASE_SLICE = slice(77, 82)
HANDOVER_PHASE_COUNT = 5


class PhaseConditionedHandoverPolicy(nn.Module):
    """Predict the complete dual-arm action through one phase-gated network."""

    def __init__(
        self,
        observation_mean: torch.Tensor,
        observation_std: torch.Tensor,
        *,
        hidden_dims: Sequence[int] = (256, 256),
        head_dim: int = 128,
    ) -> None:
        super().__init__()
        if tuple(observation_mean.shape) != (HANDOVER_OBSERVATION_DIM,):
            raise ValueError("handover observation mean must have shape [98]")
        if tuple(observation_std.shape) != (HANDOVER_OBSERVATION_DIM,):
            raise ValueError("handover observation std must have shape [98]")
        if not hidden_dims or any(int(width) <= 0 for width in hidden_dims):
            raise ValueError("successor hidden dimensions must be positive")
        if head_dim <= 0:
            raise ValueError("successor phase-head dimension must be positive")

        self.hidden_dims = tuple(int(width) for width in hidden_dims)
        self.head_dim = int(head_dim)
        self.register_buffer(
            "observation_mean",
            observation_mean.detach().float().clone(),
        )
        self.register_buffer(
            "observation_std",
            observation_std.detach().float().clamp_min(1.0e-6).clone(),
        )

        layers: list[nn.Module] = []
        input_dim = HANDOVER_OBSERVATION_DIM
        for width in self.hidden_dims:
            layers.extend(
                (
                    nn.Linear(input_dim, width),
                    nn.LayerNorm(width),
                    nn.SiLU(),
                )
            )
            input_dim = width
        self.encoder = nn.Sequential(*layers)
        self.phase_heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(input_dim, self.head_dim),
                nn.SiLU(),
                nn.Linear(self.head_dim, HANDOVER_ACTION_DIM),
            )
            for _ in range(HANDOVER_PHASE_COUNT)
        )
        for head in self.phase_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    @staticmethod
    def _policy_observation(
        observation: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(observation, Mapping):
            if "policy" not in observation:
                raise ValueError("successor observation is missing the policy group")
            return observation["policy"]
        return observation

    def forward(
        self,
        observation: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        raw = self._policy_observation(observation)
        if raw.ndim != 2 or raw.shape[-1] != HANDOVER_OBSERVATION_DIM:
            raise ValueError("successor expects observations with shape [N, 98]")
        normalized = ((raw - self.observation_mean) / self.observation_std).clamp(
            -10.0,
            10.0,
        )
        latent = self.encoder(normalized)
        all_actions = torch.stack(
            [head(latent) for head in self.phase_heads],
            dim=1,
        )
        phase_index = torch.argmax(raw[:, HANDOVER_PHASE_SLICE], dim=-1)
        batch_index = torch.arange(raw.shape[0], device=raw.device)
        return torch.tanh(all_actions[batch_index, phase_index])

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Match the stateless policy interface used by the Isaac rollout."""


def load_handover_successor_checkpoint(
    path: str,
    *,
    device: str | torch.device,
) -> tuple[PhaseConditionedHandoverPolicy, dict[str, Any]]:
    """Load a fail-closed successor checkpoint and its immutable metadata."""

    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("successor checkpoint must contain a mapping")
    if payload.get("schema_version") != SUCCESSOR_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported handover successor checkpoint")
    if payload.get("observation_dim") != HANDOVER_OBSERVATION_DIM:
        raise ValueError("successor checkpoint observation contract drifted")
    if payload.get("action_dim") != HANDOVER_ACTION_DIM:
        raise ValueError("successor checkpoint action contract drifted")
    if payload.get("phase_slice") != [
        HANDOVER_PHASE_SLICE.start,
        HANDOVER_PHASE_SLICE.stop,
    ]:
        raise ValueError("successor checkpoint phase contract drifted")
    if not payload.get("training_gate_passed"):
        raise ValueError(
            "successor checkpoint lacks an accepted demonstration-data gate"
        )

    architecture = payload.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("successor checkpoint lacks architecture metadata")
    model = PhaseConditionedHandoverPolicy(
        payload["observation_mean"],
        payload["observation_std"],
        hidden_dims=architecture["hidden_dims"],
        head_dim=int(architecture["head_dim"]),
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, payload
