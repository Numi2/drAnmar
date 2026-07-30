# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Compact full-action policy for the learned handover successor."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn


SUCCESSOR_CHECKPOINT_SCHEMA = "dranmar-handover-successor-policy-3.0"
LEGACY_SUCCESSOR_CHECKPOINT_SCHEMAS = {
    "dranmar-handover-successor-policy-2.0",
}
HANDOVER_OBSERVATION_DIM = 98
HANDOVER_ACTION_DIM = 14
HANDOVER_PHASE_SLICE = slice(77, 82)
HANDOVER_PHASE_COUNT = 5
HANDOVER_GRIPPER_INDICES = (6, 13)
HANDOVER_CONTINUOUS_INDICES = tuple(
    index
    for index in range(HANDOVER_ACTION_DIM)
    if index not in HANDOVER_GRIPPER_INDICES
)
HANDOVER_SATURATION_CLASS_COUNT = 3
HANDOVER_SATURATION_LOGIT_MARGIN = 1.5
HANDOVER_HEAD_OUTPUT_DIM = (
    HANDOVER_ACTION_DIM
    + len(HANDOVER_CONTINUOUS_INDICES)
    * HANDOVER_SATURATION_CLASS_COUNT
)


class PhaseConditionedHandoverPolicy(nn.Module):
    """Predict the complete dual-arm action through one phase-gated network."""

    def __init__(
        self,
        observation_mean: torch.Tensor,
        observation_std: torch.Tensor,
        *,
        hidden_dims: Sequence[int] = (256, 256),
        memory_dim: int = 128,
        head_dim: int = 128,
    ) -> None:
        super().__init__()
        if tuple(observation_mean.shape) != (HANDOVER_OBSERVATION_DIM,):
            raise ValueError("handover observation mean must have shape [98]")
        if tuple(observation_std.shape) != (HANDOVER_OBSERVATION_DIM,):
            raise ValueError("handover observation std must have shape [98]")
        if not hidden_dims or any(int(width) <= 0 for width in hidden_dims):
            raise ValueError("successor hidden dimensions must be positive")
        if memory_dim <= 0:
            raise ValueError("successor memory dimension must be positive")
        if head_dim <= 0:
            raise ValueError("successor phase-head dimension must be positive")

        self.hidden_dims = tuple(int(width) for width in hidden_dims)
        self.memory_dim = int(memory_dim)
        self.head_dim = int(head_dim)
        self._runtime_hidden: torch.Tensor | None = None
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
        self.memory = nn.GRU(
            input_size=input_dim,
            hidden_size=self.memory_dim,
            batch_first=True,
        )
        self.phase_heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.memory_dim, self.head_dim),
                nn.SiLU(),
                nn.Linear(self.head_dim, HANDOVER_HEAD_OUTPUT_DIM),
            )
            for _ in range(HANDOVER_PHASE_COUNT)
        )
        for head in self.phase_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            saturation_bias = head[-1].bias[
                HANDOVER_ACTION_DIM:
            ].view(
                len(HANDOVER_CONTINUOUS_INDICES),
                HANDOVER_SATURATION_CLASS_COUNT,
            )
            with torch.no_grad():
                saturation_bias[:, 1] = 2.0

    @staticmethod
    def _policy_observation(
        observation: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(observation, Mapping):
            if "policy" not in observation:
                raise ValueError("successor observation is missing the policy group")
            return observation["policy"]
        return observation

    def _normalize(
        self,
        raw: torch.Tensor,
    ) -> torch.Tensor:
        return (
            (raw - self.observation_mean) / self.observation_std
        ).clamp(-10.0, 10.0)

    def _outputs_from_latent(
        self,
        raw: torch.Tensor,
        latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if raw.ndim != 2 or raw.shape[-1] != HANDOVER_OBSERVATION_DIM:
            raise ValueError("successor expects observations with shape [N, 98]")
        all_outputs = torch.stack(
            [head(latent) for head in self.phase_heads],
            dim=1,
        )
        phase_index = torch.argmax(raw[:, HANDOVER_PHASE_SLICE], dim=-1)
        batch_index = torch.arange(raw.shape[0], device=raw.device)
        selected = all_outputs[batch_index, phase_index]
        raw_action = selected[:, :HANDOVER_ACTION_DIM]
        saturation_logits = selected[
            :, HANDOVER_ACTION_DIM:
        ].view(
            raw.shape[0],
            len(HANDOVER_CONTINUOUS_INDICES),
            HANDOVER_SATURATION_CLASS_COUNT,
        )
        return raw_action, saturation_logits, phase_index

    def _independent_outputs(
        self,
        raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(self._normalize(raw))
        latent, _ = self.memory(encoded.unsqueeze(1))
        return self._outputs_from_latent(raw, latent[:, 0])

    def _runtime_outputs(
        self,
        raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(self._normalize(raw))
        if (
            self._runtime_hidden is None
            or self._runtime_hidden.shape[1] != raw.shape[0]
            or self._runtime_hidden.device != raw.device
            or self._runtime_hidden.dtype != encoded.dtype
        ):
            self._runtime_hidden = torch.zeros(
                1,
                raw.shape[0],
                self.memory_dim,
                device=raw.device,
                dtype=encoded.dtype,
            )
        latent, hidden = self.memory(
            encoded.unsqueeze(1),
            self._runtime_hidden,
        )
        self._runtime_hidden = hidden.detach()
        return self._outputs_from_latent(raw, latent[:, 0])

    def initial_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Create the explicit recurrent state used by exported inference."""

        if batch_size <= 0:
            raise ValueError("successor recurrent batch size must be positive")
        reference = self.observation_mean
        return torch.zeros(
            1,
            batch_size,
            self.memory_dim,
            device=device if device is not None else reference.device,
            dtype=dtype if dtype is not None else reference.dtype,
        )

    @torch.jit.export
    def step(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one explicit recurrent inference step.

        This interface is independent of Python-side mutable state so the exact
        promoted actor can be exported and compared with native PyTorch.
        """

        if (
            observation.ndim != 2
            or observation.shape[-1] != HANDOVER_OBSERVATION_DIM
        ):
            raise ValueError(
                "successor step expects observations with shape [N, 98]"
            )
        if (
            hidden.ndim != 3
            or hidden.shape[0] != 1
            or hidden.shape[1] != observation.shape[0]
            or hidden.shape[2] != self.memory_dim
        ):
            raise ValueError(
                "successor hidden state must have shape [1, N, memory_dim]"
            )
        encoded = self.encoder(self._normalize(observation))
        latent, next_hidden = self.memory(encoded.unsqueeze(1), hidden)
        raw_action, saturation_logits, phase_index = (
            self._outputs_from_latent(observation, latent[:, 0])
        )
        action = self._hard_actions(
            raw_action,
            saturation_logits,
            phase_index,
        )
        return action, next_hidden

    def training_outputs(
        self,
        observation: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return precision actions and discrete logits for hybrid BC."""

        raw = self._policy_observation(observation)
        raw_action, saturation_logits, _ = (
            self._independent_outputs(raw)
        )
        return (
            torch.tanh(raw_action),
            raw_action[:, HANDOVER_GRIPPER_INDICES],
            saturation_logits,
        )

    def training_sequence_outputs(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run complete padded episodes from a zero recurrent state."""

        if (
            observation.ndim != 3
            or observation.shape[-1] != HANDOVER_OBSERVATION_DIM
        ):
            raise ValueError(
                "successor training sequences must have shape [B, T, 98]"
            )
        batch_size, sequence_length, _ = observation.shape
        encoded = self.encoder(
            self._normalize(observation).reshape(
                batch_size * sequence_length,
                HANDOVER_OBSERVATION_DIM,
            )
        ).reshape(batch_size, sequence_length, -1)
        latent, _ = self.memory(encoded)
        raw_action, saturation_logits, _ = self._outputs_from_latent(
            observation.reshape(
                batch_size * sequence_length,
                HANDOVER_OBSERVATION_DIM,
            ),
            latent.reshape(
                batch_size * sequence_length,
                self.memory_dim,
            ),
        )
        return (
            torch.tanh(raw_action).reshape(
                batch_size,
                sequence_length,
                HANDOVER_ACTION_DIM,
            ),
            raw_action[:, HANDOVER_GRIPPER_INDICES].reshape(
                batch_size,
                sequence_length,
                len(HANDOVER_GRIPPER_INDICES),
            ),
            saturation_logits.reshape(
                batch_size,
                sequence_length,
                len(HANDOVER_CONTINUOUS_INDICES),
                HANDOVER_SATURATION_CLASS_COUNT,
            ),
        )

    @staticmethod
    def _hard_actions(
        raw_action: torch.Tensor,
        saturation_logits: torch.Tensor,
        phase_index: torch.Tensor,
    ) -> torch.Tensor:
        action = torch.tanh(raw_action)
        continuous_action = action[:, HANDOVER_CONTINUOUS_INDICES]
        precision_logit = saturation_logits[:, :, 1]
        negative_limit = (
            saturation_logits[:, :, 0]
            >= precision_logit + HANDOVER_SATURATION_LOGIT_MARGIN
        )
        positive_limit = (
            saturation_logits[:, :, 2]
            >= precision_logit + HANDOVER_SATURATION_LOGIT_MARGIN
        )
        continuous_action = torch.where(
            negative_limit,
            -torch.ones_like(continuous_action),
            continuous_action,
        )
        continuous_action = torch.where(
            positive_limit,
            torch.ones_like(continuous_action),
            continuous_action,
        )
        action[:, HANDOVER_CONTINUOUS_INDICES] = continuous_action
        action[:, HANDOVER_GRIPPER_INDICES] = torch.where(
            raw_action[:, HANDOVER_GRIPPER_INDICES] >= 0.0,
            torch.ones_like(raw_action[:, HANDOVER_GRIPPER_INDICES]),
            -torch.ones_like(raw_action[:, HANDOVER_GRIPPER_INDICES]),
        )
        return torch.where(
            (phase_index == 4).unsqueeze(-1),
            torch.zeros_like(action),
            action,
        )

    def training_sequence_actions(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """Return hard runtime actions for complete offline episodes."""

        if (
            observation.ndim != 3
            or observation.shape[-1] != HANDOVER_OBSERVATION_DIM
        ):
            raise ValueError(
                "successor training sequences must have shape [B, T, 98]"
            )
        batch_size, sequence_length, _ = observation.shape
        encoded = self.encoder(
            self._normalize(observation).reshape(
                batch_size * sequence_length,
                HANDOVER_OBSERVATION_DIM,
            )
        ).reshape(batch_size, sequence_length, -1)
        latent, _ = self.memory(encoded)
        raw_action, saturation_logits, phase_index = (
            self._outputs_from_latent(
                observation.reshape(
                    batch_size * sequence_length,
                    HANDOVER_OBSERVATION_DIM,
                ),
                latent.reshape(
                    batch_size * sequence_length,
                    self.memory_dim,
                ),
            )
        )
        return self._hard_actions(
            raw_action,
            saturation_logits,
            phase_index,
        ).reshape(
            batch_size,
            sequence_length,
            HANDOVER_ACTION_DIM,
        )

    @torch.jit.ignore
    def forward(
        self,
        observation: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        raw = self._policy_observation(observation)
        if (
            self._runtime_hidden is None
            or self._runtime_hidden.shape[1] != raw.shape[0]
            or self._runtime_hidden.device != raw.device
            or self._runtime_hidden.dtype != raw.dtype
        ):
            self._runtime_hidden = self.initial_hidden(
                raw.shape[0],
                device=raw.device,
                dtype=raw.dtype,
            )
        action, hidden = self.step(
            raw,
            self._runtime_hidden,
        )
        self._runtime_hidden = hidden.detach()
        return action

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Clear recurrent state for completed Isaac environments."""

        if dones is None:
            self._runtime_hidden = None
            return
        if self._runtime_hidden is None:
            return
        done_mask = dones.bool().flatten()
        if done_mask.numel() != self._runtime_hidden.shape[1]:
            self._runtime_hidden = None
            return
        self._runtime_hidden[:, done_mask] = 0.0


def load_handover_successor_checkpoint(
    path: str,
    *,
    device: str | torch.device,
) -> tuple[PhaseConditionedHandoverPolicy, dict[str, Any]]:
    """Load a fail-closed successor checkpoint and its immutable metadata."""

    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("successor checkpoint must contain a mapping")
    schema_version = payload.get("schema_version")
    if schema_version not in {
        SUCCESSOR_CHECKPOINT_SCHEMA,
        *LEGACY_SUCCESSOR_CHECKPOINT_SCHEMAS,
    }:
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
    if architecture.get("binary_gripper_indices") != list(
        HANDOVER_GRIPPER_INDICES
    ):
        raise ValueError("successor checkpoint gripper contract drifted")
    if architecture.get("continuous_action_indices") != list(
        HANDOVER_CONTINUOUS_INDICES
    ):
        raise ValueError("successor checkpoint continuous-action contract drifted")
    if (
        architecture.get("saturation_classes")
        != ["negative_limit", "precision", "positive_limit"]
    ):
        raise ValueError("successor checkpoint saturation contract drifted")
    if (
        float(architecture.get("saturation_logit_margin", -1.0))
        != HANDOVER_SATURATION_LOGIT_MARGIN
    ):
        raise ValueError(
            "successor checkpoint saturation margin drifted"
        )
    if architecture.get("recurrent_state") != "gru_reset_per_episode":
        raise ValueError("successor checkpoint recurrent contract drifted")
    model = PhaseConditionedHandoverPolicy(
        payload["observation_mean"],
        payload["observation_std"],
        hidden_dims=architecture["hidden_dims"],
        memory_dim=int(architecture["memory_dim"]),
        head_dim=int(architecture["head_dim"]),
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, payload


class HandoverSuccessorStepExport(nn.Module):
    """TorchScript surface for explicit recurrent handover inference."""

    def __init__(self, policy: PhaseConditionedHandoverPolicy) -> None:
        super().__init__()
        self.encoder = policy.encoder
        self.memory = policy.memory
        self.phase_heads = policy.phase_heads
        self.register_buffer(
            "observation_mean",
            policy.observation_mean.detach().clone(),
        )
        self.register_buffer(
            "observation_std",
            policy.observation_std.detach().clone(),
        )
        self.register_buffer(
            "continuous_indices",
            torch.tensor(HANDOVER_CONTINUOUS_INDICES, dtype=torch.long),
        )
        self.register_buffer(
            "gripper_indices",
            torch.tensor(HANDOVER_GRIPPER_INDICES, dtype=torch.long),
        )

    def forward(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = (
            (observation - self.observation_mean)
            / self.observation_std
        ).clamp(-10.0, 10.0)
        encoded = self.encoder(normalized)
        latent, next_hidden = self.memory(
            encoded.unsqueeze(1),
            hidden,
        )
        all_outputs = torch.stack(
            [head(latent[:, 0]) for head in self.phase_heads],
            dim=1,
        )
        phase_index = torch.argmax(observation[:, 77:82], dim=-1)
        batch_index = torch.arange(
            observation.shape[0],
            device=observation.device,
        )
        selected = all_outputs[batch_index, phase_index]
        raw_action = selected[:, :14]
        saturation_logits = selected[:, 14:].view(
            observation.shape[0],
            12,
            3,
        )
        action = torch.tanh(raw_action)
        continuous_action = action.index_select(
            1,
            self.continuous_indices,
        )
        precision_logit = saturation_logits[:, :, 1]
        negative_limit = (
            saturation_logits[:, :, 0] >= precision_logit + 1.5
        )
        positive_limit = (
            saturation_logits[:, :, 2] >= precision_logit + 1.5
        )
        continuous_action = torch.where(
            negative_limit,
            -torch.ones_like(continuous_action),
            continuous_action,
        )
        continuous_action = torch.where(
            positive_limit,
            torch.ones_like(continuous_action),
            continuous_action,
        )
        action[:, self.continuous_indices] = continuous_action
        gripper_action = torch.where(
            raw_action.index_select(1, self.gripper_indices) >= 0.0,
            torch.ones_like(
                raw_action.index_select(1, self.gripper_indices)
            ),
            -torch.ones_like(
                raw_action.index_select(1, self.gripper_indices)
            ),
        )
        action[:, self.gripper_indices] = gripper_action
        action = torch.where(
            (phase_index == 4).unsqueeze(-1),
            torch.zeros_like(action),
            action,
        )
        return action, next_hidden


def export_handover_successor_checkpoint(
    checkpoint_path: str,
    output_path: str,
    *,
    device: str | torch.device = "cpu",
) -> None:
    """Export the exact recurrent checkpoint through its explicit step API."""

    policy, _ = load_handover_successor_checkpoint(
        checkpoint_path,
        device=device,
    )
    wrapper = HandoverSuccessorStepExport(policy).to(device=device).eval()
    example_observation = torch.zeros(
        2,
        HANDOVER_OBSERVATION_DIM,
        device=device,
    )
    example_observation[0, HANDOVER_PHASE_SLICE.start] = 1.0
    example_observation[1, HANDOVER_PHASE_SLICE.stop - 2] = 1.0
    example_hidden = policy.initial_hidden(
        2,
        device=example_observation.device,
        dtype=example_observation.dtype,
    )
    exported = torch.jit.trace(
        wrapper,
        (example_observation, example_hidden),
        strict=True,
    )
    torch.jit.save(exported, output_path)


def export_handover_successor_onnx(
    checkpoint_path: str,
    output_path: str,
    *,
    device: str | torch.device = "cpu",
) -> None:
    """Export the explicit recurrent step with dynamic environment count."""

    policy, payload = load_handover_successor_checkpoint(
        checkpoint_path,
        device=device,
    )
    wrapper = HandoverSuccessorStepExport(policy).to(device=device).eval()
    observation = torch.zeros(
        1,
        HANDOVER_OBSERVATION_DIM,
        device=device,
    )
    observation[:, HANDOVER_PHASE_SLICE.start] = 1.0
    hidden = policy.initial_hidden(
        1,
        device=observation.device,
        dtype=observation.dtype,
    )
    torch.onnx.export(
        wrapper,
        (observation, hidden),
        output_path,
        input_names=("observation", "hidden"),
        output_names=("action", "next_hidden"),
        dynamic_axes={
            "observation": {0: "num_envs"},
            "hidden": {1: "num_envs"},
            "action": {0: "num_envs"},
            "next_hidden": {1: "num_envs"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        export_params=True,
        verbose=False,
    )
    if payload.get("action_dim") != HANDOVER_ACTION_DIM:
        raise ValueError("exported successor action contract drifted")
