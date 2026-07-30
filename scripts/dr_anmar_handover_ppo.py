#!/usr/bin/env python3
"""Recurrent hybrid PPO refinement for the standalone handover successor."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as functional
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal


TASK = "DrAnmar-Handover-Needle-Dual-PSM-IK-Rel-v0"
SAFETY_TERMS = (
    "excessive_object_force",
    "needle_dropped_after_pickup",
    "object_dropping",
    "premature_giver_release",
    "protected_surface_force",
    "receiver_retention_lost",
)
HARD_SAFETY_TERMS = (
    "excessive_object_force",
    "protected_surface_force",
)
TRAINING_STATE_SCHEMA = "dranmar-handover-hybrid-ppo-state-1.0"
TRAINING_EVIDENCE_SCHEMA = "dranmar-handover-hybrid-ppo-evidence-1.0"
QUALIFICATION_SEEDS = {17, 2361, 4099}
PPO_CHUNK_ITERATIONS = 25
SOURCE_BATCH_SIZE = 4
SAFETY_CONSTRAINT_TARGET = 0.0
MAX_SAFETY_MULTIPLIER = 100.0
PRECISION_EXPLORATION_STD = 0.01
MOTION_EXPLORATION_TEMPERATURE = 0.25
GRIPPER_EXPLORATION_TEMPERATURE = 0.25
KL_BACKTRACK_FACTOR = 0.5
MAX_KL_BACKTRACKS = 12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _module_git_source(module_name: str) -> dict[str, str]:
    """Resolve the independently versioned source tree for one module."""

    module = importlib.import_module(module_name)
    module_file = Path(module.__file__).resolve()
    for candidate in (module_file.parent, *module_file.parents):
        if not (candidate / ".git").exists():
            continue
        return {
            "root": str(candidate),
            "revision": _git_output(candidate, "rev-parse", "HEAD"),
        }
    raise ValueError(f"{module_name} is not loaded from a Git worktree")


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not load Python module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class RecurrentHybridActorCritic(nn.Module):
    """BC-initialized hybrid actor with reward and safety critics."""

    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.policy = policy
        memory_dim = int(policy.memory_dim)
        self.precision_log_std = nn.Parameter(
            torch.full(
                (5, 12),
                math.log(PRECISION_EXPLORATION_STD),
            )
        )
        self.value_head = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.SiLU(),
            nn.Linear(memory_dim, 1),
        )
        self.safety_value_head = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.SiLU(),
            nn.Linear(memory_dim, 1),
        )
        for head in (self.value_head, self.safety_value_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.continuous_indices = (
            0,
            1,
            2,
            3,
            4,
            5,
            7,
            8,
            9,
            10,
            11,
            12,
        )
        self.gripper_indices = (6, 13)
        self.saturation_margin = 1.5

    def initial_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        return self.policy.initial_hidden(batch_size, device=device)

    def _step_outputs(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        encoded = self.policy.encoder(
            self.policy._normalize(observation)
        )
        latent_sequence, next_hidden = self.policy.memory(
            encoded.unsqueeze(1),
            hidden,
        )
        latent = latent_sequence[:, 0]
        raw_action, saturation_logits, phase = (
            self.policy._outputs_from_latent(observation, latent)
        )
        motion_logits = saturation_logits.clone()
        motion_logits[:, :, 0] -= self.saturation_margin
        motion_logits[:, :, 2] -= self.saturation_margin
        precision_mean = raw_action[:, self.continuous_indices]
        gripper_logits = raw_action[:, self.gripper_indices]
        value = self.value_head(latent).squeeze(-1)
        safety_value = self.safety_value_head(latent).squeeze(-1)
        return (
            motion_logits,
            precision_mean,
            gripper_logits,
            value,
            safety_value,
            next_hidden,
        )

    @staticmethod
    def _precision_log_probability(
        distribution: Normal,
        pre_tanh: torch.Tensor,
    ) -> torch.Tensor:
        correction = torch.log(
            1.0 - torch.tanh(pre_tanh).square() + 1.0e-6
        )
        return distribution.log_prob(pre_tanh) - correction

    def _distribution(
        self,
        motion_logits: torch.Tensor,
        precision_mean: torch.Tensor,
        gripper_logits: torch.Tensor,
        phase: torch.Tensor,
    ) -> tuple[Categorical, Normal, Bernoulli]:
        log_std = self.precision_log_std.clamp(-5.0, -1.5)[phase]
        return (
            Categorical(
                logits=(
                    motion_logits
                    / MOTION_EXPLORATION_TEMPERATURE
                )
            ),
            Normal(precision_mean, log_std.exp()),
            Bernoulli(
                logits=(
                    gripper_logits
                    / GRIPPER_EXPLORATION_TEMPERATURE
                )
            ),
        )

    def act(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        phase = torch.argmax(observation[:, 77:82], dim=-1)
        (
            motion_logits,
            precision_mean,
            gripper_logits,
            value,
            safety_value,
            next_hidden,
        ) = self._step_outputs(observation, hidden)
        motion_distribution, precision_distribution, gripper_distribution = (
            self._distribution(
                motion_logits,
                precision_mean,
                gripper_logits,
                phase,
            )
        )
        if deterministic:
            # Match PhaseConditionedHandoverPolicy._hard_actions exactly,
            # including its positive-limit precedence when both limit logits
            # clear the margin at the same time.  Plain argmax has different
            # tie semantics and can silently make PPO evaluation disagree
            # with the actor that is eventually exported.
            negative_limit = (
                motion_logits[:, :, 0] >= motion_logits[:, :, 1]
            )
            positive_limit = (
                motion_logits[:, :, 2] >= motion_logits[:, :, 1]
            )
            motion_mode = torch.ones_like(
                negative_limit,
                dtype=torch.long,
            )
            motion_mode = torch.where(
                negative_limit,
                torch.zeros_like(motion_mode),
                motion_mode,
            )
            motion_mode = torch.where(
                positive_limit,
                torch.full_like(motion_mode, 2),
                motion_mode,
            )
            precision_pre_tanh = precision_mean
            gripper_bits = (gripper_logits >= 0.0).float()
        else:
            motion_mode = motion_distribution.sample()
            precision_pre_tanh = precision_distribution.rsample()
            gripper_bits = gripper_distribution.sample()
        precision_action = torch.tanh(precision_pre_tanh)
        continuous_action = torch.where(
            motion_mode == 0,
            -torch.ones_like(precision_action),
            torch.where(
                motion_mode == 2,
                torch.ones_like(precision_action),
                precision_action,
            ),
        )
        action = torch.zeros(
            observation.shape[0],
            14,
            dtype=observation.dtype,
            device=observation.device,
        )
        action[:, self.continuous_indices] = continuous_action
        action[:, self.gripper_indices] = 2.0 * gripper_bits - 1.0
        action = torch.where(
            (phase == 4).unsqueeze(-1),
            torch.zeros_like(action),
            action,
        )
        precision_mask = (motion_mode == 1).float()
        log_probability = (
            motion_distribution.log_prob(motion_mode).sum(dim=-1)
            + (
                self._precision_log_probability(
                    precision_distribution,
                    precision_pre_tanh,
                )
                * precision_mask
            ).sum(dim=-1)
            + gripper_distribution.log_prob(gripper_bits).sum(dim=-1)
        )
        entropy = (
            motion_distribution.entropy().sum(dim=-1)
            + (
                precision_distribution.entropy()
                * motion_distribution.probs[:, :, 1]
            ).sum(dim=-1)
            + gripper_distribution.entropy().sum(dim=-1)
        )
        return {
            "action": action,
            "motion_mode": motion_mode,
            "precision_pre_tanh": precision_pre_tanh,
            "gripper_bits": gripper_bits,
            "log_probability": log_probability,
            "entropy": entropy,
            "value": value,
            "safety_value": safety_value,
            "hidden": next_hidden,
        }

    def evaluate_sequence(
        self,
        observations: torch.Tensor,
        initial_hidden: torch.Tensor,
        dones: torch.Tensor,
        motion_modes: torch.Tensor,
        precision_pre_tanh: torch.Tensor,
        gripper_bits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = initial_hidden
        log_probabilities = []
        entropies = []
        values = []
        safety_values = []
        for step in range(observations.shape[0]):
            if step:
                hidden = hidden * (
                    ~dones[step - 1]
                ).to(hidden.dtype).view(1, -1, 1)
            phase = torch.argmax(
                observations[step, :, 77:82],
                dim=-1,
            )
            (
                motion_logits,
                precision_mean,
                gripper_logits,
                value,
                safety_value,
                hidden,
            ) = self._step_outputs(observations[step], hidden)
            motion_distribution, precision_distribution, gripper_distribution = (
                self._distribution(
                    motion_logits,
                    precision_mean,
                    gripper_logits,
                    phase,
                )
            )
            precision_mask = (
                motion_modes[step] == 1
            ).float()
            log_probabilities.append(
                motion_distribution.log_prob(
                    motion_modes[step]
                ).sum(dim=-1)
                + (
                    self._precision_log_probability(
                        precision_distribution,
                        precision_pre_tanh[step],
                    )
                    * precision_mask
                ).sum(dim=-1)
                + gripper_distribution.log_prob(
                    gripper_bits[step]
                ).sum(dim=-1)
            )
            entropies.append(
                motion_distribution.entropy().sum(dim=-1)
                + (
                    precision_distribution.entropy()
                    * motion_distribution.probs[:, :, 1]
                ).sum(dim=-1)
                + gripper_distribution.entropy().sum(dim=-1)
            )
            values.append(value)
            safety_values.append(safety_value)
        return {
            "log_probability": torch.stack(log_probabilities),
            "entropy": torch.stack(entropies),
            "value": torch.stack(values),
            "safety_value": torch.stack(safety_values),
        }

    def imitation_outputs(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predicted, gripper_logits, saturation_logits = (
            self.policy.training_sequence_outputs(observations)
        )
        motion_logits = saturation_logits.clone()
        motion_logits[:, :, :, 0] -= self.saturation_margin
        motion_logits[:, :, :, 2] -= self.saturation_margin
        return predicted, gripper_logits, motion_logits


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one recurrent hybrid PPO refinement chunk"
    )
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--num_envs", type=int, default=1200)
    parser.add_argument("--rollout_steps", type=int, default=32)
    parser.add_argument(
        "--iterations",
        type=int,
        default=PPO_CHUNK_ITERATIONS,
    )
    parser.add_argument("--total_iterations", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=3.0e-5)
    parser.add_argument("--clip", type=float, default=0.1)
    parser.add_argument("--target_kl", type=float, default=0.008)
    parser.add_argument("--gamma", type=float, default=0.985)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--minibatches", type=int, default=8)
    parser.add_argument("--value_coefficient", type=float, default=0.5)
    parser.add_argument(
        "--safety_value_coefficient",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--entropy_coefficient",
        type=float,
        default=1.0e-4,
    )
    parser.add_argument("--imitation_start", type=float, default=0.5)
    parser.add_argument("--imitation_end", type=float, default=0.05)
    parser.add_argument(
        "--initial_safety_multiplier",
        type=float,
        default=1.0,
    )
    parser.add_argument("--dual_learning_rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=104729)
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.task != TASK:
        raise ValueError("hybrid PPO is restricted to the frozen needle task")
    if args.num_envs != 1200 or args.rollout_steps != 32:
        raise ValueError("hybrid PPO requires 1,200 environments and 32 steps")
    if (
        args.learning_rate != 3.0e-5
        or args.clip != 0.1
        or args.target_kl != 0.008
        or args.gamma != 0.985
        or args.gae_lambda != 0.95
        or args.epochs != 5
        or args.minibatches != 8
    ):
        raise ValueError("hybrid PPO completion hyperparameters are frozen")
    if (
        args.iterations != PPO_CHUNK_ITERATIONS
        or args.total_iterations <= 0
        or args.total_iterations % PPO_CHUNK_ITERATIONS
        or args.total_iterations < args.iterations
    ):
        raise ValueError(
            "PPO must run in 25-iteration evaluation chunks"
        )
    if args.num_envs % args.minibatches:
        raise ValueError("PPO minibatches must divide the environments")
    if (
        args.initial_safety_multiplier != 1.0
        or args.dual_learning_rate != 0.05
    ):
        raise ValueError("zero-cost safety dual hyperparameters are frozen")
    if (
        not 0.0 < args.imitation_end <= args.imitation_start
        or args.imitation_start != 0.5
        or args.imitation_end != 0.05
    ):
        raise ValueError("PPO imitation-anchor schedule is frozen")
    if args.entropy_coefficient != 1.0e-4:
        raise ValueError("calibrated PPO entropy coefficient is frozen")
    if args.seed in QUALIFICATION_SEEDS:
        raise ValueError("qualification seed cannot enter PPO training")


def _load_demonstrations(
    manifest_path: Path,
    training_tool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or len(datasets) < 96:
        raise ValueError("PPO imitation manifest is not the complete buffer")
    schema_version = manifest.get("schema_version")
    preservation = manifest.get("preservation", {})
    base_dataset_hashes: set[str] = set()
    if schema_version == "dranmar-handover-training-buffer-2.0":
        if (
            len(datasets) != 96
            or not preservation.get("all_accepted_data_retained")
        ):
            raise ValueError(
                "PPO imitation manifest is not the complete buffer"
            )
    elif (
        schema_version
        == "dranmar-handover-successor-dataset-manifest-2.0"
    ):
        base = manifest.get("base_manifest", {})
        base_path = Path(str(base.get("path", ""))).expanduser().resolve()
        if (
            not base_path.is_file()
            or _sha256(base_path) != base.get("sha256")
        ):
            raise ValueError("PPO base imitation manifest hash mismatch")
        base_manifest = json.loads(base_path.read_text())
        base_datasets = base_manifest.get("datasets")
        if (
            base_manifest.get("schema_version")
            != "dranmar-handover-training-buffer-2.0"
            or not isinstance(base_datasets, list)
            or len(base_datasets) != 96
            or int(base.get("dataset_count", -1))
            != len(base_datasets)
            or not base_manifest.get("preservation", {}).get(
                "all_accepted_data_retained"
            )
            or int(manifest.get("dataset_count", -1))
            != len(datasets)
            or int(manifest.get("new_dataset_count", -1))
            != len(datasets) - len(base_datasets)
            or not manifest.get(
                "qualification_seed_exclusion_verified"
            )
        ):
            raise ValueError(
                "PPO successor manifest does not preserve the base buffer"
            )
        base_dataset_hashes = {
            str(item.get("sha256", "")) for item in base_datasets
        }
        manifest_dataset_hashes = {
            str(item.get("sha256", "")) for item in datasets
        }
        if (
            "" in base_dataset_hashes
            or not base_dataset_hashes <= manifest_dataset_hashes
        ):
            raise ValueError(
                "PPO successor manifest dropped accepted base data"
            )
    else:
        raise ValueError("PPO imitation manifest is not the complete buffer")

    payloads = []
    artifact_paths = []
    observed_hashes: set[str] = set()
    for item in datasets:
        path = Path(item["path"]).expanduser().resolve()
        if (
            not path.is_file()
            or _sha256(path) != item["sha256"]
        ):
            raise ValueError("PPO imitation dataset hash mismatch")
        if item["sha256"] in observed_hashes:
            raise ValueError("PPO imitation buffer repeats an artifact")
        observed_hashes.add(item["sha256"])
        artifact_paths.append(path)
        payloads.append(training_tool._load_accepted_dataset(path))
    training_tool._training_example_ids(artifact_paths, payloads)
    if {int(payload["seed"]) for payload in payloads} & QUALIFICATION_SEEDS:
        raise ValueError("qualification seeds entered the PPO imitation buffer")
    source_counts = {
        source: sum(
            payload["label_source"] == source
            for payload in payloads
        )
        for source in (
            training_tool.TEACHER_LABEL_SOURCE,
            training_tool.BASELINE_LABEL_SOURCE,
            training_tool.DAGGER_LABEL_SOURCE,
        )
    }
    if (
        source_counts[training_tool.TEACHER_LABEL_SOURCE] < 80
        or source_counts[training_tool.BASELINE_LABEL_SOURCE] != 8
        or source_counts[training_tool.DAGGER_LABEL_SOURCE] != 8
        or sum(source_counts.values()) != len(payloads)
    ):
        raise ValueError(
            "PPO imitation buffer must retain every rescue and 16 scaffolds"
        )
    manifest["preservation"] = {
        **preservation,
        "all_accepted_data_retained": True,
        "base_dataset_count": (
            len(base_dataset_hashes)
            if base_dataset_hashes
            else len(datasets)
        ),
        "teacher_rescue_count": source_counts[
            training_tool.TEACHER_LABEL_SOURCE
        ],
        "baseline_scaffold_count": 8,
        "dagger_scaffold_count": 8,
        "total_episode_count": len(payloads),
    }
    return payloads, manifest


def _source_balanced_batch_indices(
    label_sources: list[str],
    *,
    teacher_source: str,
    baseline_source: str,
    dagger_source: str,
    generator: torch.Generator,
) -> list[int]:
    """Draw one exact 50/25/25 demonstration minibatch."""

    groups = {
        source: [
            index
            for index, observed in enumerate(label_sources)
            if observed == source
        ]
        for source in (
            teacher_source,
            baseline_source,
            dagger_source,
        )
    }
    if any(not indices for indices in groups.values()):
        raise ValueError(
            "source-balanced PPO imitation needs all three data sources"
        )
    requested = (
        (teacher_source, 2),
        (baseline_source, 1),
        (dagger_source, 1),
    )
    sampled: list[int] = []
    for source, count in requested:
        candidates = groups[source]
        choices = torch.randint(
            len(candidates),
            (count,),
            generator=generator,
        ).tolist()
        sampled.extend(candidates[index] for index in choices)
    order = torch.randperm(
        SOURCE_BATCH_SIZE,
        generator=generator,
    ).tolist()
    return [sampled[index] for index in order]


def _imitation_batch(
    actor_critic: RecurrentHybridActorCritic,
    payloads: list[dict[str, Any]],
    indices: list[int],
    training_tool,
    device: torch.device,
) -> torch.Tensor:
    observations = [
        payloads[index]["_ppo_observations"]
        for index in indices
    ]
    actions = [
        payloads[index]["_ppo_actions"]
        for index in indices
    ]
    lengths = torch.tensor(
        [value.shape[0] for value in observations],
        device=device,
    )
    padded_observations = torch.nn.utils.rnn.pad_sequence(
        observations,
        batch_first=True,
    )
    padded_actions = torch.nn.utils.rnn.pad_sequence(
        actions,
        batch_first=True,
    )
    valid = (
        torch.arange(
            padded_observations.shape[1],
            device=device,
        ).unsqueeze(0)
        < lengths.unsqueeze(1)
    )
    frame_weights = torch.ones_like(valid, dtype=torch.float32)
    for row, index in enumerate(indices):
        payload = payloads[index]
        frame_count = int(lengths[row].item())
        start = training_tool._teacher_training_start_frame(
            payload,
            frame_count,
        )
        if start is not None:
            frame_weights[row, start:frame_count] = 2.0

    predicted, gripper_logits, motion_logits = (
        actor_critic.imitation_outputs(padded_observations)
    )
    target_continuous = padded_actions[
        :, :, actor_critic.continuous_indices
    ]
    target_motion = torch.ones_like(
        target_continuous,
        dtype=torch.long,
    )
    target_motion[target_continuous <= -0.999] = 0
    target_motion[target_continuous >= 0.999] = 2
    precision_mask = (target_motion == 1).float()
    precision_error = functional.smooth_l1_loss(
        predicted[:, :, actor_critic.continuous_indices],
        target_continuous,
        reduction="none",
    )
    precision_loss = (
        (precision_error * precision_mask).sum(dim=-1)
        / precision_mask.sum(dim=-1).clamp_min(1.0)
    )
    motion_loss = functional.cross_entropy(
        motion_logits.reshape(-1, 3),
        target_motion.reshape(-1),
        reduction="none",
    ).reshape(target_motion.shape).mean(dim=-1)
    target_grippers = (
        padded_actions[:, :, actor_critic.gripper_indices] > 0.0
    ).float()
    gripper_loss = functional.binary_cross_entropy_with_logits(
        gripper_logits,
        target_grippers,
        reduction="none",
    ).mean(dim=-1)
    per_frame = precision_loss + motion_loss + 4.0 * gripper_loss
    valid_weights = frame_weights * valid.float()
    per_episode = (
        (per_frame * valid_weights).sum(dim=1)
        / valid_weights.sum(dim=1).clamp_min(1.0)
    )
    return per_episode.mean()


def _gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    final_value: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    accumulator = torch.zeros_like(final_value)
    next_value = final_value
    for step in reversed(range(rewards.shape[0])):
        continuing = (~dones[step]).float()
        delta = (
            rewards[step]
            + gamma * next_value * continuing
            - values[step]
        )
        accumulator = (
            delta
            + gamma * gae_lambda * continuing * accumulator
        )
        advantages[step] = accumulator
        next_value = values[step]
    return advantages, advantages + values


def _updated_safety_multiplier(
    current: float,
    costs: torch.Tensor,
    dones: torch.Tensor,
    *,
    learning_rate: float,
) -> tuple[float, float]:
    """Apply a zero-target dual update using completed-episode cost rate."""

    completed = int(dones.sum().item())
    observed_rate = (
        float(costs.sum().item()) / completed
        if completed
        else 0.0
    )
    updated = min(
        MAX_SAFETY_MULTIPLIER,
        max(
            0.0,
            current
            + learning_rate
            * (observed_rate - SAFETY_CONSTRAINT_TARGET),
        ),
    )
    return updated, observed_rate


def _backtracked_optimizer_step(
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    nominal_learning_rate: float,
    target_kl: float,
    measure_kl: Callable[[], float],
) -> tuple[bool, float, float, int]:
    """Apply one Adam step only inside the empirical KL trust region."""

    module_state = {
        key: value.detach().clone()
        for key, value in module.state_dict().items()
    }
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    maximum_kl = target_kl * 1.5
    last_kl = math.inf
    for attempt in range(MAX_KL_BACKTRACKS):
        module.load_state_dict(module_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
        scale = KL_BACKTRACK_FACTOR**attempt
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = nominal_learning_rate * scale
        optimizer.step()
        last_kl = measure_kl()
        if math.isfinite(last_kl) and last_kl <= maximum_kl:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = nominal_learning_rate
            return True, scale, last_kl, attempt

    module.load_state_dict(module_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = nominal_learning_rate
    return False, 0.0, last_kl, MAX_KL_BACKTRACKS


def _verify_jit_parity(
    policy_module,
    checkpoint_path: Path,
    jit_path: Path,
) -> dict[str, Any]:
    """Export the candidate and replay recurrent eager/JIT steps on CPU."""

    if jit_path.exists():
        raise ValueError(f"refusing to overwrite JIT artifact: {jit_path}")
    policy_module.export_handover_successor_checkpoint(
        str(checkpoint_path),
        str(jit_path),
        device="cpu",
    )
    eager, _ = policy_module.load_handover_successor_checkpoint(
        str(checkpoint_path),
        device="cpu",
    )
    eager.eval()
    exported = torch.jit.load(str(jit_path), map_location="cpu").eval()
    generator = torch.Generator().manual_seed(982_451_653)
    environments = 5
    steps = 12
    hidden_eager = eager.initial_hidden(
        environments,
        device=torch.device("cpu"),
    )
    hidden_jit = hidden_eager.clone()
    maximum_action_error = 0.0
    maximum_hidden_error = 0.0
    with torch.no_grad():
        for step in range(steps):
            observation = (
                eager.observation_mean.unsqueeze(0)
                + 0.1
                * eager.observation_std.unsqueeze(0)
                * torch.randn(
                    environments,
                    98,
                    generator=generator,
                )
            )
            observation[:, 77:82] = 0.0
            for environment in range(environments):
                observation[
                    environment,
                    77 + (step + environment) % 5,
                ] = 1.0
            eager_action, next_eager = eager.step(
                observation,
                hidden_eager,
            )
            jit_action, next_jit = exported(
                observation,
                hidden_jit,
            )
            action_error = float(
                (eager_action - jit_action).abs().max().item()
            )
            hidden_error = float(
                (next_eager - next_jit).abs().max().item()
            )
            maximum_action_error = max(
                maximum_action_error,
                action_error,
            )
            maximum_hidden_error = max(
                maximum_hidden_error,
                hidden_error,
            )
            torch.testing.assert_close(
                eager_action,
                jit_action,
                atol=1.0e-6,
                rtol=1.0e-6,
            )
            torch.testing.assert_close(
                next_eager,
                next_jit,
                atol=1.0e-6,
                rtol=1.0e-6,
            )
            reset = torch.tensor(
                [
                    (step + environment) % 7 == 0
                    for environment in range(environments)
                ],
                dtype=torch.bool,
            ).view(1, environments, 1)
            hidden_eager = torch.where(
                reset,
                torch.zeros_like(next_eager),
                next_eager,
            )
            hidden_jit = torch.where(
                reset,
                torch.zeros_like(next_jit),
                next_jit,
            )
    return {
        "steps": steps,
        "environments": environments,
        "atol": 1.0e-6,
        "rtol": 1.0e-6,
        "maximum_action_error": maximum_action_error,
        "maximum_hidden_error": maximum_hidden_error,
    }


def _candidate_checkpoint(
    initial_payload: dict[str, Any],
    actor_critic: RecurrentHybridActorCritic,
    *,
    runtime_revision: str,
    iteration: int,
    parent_checkpoint: Path,
    dataset_manifest: Path,
    training_state_sha256: str,
    metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    checkpoint = copy.deepcopy(initial_payload)
    checkpoint["deployment_status"] = "candidate_only"
    checkpoint["promotion_eligible"] = False
    checkpoint["model"] = {
        key: value.detach().cpu().clone()
        for key, value in actor_critic.policy.state_dict().items()
    }
    checkpoint["observation_mean"] = (
        actor_critic.policy.observation_mean.detach().cpu().clone()
    )
    checkpoint["observation_std"] = (
        actor_critic.policy.observation_std.detach().cpu().clone()
    )
    checkpoint["source"] = {
        **checkpoint["source"],
        "dranmar_revision": runtime_revision,
    }
    checkpoint["training"] = {
        **checkpoint.get("training", {}),
        "ppo": {
            "algorithm": "recurrent_hybrid_ppo_lagrangian",
            "iteration": iteration,
            "parent_checkpoint": {
                "path": str(parent_checkpoint),
                "sha256": _sha256(parent_checkpoint),
            },
            "dataset_manifest": {
                "path": str(dataset_manifest),
                "sha256": _sha256(dataset_manifest),
            },
            "training_state_sha256": training_state_sha256,
            "checkpoint_selection": (
                "frozen_development_outcomes_only"
            ),
            "motion_distribution": (
                "factorized_categorical_negative_precision_positive"
            ),
            "precision_distribution": "tanh_normal",
            "gripper_distribution": "factorized_bernoulli",
            "exploration_calibration": {
                "precision_std": PRECISION_EXPLORATION_STD,
                "motion_temperature": (
                    MOTION_EXPLORATION_TEMPERATURE
                ),
                "gripper_temperature": (
                    GRIPPER_EXPLORATION_TEMPERATURE
                ),
                "basis": (
                    "all_96_accepted_episodes_and_late_phase_"
                    "ultra_small_residual_rescues"
                ),
            },
            "safety_constraint_target": SAFETY_CONSTRAINT_TARGET,
            "metrics_tail": metrics[-10:],
        },
    }
    return checkpoint


def _train(args: argparse.Namespace, repo_root: Path) -> int:
    import gymnasium as gym
    from isaaclab_tasks.utils.parse_cfg import (
        load_cfg_from_registry,
        parse_env_cfg,
    )
    from isaaclab_rl.rsl_rl import (
        RslRlVecEnvWrapper,
        handle_deprecated_rsl_rl_cfg,
    )

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    manifest_path = Path(args.dataset_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not checkpoint_path.is_file() or not manifest_path.is_file():
        raise ValueError("PPO checkpoint and dataset manifest must exist")
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_revision = _git_output(repo_root, "rev-parse", "HEAD")
    if _git_output(repo_root, "status", "--porcelain"):
        raise ValueError("PPO training requires a clean source worktree")

    policy_module = _load_module(
        "dranmar_handover_ppo_policy",
        repo_root
        / "source/extensions/orbit.surgical.tasks/orbit/surgical/tasks/"
        "surgical/handover/successor_policy.py",
    )
    training_tool = _load_module(
        "dranmar_handover_ppo_dataset",
        repo_root / "scripts/dr_anmar_handover_successor.py",
    )
    policy, initial_payload = (
        policy_module.load_handover_successor_checkpoint(
            str(checkpoint_path),
            device="cuda:0",
        )
    )
    if initial_payload.get("source", {}).get(
        "dranmar_revision"
    ) != runtime_revision:
        raise ValueError("PPO parent checkpoint source does not match checkout")
    asset_source = _module_git_source("orbit.surgical.assets")
    if (
        initial_payload.get("source", {}).get("asset_revision")
        != asset_source["revision"]
    ):
        raise ValueError("PPO parent checkpoint asset revision drifted")
    payloads, dataset_manifest = _load_demonstrations(
        manifest_path,
        training_tool,
    )
    actor_critic = RecurrentHybridActorCritic(policy).to("cuda:0")
    optimizer = torch.optim.Adam(
        actor_critic.parameters(),
        lr=args.learning_rate,
    )
    start_iteration = 0
    lagrange_multiplier = args.initial_safety_multiplier
    metrics: list[dict[str, Any]] = []
    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        resume_payload = torch.load(
            resume_path,
            map_location="cuda:0",
            weights_only=False,
        )
        if (
            resume_payload.get("schema_version")
            != TRAINING_STATE_SCHEMA
            or resume_payload.get("source_revision")
            != runtime_revision
            or resume_payload.get("parent_checkpoint_sha256")
            != _sha256(checkpoint_path)
            or resume_payload.get("dataset_manifest_sha256")
            != _sha256(manifest_path)
        ):
            raise ValueError("PPO resume state provenance mismatch")
        actor_critic.load_state_dict(
            resume_payload["actor_critic"],
            strict=True,
        )
        optimizer.load_state_dict(resume_payload["optimizer"])
        start_iteration = int(resume_payload["iteration"])
        lagrange_multiplier = float(
            resume_payload["lagrange_multiplier"]
        )
        metrics = list(resume_payload.get("metrics", []))
    if (
        start_iteration % PPO_CHUNK_ITERATIONS
        or start_iteration + args.iterations > args.total_iterations
    ):
        raise ValueError(
            "PPO resume must align with the frozen evaluation schedule"
        )

    environment_seed = (
        args.seed + start_iteration * 1_000_003
    )
    if environment_seed in QUALIFICATION_SEEDS:
        raise ValueError("qualification seed entered PPO environment training")

    env_cfg = parse_env_cfg(
        args.task,
        device="cuda:0",
        num_envs=args.num_envs,
        use_fabric=True,
    )
    agent_cfg = load_cfg_from_registry(
        args.task,
        "rsl_rl_cfg_entry_point",
    )
    agent_cfg.seed = environment_seed
    env_cfg.seed = environment_seed
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg,
        metadata.version("rsl-rl-lib"),
    )
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(
        env,
        clip_actions=agent_cfg.clip_actions,
    )
    termination_manager = env.unwrapped.termination_manager
    termination_names = set(termination_manager.active_terms)
    missing_terms = set(SAFETY_TERMS) - termination_names
    if missing_terms:
        env.close()
        raise ValueError(
            "PPO safety termination contract is incomplete: "
            + ", ".join(sorted(missing_terms))
        )

    device = torch.device("cuda:0")
    for payload in payloads:
        payload["_ppo_observations"] = payload["episode"][
            "observations"
        ].float().to(device)
        payload["_ppo_actions"] = payload["episode"][
            "actions"
        ].float().to(device)
    random.seed(environment_seed)
    torch.manual_seed(environment_seed)
    torch.cuda.manual_seed_all(environment_seed)
    source_labels = [
        str(payload["label_source"]) for payload in payloads
    ]
    imitation_generator = torch.Generator().manual_seed(
        environment_seed
    )
    observation = env.get_observations()["policy"].to(device)
    hidden = actor_critic.initial_hidden(
        args.num_envs,
        device=device,
    )
    actor_critic.train()
    started = time.perf_counter()
    try:
        for iteration_offset in range(args.iterations):
            iteration = start_iteration + iteration_offset + 1
            rollout: dict[str, list[torch.Tensor]] = {
                name: []
                for name in (
                    "observation",
                    "motion_mode",
                    "precision_pre_tanh",
                    "gripper_bits",
                    "log_probability",
                    "value",
                    "safety_value",
                    "environment_reward",
                    "reward",
                    "environment_cost",
                    "cost",
                    "done",
                    "timeout",
                )
            }
            rollout_initial_hidden = hidden.detach().clone()
            termination_totals = {
                name: 0 for name in termination_manager.active_terms
            }
            with torch.no_grad():
                for _ in range(args.rollout_steps):
                    sampled = actor_critic.act(observation, hidden)
                    next_observation, reward, dones, extras = env.step(
                        sampled["action"]
                    )
                    timeout = extras.get("time_outs")
                    if timeout is None:
                        timeout = torch.zeros_like(
                            dones,
                            dtype=torch.bool,
                        )
                    else:
                        timeout = timeout.bool()
                    term_values = {
                        name: termination_manager.get_term(name).bool()
                        for name in termination_manager.active_terms
                    }
                    cost = torch.stack(
                        [term_values[name] for name in SAFETY_TERMS],
                        dim=0,
                    ).any(dim=0).float()
                    rollout["observation"].append(observation)
                    for name in (
                        "motion_mode",
                        "precision_pre_tanh",
                        "gripper_bits",
                        "log_probability",
                        "value",
                        "safety_value",
                    ):
                        rollout[name].append(sampled[name])
                    rollout["environment_reward"].append(reward)
                    rollout["reward"].append(
                        reward
                        + args.gamma
                        * sampled["value"]
                        * timeout.float()
                    )
                    rollout["environment_cost"].append(cost)
                    rollout["cost"].append(
                        cost
                        + args.gamma
                        * sampled["safety_value"]
                        * timeout.float()
                    )
                    rollout["done"].append(dones.bool())
                    rollout["timeout"].append(timeout)
                    for name, value in term_values.items():
                        termination_totals[name] += int(
                            value.sum().item()
                        )
                    hidden = sampled["hidden"].detach()
                    hidden = hidden * (
                        ~dones.bool()
                    ).to(hidden.dtype).view(1, -1, 1)
                    observation = next_observation["policy"].to(device)
                final = actor_critic.act(
                    observation,
                    hidden,
                    deterministic=True,
                )
            tensors = {
                name: torch.stack(values)
                for name, values in rollout.items()
            }
            for name, value in tensors.items():
                if value.is_floating_point() and not torch.isfinite(
                    value
                ).all():
                    raise RuntimeError(
                        f"non-finite PPO rollout tensor: {name}"
                    )
            reward_advantage, reward_return = _gae(
                tensors["reward"],
                tensors["value"],
                tensors["done"],
                final["value"],
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )
            cost_advantage, cost_return = _gae(
                tensors["cost"],
                tensors["safety_value"],
                tensors["done"],
                final["safety_value"],
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )
            reward_advantage = (
                reward_advantage - reward_advantage.mean()
            ) / reward_advantage.std(unbiased=False).clamp_min(1.0e-6)
            lagrange_multiplier, safety_event_rate = (
                _updated_safety_multiplier(
                    lagrange_multiplier,
                    tensors["environment_cost"],
                    tensors["done"],
                    learning_rate=args.dual_learning_rate,
                )
            )
            progress = min(
                1.0,
                iteration / max(args.total_iterations, 1),
            )
            imitation_coefficient = (
                args.imitation_start
                + progress
                * (args.imitation_end - args.imitation_start)
            )
            approximate_kl = 0.0
            update_count = 0
            policy_loss_total = 0.0
            value_loss_total = 0.0
            safety_value_loss_total = 0.0
            imitation_loss_total = 0.0
            backtrack_scale_total = 0.0
            backtrack_attempt_total = 0
            minimum_backtrack_scale = 1.0
            early_stop = False
            for _ in range(args.epochs):
                environment_order = torch.randperm(
                    args.num_envs,
                    device=device,
                )
                for minibatch_indices in environment_order.chunk(
                    args.minibatches
                ):
                    evaluated = actor_critic.evaluate_sequence(
                        tensors["observation"][:, minibatch_indices],
                        rollout_initial_hidden[:, minibatch_indices],
                        tensors["done"][:, minibatch_indices],
                        tensors["motion_mode"][:, minibatch_indices],
                        tensors["precision_pre_tanh"][
                            :, minibatch_indices
                        ],
                        tensors["gripper_bits"][:, minibatch_indices],
                    )
                    old_log_probability = tensors[
                        "log_probability"
                    ][:, minibatch_indices]
                    log_ratio = (
                        evaluated["log_probability"]
                        - old_log_probability
                    )
                    ratio = log_ratio.exp()
                    with torch.no_grad():
                        pre_update_kl = float(
                            (
                                (ratio - 1.0) - log_ratio
                            ).mean().item()
                        )
                    if (
                        not math.isfinite(pre_update_kl)
                        or pre_update_kl > args.target_kl * 1.5
                    ):
                        early_stop = True
                        break
                    reward_advantage_batch = reward_advantage[
                        :, minibatch_indices
                    ]
                    reward_surrogate = torch.minimum(
                        ratio * reward_advantage_batch,
                        ratio.clamp(
                            1.0 - args.clip,
                            1.0 + args.clip,
                        )
                        * reward_advantage_batch,
                    )
                    policy_loss = -reward_surrogate.mean()
                    cost_advantage_batch = cost_advantage[
                        :, minibatch_indices
                    ]
                    cost_surrogate = torch.maximum(
                        ratio * cost_advantage_batch,
                        ratio.clamp(
                            1.0 - args.clip,
                            1.0 + args.clip,
                        )
                        * cost_advantage_batch,
                    ).mean()
                    value_loss = functional.mse_loss(
                        evaluated["value"],
                        reward_return[:, minibatch_indices],
                    )
                    safety_value_loss = functional.mse_loss(
                        evaluated["safety_value"],
                        cost_return[:, minibatch_indices],
                    )
                    demonstration_indices = (
                        _source_balanced_batch_indices(
                            source_labels,
                            teacher_source=(
                                training_tool.TEACHER_LABEL_SOURCE
                            ),
                            baseline_source=(
                                training_tool.BASELINE_LABEL_SOURCE
                            ),
                            dagger_source=(
                                training_tool.DAGGER_LABEL_SOURCE
                            ),
                            generator=imitation_generator,
                        )
                    )
                    imitation_loss = _imitation_batch(
                        actor_critic,
                        payloads,
                        demonstration_indices,
                        training_tool,
                        device,
                    )
                    entropy = evaluated["entropy"].mean()
                    loss = (
                        policy_loss
                        + lagrange_multiplier * cost_surrogate
                        + args.value_coefficient * value_loss
                        + args.safety_value_coefficient
                        * safety_value_loss
                        + imitation_coefficient * imitation_loss
                        - args.entropy_coefficient * entropy
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        actor_critic.parameters(),
                        1.0,
                    )
                    def measure_post_step_kl() -> float:
                        with torch.no_grad():
                            post_step = actor_critic.evaluate_sequence(
                                tensors["observation"][
                                    :, minibatch_indices
                                ],
                                rollout_initial_hidden[
                                    :, minibatch_indices
                                ],
                                tensors["done"][:, minibatch_indices],
                                tensors["motion_mode"][
                                    :, minibatch_indices
                                ],
                                tensors["precision_pre_tanh"][
                                    :, minibatch_indices
                                ],
                                tensors["gripper_bits"][
                                    :, minibatch_indices
                                ],
                            )
                            post_log_ratio = (
                                post_step["log_probability"]
                                - old_log_probability
                            )
                            post_ratio = post_log_ratio.exp()
                            return float(
                                (
                                    (post_ratio - 1.0)
                                    - post_log_ratio
                                ).mean().item()
                            )

                    (
                        accepted,
                        backtrack_scale,
                        batch_kl,
                        backtrack_attempts,
                    ) = _backtracked_optimizer_step(
                        actor_critic,
                        optimizer,
                        nominal_learning_rate=args.learning_rate,
                        target_kl=args.target_kl,
                        measure_kl=measure_post_step_kl,
                    )
                    if not accepted:
                        early_stop = True
                        break
                    approximate_kl += batch_kl
                    update_count += 1
                    backtrack_scale_total += backtrack_scale
                    backtrack_attempt_total += backtrack_attempts
                    minimum_backtrack_scale = min(
                        minimum_backtrack_scale,
                        backtrack_scale,
                    )
                    policy_loss_total += float(policy_loss.item())
                    value_loss_total += float(value_loss.item())
                    safety_value_loss_total += float(
                        safety_value_loss.item()
                    )
                    imitation_loss_total += float(
                        imitation_loss.item()
                    )
                if early_stop:
                    break
            metric = {
                "iteration": iteration,
                "mean_reward": float(
                    tensors["environment_reward"].mean().item()
                ),
                "mean_safety_cost": float(
                    tensors["environment_cost"].mean().item()
                ),
                "safety_event_rate_per_completed_episode": (
                    safety_event_rate
                ),
                "completed_episodes": int(
                    tensors["done"].sum().item()
                ),
                "successful_episodes": termination_totals["success"],
                "hard_safety_events": sum(
                    termination_totals[name]
                    for name in HARD_SAFETY_TERMS
                ),
                "lagrange_multiplier": lagrange_multiplier,
                "imitation_coefficient": imitation_coefficient,
                "approximate_kl": (
                    approximate_kl / max(update_count, 1)
                ),
                "policy_loss": (
                    policy_loss_total / max(update_count, 1)
                ),
                "value_loss": (
                    value_loss_total / max(update_count, 1)
                ),
                "safety_value_loss": (
                    safety_value_loss_total / max(update_count, 1)
                ),
                "imitation_loss": (
                    imitation_loss_total / max(update_count, 1)
                ),
                "updates": update_count,
                "mean_accepted_step_scale": (
                    backtrack_scale_total / max(update_count, 1)
                ),
                "minimum_accepted_step_scale": (
                    minimum_backtrack_scale
                    if update_count
                    else 0.0
                ),
                "kl_backtrack_attempts": backtrack_attempt_total,
                "target_kl_early_stop": early_stop,
                "termination_counts": termination_totals,
            }
            metrics.append(metric)
            print(json.dumps(metric, sort_keys=True), flush=True)
    finally:
        env.close()

    final_iteration = start_iteration + args.iterations
    state_path = (
        output_dir
        / f"trainer-iteration-{final_iteration:06d}.pt"
    )
    state = {
        "schema_version": TRAINING_STATE_SCHEMA,
        "source_revision": runtime_revision,
        "parent_checkpoint_sha256": _sha256(checkpoint_path),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "iteration": final_iteration,
        "environment_seed": environment_seed,
        "lagrange_multiplier": lagrange_multiplier,
        "actor_critic": {
            key: value.detach().cpu().clone()
            for key, value in actor_critic.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
        "hyperparameters": {
            key: getattr(args, key)
            for key in (
                "num_envs",
                "rollout_steps",
                "learning_rate",
                "clip",
                "target_kl",
                "gamma",
                "gae_lambda",
                "epochs",
                "minibatches",
                "value_coefficient",
                "safety_value_coefficient",
                "entropy_coefficient",
                "imitation_start",
                "imitation_end",
                "initial_safety_multiplier",
                "dual_learning_rate",
                "seed",
            )
        },
        "runtime": {
            "dranmar_revision": runtime_revision,
            "asset": asset_source,
        },
        "exploration_calibration": {
            "precision_std": PRECISION_EXPLORATION_STD,
            "motion_temperature": MOTION_EXPLORATION_TEMPERATURE,
            "gripper_temperature": GRIPPER_EXPLORATION_TEMPERATURE,
        },
    }
    _atomic_torch_save(state, state_path)
    candidate_path = (
        output_dir
        / f"successor-iteration-{final_iteration:06d}.pt"
    )
    candidate = _candidate_checkpoint(
        initial_payload,
        actor_critic,
        runtime_revision=runtime_revision,
        iteration=final_iteration,
        parent_checkpoint=checkpoint_path,
        dataset_manifest=manifest_path,
        training_state_sha256=_sha256(state_path),
        metrics=metrics,
    )
    _atomic_torch_save(candidate, candidate_path)
    jit_path = (
        output_dir
        / f"successor-iteration-{final_iteration:06d}.jit.pt"
    )
    jit_parity = _verify_jit_parity(
        policy_module,
        candidate_path,
        jit_path,
    )
    evidence = {
        "schema_version": TRAINING_EVIDENCE_SCHEMA,
        "source_revision": runtime_revision,
        "asset_source": asset_source,
        "iteration": final_iteration,
        "environment_seed": environment_seed,
        "transitions_in_chunk": (
            args.num_envs * args.rollout_steps * args.iterations
        ),
        "transitions_through_iteration": (
            args.num_envs * args.rollout_steps * final_iteration
        ),
        "duration_s": time.perf_counter() - started,
        "parent_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
        },
        "dataset_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
            "episode_count": len(payloads),
            "preservation": dataset_manifest["preservation"],
        },
        "exploration_calibration": {
            "precision_std": PRECISION_EXPLORATION_STD,
            "motion_temperature": MOTION_EXPLORATION_TEMPERATURE,
            "gripper_temperature": GRIPPER_EXPLORATION_TEMPERATURE,
            "dataset_episode_count": len(payloads),
        },
        "training_state": {
            "path": str(state_path),
            "sha256": _sha256(state_path),
        },
        "candidate": {
            "path": str(candidate_path),
            "sha256": _sha256(candidate_path),
        },
        "jit": {
            "path": str(jit_path),
            "sha256": _sha256(jit_path),
            "parity": jit_parity,
        },
        "metrics": metrics,
    }
    evidence_path = (
        output_dir
        / f"evidence-iteration-{final_iteration:06d}.json"
    )
    _atomic_json_save(evidence, evidence_path)
    print(
        json.dumps(
            {
                "candidate": evidence["candidate"],
                "jit": evidence["jit"],
                "training_state": evidence["training_state"],
                "evidence": {
                    "path": str(evidence_path),
                    "sha256": _sha256(evidence_path),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    _validate_arguments(args)
    repo_root = Path(__file__).resolve().parents[1]
    isaaclab_value = os.environ.get("DR_ANMAR_ISAACLAB_ROOT")
    if not isaaclab_value:
        raise ValueError(
            "DR_ANMAR_ISAACLAB_ROOT must point to Isaac Lab"
        )
    isaaclab_root = Path(isaaclab_value).expanduser().resolve()
    for path in reversed(
        (
            repo_root / "source/extensions/orbit.surgical.tasks",
            repo_root / "source/extensions/orbit.surgical.assets",
            isaaclab_root,
        )
    ):
        sys.path.insert(0, str(path))
    torch.cuda.set_device(0)
    cuda_context_guard = torch.zeros(1, device="cuda:0")
    from isaaclab.app import AppLauncher

    app = AppLauncher(
        headless=True,
        enable_cameras=False,
        multi_gpu=False,
        anti_aliasing=0,
        denoiser=False,
        kit_args="--/persistent/physics/useActiveCudaContext=true",
    ).app
    try:
        if not app.is_running():
            raise RuntimeError("Isaac Sim did not remain running")
        import carb

        carb.settings.get_settings().set_bool(
            "/persistent/physics/useActiveCudaContext",
            False,
        )
        import orbit.surgical.tasks  # noqa: F401

        return _train(args, repo_root)
    finally:
        del cuda_context_guard
        app.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
