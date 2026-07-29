#!/usr/bin/env python3
"""Bootstrap and update the one-decision receiver attempt PPO head."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional


DEVELOPMENT_SEEDS = {104729, 130363, 196613}
SCHEMA_VERSION = "dranmar-receiver-attempt-ppo-1.0"
ROLLOUT_SCHEMA_VERSION = "dranmar-receiver-attempt-ppo-rollout-1.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_models() -> tuple[type[torch.nn.Module], type[torch.nn.Module]]:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(
        0,
        str(repo_root / "source/extensions/orbit.surgical.tasks"),
    )
    sys.path.insert(
        0,
        str(repo_root / "source/extensions/orbit.surgical.assets"),
    )
    from orbit.surgical.tasks.surgical.handover.recovery_policy import (
        ReceiverAttemptActorCritic,
        ReceiverCandidateValue,
    )

    return ReceiverAttemptActorCritic, ReceiverCandidateValue


def _project_vector(value: torch.Tensor, cap: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    return value * torch.clamp(cap / norm.clamp_min(1.0e-8), max=1.0)


def _candidate_local_offsets(
    *,
    seed: int,
    position_radius_m: float,
    orientation_radius_rad: float,
) -> torch.Tensor:
    sobol = torch.quasirandom.SobolEngine(
        dimension=6,
        scramble=True,
        seed=seed,
    )
    normalized = torch.cat(
        (
            torch.zeros(1, 6),
            2.0 * sobol.draw(31) - 1.0,
        ),
        dim=0,
    )
    return torch.cat(
        (
            normalized[:, :3] * position_radius_m,
            normalized[:, 3:] * orientation_radius_rad,
        ),
        dim=-1,
    )


def _promoted_candidate_selection(
    context: torch.Tensor,
    payload: dict[str, object],
    *,
    position_cap_m: float,
    orientation_cap_rad: float,
    local_sobol_seed: int,
    local_position_radius_m: float,
    local_orientation_radius_rad: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, candidate_model_type = _repo_models()
    model = candidate_model_type()
    model.load_state_dict(payload["receiver_candidate_value"], strict=True)
    model.eval()
    feature_mean = payload["feature_mean"].float()
    feature_std = payload["feature_std"].float()
    candidates = payload["candidate_corrections"].float()
    normalized_candidates = torch.cat(
        (
            candidates[:, :3] / position_cap_m,
            candidates[:, 3:] / orientation_cap_rad,
        ),
        dim=-1,
    )
    features = torch.cat(
        (
            context.unsqueeze(1).expand(-1, candidates.shape[0], -1),
            normalized_candidates.unsqueeze(0).expand(
                context.shape[0],
                -1,
                -1,
            ),
        ),
        dim=-1,
    )
    with torch.no_grad():
        scores = model(
            ((features - feature_mean) / feature_std).reshape(-1, 35)
        ).reshape(context.shape[0], candidates.shape[0])
    best_index = scores.argmax(dim=-1)
    zero_index = candidates.square().sum(dim=-1).argmin()
    zero_score = scores[:, zero_index]
    selected = candidates[best_index]
    local_offsets = _candidate_local_offsets(
        seed=local_sobol_seed,
        position_radius_m=local_position_radius_m,
        orientation_radius_rad=local_orientation_radius_rad,
    )
    local = selected.unsqueeze(1) + local_offsets.unsqueeze(0)
    local = torch.cat(
        (
            _project_vector(
                local[:, :, :3].reshape(-1, 3),
                position_cap_m,
            ).reshape(-1, 32, 3),
            _project_vector(
                local[:, :, 3:].reshape(-1, 3),
                orientation_cap_rad,
            ).reshape(-1, 32, 3),
        ),
        dim=-1,
    )
    normalized_local = torch.cat(
        (
            local[:, :, :3] / position_cap_m,
            local[:, :, 3:] / orientation_cap_rad,
        ),
        dim=-1,
    )
    local_features = torch.cat(
        (
            context.unsqueeze(1).expand(-1, 32, -1),
            normalized_local,
        ),
        dim=-1,
    )
    with torch.no_grad():
        local_scores = model(
            (
                (local_features - feature_mean) / feature_std
            ).reshape(-1, 35)
        ).reshape(context.shape[0], 32)
    local_score, local_index = local_scores.max(dim=-1)
    selected = local[
        torch.arange(context.shape[0]),
        local_index,
    ]
    advantage = local_score - zero_score
    applied = advantage >= 0.0
    correction = torch.where(
        applied.unsqueeze(-1),
        selected,
        candidates[zero_index].unsqueeze(0),
    )
    return correction, advantage


def _write_checkpoint(
    payload: dict[str, object],
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    report = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "receiver_attempt_actor_critic",
            "feature_mean",
            "feature_std",
        }
    }
    report["checkpoint"] = {
        "path": str(output),
        "sha256": _sha256(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _bootstrap(args: argparse.Namespace) -> int:
    actor_critic_type, _ = _repo_models()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    base_checkpoint = Path(args.base_checkpoint).expanduser().resolve()
    candidate_checkpoint = (
        Path(args.candidate_checkpoint).expanduser().resolve()
    )
    if not base_checkpoint.is_file() or not candidate_checkpoint.is_file():
        raise FileNotFoundError("base and candidate checkpoints must exist")
    candidate_payload = torch.load(
        candidate_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if (
        candidate_payload.get("schema_version")
        != "dranmar-receiver-candidate-value-1.0"
        or candidate_payload.get("base_checkpoint_sha256")
        != _sha256(base_checkpoint)
    ):
        raise ValueError(
            "candidate checkpoint does not match the frozen base policy"
        )
    receiver_orientation_cap_rad = math.radians(
        args.receiver_orientation_cap_deg
    )
    residual_orientation_cap_rad = math.radians(
        args.residual_orientation_cap_deg
    )
    if not math.isclose(
        float(candidate_payload["position_cap_m"]),
        args.receiver_position_cap_m,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ) or not math.isclose(
        float(candidate_payload["orientation_cap_rad"]),
        receiver_orientation_cap_rad,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("candidate correction caps drifted")

    dataset_paths = [
        Path(value).expanduser().resolve() for value in args.dataset
    ]
    contexts = []
    outcomes = []
    dataset_reports = []
    for path in dataset_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        attempts = payload.get("attempts") or payload
        if (
            payload.get("schema_version")
            != "dranmar-receiver-recovery-dataset-1.2"
            or int(payload["seed"]) not in DEVELOPMENT_SEEDS
            or payload["base_checkpoint_sha256"] != _sha256(base_checkpoint)
            or attempts["context"].shape[-1] != 29
        ):
            raise ValueError(f"incompatible bootstrap dataset: {path}")
        first_attempt = attempts["retry_count"].long() == 0
        context = attempts["context"].float()[first_attempt]
        outcome = attempts["full_success"].float()[first_attempt]
        if torch.unique(
            attempts["state_index"].long()[first_attempt]
        ).numel() != context.shape[0]:
            raise ValueError(
                f"bootstrap dataset does not contain unique states: {path}"
            )
        contexts.append(context)
        outcomes.append(outcome)
        dataset_reports.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "seed": int(payload["seed"]),
                "unique_states": int(context.shape[0]),
                "successful": int(outcome.sum().item()),
            }
        )
    context = torch.cat(contexts)
    reward = torch.cat(outcomes)
    if context.shape[0] != args.expected_states:
        raise ValueError(
            f"expected {args.expected_states} bootstrap states, "
            f"found {context.shape[0]}"
        )
    correction, advantage = _promoted_candidate_selection(
        context,
        candidate_payload,
        position_cap_m=args.receiver_position_cap_m,
        orientation_cap_rad=receiver_orientation_cap_rad,
        local_sobol_seed=args.local_sobol_seed,
        local_position_radius_m=args.local_position_radius_m,
        local_orientation_radius_rad=math.radians(
            args.local_orientation_radius_deg
        ),
    )
    normalized_correction = torch.cat(
        (
            correction[:, :3] / args.receiver_position_cap_m,
            correction[:, 3:] / receiver_orientation_cap_rad,
        ),
        dim=-1,
    )
    features = torch.cat(
        (context, normalized_correction, advantage.unsqueeze(-1)),
        dim=-1,
    )
    if features.shape[-1] != actor_critic_type.input_dim:
        raise ValueError("receiver attempt feature shape drifted")
    feature_mean = features.mean(dim=0)
    feature_std = features.std(dim=0).clamp_min(1.0e-6)
    normalized = (features - feature_mean) / feature_std
    model = actor_critic_type(initial_std=args.initial_std)
    actor_before = {
        key: value.detach().clone()
        for key, value in model.actor.state_dict().items()
    }
    optimizer = torch.optim.AdamW(
        model.critic.parameters(),
        lr=args.critic_learning_rate,
        weight_decay=1.0e-6,
    )
    order_generator = torch.Generator().manual_seed(args.seed)
    for _ in range(args.critic_epochs):
        order = torch.randperm(
            normalized.shape[0],
            generator=order_generator,
        )
        for start in range(0, order.numel(), args.batch_size):
            indices = order[start : start + args.batch_size]
            logits = model.critic(normalized[indices]).squeeze(-1)
            loss = functional.binary_cross_entropy_with_logits(
                logits,
                reward[indices],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 0.5)
            optimizer.step()
    if any(
        not torch.equal(value, model.actor.state_dict()[key])
        for key, value in actor_before.items()
    ):
        raise RuntimeError("critic bootstrap modified the zero actor")
    with torch.no_grad():
        critic_probability = model.value(normalized)
        critic_loss = float(
            functional.binary_cross_entropy(
                critic_probability,
                reward,
            ).item()
        )
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "receiver_attempt_actor_critic": model.state_dict(),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "base_checkpoint_sha256": _sha256(base_checkpoint),
        "receiver_candidate_checkpoint_sha256": _sha256(
            candidate_checkpoint
        ),
        "receiver_gate_step": args.receiver_gate_step,
        "receiver_position_cap_m": args.receiver_position_cap_m,
        "receiver_orientation_cap_rad": receiver_orientation_cap_rad,
        "residual_position_cap_m": args.residual_position_cap_m,
        "residual_orientation_cap_rad": residual_orientation_cap_rad,
        "candidate_local_refinement": {
            "sobol_seed": args.local_sobol_seed,
            "position_radius_m": args.local_position_radius_m,
            "orientation_radius_rad": math.radians(
                args.local_orientation_radius_deg
            ),
        },
        "ppo": {
            "learning_rate": 1.0e-4,
            "clip": 0.1,
            "value_coefficient": 0.5,
            "entropy_coefficient": 0.001,
            "target_kl": 0.015,
            "max_gradient_norm": 0.5,
            "gamma": 1.0,
            "epochs": 6,
            "minibatch_size": 512,
            "decisions_per_update": 4096,
            "maximum_decisions": 61440,
        },
        "training": {
            "algorithm": "one_decision_contextual_bandit_ppo",
            "reward": "retained_full_success_binary",
            "decisions": 0,
            "updates": 0,
            "seed": args.seed,
            "bootstrap_unique_states": int(features.shape[0]),
            "bootstrap_successes": int(reward.sum().item()),
            "bootstrap_critic_bce": critic_loss,
            "bootstrap_datasets": dataset_reports,
            "rollouts": [],
        },
    }
    _write_checkpoint(
        checkpoint,
        Path(args.output).expanduser().resolve(),
    )
    return 0


def _update(args: argparse.Namespace) -> int:
    actor_critic_type, _ = _repo_models()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported receiver attempt PPO checkpoint")
    checkpoint_hash = _sha256(checkpoint_path)
    rollout_paths = [
        Path(value).expanduser().resolve() for value in args.rollout
    ]
    features_parts = []
    action_parts = []
    log_probability_parts = []
    old_value_parts = []
    reward_parts = []
    rollout_reports = []
    for path in rollout_paths:
        rollout = torch.load(path, map_location="cpu", weights_only=False)
        if (
            rollout.get("schema_version") != ROLLOUT_SCHEMA_VERSION
            or rollout.get("receiver_attempt_checkpoint_sha256")
            != checkpoint_hash
            or rollout.get("base_checkpoint_sha256")
            != payload["base_checkpoint_sha256"]
            or rollout.get("receiver_candidate_checkpoint_sha256")
            != payload["receiver_candidate_checkpoint_sha256"]
            or int(rollout["seed"]) not in DEVELOPMENT_SEEDS
            or not bool(rollout["stochastic"])
        ):
            raise ValueError(f"incompatible PPO rollout: {path}")
        for field in (
            "receiver_position_cap_m",
            "receiver_orientation_cap_rad",
            "residual_position_cap_m",
            "residual_orientation_cap_rad",
        ):
            if not math.isclose(
                float(rollout[field]),
                float(payload[field]),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise ValueError(f"rollout {field} drifted: {path}")
        features = rollout["features"].float()
        action = rollout["action"].float()
        if (
            features.shape[-1] != actor_critic_type.input_dim
            or action.shape[-1] != actor_critic_type.action_dim
        ):
            raise ValueError(f"rollout tensor shape drifted: {path}")
        features_parts.append(features)
        action_parts.append(action)
        log_probability_parts.append(
            rollout["old_log_probability"].float()
        )
        old_value_parts.append(rollout["old_value"].float())
        reward_parts.append(rollout["full_success"].float())
        rollout_reports.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "seed": int(rollout["seed"]),
                "decisions": int(features.shape[0]),
                "successful": int(
                    rollout["full_success"].sum().item()
                ),
            }
        )
    features = torch.cat(features_parts)
    action = torch.cat(action_parts)
    old_log_probability = torch.cat(log_probability_parts)
    old_value = torch.cat(old_value_parts)
    reward = torch.cat(reward_parts)
    if features.shape[0] < args.minimum_decisions:
        raise ValueError(
            f"PPO update requires at least {args.minimum_decisions} "
            f"decisions, found {features.shape[0]}"
        )
    normalized = (
        features - payload["feature_mean"].float()
    ) / payload["feature_std"].float()
    model = actor_critic_type()
    model.load_state_dict(
        payload["receiver_attempt_actor_critic"],
        strict=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1.0e-6,
    )
    advantage = reward - old_value
    advantage = (
        advantage - advantage.mean()
    ) / advantage.std().clamp_min(1.0e-6)
    generator = torch.Generator().manual_seed(args.seed)
    epochs_completed = 0
    approximate_kl = 0.0
    policy_loss_value = 0.0
    value_loss_value = 0.0
    entropy_value = 0.0
    for epoch in range(args.epochs):
        order = torch.randperm(normalized.shape[0], generator=generator)
        stop_for_kl = False
        for start in range(0, order.numel(), args.minibatch_size):
            indices = order[start : start + args.minibatch_size]
            new_log_probability, entropy, value = model.evaluate_actions(
                normalized[indices],
                action[indices],
            )
            log_ratio = (
                new_log_probability - old_log_probability[indices]
            )
            ratio = torch.exp(log_ratio)
            unclipped = ratio * advantage[indices]
            clipped = (
                ratio.clamp(1.0 - args.clip, 1.0 + args.clip)
                * advantage[indices]
            )
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = functional.mse_loss(value, reward[indices])
            entropy_mean = entropy.mean()
            loss = (
                policy_loss
                + args.value_coefficient * value_loss
                - args.entropy_coefficient * entropy_mean
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.max_gradient_norm,
            )
            optimizer.step()
            with torch.no_grad():
                model.log_std.clamp_(
                    math.log(0.05),
                    math.log(0.5),
                )
                approximate_kl = float(
                    ((ratio - 1.0) - log_ratio).mean().item()
                )
            policy_loss_value = float(policy_loss.item())
            value_loss_value = float(value_loss.item())
            entropy_value = float(entropy_mean.item())
            if approximate_kl > args.target_kl:
                stop_for_kl = True
                break
        epochs_completed = epoch + 1
        if stop_for_kl:
            break
    payload["receiver_attempt_actor_critic"] = model.state_dict()
    training = payload["training"]
    training["decisions"] = int(training["decisions"]) + int(
        features.shape[0]
    )
    training["updates"] = int(training["updates"]) + 1
    training["rollouts"].extend(rollout_reports)
    training["last_update"] = {
        "decisions": int(features.shape[0]),
        "successful": int(reward.sum().item()),
        "success_rate": float(reward.mean().item()),
        "epochs_completed": epochs_completed,
        "approximate_kl": approximate_kl,
        "policy_loss": policy_loss_value,
        "value_loss": value_loss_value,
        "entropy": entropy_value,
    }
    _write_checkpoint(
        payload,
        Path(args.output).expanduser().resolve(),
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the one-decision receiver attempt PPO residual"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    bootstrap = subparsers.add_parser("bootstrap")
    bootstrap.add_argument("--base_checkpoint", required=True)
    bootstrap.add_argument("--candidate_checkpoint", required=True)
    bootstrap.add_argument("--dataset", action="append", required=True)
    bootstrap.add_argument("--output", required=True)
    bootstrap.add_argument("--expected_states", type=int, default=1647)
    bootstrap.add_argument("--receiver_gate_step", type=int, default=50)
    bootstrap.add_argument(
        "--receiver_position_cap_m",
        type=float,
        default=0.0025,
    )
    bootstrap.add_argument(
        "--receiver_orientation_cap_deg",
        type=float,
        default=2.0,
    )
    bootstrap.add_argument(
        "--residual_position_cap_m",
        type=float,
        default=0.001,
    )
    bootstrap.add_argument(
        "--residual_orientation_cap_deg",
        type=float,
        default=1.0,
    )
    bootstrap.add_argument("--local_sobol_seed", type=int, default=104748)
    bootstrap.add_argument(
        "--local_position_radius_m",
        type=float,
        default=0.001,
    )
    bootstrap.add_argument(
        "--local_orientation_radius_deg",
        type=float,
        default=1.0,
    )
    bootstrap.add_argument("--initial_std", type=float, default=0.25)
    bootstrap.add_argument("--critic_epochs", type=int, default=100)
    bootstrap.add_argument("--critic_learning_rate", type=float, default=1.0e-3)
    bootstrap.add_argument("--batch_size", type=int, default=512)
    bootstrap.add_argument("--seed", type=int, default=104729)

    update = subparsers.add_parser("update")
    update.add_argument("--checkpoint", required=True)
    update.add_argument("--rollout", action="append", required=True)
    update.add_argument("--output", required=True)
    update.add_argument("--minimum_decisions", type=int, default=4096)
    update.add_argument("--epochs", type=int, default=6)
    update.add_argument("--minibatch_size", type=int, default=512)
    update.add_argument("--learning_rate", type=float, default=1.0e-4)
    update.add_argument("--clip", type=float, default=0.1)
    update.add_argument("--value_coefficient", type=float, default=0.5)
    update.add_argument("--entropy_coefficient", type=float, default=0.001)
    update.add_argument("--target_kl", type=float, default=0.015)
    update.add_argument("--max_gradient_norm", type=float, default=0.5)
    update.add_argument("--seed", type=int, default=104729)
    return parser


def main(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    if args.command == "bootstrap":
        return _bootstrap(args)
    return _update(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
