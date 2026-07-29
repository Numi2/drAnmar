#!/usr/bin/env python3
"""Distill and train the paired-outcome receiver context selector."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as functional


DEVELOPMENT_SEEDS = {104729, 130363, 196613}
SCHEMA_VERSION = "dranmar-receiver-context-selector-1.0"
DATASET_SCHEMA_VERSION = "dranmar-receiver-recovery-dataset-1.2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_revision() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repo_models() -> tuple[type[torch.nn.Module], type[torch.nn.Module]]:
    module_name = "_dranmar_receiver_selector_recovery_policy"
    recovery_policy = sys.modules.get(module_name)
    if recovery_policy is None:
        for package_name in (
            "orbit",
            "orbit.surgical",
            "orbit.surgical.tasks",
            "orbit.surgical.tasks.surgical",
            "orbit.surgical.tasks.surgical.lift",
            "orbit.surgical.tasks.surgical.handover",
        ):
            package = types.ModuleType(package_name)
            package.__path__ = []
            sys.modules.setdefault(package_name, package)
        grasp_frames = types.ModuleType(
            "orbit.surgical.tasks.surgical.lift.grasp_frames"
        )
        grasp_frames.NEEDLE_PROVISIONAL_GRASP_OFFSET_M = (
            -0.0072,
            0.0015,
            0.0,
        )
        grasp_frames.needle_geometry_grasp_offset_m = lambda fraction: (
            -0.004,
            0.003,
            0.0,
        )
        sys.modules[grasp_frames.__name__] = grasp_frames
        module_path = (
            Path(__file__).resolve().parents[1]
            / "source/extensions/orbit.surgical.tasks/orbit/surgical/tasks/"
            "surgical/handover/recovery_policy.py"
        )
        spec = importlib.util.spec_from_file_location(
            module_name,
            module_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load recovery policy: {module_path}")
        recovery_policy = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = recovery_policy
        spec.loader.exec_module(recovery_policy)
    return (
        recovery_policy.ReceiverContextCandidateSelector,
        recovery_policy.ReceiverCandidateValue,
    )


def _candidate_scores(
    context: torch.Tensor,
    candidate_payload: dict[str, object],
) -> torch.Tensor:
    _, value_type = _repo_models()
    model = value_type()
    model.load_state_dict(
        candidate_payload["receiver_candidate_value"],
        strict=True,
    )
    model.eval()
    candidates = candidate_payload["candidate_corrections"].float()
    position_cap = float(candidate_payload["position_cap_m"])
    orientation_cap = float(candidate_payload["orientation_cap_rad"])
    normalized_candidates = torch.cat(
        (
            candidates[:, :3] / position_cap,
            candidates[:, 3:] / orientation_cap,
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
        return model(
            (
                (features - candidate_payload["feature_mean"].float())
                / candidate_payload["feature_std"].float()
            ).reshape(-1, 35)
        ).reshape(context.shape[0], candidates.shape[0])


def _promoted_labels(
    context: torch.Tensor,
    candidate_payload: dict[str, object],
) -> torch.Tensor:
    scores = _candidate_scores(context, candidate_payload)
    best = scores.argmax(dim=-1)
    zero = (
        candidate_payload["candidate_corrections"]
        .float()
        .square()
        .sum(dim=-1)
        .argmin()
    )
    best_score = scores.gather(1, best.unsqueeze(-1)).squeeze(-1)
    return torch.where(best_score >= scores[:, zero], best, zero)


def _write_checkpoint(payload: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    report = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "receiver_context_selector",
            "feature_mean",
            "feature_std",
            "candidate_corrections",
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


def _load_bootstrap_contexts(
    paths: list[Path],
    *,
    base_hash: str,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    contexts = []
    reports = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        attempts = payload.get("attempts") or payload
        first = attempts["retry_count"].long() == 0
        context = attempts["context"].float()[first]
        if (
            payload.get("schema_version") != DATASET_SCHEMA_VERSION
            or int(payload["seed"]) not in DEVELOPMENT_SEEDS
            or payload.get("base_checkpoint_sha256") != base_hash
            or context.shape[-1] != 29
            or torch.unique(
                attempts["state_index"].long()[first]
            ).numel()
            != context.shape[0]
        ):
            raise ValueError(f"incompatible distillation dataset: {path}")
        contexts.append(context)
        reports.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "seed": int(payload["seed"]),
                "states": int(context.shape[0]),
            }
        )
    return torch.cat(contexts), reports


def _bootstrap(args: argparse.Namespace) -> int:
    selector_type, _ = _repo_models()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    base_path = Path(args.base_checkpoint).expanduser().resolve()
    candidate_path = Path(args.candidate_checkpoint).expanduser().resolve()
    base_hash = _sha256(base_path)
    candidate_payload = torch.load(
        candidate_path,
        map_location="cpu",
        weights_only=False,
    )
    if (
        candidate_payload.get("schema_version")
        != "dranmar-receiver-candidate-value-1.0"
        or candidate_payload.get("base_checkpoint_sha256") != base_hash
        or candidate_payload["candidate_corrections"].shape != (16, 6)
    ):
        raise ValueError("candidate checkpoint is not the frozen common-16")
    context, reports = _load_bootstrap_contexts(
        [Path(value).expanduser().resolve() for value in args.dataset],
        base_hash=base_hash,
    )
    if context.shape[0] != args.expected_states:
        raise ValueError(
            f"expected {args.expected_states} distillation states, "
            f"found {context.shape[0]}"
        )
    labels = _promoted_labels(context, candidate_payload)
    feature_mean = context.mean(dim=0)
    feature_std = context.std(dim=0).clamp_min(1.0e-6)
    normalized = (context - feature_mean) / feature_std
    model = selector_type()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1.0e-6,
    )
    generator = torch.Generator().manual_seed(args.seed)
    for _ in range(args.epochs):
        order = torch.randperm(context.shape[0], generator=generator)
        for start in range(0, order.numel(), args.batch_size):
            indices = order[start : start + args.batch_size]
            loss = functional.cross_entropy(
                model(normalized[indices]),
                labels[indices],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
    with torch.no_grad():
        accuracy = float(
            (model(normalized).argmax(dim=-1) == labels).float().mean()
        )
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "source_revision": _source_revision(),
        "receiver_context_selector": model.state_dict(),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "candidate_corrections": (
            candidate_payload["candidate_corrections"].float()
        ),
        "base_checkpoint_sha256": base_hash,
        "receiver_candidate_checkpoint_sha256": _sha256(candidate_path),
        "receiver_gate_step": 50,
        "position_cap_m": float(candidate_payload["position_cap_m"]),
        "orientation_cap_rad": float(
            candidate_payload["orientation_cap_rad"]
        ),
        "training": {
            "algorithm": "promoted_selector_distillation",
            "seed": args.seed,
            "distillation_states": int(context.shape[0]),
            "distillation_accuracy": accuracy,
            "distillation_datasets": reports,
            "paired_states": 0,
            "updates": 0,
        },
    }
    _write_checkpoint(
        checkpoint,
        Path(args.output).expanduser().resolve(),
    )
    return 0


def _paired_states(
    paths: list[Path],
    *,
    checkpoint: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    contexts = []
    outcomes = []
    seeds = []
    reports = []
    candidates = checkpoint["candidate_corrections"].float()
    for dataset_id, path in enumerate(paths):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        attempts = payload.get("attempts") or payload
        seed = int(payload["seed"])
        if (
            payload.get("schema_version") != DATASET_SCHEMA_VERSION
            or seed not in DEVELOPMENT_SEEDS
            or payload.get("base_checkpoint_sha256")
            != checkpoint["base_checkpoint_sha256"]
            or int(payload["sweep_replicas"]) != 16
            or int(payload["sobol_seed"]) != 104730
        ):
            raise ValueError(f"incompatible common-16 dataset: {path}")
        first = attempts["retry_count"].long() == 0
        state_index = attempts["state_index"].long()[first]
        candidate_index = attempts["candidate_index"].long()[first]
        context = attempts["context"].float()[first]
        correction = attempts["correction"].float()[first]
        success = attempts["full_success"].bool()[first]
        dataset_states = 0
        dataset_context_max_abs_drift = 0.0
        dataset_context_mean_abs_drift = 0.0
        for state in torch.unique(state_index).tolist():
            mask = state_index == state
            order = candidate_index[mask].argsort()
            indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)[order]
            if not torch.equal(
                candidate_index[indices],
                torch.arange(16),
            ):
                continue
            state_context = context[indices]
            if not torch.allclose(
                correction[indices],
                candidates,
                atol=1.0e-7,
                rtol=0.0,
            ):
                raise ValueError(
                    f"paired state/candidate drift in dataset: {path}"
                )
            state_success = success[indices]
            if bool(state_success.all()):
                continue
            context_drift = (
                state_context - state_context[:1]
            ).abs()
            dataset_context_max_abs_drift = max(
                dataset_context_max_abs_drift,
                float(context_drift.max()),
            )
            dataset_context_mean_abs_drift += float(
                context_drift.mean()
            )
            # Grouped PhysX replicas share the reset stream but can diverge
            # before the receiver gate. Candidate zero is the canonical,
            # correction-free context available at deployment; the remaining
            # replicas contribute only their paired retained outcomes.
            contexts.append(state_context[0])
            outcomes.append(state_success)
            seeds.append(seed)
            dataset_states += 1
        reports.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "seed": seed,
                "failure_heavy_states": dataset_states,
                "dataset_id": dataset_id,
                "context_representation": (
                    "candidate_0_gate_context"
                ),
                "context_max_abs_drift": (
                    dataset_context_max_abs_drift
                ),
                "context_mean_abs_drift": (
                    dataset_context_mean_abs_drift
                    / max(dataset_states, 1)
                ),
                "seed_stream_offset": int(
                    payload.get("seed_stream_offset", 0)
                ),
            }
        )
    return (
        torch.stack(contexts),
        torch.stack(outcomes),
        torch.tensor(seeds, dtype=torch.long),
        reports,
    )


def _ranking(
    logits: torch.Tensor,
    outcomes: torch.Tensor,
) -> dict[str, object]:
    selected = logits.argmax(dim=-1)
    selected_success = outcomes.gather(
        1,
        selected.unsqueeze(-1),
    ).squeeze(-1)
    oracle = outcomes.any(dim=-1)
    return {
        "states": int(outcomes.shape[0]),
        "oracle_success_states": int(oracle.sum()),
        "selected_success_states": int(selected_success.sum()),
        "selected_success_rate": float(selected_success.float().mean()),
        "oracle_capture_rate": (
            float(selected_success[oracle].float().mean())
            if bool(oracle.any())
            else None
        ),
    }


def _update(args: argparse.Namespace) -> int:
    selector_type, _ = _repo_models()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported receiver selector checkpoint")
    context, outcomes, seeds, reports = _paired_states(
        [Path(value).expanduser().resolve() for value in args.dataset],
        checkpoint=checkpoint,
    )
    if torch.unique(context, dim=0).shape[0] != context.shape[0]:
        raise ValueError("paired selector datasets contain duplicate contexts")
    if context.shape[0] < args.minimum_states:
        raise ValueError(
            f"need at least {args.minimum_states} failure-heavy states, "
            f"found {context.shape[0]}"
        )
    for seed in DEVELOPMENT_SEEDS:
        count = int((seeds == seed).sum())
        if count < args.minimum_states_per_seed:
            raise ValueError(
                f"seed {seed} has {count} failure-heavy states; "
                f"need {args.minimum_states_per_seed}"
            )
    normalized = (
        context - checkpoint["feature_mean"].float()
    ) / checkpoint["feature_std"].float()
    model = selector_type()
    model.load_state_dict(
        checkpoint["receiver_context_selector"],
        strict=True,
    )
    with torch.no_grad():
        distilled_logits = model(normalized).detach()
    generator = torch.Generator().manual_seed(args.seed)
    validation_mask = torch.zeros(context.shape[0], dtype=torch.bool)
    for seed in DEVELOPMENT_SEEDS:
        seed_indices = torch.nonzero(
            seeds == seed,
            as_tuple=False,
        ).squeeze(-1)
        order = seed_indices[
            torch.randperm(seed_indices.numel(), generator=generator)
        ]
        validation_mask[order[: max(1, order.numel() // 10)]] = True
    train_mask = ~validation_mask
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1.0e-6,
    )
    best_state = None
    best_successes = -1
    stale_epochs = 0
    train_indices = torch.nonzero(
        train_mask,
        as_tuple=False,
    ).squeeze(-1)
    for _ in range(args.epochs):
        order = train_indices[
            torch.randperm(train_indices.numel(), generator=generator)
        ]
        for start in range(0, order.numel(), args.batch_size):
            indices = order[start : start + args.batch_size]
            logits = model(normalized[indices])
            success = outcomes[indices]
            probability = torch.softmax(logits, dim=-1)
            has_success = success.any(dim=-1)
            success_mass = (
                probability * success.float()
            ).sum(dim=-1).clamp_min(1.0e-8)
            if bool(has_success.any()):
                listwise = -torch.log(
                    success_mass[has_success]
                ).mean()
            else:
                listwise = logits.new_zeros(())
            no_success = ~has_success
            if bool(no_success.any()):
                fallback = distilled_logits[indices][no_success].argmax(
                    dim=-1
                )
                listwise = listwise + functional.cross_entropy(
                    logits[no_success],
                    fallback,
                )
            distillation = functional.kl_div(
                torch.log_softmax(logits, dim=-1),
                torch.softmax(distilled_logits[indices], dim=-1),
                reduction="batchmean",
            )
            loss = listwise + args.distillation_weight * distillation
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        with torch.no_grad():
            validation_logits = model(normalized[validation_mask])
            validation_successes = int(
                outcomes[validation_mask]
                .gather(
                    1,
                    validation_logits.argmax(dim=-1).unsqueeze(-1),
                )
                .sum()
            )
        if validation_successes > best_successes:
            best_successes = validation_successes
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state, strict=True)
    with torch.no_grad():
        before = _ranking(distilled_logits, outcomes)
        after = _ranking(model(normalized), outcomes)
        validation = _ranking(
            model(normalized[validation_mask]),
            outcomes[validation_mask],
        )
    checkpoint["receiver_context_selector"] = model.state_dict()
    checkpoint["source_revision"] = _source_revision()
    training = checkpoint["training"]
    training["algorithm"] = "paired_common16_listwise_selector"
    training["paired_states"] = int(context.shape[0])
    training["paired_datasets"] = reports
    training["before_ranking"] = before
    training["after_ranking"] = after
    training["validation_ranking"] = validation
    training["listwise_configuration"] = {
        "learning_rate": args.learning_rate,
        "distillation_weight": args.distillation_weight,
        "batch_size": args.batch_size,
        "maximum_epochs": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
    }
    training["updates"] = int(training["updates"]) + 1
    _write_checkpoint(
        checkpoint,
        Path(args.output).expanduser().resolve(),
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the 16-way receiver context selector"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    bootstrap = commands.add_parser("bootstrap")
    bootstrap.add_argument("--base_checkpoint", required=True)
    bootstrap.add_argument("--candidate_checkpoint", required=True)
    bootstrap.add_argument("--output", required=True)
    bootstrap.add_argument("--dataset", action="append", required=True)
    bootstrap.add_argument("--expected_states", type=int, default=1647)
    bootstrap.add_argument("--epochs", type=int, default=300)
    bootstrap.add_argument("--batch_size", type=int, default=512)
    bootstrap.add_argument("--learning_rate", type=float, default=1.0e-3)
    bootstrap.add_argument("--seed", type=int, default=104729)
    bootstrap.set_defaults(handler=_bootstrap)
    update = commands.add_parser("update")
    update.add_argument("--checkpoint", required=True)
    update.add_argument("--output", required=True)
    update.add_argument("--dataset", action="append", required=True)
    update.add_argument("--minimum_states", type=int, default=900)
    update.add_argument("--minimum_states_per_seed", type=int, default=300)
    update.add_argument("--epochs", type=int, default=400)
    update.add_argument("--patience", type=int, default=40)
    update.add_argument("--batch_size", type=int, default=256)
    update.add_argument("--learning_rate", type=float, default=3.0e-4)
    update.add_argument("--distillation_weight", type=float, default=0.1)
    update.add_argument("--seed", type=int, default=104729)
    update.set_defaults(handler=_update)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
