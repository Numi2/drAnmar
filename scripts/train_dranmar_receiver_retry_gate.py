#!/usr/bin/env python3
"""Train an early receiver-retry gate from canonical physical trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional


DEVELOPMENT_SEEDS = {104729, 130363, 196613}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the receiver failure gate at active approach step 100"
    )
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--baseline_evidence", action="append")
    parser.add_argument("--intervention_evidence", action="append")
    parser.add_argument(
        "--retry_intervention_target",
        action="store_true",
        help=(
            "emit a retry-gate checkpoint from paired baseline versus "
            "receiver-recovery evidence"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-6)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--validation_seed", type=int)
    parser.add_argument("--seed", type=int, default=104729)
    return parser


def _threshold_metrics(
    probabilities: torch.Tensor,
    failure: torch.Tensor,
) -> dict[str, dict[str, float | int | None]]:
    result = {}
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
        selected = probabilities >= threshold
        true_positive = int((selected & failure).sum().item())
        false_positive = int((selected & ~failure).sum().item())
        false_negative = int((~selected & failure).sum().item())
        true_negative = int((~selected & ~failure).sum().item())
        result[f"{threshold:.1f}"] = {
            "failure_selected": true_positive,
            "canonical_success_preempted": false_positive,
            "failure_missed": false_negative,
            "canonical_success_preserved": true_negative,
            "precision": (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive
                else None
            ),
            "recall": (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative
                else None
            ),
        }
    return result


def _intervention_threshold_metrics(
    probabilities: torch.Tensor,
    outcome_delta: torch.Tensor,
) -> dict[str, dict[str, float | int | None]]:
    result = {}
    benefit = outcome_delta > 0
    harm = outcome_delta < 0
    neutral = outcome_delta == 0
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
        selected = probabilities >= threshold
        benefit_selected = int((selected & benefit).sum().item())
        harm_selected = int((selected & harm).sum().item())
        neutral_selected = int((selected & neutral).sum().item())
        result[f"{threshold:.1f}"] = {
            "benefit_selected": benefit_selected,
            "harm_selected": harm_selected,
            "neutral_selected": neutral_selected,
            "net_success_delta": benefit_selected - harm_selected,
            "selection_precision": (
                benefit_selected / (benefit_selected + harm_selected)
                if benefit_selected + harm_selected
                else None
            ),
            "benefit_recall": (
                benefit_selected / int(benefit.sum().item())
                if bool(benefit.any())
                else None
            ),
        }
    return result


def main(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    if not 0.05 <= args.validation_fraction <= 0.5:
        raise ValueError("validation fraction must be in [0.05, 0.5]")
    if (
        args.validation_seed is not None
        and args.validation_seed not in DEVELOPMENT_SEEDS
    ):
        raise ValueError("validation seed must be a development seed")

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
        ReceiverRetryGate,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset_paths = [
        Path(value).expanduser().resolve() for value in args.dataset
    ]
    payloads = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in dataset_paths
    ]
    for path, payload in zip(dataset_paths, payloads, strict=True):
        if (
            payload.get("schema_version")
            != "dranmar-receiver-retry-gate-dataset-1.0"
        ):
            raise ValueError(f"unsupported receiver gate dataset: {path}")
        seed = int(payload["seed"])
        if seed not in DEVELOPMENT_SEEDS:
            raise ValueError(f"dataset seed is not development-only: {path}")
        if int(payload["active_approach_step"]) != 100:
            raise ValueError(f"receiver gate probe step drifted in {path}")

    base_hashes = {payload["base_checkpoint_sha256"] for payload in payloads}
    pickup_hashes = {
        payload.get("pickup_recovery_checkpoint_sha256")
        for payload in payloads
    }
    if len(base_hashes) != 1 or len(pickup_hashes) != 1:
        raise ValueError("receiver gate datasets do not share frozen policies")

    counterfactual_mode = bool(
        args.baseline_evidence or args.intervention_evidence
    )
    if counterfactual_mode and (
        not args.baseline_evidence
        or not args.intervention_evidence
        or len(args.baseline_evidence) != len(payloads)
        or len(args.intervention_evidence) != len(payloads)
    ):
        raise ValueError(
            "counterfactual training requires one baseline and one "
            "intervention evidence file per dataset"
        )
    if args.retry_intervention_target and not counterfactual_mode:
        raise ValueError(
            "retry intervention target requires counterfactual evidence"
        )
    baseline_evidence_paths = (
        [
            Path(value).expanduser().resolve()
            for value in args.baseline_evidence
        ]
        if counterfactual_mode
        else []
    )
    intervention_evidence_paths = (
        [
            Path(value).expanduser().resolve()
            for value in args.intervention_evidence
        ]
        if counterfactual_mode
        else []
    )
    baseline_evidence = (
        [
            json.loads(path.read_text())
            for path in baseline_evidence_paths
        ]
        if counterfactual_mode
        else []
    )
    intervention_evidence = (
        [
            json.loads(path.read_text())
            for path in intervention_evidence_paths
        ]
        if counterfactual_mode
        else []
    )

    feature_parts = []
    failure_parts = []
    seed_parts = []
    intervention_delta_parts = []
    counterfactual_counts = []
    for payload_index, payload in enumerate(payloads):
        features = torch.cat(
            (
                payload["observation"].float(),
                payload["receiver_action"].float(),
            ),
            dim=-1,
        )
        if features.shape[-1] != ReceiverRetryGate.input_dim:
            raise ValueError("receiver retry gate feature shape drifted")
        if counterfactual_mode:
            baseline = baseline_evidence[payload_index]
            intervention = intervention_evidence[payload_index]
            seed = int(payload["seed"])
            if (
                int(baseline["seed"]) != seed
                or int(intervention["seed"]) != seed
            ):
                raise ValueError(
                    "counterfactual evidence seed does not match dataset"
                )
            environment_index = payload["environment_index"].long()
            baseline_success_indices = set(
                baseline["environment_outcomes"]["successful_indices"]
            )
            intervention_success_indices = set(
                intervention["environment_outcomes"][
                    "successful_indices"
                ]
            )
            baseline_success = torch.tensor(
                [
                    int(index) in baseline_success_indices
                    for index in environment_index
                ],
                dtype=torch.bool,
            )
            intervention_success = torch.tensor(
                [
                    int(index) in intervention_success_indices
                    for index in environment_index
                ],
                dtype=torch.bool,
            )
            changed = baseline_success ^ intervention_success
            benefit = ~baseline_success & intervention_success
            counterfactual_counts.append(
                {
                    "seed": seed,
                    "changed_samples": int(changed.sum().item()),
                    "benefits": int(benefit.sum().item()),
                    "harms": int(
                        (baseline_success & ~intervention_success)
                        .sum()
                        .item()
                    ),
                }
            )
            target = benefit
            intervention_delta_parts.append(
                intervention_success.to(torch.int8)
                - baseline_success.to(torch.int8)
            )
        else:
            target = ~payload["eventual_acquisition"].bool()
        feature_parts.append(features)
        failure_parts.append(target)
        seed_parts.append(
            torch.full(
                (features.shape[0],),
                int(payload["seed"]),
                dtype=torch.long,
            )
        )
    features = torch.cat(feature_parts)
    failure = torch.cat(failure_parts)
    sample_seed = torch.cat(seed_parts)
    intervention_delta = (
        torch.cat(intervention_delta_parts)
        if counterfactual_mode
        else None
    )

    if args.validation_seed is not None:
        validation_mask = sample_seed == args.validation_seed
        training_mask = ~validation_mask
    else:
        permutation = torch.randperm(
            features.shape[0],
            generator=torch.Generator().manual_seed(args.seed),
        )
        validation_count = max(
            1,
            int(features.shape[0] * args.validation_fraction),
        )
        validation_mask = torch.zeros(features.shape[0], dtype=torch.bool)
        validation_mask[permutation[:validation_count]] = True
        training_mask = ~validation_mask
    if not bool(training_mask.any()) or not bool(validation_mask.any()):
        raise ValueError("receiver gate split produced an empty partition")

    feature_mean = features[training_mask].mean(dim=0)
    feature_std = features[training_mask].std(dim=0).clamp_min(1.0e-6)
    normalized = (features - feature_mean) / feature_std
    training_features = normalized[training_mask]
    training_failure = failure[training_mask].float()
    validation_features = normalized[validation_mask]
    validation_failure = failure[validation_mask]
    training_intervention_delta = (
        intervention_delta[training_mask]
        if intervention_delta is not None
        else None
    )
    validation_intervention_delta = (
        intervention_delta[validation_mask]
        if intervention_delta is not None
        else None
    )

    positive_count = training_failure.sum().clamp_min(1.0)
    negative_count = (
        training_failure.shape[0] - training_failure.sum()
    ).clamp_min(1.0)
    positive_weight = negative_count / positive_count

    gate = ReceiverRetryGate()
    optimizer = torch.optim.AdamW(
        gate.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_loss = float("inf")
    best_state = None
    final_training_loss = None
    for epoch in range(args.epochs):
        order = torch.randperm(
            training_features.shape[0],
            generator=torch.Generator().manual_seed(args.seed + epoch),
        )
        gate.train()
        losses = []
        for start in range(0, order.numel(), args.batch_size):
            indices = order[start : start + args.batch_size]
            logits = gate(training_features[indices])
            loss = functional.binary_cross_entropy_with_logits(
                logits,
                training_failure[indices],
                pos_weight=positive_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
        final_training_loss = sum(losses) / len(losses)
        gate.eval()
        with torch.no_grad():
            validation_loss = float(
                functional.binary_cross_entropy_with_logits(
                    gate(validation_features),
                    validation_failure.float(),
                    pos_weight=positive_weight,
                ).item()
            )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                key: value.detach().clone()
                for key, value in gate.state_dict().items()
            }

    assert best_state is not None
    gate.load_state_dict(best_state)
    gate.eval()
    with torch.no_grad():
        validation_probability = torch.sigmoid(
            gate(validation_features)
        )
        training_probability = torch.sigmoid(gate(training_features))

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    retry_gate_output = (
        not counterfactual_mode or args.retry_intervention_target
    )
    checkpoint = {
        "schema_version": (
            "dranmar-receiver-retry-gate-1.0"
            if retry_gate_output
            else "dranmar-receiver-intervention-gate-1.0"
        ),
        (
            "receiver_retry_gate"
            if retry_gate_output
            else "receiver_intervention_gate"
        ): gate.state_dict(),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "base_checkpoint_sha256": next(iter(base_hashes)),
        "pickup_recovery_checkpoint_sha256": next(iter(pickup_hashes)),
        "active_approach_step": 100,
        "intervention": (
            (
                "receiver_recovery"
                if args.retry_intervention_target
                else "stabilize_giver_during_acquisition"
            )
            if counterfactual_mode
            else None
        ),
        "counterfactual_counts": counterfactual_counts,
        "training": {
            "algorithm": (
                "class_balanced_intervention_benefit_classification"
                if counterfactual_mode
                else "class_balanced_failure_classification"
            ),
            "seed": args.seed,
            "validation_seed": args.validation_seed,
            "epochs": args.epochs,
            "dataset_samples": int(features.shape[0]),
            "training_samples": int(training_mask.sum().item()),
            "validation_samples": int(validation_mask.sum().item()),
            "training_failures": int(training_failure.sum().item()),
            "validation_failures": int(validation_failure.sum().item()),
            "final_training_loss": final_training_loss,
            "best_validation_loss": best_loss,
            "training_threshold_metrics": (
                _intervention_threshold_metrics(
                    training_probability,
                    training_intervention_delta,
                )
                if counterfactual_mode
                else _threshold_metrics(
                    training_probability,
                    training_failure.bool(),
                )
            ),
            "validation_threshold_metrics": (
                _intervention_threshold_metrics(
                    validation_probability,
                    validation_intervention_delta,
                )
                if counterfactual_mode
                else _threshold_metrics(
                    validation_probability,
                    validation_failure,
                )
            ),
        },
        "datasets": [
            {
                "path": str(path),
                "sha256": _sha256(path),
                "seed": int(payload["seed"]),
                "samples": int(payload["observation"].shape[0]),
            }
            for path, payload in zip(dataset_paths, payloads, strict=True)
        ],
        "counterfactual_evidence": (
            [
                {
                    "baseline_path": str(baseline_path),
                    "baseline_sha256": _sha256(baseline_path),
                    "intervention_path": str(intervention_path),
                    "intervention_sha256": _sha256(intervention_path),
                }
                for baseline_path, intervention_path in zip(
                    baseline_evidence_paths,
                    intervention_evidence_paths,
                    strict=True,
                )
            ]
            if counterfactual_mode
            else []
        ),
    }
    torch.save(checkpoint, output)
    report = {
        key: value
        for key, value in checkpoint.items()
        if key
        not in {
            "receiver_retry_gate",
            "receiver_intervention_gate",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
