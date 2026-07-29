#!/usr/bin/env python3
"""Learn to rank physically tested receiver-recovery corrections."""

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
        description="Rank 65 receiver-recovery correction candidates"
    )
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=3.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-6)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--validation_seed", type=int)
    parser.add_argument("--candidate_seed", type=int, default=130363)
    parser.add_argument("--first_attempt_only", action="store_true")
    parser.add_argument("--seed", type=int, default=104729)
    return parser


def _ranking_metrics(
    logits: torch.Tensor,
    retained: torch.Tensor,
    group: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float | int | None]:
    selected_success = 0
    oracle_success = 0
    groups = torch.unique(group[mask])
    for group_id in groups:
        indices = torch.nonzero(
            mask & (group == group_id),
            as_tuple=False,
        ).squeeze(-1)
        selected = indices[logits[indices].argmax()]
        selected_success += int(retained[selected].item())
        oracle_success += int(bool(retained[indices].any()))
    count = int(groups.numel())
    return {
        "states": count,
        "oracle_retained_states": oracle_success,
        "selected_retained_states": selected_success,
        "selected_retention_rate": (
            selected_success / count if count else None
        ),
        "oracle_conversion_rate": (
            oracle_success / count if count else None
        ),
        "oracle_capture_rate": (
            selected_success / oracle_success if oracle_success else None
        ),
    }


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
    if args.candidate_seed not in DEVELOPMENT_SEEDS:
        raise ValueError("candidate seed must be a development seed")

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
        ReceiverCandidateValue,
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
    base_hashes = {payload["base_checkpoint_sha256"] for payload in payloads}
    pickup_hashes = {
        payload.get("pickup_recovery_checkpoint_sha256")
        for payload in payloads
    }
    position_caps = {float(payload["position_cap_m"]) for payload in payloads}
    orientation_caps = {
        float(payload["orientation_cap_rad"]) for payload in payloads
    }
    if (
        len(base_hashes) != 1
        or len(pickup_hashes) != 1
        or len(position_caps) != 1
        or len(orientation_caps) != 1
    ):
        raise ValueError("receiver candidate datasets are incompatible")
    position_cap = position_caps.pop()
    orientation_cap = orientation_caps.pop()

    context_parts = []
    correction_parts = []
    retained_parts = []
    seed_parts = []
    group_parts = []
    candidate_parts = []
    group_id = 0
    dataset_reports = []
    for path, payload in zip(dataset_paths, payloads, strict=True):
        seed = int(payload["seed"])
        if seed not in DEVELOPMENT_SEEDS:
            raise ValueError(f"non-development dataset: {path}")
        samples = payload.get("attempts")
        if not samples:
            raise ValueError(f"candidate dataset lacks attempts: {path}")
        selected_attempt = samples["retry_count"].long() == (
            0 if args.first_attempt_only else 1
        )
        context = samples["context"].float()[selected_attempt]
        correction = samples["correction"].float()[selected_attempt]
        candidate = samples["candidate_index"].long()[selected_attempt]
        state = samples["state_index"].long()[selected_attempt]
        retained = (
            samples["retained"].bool()[selected_attempt]
            & samples["safe_acquisition"].bool()[selected_attempt]
        )
        if context.shape[-1] != 29 or correction.shape[-1] != 6:
            raise ValueError(f"receiver candidate shape drifted: {path}")
        group = torch.empty_like(state)
        for state_id in torch.unique(state):
            group[state == state_id] = group_id
            group_id += 1
        context_parts.append(context)
        correction_parts.append(correction)
        retained_parts.append(retained)
        seed_parts.append(torch.full_like(state, seed))
        group_parts.append(group)
        candidate_parts.append(candidate)
        dataset_reports.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "seed": seed,
                "attempt_stage": (
                    "canonical_first_attempt"
                    if args.first_attempt_only
                    else "first_retry"
                ),
                "selected_attempt_samples": int(
                    selected_attempt.sum().item()
                ),
                "states": int(torch.unique(state).numel()),
                "retained_candidates": int(retained.sum().item()),
            }
        )

    context = torch.cat(context_parts)
    correction = torch.cat(correction_parts)
    retained = torch.cat(retained_parts)
    sample_seed = torch.cat(seed_parts)
    group = torch.cat(group_parts)
    candidate_index = torch.cat(candidate_parts)
    normalized_correction = torch.cat(
        (
            correction[:, :3] / position_cap,
            correction[:, 3:] / orientation_cap,
        ),
        dim=-1,
    )
    features = torch.cat((context, normalized_correction), dim=-1)
    if features.shape[-1] != ReceiverCandidateValue.input_dim:
        raise ValueError("receiver candidate value feature shape drifted")

    candidates = torch.empty((65, 6), dtype=torch.float32)
    for candidate in range(65):
        values = correction[
            (candidate_index == candidate)
            & (sample_seed == args.candidate_seed)
        ]
        if values.shape[0] == 0:
            raise ValueError(f"candidate {candidate} has no samples")
        mean = values.mean(dim=0)
        if float((values - mean).abs().max().item()) > 1.0e-6:
            raise ValueError(
                f"candidate {candidate} correction is not deterministic"
            )
        candidates[candidate] = mean

    if args.validation_seed is not None:
        validation_mask = sample_seed == args.validation_seed
        training_mask = ~validation_mask
    else:
        groups = torch.unique(group)
        order = groups[
            torch.randperm(
                groups.numel(),
                generator=torch.Generator().manual_seed(args.seed),
            )
        ]
        validation_groups = order[
            : max(1, int(groups.numel() * args.validation_fraction))
        ]
        validation_mask = torch.isin(group, validation_groups)
        training_mask = ~validation_mask
    if not bool(training_mask.any()) or not bool(validation_mask.any()):
        raise ValueError("receiver candidate split is empty")

    feature_mean = features[training_mask].mean(dim=0)
    feature_std = features[training_mask].std(dim=0).clamp_min(1.0e-6)
    normalized = (features - feature_mean) / feature_std
    positive = retained[training_mask].float().sum().clamp_min(1.0)
    negative = (
        training_mask.sum() - retained[training_mask].sum()
    ).float().clamp_min(1.0)
    positive_weight = negative / positive

    model = ReceiverCandidateValue()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_loss = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        training_indices = torch.nonzero(
            training_mask,
            as_tuple=False,
        ).squeeze(-1)
        order = training_indices[
            torch.randperm(
                training_indices.numel(),
                generator=torch.Generator().manual_seed(args.seed + epoch),
            )
        ]
        model.train()
        for start in range(0, order.numel(), args.batch_size):
            indices = order[start : start + args.batch_size]
            loss = functional.binary_cross_entropy_with_logits(
                model(normalized[indices]),
                retained[indices].float(),
                pos_weight=positive_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                functional.binary_cross_entropy_with_logits(
                    model(normalized[validation_mask]),
                    retained[validation_mask].float(),
                    pos_weight=positive_weight,
                ).item()
            )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(normalized)
    training_metrics = _ranking_metrics(
        logits,
        retained,
        group,
        training_mask,
    )
    validation_metrics = _ranking_metrics(
        logits,
        retained,
        group,
        validation_mask,
    )

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": "dranmar-receiver-candidate-value-1.0",
        "receiver_candidate_value": model.state_dict(),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "candidate_corrections": candidates,
        "candidate_seed": args.candidate_seed,
        "base_checkpoint_sha256": next(iter(base_hashes)),
        "pickup_recovery_checkpoint_sha256": next(iter(pickup_hashes)),
        "position_cap_m": position_cap,
        "orientation_cap_rad": orientation_cap,
        "training": {
            "algorithm": "candidate_retained_value_ranking",
            "attempt_stage": (
                "canonical_first_attempt"
                if args.first_attempt_only
                else "first_retry"
            ),
            "seed": args.seed,
            "validation_seed": args.validation_seed,
            "epochs": args.epochs,
            "samples": int(features.shape[0]),
            "states": int(torch.unique(group).numel()),
            "retained_candidates": int(retained.sum().item()),
            "best_validation_loss": best_loss,
            "training_ranking": training_metrics,
            "validation_ranking": validation_metrics,
        },
        "datasets": dataset_reports,
    }
    torch.save(checkpoint, output)
    report = {
        key: value
        for key, value in checkpoint.items()
        if key
        not in {
            "receiver_candidate_value",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
