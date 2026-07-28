#!/usr/bin/env python3
"""Train the isolated pickup-retry head from physics-generated corrections."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Behavior-clone successful post-reset grasp corrections"
    )
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-6)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=104729)
    return parser


def main(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    if not 0.05 <= args.validation_fraction <= 0.5:
        raise ValueError("validation fraction must be in [0.05, 0.5]")

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
        PickupRecoveryHead,
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
        if payload.get("schema_version") != (
            "dranmar-pickup-recovery-dataset-1.0"
        ):
            raise ValueError(f"unsupported recovery dataset: {path}")
        if payload["context"].shape[-1] != PickupRecoveryHead.input_dim:
            raise ValueError(f"recovery context shape drifted in {path}")
        if payload["correction"].shape[-1] != PickupRecoveryHead.output_dim:
            raise ValueError(f"recovery correction shape drifted in {path}")

    base_hashes = {payload["base_checkpoint_sha256"] for payload in payloads}
    position_caps = {float(payload["position_cap_m"]) for payload in payloads}
    orientation_caps = {
        float(payload["orientation_cap_rad"]) for payload in payloads
    }
    if len(base_hashes) != 1:
        raise ValueError("recovery datasets do not share one base checkpoint")
    if len(position_caps) != 1 or len(orientation_caps) != 1:
        raise ValueError("recovery datasets do not share correction caps")
    position_cap = position_caps.pop()
    orientation_cap = orientation_caps.pop()

    context = torch.cat([payload["context"].float() for payload in payloads])
    correction = torch.cat(
        [payload["correction"].float() for payload in payloads]
    )
    recovered = torch.cat(
        [payload["recovered_custody"].bool() for payload in payloads]
    )
    lifted = torch.cat([payload["lifted"].bool() for payload in payloads])
    successful_demonstration = recovered & lifted
    positive_context = context[successful_demonstration]
    positive_correction = correction[successful_demonstration]
    if positive_context.shape[0] < 32:
        raise ValueError(
            "at least 32 recovered-and-lifted demonstrations are required"
        )
    normalized_target = torch.cat(
        (
            positive_correction[:, :3] / position_cap,
            positive_correction[:, 3:] / orientation_cap,
        ),
        dim=-1,
    ).clamp(-1.0, 1.0)

    permutation = torch.randperm(
        positive_context.shape[0],
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_count = max(
        1,
        int(positive_context.shape[0] * args.validation_fraction),
    )
    validation_indices = permutation[:validation_count]
    training_indices = permutation[validation_count:]
    training_context = positive_context[training_indices]
    training_target = normalized_target[training_indices]
    validation_context = positive_context[validation_indices]
    validation_target = normalized_target[validation_indices]

    head = PickupRecoveryHead()
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_validation_loss = float("inf")
    best_state = None
    final_training_loss = None
    for epoch in range(args.epochs):
        epoch_generator = torch.Generator().manual_seed(args.seed + epoch)
        order = torch.randperm(
            training_context.shape[0],
            generator=epoch_generator,
        )
        head.train()
        epoch_losses = []
        for start in range(0, order.numel(), args.batch_size):
            indices = order[start : start + args.batch_size]
            predicted = head(training_context[indices])
            loss = functional.smooth_l1_loss(
                predicted,
                training_target[indices],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().item()))
        final_training_loss = sum(epoch_losses) / len(epoch_losses)
        head.eval()
        with torch.no_grad():
            validation_loss = float(
                functional.smooth_l1_loss(
                    head(validation_context),
                    validation_target,
                ).item()
            )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                key: value.detach().clone()
                for key, value in head.state_dict().items()
            }

    assert best_state is not None
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        predicted = head(validation_context)
        position_error_m = (
            (predicted[:, :3] - validation_target[:, :3])
            * position_cap
        ).norm(dim=-1)
        orientation_error_rad = (
            (predicted[:, 3:] - validation_target[:, 3:])
            * orientation_cap
        ).norm(dim=-1)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": "dranmar-pickup-recovery-head-1.0",
        "pickup_recovery_head": head.state_dict(),
        "base_checkpoint_sha256": next(iter(base_hashes)),
        "position_cap_m": position_cap,
        "orientation_cap_rad": orientation_cap,
        "training": {
            "algorithm": "successful_offset_behavior_cloning",
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dataset_samples": int(context.shape[0]),
            "successful_demonstrations": int(
                successful_demonstration.sum().item()
            ),
            "training_demonstrations": int(training_context.shape[0]),
            "validation_demonstrations": int(validation_context.shape[0]),
            "final_training_loss": final_training_loss,
            "best_validation_loss": best_validation_loss,
            "validation_position_error_mean_m": float(
                position_error_m.mean().item()
            ),
            "validation_orientation_error_mean_rad": float(
                orientation_error_rad.mean().item()
            ),
        },
        "datasets": [
            {
                "path": str(path),
                "sha256": _sha256(path),
                "seed": int(payload["seed"]),
                "samples": int(payload["context"].shape[0]),
            }
            for path, payload in zip(dataset_paths, payloads, strict=True)
        ],
    }
    torch.save(checkpoint, output)
    report = {
        key: value
        for key, value in checkpoint.items()
        if key != "pickup_recovery_head"
    }
    report["checkpoint"] = {
        "path": str(output),
        "sha256": _sha256(output),
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
