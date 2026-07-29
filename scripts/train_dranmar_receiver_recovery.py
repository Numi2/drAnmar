#!/usr/bin/env python3
"""Train the isolated receiver-retry head from physics demonstrations."""

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
        description=(
            "Behavior-clone safe receiver-acquisition retry corrections"
        )
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
        ReceiverRecoveryHead,
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
        if payload.get("schema_version") not in {
            "dranmar-receiver-recovery-dataset-1.0",
            "dranmar-receiver-recovery-dataset-1.1",
            "dranmar-receiver-recovery-dataset-1.2",
        }:
            raise ValueError(f"unsupported receiver dataset: {path}")
        samples = payload.get("attempts") or payload
        if samples["context"].shape[-1] != ReceiverRecoveryHead.input_dim:
            raise ValueError(f"receiver context shape drifted in {path}")
        if (
            samples["correction"].shape[-1]
            != ReceiverRecoveryHead.output_dim
        ):
            raise ValueError(f"receiver correction shape drifted in {path}")
        if int(payload["seed"]) not in DEVELOPMENT_SEEDS:
            raise ValueError(
                f"receiver dataset seed is not development-only: {path}"
            )

    base_hashes = {payload["base_checkpoint_sha256"] for payload in payloads}
    pickup_hashes = {
        payload.get("pickup_recovery_checkpoint_sha256")
        for payload in payloads
    }
    position_caps = {float(payload["position_cap_m"]) for payload in payloads}
    orientation_caps = {
        float(payload["orientation_cap_rad"]) for payload in payloads
    }
    if len(base_hashes) != 1:
        raise ValueError("receiver datasets do not share one base checkpoint")
    if len(pickup_hashes) != 1:
        raise ValueError("receiver datasets do not share one frozen pickup head")
    if len(position_caps) != 1 or len(orientation_caps) != 1:
        raise ValueError("receiver datasets do not share correction caps")
    position_cap = position_caps.pop()
    orientation_cap = orientation_caps.pop()

    candidate_groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    total_samples = 0
    successful_candidate_count = 0
    retained_candidate_count = 0
    for payload_index, payload in enumerate(payloads):
        has_attempt_records = payload.get("attempts") is not None
        samples = payload.get("attempts") or payload
        context = samples["context"].float()
        correction = samples["correction"].float()
        safe_acquisition = samples["safe_acquisition"].bool()
        retained = samples.get(
            "retained",
            samples["full_success"],
        ).bool()
        peak_force = samples.get(
            "peak_jaw_force_n",
            torch.full((context.shape[0], 2), float("inf")),
        ).float()
        steps = samples.get(
            "steps_to_acquisition",
            torch.full((context.shape[0],), 2**31 - 1),
        ).long()
        state_index = samples.get(
            "state_index",
            torch.arange(context.shape[0]),
        ).long()
        retry_count = (
            samples["retry_count"].long()
            if has_attempt_records
            else torch.ones(context.shape[0], dtype=torch.long)
        )
        candidate_index = samples.get(
            "candidate_index",
            torch.zeros(context.shape[0], dtype=torch.long),
        ).long()
        total_samples += int(context.shape[0])
        successful_candidate_count += int(safe_acquisition.sum().item())
        retained_candidate_count += int(
            (safe_acquisition & retained).sum().item()
        )
        sweep_id = payload.get("sweep_id")
        if sweep_id:
            group_prefix = (
                f"{sweep_id}|seed={int(payload['seed'])}|"
                f"num_envs={int(payload['num_envs'])}"
            )
        else:
            group_prefix = f"dataset={payload_index}"
        for sample_index in range(context.shape[0]):
            key = (group_prefix, int(state_index[sample_index].item()))
            candidate_groups.setdefault(key, []).append(
                {
                    "context": context[sample_index],
                    "correction": correction[sample_index],
                    "safe": bool(safe_acquisition[sample_index].item()),
                    "retained": bool(retained[sample_index].item()),
                    "peak_force": float(
                        peak_force[sample_index].amax().item()
                    ),
                    "steps": int(steps[sample_index].item()),
                    "candidate_index": int(
                        candidate_index[sample_index].item()
                    ),
                    "retry_count": int(
                        retry_count[sample_index].item()
                    ),
                }
            )
    demonstration_context = []
    demonstration_correction = []
    retained_demonstrations = 0
    for key, candidates in candidate_groups.items():
        successful = [
            candidate for candidate in candidates if candidate["safe"]
        ]
        if not successful:
            continue
        chosen = min(
            successful,
            key=lambda candidate: (
                float(
                    torch.cat(
                        (
                            candidate["correction"][:3] / position_cap,
                            candidate["correction"][3:]
                            / orientation_cap,
                        )
                    )
                    .norm()
                    .item()
                ),
                candidate["peak_force"],
                candidate["steps"],
                not candidate["retained"],
            ),
        )
        demonstration_context.append(chosen["context"])
        demonstration_correction.append(chosen["correction"])
        retained_demonstrations += int(chosen["retained"])

    if len(demonstration_context) < 32:
        raise ValueError(
            "at least 32 distinct safe receiver-recovery states are required"
        )
    context = torch.stack(demonstration_context)
    physical_target = torch.stack(demonstration_correction)
    normalized_target = torch.cat(
        (
            physical_target[:, :3] / position_cap,
            physical_target[:, 3:] / orientation_cap,
        ),
        dim=-1,
    ).clamp(-1.0, 1.0)

    permutation = torch.randperm(
        context.shape[0],
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_count = max(
        1,
        int(context.shape[0] * args.validation_fraction),
    )
    validation_indices = permutation[:validation_count]
    training_indices = permutation[validation_count:]
    training_context = context[training_indices]
    training_target = normalized_target[training_indices]
    validation_context = context[validation_indices]
    validation_target = normalized_target[validation_indices]

    head = ReceiverRecoveryHead()
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_validation_loss = float("inf")
    best_state = None
    final_training_loss = None
    for epoch in range(args.epochs):
        order = torch.randperm(
            training_context.shape[0],
            generator=torch.Generator().manual_seed(args.seed + epoch),
        )
        head.train()
        epoch_losses = []
        for start in range(0, order.numel(), args.batch_size):
            indices = order[start : start + args.batch_size]
            prediction = head(training_context[indices])
            loss = functional.smooth_l1_loss(
                prediction,
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
        prediction = head(validation_context)
        position_error = (
            (prediction[:, :3] - validation_target[:, :3])
            * position_cap
        ).norm(dim=-1)
        orientation_error = (
            (prediction[:, 3:] - validation_target[:, 3:])
            * orientation_cap
        ).norm(dim=-1)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": "dranmar-receiver-recovery-head-1.0",
        "receiver_recovery_head": head.state_dict(),
        "base_checkpoint_sha256": next(iter(base_hashes)),
        "pickup_recovery_checkpoint_sha256": next(iter(pickup_hashes)),
        "position_cap_m": position_cap,
        "orientation_cap_rad": orientation_cap,
        "training": {
            "algorithm": "successful_offset_behavior_cloning",
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dataset_samples": total_samples,
            "successful_candidates": successful_candidate_count,
            "retained_successful_candidates": retained_candidate_count,
            "successful_demonstrations": int(context.shape[0]),
            "retained_demonstrations": retained_demonstrations,
            "paired_failed_states": len(candidate_groups),
            "training_demonstrations": int(training_context.shape[0]),
            "validation_demonstrations": int(validation_context.shape[0]),
            "final_training_loss": final_training_loss,
            "best_validation_loss": best_validation_loss,
            "validation_position_error_mean_m": float(
                position_error.mean().item()
            ),
            "validation_orientation_error_mean_rad": float(
                orientation_error.mean().item()
            ),
        },
        "datasets": [
            {
                "path": str(path),
                "sha256": _sha256(path),
                "seed": int(payload["seed"]),
                "samples": int(
                    (payload.get("attempts") or payload)[
                        "context"
                    ].shape[0]
                ),
            }
            for path, payload in zip(dataset_paths, payloads, strict=True)
        ],
    }
    torch.save(checkpoint, output)
    report = {
        key: value
        for key, value in checkpoint.items()
        if key != "receiver_recovery_head"
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
