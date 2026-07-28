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


DEVELOPMENT_SEEDS = {104729, 130363, 196613}


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
    parser.add_argument("--require_collection_gate", action="store_true")
    parser.add_argument("--minimum_per_cohort", type=int, default=1000)
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
        if payload.get("schema_version") not in {
            "dranmar-pickup-recovery-dataset-1.0",
            "dranmar-pickup-recovery-dataset-1.1",
            "dranmar-pickup-recovery-dataset-1.2",
        }:
            raise ValueError(f"unsupported recovery dataset: {path}")
        samples = payload.get("attempts") or payload
        if samples["context"].shape[-1] != PickupRecoveryHead.input_dim:
            raise ValueError(f"recovery context shape drifted in {path}")
        if (
            samples["correction"].shape[-1]
            != PickupRecoveryHead.output_dim
        ):
            raise ValueError(f"recovery correction shape drifted in {path}")
        if int(payload["seed"]) not in DEVELOPMENT_SEEDS:
            raise ValueError(
                f"dataset seed is not development-only: {path}"
            )

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

    total_samples = sum(
        int((payload.get("attempts") or payload)["context"].shape[0])
        for payload in payloads
    )
    safe_lift_candidate_count = 0
    end_to_end_successful_candidate_count = 0
    positive_context_parts = []
    positive_correction_parts = []
    candidate_groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    strict_replay_groups: set[tuple[str, int]] = set()
    collection_cohorts = {
        f"{failure}|retry_{retry}": 0
        for failure in (
            "never_bilateral",
            "lost_jaw_1",
            "lost_jaw_2",
            "lost_both",
        )
        for retry in ("1", "2", "3_plus")
    }
    for payload_index, payload in enumerate(payloads):
        has_attempt_records = payload.get("attempts") is not None
        samples = payload.get("attempts") or payload
        payload_context = samples["context"].float()
        payload_correction = samples["correction"].float()
        safe_lift = samples.get(
            "safe_lift",
            samples["recovered_custody"].bool()
            & samples["lifted"].bool(),
        ).bool()
        full_success = samples["full_success"].bool()
        safe_lift_candidate_count += int(safe_lift.sum().item())
        end_to_end_successful_candidate_count += int(
            full_success.sum().item()
        )
        state_index = samples.get(
            "state_index",
            torch.arange(payload_context.shape[0]),
        ).long()
        candidate_index = samples.get(
            "candidate_index",
            torch.zeros(payload_context.shape[0], dtype=torch.long),
        ).long()
        peak_force = samples.get(
            "peak_jaw_force_n",
            torch.full((payload_context.shape[0], 2), float("inf")),
        ).float()
        steps_to_lift = samples.get(
            "steps_to_lift",
            torch.full((payload_context.shape[0],), 2**31 - 1),
        ).long()
        retry_count = (
            samples["retry_count"].long()
            if has_attempt_records
            else torch.ones(
                payload_context.shape[0],
                dtype=torch.long,
            )
        )
        sweep_id = payload.get("sweep_id")
        if sweep_id:
            group_prefix = (
                f"{sweep_id}|seed={int(payload['seed'])}|"
                f"num_envs={int(payload['num_envs'])}"
            )
        else:
            group_prefix = f"dataset={payload_index}"
        for sample_index in range(payload_context.shape[0]):
            key = (
                group_prefix,
                int(state_index[sample_index].item()),
            )
            candidate_groups.setdefault(key, []).append(
                {
                    "context": payload_context[sample_index],
                    "correction": payload_correction[sample_index],
                    "safe_lift": bool(safe_lift[sample_index].item()),
                    "full_success": bool(
                        full_success[sample_index].item()
                    ),
                    "peak_force": float(
                        peak_force[sample_index].amax().item()
                    ),
                    "steps_to_lift": int(
                        steps_to_lift[sample_index].item()
                    ),
                    "candidate_index": int(
                        candidate_index[sample_index].item()
                    ),
                    "retry_count": int(
                        retry_count[sample_index].item()
                    ),
                }
            )
            if sweep_id:
                strict_replay_groups.add(key)
    collection_state_count = 0
    for key, candidates in candidate_groups.items():
        if key in strict_replay_groups:
            reference = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate["candidate_index"] == 0
                ),
                None,
            )
            if reference is None:
                continue
        else:
            reference = candidates[0]
        context = reference["context"]
        loss_flags = context[18:20] > 0.5
        ever_bilateral = bool((context[20] > 0.5).item())
        if not ever_bilateral:
            failure_cohort = "never_bilateral"
        elif bool(loss_flags[0].item()) and bool(loss_flags[1].item()):
            failure_cohort = "lost_both"
        elif bool(loss_flags[0].item()):
            failure_cohort = "lost_jaw_1"
        elif bool(loss_flags[1].item()):
            failure_cohort = "lost_jaw_2"
        else:
            failure_cohort = "lost_both"
        retry_value = int(reference["retry_count"])
        retry_cohort = (
            "1"
            if retry_value <= 1
            else "2"
            if retry_value == 2
            else "3_plus"
        )
        collection_cohorts[
            f"{failure_cohort}|retry_{retry_cohort}"
        ] += 1
        collection_state_count += 1
    collection_gate_passed = (
        collection_state_count >= 12_000
        and all(
            count >= args.minimum_per_cohort
            for count in collection_cohorts.values()
        )
    )
    if args.require_collection_gate and not collection_gate_passed:
        raise ValueError(
            "pickup recovery collection gate failed: "
            f"unique_states={collection_state_count}, "
            f"raw_samples={total_samples}, cohorts={collection_cohorts}"
        )
    maximum_replay_context_spread = 0.0
    maximum_observed_replay_context_spread = 0.0
    excluded_context_drift_candidates = 0
    replay_groups_without_canonical_reference = 0
    for key, candidates in candidate_groups.items():
        candidate_contexts = torch.stack(
            [candidate["context"] for candidate in candidates]
        )
        if key in strict_replay_groups:
            canonical = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate["candidate_index"] == 0
                ),
                None,
            )
            if canonical is None:
                replay_groups_without_canonical_reference += 1
                continue
            reference = canonical["context"]
            per_candidate_spread = (
                candidate_contexts - reference
            ).abs().amax(dim=-1)
            maximum_observed_replay_context_spread = max(
                maximum_observed_replay_context_spread,
                float(per_candidate_spread.max().item()),
            )
            exact = per_candidate_spread <= 1.0e-5
            excluded_context_drift_candidates += int(
                (~exact).sum().item()
            )
            candidates = [
                candidate
                for candidate, keep in zip(
                    candidates,
                    exact.tolist(),
                    strict=True,
                )
                if keep
            ]
            if not candidates:
                continue
            exact_contexts = torch.stack(
                [candidate["context"] for candidate in candidates]
            )
            exact_spread = float(
                (exact_contexts - exact_contexts[:1])
                .abs()
                .max()
                .item()
            )
            maximum_replay_context_spread = max(
                maximum_replay_context_spread,
                exact_spread,
            )
        successful = [
            candidate
            for candidate in candidates
            if candidate["full_success"]
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
                candidate["steps_to_lift"],
            ),
        )
        positive_context_parts.append(chosen["context"])
        positive_correction_parts.append(chosen["correction"])
    if positive_context_parts:
        positive_context = torch.stack(positive_context_parts)
        positive_correction = torch.stack(positive_correction_parts)
    else:
        positive_context = torch.empty(0, PickupRecoveryHead.input_dim)
        positive_correction = torch.empty(0, PickupRecoveryHead.output_dim)
    if positive_context.shape[0] < 32:
        raise ValueError(
            "at least 32 distinct end-to-end successful recovery states "
            "are required"
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
            "algorithm": "end_to_end_successful_offset_behavior_cloning",
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dataset_samples": total_samples,
            "safe_lift_candidates": safe_lift_candidate_count,
            "end_to_end_successful_candidates": (
                end_to_end_successful_candidate_count
            ),
            "successful_demonstrations": int(
                positive_context.shape[0]
            ),
            "paired_failed_states": len(candidate_groups),
            "maximum_replay_context_spread": (
                maximum_replay_context_spread
            ),
            "maximum_observed_replay_context_spread": (
                maximum_observed_replay_context_spread
            ),
            "excluded_context_drift_candidates": (
                excluded_context_drift_candidates
            ),
            "replay_groups_without_canonical_reference": (
                replay_groups_without_canonical_reference
            ),
            "collection_gate": {
                "passed": collection_gate_passed,
                "unique_states": collection_state_count,
                "raw_candidate_samples": total_samples,
                "minimum_total_unique_states": 12_000,
                "minimum_per_cohort": args.minimum_per_cohort,
                "cohorts": collection_cohorts,
            },
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
