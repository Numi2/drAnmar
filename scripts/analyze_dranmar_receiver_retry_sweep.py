#!/usr/bin/env python3
"""Analyze paired common-16 outcomes from genuine receiver retries."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch


DATASET_SCHEMA_VERSION = "dranmar-receiver-recovery-dataset-1.2"
PORTFOLIO_SCHEMA_VERSION = "dranmar-receiver-retry-portfolio-1.0"
CANDIDATE_SCHEMA_VERSION = "dranmar-receiver-candidate-value-1.0"
RETRY_SEARCH_MODE = "retry_paired_common16"
DROP_TERMINATIONS = {
    "needle_dropped_after_pickup",
    "object_dropping",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _greedy_coverage_order(
    retained: torch.Tensor,
    acquired: torch.Tensor,
    dropped: torch.Tensor,
) -> tuple[list[int], list[dict[str, int]]]:
    covered = torch.zeros(retained.shape[0], dtype=torch.bool)
    remaining = list(range(retained.shape[1]))
    order: list[int] = []
    trace: list[dict[str, int]] = []
    while remaining:
        ranked: list[tuple[tuple[int, int, int, int, int], int]] = []
        for candidate in remaining:
            new_success = ~covered & retained[:, candidate]
            objective = (
                int(new_success.sum()),
                int(retained[:, candidate].sum()),
                int(acquired[:, candidate].sum()),
                -int(dropped[:, candidate].sum()),
                -candidate,
            )
            ranked.append((objective, candidate))
        _, candidate = max(ranked)
        marginal = ~covered & retained[:, candidate]
        covered |= retained[:, candidate]
        order.append(candidate)
        remaining.remove(candidate)
        trace.append(
            {
                "portfolio_rank": len(order) - 1,
                "candidate_index": candidate,
                "marginal_retained_successes": int(marginal.sum()),
                "cumulative_oracle_successes": int(covered.sum()),
                "candidate_retained_successes": int(
                    retained[:, candidate].sum()
                ),
                "candidate_acquisitions": int(
                    acquired[:, candidate].sum()
                ),
                "candidate_drops": int(dropped[:, candidate].sum()),
            }
        )
    return order, trace


def _analyze(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset).expanduser().resolve()
    base_path = Path(args.base_checkpoint).expanduser().resolve()
    candidate_path = Path(args.candidate_checkpoint).expanduser().resolve()
    report_path = Path(args.output_report).expanduser().resolve()
    payload = torch.load(
        dataset_path,
        map_location="cpu",
        weights_only=False,
    )
    candidate_payload = torch.load(
        candidate_path,
        map_location="cpu",
        weights_only=False,
    )
    base_hash = _sha256(base_path)
    candidate_hash = _sha256(candidate_path)
    if (
        payload.get("schema_version") != DATASET_SCHEMA_VERSION
        or payload.get("search_mode") != RETRY_SEARCH_MODE
        or int(payload.get("sweep_replicas", 0)) != 16
        or payload.get("base_checkpoint_sha256") != base_hash
        or payload.get("receiver_candidate_checkpoint_sha256")
        != candidate_hash
    ):
        raise ValueError("dataset is not a hash-bound retry common-16 sweep")
    if (
        candidate_payload.get("schema_version")
        != CANDIDATE_SCHEMA_VERSION
        or candidate_payload.get("base_checkpoint_sha256") != base_hash
        or candidate_payload["candidate_corrections"].shape != (16, 6)
    ):
        raise ValueError("candidate checkpoint is not the frozen common-16")
    attempts = payload.get("attempts")
    if not isinstance(attempts, dict):
        raise ValueError("retry sweep dataset has no attempt records")
    required = {
        "state_index",
        "candidate_index",
        "context",
        "correction",
        "full_success",
        "acquired",
        "retry_count",
        "termination_flags",
        "giver_bilateral_at_activation",
    }
    missing = sorted(required - attempts.keys())
    if missing:
        raise ValueError(f"retry attempt records missing {missing}")
    retry_mask = (
        attempts["retry_count"].long() == 1
    ) & attempts["giver_bilateral_at_activation"].bool()
    state_index = attempts["state_index"].long()[retry_mask]
    candidate_index = attempts["candidate_index"].long()[retry_mask]
    context = attempts["context"].float()[retry_mask]
    correction = attempts["correction"].float()[retry_mask]
    retained = attempts["full_success"].bool()[retry_mask]
    acquired = attempts["acquired"].bool()[retry_mask]
    terminal = attempts["termination_flags"].bool()[retry_mask]
    termination_names = list(payload["termination_names"])
    drop_indices = [
        termination_names.index(name)
        for name in DROP_TERMINATIONS
        if name in termination_names
    ]
    dropped = (
        terminal[:, drop_indices].any(dim=-1)
        if drop_indices
        else torch.zeros_like(retained)
    )
    candidate_corrections = candidate_payload[
        "candidate_corrections"
    ].float()
    complete_states: list[int] = []
    state_context: list[torch.Tensor] = []
    retained_rows: list[torch.Tensor] = []
    acquired_rows: list[torch.Tensor] = []
    dropped_rows: list[torch.Tensor] = []
    for state in torch.unique(state_index).tolist():
        mask = state_index == state
        ordered = torch.nonzero(mask, as_tuple=False).squeeze(-1)[
            candidate_index[mask].argsort()
        ]
        if not torch.equal(
            candidate_index[ordered],
            torch.arange(16),
        ):
            continue
        if not torch.allclose(
            correction[ordered],
            candidate_corrections,
            atol=1.0e-7,
            rtol=0.0,
        ):
            raise ValueError(
                f"candidate correction drift in paired state {state}"
            )
        complete_states.append(int(state))
        state_context.append(context[ordered][0])
        retained_rows.append(retained[ordered])
        acquired_rows.append(acquired[ordered])
        dropped_rows.append(dropped[ordered])
    if len(complete_states) < args.minimum_complete_states:
        raise ValueError(
            f"expected at least {args.minimum_complete_states} complete "
            f"retry states, found {len(complete_states)}"
        )
    contexts = torch.stack(state_context)
    retained_matrix = torch.stack(retained_rows)
    acquired_matrix = torch.stack(acquired_rows)
    dropped_matrix = torch.stack(dropped_rows)
    order, trace = _greedy_coverage_order(
        retained_matrix,
        acquired_matrix,
        dropped_matrix,
    )
    oracle = retained_matrix.any(dim=-1)
    report: dict[str, object] = {
        "schema_version": "dranmar-receiver-retry-sweep-analysis-1.0",
        "source_revision": _source_revision(),
        "seed": int(payload["seed"]),
        "dataset": {
            "path": str(dataset_path),
            "sha256": _sha256(dataset_path),
        },
        "base_checkpoint_sha256": base_hash,
        "receiver_candidate_checkpoint_sha256": candidate_hash,
        "eligible_retry_activations": int(retry_mask.sum()),
        "complete_paired_states": len(complete_states),
        "complete_state_indices": complete_states,
        "oracle_retained_successes": int(oracle.sum()),
        "oracle_retained_rate": float(oracle.float().mean()),
        "greedy_order": order,
        "greedy_trace": trace,
        "candidate_results": [
            {
                "candidate_index": candidate,
                "retained_successes": int(
                    retained_matrix[:, candidate].sum()
                ),
                "acquisitions": int(
                    acquired_matrix[:, candidate].sum()
                ),
                "drops": int(dropped_matrix[:, candidate].sum()),
            }
            for candidate in range(16)
        ],
        "routing_context": {
            "jaw_1_loss_states": int(
                (contexts[:, 18] > 0.5).sum()
            ),
            "jaw_2_loss_states": int(
                (contexts[:, 19] > 0.5).sum()
            ),
            "nonzero_force_imbalance_states": int(
                (contexts[:, 17].abs() > 0.005).sum()
            ),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    if args.portfolio_output:
        portfolio_path = (
            Path(args.portfolio_output).expanduser().resolve()
        )
        portfolio_path.parent.mkdir(parents=True, exist_ok=True)
        portfolio = {
            "schema_version": PORTFOLIO_SCHEMA_VERSION,
            "source_revision": _source_revision(),
            "base_checkpoint_sha256": base_hash,
            "receiver_candidate_checkpoint_sha256": candidate_hash,
            "candidate_corrections": candidate_corrections,
            "portfolio_orders": torch.tensor(
                [order, order, order],
                dtype=torch.long,
            ),
            "portfolio_names": [
                "retry_pilot_global",
                "retry_pilot_global",
                "retry_pilot_global",
            ],
            "force_imbalance_threshold": 0.005,
            "routing_signal_observed_in_paired_contexts": bool(
                (contexts[:, 18:20] > 0.5).any()
                or (contexts[:, 17].abs() > 0.005).any()
            ),
            "routing_status": "single_seed_retry_pilot_global_order",
            "selection": {
                "algorithm": (
                    "retry_only_greedy_retained_maximum_coverage"
                ),
                "complete_paired_states": len(complete_states),
                "oracle_retained_successes": int(oracle.sum()),
                "first_three": order[:3],
                "trace": trace,
            },
            "datasets": [report["dataset"]],
        }
        torch.save(portfolio, portfolio_path)
        report["portfolio_checkpoint"] = {
            "path": str(portfolio_path),
            "sha256": _sha256(portfolio_path),
        }
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze genuine post-reopen common-16 retry outcomes"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--candidate_checkpoint", required=True)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--portfolio_output")
    parser.add_argument("--minimum_complete_states", type=int, default=1)
    return parser


if __name__ == "__main__":
    raise SystemExit(_analyze(_parser().parse_args()))
