#!/usr/bin/env python3
"""Combine seed-identical retry-only runs for all frozen candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch


DATASET_SCHEMA = "dranmar-receiver-recovery-dataset-1.2"
CANDIDATE_SCHEMA = "dranmar-receiver-candidate-value-1.0"
PORTFOLIO_SCHEMA = "dranmar-receiver-retry-portfolio-1.0"
SEARCH_MODE = "retry_common16_candidate"
DROP_NAMES = {"needle_dropped_after_pickup", "object_dropping"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _greedy_order(
    retained: torch.Tensor,
    acquired: torch.Tensor,
    dropped: torch.Tensor,
) -> tuple[list[int], list[dict[str, int]]]:
    covered = torch.zeros(retained.shape[0], dtype=torch.bool)
    remaining = list(range(16))
    order: list[int] = []
    trace: list[dict[str, int]] = []
    while remaining:
        choices: list[tuple[tuple[int, int, int, int, int], int]] = []
        for candidate in remaining:
            marginal = ~covered & retained[:, candidate]
            choices.append(
                (
                    (
                        int(marginal.sum()),
                        int(retained[:, candidate].sum()),
                        int(acquired[:, candidate].sum()),
                        -int(dropped[:, candidate].sum()),
                        -candidate,
                    ),
                    candidate,
                )
            )
        _, selected = max(choices)
        marginal = ~covered & retained[:, selected]
        covered |= retained[:, selected]
        order.append(selected)
        remaining.remove(selected)
        trace.append(
            {
                "portfolio_rank": len(order) - 1,
                "candidate_index": selected,
                "marginal_retained_successes": int(marginal.sum()),
                "cumulative_oracle_successes": int(covered.sum()),
                "paired_retained_successes": int(
                    retained[:, selected].sum()
                ),
                "paired_acquisitions": int(acquired[:, selected].sum()),
                "paired_drops": int(dropped[:, selected].sum()),
            }
        )
    return order, trace


def _main(args: argparse.Namespace) -> int:
    base_path = Path(args.base_checkpoint).expanduser().resolve()
    candidate_path = Path(
        args.candidate_checkpoint
    ).expanduser().resolve()
    base_hash = _sha256(base_path)
    candidate_hash = _sha256(candidate_path)
    candidate_payload = torch.load(
        candidate_path,
        map_location="cpu",
        weights_only=False,
    )
    if (
        candidate_payload.get("schema_version") != CANDIDATE_SCHEMA
        or candidate_payload.get("base_checkpoint_sha256") != base_hash
        or candidate_payload["candidate_corrections"].shape != (16, 6)
    ):
        raise ValueError("candidate checkpoint is not the frozen common-16")
    corrections = candidate_payload["candidate_corrections"].float()
    rows_by_candidate: dict[int, dict[int, dict[str, object]]] = {}
    dataset_reports: list[dict[str, object]] = []
    expected_seed: int | None = None
    for encoded_path in args.dataset:
        path = Path(encoded_path).expanduser().resolve()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        attempts = payload.get("attempts")
        if not isinstance(attempts, dict):
            raise ValueError(f"dataset has no attempt records: {path}")
        candidate = payload.get("retry_candidate_index")
        if candidate is None:
            retry_one = attempts["retry_count"].long() == 1
            observed = torch.unique(
                attempts["candidate_index"].long()[retry_one]
            )
            if observed.numel() == 1:
                candidate = int(observed.item())
        if (
            payload.get("schema_version") != DATASET_SCHEMA
            or payload.get("search_mode") != SEARCH_MODE
            or payload.get("base_checkpoint_sha256") != base_hash
            or payload.get("receiver_candidate_checkpoint_sha256")
            != candidate_hash
            or not isinstance(candidate, int)
            or not 0 <= candidate < 16
        ):
            raise ValueError(f"incompatible retry candidate dataset: {path}")
        if candidate in rows_by_candidate:
            raise ValueError(f"duplicate candidate dataset: {candidate}")
        seed = int(payload["seed"])
        if expected_seed is None:
            expected_seed = seed
        elif seed != expected_seed:
            raise ValueError("retry candidate datasets must share one seed")
        retry = attempts["retry_count"].long()
        mask = (
            (retry == 1)
            & attempts["giver_bilateral_at_activation"].bool()
            & (attempts["candidate_index"].long() == candidate)
        )
        termination_names = list(payload["termination_names"])
        drop_indices = [
            termination_names.index(name)
            for name in DROP_NAMES
            if name in termination_names
        ]
        environment = attempts["environment_index"].long()[mask]
        terminal = attempts["termination_flags"].bool()[mask]
        dropped = (
            terminal[:, drop_indices].any(dim=-1)
            if drop_indices
            else torch.zeros(environment.shape[0], dtype=torch.bool)
        )
        candidate_rows: dict[int, dict[str, object]] = {}
        for index, env_id in enumerate(environment.tolist()):
            if env_id in candidate_rows:
                raise ValueError(
                    f"candidate {candidate} retried environment {env_id} "
                    "more than once at retry index one"
                )
            correction = attempts["correction"][mask][index].float()
            if not torch.allclose(
                correction,
                corrections[candidate],
                atol=1.0e-7,
                rtol=0.0,
            ):
                raise ValueError(
                    f"candidate {candidate} correction drifted"
                )
            candidate_rows[env_id] = {
                "context": attempts["context"][mask][index].float(),
                "retained": bool(
                    attempts["full_success"][mask][index]
                ),
                "acquired": bool(attempts["acquired"][mask][index]),
                "dropped": bool(dropped[index]),
            }
        rows_by_candidate[candidate] = candidate_rows
        dataset_reports.append(
            {
                "candidate_index": candidate,
                "path": str(path),
                "sha256": _sha256(path),
                "retry_states": len(candidate_rows),
            }
        )
    if sorted(rows_by_candidate) != list(range(16)):
        raise ValueError(
            "analysis requires exactly one retry dataset for candidates 0..15"
        )
    complete_environments = sorted(
        set.intersection(
            *(
                set(rows_by_candidate[candidate])
                for candidate in range(16)
            )
        )
    )
    if len(complete_environments) < args.minimum_complete_states:
        raise ValueError(
            f"expected at least {args.minimum_complete_states} complete "
            f"paired retry states, found {len(complete_environments)}"
        )
    retained = torch.tensor(
        [
            [
                rows_by_candidate[candidate][environment]["retained"]
                for candidate in range(16)
            ]
            for environment in complete_environments
        ],
        dtype=torch.bool,
    )
    acquired = torch.tensor(
        [
            [
                rows_by_candidate[candidate][environment]["acquired"]
                for candidate in range(16)
            ]
            for environment in complete_environments
        ],
        dtype=torch.bool,
    )
    dropped = torch.tensor(
        [
            [
                rows_by_candidate[candidate][environment]["dropped"]
                for candidate in range(16)
            ]
            for environment in complete_environments
        ],
        dtype=torch.bool,
    )
    context_drift = []
    for environment in complete_environments:
        reference = rows_by_candidate[0][environment]["context"]
        context_drift.append(
            max(
                float(
                    (
                        rows_by_candidate[candidate][environment]["context"]
                        - reference
                    )
                    .abs()
                    .max()
                )
                for candidate in range(1, 16)
            )
        )
    order, trace = _greedy_order(retained, acquired, dropped)
    oracle = retained.any(dim=-1)
    report: dict[str, object] = {
        "schema_version": (
            "dranmar-receiver-retry-candidate-analysis-1.0"
        ),
        "source_revision": _revision(),
        "seed": expected_seed,
        "base_checkpoint_sha256": base_hash,
        "receiver_candidate_checkpoint_sha256": candidate_hash,
        "complete_paired_retry_states": len(complete_environments),
        "complete_environment_indices": complete_environments,
        "oracle_retained_successes": int(oracle.sum()),
        "oracle_retained_rate": float(oracle.float().mean()),
        "maximum_context_drift": max(context_drift),
        "greedy_order": order,
        "greedy_trace": trace,
        "candidate_results": [
            {
                "candidate_index": candidate,
                "all_retry_states": len(rows_by_candidate[candidate]),
                "all_retained_successes": sum(
                    int(row["retained"])
                    for row in rows_by_candidate[candidate].values()
                ),
                "all_acquisitions": sum(
                    int(row["acquired"])
                    for row in rows_by_candidate[candidate].values()
                ),
                "all_drops": sum(
                    int(row["dropped"])
                    for row in rows_by_candidate[candidate].values()
                ),
                "paired_retained_successes": int(
                    retained[:, candidate].sum()
                ),
                "paired_acquisitions": int(
                    acquired[:, candidate].sum()
                ),
                "paired_drops": int(dropped[:, candidate].sum()),
            }
            for candidate in range(16)
        ],
        "datasets": sorted(
            dataset_reports,
            key=lambda item: item["candidate_index"],
        ),
    }
    if args.portfolio_output:
        output = Path(args.portfolio_output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        portfolio = {
            "schema_version": PORTFOLIO_SCHEMA,
            "source_revision": _revision(),
            "base_checkpoint_sha256": base_hash,
            "receiver_candidate_checkpoint_sha256": candidate_hash,
            "candidate_corrections": corrections,
            "portfolio_orders": torch.tensor(
                [order, order, order],
                dtype=torch.long,
            ),
            "portfolio_names": [
                "seed104729_retry_evidence",
                "seed104729_retry_evidence",
                "seed104729_retry_evidence",
            ],
            "force_imbalance_threshold": 0.005,
            "routing_signal_observed_in_paired_contexts": False,
            "routing_status": "single_seed_retry_pilot_global_order",
            "selection": {
                "algorithm": (
                    "retry_only_greedy_retained_maximum_coverage"
                ),
                "complete_paired_states": len(complete_environments),
                "oracle_retained_successes": int(oracle.sum()),
                "first_three": order[:3],
                "trace": trace,
            },
            "datasets": report["datasets"],
        }
        torch.save(portfolio, output)
        report["portfolio_checkpoint"] = {
            "path": str(output),
            "sha256": _sha256(output),
        }
    output_report = Path(args.output_report).expanduser().resolve()
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine all-16 seed-identical retry candidate outcomes"
    )
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--candidate_checkpoint", required=True)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--portfolio_output")
    parser.add_argument("--minimum_complete_states", type=int, default=1)
    return parser


if __name__ == "__main__":
    raise SystemExit(_main(_parser().parse_args()))
