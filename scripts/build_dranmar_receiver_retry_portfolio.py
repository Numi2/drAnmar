#!/usr/bin/env python3
"""Build a deterministic receiver-retry portfolio from paired outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch


DEVELOPMENT_SEEDS = (104729, 130363, 196613)
DATASET_SCHEMA_VERSION = "dranmar-receiver-recovery-dataset-1.2"
PORTFOLIO_SCHEMA_VERSION = "dranmar-receiver-retry-portfolio-1.0"
CANDIDATE_SCHEMA_VERSION = "dranmar-receiver-candidate-value-1.0"


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


def _load_paired_states(
    paths: list[Path],
    *,
    base_hash: str,
    candidates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    contexts: list[torch.Tensor] = []
    outcomes: list[torch.Tensor] = []
    seeds: list[int] = []
    reports: list[dict[str, object]] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        attempts = payload.get("attempts") or payload
        seed = int(payload["seed"])
        if (
            payload.get("schema_version") != DATASET_SCHEMA_VERSION
            or seed not in DEVELOPMENT_SEEDS
            or payload.get("base_checkpoint_sha256") != base_hash
            or int(payload.get("sweep_replicas", 0)) != 16
            or int(payload.get("sobol_seed", -1)) != 104730
        ):
            raise ValueError(f"incompatible common-16 dataset: {path}")
        first = attempts["retry_count"].long() == 0
        state_index = attempts["state_index"].long()[first]
        candidate_index = attempts["candidate_index"].long()[first]
        context = attempts["context"].float()[first]
        correction = attempts["correction"].float()[first]
        success = attempts["full_success"].bool()[first]
        state_count = 0
        for state in torch.unique(state_index).tolist():
            mask = state_index == state
            order = candidate_index[mask].argsort()
            indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)[order]
            if not torch.equal(candidate_index[indices], torch.arange(16)):
                continue
            if not torch.allclose(
                correction[indices],
                candidates,
                atol=1.0e-7,
                rtol=0.0,
            ):
                raise ValueError(f"candidate correction drift: {path}")
            state_success = success[indices]
            if bool(state_success.all()):
                continue
            # Candidate zero is the correction-free gate context. The PhysX
            # replicas provide paired outcomes, but are not treated as exact
            # per-frame context replicas.
            contexts.append(context[indices][0])
            outcomes.append(state_success)
            seeds.append(seed)
            state_count += 1
        reports.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "seed": seed,
                "failure_heavy_states": state_count,
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


def _coverage_by_seed(
    covered: torch.Tensor,
    seeds: torch.Tensor,
) -> list[float]:
    return [
        float(covered[seeds == seed].float().mean())
        for seed in DEVELOPMENT_SEEDS
    ]


def _greedy_worst_seed_order(
    outcomes: torch.Tensor,
    seeds: torch.Tensor,
    zero_index: int,
) -> tuple[list[int], list[dict[str, object]]]:
    selected: list[int] = []
    covered = torch.zeros(outcomes.shape[0], dtype=torch.bool)
    trace: list[dict[str, object]] = []
    eligible = [
        index for index in range(outcomes.shape[1]) if index != zero_index
    ]
    while eligible:
        ranked: list[
            tuple[
                tuple[float, float, int, int, int],
                int,
                torch.Tensor,
                list[int],
            ]
        ] = []
        for candidate in eligible:
            proposed = covered | outcomes[:, candidate]
            cumulative = _coverage_by_seed(proposed, seeds)
            marginal = [
                int(
                    (
                        ~covered
                        & outcomes[:, candidate]
                        & (seeds == seed)
                    ).sum()
                )
                for seed in DEVELOPMENT_SEEDS
            ]
            # Worst-seed cumulative coverage is the primary objective.
            # Aggregate coverage and a stable candidate-index tie break only
            # decide otherwise-equal choices.
            objective = (
                min(cumulative),
                sum(cumulative),
                min(marginal),
                sum(marginal),
                -candidate,
            )
            ranked.append((objective, candidate, proposed, marginal))
        _, candidate, proposed, marginal = max(ranked, key=lambda row: row[0])
        selected.append(candidate)
        eligible.remove(candidate)
        covered = proposed
        trace.append(
            {
                "attempt": len(selected),
                "candidate_index": candidate,
                "marginal_successes_by_seed": dict(
                    zip(
                        (str(seed) for seed in DEVELOPMENT_SEEDS),
                        marginal,
                        strict=True,
                    )
                ),
                "cumulative_coverage_by_seed": dict(
                    zip(
                        (str(seed) for seed in DEVELOPMENT_SEEDS),
                        _coverage_by_seed(covered, seeds),
                        strict=True,
                    )
                ),
                "cumulative_covered_states": int(covered.sum()),
            }
        )
    selected.append(zero_index)
    return selected, trace


def _write(payload: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    report = {
        key: value
        for key, value in payload.items()
        if key not in {"candidate_corrections", "portfolio_orders"}
    }
    report["portfolio_orders"] = payload["portfolio_orders"].tolist()
    report["checkpoint"] = {
        "path": str(output),
        "sha256": _sha256(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _build(args: argparse.Namespace) -> int:
    base_path = Path(args.base_checkpoint).expanduser().resolve()
    candidate_path = Path(args.candidate_checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    base_hash = _sha256(base_path)
    candidate_payload = torch.load(
        candidate_path,
        map_location="cpu",
        weights_only=False,
    )
    if (
        candidate_payload.get("schema_version") != CANDIDATE_SCHEMA_VERSION
        or candidate_payload.get("base_checkpoint_sha256") != base_hash
        or candidate_payload["candidate_corrections"].shape != (16, 6)
    ):
        raise ValueError("candidate checkpoint is not the frozen common-16")
    candidates = candidate_payload["candidate_corrections"].float()
    contexts, outcomes, seeds, reports = _load_paired_states(
        [Path(value).expanduser().resolve() for value in args.dataset],
        base_hash=base_hash,
        candidates=candidates,
    )
    if contexts.shape[0] != args.expected_states:
        raise ValueError(
            f"expected {args.expected_states} paired states, "
            f"found {contexts.shape[0]}"
        )
    for seed in DEVELOPMENT_SEEDS:
        count = int((seeds == seed).sum())
        if count < args.minimum_states_per_seed:
            raise ValueError(
                f"seed {seed} has only {count} paired states"
            )
    zero_index = int(candidates.square().sum(dim=-1).argmin())
    order, trace = _greedy_worst_seed_order(
        outcomes,
        seeds,
        zero_index,
    )
    loss_flags = contexts[:, 18:20] > 0.5
    force_imbalance = contexts[:, 17].abs()
    routing_signal_observed = bool(
        (~loss_flags.all(dim=-1)).any()
        or (force_imbalance > args.force_imbalance_threshold).any()
    )
    # The 996 canonical contexts all have both jaws missing and zero force.
    # Keep three runtime cohort IDs, but bind them to the one evidence-backed
    # ordering until genuine post-reopen states justify distinct orders.
    portfolio_orders = torch.tensor(
        [order, order, order],
        dtype=torch.long,
    )
    oracle = outcomes.any(dim=-1)
    payload: dict[str, object] = {
        "schema_version": PORTFOLIO_SCHEMA_VERSION,
        "source_revision": _source_revision(),
        "base_checkpoint_sha256": base_hash,
        "receiver_candidate_checkpoint_sha256": _sha256(candidate_path),
        "candidate_corrections": candidates,
        "portfolio_orders": portfolio_orders,
        "portfolio_names": [
            "both_or_balanced",
            "jaw_1_missing_or_weaker",
            "jaw_2_missing_or_weaker",
        ],
        "force_imbalance_threshold": float(
            args.force_imbalance_threshold
        ),
        "routing_signal_observed_in_paired_contexts": (
            routing_signal_observed
        ),
        "routing_status": (
            "three_ids_share_global_order_until_genuine_retry_states"
            if not routing_signal_observed
            else "cohort_specific_rebuild_required"
        ),
        "selection": {
            "algorithm": "greedy_worst_seed_maximum_coverage",
            "states": int(contexts.shape[0]),
            "states_by_seed": {
                str(seed): int((seeds == seed).sum())
                for seed in DEVELOPMENT_SEEDS
            },
            "oracle_success_states": int(oracle.sum()),
            "oracle_coverage_by_seed": dict(
                zip(
                    (str(seed) for seed in DEVELOPMENT_SEEDS),
                    _coverage_by_seed(oracle, seeds),
                    strict=True,
                )
            ),
            "first_three": order[:3],
            "trace": trace,
        },
        "datasets": reports,
    }
    _write(payload, output)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic receiver retry portfolios"
    )
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--candidate_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--expected_states", type=int, default=996)
    parser.add_argument("--minimum_states_per_seed", type=int, default=300)
    parser.add_argument(
        "--force_imbalance_threshold",
        type=float,
        default=0.005,
    )
    return parser


def main() -> int:
    return _build(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
