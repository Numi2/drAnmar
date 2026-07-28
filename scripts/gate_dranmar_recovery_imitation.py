#!/usr/bin/env python3
"""Gate RL behind real held-out imitation recovery evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


DEVELOPMENT_SEEDS = {104729, 130363, 196613}
QUALIFICATION_SEEDS = {17, 2361, 4099}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Require 70% held-out recovery conversion before PPO"
    )
    parser.add_argument("--stage", choices=("pickup", "receiver"), required=True)
    parser.add_argument("--evidence", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser


def evaluate(stage: str, evidence: list[dict]) -> dict:
    if not evidence:
        raise ValueError("at least one development result is required")
    payload_name = f"{stage}_recovery"
    failure_name = (
        "first_attempt_failures"
        if stage == "pickup"
        else "first_attempt_failures"
    )
    converted_name = (
        "lifted_10mm_after_retry"
        if stage == "pickup"
        else "acquired_after_retry"
    )
    seeds = {int(result["seed"]) for result in evidence}
    if seeds & QUALIFICATION_SEEDS:
        raise ValueError("qualification seeds cannot gate training")
    if not seeds <= DEVELOPMENT_SEEDS:
        raise ValueError(f"unexpected development seeds: {sorted(seeds)}")

    failures = 0
    converted = 0
    mismatch_count = 0
    maximum_difference = 0.0
    head_hashes = set()
    sources = []
    for result in evidence:
        payload = result.get(payload_name)
        if not isinstance(payload, dict) or not payload.get("enabled"):
            raise ValueError(f"{payload_name} is not enabled")
        checkpoint = payload.get("head_checkpoint")
        if not isinstance(checkpoint, dict) or not checkpoint.get("sha256"):
            raise ValueError(f"{payload_name} lacks a frozen head checkpoint")
        head_hashes.add(checkpoint["sha256"])
        failures += int(payload[failure_name])
        converted += int(payload[converted_name])
        mismatch_count += int(payload["first_attempt_action_mismatches"])
        maximum_difference = max(
            maximum_difference,
            float(payload["first_attempt_action_max_abs_difference"]),
        )
        source = result.get("_source")
        if source is not None:
            sources.append(source)
    conversion_rate = converted / failures if failures else 0.0
    gates = {
        "development_seeds_only": True,
        "single_frozen_head": len(head_hashes) == 1,
        "at_least_70_percent_conversion": conversion_rate >= 0.70,
        "first_attempt_action_mismatches_zero": mismatch_count == 0,
        "first_attempt_action_difference_zero": maximum_difference == 0.0,
    }
    return {
        "schema_version": "dranmar-recovery-imitation-gate-1.0",
        "stage": stage,
        "passed": all(gates.values()),
        "gates": gates,
        "development_seeds": sorted(seeds),
        "failed_first_attempts": failures,
        "converted_failed_attempts": converted,
        "conversion_rate": conversion_rate,
        "head_checkpoint_sha256": (
            next(iter(head_hashes)) if len(head_hashes) == 1 else None
        ),
        "first_attempt_action_mismatches": mismatch_count,
        "first_attempt_action_max_abs_difference": maximum_difference,
        "sources": sources,
    }


def main(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    evidence = []
    for value in args.evidence:
        path = Path(value).expanduser().resolve()
        result = json.loads(path.read_text())
        result["_source"] = {
            "path": str(path),
            "sha256": _sha256(path),
        }
        evidence.append(result)
    report = evaluate(args.stage, evidence)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
