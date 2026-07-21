#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Compare physics-next results without treating missing gates as passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def evaluate_gate(name: str, limit: float, metrics: dict[str, Any]) -> dict[str, Any]:
    if name.endswith("_min"):
        shorthand_metric = name.removesuffix("_min")
        direction = "minimum"
        # Some canonical metric names themselves end in _min/_max. Prefer an
        # exact result key, then fall back to the compact gate-key shorthand.
        metric = name if isinstance(metrics.get(name), (int, float)) else shorthand_metric
        observed = metrics.get(metric)
        passed = observed >= limit if isinstance(observed, (int, float)) else None
    elif name.endswith("_max"):
        shorthand_metric = name.removesuffix("_max")
        direction = "maximum"
        metric = name if isinstance(metrics.get(name), (int, float)) else shorthand_metric
        observed = metrics.get(metric)
        passed = observed <= limit if isinstance(observed, (int, float)) else None
    else:
        raise ValueError(f"Gate name does not identify its direction: {name}")
    return {
        "metric": metric,
        "direction": direction,
        "limit": limit,
        "observed": observed,
        "passed": passed,
        "status": "pass" if passed is True else "fail" if passed is False else "not_measured",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Dr.Anmar physics-next benchmark results")
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--benchmark", default="liver-retraction")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    contract_path = REPOSITORY_ROOT / "physics_next/benchmarks" / f"{args.benchmark}.json"
    contract = load(contract_path)
    gates = contract.get("engineering_gates", {})
    comparisons = []
    for result_path in args.results:
        result = load(result_path)
        if result.get("benchmark") != args.benchmark:
            raise ValueError(f"Benchmark mismatch in {result_path}")
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        evaluated = {name: evaluate_gate(name, float(limit), metrics) for name, limit in gates.items()}
        statuses = {gate["status"] for gate in evaluated.values()}
        comparisons.append(
            {
                "path": str(result_path.expanduser().resolve()),
                "backend": result.get("backend"),
                "scope": result.get("benchmark_scope"),
                "gate_status": "fail" if "fail" in statuses else "incomplete" if "not_measured" in statuses else "pass",
                "gates": evaluated,
                "metrics": metrics,
                "clinical_validation": False,
            }
        )
    output = {
        "schema": "dr.anmar.physics-benchmark-comparison.v1",
        "benchmark": args.benchmark,
        "contract": str(contract_path),
        "results": comparisons,
        "promotion_allowed": bool(comparisons) and all(row["gate_status"] == "pass" for row in comparisons),
        "clinical_validation": False,
    }
    rendered = json.dumps(output, indent=2) + "\n"
    if args.output:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
