#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise the pinned tissue-entry backend on the qualification GPU."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET_SOURCE = ROOT / "source/extensions/orbit.surgical.assets"
TASK_SOURCE = ROOT / "source/extensions/orbit.surgical.tasks"
sys.path[:0] = [str(ASSET_SOURCE), str(TASK_SOURCE)]

from orbit.surgical.tasks.surgical.penetration.backend import (  # noqa: E402
    create_tissue_entry_backend,
)
from orbit.surgical.tasks.surgical.penetration.cressim import NeedlePose  # noqa: E402


def _norm(values: tuple[float, float, float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _p95(values: list[float]) -> float:
    return sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1)]


def _run(library: Path, scenes: int, integration_step_s: float) -> dict[str, object]:
    backend = create_tissue_entry_backend(
        scenes,
        integration_step_s=integration_step_s,
        library_path=library,
    )
    quaternion = (0.0, 0.0, 0.0, 1.0)
    outside = NeedlePose((0.0, 0.0, 0.012), quaternion)
    outside_forces: list[float] = []
    contact_forces: list[float] = []
    step_ms: list[float] = []
    trace: list[float] = []
    finite = True
    try:
        for _ in range(3):
            wrenches = backend.step(
                [outside] * scenes,
                [outside] * scenes,
                [False] * scenes,
                dt_s=0.02,
            )
            outside_forces.extend(_norm(wrench.force_n) for wrench in wrenches)

        for index in range(35):
            z_m = 0.006 - index * 0.00012
            tip = NeedlePose(
                (0.0, 0.0, z_m),
                quaternion,
                linear_velocity=(0.0, 0.0, -0.005),
            )
            start = time.perf_counter()
            wrenches = backend.step(
                [tip] * scenes,
                [tip] * scenes,
                [False] * scenes,
                dt_s=0.02,
            )
            step_ms.append((time.perf_counter() - start) * 1000.0)
            magnitudes = [_norm(wrench.force_n) for wrench in wrenches]
            contact_forces.extend(magnitudes)
            trace.append(statistics.fmean(magnitudes))
            finite &= all(
                math.isfinite(value)
                for wrench in wrenches
                for value in (*wrench.force_n, *wrench.torque_nm)
            )

        # Exercise the environment-owned one-shot representation switch.  The
        # adapter receives state; it never decides or emits the puncture event.
        arc = NeedlePose(
            (0.0, 0.0, 0.0),
            quaternion,
            linear_velocity=(0.0, 0.0, -0.005),
        )
        switched = backend.step(
            [outside] * scenes,
            [arc] * scenes,
            [True] * scenes,
            dt_s=0.02,
        )
        switch_finite = all(
            math.isfinite(value)
            for wrench in switched
            for value in (*wrench.force_n, *wrench.torque_nm)
        )
        finite &= switch_finite
        return {
            "integration_step_s": integration_step_s,
            "scenes": scenes,
            "outside_force_n_max": max(outside_forces, default=0.0),
            "contact_force_n_peak": max(contact_forces, default=0.0),
            "physics_step_ms_p95": _p95(step_ms),
            "finite": finite,
            "representation_switch_finite": switch_finite,
            "trace": trace,
            "backend": backend.metadata.__dict__,
        }
    finally:
        backend.close()


def _relative_rmse(reference: list[float], candidate: list[float]) -> float:
    squared = [(left - right) ** 2 for left, right in zip(reference, candidate, strict=True)]
    scale = max(max(reference, default=0.0), max(candidate, default=0.0), 1.0e-12)
    return math.sqrt(statistics.fmean(squared)) / scale


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    at_2ms = _run(args.library.resolve(), args.scenes, 0.002)
    at_1ms = _run(args.library.resolve(), args.scenes, 0.001)
    relative_rmse = _relative_rmse(at_1ms["trace"], at_2ms["trace"])
    qualified = bool(
        at_2ms["finite"]
        and at_1ms["finite"]
        and at_2ms["representation_switch_finite"]
        and at_2ms["contact_force_n_peak"] > 1.0e-9
        and at_2ms["outside_force_n_max"] <= 1.0e-9
        and at_2ms["physics_step_ms_p95"] <= 20.0
        and relative_rmse <= 0.25
    )
    receipt = {
        "schema": "dr.anmar.tissue-entry-backend-probe.v1",
        "qualified": qualified,
        "evidence_level": "simulator_engineering_only",
        "clinical_validation": False,
        "convergence_relative_rmse": relative_rmse,
        "at_2ms": at_2ms,
        "at_1ms": at_1ms,
    }
    receipt["at_2ms"].pop("trace")
    receipt["at_1ms"].pop("trace")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
