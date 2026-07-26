#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the deterministic, safely bounded dual-PSM action-stream fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / (
    "assets/dr_anmar/multimodal/cosmos_h_dreams_knot_tying_v1/"
    "dranmar_action_stream.json"
)
SAMPLE_HZ = 20
ARM_DIM = 6
ACTION_DIM = 14
NEUTRAL = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0) * 2

# Times, phases, six normalized joint-position inputs per arm, and the
# canonical binary gripper sign. Values are intentionally conservative. This
# is a transport/safety fixture, not a recorded surgical demonstration.
WAYPOINTS = (
    (0.0, "rest", (0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.0), (0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.0)),
    (1.0, "approach", (0.08, -0.04, 0.10, 0.02, 0.00, -0.02, 1.0), (-0.08, -0.04, 0.10, -0.02, 0.00, 0.02, 1.0)),
    (2.0, "align", (0.11, -0.02, 0.14, 0.04, 0.02, -0.04, 1.0), (-0.11, -0.02, 0.14, -0.04, -0.02, 0.04, 1.0)),
    (3.0, "contact", (0.12, 0.00, 0.17, 0.06, 0.03, -0.05, 1.0), (-0.12, 0.00, 0.17, -0.06, -0.03, 0.05, 1.0)),
    (3.5, "grasp", (0.12, 0.00, 0.17, 0.06, 0.03, -0.05, -1.0), (-0.12, 0.00, 0.17, -0.06, -0.03, 0.05, 1.0)),
    (4.5, "manipulate", (0.04, 0.05, 0.20, 0.10, 0.05, -0.09, -1.0), (-0.04, 0.05, 0.20, -0.10, -0.05, 0.09, 1.0)),
    (5.0, "handoff", (0.02, 0.06, 0.20, 0.11, 0.05, -0.10, -1.0), (-0.02, 0.06, 0.20, -0.11, -0.05, 0.10, -1.0)),
    (5.5, "release", (0.02, 0.06, 0.20, 0.11, 0.05, -0.10, 1.0), (-0.02, 0.06, 0.20, -0.11, -0.05, 0.10, -1.0)),
    (6.5, "verify", (0.00, 0.03, 0.15, 0.06, 0.02, -0.05, 1.0), (0.00, 0.03, 0.15, -0.06, -0.02, 0.05, -1.0)),
    (7.0, "release", (0.00, 0.03, 0.15, 0.06, 0.02, -0.05, 1.0), (0.00, 0.03, 0.15, -0.06, -0.02, 0.05, 1.0)),
    (8.0, "recover", (0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.0), (0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.0)),
)


def _smootherstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value**3 * (value * (value * 6.0 - 15.0) + 10.0)


def _interpolate_arm(
    left: tuple[float, ...],
    right: tuple[float, ...],
    fraction: float,
) -> tuple[float, ...]:
    blend = _smootherstep(fraction)
    continuous = tuple(
        left[index] + blend * (right[index] - left[index])
        for index in range(ARM_DIM)
    )
    gripper = left[ARM_DIM] if fraction < 1.0 else right[ARM_DIM]
    return (*continuous, gripper)


def build_stream() -> dict[str, Any]:
    frame_count = int(round(WAYPOINTS[-1][0] * SAMPLE_HZ)) + 1
    frames = []
    segment = 0
    for frame_index in range(frame_count):
        timestamp = frame_index / SAMPLE_HZ
        while segment + 1 < len(WAYPOINTS) - 1 and timestamp > WAYPOINTS[segment + 1][0]:
            segment += 1
        left = WAYPOINTS[segment]
        right = WAYPOINTS[segment + 1]
        fraction = (timestamp - left[0]) / (right[0] - left[0])
        psm1 = _interpolate_arm(left[2], right[2], fraction)
        psm2 = _interpolate_arm(left[3], right[3], fraction)
        phase = right[1] if abs(timestamp - right[0]) <= 1.0e-12 else left[1]
        frames.append(
            {
                "timestamp_s": round(timestamp, 6),
                "phase": phase,
                "action": [round(value, 8) for value in (*psm1, *psm2)],
            }
        )
    assert tuple(frames[0]["action"]) == NEUTRAL
    assert tuple(frames[-1]["action"]) == NEUTRAL
    return {
        "schema": "dr.anmar.timestamped-action-stream.v1",
        "id": "dranmar-dual-psm-handoff-safety-fixture-v1",
        "action_contract": "action_contract.json",
        "source": "deterministic_repository_generator",
        "source_generator": "scripts/generate_dranmar_multimodal_fixture.py",
        "purpose": "transport_resampling_stale_stop_and_bundle_validation",
        "paired_visual_observation": False,
        "training_reference_eligible": False,
        "clinician_review_status": "not_submitted",
        "native_simulator_evidence": "not_recorded",
        "real_world_evidence": "not_established",
        "clinical_validation": False,
        "sample_hz": SAMPLE_HZ,
        "duration_s": WAYPOINTS[-1][0],
        "frame_count": frame_count,
        "frames": frames,
    }


def render_stream() -> str:
    return json.dumps(build_stream(), indent=2, sort_keys=True) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output.expanduser().resolve()
    rendered = render_stream()
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != rendered:
            print(f"stale multimodal action fixture: {output}")
            return 1
        print(f"current multimodal action fixture: {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
