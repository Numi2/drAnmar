from __future__ import annotations

import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _evidence(
    path: Path,
    *,
    seed: int,
    successes: int,
    hard_failures: int = 0,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "task": (
                    "DrAnmar-Handover-Needle-Dual-PSM-IK-Rel-Structured-v0"
                ),
                "kind": "held_out_play",
                "seed": seed,
                "num_envs": 2000,
                "frames_per_env": 1000,
                "completed_episodes": 2000,
                "successful_episodes": successes,
                "failure_distribution": {
                    "object_dropping": hard_failures,
                    "excessive_object_force": 0,
                    "protected_surface_force": 0,
                },
                "checkpoint": {
                    "path": "model.pt",
                    "sha256": "frozen-checkpoint",
                },
                "first_terminal_outcome_per_environment": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_multiseed_gate_requires_population_confidence_and_zero_hard_failures(
    tmp_path: Path,
) -> None:
    module = runpy.run_path(
        str(ROOT / "scripts/dr_anmar_handover_multiseed_promotion.py")
    )
    seeds = (2361, 4099, 7919)
    baselines = [
        _evidence(
            tmp_path / f"baseline-{seed}.json",
            seed=seed,
            successes=1020,
        )
        for seed in seeds
    ]
    candidates = [
        _evidence(
            tmp_path / f"candidate-{seed}.json",
            seed=seed,
            successes=1500,
        )
        for seed in seeds
    ]
    result = module["evaluate_multiseed"](
        baselines,
        candidates,
        required_seeds=set(seeds),
        minimum_success_rate=0.70,
        maximum_seed_regression=0.0,
    )
    assert result["decision"] == "candidate_promoted"
    assert result["aggregate"]["candidate_success_rate"] == 0.75
    assert result["aggregate"]["candidate_success_wilson_95"][0] > 0.70

    _evidence(
        candidates[0],
        seed=seeds[0],
        successes=1500,
        hard_failures=1,
    )
    rejected = module["evaluate_multiseed"](
        baselines,
        candidates,
        required_seeds=set(seeds),
        minimum_success_rate=0.70,
        maximum_seed_regression=0.0,
    )
    assert rejected["decision"] == "baseline_retained"
    assert rejected["gates"]["zero_hard_failure_gate_passed"] is False
