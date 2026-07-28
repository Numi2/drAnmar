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
    protected_surface_failures: int = 0,
    population_sha256: str | None = None,
    runtime_contract_sha256: str = "frozen-runtime-contract",
    environment_contract_sha256: str = "frozen-environment-contract",
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
                    "protected_surface_force": (
                        protected_surface_failures
                    ),
                },
                "checkpoint": {
                    "path": "model.pt",
                    "sha256": "frozen-checkpoint",
                },
                "policy_runtime_contract_sha256": (
                    runtime_contract_sha256
                ),
                "environment_runtime_contract_sha256": (
                    environment_contract_sha256
                ),
                "first_terminal_outcome_per_environment": True,
                "initial_state_population_sha256": (
                    population_sha256 or f"population-{seed}"
                ),
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
    assert result["gates"]["aggregate_improvement_gate_passed"] is True

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
    assert (
        rejected["gates"]["zero_catastrophic_failure_gate_passed"]
        is False
    )


def test_multiseed_gate_uses_paired_populations_and_safety_noninferiority(
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
            successes=1200,
            protected_surface_failures=2,
        )
        for seed in seeds
    ]
    candidates = [
        _evidence(
            tmp_path / f"candidate-{seed}.json",
            seed=seed,
            successes=1500,
            protected_surface_failures=3,
        )
        for seed in seeds
    ]
    result = module["evaluate_multiseed"](
        baselines,
        candidates,
        required_seeds=set(seeds),
        minimum_success_rate=0.70,
        maximum_seed_regression=0.0,
        maximum_protected_surface_rate_increase=0.001,
    )
    assert result["decision"] == "candidate_promoted"
    assert result["gates"]["paired_population_gate_passed"] is True
    assert (
        result["gates"][
            "protected_surface_noninferiority_gate_passed"
        ]
        is True
    )

    _evidence(
        candidates[0],
        seed=seeds[0],
        successes=1500,
        protected_surface_failures=3,
        population_sha256="different-population",
    )
    unpaired = module["evaluate_multiseed"](
        baselines,
        candidates,
        required_seeds=set(seeds),
        minimum_success_rate=0.70,
        maximum_seed_regression=0.0,
    )
    assert unpaired["decision"] == "baseline_retained"
    assert unpaired["gates"]["paired_population_gate_passed"] is False


def test_multiseed_gate_requires_preregistered_aggregate_improvement(
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
            successes=1500,
            runtime_contract_sha256="baseline-runtime",
        )
        for seed in seeds
    ]
    candidates = [
        _evidence(
            tmp_path / f"candidate-{seed}.json",
            seed=seed,
            successes=1504,
            runtime_contract_sha256="candidate-runtime",
        )
        for seed in seeds
    ]
    result = module["evaluate_multiseed"](
        baselines,
        candidates,
        required_seeds=set(seeds),
        minimum_success_rate=0.70,
        maximum_seed_regression=0.0,
        minimum_aggregate_improvement=0.005,
    )
    assert result["decision"] == "baseline_retained"
    assert abs(
        result["aggregate"]["candidate_minus_baseline"] - 0.002
    ) < 1.0e-12
    assert (
        result["gates"]["aggregate_improvement_gate_passed"] is False
    )


def test_multiseed_gate_rejects_environment_contract_drift(
    tmp_path: Path,
) -> None:
    module = runpy.run_path(
        str(ROOT / "scripts/dr_anmar_handover_multiseed_promotion.py")
    )
    seeds = (2361, 4099, 7919)
    baselines = [
        _evidence(
            tmp_path / f"baseline-env-{seed}.json",
            seed=seed,
            successes=1200,
        )
        for seed in seeds
    ]
    candidates = [
        _evidence(
            tmp_path / f"candidate-env-{seed}.json",
            seed=seed,
            successes=1500,
            environment_contract_sha256="changed-environment-contract",
        )
        for seed in seeds
    ]
    try:
        module["evaluate_multiseed"](
            baselines,
            candidates,
            required_seeds=set(seeds),
            minimum_success_rate=0.70,
            maximum_seed_regression=0.0,
        )
    except ValueError as error:
        assert "environment contracts differ" in str(error)
    else:
        raise AssertionError("environment contract drift must fail closed")
