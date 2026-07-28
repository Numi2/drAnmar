from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/gate_dranmar_recovery_imitation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "gate_dranmar_recovery_imitation",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def _evidence(seed: int, *, converted: int) -> dict:
    return {
        "seed": seed,
        "pickup_recovery": {
            "enabled": True,
            "head_checkpoint": {"sha256": "pickup-head"},
            "first_attempt_failures": 100,
            "lifted_10mm_after_retry": converted,
            "first_attempt_action_mismatches": 0,
            "first_attempt_action_max_abs_difference": 0.0,
        },
    }


def test_imitation_gate_allows_ppo_only_after_70_percent_conversion() -> None:
    report = GATE.evaluate(
        "pickup",
        [
            _evidence(104729, converted=71),
            _evidence(130363, converted=70),
        ],
    )

    assert report["passed"] is True
    assert report["conversion_rate"] == 0.705


def test_imitation_gate_rejects_qualification_seed_leakage() -> None:
    with pytest.raises(ValueError, match="qualification seeds"):
        GATE.evaluate("pickup", [_evidence(17, converted=100)])


def test_imitation_gate_fails_below_conversion_threshold() -> None:
    report = GATE.evaluate(
        "pickup",
        [_evidence(196613, converted=69)],
    )

    assert report["passed"] is False
    assert report["gates"]["at_least_70_percent_conversion"] is False
