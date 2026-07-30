from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_dranmar_active_custody_replication.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location(
    "evaluate_dranmar_active_custody_replication",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_locked_policy_only_pulses_at_or_above_threshold() -> None:
    risk = torch.tensor((0.01, 0.1, 0.106, 0.107, 0.4))

    action = MODULE._locked_policy_actions(
        risk,
        threshold=0.106,
    )

    assert torch.equal(action, torch.tensor((1, 1, 0, 0, 0)))


def test_sample_plan_is_zero_when_lower_bound_is_positive() -> None:
    evaluation = {
        "success_effect_vs_no_op": {
            "estimate": 0.05,
            "standard_error": 0.01,
            "samples": 500,
        }
    }

    plan = MODULE._required_sample_plan(evaluation)

    assert plan["estimated_additional_samples"] == 0


def test_sample_plan_refuses_to_extrapolate_negative_effect() -> None:
    evaluation = {
        "success_effect_vs_no_op": {
            "estimate": -0.01,
            "standard_error": 0.02,
            "samples": 500,
        }
    }

    plan = MODULE._required_sample_plan(evaluation)

    assert plan["estimated_total_samples_for_positive_lcb"] is None
