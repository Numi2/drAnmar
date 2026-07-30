from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_dranmar_release_delay_gate.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location(
    "evaluate_dranmar_release_delay_gate",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_release_compliance_retains_terminally_truncated_three_frame_arm(
) -> None:
    assigned = torch.tensor((0, 1, 3, 3, 3))
    applied = torch.tensor((0, 1, 1, 2, 3))

    result = MODULE._release_compliance(assigned, applied)

    assert result == {
        "fully_applied": 3,
        "terminally_truncated": 2,
    }


def test_release_compliance_rejects_under_applied_one_frame_arm() -> None:
    with pytest.raises(
        ValueError,
        match="application contract drifted",
    ):
        MODULE._release_compliance(
            torch.tensor((0, 1, 3)),
            torch.tensor((0, 0, 3)),
        )


def test_difference_in_success_reports_delay_harm() -> None:
    assigned = torch.tensor((0, 0, 0, 0, 1, 1, 1, 1))
    success = torch.tensor(
        (True, True, True, True, True, False, False, False)
    )

    result = MODULE._difference_in_success(
        assigned,
        success,
        treatment=1,
    )

    assert result["estimate"] == pytest.approx(-0.75)
    assert result["upper_one_sided_95"] < 0.0
