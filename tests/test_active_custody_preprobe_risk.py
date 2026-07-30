from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "train_dranmar_preprobe_risk_model.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location(
    "train_dranmar_preprobe_risk_model",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_natural_arm_mask_selects_no_motion_and_immediate_release() -> None:
    symmetric = {
        "schema_version": MODULE.SYMMETRIC_SCHEMA_VERSION,
        "assigned_action_id": torch.tensor((-1, 0, 1, 0)),
    }
    release = {
        "schema_version": MODULE.RELEASE_SCHEMA_VERSION,
        "assigned_release_delay_frames": torch.tensor((0, 1, 3, 0)),
    }

    assert torch.equal(
        MODULE._natural_arm_mask(symmetric),
        torch.tensor((False, True, False, True)),
    )
    assert torch.equal(
        MODULE._natural_arm_mask(release),
        torch.tensor((True, False, False, True)),
    )


def test_preprobe_features_do_not_depend_on_post_observation() -> None:
    payload = {
        "pre_observation": torch.zeros((3, 98)),
        "post_observation": torch.randn((3, 98)),
        "receiver_correction": torch.zeros((3, 6)),
        "retry_count": torch.zeros(3),
    }
    payload["pre_observation"][:, 82] = 1.0

    first = MODULE._preprobe_features(payload)
    payload["post_observation"] = torch.randn((3, 98)) * 100.0
    second = MODULE._preprobe_features(payload)

    assert first.shape == (3, 89)
    assert torch.equal(first, second)


def test_natural_arm_mask_rejects_unknown_schema() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        MODULE._natural_arm_mask({"schema_version": "unknown"})
