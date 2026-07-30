from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_dranmar_active_custody_probe.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_dranmar_active_custody_probe",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
LAUNCHER = Path(__file__).resolve().parents[1] / "dr_anmar_learning.sh"


def _role_swapped_observations() -> torch.Tensor:
    observation = torch.zeros(2, 98)
    observation[0, 82] = 1.0
    observation[1, 83] = 1.0
    paired_slices = (
        (slice(0, 8), slice(16, 24)),
        (slice(8, 16), slice(24, 32)),
        (slice(32, 39), slice(39, 46)),
        (slice(46, 53), slice(53, 60)),
        (slice(66, 68), slice(68, 70)),
        (slice(84, 91), slice(91, 98)),
    )
    for index, (robot_1_slice, robot_2_slice) in enumerate(
        paired_slices,
        start=1,
    ):
        width = robot_1_slice.stop - robot_1_slice.start
        giver = torch.arange(width, dtype=torch.float32) + 10 * index
        receiver = giver + 100
        observation[0, robot_1_slice] = giver
        observation[0, robot_2_slice] = receiver
        observation[1, robot_1_slice] = receiver
        observation[1, robot_2_slice] = giver
    observation[:, 60:66] = torch.arange(6, dtype=torch.float32)
    observation[:, 70:82] = torch.arange(12, dtype=torch.float32)
    return observation


def test_role_invariant_probe_features_survive_robot_role_swap() -> None:
    observation = _role_swapped_observations()

    features = MODULE._role_invariant_observation(observation)

    assert features.shape == (2, 82)
    assert torch.equal(features[0], features[1])


def test_probe_auc_handles_ordering_and_ties() -> None:
    probability = torch.tensor([0.1, 0.4, 0.4, 0.9])
    target = torch.tensor([False, True, False, True])

    auc = MODULE._roc_auc(probability, target)

    assert auc == 0.875


def test_platt_calibration_is_monotonic_and_rank_preserving() -> None:
    logits = torch.tensor(
        [-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0]
    )
    target = torch.tensor(
        [False, False, False, True, False, True, True, True]
    )

    calibrator = MODULE._fit_platt_calibrator(
        logits,
        target,
        l2=1.0e-4,
        max_iterations=100,
    )
    probability = MODULE._apply_platt_calibrator(
        logits,
        calibrator,
    )

    assert calibrator["slope"] > 0.0
    assert torch.equal(
        probability.argsort(),
        logits.argsort(),
    )
    assert MODULE._roc_auc(probability, target) == MODULE._roc_auc(
        logits,
        target,
    )


def test_launcher_exposes_temporal_probe_signal_audit() -> None:
    launcher = LAUNCHER.read_text()
    benchmark = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "dr_anmar_learning_benchmark.py"
    ).read_text()

    assert "receiver-custody-audit)" in launcher
    assert "receiver-custody-interventions)" in launcher
    assert "analyze_dranmar_active_custody_probe.py" in launcher
    assert "--receiver_active_custody_probe_dataset" in launcher
    assert "--receiver_active_custody_intervention" in launcher
    assert '"intervention-dataset-1.0"' in benchmark
    assert '"assigned_action_probability"' in benchmark
