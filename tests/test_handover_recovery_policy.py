from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import torch
from torch import nn

for package_name in (
    "orbit",
    "orbit.surgical",
    "orbit.surgical.tasks",
    "orbit.surgical.tasks.surgical",
    "orbit.surgical.tasks.surgical.lift",
    "orbit.surgical.tasks.surgical.handover",
):
    package = types.ModuleType(package_name)
    package.__path__ = []
    sys.modules.setdefault(package_name, package)

grasp_frames = types.ModuleType(
    "orbit.surgical.tasks.surgical.lift.grasp_frames"
)
grasp_frames.NEEDLE_PROVISIONAL_GRASP_OFFSET_M = (
    -0.0072,
    0.0015,
    0.0,
)
sys.modules[grasp_frames.__name__] = grasp_frames

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/extensions/orbit.surgical.tasks/orbit/surgical/tasks/"
    "surgical/handover/recovery_policy.py"
)
SPEC = importlib.util.spec_from_file_location(
    "orbit.surgical.tasks.surgical.handover.recovery_policy",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
RECOVERY_POLICY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RECOVERY_POLICY
SPEC.loader.exec_module(RECOVERY_POLICY)
HandoverPickupRecoveryPolicy = RECOVERY_POLICY.HandoverPickupRecoveryPolicy
PickupRecoveryHead = RECOVERY_POLICY.PickupRecoveryHead


class _FixedBasePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        action = torch.linspace(-0.7, 0.6, 14)
        action[6] = -1.0
        action[13] = 1.0
        self.register_buffer("action", action)

    def forward(self, obs):
        return self.action.expand(obs["policy"].shape[0], -1).clone()

    def reset(self, dones=None, hidden_state=None) -> None:
        pass


def _observation(batch_size: int = 2) -> dict[str, torch.Tensor]:
    raw = torch.zeros(batch_size, 98)
    raw[:, 35:39] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    raw[:, 42:46] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    raw[:, 46:49] = torch.tensor([-0.1, 0.0, -0.15])
    raw[:, 49:53] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    raw[:, 53:56] = torch.tensor([-0.1, 0.0, -0.15])
    raw[:, 56:60] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    raw[:, 77] = 1.0
    raw[:, 82] = 1.0
    raw[:, 90] = -1.0
    raw[:, 97] = 1.0
    return {"policy": raw}


def test_first_pickup_action_is_bit_identical() -> None:
    base = _FixedBasePolicy()
    policy = HandoverPickupRecoveryPolicy(base)
    observation = _observation()

    expected = base(observation)
    actual = policy(observation)

    assert torch.equal(actual, expected)
    assert torch.equal(policy.retry_count, torch.zeros_like(policy.retry_count))


def test_failed_pickup_fully_reopens_before_recovery_activation() -> None:
    policy = HandoverPickupRecoveryPolicy(_FixedBasePolicy())
    observation = _observation(batch_size=1)
    raw = observation["policy"]
    raw[:, 6:8] = 0.30
    for _ in range(policy.close_dwell_steps):
        assert torch.equal(
            policy(observation),
            policy.base_policy(observation),
        )

    raw[:, 6:8] = 0.43

    failed_action = policy(observation)

    assert torch.equal(failed_action[:, :6], torch.zeros(1, 6))
    assert failed_action[0, 6].item() == 1.0
    assert policy.first_attempt_failed.item()
    assert policy.retry_count.item() == 0

    raw[:, 6:8] = 0.0
    for _ in range(2):
        settling_action = policy(observation)
        assert torch.equal(settling_action[:, :6], torch.zeros(1, 6))
        assert settling_action[0, 6].item() == 1.0
        assert policy.retry_count.item() == 0

    retry_action = policy(observation)

    assert policy.retry_count.item() == 1
    assert policy.activation_count.item() == 1
    assert not torch.equal(retry_action[:, :6], torch.zeros(1, 6))


def test_recovery_correction_is_latched_bounded_and_loses_authority_on_custody() -> None:
    policy = HandoverPickupRecoveryPolicy(_FixedBasePolicy())
    observation = _observation(batch_size=1)
    raw = observation["policy"]
    policy.set_fixed_correction(
        torch.tensor([0.02, 0.0, 0.0, math.radians(20.0), 0.0, 0.0])
    )
    raw[:, 6:8] = 0.30
    for _ in range(policy.close_dwell_steps):
        policy(observation)
    raw[:, 6:8] = 0.43
    policy(observation)
    raw[:, 6:8] = 0.0
    for _ in range(3):
        policy(observation)

    assert policy.correction[0, :3].norm().item() <= 0.0050001
    assert policy.correction[0, 3:].norm().item() <= math.radians(5.0) + 1.0e-6
    latched = policy.correction.clone()
    policy(observation)
    assert torch.equal(policy.correction, latched)

    raw[:, 77] = 0.0
    raw[:, 78] = 1.0
    raw[:, 66:68] = 0.01
    expected = policy.base_policy(observation)
    for _ in range(3):
        actual = policy(observation)

    assert torch.equal(actual, expected)
    assert policy.recovered_custody.item()


def test_recovery_head_contract_has_no_gripper_channel() -> None:
    head = PickupRecoveryHead()
    output = head(torch.zeros(4, PickupRecoveryHead.input_dim))

    assert output.shape == (4, 6)
    assert torch.equal(output, torch.zeros_like(output))


def test_custody_loss_is_not_redeclared_during_reapproach() -> None:
    policy = HandoverPickupRecoveryPolicy(_FixedBasePolicy())
    observation = _observation(batch_size=1)
    raw = observation["policy"]
    raw[:, 77] = 0.0
    raw[:, 78] = 1.0
    raw[:, 66:68] = 0.01
    for _ in range(3):
        policy(observation)

    raw[:, 66:68] = 0.0
    for _ in range(policy.custody_loss_steps):
        policy(observation)
    assert policy.first_attempt_failed.item()

    raw[:, 6:8] = 0.0
    for _ in range(policy.open_settle_steps + 2):
        policy(observation)
    assert policy.retry_count.item() == 1

    for _ in range(10):
        policy(observation)
    assert policy.retry_count.item() == 1
    assert policy.activation_count.item() == 1
