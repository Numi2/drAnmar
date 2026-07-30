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
grasp_frames.needle_geometry_grasp_offset_m = lambda fraction: (
    -0.004,
    0.003,
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
HandoverReceiverRecoveryPolicy = (
    RECOVERY_POLICY.HandoverReceiverRecoveryPolicy
)
PickupRecoveryHead = RECOVERY_POLICY.PickupRecoveryHead
ReceiverRecoveryHead = RECOVERY_POLICY.ReceiverRecoveryHead
ReceiverAttemptActorCritic = RECOVERY_POLICY.ReceiverAttemptActorCritic


class _FixedBasePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        action = torch.linspace(-0.7, 0.6, 14)
        action[6] = -1.0
        action[13] = -1.0
        self.register_buffer("action", action)

    def forward(self, obs):
        return self.action.expand(obs["policy"].shape[0], -1).clone()

    def reset(self, dones=None, hidden_state=None) -> None:
        pass

    def as_jit(self) -> nn.Module:
        return _FixedTensorPolicy(self.action)


class _FixedTensorPolicy(nn.Module):
    def __init__(self, action: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("action", action.clone())

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.action.expand(observation.shape[0], -1).clone()


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


def test_failed_pickup_fully_reopens_before_recovery_activation() -> None:
    policy = HandoverPickupRecoveryPolicy(_FixedBasePolicy())
    observation = _observation(batch_size=1)
    raw = observation["policy"]
    raw[:, 6:8] = 0.30
    for _ in range(policy.close_dwell_steps):
        policy(observation)

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


def test_dagger_offset_is_added_to_head_prediction_and_latched() -> None:
    policy = HandoverPickupRecoveryPolicy(_FixedBasePolicy())
    observation = _observation(batch_size=1)
    raw = observation["policy"]
    local_delta = torch.tensor(
        [0.001, -0.0005, 0.0, math.radians(0.5), 0.0, 0.0]
    )
    policy.set_fixed_correction_delta(local_delta)
    raw[:, 6:8] = 0.30
    for _ in range(policy.close_dwell_steps):
        policy(observation)
    raw[:, 6:8] = 0.43
    policy(observation)
    raw[:, 6:8] = 0.0
    for _ in range(policy.open_settle_steps):
        policy(observation)

    assert torch.allclose(policy.correction[0], local_delta)
    latched = policy.correction.clone()
    policy(observation)
    assert torch.equal(policy.correction, latched)


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


def test_receiver_reopens_before_retry() -> None:
    base = _FixedBasePolicy()
    policy = HandoverReceiverRecoveryPolicy(base)
    observation = _observation(batch_size=1)
    raw = observation["policy"]
    raw[:, 77] = 0.0
    raw[:, 79] = 1.0
    raw[:, 97] = -1.0
    raw[:, 22:24] = 0.30
    raw[:, 66:68] = 0.01

    for _ in range(policy.close_dwell_steps):
        policy(observation)

    raw[:, 22:24] = 0.43
    failed_action = policy(observation)
    assert torch.equal(failed_action[:, :6], torch.zeros(1, 6))
    assert failed_action[0, 6].item() == -1.0
    assert torch.equal(failed_action[:, 7:13], torch.zeros(1, 6))
    assert failed_action[0, 13].item() == 1.0

    raw[:, 22:24] = 0.0
    for _ in range(policy.open_settle_steps):
        retry_action = policy(observation)
    assert policy.retry_count.item() == 1
    assert policy.activation_count.item() == 1
    assert torch.equal(retry_action[:, :6], torch.zeros(1, 6))
    assert retry_action[0, 6].item() == -1.0

    raw[:, 79] = 0.0
    raw[:, 80] = 1.0
    raw[:, 68:70] = 0.01
    for _ in range(3):
        policy(observation)
    assert policy.recovered_acquisition.item()


def test_receiver_recovery_head_contract_has_no_gripper_channel() -> None:
    head = ReceiverRecoveryHead()
    output = head(torch.zeros(4, ReceiverRecoveryHead.input_dim))
    assert output.shape == (4, 6)
    assert torch.equal(output, torch.zeros_like(output))


def test_receiver_attempt_actor_starts_as_exact_zero_residual() -> None:
    actor_critic = ReceiverAttemptActorCritic()
    features = torch.randn(32, ReceiverAttemptActorCritic.input_dim)

    action, log_probability, value = actor_critic.act(
        features,
        stochastic=False,
    )
    evaluated_log_probability, entropy, evaluated_value = (
        actor_critic.evaluate_actions(features, action)
    )

    assert torch.equal(action, torch.zeros_like(action))
    assert action.shape == (32, 6)
    assert torch.allclose(log_probability, evaluated_log_probability)
    assert torch.allclose(value, evaluated_value)
    assert entropy.shape == (32,)


def test_receiver_attempt_stochastic_action_is_bounded() -> None:
    actor_critic = ReceiverAttemptActorCritic(initial_std=0.25)
    action, _, value = actor_critic.act(
        torch.zeros(1024, ReceiverAttemptActorCritic.input_dim),
        stochastic=True,
    )

    assert torch.all(action > -1.0)
    assert torch.all(action < 1.0)
    assert torch.all((0.0 <= value) & (value <= 1.0))


def test_receiver_contact_loss_before_transfer_forces_full_reset() -> None:
    policy = HandoverReceiverRecoveryPolicy(_FixedBasePolicy())
    observation = _observation(batch_size=1)
    raw = observation["policy"]
    raw[:, 77] = 0.0
    raw[:, 79] = 1.0
    raw[:, 68:70] = 0.01
    raw[:, 97] = -1.0
    for _ in range(3):
        policy(observation)

    assert policy.retry_state.item() == policy.state_secure

    raw[:, 68:70] = 0.0
    for _ in range(3):
        action = policy(observation)

    assert policy.first_attempt_failed.item()
    assert action[0, 13].item() == 1.0

    for _ in range(2):
        action = policy(observation)
    assert action[0, 13].item() == 1.0


def test_active_custody_probe_pulses_once_per_acquisition_attempt() -> None:
    policy = HandoverReceiverRecoveryPolicy(
        _FixedBasePolicy(),
        active_custody_verification=True,
    )
    observation = _observation(batch_size=1)
    raw = observation["policy"]
    raw[:, 77] = 0.0
    raw[:, 80] = 1.0
    raw[:, 66:68] = 0.01
    raw[:, 68:70] = 0.01

    for _ in range(3):
        pulse_action = policy(observation)

    assert pulse_action[0, 6].item() == 1.0
    assert policy.active_custody_probe_pending.item()
    assert policy.active_custody_probe_attempted.item()
    assert torch.equal(
        policy.active_custody_probe_pre_observation,
        raw,
    )

    raw[:, 46] = -0.123
    policy(observation)
    assert not policy.active_custody_probe_pending.item()
    assert policy.active_custody_probe_evaluated.item()
    assert policy.active_custody_probe_survived.item()
    assert policy.receiver_release_authorized.item()
    assert policy.active_custody_probe_post_observation[0, 46].item() == (
        raw[0, 46].item()
    )

    post_probe_action = policy(observation)
    assert post_probe_action[0, 6].item() == -1.0


def test_active_custody_intervention_is_seeded_uniform_and_bounded() -> None:
    seed = 4_104_736
    policy = HandoverReceiverRecoveryPolicy(
        _FixedBasePolicy(),
        active_custody_verification=True,
        active_custody_intervention=True,
        active_custody_intervention_seed=seed,
    )
    observation = _observation(batch_size=9)
    raw = observation["policy"]
    raw[:, 77] = 0.0
    raw[:, 80] = 1.0
    raw[:, 66:68] = 0.01
    raw[:, 68:70] = torch.tensor(
        [[0.005, 0.01], [0.01, 0.005], [0.005, 0.01]]
    ).repeat(3, 1)

    for _ in range(3):
        policy(observation)
    expected_id = policy._randomized_active_custody_action_id()
    expected_direction = torch.sign(
        raw[:, 69] - raw[:, 68]
    )

    action = policy(observation)

    assert set(expected_id.tolist()) == {-1, 0, 1}
    assert torch.equal(
        policy.active_custody_intervention_action_id,
        expected_id,
    )
    assert bool(policy.active_custody_intervention_assigned.all())
    assert torch.allclose(
        policy.active_custody_intervention_probability,
        torch.full((9,), 1.0 / 3.0),
    )
    assert torch.allclose(
        action[:, 9],
        expected_id.float()
        * expected_direction
        * policy.active_custody_intervention_action_limit,
    )
    assert torch.equal(
        policy.active_custody_intervention_action,
        action[:, 7:14],
    )
    assert bool(
        (
            policy.last_receiver_action_owner
            == policy._RECEIVER_OWNER_ACTIVE_CUSTODY_INTERVENTION
        ).all()
    )


def test_active_custody_release_delay_randomizes_zero_one_three_frames() -> None:
    policy = HandoverReceiverRecoveryPolicy(
        _FixedBasePolicy(),
        active_custody_verification=True,
        active_custody_intervention=True,
        active_custody_intervention_profile="release_delay",
        active_custody_intervention_seed=4_104_736,
    )
    observation = _observation(batch_size=9)
    raw = observation["policy"]
    raw[:, 77] = 0.0
    raw[:, 80] = 1.0
    raw[:, 66:68] = 0.01
    raw[:, 68:70] = 0.01

    for _ in range(3):
        policy(observation)
    randomized_bucket = policy._randomized_active_custody_action_id()
    expected_delay = torch.where(
        randomized_bucket < 0,
        torch.zeros_like(randomized_bucket),
        torch.where(
            randomized_bucket == 0,
            torch.ones_like(randomized_bucket),
            torch.full_like(randomized_bucket, 3),
        ),
    )

    policy(observation)

    assert set(expected_delay.tolist()) == {0, 1, 3}
    assert torch.equal(
        policy.active_custody_intervention_action_id,
        expected_delay,
    )
    assert torch.equal(
        policy.active_custody_intervention_applied_frames,
        (expected_delay > 0).long(),
    )
    delayed = expected_delay > 0
    assert bool(
        (policy.last_giver_action_owner[delayed] == policy._GIVER_OWNER_RELEASE_DELAY).all()
    )
    assert bool(
        (
            policy.last_receiver_action_owner[delayed]
            == policy._RECEIVER_OWNER_ACTIVE_CUSTODY_INTERVENTION
        ).all()
    )

    policy(observation)
    policy(observation)

    assert torch.equal(
        policy.active_custody_intervention_applied_frames,
        expected_delay,
    )
    assert bool((policy.active_custody_intervention_delay_remaining == 0).all())
    assert bool(policy.receiver_release_authorized.all())


def test_active_custody_preemptive_retry_randomizes_binary_decision() -> None:
    policy = HandoverReceiverRecoveryPolicy(
        _FixedBasePolicy(),
        active_custody_verification=True,
        active_custody_intervention=True,
        active_custody_intervention_profile="preemptive_retry",
        active_custody_intervention_seed=4_104_736,
    )
    observation = _observation(batch_size=10)
    raw = observation["policy"]
    raw[:, 77] = 0.0
    raw[:, 80] = 1.0
    raw[:, 66:68] = 0.01
    raw[:, 68:70] = 0.01

    for _ in range(3):
        policy(observation)
    expected_decision = (
        policy._randomized_active_custody_binary_action_id()
    )

    policy(observation)

    treatment = expected_decision == 1
    control = ~treatment
    assert set(expected_decision.tolist()) == {0, 1}
    assert torch.equal(
        policy.active_custody_intervention_action_id,
        expected_decision,
    )
    assert torch.allclose(
        policy.active_custody_intervention_probability,
        torch.full((10,), 0.5),
    )
    assert bool(policy.receiver_release_authorized[control].all())
    assert not bool(policy.receiver_release_authorized[treatment].any())
    assert bool(policy.first_attempt_failed[treatment].all())
    assert not bool(policy.first_attempt_failed[control].any())
    assert bool(
        (policy.retry_state[treatment] == policy.state_reopening).all()
    )
    assert torch.equal(
        policy.active_custody_intervention_action[treatment, :6],
        torch.zeros_like(
            policy.active_custody_intervention_action[treatment, :6]
        ),
    )
    assert torch.equal(
        policy.active_custody_intervention_action[treatment, 6],
        torch.ones_like(
            policy.active_custody_intervention_action[treatment, 6]
        ),
    )
    assert torch.equal(
        policy.active_custody_intervention_giver_action[treatment, 6],
        -torch.ones_like(
            policy.active_custody_intervention_giver_action[treatment, 6]
        ),
    )
    assert torch.equal(
        policy.active_custody_intervention_action[control],
        torch.zeros_like(
            policy.active_custody_intervention_action[control]
        ),
    )
    assert bool(
        (
            policy.last_receiver_action_owner[treatment]
            == policy._RECEIVER_OWNER_RESET_OPEN
        ).all()
    )
    assert bool(
        (
            policy.last_giver_action_owner[treatment]
            == policy._GIVER_OWNER_HOLD
        ).all()
    )


def test_preprobe_risk_trial_randomizes_before_giver_probe() -> None:
    policy = HandoverReceiverRecoveryPolicy(
        _FixedBasePolicy(),
        enable_retries=False,
        active_custody_verification=True,
        active_custody_intervention=True,
        active_custody_intervention_profile="preprobe_retry",
        active_custody_preprobe_risk_feature_mean=torch.zeros(89),
        active_custody_preprobe_risk_feature_std=torch.ones(89),
        active_custody_preprobe_risk_weight=torch.zeros(89),
        active_custody_preprobe_risk_bias=-5.0,
        active_custody_preprobe_risk_threshold=0.5,
        active_custody_intervention_seed=4_104_736,
    )
    observation = _observation(batch_size=10)
    raw = observation["policy"]
    raw[:, 77] = 0.0
    raw[:, 80] = 1.0
    raw[:, 66:68] = 0.01
    raw[:, 68:70] = 0.01

    action = None
    for _ in range(4):
        action = policy(observation)
        if bool(policy.active_custody_intervention_assigned.all()):
            break

    assert action is not None
    decision = policy.active_custody_intervention_action_id
    treatment = decision == 1
    control = decision == 0
    assert set(decision.tolist()) == {0, 1}
    assert torch.allclose(
        policy.active_custody_intervention_probability,
        torch.full((10,), 0.5),
    )
    assert bool(
        (
            policy.active_custody_preprobe_risk
            > policy.active_custody_preprobe_risk_threshold
        ).all()
    )
    assert bool(policy.active_custody_probe_attempted[control].all())
    assert not bool(
        policy.active_custody_probe_attempted[treatment].any()
    )
    assert bool(policy.first_attempt_failed[treatment].all())
    assert not bool(policy.first_attempt_failed[control].any())
    assert bool(
        (policy.retry_state[treatment] == policy.state_reopening).all()
    )
    assert torch.equal(
        policy.active_custody_intervention_action[treatment, 6],
        torch.ones_like(
            policy.active_custody_intervention_action[treatment, 6]
        ),
    )
    assert torch.equal(
        policy.active_custody_intervention_giver_action[treatment, 6],
        -torch.ones_like(
            policy.active_custody_intervention_giver_action[treatment, 6]
        ),
    )
    assert torch.equal(
        policy.active_custody_intervention_action[control],
        torch.zeros_like(
            policy.active_custody_intervention_action[control]
        ),
    )
    assert bool(
        (
            policy.last_receiver_action_owner[treatment]
            == policy._RECEIVER_OWNER_RESET_OPEN
        ).all()
    )
    assert bool(
        (
            policy.last_giver_action_owner[treatment]
            == policy._GIVER_OWNER_HOLD
        ).all()
    )


def test_retry_retreat_budget_is_cumulative_across_contact_flicker() -> None:
    policy = HandoverReceiverRecoveryPolicy(
        _FixedBasePolicy(),
        retry_clearance_retreat=True,
    )
    observation = _observation(batch_size=1)
    raw = observation["policy"]
    raw[:, 77] = 0.0
    raw[:, 79] = 1.0
    raw[:, 22:24] = 0.30
    policy(observation)
    policy.retry_state[:] = policy.state_reopening

    raw[:, 66:68] = torch.tensor([[0.01, 0.0]])
    for _ in range(5):
        policy(observation)
    raw[:, 66:68] = 0.01
    policy(observation)
    raw[:, 66:68] = torch.tensor([[0.01, 0.0]])
    for _ in range(5):
        tenth_retreat_action = policy(observation)

    assert policy.clearance_retreat_dwell.item() == 10
    assert not torch.equal(
        tenth_retreat_action[:, 7:10],
        torch.zeros(1, 3),
    )

    exhausted_action = policy(observation)
    assert policy.clearance_retreat_dwell.item() == 11
    assert torch.equal(exhausted_action[:, 7:10], torch.zeros(1, 3))


def test_retry_force_centering_sets_the_requested_direction() -> None:
    policy = HandoverReceiverRecoveryPolicy(
        _FixedBasePolicy(),
        retry_force_centering=True,
    )
    observation = _observation(batch_size=1)
    raw = observation["policy"]
    raw[:, 77] = 0.0
    raw[:, 79] = 1.0
    raw[:, 66:68] = 0.01
    raw[:, 68:70] = torch.tensor([[0.0, 0.01]])
    policy(observation)
    policy.retry_state[:] = policy.state_learned_retry
    policy.retry_count[:] = 1
    policy.acquisition_started[:] = True

    action = policy(observation)

    assert math.isclose(
        action[0, 9].item(),
        -policy.contact_centering_action_limit,
        abs_tol=1.0e-7,
    )
    assert (
        policy.last_receiver_action_owner.item()
        == policy._RECEIVER_OWNER_FORCE_CENTERING
    )


def test_phase_three_retry_approach_is_not_blocked_by_custody_hold() -> None:
    policy = HandoverReceiverRecoveryPolicy(_FixedBasePolicy())
    observation = _observation(batch_size=1)
    raw = observation["policy"]
    raw[:, 77] = 0.0
    raw[:, 80] = 1.0
    raw[:, 66:68] = 0.01
    policy(observation)
    policy.retry_state[:] = policy.state_learned_retry
    policy.retry_count[:] = 1
    policy.acquisition_started[:] = True

    policy(observation)

    assert (
        policy.last_receiver_action_owner.item()
        == policy._RECEIVER_OWNER_CORRECTED_APPROACH
    )


def test_canonical_retention_centering_does_not_touch_recovered_retry() -> None:
    policy = HandoverReceiverRecoveryPolicy(
        _FixedBasePolicy(),
        receiver_retention_contact_centering=True,
    )
    observation = _observation(batch_size=1)
    raw = observation["policy"]
    raw[:, 77] = 0.0
    raw[:, 80] = 1.0
    raw[:, 66:68] = 0.0
    raw[:, 68:70] = torch.tensor([[0.0, 0.01]])
    policy(observation)
    policy.retry_count[:] = 1
    expected = policy.base_policy(observation)

    action = policy(observation)

    assert torch.equal(action, expected)
    assert policy.last_receiver_action_owner.item() == 0
