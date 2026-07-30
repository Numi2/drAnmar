from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "train_dranmar_receiver_attempt_ppo.py"
)
LAUNCHER = Path(__file__).resolve().parents[1] / "dr_anmar_learning.sh"
SPEC = importlib.util.spec_from_file_location(
    "train_dranmar_receiver_attempt_ppo",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _legacy_checkpoint(path: Path) -> None:
    torch.save(
        {
            "schema_version": MODULE.SCHEMA_VERSION,
            "receiver_attempt_actor_critic": {
                "actor.weight": torch.tensor([1.0]),
            },
            "feature_mean": torch.zeros(36),
            "feature_std": torch.ones(36),
        },
        path,
    )


def test_bind_source_migrates_metadata_without_changing_policy(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy.pt"
    output = tmp_path / "bound.pt"
    _legacy_checkpoint(source)
    original = torch.load(source, map_location="cpu", weights_only=False)
    revision = MODULE._source_revision()

    result = MODULE.main(
        [
            "bind-source",
            "--checkpoint",
            str(source),
            "--source_revision",
            revision,
            "--output",
            str(output),
        ]
    )

    assert result == 0
    bound = torch.load(output, map_location="cpu", weights_only=False)
    assert bound["source_revision"] == revision
    assert torch.equal(
        bound["receiver_attempt_actor_critic"]["actor.weight"],
        original["receiver_attempt_actor_critic"]["actor.weight"],
    )
    assert bound["source_binding"][
        "policy_weights_unchanged_during_binding"
    ]


def test_update_rejects_unbound_legacy_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "legacy.pt"
    _legacy_checkpoint(source)

    with pytest.raises(ValueError, match="source revision"):
        MODULE.main(
            [
                "update",
                "--checkpoint",
                str(source),
                "--rollout",
                str(tmp_path / "unused.pt"),
                "--output",
                str(tmp_path / "updated.pt"),
            ]
        )


def test_bind_source_rejects_an_already_bound_checkpoint(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bound.pt"
    output = tmp_path / "rebound.pt"
    _legacy_checkpoint(source)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    payload["source_revision"] = MODULE._source_revision()
    torch.save(payload, source)

    with pytest.raises(ValueError, match="already source-bound"):
        MODULE.main(
            [
                "bind-source",
                "--checkpoint",
                str(source),
                "--source_revision",
                MODULE._source_revision(),
                "--output",
                str(output),
            ]
        )


def test_attempt_handover_forwards_seed_stream_offset() -> None:
    launcher = LAUNCHER.read_text()
    benchmark = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "dr_anmar_learning_benchmark.py"
    ).read_text()
    attempt_block = launcher.split("    attempt-handover)", 1)[1].split(
        "    selector-handover)", 1
    )[0]
    promoted_block = launcher.split("    promoted-handover)", 1)[1].split(
        "    play)", 1
    )[0]

    assert 'DR_ANMAR_SEED_STREAM_OFFSET="${7:-0}"' in attempt_block
    assert (
        'selector_seed_stream_offset_env="${DR_ANMAR_SEED_STREAM_OFFSET:-0}"'
        in promoted_block
    )
    attempt_rollout_block = benchmark.split(
        '"dranmar-receiver-attempt-ppo-rollout-1.0"', 1
    )[1].split('"receiver_gate_step"', 1)[0]
    assert '"seed_stream_offset": args.seed_stream_offset' in (
        attempt_rollout_block
    )


def test_risk_auxiliary_is_centered_within_seed_and_outcome() -> None:
    seed = torch.tensor((104729, 104729, 104729, 104729, 130363, 130363))
    success = torch.tensor((False, False, True, True, True, True))
    observed = torch.tensor((True, True, True, False, True, True))
    predicted_risk = torch.tensor((0.8, 0.2, 0.1, 0.0, 0.7, 0.3))

    auxiliary, report = MODULE._within_outcome_risk_auxiliary(
        seed,
        success,
        observed,
        predicted_risk,
    )

    assert torch.allclose(auxiliary[:2], torch.tensor((-1.0, 1.0)))
    assert auxiliary[2].item() == 0.0
    assert auxiliary[3].item() == 0.0
    assert torch.allclose(auxiliary[4:], torch.tensor((-1.0, 1.0)))
    assert report["observed"] == 5
    assert report["centering"] == (
        "standardized_and_clipped_within_seed_and_terminal_outcome"
    )


def test_launcher_exposes_risk_monitored_attempt_update() -> None:
    launcher = LAUNCHER.read_text()
    benchmark = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "dr_anmar_learning_benchmark.py"
    ).read_text()

    assert "receiver-attempt-risk-update)" in launcher
    assert "attempt-risk-handover)" in launcher
    assert "DR_ANMAR_RECEIVER_ACTIVE_CUSTODY_PREPROBE_RISK_MONITOR=1" in launcher
    assert '"dranmar-receiver-attempt-risk-ppo-rollout-1.0"' in benchmark
    assert '"preprobe_risk_observed"' in benchmark
    assert '"preprobe_risk_checkpoint_sha256"' in benchmark


def test_risk_update_consumes_all_seed_monitored_rollouts(
    tmp_path: Path,
) -> None:
    actor_critic_type, _ = MODULE._repo_models()
    model = actor_critic_type()
    checkpoint = tmp_path / "attempt.pt"
    torch.save(
        {
            "schema_version": MODULE.SCHEMA_VERSION,
            "source_revision": MODULE._source_revision(),
            "receiver_attempt_actor_critic": model.state_dict(),
            "feature_mean": torch.zeros(actor_critic_type.input_dim),
            "feature_std": torch.ones(actor_critic_type.input_dim),
            "base_checkpoint_sha256": "a" * 64,
            "receiver_candidate_checkpoint_sha256": "b" * 64,
            "receiver_gate_step": 50,
            "receiver_position_cap_m": 0.0025,
            "receiver_orientation_cap_rad": 0.03490658503988659,
            "residual_position_cap_m": 0.001,
            "residual_orientation_cap_rad": 0.017453292519943295,
            "training": {
                "decisions": 0,
                "updates": 0,
                "rollouts": [],
            },
        },
        checkpoint,
    )
    risk_checkpoint = tmp_path / "risk.pt"
    torch.save(
        {
            "schema_version": (
                "dranmar-active-custody-preprobe-risk-model-1.0"
            ),
            "source_revision": MODULE._source_revision(),
            "base_checkpoint_sha256": "a" * 64,
            "receiver_candidate_checkpoint_sha256": "b" * 64,
            "cross_fit_gate": {"signal_gate_passed": True},
            "motion_control_authorized": False,
        },
        risk_checkpoint,
    )
    rollout_paths = []
    for seed in sorted(MODULE.DEVELOPMENT_SEEDS):
        features = torch.randn(4, actor_critic_type.input_dim)
        action = torch.zeros(4, actor_critic_type.action_dim)
        with torch.no_grad():
            old_log_probability, _, old_value = model.evaluate_actions(
                features,
                action,
            )
        rollout_path = tmp_path / f"rollout-{seed}.pt"
        torch.save(
            {
                "schema_version": MODULE.RISK_ROLLOUT_SCHEMA_VERSION,
                "seed": seed,
                "seed_stream_offset": 1,
                "receiver_attempt_checkpoint_sha256": MODULE._sha256(
                    checkpoint
                ),
                "preprobe_risk_checkpoint_sha256": MODULE._sha256(
                    risk_checkpoint
                ),
                "base_checkpoint_sha256": "a" * 64,
                "receiver_candidate_checkpoint_sha256": "b" * 64,
                "receiver_position_cap_m": 0.0025,
                "receiver_orientation_cap_rad": 0.03490658503988659,
                "residual_position_cap_m": 0.001,
                "residual_orientation_cap_rad": 0.017453292519943295,
                "stochastic": True,
                "features": features,
                "action": action,
                "old_log_probability": old_log_probability,
                "old_value": old_value,
                "full_success": torch.tensor(
                    (False, False, True, True)
                ),
                "preprobe_risk_observed": torch.ones(
                    4,
                    dtype=torch.bool,
                ),
                "predicted_preprobe_risk": torch.tensor(
                    (0.8, 0.2, 0.7, 0.1)
                ),
            },
            rollout_path,
        )
        rollout_paths.append(rollout_path)
    output = tmp_path / "updated.pt"
    arguments = [
        "risk-update",
        "--checkpoint",
        str(checkpoint),
        "--risk_checkpoint",
        str(risk_checkpoint),
        "--output",
        str(output),
        "--minimum_decisions",
        "12",
        "--minimum_risk_observations",
        "12",
        "--epochs",
        "1",
        "--minibatch_size",
        "12",
    ]
    for path in rollout_paths:
        arguments.extend(("--rollout", str(path)))

    assert MODULE.main(arguments) == 0
    updated = torch.load(output, map_location="cpu", weights_only=False)
    last_update = updated["training"]["last_update"]
    assert last_update["objective"] == (
        "terminal_success_plus_bounded_within_outcome_risk_auxiliary"
    )
    assert last_update["risk_auxiliary"]["observed"] == 12
    assert last_update["risk_checkpoint"]["sha256"] == MODULE._sha256(
        risk_checkpoint
    )
