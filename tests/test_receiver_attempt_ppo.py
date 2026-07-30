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
