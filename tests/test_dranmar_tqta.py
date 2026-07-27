from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/dr_anmar_tqta.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("dr_anmar_tqta", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def _contract(path: Path) -> Path:
    return _write_json(
        path,
        {
            "defaults": {
                "held_out_seeds": [101, 202],
                "success_threshold": 0.9,
            },
            "stages": [
                {
                    "task": "DrAnmar-Test-v0",
                    "promotion": {
                        "minimum_success_rate": 0.9,
                        "held_out_seed_passes": 2,
                    },
                }
            ],
        },
    )


def _training(path: Path, checkpoint: str) -> Path:
    return _write_json(
        path,
        {
            "kind": "training",
            "task": "DrAnmar-Test-v0",
            "seed": 17,
            "wall_time_s": 180.0,
            "simulated_frames": 123456,
            "checkpoint": {"sha256": checkpoint},
            "runtime": {
                "source": {"dranmar_revision": "revision-a"}
            },
        },
    )


def _play(
    path: Path,
    checkpoint: str | None,
    *,
    seed: int,
    success_rate: float,
    analytic_only: bool = False,
    sustained_losses: int = 0,
    protected_surface_force: int = 0,
    num_envs: int = 100,
) -> Path:
    return _write_json(
        path,
        {
            "kind": "held_out_play",
            "task": "DrAnmar-Test-v0",
            "seed": seed,
            "success_rate": success_rate,
            "checkpoint": (
                {"sha256": checkpoint}
                if checkpoint is not None
                else None
            ),
            "analytic_only": analytic_only,
            "num_envs": num_envs,
            "completed_episodes": num_envs,
            "unresolved_episodes": 0,
            "first_terminal_outcome_per_environment": True,
            "termination_term_counts": {
                "object_dropping": 0,
                "excessive_object_force": 0,
                "protected_surface_force": protected_surface_force,
            },
            "first_episode_handover_diagnostics": {
                "transport_retention_diagnostics": {
                    "overall": {
                        "episodes_with_sustained_midair_loss_3_steps": (
                            sustained_losses
                        )
                    }
                }
            },
            "policy_residual_scale": 0.01,
            "policy_giver_residual_axes": ["x", "y"],
            "policy_analytic_vertical_authority": True,
            "policy_receiver_residual_enabled": False,
            "runtime": {
                "source": {"dranmar_revision": "revision-a"}
            },
        },
    )


def _enable_physical_gate(contract: Path) -> None:
    value = json.loads(contract.read_text())
    promotion = value["stages"][0]["promotion"]
    promotion.update(
        {
            "held_out_seed_passes": 1,
            "require_complete_first_terminal_population": True,
            "hard_termination_terms_must_be_zero": [
                "object_dropping",
                "excessive_object_force",
                "protected_surface_force",
            ],
            "require_matching_analytic_baseline": True,
            "candidate_success_must_not_trail_analytic_baseline": True,
            "analytic_baseline_retention_comparison": (
                "strictly_lower_rate_unless_baseline_zero"
            ),
            "require_training_play_source_parity": True,
            "required_policy_contract": {
                "policy_residual_scale": 0.01,
                "policy_giver_residual_axes": ["x", "y"],
                "policy_analytic_vertical_authority": True,
                "policy_receiver_residual_enabled": False,
            },
        }
    )
    _write_json(contract, value)


def test_tqta_stops_only_after_one_checkpoint_passes_all_seed_gates(
    tmp_path: Path,
) -> None:
    module = _load_module()
    started = datetime(2026, 7, 26, 10, 0, tzinfo=timezone.utc)
    contract = _contract(tmp_path / "contract.json")
    tracker = module.new_tracker(
        task="DrAnmar-Test-v0",
        contract_path=contract,
        started_at=started,
    )
    checkpoint = "a" * 64
    training = _training(tmp_path / "training.json", checkpoint)
    play_one = _play(
        tmp_path / "play-one.json",
        checkpoint,
        seed=101,
        success_rate=0.95,
    )
    module.ingest(
        tracker,
        [training, play_one],
        expert_minutes=4.5,
        observed_at=started + timedelta(minutes=5),
    )
    assert tracker["qualification"]["achieved"] is False
    assert tracker["resource_totals"]["gpu_device_hours"] == 0.05
    assert tracker["resource_totals"]["simulated_steps"] == 123456

    play_two = _play(
        tmp_path / "play-two.json",
        checkpoint,
        seed=202,
        success_rate=0.91,
    )
    module.ingest(
        tracker,
        [training, play_two],
        observed_at=started + timedelta(minutes=7),
    )
    assert tracker["qualification"]["achieved"] is True
    assert tracker["qualification"]["checkpoint_sha256"] == checkpoint
    assert tracker["qualification"]["passing_held_out_seeds"] == [101, 202]
    assert tracker["qualification"]["wall_clock_seconds_to_gate"] == 420.0
    assert tracker["resource_totals"]["experiment_count"] == 3
    assert (
        tracker["resource_totals"]["successful_expert_demonstration_minutes"]
        == 4.5
    )
    module.ingest(tracker, [training, play_two])
    assert tracker["resource_totals"]["experiment_count"] == 3


def test_tqta_rejects_changed_contract(tmp_path: Path) -> None:
    module = _load_module()
    contract = _contract(tmp_path / "contract.json")
    tracker = module.new_tracker(
        task="DrAnmar-Test-v0",
        contract_path=contract,
    )
    value = json.loads(contract.read_text())
    value["stages"][0]["promotion"]["minimum_success_rate"] = 0.5
    _write_json(contract, value)
    try:
        module.ingest(tracker, [])
    except ValueError as exc:
        assert "contract changed" in str(exc)
    else:
        raise AssertionError("changed contract was accepted")


def test_tqta_does_not_mix_checkpoints_across_held_out_seeds(
    tmp_path: Path,
) -> None:
    module = _load_module()
    contract = _contract(tmp_path / "contract.json")
    tracker = module.new_tracker(
        task="DrAnmar-Test-v0",
        contract_path=contract,
    )
    checkpoint_a = "a" * 64
    checkpoint_b = "b" * 64
    evidence = [
        _training(tmp_path / "train-a.json", checkpoint_a),
        _training(tmp_path / "train-b.json", checkpoint_b),
        _play(
            tmp_path / "play-a.json",
            checkpoint_a,
            seed=101,
            success_rate=1.0,
        ),
        _play(
            tmp_path / "play-b.json",
            checkpoint_b,
            seed=202,
            success_rate=1.0,
        ),
    ]
    module.ingest(tracker, evidence)
    assert tracker["qualification"]["achieved"] is False


def test_tqta_requires_matching_baseline_and_lower_sustained_slip(
    tmp_path: Path,
) -> None:
    module = _load_module()
    contract = _contract(tmp_path / "contract.json")
    _enable_physical_gate(contract)
    tracker = module.new_tracker(
        task="DrAnmar-Test-v0",
        contract_path=contract,
    )
    checkpoint = "c" * 64
    training = _training(tmp_path / "training.json", checkpoint)
    candidate = _play(
        tmp_path / "candidate.json",
        checkpoint,
        seed=101,
        success_rate=0.95,
        sustained_losses=5,
    )
    module.ingest(tracker, [training, candidate])
    assert tracker["qualification"]["achieved"] is False

    baseline = _play(
        tmp_path / "baseline.json",
        None,
        seed=101,
        success_rate=0.91,
        analytic_only=True,
        sustained_losses=10,
    )
    module.ingest(tracker, [baseline])
    assert tracker["qualification"]["achieved"] is True


def test_tqta_rejects_successful_candidate_with_hard_failure(
    tmp_path: Path,
) -> None:
    module = _load_module()
    contract = _contract(tmp_path / "contract.json")
    _enable_physical_gate(contract)
    tracker = module.new_tracker(
        task="DrAnmar-Test-v0",
        contract_path=contract,
    )
    checkpoint = "d" * 64
    evidence = [
        _training(tmp_path / "training.json", checkpoint),
        _play(
            tmp_path / "baseline.json",
            None,
            seed=101,
            success_rate=0.91,
            analytic_only=True,
            sustained_losses=10,
        ),
        _play(
            tmp_path / "unsafe-candidate.json",
            checkpoint,
            seed=101,
            success_rate=0.99,
            sustained_losses=1,
            protected_surface_force=1,
        ),
    ]
    module.ingest(tracker, evidence)
    assert tracker["qualification"]["achieved"] is False


def test_learning_launcher_exposes_tqta_commands() -> None:
    launcher = (ROOT / "dr_anmar_learning.sh").read_text()
    assert "tqta-start" in launcher
    assert "tqta-ingest" in launcher
    assert "tqta-report" in launcher
    assert "scripts/dr_anmar_tqta.py" in launcher
