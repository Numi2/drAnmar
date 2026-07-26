from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "config/dranmar_rl_direction.json"
VALIDATOR_PATH = ROOT / "scripts/validate_dranmar_rl_direction.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_dranmar_rl_direction", VALIDATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direction_contract_is_self_consistent() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    document = (
        ROOT / "docs/ROBOTIC_SURGERY_RL_TECHNICAL_DIRECTION.md"
    ).read_text()
    validator = _load_validator()
    assert validator.validate(contract, document) == []


def test_learned_policy_cannot_own_unshielded_low_level_control() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    layers = {
        layer["id"]: layer for layer in contract["policy_architecture"]
    }
    local = layers["local_skill_policy"]
    assert local["output"] == (
        "relative_task_space_target_or_short_action_chunk"
    )
    assert "unshielded_VLA" in local["forbidden_policy_families"]
    assert layers["safety_shield"]["order"] < (
        layers["deterministic_controller"]["order"]
    )


def test_direction_requires_outcome_safety_and_failure_evidence() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    thesis_evidence = set(contract["technical_thesis"]["primary_evidence"])
    metrics = set(contract["evaluation"]["required_metrics"])
    assert {"patient_benefit", "patient_harm", "recovery"}.issubset(
        thesis_evidence
    )
    assert {
        "hard_constraint_violations",
        "failure_distribution_by_stratum",
        "abstention_precision_recall",
    }.issubset(metrics)
    assert contract["technical_thesis"]["reward_is_not_evidence"] is True


def test_program_goal_is_time_to_qualified_task_achievement() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    goal = contract["program_goal"]
    assert goal["objective"] == "minimize_time_to_qualified_task_achievement"
    assert "held_out_competence_safety_and_recovery" in goal["stop_event"]
    assert {
        "gpu_hours_to_gate",
        "simulated_steps_to_gate",
        "successful_expert_demonstration_minutes_to_gate",
        "experiment_count_to_gate",
    }.issubset(set(goal["decomposition"]))
    fastest_path = contract["efficiency_policy"]["fastest_path_loop"]
    assert fastest_path[0]["id"] == "contract_smoke"
    assert fastest_path[-1]["id"] == "transfer_forward"
    assert any(item["id"] == "successive_halving" for item in fastest_path)


def test_generative_models_have_no_physics_or_safety_authority() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    forbidden = set(
        contract["nvidia_stack"]["generative_world_models"][
            "forbidden_authority"
        ]
    )
    assert {
        "contact",
        "force",
        "tissue_state",
        "patient_effects",
        "task_success",
        "safety",
    }.issubset(forbidden)
