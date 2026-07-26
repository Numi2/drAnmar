#!/usr/bin/env python3
"""Validate the adopted Dr.Anmar robotic-surgery learning direction."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config/dranmar_rl_direction.json"
DOCUMENT = ROOT / "docs/ROBOTIC_SURGERY_RL_TECHNICAL_DIRECTION.md"


def _ordered_ids(items: list[dict[str, Any]]) -> list[str]:
    expected = list(range(1, len(items) + 1))
    observed = [int(item.get("order", -1)) for item in items]
    if observed != expected:
        raise ValueError(f"non-contiguous order: {observed}; expected {expected}")
    return [str(item.get("id", "")) for item in items]


def validate(contract: dict[str, Any], document: str) -> list[str]:
    failures: list[str] = []

    thesis = contract.get("technical_thesis", {})
    if thesis.get("target") != "bounded_procedural_autonomy":
        failures.append("technical target must be bounded procedural autonomy")
    if thesis.get("end_to_end_autonomous_surgeon") is not False:
        failures.append("end-to-end autonomous surgeon direction must remain false")
    if thesis.get("reward_is_not_evidence") is not True:
        failures.append("reward must not be accepted as evidence")
    if thesis.get("patient_effects_are_physics_owned") is not True:
        failures.append("patient effects must remain physics-owned")
    required_evidence = {
        "patient_benefit",
        "patient_harm",
        "constraint_violations",
        "failure_distribution",
        "recovery",
        "abstention",
    }
    if not required_evidence.issubset(set(thesis.get("primary_evidence", []))):
        failures.append("primary evidence omits benefit, harm, safety, or recovery")

    goal = contract.get("program_goal", {})
    if goal.get("objective") != "minimize_time_to_qualified_task_achievement":
        failures.append("program goal must minimize TQTA")
    if goal.get("start_event") != "versioned_task_contract_is_frozen":
        failures.append("TQTA must start from a frozen task contract")
    if goal.get("stop_event") != (
        "first_frozen_checkpoint_passes_held_out_competence_safety_and_recovery"
    ):
        failures.append("TQTA must stop only at qualified task achievement")
    required_tqta_components = {
        "gpu_hours_to_gate",
        "simulated_steps_to_gate",
        "successful_expert_demonstration_minutes_to_gate",
        "experiment_count_to_gate",
        "human_intervention_minutes_to_gate",
    }
    if not required_tqta_components.issubset(set(goal.get("decomposition", []))):
        failures.append("TQTA decomposition is incomplete")

    try:
        layers = _ordered_ids(contract.get("policy_architecture", []))
    except ValueError as exc:
        failures.append(f"policy architecture {exc}")
        layers = []
    expected_layers = [
        "synchronized_sensing",
        "state_estimation",
        "procedure_intent",
        "local_skill_policy",
        "safety_shield",
        "deterministic_controller",
        "outcome_and_runtime_monitor",
    ]
    if layers != expected_layers:
        failures.append(f"policy architecture mismatch: {layers}")

    local_policy = next(
        (
            layer
            for layer in contract.get("policy_architecture", [])
            if layer.get("id") == "local_skill_policy"
        ),
        {},
    )
    forbidden_low_level = set(local_policy.get("forbidden_policy_families", []))
    if not {"unshielded_VLM", "unshielded_VLA"}.issubset(forbidden_low_level):
        failures.append("unshielded VLM/VLA must be forbidden at low level")
    if local_policy.get("output") != (
        "relative_task_space_target_or_short_action_chunk"
    ):
        failures.append("local policy output must remain relative and bounded")

    required_algorithms = {
        "rsl_rl_ppo",
        "recurrent_behavior_cloning",
        "act",
        "diffusion_policy",
        "residual_rl",
        "control_barrier_function",
        "classical_servoing_and_planning",
    }
    algorithms = {
        item.get("id") for item in contract.get("algorithm_portfolio", [])
    }
    missing_algorithms = sorted(required_algorithms - algorithms)
    if missing_algorithms:
        failures.append(f"algorithm portfolio missing: {missing_algorithms}")

    try:
        tracks = _ordered_ids(contract.get("learning_tracks", []))
    except ValueError as exc:
        failures.append(f"learning tracks {exc}")
        tracks = []
    if not tracks or tracks[0] != "motor_foundations":
        failures.append("motor foundations must remain the first learning track")
    motor_track = next(
        (
            track
            for track in contract.get("learning_tracks", [])
            if track.get("id") == "motor_foundations"
        ),
        {},
    )
    if motor_track.get("existing_contract") != "config/dranmar_learning_path.json":
        failures.append("direction must compose with the active learning path")

    efficiency = contract.get("efficiency_policy", {})
    if efficiency.get("objective") != "minimum_time_to_qualified_task_achievement":
        failures.append("efficiency policy must optimize minimum TQTA")
    try:
        fastest_path = _ordered_ids(efficiency.get("fastest_path_loop", []))
    except ValueError as exc:
        failures.append(f"fastest-path loop {exc}")
        fastest_path = []
    required_fastest_path = [
        "contract_smoke",
        "expert_bootstrap",
        "fixed_budget_short_race",
        "successive_halving",
        "failure_targeted_curriculum",
        "residual_refinement",
        "frozen_qualification",
        "transfer_forward",
    ]
    if fastest_path != required_fastest_path:
        failures.append(f"fastest-path loop mismatch: {fastest_path}")

    evaluation = contract.get("evaluation", {})
    required_splits = {
        "train",
        "validation",
        "held_out",
        "stress",
        "counterfactual",
        "intervention_and_failure",
    }
    if not required_splits.issubset(set(evaluation.get("required_splits", []))):
        failures.append("evaluation splits are incomplete")
    try:
        gates = _ordered_ids(evaluation.get("promotion_gates", []))
    except ValueError as exc:
        failures.append(f"promotion gates {exc}")
        gates = []
    for gate_id in (
        "contract",
        "competence",
        "physical_behavior",
        "robustness",
        "safety",
        "recovery",
        "efficiency",
        "claim_boundary",
    ):
        if gate_id not in gates:
            failures.append(f"promotion gate missing: {gate_id}")
    safety_gate = next(
        (
            gate
            for gate in evaluation.get("promotion_gates", [])
            if gate.get("id") == "safety"
        ),
        {},
    )
    if safety_gate.get("hard_constraint_violations") != 0:
        failures.append("hard safety violations must be zero at promotion")
    if safety_gate.get("report_statistical_limit") is not True:
        failures.append("zero observed violations must report statistical limits")

    generative = contract.get("nvidia_stack", {}).get(
        "generative_world_models", {}
    )
    forbidden_authority = set(generative.get("forbidden_authority", []))
    if not {
        "contact",
        "force",
        "tissue_state",
        "patient_effects",
        "task_success",
        "safety",
    }.issubset(forbidden_authority):
        failures.append("generative world-model authority boundary is incomplete")

    boundary = contract.get("evidence_boundary", {})
    for key in (
        "simulation_success_is_clinical_validation",
        "physical_calibration_claimed",
        "medical_device_status",
        "patient_care_allowed",
    ):
        if boundary.get(key) is not False:
            failures.append(f"evidence boundary must keep {key}=false")

    sources = contract.get("research_basis", [])
    if len(sources) < 10 or any(
        not isinstance(source, str) or not source.startswith("https://")
        for source in sources
    ):
        failures.append("research basis must contain at least ten HTTPS sources")

    required_document_phrases = (
        "bounded procedural autonomy",
        "time to qualified task achievement",
        "patient benefit, harm, recovery",
        "Reward is a training signal",
        "Isaac Lab Mimic",
        "Control Barrier Function",
        "runtime monitoring",
        "not establish clinical",
    )
    for phrase in required_document_phrases:
        if phrase not in document:
            failures.append(f"direction document missing: {phrase}")

    return failures


def main() -> int:
    try:
        contract = json.loads(CONTRACT.read_text())
        document = DOCUMENT.read_text()
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Dr.Anmar RL direction validation: FAIL\n- {exc}", file=sys.stderr)
        return 1

    failures = validate(contract, document)
    if failures:
        print("Dr.Anmar RL direction validation: FAIL", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(
        "Dr.Anmar RL direction validation: PASS "
        f"({len(contract['policy_architecture'])} policy layers, "
        f"{len(contract['algorithm_portfolio'])} algorithm roles, "
        f"{len(contract['evaluation']['promotion_gates'])} promotion gates, "
        "TQTA objective)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
