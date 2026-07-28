"""Static and pure-Python gates for the v24 RL hardening contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _source(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _load_matrix_module():
    path = REPO_ROOT / "scripts/dr_anmar_rl_efficiency_matrix.py"
    spec = importlib.util.spec_from_file_location(
        "dranmar_efficiency_matrix",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MATRIX = _load_matrix_module()


def _training_evidence(
    *,
    num_envs: int,
    seed: int,
    iterations: int,
    threshold_iteration: int,
) -> dict:
    history = [
        0.5 if index < threshold_iteration else 0.8
        for index in range(1, iterations + 1)
    ]
    return {
        "kind": "training",
        "task": "DrAnmar-Handover-Needle-Frontier-Hardening-v0",
        "num_envs": num_envs,
        "seed": seed,
        "simulated_frames": 13_824_000,
        "rollout_steps_per_env": 64,
        "total_fps": float(num_envs * 20),
        "initial_policy_bundle": {"bound": True},
        "output_policy_bundle": {"contract_sha256": f"{num_envs:064x}"},
        "checkpoint": {
            "sha256": f"{num_envs * 1_000_000 + seed:064x}"
        },
        "success": {
            "history": history,
            "tail_mean": history[-1],
        },
    }


def _qualification_evidence(
    *,
    task: str,
    seed: int,
    checkpoint_sha256: str,
    success_rate: float,
) -> dict:
    return {
        "kind": "held_out_play",
        "task": task,
        "seed": seed,
        "requested_num_envs": 100,
        "completed_episodes": 100,
        "success_rate": success_rate,
        "checkpoint": {"sha256": checkpoint_sha256},
        "termination_term_counts": {
            "excessive_object_force": 0,
            "protected_surface_force": 0,
        },
        "first_episode_handover_diagnostics": {
            "initial_giver_role_population": {
                "robot_1": 50,
                "robot_2": 50,
                "absolute_imbalance": 0,
            }
        },
    }


def test_equal_frame_matrix_is_exact_for_all_environment_counts():
    config = json.loads(
        (
            REPO_ROOT
            / "config/experiments/dranmar_rl_efficiency_matrix.json"
        ).read_text()
    )
    rows = MATRIX._matrix_rows(config)
    assert rows == {600: 360, 1200: 180, 2400: 90}
    assert all(
        count * 64 * iterations == 13_824_000
        for count, iterations in rows.items()
    )
    assert config["gpu_execution_requires_explicit_approval"] is True


def test_matrix_rejects_missing_cells_and_ranks_equal_frame_runs():
    config = json.loads(
        (
            REPO_ROOT
            / "config/experiments/dranmar_rl_efficiency_matrix.json"
        ).read_text()
    )
    evidence = []
    threshold_iterations = {600: 120, 1200: 70, 2400: 50}
    for row in config["environment_counts"]:
        for seed in config["seeds"]:
            document = _training_evidence(
                num_envs=row["num_envs"],
                seed=seed,
                iterations=row["max_iterations"],
                threshold_iteration=threshold_iterations[row["num_envs"]],
            )
            evidence.append(
                (
                    Path(f"{row['num_envs']}-{seed}.json"),
                    document,
                )
            )
    result = MATRIX.analyze_matrix(config, evidence)
    assert result["comparison_basis"] == "equal_simulated_frames"
    assert result["provisional_fastest_num_envs"] in {600, 1200, 2400}

    qualifications = []
    for _, training in evidence:
        checkpoint_sha256 = training["checkpoint"]["sha256"]
        for task, success_rate in (
            (config["qualification"]["task"], 0.80),
            (config["qualification"]["durability_task"], 0.70),
        ):
            qualifications.append(
                (
                    Path(f"{checkpoint_sha256}-{task}.json"),
                    _qualification_evidence(
                        task=task,
                        seed=training["seed"],
                        checkpoint_sha256=checkpoint_sha256,
                        success_rate=success_rate,
                    ),
                )
            )
    qualified = MATRIX.analyze_matrix(
        config,
        evidence,
        qualifications,
    )
    assert qualified["selected_num_envs"] in {600, 1200, 2400}
    assert qualified["promotion_status"] == "qualified_candidate_selected"

    with pytest.raises(MATRIX.MatrixError, match="missing matrix cells"):
        MATRIX.analyze_matrix(config, evidence[:-1])


def test_frontier_task_preserves_terminal_truth_and_uses_potential_delta():
    environment = _source(
        "source/extensions/orbit.surgical.tasks/orbit/surgical/tasks/"
        "surgical/handover/config/needle/e2e_ik_rel_env_cfg.py"
    )
    rewards = _source(
        "source/extensions/orbit.surgical.tasks/orbit/surgical/tasks/"
        "surgical/handover/mdp/rewards.py"
    )
    assert "NeedleHandoverFrontierHardeningEnvCfg" in environment
    assert "self.rewards.success.weight = 80.0" in environment
    assert "terminal_transfer_failure" in environment
    assert "potential_based_handover_progress" in environment
    assert "gamma * next_potential - previous" in rewards
    assert "torch.zeros_like(potential)" in rewards


def test_v24_uses_canonical_geometry_balanced_roles_and_zero_adapter():
    controller = _source(
        "source/extensions/orbit.surgical.tasks/orbit/surgical/tasks/"
        "surgical/handover/residual_model.py"
    )
    model = _source(
        "source/extensions/orbit.surgical.tasks/orbit/surgical/tasks/"
        "surgical/handover/end_to_end_model.py"
    )
    state = _source(
        "source/extensions/orbit.surgical.tasks/orbit/surgical/tasks/"
        "surgical/handover/mdp/state.py"
    )
    assert "canonical_needle_local_frames_enabled" in controller
    assert "return quat_apply(object_orientation, offset)" in controller
    assert "last_giver_custody_quality" in controller
    assert "custody_transport_scale" in controller
    assert "class _FrontierHardeningAdapter" in model
    assert "nn.init.zeros_(self.output.weight)" in model
    assert "frontier_hardening_features" in model
    assert "def assign_balanced_handover_roles(" in state
    assert "_failure_stratified_receiver_sources" in state
    assert "return target_env_ids.to(dtype=torch.long)" in state
    assert "selected = torch.empty(" in state
    assert "if target_offset >= target_env_ids.numel():" in state


def test_durability_is_separate_from_legacy_success_contract():
    environment = _source(
        "source/extensions/orbit.surgical.tasks/orbit/surgical/tasks/"
        "surgical/handover/config/needle/e2e_ik_rel_env_cfg.py"
    )
    registration = _source(
        "source/extensions/orbit.surgical.tasks/orbit/surgical/tasks/"
        "surgical/handover/config/needle/__init__.py"
    )
    assert "required_receiver_only_steps" in environment
    assert '"required_receiver_only_steps"' in environment
    assert "= 60" in environment
    assert "Frontier-Durability-Eval-v0" in registration


def test_native_dranmar_tasks_are_included_in_public_catalog():
    task_catalog = _source(
        "source/extensions/orbit.surgical.tasks/orbit/surgical/tasks/"
        "__init__.py"
    )
    assert 'if task_id.startswith("DrAnmar-")' in task_catalog
    assert "tuple(sorted(set(registered)))" in task_catalog
