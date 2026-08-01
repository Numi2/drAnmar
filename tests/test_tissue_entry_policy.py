# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = (
    ROOT / "source/extensions/orbit.surgical.tasks/orbit/surgical/tasks/"
    "surgical/penetration"
)


def _contract():
    name = "dranmar_test_penetration_contract"
    spec = importlib.util.spec_from_file_location(name, TASK_ROOT / "contract.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _measurement(module, **overrides):
    values = {
        "entry_error_m": 0.0005,
        "tangent_error_deg": 5.0,
        "plane_error_deg": 5.0,
        "indentation_m": 0.0015,
        "embedded_depth_m": 0.0,
        "normal_force_n": 2.0,
        "accumulated_work_j": 0.001,
        "bilateral_custody": True,
        "target_patch_contact": True,
    }
    values.update(overrides)
    return module.PunctureMeasurement(**values)


def test_force_gate_requires_target_alignment_force_depth_and_work():
    module = _contract()
    state = module.PunctureGateState(phase=module.PenetrationPhase.INDENT)
    module.advance_puncture_gate(
        state,
        _measurement(module, normal_force_n=1.99),
        puncture_force_n=2.0,
    )
    assert state.event_count == 0
    module.advance_puncture_gate(
        state,
        _measurement(module, entry_error_m=0.003),
        puncture_force_n=2.0,
    )
    assert state.event_count == 0
    module.advance_puncture_gate(
        state,
        _measurement(module),
        puncture_force_n=2.0,
    )
    assert state.event_count == 1
    assert state.phase == module.PenetrationPhase.PUNCTURE


def test_puncture_event_is_one_shot_and_success_needs_depth_hold():
    module = _contract()
    state = module.PunctureGateState(phase=module.PenetrationPhase.INDENT)
    module.advance_puncture_gate(state, _measurement(module), puncture_force_n=2.0)
    depth = _measurement(module, embedded_depth_m=0.002)
    module.advance_puncture_gate(state, depth, puncture_force_n=2.0)
    for _ in range(9):
        module.advance_puncture_gate(state, depth, puncture_force_n=2.0)
    assert state.event_count == 1
    assert state.stabilized_steps == 10
    assert module.puncture_success(state, depth)


def test_prepuncture_overshoot_is_a_hard_failure():
    module = _contract()
    state = module.PunctureGateState(phase=module.PenetrationPhase.INDENT)
    module.advance_puncture_gate(
        state,
        _measurement(module, normal_force_n=2.51),
        puncture_force_n=2.0,
    )
    assert state.hard_failures == {"prepuncture_force_limit"}
    assert state.event_count == 0


def test_force_model_preserves_compression_cutting_sweep_and_shaft_terms():
    module = _contract()
    before = module.needle_tissue_force_components(
        indentation_m=0.00075,
        embedded_length_m=0.0,
        puncture_force_n=2.0,
        prepuncture_depth_m=0.0015,
        cutting_fraction=0.55,
        shaft_drag_n_per_m=25.0,
        sweep_stiffness_n_m2=40_000.0,
        swept_area_m2=0.0,
    )
    assert before["compression_n"] == 0.5
    assert before["cutting_n"] == 0.0
    after = module.needle_tissue_force_components(
        indentation_m=0.002,
        embedded_length_m=0.002,
        puncture_force_n=2.0,
        prepuncture_depth_m=0.0015,
        cutting_fraction=0.55,
        shaft_drag_n_per_m=25.0,
        sweep_stiffness_n_m2=40_000.0,
        swept_area_m2=1.0e-6,
    )
    assert after["total_n"] == after["cutting_n"] + after["sweep_n"] + after["shaft_friction_n"]


def test_qualification_requires_three_isolated_256_episode_seeds():
    module = _contract()
    result = {
        "episodes": 256,
        "successes": 205,
        "hard_safety_failures": 0,
        "unintended_crossings": 0,
        "entry_rmse_m": 0.0008,
        "tangent_error_deg_max": 9.0,
        "plane_error_deg_max": 9.0,
        "force_overshoot_fraction_max": 0.2,
        "physics_step_ms_p95": 18.0,
        "replay_rmse_m": 0.0004,
        "replay_event_sequence_identical": True,
    }
    decision = module.evaluate_qualification([result, result, result])
    assert decision["qualified"]
    assert decision["episodes"] == 768
    unsafe = dict(result, hard_safety_failures=1)
    assert not module.evaluate_qualification([result, unsafe, result])["qualified"]


def test_paired_baseline_rule_requires_real_improvement():
    module = _contract()
    baseline = {"success_rate": 0.80, "entry_rmse_m": 0.001, "normalized_peak_force": 1.0}
    learned = {"success_rate": 0.80, "entry_rmse_m": 0.00085, "normalized_peak_force": 1.04}
    assert module.learned_policy_beats_baseline(learned, baseline) == (
        True,
        "entry_rmse_improvement",
    )
    unchanged = dict(baseline)
    assert not module.learned_policy_beats_baseline(unchanged, baseline)[0]


def test_task_is_dranmar_owned_and_uses_a_private_backend_provider():
    registration = (TASK_ROOT / "config/needle/__init__.py").read_text(encoding="utf-8")
    assert "DrAnmar-Penetrate-Tissue-Needle-PSM-IK-Rel-v0" in registration
    assert "DrAnmar-Penetrate-Tissue-Needle-PSM-IK-Rel-Play-v0" in registration
    backend = (TASK_ROOT / "backend.py").read_text(encoding="utf-8")
    state = (TASK_ROOT / "mdp/state.py").read_text(encoding="utf-8")
    assert "class DrAnmarTissueEntryBackend(Protocol):" in backend
    assert "create_tissue_entry_backend" in state
    assert "CressimMpmAdapter" not in state


def test_policy_is_gru_128_with_zero_initialized_bounded_residual():
    model = (TASK_ROOT / "residual_model.py").read_text(encoding="utf-8")
    learning = (
        ROOT / "source/extensions/orbit.surgical.tasks/orbit/surgical/tasks/"
        "surgical/learning_cfg.py"
    ).read_text(encoding="utf-8")
    assert "class PenetrationResidualGRUModel(RNNModel):" in model
    assert "nn.init.zeros_(final_linear.weight)" in model
    assert 'rnn_type="gru"' in learning
    assert "rnn_hidden_dim=128" in learning
    assert "residual_scale=0.25" in learning


def test_entry_benchmark_is_explicitly_not_full_suturing():
    benchmark = json.loads(
        (ROOT / "physics_next/benchmarks/needle-entry-v1.json").read_text(encoding="utf-8")
    )
    assert benchmark["scope"] == "single_force_gated_entry_only"
    assert benchmark["engineering_acceptance"]["overall_success_rate_min"] == 0.8
    assert "persistent_tract" in benchmark["excluded"]
    assert benchmark["clinical_validation"] is False


def test_runtime_lock_receipts_the_shared_cressim_artifact():
    lock = json.loads((ROOT / "config/physics-next-lock.json").read_text(encoding="utf-8"))
    build = lock["builds"]["cressim_mpm"]
    assert build["library_relative_path"].endswith("libcrmpm_c_api.so")
    assert build["cuda_architectures"] == "89"
    assert build["tests"] is True
    writer = (ROOT / "scripts/write_dranmar_physics_next_receipt.py").read_text(encoding="utf-8")
    verifier = (ROOT / "scripts/verify_dranmar_physics_next_receipt.py").read_text(encoding="utf-8")
    assert '"cressim_mpm_c_api"' in writer
    assert "CRESSim shared library digest mismatch" in verifier
