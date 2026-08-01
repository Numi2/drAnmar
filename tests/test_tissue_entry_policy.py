# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import json
import sys
import types
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


def _backend():
    name = "dranmar_test_penetration_backend"
    spec = importlib.util.spec_from_file_location(name, TASK_ROOT / "backend.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _through_contract():
    name = "dranmar_test_through_puncture_contract"
    spec = importlib.util.spec_from_file_location(name, TASK_ROOT / "through_contract.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _through_backend():
    package_name = "dranmar_test_through_backend_package"
    package = types.ModuleType(package_name)
    package.__path__ = [str(TASK_ROOT)]
    sys.modules[package_name] = package
    for module_name in ("backend", "through_backend"):
        qualified_name = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(
            qualified_name, TASK_ROOT / f"{module_name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[qualified_name] = module
        spec.loader.exec_module(module)
    return sys.modules[f"{package_name}.through_backend"]


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
        "target_region_valid": True,
        "tissue_contact": True,
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


def test_alignment_enters_indent_before_tissue_contact():
    module = _contract()
    state = module.PunctureGateState(phase=module.PenetrationPhase.ALIGN)
    module.advance_puncture_gate(
        state,
        _measurement(
            module,
            indentation_m=0.0,
            normal_force_n=0.0,
            accumulated_work_j=0.0,
            tissue_contact=False,
        ),
        puncture_force_n=2.0,
    )
    assert state.phase == module.PenetrationPhase.INDENT
    assert state.event_count == 0


def test_alignment_uses_coarse_radius_but_contact_keeps_one_mm_target():
    module = _contract()
    state = module.PunctureGateState(phase=module.PenetrationPhase.ALIGN)
    module.advance_puncture_gate(
        state,
        _measurement(
            module,
            entry_error_m=0.0015,
            indentation_m=0.0,
            normal_force_n=0.0,
            accumulated_work_j=0.0,
            target_region_valid=False,
            tissue_contact=False,
        ),
        puncture_force_n=2.0,
    )
    assert state.phase == module.PenetrationPhase.INDENT


def test_off_target_contact_still_fails_closed():
    module = _contract()
    state = module.PunctureGateState(phase=module.PenetrationPhase.INDENT)
    module.advance_puncture_gate(
        state,
        _measurement(module, target_region_valid=False, tissue_contact=True),
        puncture_force_n=2.0,
    )
    assert state.hard_failures == {"off_target_contact"}


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
    assert "class DrAnmarNativeTissueEntryBackend:" in backend
    assert 'provider="dranmar_native_entry"' in backend
    assert "create_tissue_entry_backend" in state
    assert "CressimMpmAdapter" not in state
    assert "cressim" not in backend.lower()


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
    assert 'std_type="log"' in learning
    assert "desired_tip_position - needle_position" in model
    assert "contact_phase = indent_phase & (indentation > 0.0)" in model
    assert "rotation = torch.where(indent_phase.unsqueeze(-1)" in model
    assert 'kwargs.pop(deprecated_key, None)' in model
    assert "raw = unpad_trajectories(raw, masks)" in model
    assert "residual = torch.tanh(self.mlp(latent))" in model
    assert "self.distribution.update(action_mean)" in model


def test_reset_settling_cannot_advance_authoritative_phase():
    state = (TASK_ROOT / "mdp/state.py").read_text(encoding="utf-8")
    assert "if bool(settled[index]):" in state
    assert "advance_puncture_gate(" in state


def test_play_task_uses_canonical_fixed_domain_materials():
    config = (TASK_ROOT / "penetration_env_cfg.py").read_text(encoding="utf-8")
    events = (TASK_ROOT / "mdp/events.py").read_text(encoding="utf-8")
    assert 'self.events.reset_evidence.params = {"fixed_domain": True}' in config
    assert "state[\"puncture_force_n\"][env_ids] = 2.0" in events


def test_entry_proxy_is_static_and_table_keeps_handover_clearance():
    config = (TASK_ROOT / "penetration_env_cfg.py").read_text(encoding="utf-8")
    proxy = (
        ROOT / "source/extensions/orbit.surgical.assets/data/Props/"
        "SurgicalClosure/Needle/dranmar_needle_entry_proxy.usda"
    ).read_text(encoding="utf-8")
    assert "dranmar_needle_entry_proxy.usda" in config
    assert 'pos=(0.0, 0.0, -0.457)' in config
    assert proxy.count("physics:collisionEnabled = false") == 48
    assert "physics:collisionEnabled = true" not in proxy


def test_receipt_discloses_pregrasp_coupling_and_simulator_evidence():
    module = _contract()
    fields = module.PunctureReceipt.__dataclass_fields__
    assert fields["custody_model"].default == "pregrasped_pose_coupling"
    assert fields["rigid_needle_collisions_enabled"].default is False
    assert fields["evidence_level"].default == "simulator_engineering_only"


def test_entry_benchmark_is_explicitly_not_full_suturing():
    benchmark = json.loads(
        (ROOT / "physics_next/benchmarks/needle-entry-v1.json").read_text(encoding="utf-8")
    )
    assert benchmark["scope"] == "single_force_gated_entry_only"
    assert benchmark["engineering_acceptance"]["overall_success_rate_min"] == 0.8
    assert "persistent_tract" in benchmark["excluded"]
    assert benchmark["clinical_validation"] is False


def test_entry_policy_has_no_cressim_runtime_or_lock_dependency():
    lock = json.loads((ROOT / "config/physics-next-lock.json").read_text(encoding="utf-8"))
    assert "cressim_mpm" not in lock.get("sources", {})
    assert "cressim_mpm" not in lock.get("builds", {})
    assert not (TASK_ROOT / "cressim.py").exists()
    probe = (ROOT / "scripts/probe_dranmar_tissue_entry_backend.py").read_text(
        encoding="utf-8"
    )
    assert "--library" not in probe


def test_native_backend_blocks_indents_and_switches_representation_once():
    module = _backend()
    backend = module.DrAnmarNativeTissueEntryBackend(1)
    outside = module.NeedlePose((0.0, 0.0, 0.004), (0.0, 0.0, 0.0, 1.0))
    contact = module.NeedlePose(
        (0.0, 0.0, 0.002),
        (0.0, 0.0, 0.0, 1.0),
        linear_velocity=(0.0, 0.0, -0.001),
    )
    assert backend.step([outside], [outside], [False])[0].force_n == (0.0, 0.0, 0.0)
    assert backend.step([contact], [contact], [False])[0].force_n[2] > 0.0
    assert backend.scene_state[0].representation == "tip"
    switched = backend.step([contact], [contact], [True])[0]
    backend.step([contact], [contact], [True])
    assert backend.scene_state[0].representation == "arc"
    assert backend.scene_state[0].representation_switch_count == 1
    assert any(abs(value) > 0.0 for value in switched.torque_nm)


def test_native_backend_reset_and_nonfinite_state_fail_closed():
    module = _backend()
    backend = module.DrAnmarNativeTissueEntryBackend(1)
    contact = module.NeedlePose((0.0, 0.0, 0.002), (0.0, 0.0, 0.0, 1.0))
    backend.step([contact], [contact], [True])
    backend.step([contact], [contact], [False])
    assert backend.scene_state[0].representation_switch_count == 0
    invalid = module.NeedlePose((0.0, 0.0, float("nan")), (0.0, 0.0, 0.0, 1.0))
    wrench = backend.step([invalid], [invalid], [False])[0]
    assert backend.scene_state[0].finite is False
    assert all(value != value for value in wrench.force_n)


def test_rsl_rl_training_fails_closed_on_native_analytical_gate():
    training = (ROOT / "source/standalone/workflows/rsl_rl/train.py").read_text(
        encoding="utf-8"
    )
    assert "_require_tissue_entry_gate()" in training
    assert 'receipt.get("qualified_for_ppo") is True' in training
    assert 'receipt.get("representation_switch_count") == 1' in training
    assert 'receipt.get("backend_implementation_sha256") == source_sha256' in training
    assert "except Exception:" in training
    assert "os._exit(1)" in training
    assert "_runner_cfg_for_installed_rsl_rl(agent_cfg)" in training
    assert "env_cfg.seed = args_cli.seed" in training


def test_isolated_entry_evaluator_resets_recurrent_state_and_reports_physics():
    evaluation = (
        ROOT / "scripts/evaluate_dranmar_tissue_entry_policy.py"
    ).read_text(encoding="utf-8")
    assert "policy.reset(dones)" in evaluation
    assert 'termination_manager.get_term("success")' in evaluation
    assert '"hard_safety_failures": hard_failures' in evaluation
    assert '"exactly_one_event_per_success"' in evaluation
    assert '"normalized_peak_force_mean"' in evaluation


def test_through_puncture_requires_one_entry_one_exit_and_twenty_percent_exposure():
    module = _through_contract()
    thresholds = module.ThroughPunctureThresholds()
    values = {
        "entry_error_m": 0.0005,
        "exit_error_m": 0.0005,
        "tangent_error_deg": 5.0,
        "plane_error_deg": 5.0,
        "indentation_m": 0.0015,
        "embedded_arc_length_m": 0.0,
        "exposed_arc_length_m": 0.0,
        "exposed_fraction": 0.0,
        "normal_force_n": 2.0,
        "accumulated_work_j": 0.001,
        "bilateral_custody": True,
        "target_region_valid": True,
        "tissue_contact": True,
    }
    state = module.ThroughPunctureGateState(
        phase=module.ThroughPuncturePhase.INDENT
    )
    module.advance_through_puncture_gate(
        state, module.ThroughPunctureMeasurement(**values), puncture_force_n=2.0
    )
    assert state.entry_event_count == 1
    assert state.phase == module.ThroughPuncturePhase.PUNCTURE
    values["embedded_arc_length_m"] = 0.004
    module.advance_through_puncture_gate(
        state, module.ThroughPunctureMeasurement(**values), puncture_force_n=2.0
    )
    assert state.phase == module.ThroughPuncturePhase.DRIVE
    values.update(
        backend_exit_count=1,
        exposed_arc_length_m=0.0002,
        exposed_fraction=0.01,
    )
    module.advance_through_puncture_gate(
        state, module.ThroughPunctureMeasurement(**values), puncture_force_n=2.0
    )
    assert state.exit_event_count == 1
    assert state.phase == module.ThroughPuncturePhase.EXIT
    values.update(exposed_arc_length_m=0.0045, exposed_fraction=0.205)
    measurement = module.ThroughPunctureMeasurement(**values)
    module.advance_through_puncture_gate(state, measurement, puncture_force_n=2.0)
    for _ in range(thresholds.presentation_steps - 1):
        module.advance_through_puncture_gate(state, measurement, puncture_force_n=2.0)
    assert module.through_puncture_success(state, measurement, thresholds)


def test_through_puncture_backend_and_task_are_distinct_from_qualified_entry():
    backend = (TASK_ROOT / "through_backend.py").read_text(encoding="utf-8")
    registration = (TASK_ROOT / "config/needle/__init__.py").read_text(
        encoding="utf-8"
    )
    controller = (TASK_ROOT / "residual_model.py").read_text(encoding="utf-8")
    assert 'DRANMAR_NATIVE_THROUGH_REVISION = "dranmar-native-tissue-through-v1"' in backend
    assert "through_sample_count = 129" in backend
    assert "exit_event_count" in backend
    assert "DrAnmar-Through-Puncture-Tissue-Needle-PSM-IK-Rel-v0" in registration
    assert "class ThroughPunctureAnalyticController" in controller
    assert "target_exposed_fraction = 0.22" in controller


def test_through_backend_emits_one_exit_and_measures_trailing_arc_exposure():
    module = _through_backend()
    backend = module.DrAnmarNativeTissueThroughBackend(1)
    root = module.NeedlePose(
        (0.0, 0.0, 0.002),
        (2**-0.5, 0.0, 0.0, 2**-0.5),
    )
    backend.step([root], [root], [True])
    first = backend.scene_state[0]
    backend.step([root], [root], [True])
    second = backend.scene_state[0]
    assert first.exit_event_count == 1
    assert second.exit_event_count == 1
    assert first.exposed_arc_length_m > 0.0
    assert first.embedded_arc_length_m > 0.0
    assert 0.0 < first.exposed_fraction < 1.0


def test_through_puncture_receipt_reports_grippable_exposure_and_exit_error():
    module = _through_contract()
    fields = module.ThroughPunctureReceipt.__dataclass_fields__
    assert "exit_error_m" in fields
    assert "exposed_arc_length_m" in fields
    assert "exposed_fraction" in fields
    assert fields["evidence_level"].default == "simulator_engineering_only"


def test_through_puncture_benchmark_requires_a_grippable_twenty_percent_exit():
    benchmark = json.loads(
        (
            ROOT
            / "physics_next/benchmarks/needle-through-puncture-v1.json"
        ).read_text(encoding="utf-8")
    )
    acceptance = benchmark["engineering_acceptance"]
    assert acceptance["entry_event_count"] == 1
    assert acceptance["exit_event_count"] == 1
    assert acceptance["exposed_fraction_min"] == 0.2
    assert acceptance["exposed_arc_length_m_min"] == 0.0044
    assert "second_psm_pullout" in benchmark["excluded"]
