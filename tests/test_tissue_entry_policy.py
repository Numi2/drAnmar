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


def _pullout_contract():
    name = "dranmar_test_pullout_contract"
    spec = importlib.util.spec_from_file_location(name, TASK_ROOT / "pullout_contract.py")
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


def test_unintended_robot_tissue_contact_fails_closed():
    module = _contract()
    state = module.PunctureGateState(phase=module.PenetrationPhase.INDENT)
    module.advance_puncture_gate(
        state,
        _measurement(module, unintended_robot_contact=True),
        puncture_force_n=2.0,
    )
    assert state.hard_failures == {"unintended_robot_tissue_contact"}


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
    assert "indent_phase = phase >= 2" in model
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


def test_giver_reset_lifts_base_and_targets_the_left_tissue_span():
    config = (TASK_ROOT / "penetration_env_cfg.py").read_text(encoding="utf-8")
    assert "pos=(-0.040, -0.120, 0.140)" in config
    assert "rot=(0.4816338, -0.15930964, 0.0, 0.86177104)" in config
    assert "pos_x=(-0.01257941, -0.01257941)" in config
    assert "Enter the left tissue span 5 mm from the wound edge" in config


def test_native_threaded_needle_keeps_single_rigid_task_abi_and_table_clearance():
    config = (TASK_ROOT / "penetration_env_cfg.py").read_text(encoding="utf-8")
    threaded_asset = (
        ROOT / "source/extensions/orbit.surgical.assets/data/Props/"
        "SurgicalClosure/Needle/dranmar_needle_thread_fem.usda"
    ).read_text(encoding="utf-8")
    rigid_proxy = (
        ROOT / "source/extensions/orbit.surgical.assets/data/Props/"
        "SurgicalClosure/Needle/dranmar_needle_entry_proxy.usda"
    ).read_text(encoding="utf-8")
    assert "dranmar_needle_thread_fem.usda" in config
    assert 'filter_prim_paths_expr=["{ENV_REGEX_NS}/Needle/NeedleRigid"]' in config
    assert "self.scene.needle_thread = DeformableObjectCfg(" in config
    assert 'prim_path="{ENV_REGEX_NS}/Needle"' in config
    assert "spawn=None" in config
    assert 'pos=(0.0, 0.0, -0.457)' in config
    assert "dranmar_needle_entry_proxy.usda" in threaded_asset
    assert threaded_asset.count("OmniPhysicsDeformableBodyAPI") == 1
    assert threaded_asset.count("OmniPhysicsVtxXformAttachment") == 1
    assert 'def Material "NeedleGripPhysics"' in threaded_asset
    assert "physics:staticFriction = 2" in threaded_asset
    assert "physics:dynamicFriction = 1.5" in threaded_asset
    assert (
        "rel material:binding:physics = </DrAnmarNeedle/ThreadMaterial>"
        in threaded_asset
    )
    assert "physics_material=sim_utils.RigidBodyMaterialCfg" not in config
    assert rigid_proxy.count("physics:collisionEnabled = false") == 48
    assert "physics:collisionEnabled = true" not in rigid_proxy

    evaluator = (ROOT / "scripts/evaluate_dranmar_tissue_entry_policy.py").read_text(
        encoding="utf-8"
    )
    assert '"needle_asset_sha256": _sha256(needle_asset)' in evaluator
    assert '"single_rigid_needle_plus_surface_fem_suture"' in evaluator


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
    assert '"receiver_pull_steps_min"' in evaluation


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
        "entry_slab": "left",
        "exit_slab": "right",
        "cross_slab_route_valid": True,
        "backend_right_underside_count": 1,
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
    assert (
        'DRANMAR_NATIVE_THROUGH_REVISION = "dranmar-native-tissue-through-v13-right-underside-gate"'
        in backend
    )
    assert "through_sample_count = 129" in backend
    assert "exit_event_count" in backend
    assert "DrAnmar-Through-Puncture-Tissue-Needle-PSM-IK-Rel-v0" in registration
    assert "class ThroughPunctureAnalyticController" in controller
    assert "target_exposed_fraction = 0.22" in controller


def test_through_backend_requires_one_right_underside_puncture_before_top_exit():
    module = _through_backend()
    backend = module.DrAnmarNativeTissueThroughBackend(1)
    entered = module.NeedlePose(
        (0.0, 0.0, 0.002),
        (2**-0.5, 0.0, 0.0, 2**-0.5),
    )
    reemerged = module.NeedlePose(
        (0.0, 0.0, 0.002),
        (-(2**-0.5), 0.0, 0.0, 2**-0.5),
    )
    entry_tip = module.NeedlePose(
        (-0.007, 0.0, 0.002), entered.quaternion_xyzw
    )
    backend.step([entry_tip], [entered], [True])
    before_exit = backend.scene_state[0]
    underside_tip = module.NeedlePose(
        (0.010, 0.0, -0.0020), (0.0, 0.0, 0.0, 1.0)
    )
    underside_arc = module.NeedlePose(
        (0.010, 0.007, -0.0020), underside_tip.quaternion_xyzw
    )
    backend.step([underside_tip], [underside_arc], [True])
    backend.step([entry_tip], [reemerged], [True])
    first = backend.scene_state[0]
    moved_after_exit = module.NeedlePose(
        (0.002, 0.001, 0.004), reemerged.quaternion_xyzw
    )
    backend.step([entry_tip], [moved_after_exit], [True])
    second = backend.scene_state[0]
    backend._through[0].tip_has_entered = True
    backend.step([entry_tip], [reemerged], [True])
    replay_crossing = backend.scene_state[0]
    assert before_exit.exit_event_count == 0
    assert first.right_underside_event_count == 1
    assert first.exit_event_count == 1
    assert second.exit_event_count == 1
    assert second.exit_position_m == first.exit_position_m
    assert replay_crossing.exit_event_count == 1
    assert first.exposed_arc_length_m > 0.0
    assert first.embedded_arc_length_m > 0.0
    assert 0.0 < first.exposed_fraction < 1.0
    assert first.entry_slab == "left"
    assert first.exit_slab == "right"
    assert first.cross_slab_route_valid


def test_through_backend_rejects_same_slab_or_gap_exit_route():
    module = _through_backend()
    backend = module.DrAnmarNativeTissueThroughBackend(1)
    entered = module.NeedlePose(
        (-0.020, 0.0, 0.002),
        (2**-0.5, 0.0, 0.0, 2**-0.5),
    )
    same_slab_exit = module.NeedlePose(
        (-0.020, 0.0, 0.002),
        (-(2**-0.5), 0.0, 0.0, 2**-0.5),
    )
    backend.step([entered], [entered], [True])
    backend.step([entered], [same_slab_exit], [True])
    state = backend.scene_state[0]
    assert state.exit_event_count == 0
    assert state.entry_slab == "left"
    assert state.exit_slab in {"left", "wound_gap"}
    assert state.invalid_exit_route
    assert not state.cross_slab_route_valid


def test_through_backend_rejects_right_top_exit_without_underside_event():
    module = _through_backend()
    backend = module.DrAnmarNativeTissueThroughBackend(1)
    entered = module.NeedlePose(
        (0.0, 0.0, 0.002),
        (2**-0.5, 0.0, 0.0, 2**-0.5),
    )
    reemerged = module.NeedlePose(
        (0.0, 0.0, 0.002),
        (-(2**-0.5), 0.0, 0.0, 2**-0.5),
    )
    entry_tip = module.NeedlePose(
        (-0.007, 0.0, 0.002), entered.quaternion_xyzw
    )
    backend.step([entry_tip], [entered], [True])
    backend.step([entry_tip], [reemerged], [True])
    state = backend.scene_state[0]
    assert state.right_underside_event_count == 0
    assert state.exit_event_count == 0
    assert state.missing_right_underside_puncture
    assert state.invalid_exit_route


def test_through_backend_authorizes_four_bounded_embedded_tract_regrasps():
    module = _through_backend()
    backend = module.DrAnmarNativeTissueThroughBackend(1)
    pose = module.NeedlePose(
        (-0.009, 0.0, 0.003),
        (0.2588190451, 0.0, 0.0, 0.9659258263),
    )
    tip = module.NeedlePose((-0.006, 0.0, 0.002), pose.quaternion_xyzw)
    backend.step([tip], [pose], [True])
    before = backend.scene_state[0]
    assert before.embedded_arc_length_m >= 0.0015
    assert before.trailing_exposed_arc_length_m >= 0.002
    assert not before.trailing_grasp_over_wound_gap
    assert before.trailing_grasp_position_m[2] >= backend.surface_z_m + 0.0015
    assert backend.request_tract_support([0]) == (True,)
    active = backend.scene_state[0]
    assert active.tract_support_active
    assert active.tract_support_event_count == 1
    backend.release_tract_support([0])
    assert backend.request_tract_support([0]) == (True,)
    backend.release_tract_support([0])
    assert backend.request_tract_support([0]) == (True,)
    backend.release_tract_support([0])
    assert backend.request_tract_support([0]) == (True,)
    backend.release_tract_support([0])
    assert backend.request_tract_support([0]) == (False,)
    released = backend.scene_state[0]
    assert not released.tract_support_active
    assert released.tract_support_event_count == 4


def test_through_backend_bounds_match_authored_fem_flaps():
    module = _through_backend()
    assert module.LEFT_SLAB_X_BOUNDS_M == (-0.035, -0.00155)
    assert module.RIGHT_SLAB_X_BOUNDS_M == (0.00157713832065, 0.035)
    assert module.DrAnmarNativeTissueThroughBackend._classify_slab(0.0) == "wound_gap"


def test_right_fem_flap_has_bounded_exit_tenting_patch():
    state = (TASK_ROOT / "mdp/state.py").read_text(encoding="utf-8")
    assert "exit_influence_depth_m = 0.004" in state
    assert "exit_lift_max_m = 0.0012" in state
    assert 'getattr(item, "exit_event_count", 0)' in state
    assert "positions[..., 2] + exit_lift.unsqueeze(-1) * falloff" in state


def test_fem_tract_load_stretches_and_recoils_through_free_nodes():
    backend = (TASK_ROOT / "backend.py").read_text(encoding="utf-8")
    state = (TASK_ROOT / "mdp/state.py").read_text(encoding="utf-8")
    left = (ROOT / "assets/dr_anmar/tissue/DrAnmarSuturableTissue.left.tet.usda").read_text(
        encoding="utf-8"
    )
    right = (ROOT / "assets/dr_anmar/tissue/DrAnmarSuturableTissue.right.tet.usda").read_text(
        encoding="utf-8"
    )
    assert "contact_position_m" in backend
    assert "patch_radius_m = 0.008" in state
    assert "entry_patch_radius_m = 0.0045" in state
    assert "tract_displacement_max_m = 0.0035" in state
    assert "tract_lateral_displacement_max_m = 0.002" in state
    assert "deformation_falloff.unsqueeze(-1)" in state
    for asset in (left, right):
        assert "float omniphysics:youngsModulus = 80000" in asset
        assert "float physxDeformableMaterial:elasticityDamping = 0.08" in asset


def test_receiver_rotates_about_needle_curvature_center_through_final_clearance():
    state = (TASK_ROOT / "mdp/state.py").read_text(encoding="utf-8")
    controller = (TASK_ROOT / "residual_model.py").read_text(encoding="utf-8")
    assert "pull_rotation_axis_w = -torch.nn.functional.normalize" in state
    assert "torch.linalg.cross(pull_rotation_axis_w, receiver_grasp_radial)" in state
    assert "pull_chord_w = receiver_grasp_radius" in state
    assert 'state["receiver_curve_center_w"] - root_pos' in state
    assert "pull_rotation_r" in state
    assert "late_exit_lift" not in controller
    assert "surface_normal * 0.4" not in controller
    assert 'state["custody_owner"] == 0' in state
    assert 'receiver_held = state["custody_owner"] >= 1' in state


def test_whole_psm_link_tissue_contacts_are_authoritative():
    scene = (TASK_ROOT / "penetration_env_cfg.py").read_text(encoding="utf-8")
    pullout = (TASK_ROOT / "pullout_env_cfg.py").read_text(encoding="utf-8")
    state = (TASK_ROOT / "mdp/state.py").read_text(encoding="utf-8")
    assert "giver_all_links_tissue_contact = ContactSensorCfg" in scene
    assert "receiver_all_links_tissue_contact = ContactSensorCfg" in pullout
    assert 'env, "giver_all_links_tissue_contact"' in state
    assert 'env, "receiver_all_links_tissue_contact"' in state
    assert "torch.maximum(giver_tissue_force, receiver_tissue_force) > 0.02" in state
    assert "solver_position_iteration_count=16" in scene
    assert "solver_position_iteration_count=16" in pullout
    assert "filter_prim_paths_expr" not in scene.split(
        "giver_all_links_tissue_contact = ContactSensorCfg", 1
    )[1].split("giver_tip_tissue_contact", 1)[0]


def test_through_backend_samples_the_rendered_positive_x_semicircle():
    module = _through_backend()
    pose = module.NeedlePose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    points = module.DrAnmarNativeTissueThroughBackend._arc_points(pose)
    assert min(point[0] for point in points) >= -1.0e-12
    assert points[0][1] < 0.0
    assert points[-1][1] > 0.0


def test_policy_uses_authored_grasp_and_negative_x_sharp_tangent():
    events = (TASK_ROOT / "mdp/events.py").read_text(encoding="utf-8")
    state = (TASK_ROOT / "mdp/state.py").read_text(encoding="utf-8")
    assert "NEEDLE_MID_GRASP_POSITION_M = (0.009204923626365" in events
    assert "NEEDLE_TANGENT_LOCAL = (-1.0, 0.0, 0.0)" in state


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
    assert acceptance["entry_slab"] == "left"
    assert acceptance["exit_slab"] == "right"
    assert acceptance["exposed_fraction_min"] == 0.2
    assert acceptance["exposed_arc_length_m_min"] == 0.0044
    assert "second_psm_pullout" in benchmark["excluded"]


def test_pullout_requires_receiver_contact_transfer_and_complete_clearance():
    module = _pullout_contract()
    thresholds = module.PulloutThresholds()
    state = module.PulloutGateState(
        phase=module.PulloutPhase.RECEIVER_APPROACH,
        entry_event_count=1,
        exit_event_count=1,
    )
    values = {
        "entry_error_m": 0.0005,
        "exit_error_m": 0.0005,
        "tangent_error_deg": 5.0,
        "plane_error_deg": 5.0,
        "indentation_m": 0.006,
        "embedded_arc_length_m": 0.015,
        "exposed_arc_length_m": 0.005,
        "exposed_fraction": 0.22,
        "normal_force_n": 1.5,
        "accumulated_work_j": 0.01,
        "bilateral_custody": True,
        "giver_custody": True,
        "receiver_distance_m": 0.003,
        "receiver_bilateral_contact": False,
        "receiver_custody": False,
        "giver_released": False,
        "receiver_curve_rotation_deg": 0.0,
        "receiver_curve_center_error_m": 0.0,
        "target_region_valid": True,
        "tissue_contact": True,
        "backend_exit_count": 1,
        "backend_right_underside_count": 1,
        "entry_slab": "left",
        "exit_slab": "right",
        "cross_slab_route_valid": True,
        "tract_support_event_count": 4,
        "giver_regrasp_stage": 5,
        "giver_regrasp_complete": True,
    }
    measurement = module.PulloutMeasurement(**values)
    drive_state = module.PulloutGateState(
        phase=module.PulloutPhase.DRIVE,
        entry_event_count=1,
    )
    module.advance_pullout_gate(drive_state, measurement, puncture_force_n=2.0)
    assert drive_state.phase == module.PulloutPhase.EXIT
    assert drive_state.exit_error_at_event_m == values["exit_error_m"]
    module.advance_pullout_gate(state, measurement, puncture_force_n=2.0)
    assert state.phase == module.PulloutPhase.RECEIVER_GRASP
    for _ in range(thresholds.receiver_contact_steps):
        values["receiver_bilateral_contact"] = True
        measurement = module.PulloutMeasurement(**values)
        module.advance_pullout_gate(state, measurement, puncture_force_n=2.0)
    assert state.phase == module.PulloutPhase.TRANSFER
    for _ in range(thresholds.transfer_release_steps):
        values.update(
            giver_custody=False,
            receiver_custody=True,
            giver_released=True,
        )
        measurement = module.PulloutMeasurement(**values)
        module.advance_pullout_gate(state, measurement, puncture_force_n=2.0)
    assert state.phase == module.PulloutPhase.PULL
    values.update(
        embedded_arc_length_m=0.0001,
        exposed_arc_length_m=0.0218,
        exposed_fraction=0.995,
        receiver_curve_rotation_deg=135.0,
    )
    measurement = module.PulloutMeasurement(**values)
    for _ in range(thresholds.receiver_pull_steps_min):
        module.advance_pullout_gate(state, measurement, puncture_force_n=2.0)
    assert state.receiver_pull_steps == thresholds.receiver_pull_steps_min
    assert state.phase == module.PulloutPhase.CLEAR
    for _ in range(thresholds.clearance_steps - 1):
        module.advance_pullout_gate(state, measurement, puncture_force_n=2.0)
    assert module.pullout_success(state, measurement, thresholds)


def test_pullout_task_has_dual_psm_controller_and_receipt_contract():
    registration = (TASK_ROOT / "config/needle/__init__.py").read_text(encoding="utf-8")
    scene = (TASK_ROOT / "pullout_env_cfg.py").read_text(encoding="utf-8")
    controller = (TASK_ROOT / "residual_model.py").read_text(encoding="utf-8")
    fields = _pullout_contract().PulloutReceipt.__dataclass_fields__
    assert "DrAnmar-Puncture-Pullout-Tissue-Needle-PSM-IK-Rel-v0" in registration
    assert 'asset_name="robot_receiver"' in scene
    assert "receiver_jaw_1_needle_contact" in scene
    assert "receiver_tip_tissue_contact" in scene
    assert "pos=(0.040, -0.120, 0.140)" in scene
    assert "class PulloutAnalyticController" in controller
    assert "giver_regrasp_guidance" in controller
    assert "tract_regrasp_active" in controller
    assert "normal_advance_limit = 0.10" not in controller
    assert "entry_lateral_delta" in controller
    assert "tangential_error <= 0.0009" in controller
    assert "tangential_limit_m = 0.00005" in controller
    assert "remains authoritative to presentation" in controller
    assert "self.drive_rotation_command = 1.0" in controller
    assert "torch.linalg.cross(current_tangent, start_tangent)" in controller
    assert ") & (phase <= 6)" in controller
    assert "receiver retracting through CLEAR" in controller
    assert "(phase >= 11).unsqueeze(-1)" not in controller
    assert fields["custody_model"].default == (
        "bilateral_force_or_calibrated_geometry_then_receiver_pose_coupling"
    )
    assert "tract_support_event_count" in fields
    assert "giver_regrasp_complete" in fields


def test_tissue_blocks_psms_but_filters_only_the_needle_pair():
    penetration = (TASK_ROOT / "penetration_env_cfg.py").read_text(encoding="utf-8")
    pullout = (TASK_ROOT / "pullout_env_cfg.py").read_text(encoding="utf-8")
    events = (TASK_ROOT / "mdp/events.py").read_text(encoding="utf-8")
    state = (TASK_ROOT / "mdp/state.py").read_text(encoding="utf-8")
    left_tet = (
        ROOT / "assets/dr_anmar/tissue/DrAnmarSuturableTissue.left.tet.usda"
    ).read_text(encoding="utf-8")
    assert 'tissue_left = _make_volume_tissue_flap_cfg(side="left")' in penetration
    assert 'tissue_right = _make_volume_tissue_flap_cfg(side="right")' in penetration
    assert "DeformableObjectCfg(" in penetration
    assert "DrAnmarSuturableTissue.{side}.tet.usda" in penetration
    assert 'def TetMesh "SimulationMesh"' in left_tet
    assert 'def TetMesh "CollisionMesh"' in left_tet
    assert "OmniPhysicsVolumeDeformableSimAPI" in left_tet
    assert "PhysicsCollisionAPI" in left_tet
    assert "physxDeformableBody:solverPositionIterationCount = 24" in left_tet
    assert "physxDeformableBody:selfCollision = true" in left_tet
    assert "float physxCollision:contactOffset = 0.0005" in left_tet
    assert "float physxCollision:restOffset = 0.0001" in left_tet
    assert "reset_and_anchor_tissue_fem" in events
    assert "TISSUE_OUTER_ANCHOR_WIDTH_M = 0.004" in events
    assert "def _couple_fem_contact_patch(" in state
    assert "patch_radius_m = 0.008" in state
    assert "item.surface_displacement_m" in state
    assert "contact_patch" in state
    assert "underside_patch" in state
    assert "contact_patch | underside_patch | exit_patch" in state
    assert "right_underside_event_count" in state
    assert "giver_tip_tissue_contact" in penetration
    assert "collision_enabled=False" not in pullout
    assert "dranmar_needle.usda" not in pullout
    assert "configure_tissue_collision_filter(env)" in events
    assert "Creating a second USD" in events
    assert "no needle/tissue filter is needed" in events
    assert "unintended_robot_contact" in state


def test_analytical_controller_finishes_lateral_alignment_before_fem_contact():
    controller = (TASK_ROOT / "residual_model.py").read_text(encoding="utf-8")
    qualification = (
        ROOT / "scripts/qualify_dranmar_tissue_entry_analytical.py"
    ).read_text(encoding="utf-8")
    assert "hold_entry_standoff" in controller
    assert "tangential_error > 0.0009" in controller
    assert "alignment_tangential_limit_m = self.translation_scale_m" in controller
    assert '"max_fem_displacement_m": max_fem_displacement_m' in qualification


def test_pullout_benchmark_requires_receiver_only_complete_clearance():
    benchmark = json.loads(
        (ROOT / "physics_next/benchmarks/needle-puncture-pullout-v1.json").read_text(
            encoding="utf-8"
        )
    )
    acceptance = benchmark["engineering_acceptance"]
    assert acceptance["receiver_bilateral_contact_steps_min"] == 3
    assert acceptance["receiver_pull_steps_min"] == 270
    assert acceptance["receiver_curve_rotation_deg_min"] == 135.0
    assert acceptance["receiver_curve_center_error_m_max"] == 0.0015
    assert acceptance["embedded_arc_length_m_max"] == 0.0001
    assert acceptance["exposed_fraction_min"] == 0.995
    assert acceptance["giver_released"] is True
