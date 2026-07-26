from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ASSET_MODULES = ROOT / ("source/extensions/orbit.surgical.assets/orbit/surgical/assets")


def load_rescue_modules():
    package_name = "dranmar_rescue_test_package"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ASSET_MODULES)]
    sys.modules[package_name] = package
    loaded = {}
    for module_name in (
        "deformable_rescue",
        "resuscitation_effects",
        "autonomous_rescue_or",
    ):
        qualified = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(
            qualified,
            ASSET_MODULES / f"{module_name}.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
        loaded[module_name] = module
    return loaded["deformable_rescue"], loaded["autonomous_rescue_or"]


def frame(
    runtime,
    step,
    *,
    left=1.8,
    right=1.8,
    retained=0,
    target="rescue_vessel",
    upstream_pressure_mmhg=None,
):
    deformable, _ = runtime
    return deformable.PhysicsEvidenceFrame(
        physics_step=step,
        simulation_time_s=step * 0.05,
        dt_s=0.05,
        station_id="rescue",
        tool_id="adaptive_hemostasis",
        target_id=target,
        left_normal_force_n=left,
        right_normal_force_n=right,
        separation_m=0.0005,
        tool_speed_m_s=0.0,
        target_distance_m=0.0,
        retained_attachment_count=retained,
        measured_upstream_pressure_mmhg=upstream_pressure_mmhg,
    )


def pump_frame(
    runtime,
    step,
    *,
    channel="blood_product",
    position_m,
    reservoir_mass_g,
):
    _, rescue = runtime
    return rescue.PumpEvidenceFrame(
        physics_step=step,
        simulation_time_s=step * 0.05,
        dt_s=0.05,
        channel_id=channel,
        access_attachment_count=1,
        plunger_position_m=position_m,
        downstream_flow_ml_s=600.0,
        reservoir_mass_g=reservoir_mass_g,
        line_pressure_kpa=10.0,
    )


def test_policy_intent_cannot_write_patient_outcomes():
    modules = load_rescue_modules()
    _, rescue = modules
    runtime = rescue.AutonomousRescueORRuntime(seed=7)
    action = rescue.PolicyAction(
        "clip-1",
        "rescue",
        "adaptive_hemostasis",
        "rescue_vessel",
        "clip",
    )
    before = runtime.effects.snapshot().vessel
    runtime.request_action(action)
    after = runtime.effects.snapshot().vessel
    assert dict(before) == dict(after)
    with pytest.raises(ValueError, match="cannot author patient outcomes"):
        runtime.request_action(action, hemostasis_verified=True)


@pytest.mark.parametrize(
    ("action", "reason"),
    (
        (
            ("unknown-station", "gantry", "adaptive_hemostasis", "clip"),
            "unknown_robot_station",
        ),
        (
            ("unknown-tool", "rescue", "mystery_tool", "clip"),
            "unknown_registered_tool",
        ),
        (
            ("wrong-tool", "rescue", "closure_robot", "clip"),
            "tool_lacks_requested_capability",
        ),
        (
            (
                "wrong-system-station",
                "rescue",
                "adaptive_hemostasis",
                "transfuse",
            ),
            "system_intent_requires_system_station",
        ),
    ),
)
def test_policy_action_admission_fails_closed(action, reason):
    modules = load_rescue_modules()
    _, rescue = modules
    runtime = rescue.AutonomousRescueORRuntime(seed=19)
    action_id, station_id, tool_id, requested_action = action

    record = runtime.request_action(
        rescue.PolicyAction(
            action_id,
            station_id,
            tool_id,
            "rescue_vessel",
            requested_action,
        )
    )

    assert record.status is rescue.ActionStatus.REJECTED
    assert record.reason == reason


def test_policy_action_admission_accepts_registered_capability_and_unique_id():
    modules = load_rescue_modules()
    _, rescue = modules
    runtime = rescue.AutonomousRescueORRuntime(seed=23)
    action = rescue.PolicyAction(
        "clip-admitted",
        "rescue",
        "adaptive_hemostasis",
        "rescue_vessel",
        "clip",
    )

    accepted = runtime.request_action(action)
    duplicate = runtime.request_action(action)
    system_action = runtime.request_action(
        rescue.PolicyAction(
            "transfuse-admitted",
            "system",
            "resuscitation_module",
            "patient",
            "transfuse",
        )
    )

    assert accepted.status is rescue.ActionStatus.REQUESTED
    assert duplicate.status is rescue.ActionStatus.REJECTED
    assert duplicate.reason == "duplicate_action_id"
    assert system_action.status is rescue.ActionStatus.REQUESTED


def test_every_protocol_tool_declares_its_requested_capability():
    modules = load_rescue_modules()
    _, rescue = modules
    tools = rescue._load_contract("tools")
    protocols = rescue._load_contract("rescue_protocols")
    tool_capabilities = {
        item["id"]: set(item["capabilities"]) for item in tools["tools"]
    }

    mismatches = []
    for protocol_id, actions in protocols["protocols"].items():
        for action in actions:
            tool_id = action.get("tool")
            if tool_id is None:
                continue
            capability = action["capability"]
            if capability not in tool_capabilities.get(tool_id, set()):
                mismatches.append(
                    f"{protocol_id}/{action['id']}: " f"{tool_id} lacks {capability}"
                )

    assert mismatches == []


def test_unilateral_contact_does_not_create_hemostasis():
    modules = load_rescue_modules()
    deformable, _ = modules
    effects = deformable.ContactDrivenRescueEffects(seed=11)
    adapter = effects.create_scene_adapter()
    adapter.publish(frame(modules, 1, left=2.0, right=0.0))
    vessel = effects.snapshot().vessel
    assert vessel["transient_compression_fraction"] == 0.0
    assert vessel["residual_flow_ml_s"] > 0.0
    assert vessel["hemostasis_verified"] is False


def test_retained_contact_effect_survives_pressure_challenge():
    modules = load_rescue_modules()
    _, rescue = modules
    runtime = rescue.AutonomousRescueORRuntime(seed=3)
    verify = rescue.PolicyAction(
        "verify-1",
        "assessment",
        "perfusion_viability",
        "rescue_vessel",
        "pressure_challenge",
    )
    runtime.request_action(verify)
    for step in range(1, 41):
        runtime.advance_scene(
            frame(
                modules,
                step,
                retained=1,
                upstream_pressure_mmhg=118.0,
            )
        )
    vessel = runtime.effects.snapshot().vessel
    assert vessel["retained_clip_fraction"] == pytest.approx(1.0)
    assert vessel["residual_flow_ml_s"] <= 0.08
    assert vessel["distal_perfusion_fraction"] >= 0.52
    assert vessel["hemostasis_verified"] is True


def test_replayed_physics_frame_is_rejected():
    modules = load_rescue_modules()
    deformable, _ = modules
    effects = deformable.ContactDrivenRescueEffects()
    adapter = effects.create_scene_adapter()
    evidence = frame(modules, 1)
    adapter.publish(evidence)
    with pytest.raises(ValueError, match="strictly increasing"):
        adapter.publish(evidence)


def test_complication_and_rescue_plan_come_from_scene_state():
    modules = load_rescue_modules()
    _, rescue = modules
    runtime = rescue.AutonomousRescueORRuntime(seed=5)
    observation = runtime.advance_scene(frame(modules, 1, left=0.0, right=0.0))
    assert observation["active_complications"][0]["id"] == "catastrophic_hemorrhage"
    assert observation["rescue_plan"]["protocol_id"] == "hemorrhage_control"


def test_repair_effect_requires_geometry_and_retention():
    modules = load_rescue_modules()
    deformable, _ = modules
    effects = deformable.ContactDrivenRescueEffects()
    adapter = effects.create_scene_adapter()
    adapter.publish(
        deformable.PhysicsEvidenceFrame(
            physics_step=1,
            simulation_time_s=0.05,
            dt_s=0.05,
            station_id="primary",
            tool_id="adaptive_anastomosis",
            target_id="bowel_anastomosis",
            left_normal_force_n=1.5,
            right_normal_force_n=1.5,
            separation_m=0.0012,
            tool_speed_m_s=0.0,
            target_distance_m=0.0,
            retained_attachment_count=12,
        )
    )
    repair = effects.snapshot().repairs["bowel_anastomosis"]
    assert repair["approximation_fraction"] == pytest.approx(1.0)
    assert repair["retention_fraction"] == pytest.approx(1.0)
    assert repair["leak_rate_ml_s"] == pytest.approx(0.0)


def test_resource_ledger_rejects_nonfinite_consumption_without_mutation():
    modules = load_rescue_modules()
    _, rescue = modules
    ledger = rescue.ResourceLedger({"blood_products_ml": 1200.0})
    before = dict(ledger.snapshot())

    with pytest.raises(ValueError, match="finite and positive"):
        ledger.consume("blood_products_ml", float("nan"))

    assert dict(ledger.snapshot()) == before
    assert ledger.consumed == {}


def test_resuscitation_inventory_rejection_is_fail_closed():
    modules = load_rescue_modules()
    _, rescue = modules
    runtime = rescue.AutonomousRescueORRuntime(seed=13)
    runtime.advance_resuscitation(
        pump_frame(
            modules,
            1,
            position_m=0.0,
            reservoir_mass_g=500.0,
        )
    )
    runtime.resources.resources["blood_products_ml"] = 0.0
    before = runtime.resuscitation.snapshot()

    with pytest.raises(RuntimeError, match="insufficient blood_products_ml"):
        runtime.advance_resuscitation(
            pump_frame(
                modules,
                2,
                position_m=0.018,
                reservoir_mass_g=468.2,
            )
        )

    after = runtime.resuscitation.snapshot()
    assert after == before
    assert runtime.resources.available("blood_products_ml") == 0.0


def test_resuscitation_withdrawal_conserves_runtime_inventory():
    modules = load_rescue_modules()
    _, rescue = modules
    runtime = rescue.AutonomousRescueORRuntime(seed=17)
    runtime.advance_resuscitation(
        pump_frame(
            modules,
            1,
            position_m=0.0,
            reservoir_mass_g=500.0,
        )
    )
    before = runtime.resources.available("blood_products_ml")
    runtime.advance_resuscitation(
        pump_frame(
            modules,
            2,
            position_m=0.018,
            reservoir_mass_g=468.2,
        )
    )

    channel = runtime.resuscitation.snapshot().channels["blood_product"]
    assert channel["withdrawn_from_reservoir_ml"] == pytest.approx(30.0)
    assert runtime.resources.available("blood_products_ml") == pytest.approx(
        before - 30.0
    )
