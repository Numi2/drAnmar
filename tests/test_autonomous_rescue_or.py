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


def test_resource_ledger_rejects_nonfinite_consumption_without_mutation():
    modules = load_rescue_modules()
    _, rescue = modules
    ledger = rescue.ResourceLedger({"blood_products_ml": 1200.0})
    before = dict(ledger.snapshot())

    with pytest.raises(ValueError, match="finite and positive"):
        ledger.consume("blood_products_ml", float("nan"))

    assert dict(ledger.snapshot()) == before
    assert ledger.consumed == {}
