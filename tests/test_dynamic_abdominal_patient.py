from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
RUNTIME_PATH = (
    ROOT
    / "source/extensions/orbit.surgical.assets/orbit/surgical/assets"
    / "dynamic_abdominal_patient.py"
)
ASSET_ROOT = (
    ROOT
    / "source/extensions/orbit.surgical.assets/data/Props/Patients"
    / "DynamicAbdominalPatient"
)


def load_runtime():
    spec = importlib.util.spec_from_file_location(
        "dranmar_dynamic_patient_test_runtime", RUNTIME_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runtime = load_runtime()


def test_repository_contract_is_complete_and_registered_once() -> None:
    required_assets = (
        "dranmar_dynamic_abdominal_patient.usda",
        "dranmar_dynamic_abdominal_patient_rigid_proxy.usda",
        "dranmar_dynamic_abdominal_patient_operating_scene.usda",
        "anatomy_manifest.json",
        "patient_runtime_contract.json",
        "mechanics_contract.json",
        "robot_compatibility.json",
        "procedure_scenarios.json",
    )
    assert all((ASSET_ROOT / name).is_file() for name in required_assets)

    anatomy = json.loads((ASSET_ROOT / "anatomy_manifest.json").read_text())
    assert len(anatomy["components"]) == 23
    assert len({component["id"] for component in anatomy["components"]}) == 23

    portfolio = json.loads((ROOT / "physics_next/dr-anmar-assets.json").read_text())
    entries = [
        item
        for item in portfolio["assets"]
        if item["id"] == "dranmar-dynamic-abdominal-patient-v1"
    ]
    assert len(entries) == 1
    assert entries[0]["clinical_validation"] is False

    from dr_anmar_procedures import PROCEDURES_BY_ID

    room = PROCEDURES_BY_ID["dr-anmar-dynamic-abdominal-patient"]
    assert room["dynamic_abdominal_patient"] is True
    assert room["dynamic_patient_access_state"] == "open"
    assert room["nvidia_native_bench"] is True
    assert room["hide_anatomy"] is True

    profile = json.loads(
        (
            ROOT
            / "physics_next/dynamic-patient"
            / "dranmar-dynamic-abdominal-patient-v1.json"
        ).read_text()
    )
    assert profile["deployment"]["doctor_room_id"] == room["id"]
    assert profile["deployment"]["mechanics_policy"].startswith("fail_closed")


@pytest.mark.parametrize("condition", sorted(runtime.VALID_CONDITIONS))
def test_every_condition_remains_finite_and_serializable(condition: str) -> None:
    patient = runtime.DynamicSurgicalPatient(condition=condition)
    for _ in range(100):
        patient.step(0.1)
    snapshot = patient.snapshot()
    json.dumps(snapshot, allow_nan=False)
    assert math.isfinite(snapshot["vital_signs"]["mean_arterial_pressure_mmhg"])
    assert 0.0 <= snapshot["vital_signs"]["spo2_fraction"] <= 1.0
    assert 0.0 <= snapshot["fluid_balance"]["intravascular_volume_ml"]


def test_public_inputs_reject_nonfinite_or_impossible_values() -> None:
    patient = runtime.DynamicSurgicalPatient()
    with pytest.raises(ValueError, match="positive and finite"):
        patient.step(float("nan"))
    with pytest.raises(ValueError, match="positive and finite"):
        patient.start_bleeding(
            "bad",
            "major_vessels",
            vessel_radius_m=0.0,
            injury_fraction=0.5,
            kind="arterial",
        )
    with pytest.raises(ValueError, match="between 0 and 1"):
        patient.start_bleeding(
            "bad",
            "major_vessels",
            vessel_radius_m=0.001,
            injury_fraction=1.1,
            kind="arterial",
        )
    with pytest.raises(ValueError, match="non-negative"):
        patient.fluid_balance.infuse_crystalloid(-1.0)


def test_fluid_ledgers_are_bounded_and_duplicate_bleeds_are_rejected() -> None:
    balance = runtime.FluidBalanceModel(
        baseline_blood_volume_ml=100.0,
        intravascular_volume_ml=100.0,
    )
    balance.lose_blood(150.0)
    assert balance.intravascular_volume_ml == 0.0
    assert balance.cumulative_blood_loss_ml == 100.0

    balance.add_irrigation(10.0)
    balance.recover_irrigation(15.0)
    assert balance.irrigation_recovered_ml == 10.0

    patient = runtime.DynamicSurgicalPatient()
    patient.start_bleeding(
        "source",
        "major_vessels",
        vessel_radius_m=0.001,
        kind="arterial",
    )
    with pytest.raises(ValueError, match="already exists"):
        patient.start_bleeding(
            "source",
            "major_vessels",
            vessel_radius_m=0.001,
            kind="arterial",
        )


def test_damage_adhesion_and_hemostasis_transitions_are_consistent() -> None:
    patient = runtime.DynamicSurgicalPatient(condition="dense_adhesions")
    adhesion_state = patient.tissue_state.get("adhesions")
    assert "adhesion_03" in adhesion_state.active_adhesions
    patient.robot.dissection(target="adhesion_03", method="hydrodissection")
    assert "adhesion_03" not in adhesion_state.active_adhesions
    assert "adhesion_03" in patient.released_adhesions

    vessel = patient.tissue_state.get("major_vessels")
    patient.puncture("major_vessels", severity=0.2)
    assert vessel.punctures == 1
    assert vessel.cuts == 0

    patient.start_bleeding(
        "control_target",
        "major_vessels",
        vessel_radius_m=0.0015,
        injury_fraction=0.8,
        kind="arterial",
    )
    patient.step(0.1)
    before = patient.bleeding.total_flow_ml_s
    event = patient.robot.hemostasis(
        source_id="control_target",
        method="clip",
        effectiveness=0.99,
    )
    patient.step(0.1)
    assert patient.bleeding.total_flow_ml_s < before
    assert event.result["controlled_fraction"] == pytest.approx(0.99)


def test_reset_and_scenario_orchestration_restore_episode_contract() -> None:
    patient = runtime.DynamicSurgicalPatient(
        condition="bowel_ischemia",
        procedure_stage="access_open",
    )
    patient.set_procedure_stage("dressed")
    patient.step(0.1)
    patient.reset()
    assert patient.condition == "bowel_ischemia"
    assert patient.procedure_stage == "access_open"
    assert patient.time_s == 0.0

    orchestrator = runtime.ProcedureOrchestrator(patient)
    with pytest.raises(RuntimeError, match="begin"):
        orchestrator.mark_step("open_abdominal_layers")
    scenario = orchestrator.begin("laparotomy_cholecystectomy")
    first_step = scenario["steps"][0]
    orchestrator.mark_step(first_step)
    orchestrator.mark_step(first_step)
    assert orchestrator.completed_steps == [first_step]
    with pytest.raises(ValueError, match="not part"):
        orchestrator.mark_step("invented_step")


def test_codeless_usd_apis_are_applied_only_when_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRegistry:
        def FindAppliedAPIPrimDefinition(self, name: str):
            return object() if name == "AvailableAPI" else None

    class FakeUsd:
        @staticmethod
        def SchemaRegistry():
            return FakeRegistry()

    class FakePrim:
        def __init__(self):
            self.applied: list[str] = []

        def ApplyAPI(self, name: str) -> None:
            self.applied.append(name)

    monkeypatch.setitem(sys.modules, "pxr", SimpleNamespace(Usd=FakeUsd))
    prim = FakePrim()

    assert runtime._apply_registered_api(prim, "AvailableAPI") is True
    assert runtime._apply_registered_api(prim, "MissingAPI", required=False) is False
    assert prim.applied == ["AvailableAPI"]
    with pytest.raises(RuntimeError, match="Required USD API schema"):
        runtime._apply_registered_api(prim, "MissingAPI")


def test_patient_collision_group_filters_only_internal_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRelationship:
        def __init__(self):
            self.targets: list[str] = []

        def AddTarget(self, path: str) -> None:
            self.targets.append(str(path))

    class FakeAttribute:
        def __init__(self):
            self.value = None

        def Set(self, value: str) -> None:
            self.value = value

    class FakePrim:
        def __init__(self, path: str, valid: bool = True):
            self.path = path
            self.valid = valid

        def IsValid(self) -> bool:
            return self.valid

    class FakeStage:
        def __init__(self):
            self.prims = {
                "/World/envs/env_0/Patient": FakePrim("/World/envs/env_0/Patient")
            }

        def GetPrimAtPath(self, path: str) -> FakePrim:
            return self.prims.get(str(path), FakePrim(str(path), valid=False))

    class FakeCollisionGroup:
        def __init__(self, path: str):
            self.prim = FakePrim(path)
            self.filtered = FakeRelationship()

        def GetPrim(self) -> FakePrim:
            return self.prim

        def CreateFilteredGroupsRel(self) -> FakeRelationship:
            return self.filtered

    class FakeCollisionGroupSchema:
        group: FakeCollisionGroup | None = None

        @classmethod
        def Define(cls, _stage: FakeStage, path: str) -> FakeCollisionGroup:
            cls.group = FakeCollisionGroup(str(path))
            return cls.group

    class FakeCollection:
        instance: "FakeCollection | None" = None

        def __init__(self):
            self.expansion = FakeAttribute()
            self.includes = FakeRelationship()

        @classmethod
        def Apply(cls, _prim: FakePrim, _name: str) -> "FakeCollection":
            cls.instance = cls()
            return cls.instance

        def CreateExpansionRuleAttr(self) -> FakeAttribute:
            return self.expansion

        def CreateIncludesRel(self) -> FakeRelationship:
            return self.includes

    fake_pxr = SimpleNamespace(
        Sdf=SimpleNamespace(Path=lambda value: value),
        Usd=SimpleNamespace(
            Tokens=SimpleNamespace(expandPrims="expandPrims"),
            CollectionAPI=FakeCollection,
        ),
        UsdPhysics=SimpleNamespace(CollisionGroup=FakeCollisionGroupSchema),
    )
    monkeypatch.setitem(sys.modules, "pxr", fake_pxr)

    result = runtime.configure_patient_internal_collision_filter(
        "/World/envs/env_0/Patient",
        stage=FakeStage(),
    )

    group_path = "/World/envs/env_0/DynamicAbdominalPatientInternalCollisionGroup"
    assert result["collision_group_path"] == group_path
    assert FakeCollection.instance is not None
    assert FakeCollection.instance.includes.targets == ["/World/envs/env_0/Patient"]
    assert FakeCollection.instance.expansion.value == "expandPrims"
    assert FakeCollisionGroupSchema.group is not None
    assert FakeCollisionGroupSchema.group.filtered.targets == [group_path]
