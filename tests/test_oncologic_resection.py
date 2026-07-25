from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "source/extensions/orbit.surgical.assets/orbit/surgical/assets"
    / "oncologic_resection.py"
)
ASSET_ROOT = (
    ROOT
    / "source/extensions/orbit.surgical.assets/data"
    / "Props/SurgicalOncology/OncoSurgeryCell"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "oncologic_resection_test_module", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runtime = load_module()


def sensor_readings(timestamp_s: float = 1.0):
    return (
        runtime.SensorReading(
            modality="nir_fluorescence",
            tumor_probability=0.82,
            margin_probability=0.76,
            protected_structure_probability=0.12,
            confidence=0.91,
            timestamp_s=timestamp_s,
            registration_error_m=0.0010,
        ),
        runtime.SensorReading(
            modality="ultrasound",
            tumor_probability=0.78,
            margin_probability=0.72,
            protected_structure_probability=0.16,
            confidence=0.88,
            timestamp_s=timestamp_s + 0.015,
            registration_error_m=0.0015,
        ),
        runtime.SensorReading(
            modality="hyperspectral",
            tumor_probability=0.80,
            margin_probability=0.74,
            protected_structure_probability=0.14,
            confidence=0.86,
            timestamp_s=timestamp_s + 0.025,
            registration_error_m=0.0012,
        ),
    )


def test_asset_contract_and_runtime_paths_are_complete() -> None:
    manifest = json.loads((ASSET_ROOT / "asset_manifest.json").read_text())
    task = json.loads(
        (ASSET_ROOT / "oncologic_resection_task_contract.json").read_text()
    )
    profile = json.loads((ASSET_ROOT / "physics_profile.json").read_text())

    assert manifest["version"] == "0.2.0"
    assert tuple(task["phases"]) == runtime.TASK_PHASES
    assert manifest["joint_count"] == len(runtime.TOOL_JOINTS) == 22
    assert profile["tool"]["authored_mass_kg"] == pytest.approx(2.5534)
    for path in (
        runtime.TOOL_STANDALONE_USD,
        runtime.TOOL_PAYLOAD_USD,
        runtime.TOOL_RIGID_PROXY_USD,
        runtime.LIVER_DEMO_USD,
        runtime.SPECIMEN_BAG_USD,
        runtime.WORKCELL_USD,
    ):
        assert path.is_file()


def test_phase_targets_are_complete_finite_and_within_authored_limits() -> None:
    assert tuple(runtime.PHASE_TARGETS) == runtime.TASK_PHASES
    expected = set(runtime.TOOL_JOINTS.values())
    for phase in runtime.TASK_PHASES:
        targets = runtime.phase_targets(phase)
        assert set(targets) == expected
        assert all(math.isfinite(value) for value in targets.values())
    assert abs(runtime.PHASE_TARGETS["map"]["oct_scan_x_joint"]) <= 0.014
    assert runtime.PHASE_TARGETS["deploy_bag"]["bag_deployment_joint"] <= 0.095
    assert runtime.PHASE_TARGETS["close_bag"]["bag_closure_joint"] <= 0.025


def test_isaac_quaternion_contract_remains_wxyz() -> None:
    assert runtime._wxyz_quaternion((1.0, 0.0, 0.0, 0.0)) == (
        1.0,
        0.0,
        0.0,
        0.0,
    )
    assert runtime._wxyz_quaternion((0.5, 0.5, 0.5, 0.5)) == (
        0.5,
        0.5,
        0.5,
        0.5,
    )
    with pytest.raises(ValueError):
        runtime._wxyz_quaternion((0.0, 0.0, 0.0, 0.0))


def test_registered_frames_cover_all_declared_cameras() -> None:
    assert len(runtime.REGISTERED_CAMERA_FRAMES) == 4
    for name in runtime.REGISTERED_CAMERA_FRAMES:
        assert name in runtime.TOOL_FRAME_PATHS
        assert runtime.frame_path("/World/Tool", name).startswith("/World/Tool/")
    with pytest.raises(KeyError):
        runtime.frame_path("/World/Tool", "invented_sensor")
    contract = runtime.sensor_runtime_contract("/World/Tool")
    assert set(contract["rtx_cameras"]) == set(runtime.REGISTERED_CAMERA_FRAMES)
    assert contract["fusion"]["failure_policy"] == "abstain"
    assert contract["fusion"]["maximum_timestamp_skew_s"] == 0.050


def test_multimodal_fusion_accepts_registered_consistent_samples() -> None:
    result = runtime.MultimodalOncologyFusion().fuse(
        sensor_readings(), reference_time_s=1.025
    )
    assert result.actionable
    assert result.abstention_reason is None
    assert result.modalities == (
        "hyperspectral",
        "nir_fluorescence",
        "ultrasound",
    )
    assert 0.78 <= result.tumor_probability <= 0.82
    assert result.sensor_disagreement < 0.1
    assert result.confidence > 0.55


def test_multimodal_fusion_abstains_on_stale_registration_and_disagreement() -> None:
    fusion = runtime.MultimodalOncologyFusion()
    stale = fusion.fuse(sensor_readings(), reference_time_s=2.0)
    assert not stale.actionable
    assert stale.abstention_reason == "insufficient_registered_modalities"

    disagreement = (
        runtime.SensorReading(
            "nir_fluorescence", 0.95, 0.95, 0.05, 0.95, 1.0, 0.001
        ),
        runtime.SensorReading(
            "ultrasound", 0.10, 0.10, 0.90, 0.95, 1.01, 0.001
        ),
    )
    result = fusion.fuse(disagreement, reference_time_s=1.01)
    assert not result.actionable
    assert result.abstention_reason == "sensor_disagreement"


def test_sensor_reading_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError):
        runtime.SensorReading(
            "oct", 1.2, 0.4, 0.1, 0.8, 1.0, 0.001
        )
    with pytest.raises(ValueError):
        runtime.SensorReading(
            "oct", 0.2, 0.4, 0.1, 0.8, 1.0, -0.001
        )


def test_tumor_field_matches_authored_volume_and_progress_is_monotonic() -> None:
    model = runtime.TumorFieldModel.from_json()
    assert len(model.cells) == 3028
    assert len(model.planned_cell_ids) == 220
    classes = {
        name: sum(cell.tissue_class == name for cell in model.cells.values())
        for name in {"healthy_parenchyma", "infiltrative_halo", "tumor_core"}
    }
    assert classes == {
        "healthy_parenchyma": 2790,
        "infiltrative_halo": 206,
        "tumor_core": 32,
    }
    initial_residual = model.residual_tumor_volume_mm3
    tumor_id = model.tumor_cell_ids[0]
    assert model.remove([tumor_id, tumor_id]) == 1
    assert model.remove([tumor_id]) == 0
    assert model.residual_tumor_volume_mm3 < initial_residual
    assert model.tumor_removed_volume_mm3 == pytest.approx(
        model.cell_volume_mm3
    )


def test_tumor_field_rejects_unknown_cells() -> None:
    model = runtime.TumorFieldModel.from_json()
    with pytest.raises(KeyError):
        model.remove(["cell_does_not_exist"])
    with pytest.raises(KeyError):
        model.ablate(["cell_does_not_exist"])


def test_protected_pedicle_release_is_fail_closed() -> None:
    topology = runtime.ResectionTopologyModel.from_json()
    pedicle = next(
        bond for bond in topology.bonds.values() if bond.seal_required
    )
    with pytest.raises(runtime.SafetyInterlockError):
        topology.release(
            pedicle.id,
            modality=pedicle.recommended_modality,
            mechanical_work_j=1.0,
        )
    assert not pedicle.released
    assert topology.unsafe_attempts == 1


def test_pedicle_seal_requires_force_and_energy_then_allows_division() -> None:
    topology = runtime.ResectionTopologyModel.from_json()
    pedicle = next(
        bond for bond in topology.bonds.values() if bond.seal_required
    )
    with pytest.raises(runtime.SafetyInterlockError):
        topology.seal(
            pedicle.id, compression_force_n=2.0, energy_j=1.0
        )
    with pytest.raises(runtime.SafetyInterlockError):
        topology.seal(
            pedicle.id, compression_force_n=12.0, energy_j=0.001
        )
    assert topology.seal(
        pedicle.id, compression_force_n=12.0, energy_j=1.0
    )
    assert topology.release(
        pedicle.id,
        modality=pedicle.recommended_modality,
        mechanical_work_j=1.0,
    )


def test_non_pedicle_requires_authored_modality_and_threshold() -> None:
    topology = runtime.ResectionTopologyModel.from_json()
    bond = next(
        item for item in topology.bonds.values() if not item.seal_required
    )
    with pytest.raises(runtime.SafetyInterlockError):
        topology.release(
            bond.id,
            modality="seal_divide",
            mechanical_work_j=10.0,
            aspiration_energy_j=10.0,
        )
    with pytest.raises(runtime.SafetyInterlockError):
        topology.release(
            bond.id,
            modality=bond.recommended_modality,
            mechanical_work_j=0.0,
            aspiration_energy_j=0.0,
        )


def test_specimen_workflow_enforces_detachment_containment_and_marking() -> None:
    specimen = runtime.SpecimenWorkflow()
    with pytest.raises(runtime.SafetyInterlockError):
        specimen.capture(specimen_detached=True)
    specimen.deploy()
    with pytest.raises(runtime.SafetyInterlockError):
        specimen.capture(specimen_detached=False)
    specimen.capture(specimen_detached=True)
    specimen.close()
    for marker in runtime.ORIENTATION_MARKERS:
        specimen.mark_orientation(marker)
    assert specimen.orientation_complete
    with pytest.raises(ValueError):
        specimen.mark_orientation("invented")


def test_domain_randomization_is_deterministic_bounded_and_nonclinical() -> None:
    first = runtime.sample_domain_parameters(42)
    second = runtime.sample_domain_parameters(42)
    assert first == second
    assert all(abs(value) <= 0.002 for value in first.registration_bias_m)
    assert 0.75 <= first.tissue_stiffness_scale <= 1.30
    assert 0.80 <= first.tool_friction_scale <= 1.20
    binding = runtime.dynamic_patient_oncology_binding("/World/Patient")
    assert binding["liver_prim"] == "/World/Patient/Anatomy/liver"
    assert binding["tumor_prim"] == "/World/Patient/Anatomy/liver_tumor"
    assert binding["demo_liver_active"] is False
    assert binding["deformable_representation"] == "gpu_volume_tetmesh"
    assert binding["maximum_active_deformable_components"] == 1
    assert binding["shared_ledgers"] == ("blood", "bile")
    assert binding["clinical_validation"] is False


def test_native_volume_route_rejects_proxy_fallback_and_incomplete_paths() -> None:
    valid = runtime._require_native_volume_route(
        {
            "route": "current_explicit_tetmesh_volume_hierarchy",
            "body_prim_path": "/World/Liver/Geometry",
            "simulation_mesh_path": "/World/Liver/Geometry/SimulationTetMesh",
        }
    )
    assert valid["route"] == "current_explicit_tetmesh_volume_hierarchy"
    with pytest.raises(RuntimeError, match="native GPU volume deformable"):
        runtime._require_native_volume_route(
            {"route": "not_applied", "error": "CUDA unavailable"}
        )
    with pytest.raises(RuntimeError, match="omitted runtime paths"):
        runtime._require_native_volume_route(
            {"route": "current_auto_cooked_volume_hierarchy"}
        )


def test_resection_bonds_retain_registered_spatial_contract() -> None:
    topology = runtime.ResectionTopologyModel.from_json()
    for bond in topology.bonds.values():
        assert len(bond.center_m) == 3
        assert len(bond.normal) == 3
        assert all(math.isfinite(value) for value in bond.center_m)
        assert math.sqrt(sum(value * value for value in bond.normal)) == pytest.approx(
            1.0, abs=1.0e-5
        )


def test_episode_rejects_bad_registration_and_out_of_plan_resection() -> None:
    episode = runtime.OncologicResectionEpisode()
    assert episode.advance() == "register"
    episode.register(0.004)
    with pytest.raises(runtime.SafetyInterlockError):
        episode.advance()
    episode.register(0.001)
    assert episode.advance() == "map"
    episode.map_sensors(sensor_readings(), reference_time_s=1.025)
    assert episode.advance() == "plan"
    episode.set_plan()
    assert episode.advance() == "capture"
    outside = next(
        cell_id
        for cell_id in episode.tumor_field.cells
        if cell_id not in episode.planned_cell_ids
    )
    with pytest.raises(runtime.SafetyInterlockError):
        episode.resect_cells([outside])


def test_complete_episode_reaches_final_contract_without_inflated_success() -> None:
    episode = runtime.OncologicResectionEpisode()
    episode.advance()
    episode.register(0.001)
    episode.advance()
    episode.map_sensors(sensor_readings(), reference_time_s=1.025)
    episode.advance()
    episode.set_plan()
    episode.advance()
    episode.confirm_traction_capture()
    episode.advance()
    episode.resect_cells(episode.planned_cell_ids)

    for bond in episode.topology.bonds.values():
        if not bond.seal_required:
            episode.topology.release(
                bond.id,
                modality=bond.recommended_modality,
                mechanical_work_j=1.0,
                aspiration_energy_j=1.0,
            )
    episode.advance()
    for bond in episode.topology.bonds.values():
        if bond.seal_required:
            episode.topology.seal(
                bond.id, compression_force_n=12.0, energy_j=1.0
            )
            episode.topology.release(
                bond.id,
                modality=bond.recommended_modality,
                mechanical_work_j=1.0,
            )
    episode.advance()
    assert episode.topology.specimen_detached
    episode.advance()
    episode.specimen.deploy()
    episode.advance()
    episode.specimen.capture(specimen_detached=True)
    episode.advance()
    episode.specimen.close()
    episode.advance()
    for marker in runtime.ORIENTATION_MARKERS:
        episode.specimen.mark_orientation(marker)
    episode.advance()
    episode.record_cavity_scan(sensor_readings(2.0), reference_time_s=2.025)
    episode.advance()
    residual_ids = [
        cell.id
        for cell in episode.tumor_field.cells.values()
        if cell.tissue_class != "healthy_parenchyma"
        and not cell.removed
        and not cell.ablated
    ]
    episode.resect_cells(residual_ids, corrective=True)
    episode.advance()
    episode.record_losses(blood_ml=1.2, bile_ml=0.05)
    episode.confirm_hemostasis_and_bile_check()
    episode.advance()
    report = episode.finalize()

    assert report["metrics"]["residual_tumor_volume_mm3"] == 0.0
    assert report["metrics"]["protected_structure_injury_count"] == 0
    assert report["metrics"]["orientation_marker_completeness"] == 1.0
    assert report["success_checks"]["specimen_contained"]
    # Removing only the authored plan plus residual tumor is not automatically
    # represented as a validated 10 mm margin. The report must not inflate it.
    assert report["success"] is False
    assert math.isfinite(episode.reward())
    assert len(episode.observation()) == 12
