from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = (
    ROOT
    / "source/extensions/orbit.surgical.assets"
    / "data/Props/SurgicalTissue/NeedleReadyTissueUnit"
)
RUNTIME_CONTRACT = ROOT / "config/dranmar_needle_ready_tissue.json"
EVIDENCE_ROOT = (
    ROOT / "physics_next/benchmarks/needle-ready-tissue"
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_runtime_contract_locks_the_canonical_asset_and_fail_closed_scope():
    runtime = load_json(RUNTIME_CONTRACT)
    geometry = load_json(ASSET_ROOT / "geometry_contract.json")
    qualification = load_json(ASSET_ROOT / "qualification_contract.json")
    assert runtime["asset"]["catalog_id"] == geometry["id"]
    assert runtime["asset"]["dr_assets_commit"] == (
        "b1155e2577210e913de8fa2c36b2e37692ec43be"
    )
    assert runtime["promotion_boundaries"]["geometry"] is True
    assert runtime["promotion_boundaries"]["intact_newton_vbd"] is True
    for capability in (
        "needle_tissue_puncture",
        "persistent_tract",
        "thread_passage",
        "damage_and_tear",
        "physical_calibration",
        "clinical_validation",
    ):
        assert runtime["promotion_boundaries"][capability] is False
    assert "policy-written puncture flag" in (
        runtime["reward_boundary"]["forbidden_success_shortcuts"]
    )
    assert qualification["policy"] == "fail_closed"


def test_geometry_report_and_lod_hashes_are_consistent():
    report = load_json(ASSET_ROOT / "geometry_report.json")
    contract = load_json(ASSET_ROOT / "geometry_contract.json")
    assert report["lods_point_nested"] is True
    assert report["material_interfaces_conforming"] is True
    assert report["clinical_validation"] is False
    for lod, lod_contract in contract["lods"].items():
        values = report["lods"][lod]
        path = ASSET_ROOT / values["usd"]
        assert values["point_count"] == lod_contract["expected_points"]
        assert (
            values["tetrahedron_count"]
            == lod_contract["expected_tetrahedra"]
        )
        assert sha256(path) == values["usd_sha256"]


def test_newton_and_isaaclab_evidence_preserve_claim_boundaries():
    required = (
        "training-newton.json",
        "contact-newton.json",
        "contact-newton-replay.json",
        "validation-newton.json",
        "training-1200env-newton.json",
        "training-2400env-newton.json",
        "contact-isaaclab-smoke.json",
    )
    for name in required:
        assert (EVIDENCE_ROOT / name).is_file(), name

    for name in (
        "training-newton.json",
        "contact-newton.json",
        "contact-newton-replay.json",
        "validation-newton.json",
        "training-1200env-newton.json",
        "training-2400env-newton.json",
    ):
        evidence = load_json(EVIDENCE_ROOT / name)
        assert evidence["runtime_gate_passed"] is True, name
        assert evidence["metrics"]["finite_state_fraction"] == 1.0, name
        assert evidence["metrics"]["inverted_tetrahedra_peak"] == 0, name
        assert evidence["clinical_validation"] is False, name
        assert evidence["promotion_boundaries"]["puncture"] is False, name
        assert evidence["promotion_boundaries"]["thread_passage"] is False, name

    contact = load_json(EVIDENCE_ROOT / "contact-newton.json")
    replay = load_json(EVIDENCE_ROOT / "contact-newton-replay.json")
    assert replay["deterministic_replay"][
        "exact_final_state_hash_match"
    ] is True
    assert contact["final_state_sha256"] == replay["final_state_sha256"]
    assert contact["metrics"]["contact_candidates_peak"] > 0
    assert contact["metrics"]["geometric_contact_samples"] > 0

    capacity = load_json(
        EVIDENCE_ROOT / "training-2400env-newton.json"
    )
    assert capacity["asset"]["instances"] == 2400
    assert capacity["asset"]["total_particles"] == 1_344_000
    assert capacity["asset"]["total_tetrahedra"] == 4_665_600
    assert capacity["solver"]["contact_probe"] is False
    capacity_contract = load_json(RUNTIME_CONTRACT)["batching"][
        "measured_capacity_lane"
    ]
    assert (
        capacity_contract["physics_step_ms_p50"]
        == capacity["metrics"]["physics_step_ms_p50"]
    )
    assert (
        capacity_contract["physics_step_ms_p95"]
        == capacity["metrics"]["physics_step_ms_p95"]
    )
    assert capacity_contract["measured_gpu_memory_delta_gib"] == (
        capacity["runtime"]["gpu_memory_delta_peak_bytes"] / 2**30
    )

    isaaclab = load_json(EVIDENCE_ROOT / "contact-isaaclab-smoke.json")
    assert isaaclab["spawn_gate_passed"] is True
    assert isaaclab["clinical_validation"] is False
    assert isaaclab["scope"] == "spawn_reset_and_finite_step_only"


def test_batch_benchmark_uses_newton_template_replication():
    source = (
        ROOT / "scripts/dr_anmar_needle_ready_tissue_newton.py"
    ).read_text(encoding="utf-8")
    assert "template_builder.color()" in source
    assert "builder.replicate(" in source
    assert "for instance in range(args.instances):" in source
    replication_block = source[
        source.index("builder.replicate(") : source.index(
            "finalize_started"
        )
    ]
    assert "add_soft_mesh" not in replication_block
