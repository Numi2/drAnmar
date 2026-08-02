# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/dr_anmar_persistent_cut_topology.py"
RECEIPT_PATH = ROOT / "physics_next/receipts/cuttable-tissue-persistent-topology-reference.json"


def _module():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    name = "dranmar_test_persistent_cut_topology"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_triangle_box_overlap_is_bounded_and_not_a_center_distance_guess():
    module = _module()
    triangle = np.asarray(((-0.2, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.2, 0.0)))
    extent = np.asarray((0.1, 0.1, 0.1))
    assert module._triangle_box_overlap(triangle, np.zeros(3), extent)
    assert not module._triangle_box_overlap(triangle, np.asarray((0.0, 0.0, 0.25)), extent)


def test_fracture_work_accumulates_but_other_channels_cannot_cut():
    module = _module()
    profile = module.load_profile()
    critical = profile["fracture"]["mode_i_fracture_energy_j_m2"]
    start, end = module._incision_sweeps([(-0.004, 0.0, 0.0), (0.004, 0.0, 0.0)])[0]
    friction = module.PersistentCutCellField(profile)
    friction.apply_sweep(
        start,
        end,
        module.WorkChannels(
            adhesion_j_m2=critical,
            wear_j_m2=critical,
            viscous_j_m2=critical,
            friction_j_m2=critical,
        ),
    )
    assert friction.fracture_event_count == 0

    cumulative = module.PersistentCutCellField(profile)
    first = cumulative.apply_sweep(start, end, module.WorkChannels(fracture_j_m2=0.51 * critical))
    second = cumulative.apply_sweep(start, end, module.WorkChannels(fracture_j_m2=0.51 * critical))
    assert not first
    assert second
    assert cumulative.fracture_event_count == len(second)


def test_reconstruction_creates_equal_opposed_zero_volume_collision_sheets():
    module = _module()
    field, metrics = module._run_reference_topology(module.load_profile())
    mesh = field.reconstruct_wound_surfaces()
    assert metrics["repeat_hash_stable"]
    assert metrics["repeated_events"] == 0
    assert field.intersection_cell_count() > 0
    assert len(mesh.triangles) > 0
    assert np.isclose(mesh.positive_area_m2, mesh.negative_area_m2, atol=1.0e-15)
    assert set(mesh.triangle_sides.tolist()) == {-1, 1}
    triangle = mesh.vertices_m[mesh.triangles[0]]
    center = np.mean(triangle, axis=0)
    normal = mesh.triangle_normals[0]
    radius = 0.25 * float(np.min(field.cell_size))
    force = module.wound_contact_force(mesh, center + 0.5 * radius * normal, radius, 2000.0)
    assert np.linalg.norm(force) > 0.0


def test_persistent_topology_qualifies_replays_and_matches_retained_receipt():
    module = _module()
    first = module.run_persistent_topology_qualification()
    second = module.run_persistent_topology_qualification()
    assert first.qualified, first.failed_gates
    assert first.payload() == second.payload()
    assert first.arbitrary_origin_coverage_fraction == 1.0
    assert first.repeated_path_additional_events == 0
    assert first.intersection_cell_count > 0
    assert first.removed_volume_m3 == 0.0
    assert first.wound_collision_coverage_fraction == 1.0
    retained = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert first.payload() == retained
