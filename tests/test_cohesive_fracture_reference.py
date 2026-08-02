# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/dr_anmar_cohesive_fracture.py"
RECEIPT_PATH = ROOT / "physics_next/receipts/cuttable-tissue-cohesive-fracture-reference.json"


def _module():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    name = "dranmar_test_cohesive_fracture"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_every_internal_tetrahedral_face_is_a_connected_eligible_interface():
    module = _module()
    profile = module.load_profile()
    points, tetrahedra = module.build_regular_tetrahedral_coupon(profile)
    interfaces, adjacency = module.build_cohesive_interfaces(points, tetrahedra)
    assert len(interfaces) == 1000
    assert len({interface.nodes for interface in interfaces}) == len(interfaces)
    assert all(len(interface.tetrahedra) == 2 for interface in interfaces)
    assert all(interface.area_m2 > 0.0 for interface in interfaces)
    assert module._interface_graph_connected(adjacency)
    assert profile["fracture"]["all_internal_faces_eligible"] is True


def test_cohesive_law_is_energy_based_irreversible_and_compression_safe():
    module = _module()
    profile = module.load_profile()
    law = module.MixedModeCohesiveLaw(profile)
    normal = np.asarray((0.0, 0.0, 1.0))
    state = module.CohesiveState()
    _, final, _, expected = law.envelope(1.0, 0.0)
    work, integrated_expected, _ = module._monotonic_work(law, normal)
    assert np.isclose(integrated_expected, expected)
    assert (
        abs(work - expected) / expected
        <= profile["fracture"]["qualification"]["maximum_energy_relative_error"]
    )
    law.evaluate(final * normal * 1.1, normal, np.zeros(3), 1.0, state, seeded=True)
    assert state.failed
    failed_damage = state.damage
    response = law.evaluate(
        -1.0e-5 * normal,
        normal,
        np.zeros(3),
        1.0,
        state,
        seeded=True,
    )
    assert state.damage == failed_damage
    assert np.dot(response.traction_on_positive_face_pa, normal) > 0.0


def test_damage_cannot_start_without_a_blade_seed():
    module = _module()
    profile = module.load_profile()
    law = module.MixedModeCohesiveLaw(profile)
    normal = np.asarray((0.0, 0.0, 1.0))
    _, final, _, _ = law.envelope(1.0, 0.0)
    state = module.CohesiveState()
    law.evaluate(1.1 * final * normal, normal, np.zeros(3), 1.0, state, seeded=False)
    assert state.damage == 0.0
    assert not state.failed


def test_blade_sweep_seeds_only_a_surface_connected_dynamic_front():
    module = _module()
    profile = module.load_profile()
    points, tetrahedra = module.build_regular_tetrahedral_coupon(profile)
    front = module.BladeSeededCohesiveFront(points, tetrahedra, profile)
    top = float(np.max(points[:, 2]))
    first, first_remote = front.seed_sweep(
        module.ScalpelPose((0.0017, 0.0, top + 0.001)),
        module.ScalpelPose((0.0017, 0.0, top - 0.0015), velocity_m_s=(0.0, 0.0, -0.005)),
    )
    assert first
    assert first_remote == 0
    assert all(front.interfaces[index].touches_top for index in first)
    original = set(front.seeded)
    second, _ = front.seed_sweep(
        module.ScalpelPose((0.0017, 0.0, top - 0.0014)),
        module.ScalpelPose((0.0017, 0.0, top - 0.0035), velocity_m_s=(0.0, 0.0, -0.005)),
    )
    assert second
    assert all(front.adjacency[index].intersection(original) for index in second)

    buried = module.BladeSeededCohesiveFront(points, tetrahedra, profile)
    disconnected, rejected = buried.seed_sweep(
        module.ScalpelPose((0.0, 0.0, top - 0.004)),
        module.ScalpelPose((0.0, 0.0, top - 0.006), velocity_m_s=(0.0, 0.0, -0.005)),
    )
    assert not disconnected
    assert rejected > 0

    static = module.BladeSeededCohesiveFront(points, tetrahedra, profile)
    stationary, _ = static.seed_sweep(
        module.ScalpelPose((0.0, 0.0, top + 0.001)),
        module.ScalpelPose((0.0, 0.0, top - 0.0015)),
    )
    assert not stationary


def test_cohesive_receipt_qualifies_replays_and_matches_retained_evidence():
    module = _module()
    first = module.run_cohesive_fracture_qualification()
    second = module.run_cohesive_fracture_qualification()
    assert first.qualified, first.failed_gates
    assert first.payload() == second.payload()
    assert first.eligible_interface_fraction == 1.0
    assert first.off_grid_blade_seed_coverage_fraction == 1.0
    assert first.maximum_remote_seed_events == 0
    assert first.buried_sweep_seed_events == 0
    assert first.buried_sweep_candidates_rejected > 0
    assert first.stationary_overlap_seed_events == 0
    assert first.dynamic_solver_fracture_enabled is False
    retained = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert first.payload() == retained
