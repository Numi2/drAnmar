# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/dr_anmar_cuttable_tissue_solver.py"


def _module():
    name = "dranmar_test_cuttable_tissue_solver"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _assert_retained_receipt_matches(payload, retained):
    assert payload.keys() == retained.keys()
    for key, expected in retained.items():
        actual = payload[key]
        if isinstance(expected, float):
            assert np.isclose(actual, expected, rtol=1.0e-12, atol=1.0e-12), key
        else:
            assert actual == expected, key


def test_coupon_is_connected_positive_and_resolution_independent_of_cut_points():
    module = _module()
    profile = module.load_profile()
    points, tets = module.build_regular_tetrahedral_coupon(profile)
    expected_cells = np.prod([profile["geometry"][f"cells_{axis}"] for axis in "xyz"])
    assert len(tets) == expected_cells * 6
    assert np.unique(tets).size == len(points)
    matrices = np.stack(
        (
            points[tets[:, 1]] - points[tets[:, 0]],
            points[tets[:, 2]] - points[tets[:, 0]],
            points[tets[:, 3]] - points[tets[:, 0]],
        ),
        axis=2,
    )
    assert np.all(np.linalg.det(matrices) > 0.0)
    assert "cut_points" not in profile["fracture"]
    assert profile["fracture"]["all_internal_faces_eligible"] is True
    assert "dynamic_discontinuous_tetrahedral_dofs" in profile["fracture"]["required_future_terms"]


def test_reference_solver_uses_nonlinear_stress_relaxation_and_fixed_anchors():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "np.log(safe_j)" in source
    assert "inverse_transpose" in source
    assert "prony_history" in source
    assert "self.position[self.fixed] = self.fixed_position" in source
    assert "normal_stiffness_pa_m" in source
    assert "selected_triangles" in source
    assert "strip_width" in source


def test_neo_hookean_reference_is_objective_under_rigid_rotation():
    module = _module()
    profile = module.load_profile()
    profile["geometry"]["prestrain_x"] = 0.0
    solver = module.CuttableTissueReferenceSolver(profile)
    angle = np.deg2rad(31.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    solver.position = solver.rest @ rotation.T
    force, jacobian = solver._internal_force(float(profile["solver"]["time_step_s"]))
    assert np.allclose(jacobian, 1.0, atol=1.0e-12)
    assert np.max(np.linalg.norm(force, axis=1)) < 1.0e-9


def test_scalpel_contact_is_two_way_and_uses_a_finite_edge():
    module = _module()
    profile = module.load_profile()
    solver = module.CuttableTissueReferenceSolver(profile)
    radius = float(profile["scalpel_contact"]["edge_radius_m"])
    surface_z = float(np.max(solver.position[:, 2]))
    pose = module.ScalpelPose((0.0, 0.0, surface_z + radius - 0.0001))
    tissue_force, scalpel_reaction, penetration = solver._scalpel_contact(pose)
    assert penetration > 0.0
    assert np.linalg.norm(scalpel_reaction) > 0.0
    assert np.allclose(np.sum(tissue_force, axis=0) + scalpel_reaction, 0.0)


def test_surface_quadrature_prevents_off_grid_contact_holes():
    module = _module()
    solver = module.CuttableTissueReferenceSolver(module.load_profile())
    coverage, variation = module._off_grid_contact_sweep(solver)
    assert coverage == 1.0
    assert variation <= solver.profile["qualification"]["maximum_off_grid_force_variation_fraction"]


def test_fracture_is_fail_closed_for_the_first_milestone():
    module = _module()
    profile = module.load_profile()
    profile["fracture"]["enabled"] = True
    solver = module.CuttableTissueReferenceSolver(profile)
    with pytest.raises(RuntimeError, match="fracture disabled"):
        solver.step(float(profile["solver"]["time_step_s"]))


def test_intact_scalpel_contact_qualifies_and_replays_exactly():
    module = _module()
    first = module.run_intact_scalpel_qualification()
    second = module.run_intact_scalpel_qualification()
    assert first.qualified, first.failed_gates
    assert first.finite
    assert first.inverted_tetrahedra_peak == 0
    assert first.fracture_event_count == 0
    assert first.peak_scalpel_force_n > 0.0
    assert first.peak_scalpel_tangential_force_n > 0.0
    assert first.off_grid_contact_coverage_fraction == 1.0
    assert first.force_at_hold_end_n < first.force_at_hold_start_n
    assert first.deterministic_trace_sha256 == second.deterministic_trace_sha256
    assert first.payload() == second.payload()
    retained = json.loads(
        (ROOT / "physics_next/receipts/cuttable-tissue-intact-contact-reference.json").read_text(
            encoding="utf-8"
        )
    )
    _assert_retained_receipt_matches(first.payload(), retained)
