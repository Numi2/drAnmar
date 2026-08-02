# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/dr_anmar_dynamic_discontinuous_fem.py"
RECEIPT_PATH = ROOT / "physics_next/receipts/dynamic-planar-cut-reference.json"


def _module():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    name = "dranmar_test_dynamic_discontinuous_fem"
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


def test_dynamic_mesh_has_mass_preserving_independent_tetrahedral_dofs():
    module = _module()
    dynamic = module.load_dynamic_profile()
    base = module.load_profile(module.REPOSITORY_ROOT / dynamic["base_profile"])
    solver = module.DynamicDiscontinuousFEM(base, dynamic)
    expected_mass = (
        base["material"]["density_kg_m3"]
        * base["geometry"]["width_m"]
        * base["geometry"]["depth_m"]
        * base["geometry"]["thickness_m"]
    )
    assert len(solver.position) == len(solver.tetrahedra) * 4
    assert np.isclose(np.sum(solver.mass), expected_mass, rtol=1.0e-14)
    assert np.count_nonzero(solver.cut_interfaces) == 24
    assert np.count_nonzero(solver.released) == 0
    _, jump = solver._cohesive_force()
    assert np.max(np.linalg.norm(jump, axis=1)) == 0.0


def test_cut_release_creates_opposed_deforming_surfaces_and_one_sided_contact():
    module = _module()
    dynamic = module.load_dynamic_profile()
    base = module.load_profile(module.REPOSITORY_ROOT / dynamic["base_profile"])
    solver = module.DynamicDiscontinuousFEM(base, dynamic)
    solver.release_cut()
    mesh = solver.wound_surface_mesh()
    assert len(mesh.triangles) == 48
    assert set(mesh.triangle_sides.tolist()) == {-1, 1}
    assert np.isclose(mesh.positive_area_m2, mesh.negative_area_m2, atol=1.0e-15)
    for side in (-1, 1):
        triangle_index = int(np.flatnonzero(mesh.triangle_sides == side)[0])
        triangle = mesh.vertices_m[mesh.triangles[triangle_index]]
        normal = mesh.triangle_normals[triangle_index]
        radius = dynamic["wound_collision"]["probe_radius_m"]
        point = np.mean(triangle, axis=0) + 0.5 * radius * normal
        force, crossing = module.one_sided_wound_contact_force(
            mesh,
            point,
            -0.01 * normal,
            radius,
            dynamic["wound_collision"]["normal_stiffness_n_m"],
            dynamic["wound_collision"]["normal_damping_n_s_m"],
            side=side,
        )
        assert np.dot(force, normal) > 0.0
        assert crossing == 0.0


def test_dynamic_planar_cut_qualifies_replays_and_matches_retained_receipt():
    module = _module()
    first = module.run_dynamic_planar_cut_qualification()
    second = module.run_dynamic_planar_cut_qualification()
    assert first.qualified, first.failed_gates
    assert first.payload() == second.payload()
    assert first.mass_relative_error == 0.0
    assert first.inverted_tetrahedra_peak == 0
    assert first.mean_wound_gap_m > 0.0
    assert first.two_sided_collision_coverage_fraction == 1.0
    assert first.maximum_probe_surface_crossing_m == 0.0
    retained = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    _assert_retained_receipt_matches(first.payload(), retained)
