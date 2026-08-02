# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/dr_anmar_dynamic_curved_cut_fem.py"
RECEIPT_PATH = ROOT / "physics_next/receipts/dynamic-curved-cut-reference.json"


def _module():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    name = "dranmar_test_dynamic_curved_cut_fem"
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
            assert np.isclose(actual, expected, rtol=1.0e-10, atol=1.0e-12), key
        else:
            assert actual == expected, key


def test_cut_requires_cohesive_energy_gate():
    module = _module()
    curved = module.load_curved_profile()
    curved["implicit_cut"]["applied_fracture_work_ratio"] = 0.99
    base = module.load_profile(module.REPOSITORY_ROOT / curved["base_profile"])
    with pytest.raises(ValueError, match="cohesive-energy"):
        module._build_settled_mesh(base, curved)


def test_curved_cut_is_in_element_conforming_and_mass_preserving():
    module = _module()
    curved = module.load_curved_profile()
    base = module.load_profile(module.REPOSITORY_ROOT / curved["base_profile"])
    solver = module._build_settled_mesh(base, curved)
    unexpected, nonmanifold = solver.topology_metrics()
    assert solver.cut_original_tetrahedron_count > 0
    assert len(solver.tetrahedra) > len(solver.original_tets)
    assert unexpected == 0
    assert nonmanifold == 0
    original_volume = np.prod([base["geometry"][key] for key in ("width_m", "depth_m", "thickness_m")])
    assert np.isclose(np.sum(solver.rest_volume), original_volume, rtol=1e-11)
    assert np.isclose(np.sum(solver.mass), base["material"]["density_kg_m3"] * original_volume, rtol=1e-11)
    wound = solver.wound_surface_mesh()
    assert set(wound.triangle_sides.tolist()) == {-1, 1}
    assert np.isclose(wound.positive_area_m2, wound.negative_area_m2, rtol=1e-12)


def test_dynamic_curved_cut_qualifies_replays_and_matches_receipt():
    module = _module()
    first = module.run_dynamic_curved_cut_qualification()
    second = module.run_dynamic_curved_cut_qualification()
    assert first.qualified, first.failed_gates
    assert first.payload() == second.payload()
    assert first.curve_midpoint_deviation_from_chord_m > 0.0
    assert first.unexpected_boundary_face_count == 0
    assert first.nonmanifold_face_count == 0
    retained = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    _assert_retained_receipt_matches(first.payload(), retained)
