# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/dr_anmar_moving_scalpel_cut_fem.py"
RECEIPT_PATH = ROOT / "physics_next/receipts/moving-scalpel-cut-reference.json"


def _module():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    name = "dranmar_test_moving_scalpel_cut_fem"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_blade_release_is_local_irreversible_and_subcritical_safe():
    module = _module()
    moving = module.load_moving_profile()
    base = module.load_profile(module.REPOSITORY_ROOT / moving["base_profile"])
    curved = module.load_curved_profile(module.REPOSITORY_ROOT / moving["embedded_profile"])
    solver = module.MovingScalpelCutFEM(base, curved, moving)
    poses = module._path_poses(moving, curved)
    work = module._work_channels(base, moving)
    before = np.count_nonzero(solver.released)
    solver.advance_blade(0, poses[0], poses[1], work)
    after = np.count_nonzero(solver.released)
    assert after > before
    assert np.max(solver.mesh.gap_rest_points[solver.released, 0]) <= poses[1].center_m[0] + moving["qualification"]["maximum_release_ahead_of_blade_m"]
    solver.advance_blade(1, poses[1], poses[2], work)
    assert np.count_nonzero(solver.released) >= after
    subcritical = module.PersistentCutCellField(base)
    subcritical.apply_sweep(poses[0], poses[1], module._work_channels(base, moving, ratio=0.99))
    assert subcritical.fracture_event_count == 0


def test_moving_scalpel_qualification_replays_and_matches_receipt():
    module = _module()
    first = module.run_moving_scalpel_qualification()
    second = module.run_moving_scalpel_qualification()
    assert first.qualified, first.failed_gates
    assert first.event_trace_sha256 == second.event_trace_sha256
    assert first.fracture_event_count == second.fracture_event_count
    retained = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert first.payload().keys() == retained.keys()
    for key, expected in retained.items():
        actual = first.payload()[key]
        if isinstance(expected, float):
            assert np.isclose(actual, expected, rtol=1e-10, atol=1e-12), key
        else:
            assert actual == expected, key
