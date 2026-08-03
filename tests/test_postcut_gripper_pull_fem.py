# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/dr_anmar_postcut_gripper_pull_fem.py"
RECEIPT_PATH = ROOT / "physics_next/receipts/postcut-gripper-pull-reference.json"
GIF_PATH = ROOT / "docs/media/dranmar-postcut-gripper-pull.gif"
RENDER_RECEIPT_PATH = ROOT / "docs/media/dranmar-postcut-gripper-pull.json"


def _module():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    name = "dranmar_test_postcut_gripper_pull_fem"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_postcut_gripper_pull_is_bilateral_differential_and_recoverable():
    module = _module()
    receipt = module.run_postcut_gripper_pull_qualification()
    assert receipt.qualified, receipt.failed_gates
    assert receipt.released_pair_count == 85
    assert receipt.topology_event_delta == 0
    assert receipt.retained_anchor_node_count > 0
    assert receipt.pull_bilateral_custody_fraction == 1.0
    assert receipt.peak_top_contact_count >= 3
    assert receipt.peak_bottom_contact_count >= 3
    assert receipt.gripped_flap_lateral_displacement_m >= 0.002
    assert receipt.differential_flap_displacement_m >= 0.0015
    assert receipt.local_wound_gap_increase_m >= 0.0003
    assert receipt.maximum_anchor_drift_m == 0.0
    assert receipt.inversion_observation_count == 0
    assert receipt.deterministic_replay is True


def test_postcut_gripper_pull_matches_retained_receipt():
    module = _module()
    actual = module.run_postcut_gripper_pull_qualification().payload()
    retained = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert actual.keys() == retained.keys()
    for key, expected in retained.items():
        value = actual[key]
        if isinstance(expected, float):
            assert np.isclose(value, expected, rtol=1.0e-10, atol=1.0e-12), key
        else:
            assert value == expected, key


def test_postcut_gripper_pull_render_is_bound_to_qualified_mechanics():
    receipt = json.loads(RENDER_RECEIPT_PATH.read_text(encoding="utf-8"))
    assert hashlib.sha256(GIF_PATH.read_bytes()).hexdigest() == receipt["gif_sha256"]
    with Image.open(GIF_PATH) as gif:
        assert gif.size == (receipt["width"], receipt["height"])
        assert gif.n_frames == receipt["encoded_frame_count"]
    assert receipt["mechanics_qualified"] is True
    assert receipt["pull_bilateral_custody_fraction"] == 1.0
    assert receipt["topology_event_delta"] == 0
    assert receipt["inversion_observation_count"] == 0
    assert receipt["displacement_exaggeration"] == 1.0
    assert receipt["generated_imagery"] is False
