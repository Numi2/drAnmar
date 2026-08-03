# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/dr_anmar_cohesive_fracture_warp.py"
PROFILE = ROOT / "physics_next/tissues/dr-anmar-cuttable-tissue-v1.json"
RECEIPT = ROOT / "physics_next/receipts/cuttable-tissue-cohesive-warp-cpu-parity.json"
REFERENCE = ROOT / "physics_next/receipts/cuttable-tissue-cohesive-fracture-reference.json"


def test_warp_kernel_carries_the_complete_cohesive_contract():
    source = SOURCE.read_text(encoding="utf-8")
    assert "@wp.kernel\ndef _mixed_mode_cohesive_response" in source
    assert "compression_penalty" in source
    assert "bk_exponent" in source
    assert "previous_maximum" in source
    assert "previous_damage" in source
    assert "seeded" in source
    assert 'device: str = "cpu"' in source


def test_retained_cohesive_warp_parity_passes_but_cuda_is_pending():
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    limits = profile["warp_parity"]
    assert receipt["profile_sha256"] == reference["profile_sha256"]
    assert receipt["qualified"]
    assert receipt["device_is_cuda"] is False
    assert receipt["cuda_promotion_pending"] is True
    assert (
        receipt["maximum_traction_relative_l2_error"]
        <= limits["maximum_cohesive_traction_relative_l2_error"]
    )
    assert (
        receipt["maximum_damage_absolute_error"] <= limits["maximum_cohesive_damage_absolute_error"]
    )
    assert (
        receipt["maximum_envelope_separation_absolute_error_m"]
        <= limits["maximum_cohesive_separation_absolute_error_m"]
    )
