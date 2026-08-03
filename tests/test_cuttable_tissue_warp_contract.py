# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/dr_anmar_cuttable_tissue_warp.py"
PROFILE = ROOT / "physics_next/tissues/dr-anmar-cuttable-tissue-v1.json"
RECEIPT = ROOT / "physics_next/receipts/cuttable-tissue-warp-cpu-parity.json"
REFERENCE_RECEIPT = ROOT / "physics_next/receipts/cuttable-tissue-intact-contact-reference.json"


def test_warp_backend_owns_continuum_and_surface_contact_kernels():
    source = SOURCE.read_text(encoding="utf-8")
    assert "@wp.kernel\ndef _neo_hookean_prony_force" in source
    assert "@wp.kernel\ndef _surface_scalpel_contact" in source
    assert "wp.atomic_add(force_x" in source
    assert "strip_width" in source
    assert "selected_triangles" not in source
    assert 'device: str = "cpu"' in source


def test_cpu_parity_receipt_passes_but_cannot_promote_cuda():
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    reference_receipt = json.loads(REFERENCE_RECEIPT.read_text(encoding="utf-8"))
    limits = profile["warp_parity"]
    assert receipt["profile_sha256"] == reference_receipt["profile_sha256"]
    assert receipt["qualified"]
    assert receipt["device_is_cuda"] is False
    assert receipt["cuda_promotion_pending"] is True
    assert (
        receipt["maximum_internal_force_relative_l2_error"]
        <= limits["maximum_internal_force_relative_l2_error"]
    )
    assert (
        receipt["maximum_contact_force_relative_l2_error"]
        <= limits["maximum_contact_force_relative_l2_error"]
    )
    assert (
        receipt["maximum_contact_penetration_absolute_error_m"]
        <= limits["maximum_contact_penetration_absolute_error_m"]
    )
    assert limits["cuda_qualification_required_for_promotion"] is True


def test_warp_is_kept_out_of_the_locked_repository_validation_environment():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    requirements = (ROOT / "scripts/requirements_cuttable_tissue_warp.txt").read_text(
        encoding="utf-8"
    )
    dependencies = pyproject.split("[project]", 1)[1].split("[dependency-groups]", 1)[0]
    assert "warp-lang" not in dependencies
    assert "warp-lang>=1.9,<2" in requirements
