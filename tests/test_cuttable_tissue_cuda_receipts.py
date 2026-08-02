# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = ROOT / "physics_next/receipts"
LOCK = RECEIPTS / "cuttable-tissue-cuda-promotion-lock.json"


def test_cuda_receipts_are_profile_bound_qualified_and_not_pending():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for entry in lock["raw_receipts"]:
        path = ROOT / entry["path"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        assert payload["profile_sha256"] == lock["profile_sha256"]
        assert payload["device"] == "cuda:0"
        assert payload["device_is_cuda"] is True
        assert payload["qualified"] is True
        assert payload["failed_gates"] == []
        assert payload["cuda_promotion_pending"] is False


def test_cuda_promotion_lock_records_replay_envelope_and_remote_suite():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    replay = lock["cuda_replay"]
    assert lock["qualified"] is True
    assert lock["qualified_source_revision"] == "b5798974ea238a7a6ac6f677ed7945e5c9503d4c"
    assert lock["gpu"]["name"] == "NVIDIA GeForce RTX 4090"
    assert lock["runtime"]["warp"] == "1.15.0"
    assert replay["runs"] == 5
    assert replay["all_runs_qualified"] is True
    assert replay["intact_internal_force_relative_l2_error"]["maximum"] < 5.0e-4
    assert replay["intact_internal_force_relative_l2_error"]["span"] < 1.0e-8
    assert replay["cohesive_traction_relative_l2_error"]["maximum"] < 1.0e-3
    assert lock["remote_reference_tests"] == {
        "total_passed": 21,
        "intact_and_contact": 10,
        "cohesive_fracture": 7,
        "persistent_topology": 4,
        "failed": 0,
    }
