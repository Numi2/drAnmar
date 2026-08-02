# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts/dr_anmar_moving_scalpel_cut_warp.py"
PROMOTION_LOCK = ROOT / "physics_next/receipts/moving-scalpel-cut-cuda-promotion-lock.json"


def test_moving_scalpel_warp_contract_requires_real_cuda_and_event_parity():
    source = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    kernels = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(getattr(decorator, "attr", None) == "kernel" for decorator in node.decorator_list)
    }
    assert "_moving_cohesive_and_wedge_force" in kernels
    assert "released.assign" in source
    assert '"cuda_device": is_cuda' in source
    assert '"event_trace": event_match' in source
    assert "cuda_promotion_pending=not (is_cuda and not failed)" in source


def test_moving_scalpel_cuda_lock_hashes_real_receipt_and_bounded_claim():
    lock = json.loads(PROMOTION_LOCK.read_text(encoding="utf-8"))
    for field in ("profile", "implementation", "front_authority", "cpu_oracle", "cuda_receipt"):
        path = ROOT / lock[field]["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == lock[field]["sha256"]
    receipt = json.loads((ROOT / lock["cuda_receipt"]["path"]).read_text())
    assert receipt["device_is_cuda"] is True
    assert receipt["event_trace_matches_cpu"] is True
    assert receipt["qualified"] is True
    assert lock["fresh_cuda_replays"]["count"] == 5
    assert lock["fresh_cuda_replays"]["event_trace_identical"] is True
    assert lock["cuda_moving_scalpel_qualified"] is True
    assert "real_time_transient_cutting" in lock["blocked_claims"]
