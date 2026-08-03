# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts/dr_anmar_dynamic_curved_cut_warp.py"
PROMOTION_LOCK = ROOT / "physics_next/receipts/dynamic-curved-cut-cuda-promotion-lock.json"


def test_dynamic_curved_warp_contract_contains_gpu_physics_stages():
    source = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    kernels = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(getattr(decorator, "attr", None) == "kernel" for decorator in node.decorator_list)
    }
    assert {"_wound_compression_force", "_wound_opening_force", "_integrate_nodes", "_accumulate_jacobian_bounds"} <= kernels
    assert "_neo_hookean_prony_force" in source
    assert '"cuda_device": is_cuda' in source
    assert "cuda_promotion_pending=not (is_cuda and not failed)" in source


def test_cuda_promotion_lock_hashes_real_cuda_receipt_and_keeps_claim_bounded():
    lock = json.loads(PROMOTION_LOCK.read_text(encoding="utf-8"))
    for field in ("profile", "implementation", "cpu_oracle", "cuda_receipt"):
        path = ROOT / lock[field]["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == lock[field]["sha256"]
    receipt = json.loads((ROOT / lock["cuda_receipt"]["path"]).read_text())
    assert receipt["device_is_cuda"] is True
    assert receipt["qualified"] is True
    assert receipt["cuda_promotion_pending"] is False
    assert lock["fresh_cuda_replays"]["count"] == 5
    assert lock["fresh_cuda_replays"]["all_qualified"] is True
    assert lock["cuda_dynamic_cut_qualified"] is True
    assert "sequential_intersecting_dynamic_remesh" in lock["blocked_claims"]
