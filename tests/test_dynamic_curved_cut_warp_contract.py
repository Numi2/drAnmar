# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts/dr_anmar_dynamic_curved_cut_warp.py"


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
