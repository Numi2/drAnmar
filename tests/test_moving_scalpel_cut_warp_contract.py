# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts/dr_anmar_moving_scalpel_cut_warp.py"


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
