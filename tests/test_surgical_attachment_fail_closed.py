from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET_MODULES = (
    "wound_preparation_robot.py",
    "atraumatic_exposure_robot.py",
    "adaptive_hemostasis_robot.py",
    "adaptive_anastomosis_robot.py",
    "adaptive_seal_divide_robot.py",
    "safeplane_dissection_robot.py",
)
ASSET_SOURCE = (
    ROOT
    / "source/extensions/orbit.surgical.assets/orbit/surgical/assets"
)


def _attachment_function(path: Path) -> ast.FunctionDef:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "create_deformable_attachment"
    ]
    assert len(functions) == 1, f"{path} has {len(functions)} attachment helpers"
    return functions[0]


def test_all_six_attachment_helpers_reject_insufficient_overlap() -> None:
    for name in ASSET_MODULES:
        path = ASSET_SOURCE / name
        function = _attachment_function(path)
        source = ast.get_source_segment(
            path.read_text(encoding="utf-8"), function
        )
        assert source is not None
        assert "overlapping={len(selected)}" in source
        assert "required=4" in source
        assert "UsdGeom.Tokens.guide" in source
        assert "selected = ranked[:" not in source
        assert "selected=ranked[:" not in source


def test_deterministic_generators_cannot_restore_nearest_vertex_fallback() -> None:
    for name in (
        "generate_dranmar_wound_preparation_robot.py",
        "generate_dranmar_atraumatic_exposure_robot.py",
    ):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "selected = ranked[: min(4, len(ranked))]" not in source
        assert "required=4, overlap_margin_m=0.0025" in source


def test_all_six_helpers_convert_public_wxyz_to_isaac6_xyzw() -> None:
    for index, name in enumerate(ASSET_MODULES):
        path = ASSET_SOURCE / name
        spec = importlib.util.spec_from_file_location(
            f"dranmar_quaternion_contract_{index}", path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            assert module._xyzw_from_wxyz((1.0, 0.0, 0.0, 0.0)) == (
                0.0,
                0.0,
                0.0,
                1.0,
            )
            assert module._xyzw_from_wxyz((0.0, 1.0, 0.0, 0.0)) == (
                1.0,
                0.0,
                0.0,
                0.0,
            )
        finally:
            sys.modules.pop(spec.name, None)
