#!/usr/bin/env python3
"""Require every repository-owned USD layer dependency to resolve natively."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from pxr import Sdf, UsdUtils
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Native OpenUSD Python bindings are required. Install "
        "scripts/requirements_openusd_validation.txt."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
USD_SUFFIXES = {".usd", ".usda", ".usdc", ".usdz"}
SCAN_ROOTS = (
    ROOT / "source/extensions/orbit.surgical.assets/data",
    ROOT / "assets/dr_anmar",
)


def validate_dependencies() -> dict[str, object]:
    layers = sorted(
        path
        for scan_root in SCAN_ROOTS
        if scan_root.is_dir()
        for path in scan_root.rglob("*")
        if path.is_file() and path.suffix.lower() in USD_SUFFIXES
    )
    issues: list[dict[str, object]] = []
    dependency_count = 0
    for path in layers:
        layer = Sdf.Layer.FindOrOpen(str(path.resolve()))  # type: ignore[attr-defined]
        if layer is None:
            issues.append({"layer": str(path.relative_to(ROOT)), "unopenable": True})
            continue
        dependency_layers, dependency_assets, unresolved = UsdUtils.ComputeAllDependencies(  # type: ignore[attr-defined]
            layer.identifier
        )
        dependency_count += len(dependency_layers) + len(dependency_assets)
        if unresolved:
            issues.append(
                {
                    "layer": str(path.relative_to(ROOT)),
                    "unresolved": sorted(str(item) for item in unresolved),
                }
            )
    return {
        "schema": "dr.anmar.openusd-dependency-validation.v1",
        "passed": not issues,
        "layer_count": len(layers),
        "resolved_dependency_count": dependency_count,
        "issues": issues,
    }


def main() -> int:
    report = validate_dependencies()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
