#!/usr/bin/env python3
"""Require native OpenUSD parsing and composition for every repository USD layer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAYER_ROOTS = (REPOSITORY_ROOT,)
USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc"})
IGNORED_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "node_modules",
    }
)
OPENUSD_REQUIREMENT = "usd-core==25.11"


def discover_usd_layers(roots: Iterable[Path]) -> list[Path]:
    """Return each canonical USD layer exactly once in stable path order."""
    discovered: set[Path] = set()
    for root in roots:
        root = root.resolve()
        if not root.is_dir():
            continue
        for current_root, directories, filenames in os.walk(root):
            directories[:] = sorted(
                name for name in directories if name not in IGNORED_PARTS
            )
            current = Path(current_root)
            for filename in sorted(filenames):
                path = current / filename
                if path.suffix.lower() in USD_SUFFIXES and path.is_file():
                    discovered.add(path.resolve())
    return sorted(discovered, key=lambda path: path.as_posix())


def _native_modules() -> tuple[Any, Any]:
    try:
        from pxr import Sdf, Usd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Native OpenUSD Python bindings are required; validation cannot "
            f"fall back to text or brace counting. Install {OPENUSD_REQUIREMENT}."
        ) from exc
    return Sdf, Usd


def validate_openusd_layers(
    roots: Sequence[Path] = DEFAULT_LAYER_ROOTS,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Parse and compose every discovered layer with native OpenUSD."""
    Sdf, Usd = _native_modules()
    resolved_roots = [root.resolve() for root in roots]
    missing_roots = [str(root) for root in resolved_roots if not root.is_dir()]
    empty_roots = [
        str(root)
        for root in resolved_roots
        if root.is_dir() and not discover_usd_layers((root,))
    ]
    layers = discover_usd_layers(resolved_roots)
    failures: list[dict[str, str]] = []
    parsed_count = 0
    composed_count = 0
    default_prim_count = 0

    for path in layers:
        relative_path = (
            path.relative_to(repository_root.resolve()).as_posix()
            if path.is_relative_to(repository_root.resolve())
            else str(path)
        )
        try:
            layer = Sdf.Layer.FindOrOpen(str(path))
            if layer is None:
                raise RuntimeError("Sdf.Layer.FindOrOpen returned no layer")
            parsed_count += 1
            stage = Usd.Stage.Open(layer, Usd.Stage.LoadNone)
            if stage is None:
                raise RuntimeError("Usd.Stage.Open returned no stage")
            composed_count += 1
            if not stage.GetDefaultPrim().IsValid():
                raise RuntimeError("composed stage has no valid default prim")
            default_prim_count += 1
        except Exception as exc:
            failures.append(
                {
                    "path": relative_path,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    version = ".".join(str(part) for part in Usd.GetVersion())
    discovery_failures = [
        {"path": path, "error": "required USD root is missing"}
        for path in missing_roots
    ] + [
        {"path": path, "error": "required USD root contains no USD layers"}
        for path in empty_roots
    ]
    failures = discovery_failures + failures
    return {
        "schema": "dr.anmar.native-openusd-layer-validation.v1",
        "passed": bool(layers) and not failures,
        "native_binding": "pxr",
        "openusd_version": version,
        "required_dependency": OPENUSD_REQUIREMENT,
        "roots": [
            (
                root.relative_to(repository_root.resolve()).as_posix()
                if root.is_relative_to(repository_root.resolve())
                else str(root)
            )
            for root in resolved_roots
        ],
        "layer_count": len(layers),
        "parsed_count": parsed_count,
        "composed_count": composed_count,
        "default_prim_count": default_prim_count,
        "failure_count": len(failures),
        "failures": failures,
    }


def require_openusd_layers(
    roots: Sequence[Path] = DEFAULT_LAYER_ROOTS,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    report = validate_openusd_layers(roots, repository_root=repository_root)
    if not report["passed"]:
        summary = "; ".join(
            f"{failure['path']}: {failure['error']}"
            for failure in report["failures"][:5]
        )
        raise RuntimeError(
            "Native OpenUSD layer validation failed "
            f"({report['failure_count']} failure(s)): {summary}"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        dest="roots",
        action="append",
        type=Path,
        help="layer root to validate; repeat for multiple roots",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    roots = tuple(args.roots) if args.roots else DEFAULT_LAYER_ROOTS
    report = validate_openusd_layers(roots)
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(output, encoding="utf-8")
    print(output, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
