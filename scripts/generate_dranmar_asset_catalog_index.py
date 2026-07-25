#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Generate a deterministic, content-addressed DrAnmar catalog index.

The index lives outside the asset folders it identifies, avoiding circular
hashes.  It records the same folder-hash contract as NVIDIA i4h asset-catalog
v0.7.0: sorted POSIX relative paths followed by each file's bytes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CATALOG_MODULE = ROOT / (
    "source/extensions/orbit.surgical.assets/orbit/surgical/assets/"
    "dranmar_asset_catalog.py"
)
DEFAULT_OUTPUT = ROOT / "physics_next/dranmar-asset-catalog-index.json"


def load_catalog():
    spec = importlib.util.spec_from_file_location(
        "dranmar_asset_catalog_index_generator", CATALOG_MODULE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load catalog module: {CATALOG_MODULE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_index() -> dict[str, object]:
    catalog = load_catalog()
    assets: dict[str, object] = {}
    for name, descriptor in sorted(catalog.DRANMAR_SIM_READY_ASSETS.items()):
        directory = catalog.asset_directory(name)
        closure = catalog.validate_usd_dependency_closure(name)
        assets[name] = {
            "asset_id": descriptor.asset_id,
            "catalog_subpath": descriptor.catalog_subpath,
            "primary_usd": descriptor.primary_usd,
            "sha256": catalog.sha256_of_folder(directory),
            "file_count": sum(1 for _ in catalog.iter_hashed_files(directory)),
            "usd_references_checked": closure["references_checked"],
        }
    return {
        "schema": "dr.anmar.sim-ready-asset-catalog-index.v1",
        "catalog_version": catalog.CATALOG_VERSION,
        "hash_contract": (
            "sha256(sorted POSIX relative path bytes followed by file bytes)"
        ),
        "reference_implementation": {
            "repository": "isaac-for-healthcare/i4h-asset-catalog",
            "release": catalog.I4H_REFERENCE_RELEASE,
            "commit": catalog.I4H_REFERENCE_COMMIT,
        },
        "asset_count": len(assets),
        "assets": assets,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the current output differs from the generated index",
    )
    args = parser.parse_args()
    payload = json.dumps(build_index(), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.is_file():
            raise FileNotFoundError(f"Catalog index is missing: {args.output}")
        if args.output.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"Catalog index is stale: {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
