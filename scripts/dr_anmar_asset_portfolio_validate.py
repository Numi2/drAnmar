#!/usr/bin/env python3
"""Validate the truthful cross-asset DrAnmar portfolio boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "physics_next/dr-anmar-assets.json"


def iter_strings(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_strings(item)
    elif isinstance(value, str):
        yield value


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    errors: list[str] = []
    if manifest.get("schema") != "dr.anmar.asset-portfolio.v1":
        errors.append("unsupported portfolio schema")
    assets = manifest.get("assets", [])
    ids = [asset.get("id") for asset in assets]
    if len(assets) != 4 or len(ids) != len(set(ids)):
        errors.append("portfolio must contain four uniquely identified systems")
    for asset in assets:
        if asset.get("clinical_validation") is not False:
            errors.append(f"{asset.get('id')}: clinical validation must remain false")
        for key in ("asset", "profile", "report"):
            relative = asset.get(key)
            if not relative or not (REPOSITORY_ROOT / relative).is_file():
                errors.append(f"{asset.get('id')}: missing {key} {relative!r}")
        for key in (
            "explicit_tetmesh",
            "auxiliary_asset",
            "material_texture",
        ):
            relative = asset.get(key)
            if relative and not (REPOSITORY_ROOT / relative).is_file():
                errors.append(f"{asset.get('id')}: missing {key} {relative!r}")
        if not str(asset.get("native_gpu_qualification", "")).startswith("blocked_pending"):
            errors.append(f"{asset.get('id')}: native qualification promoted without evidence")
        if not str(asset.get("physical_qualification", "")).startswith("blocked_pending"):
            errors.append(f"{asset.get('id')}: physical qualification promoted without evidence")
    if manifest.get("validation", {}).get("clinical_use") != "blocked":
        errors.append("portfolio clinical-use boundary must remain blocked")
    if manifest.get("ownership", {}).get("external_geometry_dependencies") != []:
        errors.append("DrAnmar portfolio declares an external geometry dependency")
    absolute_strings = [value for value in iter_strings(manifest) if value.startswith("/")]
    if absolute_strings:
        errors.append(f"portfolio contains absolute paths: {absolute_strings}")
    report = {
        "schema": "dr.anmar.asset-portfolio-validation.v1",
        "passed": not errors,
        "asset_count": len(assets),
        "asset_ids": ids,
        "errors": errors,
        "clinical_validation": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
