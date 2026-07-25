#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Refresh repository-resolvable entries in installed DrAnmar manifests.

Legacy development-package manifests also contain paths that only exist inside
the downloadable package. Those records are preserved. Paths present in the
repository overlay are always re-hashed from the current bytes so installed
integrity checks cannot silently validate a stale pre-pruning source file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = (
    "source/extensions/orbit.surgical.assets/data/Props/"
    "Patients/DynamicAbdominalPatient/asset_manifest.json",
    "source/extensions/orbit.surgical.assets/data/Props/"
    "SurgicalClosure/SkinAdhesive/asset_manifest.json",
    "source/extensions/orbit.surgical.assets/data/Props/"
    "SurgicalPreparation/WoundPreparationRobot/asset_manifest.json",
    "source/extensions/orbit.surgical.assets/data/Props/"
    "SurgicalExposure/AtraumaticExposureRobot/asset_manifest.json",
    "source/extensions/orbit.surgical.assets/data/Props/"
    "SurgicalHemostasis/AdaptiveHemostasisRobot/asset_manifest.json",
    "source/extensions/orbit.surgical.assets/data/Props/"
    "SurgicalReconstruction/AdaptiveAnastomosisRobot/asset_manifest.json",
    "source/extensions/orbit.surgical.assets/data/Props/"
    "SurgicalDivision/AdaptiveSealDivideRobot/asset_manifest.json",
    "source/extensions/orbit.surgical.assets/data/Props/"
    "SurgicalDissection/SafePlaneDissectionRobot/asset_manifest.json",
)
INSTALLED_PREFIXES = (
    "source/",
    "physics_next/",
    "docs/",
    "examples/",
    "tests/",
    "scripts/",
)


def installed_overlay_source(relative_path: str) -> Path | None:
    if relative_path.startswith("assets/Props/"):
        source = (
            ROOT
            / "source/extensions/orbit.surgical.assets/data"
            / relative_path.removeprefix("assets/")
        )
        return source if source.is_file() else None
    if not relative_path.startswith(INSTALLED_PREFIXES):
        return None
    source = ROOT / relative_path
    return source if source.is_file() else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def refresh_manifest(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["files"] = [
        entry
        for entry in payload.get("files", [])
        if not str(entry.get("path", "")).endswith("/qualification_report.json")
    ]
    if "file_count" in payload:
        payload["file_count"] = len(payload["files"])
    refreshed = 0
    for entry in payload.get("files", []):
        source = installed_overlay_source(entry["path"])
        if source is None:
            continue
        entry["bytes"] = source.stat().st_size
        entry["sha256"] = sha256(source)
        refreshed += 1
    payload["installed_overlay_entries_refreshed"] = refreshed
    payload["installed_overlay_hash_contract"] = (
        "sha256 of exact current repository file bytes"
    )
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return refreshed


def main() -> int:
    for relative in MANIFESTS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if refresh_manifest(path) < 1:
            raise RuntimeError(f"No installed entries resolved in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
