#!/usr/bin/env python3
"""Refresh or verify the Dynamic Abdominal Patient content manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "source/extensions/orbit.surgical.assets/data"
PATIENT_ROOT = DATA_ROOT / "Props/Patients/DynamicAbdominalPatient"
MANIFEST = PATIENT_ROOT / "asset_manifest.json"
PATH_PREFIX = "assets/"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest() -> dict[str, object]:
    current = json.loads(MANIFEST.read_text(encoding="utf-8"))
    files = []
    for source in sorted(
        path
        for path in PATIENT_ROOT.rglob("*")
        if path.is_file() and path != MANIFEST
    ):
        logical_path = PATH_PREFIX + source.relative_to(DATA_ROOT).as_posix()
        files.append(
            {
                "bytes": source.stat().st_size,
                "path": logical_path,
                "sha256": _sha256(source),
            }
        )
    return {**current, "files": sorted(files, key=lambda item: item["path"])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    serialized = json.dumps(build_manifest(), indent=2) + "\n"
    if args.check:
        if MANIFEST.read_text(encoding="utf-8") != serialized:
            print("Dynamic-patient asset manifest is stale.")
            return 1
        print("Dynamic-patient asset manifest is content-consistent.")
        return 0
    MANIFEST.write_text(serialized, encoding="utf-8")
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
