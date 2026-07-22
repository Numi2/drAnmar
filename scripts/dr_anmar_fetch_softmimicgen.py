#!/usr/bin/env python3
"""Fetch and verify the small, pinned SoftMimicGen assets Dr.Anmar qualifies.

The assets stay in the mutable runtime data directory. They are never silently
updated and are not vendored into the Dr.Anmar repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "physics_next" / "softmimicgen.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "DrAnmar/SoftMimicGen-fetch"})
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    report: dict[str, object] = {
        "schema": "dr.anmar.softmimicgen-assets.v1",
        "source_revision": manifest["repository"]["revision"],
        "asset_dir": str(args.output.resolve()),
        "assets": {},
    }
    valid = True
    for name, contract in manifest["assets"].items():
        path = args.output / name
        expected = contract["sha256"]
        if not path.exists() and not args.verify_only:
            download(contract["url"], path)
        actual = sha256(path) if path.exists() else None
        asset_valid = actual == expected
        valid = valid and asset_valid
        report["assets"][name] = {
            "path": str(path.resolve()),
            "exists": path.exists(),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "valid": asset_valid,
        }
    report["valid"] = valid
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
