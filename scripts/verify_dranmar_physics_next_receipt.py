#!/usr/bin/env python3
"""Verify a content-addressed Dr.Anmar physics-next installation receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "config/physics-next-lock.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def verify(next_root: Path, lock_path: Path) -> list[str]:
    failures: list[str] = []
    receipt_path = next_root / "runtime.json"
    freeze_path = next_root / "python-freeze.txt"
    ready_path = next_root / "READY"
    if not receipt_path.is_file():
        return [f"missing receipt: {receipt_path}"]
    if not freeze_path.is_file():
        failures.append(f"missing dependency freeze: {freeze_path}")
    if not ready_path.is_file():
        failures.append(f"missing readiness digest: {ready_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "dr.anmar.physics-next-installation.v1":
        failures.append("unsupported physics-next receipt schema")
    if receipt.get("lock_sha256") != _sha256(lock_path):
        failures.append("physics-next lock digest mismatch")
    if freeze_path.is_file() and receipt.get("python_freeze_sha256") != _sha256(freeze_path):
        failures.append("Python dependency freeze digest mismatch")
    for package_name, expected_version in lock["runtime_packages"].items():
        if receipt.get("packages", {}).get(package_name) != expected_version:
            failures.append(f"{package_name} runtime package mismatch")
    for source_id, relative in (
        ("isaaclab", "IsaacLab"),
        ("cressim_mpm", "CRESSim-MPM"),
    ):
        expected = lock["sources"][source_id]["revision"]
        if receipt.get("sources", {}).get(source_id) != expected:
            failures.append(f"{source_id} receipt revision mismatch")
        if _git_head(next_root / relative) != expected:
            failures.append(f"{source_id} checkout revision mismatch")
    if ready_path.is_file():
        expected_ready = _sha256(receipt_path)
        if ready_path.read_text(encoding="utf-8").strip() != expected_ready:
            failures.append("READY digest does not match runtime receipt")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    failures = verify(args.root.expanduser().resolve(), args.lock.resolve())
    print(
        json.dumps(
            {
                "passed": not failures,
                "root": str(args.root),
                "failures": failures,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
