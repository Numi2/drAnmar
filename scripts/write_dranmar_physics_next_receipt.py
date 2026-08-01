#!/usr/bin/env python3
"""Write an atomic, content-addressed physics-next runtime receipt."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
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


def _package(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()

    next_root = args.root.resolve()
    lock_path = args.lock.resolve()
    freeze_path = next_root / "python-freeze.txt"
    dependency_check_path = next_root / "dependency-check.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    dependency_check = json.loads(
        dependency_check_path.read_text(encoding="utf-8")
    )
    if not dependency_check.get("passed"):
        raise SystemExit("dependency check did not pass")

    installed_packages = {
        name: _package(name) for name in lock["runtime_packages"]
    }
    mismatches = {
        name: {
            "expected": expected,
            "observed": installed_packages.get(name),
        }
        for name, expected in lock["runtime_packages"].items()
        if installed_packages.get(name) != expected
    }
    if mismatches:
        raise SystemExit(
            "runtime package lock mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )

    source_paths = {
        "isaaclab": next_root / "IsaacLab",
        "cressim_mpm": next_root / "CRESSim-MPM",
    }
    source_heads = {
        source_id: _git_head(path) for source_id, path in source_paths.items()
    }
    source_mismatches = {
        source_id: {
            "expected": lock["sources"][source_id]["revision"],
            "observed": observed,
        }
        for source_id, observed in source_heads.items()
        if observed != lock["sources"][source_id]["revision"]
    }
    if source_mismatches:
        raise SystemExit(
            "runtime source lock mismatch: "
            + json.dumps(source_mismatches, sort_keys=True)
        )

    cressim_build = lock["builds"]["cressim_mpm"]
    cressim_library = next_root / cressim_build["library_relative_path"]
    if not cressim_library.is_file():
        raise SystemExit(f"missing pinned CRESSim shared library: {cressim_library}")

    try:
        gpu_driver = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        gpu_driver = []

    receipt = {
        "schema": "dr.anmar.physics-next-installation.v1",
        "ready": True,
        "lock_sha256": _sha256(lock_path),
        "python_freeze_sha256": _sha256(freeze_path),
        "dependency_check_sha256": _sha256(dependency_check_path),
        "dependency_check": dependency_check,
        "installation_profile": lock["dependency_policy"][
            "isaaclab_install_profile"
        ],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu_driver": gpu_driver,
        "packages": installed_packages,
        "sources": source_heads,
        "artifacts": {
            "cressim_mpm_c_api": {
                "path": cressim_build["library_relative_path"],
                "sha256": _sha256(cressim_library),
                "build": cressim_build,
            }
        },
        "clinical_validation": False,
    }
    receipt_path = next_root / "runtime.json"
    temporary_path = next_root / f".runtime.json.{os.getpid()}.tmp"
    temporary_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, receipt_path)
    (next_root / "READY").write_text(
        _sha256(receipt_path) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
