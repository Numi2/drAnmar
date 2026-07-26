#!/usr/bin/env python3
"""Fail closed around the exact dependency overrides required by Isaac Lab."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "config/physics-next-lock.json"


def evaluate(
    observed: list[str],
    allowed: list[str],
) -> dict[str, object]:
    observed_set = {line.strip() for line in observed if line.strip()}
    allowed_set = {line.strip() for line in allowed if line.strip()}
    unexpected = sorted(observed_set - allowed_set)
    missing = sorted(allowed_set - observed_set)
    return {
        "passed": not unexpected and not missing,
        "observed_conflicts": sorted(observed_set),
        "allowed_conflicts": sorted(allowed_set),
        "unexpected_conflicts": unexpected,
        "missing_expected_conflicts": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lock_path = args.lock.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    allowed = lock["dependency_policy"]["allowed_pip_check_conflicts"]
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    lines = [
        *result.stdout.splitlines(),
        *result.stderr.splitlines(),
    ]
    report = {
        "schema": "dr.anmar.physics-next-dependency-check.v1",
        "pip_check_returncode": result.returncode,
        "override_basis": lock["dependency_policy"]["override_basis"],
        **evaluate(lines, allowed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
