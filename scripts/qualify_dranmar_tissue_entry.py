#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Create an entry-only qualification result and immutable promotion lock."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT / "source/extensions/orbit.surgical.tasks/orbit/surgical/tasks/"
    "surgical/penetration/contract.py"
)


def _load_contract():
    spec = importlib.util.spec_from_file_location("dranmar_penetration_contract", CONTRACT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-result", type=Path, action="append", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--learned-summary", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--promotion-lock", type=Path, required=True)
    args = parser.parse_args()

    contract = _load_contract()
    seed_results = [json.loads(path.read_text(encoding="utf-8")) for path in args.seed_result]
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    learned = json.loads(args.learned_summary.read_text(encoding="utf-8"))
    runtime = json.loads(args.runtime_receipt.read_text(encoding="utf-8"))
    qualification = contract.evaluate_qualification(seed_results)
    paired_passed, paired_basis = contract.learned_policy_beats_baseline(learned, baseline)
    qualification["paired_baseline_passed"] = paired_passed
    qualification["paired_baseline_basis"] = paired_basis
    qualification["qualified"] = bool(qualification["qualified"] and paired_passed)
    if not paired_passed:
        qualification["failures"] = sorted(
            set((*qualification["failures"], "paired_analytical_baseline"))
        )
    qualification["created_at"] = datetime.now(timezone.utc).isoformat()
    qualification["task"] = "DrAnmar-Penetrate-Tissue-Needle-PSM-IK-Rel-v0"
    qualification["evidence_level"] = "simulator_engineering_only"
    _atomic_json(args.output.resolve(), qualification)

    if not qualification["qualified"]:
        print(json.dumps(qualification, indent=2, sort_keys=True))
        return 1

    source_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    cressim = runtime["artifacts"]["cressim_mpm_c_api"]
    promotion = {
        "schema": "dr.anmar.tissue-entry-promotion-lock.v1",
        "task": qualification["task"],
        "source_revision": source_revision,
        "policy_sha256": _sha256(args.policy.resolve()),
        "backend_provider": "cressim_mpm",
        "backend_revision": runtime["sources"]["cressim_mpm"],
        "backend_library_sha256": cressim["sha256"],
        "tissue_profile": "dr-anmar-suturable-tissue-v1",
        "needle_profile": "dr-anmar-needle-v1",
        "evaluation_seeds": [result["seed"] for result in seed_results],
        "qualification_sha256": _sha256(args.output.resolve()),
        "clinical_validation": False,
        "evidence_level": "simulator_engineering_only"
    }
    _atomic_json(args.promotion_lock.resolve(), promotion)
    print(json.dumps(promotion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
