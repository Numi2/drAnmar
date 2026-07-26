#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the revision-bound evidence index for the Dr.Anmar portfolio."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PORTFOLIO_PATH = REPOSITORY_ROOT / "physics_next/dr-anmar-assets.json"
OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "physics_next/benchmarks/dranmar-portfolio-evidence-index.json"
)
ASSET_SUBMODULE = REPOSITORY_ROOT / "source/extensions/orbit.surgical.assets"
SCHEMA = "dr.anmar.portfolio-evidence-index.v1"
CLAIM_FIELDS = (
    "product_capability",
    "training_readiness",
    "software_evidence",
    "native_simulator_evidence",
    "real_world_evidence",
    "clinical_validation",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments],
        text=True,
    ).strip()


def is_declared_artifact(repository_root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or "/" not in value:
        return None
    candidate = (repository_root / value).resolve()
    try:
        candidate.relative_to(repository_root.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def evidence_summary(path: Path) -> dict[str, Any] | None:
    if path.suffix.lower() != ".json":
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    summary = {}
    for key in (
        "schema",
        "passed",
        "qualified",
        "overall_qualified",
        "evidence_result",
        "clinical_validation",
        "promotion_allowed",
        "tested_commit",
        "tested_parent_revision",
        "tested_asset_submodule_revision",
    ):
        if key in payload:
            summary[key] = payload[key]
    for key in (
        "evidence_not_established",
        "qualification_boundary",
        "representation_boundary",
    ):
        if key in payload:
            summary[key] = payload[key]
    return summary or None


def build_index(
    repository_root: Path,
    *,
    source_parent_revision: str,
    source_submodule_revision: str,
) -> dict[str, Any]:
    portfolio_path = repository_root / PORTFOLIO_PATH.relative_to(REPOSITORY_ROOT)
    portfolio = json.loads(portfolio_path.read_text(encoding="utf-8"))
    assets = []
    for entry in portfolio["assets"]:
        artifacts = []
        for role, value in sorted(entry.items()):
            path = is_declared_artifact(repository_root, value)
            if path is None:
                continue
            relative = path.relative_to(repository_root).as_posix()
            authority = (
                "asset_submodule"
                if relative.startswith("source/extensions/orbit.surgical.assets/")
                else "parent_repository"
            )
            artifact = {
                "role": role,
                "path": relative,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "revision_authority": authority,
                "containing_revision": (
                    source_submodule_revision
                    if authority == "asset_submodule"
                    else source_parent_revision
                ),
            }
            if role in {"native_evidence", "report", "qualification"}:
                summary = evidence_summary(path)
                if summary is not None:
                    artifact["machine_readable_summary"] = summary
            artifacts.append(artifact)
        assets.append(
            {
                "id": entry["id"],
                "claims": {
                    field: entry.get(field)
                    for field in CLAIM_FIELDS
                },
                "artifacts": artifacts,
                "artifact_count": len(artifacts),
                "all_declared_artifacts_content_addressed": bool(artifacts),
            }
        )
    return {
        "schema": SCHEMA,
        "source": {
            "parent_revision": source_parent_revision,
            "asset_submodule_revision": source_submodule_revision,
            "portfolio_path": portfolio_path.relative_to(repository_root).as_posix(),
            "portfolio_sha256": sha256_file(portfolio_path),
            "generator_path": Path(__file__).resolve().relative_to(repository_root).as_posix(),
            "generator_sha256": sha256_file(Path(__file__).resolve()),
        },
        "claim_policy": {
            "product_capability": "repository integration claim",
            "software_evidence": "repository verification only",
            "native_simulator_evidence": "requires an asset-specific native_evidence artifact",
            "real_world_evidence": "requires instrumented physical correlation",
            "clinical_validation": "false unless a scoped clinical artifact is retained",
            "generic_bench_matrix_exclusion": (
                "startup, selection, stepping, camera output, diagnostics, and "
                "shutdown are not complete procedure or deformable evidence"
            ),
        },
        "asset_count": len(assets),
        "assets": assets,
    }


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--source-parent-revision")
    arguments = parser.parse_args()

    if arguments.check:
        retained = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        parent_revision = str(retained["source"]["parent_revision"])
        submodule_revision = str(retained["source"]["asset_submodule_revision"])
        subprocess.run(
            [
                "git",
                "-C",
                str(REPOSITORY_ROOT),
                "merge-base",
                "--is-ancestor",
                parent_revision,
                "HEAD",
            ],
            check=True,
        )
        if git_output(ASSET_SUBMODULE, "rev-parse", "HEAD") != submodule_revision:
            raise SystemExit("asset submodule revision differs from retained evidence")
    else:
        parent_revision = git_output(
            REPOSITORY_ROOT,
            "rev-parse",
            arguments.source_parent_revision or "HEAD",
        )
        submodule_revision = git_output(ASSET_SUBMODULE, "rev-parse", "HEAD")

    generated = canonical_bytes(
        build_index(
            REPOSITORY_ROOT,
            source_parent_revision=parent_revision,
            source_submodule_revision=submodule_revision,
        )
    )
    if arguments.check:
        if generated != OUTPUT_PATH.read_bytes():
            raise SystemExit("portfolio evidence index is stale")
        print("Portfolio evidence index is content-consistent.")
        return 0
    OUTPUT_PATH.write_bytes(generated)
    print(OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
