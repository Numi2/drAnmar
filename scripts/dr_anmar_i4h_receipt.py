#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Record and verify content-addressed receipts for partial NVIDIA i4h bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from dr_anmar_asset_registry import i4h_provider, load_policy


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RECEIPT_SCHEMA = "dr.anmar.i4h-asset-installation.v2"
LEGACY_RECEIPT_SCHEMA = "dr.anmar.i4h-asset-catalog-installation.v1"
_SKIP_NAMES = frozenset({".DS_Store", ".gitattributes", ".gitignore"})
_SKIP_DIRS = frozenset({".git", "__pycache__"})


def _safe_subpath(value: str) -> Path:
    pure = PurePosixPath(value)
    if (
        not value
        or "\x00" in value
        or "\\" in value
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"Unsafe i4h asset subpath: {value!r}")
    return Path(*pure.parts)


def _contains(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _iter_hashed_files(target: Path) -> tuple[tuple[str, Path], ...]:
    if target.is_file():
        return ((target.name, target),)
    if not target.is_dir():
        raise FileNotFoundError(f"i4h bundle subpath is missing: {target}")
    entries: list[tuple[str, Path]] = []
    for current_root, directories, filenames in os.walk(target):
        directories[:] = sorted(name for name in directories if name not in _SKIP_DIRS)
        for filename in sorted(filenames):
            if filename in _SKIP_NAMES:
                continue
            file_path = Path(current_root) / filename
            entries.append((file_path.relative_to(target).as_posix(), file_path))
    if not entries:
        raise ValueError(f"i4h bundle subpath contains no files: {target}")
    return tuple(sorted(entries))


def hash_bundle_subpath(content_root: Path, subpath: str) -> dict[str, Any]:
    """Hash one downloaded file or dependency-complete directory."""

    content_root = content_root.expanduser().resolve()
    relative = _safe_subpath(subpath)
    target = (content_root / relative).resolve()
    if not _contains(content_root, target):
        raise ValueError(f"i4h subpath escapes the content root: {subpath!r}")
    entries = _iter_hashed_files(target)
    digest = hashlib.sha256()
    total_bytes = 0
    for relative_name, file_path in entries:
        if file_path.is_symlink():
            resolved = file_path.resolve()
            if not _contains(content_root, resolved):
                raise ValueError(f"i4h symlink escapes the content root: {file_path}")
            target_text = os.readlink(file_path)
            digest.update(relative_name.encode("utf-8"))
            digest.update(b"\0symlink\0")
            digest.update(target_text.encode("utf-8"))
            total_bytes += len(target_text.encode("utf-8"))
            continue
        digest.update(relative_name.encode("utf-8"))
        with file_path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                total_bytes += len(chunk)
    return {
        "path": relative.as_posix(),
        "kind": "file" if target.is_file() else "directory",
        "file_count": len(entries),
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def _receipt_pins(policy: Mapping[str, Any]) -> dict[str, Any]:
    provider = i4h_provider(policy)
    return {
        "catalog_release": provider["release"],
        "catalog_commit": provider["catalog_commit"],
        "asset_version": provider["asset_version"],
        "asset_hash": provider["content_hash"],
    }


def update_receipt(
    policy: Mapping[str, Any],
    content_root: Path,
    bundle: str,
    *,
    existing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record or refresh one bundle while retaining compatible prior bundles."""

    bundles = policy.get("i4h_bundles", {})
    requested = bundles.get(bundle)
    if not isinstance(requested, list) or not requested:
        raise ValueError(f"Unknown or empty i4h bundle: {bundle!r}")
    content_root = content_root.expanduser().resolve()
    if not content_root.is_dir():
        raise FileNotFoundError(f"i4h content root is missing: {content_root}")
    pins = _receipt_pins(policy)
    retained: dict[str, Any] = {}
    if existing is not None:
        if existing.get("schema") != RECEIPT_SCHEMA:
            raise ValueError("Existing i4h receipt has an unsupported schema.")
        if any(existing.get(key) != value for key, value in pins.items()):
            raise ValueError("Existing i4h receipt uses a different catalog pin.")
        if Path(str(existing.get("content_root", ""))).expanduser().resolve() != content_root:
            raise ValueError("Existing i4h receipt uses a different content root.")
        existing_bundles = existing.get("bundles", {})
        if not isinstance(existing_bundles, Mapping):
            raise ValueError("Existing i4h receipt bundles must be an object.")
        retained = {str(key): value for key, value in existing_bundles.items()}

    recorded_at = datetime.now(timezone.utc).isoformat()
    retained[bundle] = {
        "requested_subpaths": [str(value) for value in requested],
        "artifacts": [hash_bundle_subpath(content_root, str(value)) for value in requested],
        "recorded_at": recorded_at,
    }
    all_files = tuple(path for path in content_root.rglob("*") if path.is_file() and path.name not in _SKIP_NAMES)
    return {
        "schema": RECEIPT_SCHEMA,
        "provider": "NVIDIA Isaac for Healthcare asset catalog",
        **pins,
        "content_root": str(content_root),
        "bundles": dict(sorted(retained.items())),
        "cache_file_count": len(all_files),
        "cache_bytes": sum(path.stat().st_size for path in all_files),
        "recorded_at": recorded_at,
        "license_review_required": True,
        "clinical_validation": False,
    }


def verify_receipt(
    receipt: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[str, ...]:
    """Verify all recorded bundle closures against their current local bytes."""

    failures: list[str] = []
    if receipt.get("schema") != RECEIPT_SCHEMA:
        return ("Unsupported or missing i4h installation receipt schema.",)
    for key, value in _receipt_pins(policy).items():
        if receipt.get(key) != value:
            failures.append(f"Receipt pin mismatch: {key}")
    if receipt.get("license_review_required") is not True:
        failures.append("Receipt must retain the provider license-review requirement.")
    if receipt.get("clinical_validation") is not False:
        failures.append("Receipt clinical_validation must be exactly false.")
    content_root = Path(str(receipt.get("content_root", ""))).expanduser().resolve()
    if not content_root.is_dir():
        failures.append(f"Receipt content root is missing: {content_root}")
        return tuple(failures)
    recorded_bundles = receipt.get("bundles")
    if not isinstance(recorded_bundles, Mapping) or not recorded_bundles:
        failures.append("Receipt must contain at least one recorded bundle.")
        return tuple(failures)

    policy_bundles = policy.get("i4h_bundles", {})
    for bundle_name, bundle in recorded_bundles.items():
        if not isinstance(bundle, Mapping):
            failures.append(f"Invalid receipt bundle: {bundle_name}")
            continue
        expected_subpaths = policy_bundles.get(bundle_name)
        if bundle.get("requested_subpaths") != expected_subpaths:
            failures.append(f"Bundle subpaths changed: {bundle_name}")
        artifacts = bundle.get("artifacts")
        if not isinstance(artifacts, list):
            failures.append(f"Bundle artifacts must be a list: {bundle_name}")
            continue
        artifact_paths = [str(artifact.get("path")) for artifact in artifacts if isinstance(artifact, Mapping)]
        if artifact_paths != expected_subpaths:
            failures.append(f"Bundle artifact coverage changed: {bundle_name}")
            continue
        for artifact in artifacts:
            try:
                current = hash_bundle_subpath(
                    content_root,
                    str(artifact["path"]),
                )
            except (KeyError, OSError, TypeError, ValueError) as error:
                failures.append(f"{bundle_name}: {error}")
                continue
            if artifact != current:
                failures.append(f"Bundle subpath changed: {bundle_name}/{artifact['path']}")
    return tuple(failures)


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _default_receipt_path() -> Path:
    app_root = Path(
        os.environ.get(
            "DR_ANMAR_ROOT",
            Path.home() / ".local/share/dr-anmar",
        )
    ).expanduser()
    return app_root / "run/i4h_asset_catalog.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record and verify partial i4h asset bundle integrity")
    parser.add_argument(
        "--policy-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="Dr.Anmar repository containing the asset policy",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record")
    record.add_argument("--content-root", type=Path, required=True)
    record.add_argument("--bundle", required=True)
    record.add_argument("--output", type=Path, default=_default_receipt_path())

    verify = subparsers.add_parser("verify")
    verify.add_argument("--receipt", type=Path, default=_default_receipt_path())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    policy = load_policy(args.policy_root.expanduser().resolve())
    if args.command == "record":
        existing = None
        if args.output.is_file():
            existing = json.loads(args.output.read_text(encoding="utf-8"))
            if existing.get("schema") == LEGACY_RECEIPT_SCHEMA:
                existing = None
        receipt = update_receipt(
            policy,
            args.content_root,
            args.bundle,
            existing=existing,
        )
        _atomic_write(args.output, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    failures = verify_receipt(receipt, policy)
    print(
        json.dumps(
            {
                "schema": "dr.anmar.i4h-asset-installation-verification.v1",
                "passed": not failures,
                "receipt": str(args.receipt),
                "failures": list(failures),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
