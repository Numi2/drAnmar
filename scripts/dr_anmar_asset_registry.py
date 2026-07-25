#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Catalog, resolve, verify, and lock every Dr.Anmar simulation asset family.

The design follows the i4h asset-catalog contract: stable provider-relative
paths, a pinned external content release, lazy provider roots, deterministic
folder hashes, and a small CLI that works without importing Isaac Sim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
POLICY_RELATIVE_PATH = Path("config/dranmar_asset_catalog.json")
POLICY_SCHEMA = "dr.anmar.asset-catalog-policy.v1"
LOCK_SCHEMA = "dr.anmar.asset-catalog-lock.v1"
USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc"})
_USD_REFERENCE = re.compile(r"@([^@\r\n]+)@")
_HASH_PATTERN = re.compile(r"^[0-9a-f]{7,64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SKIP_HASH_NAMES = frozenset({".DS_Store", ".gitattributes", ".gitignore"})
_SKIP_HASH_DIRS = frozenset({".git", "__pycache__"})


@dataclass(frozen=True)
class CatalogIssue:
    severity: str
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class AssetUnit:
    asset_id: str
    provider: str
    relative_path: str
    entrypoints: tuple[str, ...]
    metadata: tuple[str, ...]
    license_path: str | None
    file_count: int
    bytes: int
    sha256: str | None = None


def load_policy(repository_root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    """Load the repository's authoritative asset-catalog policy."""

    policy_path = repository_root / POLICY_RELATIVE_PATH
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    if payload.get("schema") != POLICY_SCHEMA:
        raise ValueError(f"Unsupported asset-catalog policy schema in {policy_path}")
    return payload


def i4h_provider(policy: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the pinned external i4h provider contract."""

    provider = policy.get("providers", {}).get("nvidia_i4h")
    if not isinstance(provider, Mapping):
        raise TypeError("The asset policy does not define providers.nvidia_i4h")
    return provider


def _safe_relative_path(value: str | os.PathLike[str]) -> Path:
    """Convert one provider-relative POSIX path and reject traversal."""

    rendered = os.fspath(value)
    if not rendered or "\x00" in rendered or "\\" in rendered:
        raise ValueError(f"Invalid provider-relative asset path: {rendered!r}")
    pure = PurePosixPath(rendered)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"Asset path must be a normalized relative path: {rendered!r}")
    if pure.parts and ":" in pure.parts[0]:
        raise ValueError(f"Asset path must not contain a drive or URI: {rendered!r}")
    return Path(*pure.parts)


def _contains(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def provider_roots(
    repository_root: Path = REPOSITORY_ROOT,
    *,
    i4h_content_root: Path | None = None,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Resolve local and external catalog providers using one shared contract."""

    repository_root = repository_root.expanduser().resolve()
    policy = load_policy(repository_root) if policy is None else policy
    providers = policy.get("providers", {})
    roots: dict[str, Path] = {}
    for provider_id, provider in providers.items():
        if not isinstance(provider, Mapping):
            raise TypeError(f"Invalid provider contract: {provider_id}")
        if provider.get("kind") == "local":
            relative_root = (
                _safe_relative_path(str(provider.get("root", ""))) if provider.get("root") != "." else Path()
            )
            roots[str(provider_id)] = (repository_root / relative_root).resolve()

    upstream = i4h_provider(policy)
    if i4h_content_root is None:
        environment_name = str(
            upstream.get(
                "content_root_environment",
                "DR_ANMAR_I4H_ASSET_CONTENT_ROOT",
            )
        )
        configured = os.environ.get(environment_name)
        if configured:
            i4h_content_root = Path(configured)
        else:
            application_root = Path(
                os.environ.get(
                    "DR_ANMAR_ROOT",
                    Path.home() / ".local/share/dr-anmar",
                )
            )
            cache_subpath = _safe_relative_path(str(upstream.get("default_cache_subpath", "assets/i4h-catalog")))
            i4h_content_root = application_root / cache_subpath / str(upstream["content_hash"])
    roots["nvidia_i4h"] = i4h_content_root.expanduser().resolve()
    return roots


def resolve_provider_asset(
    provider: str,
    relative_path: str | os.PathLike[str],
    roots: Mapping[str, Path],
    *,
    require: bool = False,
) -> Path:
    """Resolve one catalog path without allowing it to escape its provider."""

    if provider not in roots:
        raise ValueError(f"Unknown Dr.Anmar asset provider: {provider}")
    root = roots[provider].expanduser().resolve()
    relative = _safe_relative_path(relative_path)
    candidate = (root / relative).resolve()
    if not _contains(root, candidate):
        raise ValueError(f"Asset path {relative.as_posix()!r} escapes provider {provider!r}")
    if require and not candidate.is_file():
        raise FileNotFoundError(f"Missing {provider} asset {relative.as_posix()}: {candidate}")
    return candidate


def sha256_of_folder(folder: Path, *, ignored_names: Iterable[str] = ()) -> str:
    """Hash sorted relative paths and bytes, matching the i4h catalog approach."""

    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"Asset folder does not exist: {folder}")
    ignored = _SKIP_HASH_NAMES | frozenset(ignored_names)
    digest = hashlib.sha256()
    for current_root, directories, filenames in os.walk(folder):
        directories[:] = sorted(name for name in directories if name not in _SKIP_HASH_DIRS)
        for filename in sorted(filenames):
            if filename in ignored:
                continue
            file_path = Path(current_root) / filename
            if file_path.is_symlink():
                target = os.readlink(file_path)
                relative = file_path.relative_to(folder).as_posix()
                digest.update(relative.encode("utf-8"))
                digest.update(b"\0symlink\0")
                digest.update(target.encode("utf-8"))
                continue
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(folder).as_posix()
            digest.update(relative.encode("utf-8"))
            with file_path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
    return digest.hexdigest()


def _inventory_search_roots(
    repository_root: Path,
    policy: Mapping[str, Any],
) -> list[tuple[str, Path, Path]]:
    roots = provider_roots(repository_root, policy=policy)
    searches: list[tuple[str, Path, Path]] = []
    for provider_id, provider in policy.get("providers", {}).items():
        if not isinstance(provider, Mapping) or provider.get("kind") != "local":
            continue
        provider_root = roots[str(provider_id)]
        for relative in provider.get("inventory_roots", ()):
            inventory_relative = Path() if relative == "." else _safe_relative_path(str(relative))
            search_root = (provider_root / inventory_relative).resolve()
            if not _contains(provider_root, search_root):
                raise ValueError(f"Inventory root escapes provider {provider_id}: {relative}")
            searches.append((str(provider_id), provider_root, search_root))
    return searches


def _asset_unit_directories(search_root: Path) -> tuple[Path, ...]:
    candidates = {
        path.parent for path in search_root.rglob("*") if path.is_file() and path.suffix.lower() in USD_SUFFIXES
    }
    return tuple(
        sorted(
            (candidate for candidate in candidates if not any(parent in candidates for parent in candidate.parents)),
            key=lambda item: item.as_posix(),
        )
    )


def _license_for_unit(
    unit_root: Path,
    provider_root: Path,
    repository_root: Path,
    provider: Mapping[str, Any],
) -> Path | None:
    current = unit_root
    while _contains(provider_root, current):
        for name in ("LICENSE", "LICENSE.md", "LICENSE.txt"):
            candidate = current / name
            if candidate.is_file():
                return candidate
        if current == provider_root:
            break
        current = current.parent
    fallback = provider.get("license_fallback")
    if fallback:
        candidate = (repository_root / _safe_relative_path(str(fallback))).resolve()
        if candidate.is_file() and _contains(repository_root, candidate):
            return candidate
    return None


def discover_asset_units(
    repository_root: Path = REPOSITORY_ROOT,
    *,
    include_hashes: bool = False,
    policy: Mapping[str, Any] | None = None,
) -> tuple[AssetUnit, ...]:
    """Discover each local folder that directly owns one or more USD stages."""

    repository_root = repository_root.expanduser().resolve()
    policy = load_policy(repository_root) if policy is None else policy
    ignored_names = frozenset(str(name) for name in policy.get("quality", {}).get("ignored_names", ()))
    units: list[AssetUnit] = []
    for provider_id, provider_root, search_root in _inventory_search_roots(repository_root, policy):
        provider = policy["providers"][provider_id]
        if not search_root.is_dir():
            continue
        for unit_root in _asset_unit_directories(search_root):
            files = tuple(
                sorted(
                    (path for path in unit_root.rglob("*") if path.is_file() and path.name not in ignored_names),
                    key=lambda item: item.as_posix(),
                )
            )
            entrypoints = tuple(
                path.relative_to(unit_root).as_posix()
                for path in files
                if path.parent == unit_root and path.suffix.lower() in USD_SUFFIXES
            )
            metadata = tuple(
                path.relative_to(unit_root).as_posix()
                for path in files
                if path.parent == unit_root and path.suffix.lower() in {".json", ".md", ".txt"}
            )
            relative_root = unit_root.relative_to(provider_root).as_posix()
            license_path = _license_for_unit(
                unit_root,
                provider_root,
                repository_root,
                provider,
            )
            units.append(
                AssetUnit(
                    asset_id=f"{provider_id}:{relative_root}",
                    provider=provider_id,
                    relative_path=relative_root,
                    entrypoints=entrypoints,
                    metadata=metadata,
                    license_path=(
                        license_path.relative_to(repository_root).as_posix()
                        if license_path is not None and _contains(repository_root, license_path)
                        else str(license_path) if license_path is not None else None
                    ),
                    file_count=len(files),
                    bytes=sum(path.stat().st_size for path in files),
                    sha256=(
                        sha256_of_folder(
                            unit_root,
                            ignored_names=ignored_names,
                        )
                        if include_hashes
                        else None
                    ),
                )
            )
    return tuple(sorted(units, key=lambda unit: unit.asset_id))


def _textual_usd(path: Path) -> str | None:
    try:
        with path.open("rb") as stream:
            prefix = stream.read(16)
            if not prefix.lstrip().startswith(b"#usda"):
                return None
            return prefix.decode("utf-8") + stream.read().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _issue(
    issues: list[CatalogIssue],
    severity: str,
    code: str,
    path: Path | str,
    message: str,
    repository_root: Path,
) -> None:
    rendered_path = str(path)
    if isinstance(path, Path) and _contains(repository_root, path.resolve()):
        rendered_path = path.resolve().relative_to(repository_root).as_posix()
    issues.append(CatalogIssue(severity, code, rendered_path, message))


def _validate_policy(
    policy: Mapping[str, Any],
    repository_root: Path,
    issues: list[CatalogIssue],
) -> None:
    upstream = i4h_provider(policy)
    if not _COMMIT_PATTERN.fullmatch(str(upstream.get("catalog_commit", ""))):
        _issue(
            issues,
            "error",
            "i4h_commit_pin",
            POLICY_RELATIVE_PATH,
            "The i4h catalog commit must be a full 40-character Git SHA.",
            repository_root,
        )
    if not _HASH_PATTERN.fullmatch(str(upstream.get("content_hash", ""))):
        _issue(
            issues,
            "error",
            "i4h_content_hash",
            POLICY_RELATIVE_PATH,
            "The i4h content hash must be a lowercase hexadecimal prefix or SHA-256.",
            repository_root,
        )
    bundles = policy.get("i4h_bundles", {})
    for bundle_name, paths in bundles.items():
        if not isinstance(paths, list) or not paths:
            _issue(
                issues,
                "error",
                "empty_i4h_bundle",
                POLICY_RELATIVE_PATH,
                f"i4h bundle {bundle_name!r} must contain at least one subpath.",
                repository_root,
            )
            continue
        for relative in paths:
            try:
                _safe_relative_path(str(relative))
            except ValueError as error:
                _issue(
                    issues,
                    "error",
                    "unsafe_i4h_bundle_path",
                    POLICY_RELATIVE_PATH,
                    str(error),
                    repository_root,
                )


def _validate_json_and_usd(
    repository_root: Path,
    policy: Mapping[str, Any],
    issues: list[CatalogIssue],
) -> None:
    allowed_schemes = frozenset(
        str(value) for value in policy.get("quality", {}).get("allowed_external_usd_schemes", ())
    )
    ignored_names = frozenset(str(value) for value in policy.get("quality", {}).get("ignored_names", ()))
    for _provider_id, _provider_root, search_root in _inventory_search_roots(repository_root, policy):
        if not search_root.is_dir():
            _issue(
                issues,
                "error",
                "missing_inventory_root",
                search_root,
                "Configured local asset inventory root does not exist.",
                repository_root,
            )
            continue
        for path in sorted(search_root.rglob("*"), key=lambda item: item.as_posix()):
            if path.name in ignored_names:
                continue
            if path.is_symlink():
                resolved = path.resolve()
                if not _contains(repository_root, resolved):
                    _issue(
                        issues,
                        "error",
                        "escaping_symlink",
                        path,
                        f"Asset symlink escapes the repository: {resolved}",
                        repository_root,
                    )
                continue
            if not path.is_file():
                continue
            if path.suffix.lower() == ".json":
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                    _issue(
                        issues,
                        "error",
                        "invalid_json",
                        path,
                        str(error),
                        repository_root,
                    )
                else:
                    if isinstance(payload, Mapping) and payload.get("schema") == "dr.anmar.asset-unit.v1":
                        manifest_references: list[str] = []
                        for key in (
                            "license_file",
                            "notice_file",
                            "physics_profile",
                            "validation_report",
                            "gpu_qualification",
                            "external_geometry_dependency",
                            "suture_dependency",
                        ):
                            if payload.get(key):
                                manifest_references.append(str(payload[key]))
                        manifest_references.extend(str(value) for value in payload.get("entrypoints", {}).values())
                        manifest_references.extend(str(value) for value in payload.get("layers", ()))
                        for reference in manifest_references:
                            referenced = Path(reference)
                            resolved = (path.parent / referenced).resolve()
                            if referenced.is_absolute() or not _contains(repository_root, resolved):
                                _issue(
                                    issues,
                                    "error",
                                    "unsafe_manifest_reference",
                                    path,
                                    f"Manifest reference escapes the repository: {reference}",
                                    repository_root,
                                )
                            elif not resolved.is_file():
                                _issue(
                                    issues,
                                    "error",
                                    "missing_manifest_reference",
                                    path,
                                    f"Manifest reference does not exist: {reference}",
                                    repository_root,
                                )
            if path.suffix.lower() not in USD_SUFFIXES:
                continue
            source = _textual_usd(path)
            if source is None:
                continue
            for reference in _USD_REFERENCE.findall(source):
                parsed = urlparse(reference)
                if parsed.scheme:
                    if parsed.scheme not in allowed_schemes:
                        _issue(
                            issues,
                            "error",
                            "unsupported_usd_scheme",
                            path,
                            f"Unsupported external USD reference: {reference}",
                            repository_root,
                        )
                    continue
                referenced = Path(reference)
                if referenced.is_absolute():
                    _issue(
                        issues,
                        "error",
                        "absolute_usd_reference",
                        path,
                        f"Absolute USD dependency is not portable: {reference}",
                        repository_root,
                    )
                    continue
                resolved = (path.parent / referenced).resolve()
                if not _contains(repository_root, resolved):
                    _issue(
                        issues,
                        "error",
                        "escaping_usd_reference",
                        path,
                        f"USD dependency escapes the repository: {reference}",
                        repository_root,
                    )
                elif not resolved.is_file():
                    _issue(
                        issues,
                        "error",
                        "missing_usd_dependency",
                        path,
                        f"USD dependency does not exist: {reference}",
                        repository_root,
                    )


def _requires_metadata(
    unit: AssetUnit,
    policy: Mapping[str, Any],
) -> bool:
    prefixes = policy.get("quality", {}).get("metadata_required_prefixes", {}).get(unit.provider, ())
    return any(unit.relative_path == prefix or unit.relative_path.startswith(f"{prefix}/") for prefix in prefixes)


def _validate_unit_metadata(
    units: Sequence[AssetUnit],
    policy: Mapping[str, Any],
    repository_root: Path,
    issues: list[CatalogIssue],
) -> None:
    contract_names = frozenset(str(value) for value in policy.get("quality", {}).get("metadata_contract_names", ()))
    seen: set[str] = set()
    for unit in units:
        if unit.asset_id in seen:
            _issue(
                issues,
                "error",
                "duplicate_asset_id",
                unit.relative_path,
                f"Duplicate asset ID: {unit.asset_id}",
                repository_root,
            )
        seen.add(unit.asset_id)
        if not unit.entrypoints:
            _issue(
                issues,
                "error",
                "missing_entrypoint",
                unit.relative_path,
                "Asset unit has no direct USD entrypoint.",
                repository_root,
            )
        if not _requires_metadata(unit, policy):
            continue
        if unit.license_path is None:
            _issue(
                issues,
                "error",
                "missing_license",
                unit.relative_path,
                "Curated asset unit has no local or provider-fallback license.",
                repository_root,
            )
        if not contract_names.intersection(unit.metadata):
            _issue(
                issues,
                "error",
                "missing_asset_contract",
                unit.relative_path,
                "Curated asset unit needs a manifest, physics profile, or qualification contract.",
                repository_root,
            )
        if "README.md" not in unit.metadata:
            _issue(
                issues,
                "warning",
                "missing_asset_readme",
                unit.relative_path,
                "Curated asset unit has no README.md at its catalog root.",
                repository_root,
            )


def validate_procedure_catalog(
    procedures: Sequence[Mapping[str, Any]],
    roots: Mapping[str, Path],
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[CatalogIssue, ...]:
    """Validate every runtime room reference against the shared providers."""

    repository_root = repository_root.expanduser().resolve()
    issues: list[CatalogIssue] = []
    seen_procedures: set[str] = set()
    related_keys = ("payload_path", "rigid_proxy_path", "auxiliary_path")
    for procedure in procedures:
        procedure_id = str(procedure.get("id", ""))
        if not procedure_id or procedure_id in seen_procedures:
            _issue(
                issues,
                "error",
                "duplicate_procedure_id",
                "scripts/dr_anmar_procedures.py",
                f"Missing or duplicate procedure ID: {procedure_id!r}",
                repository_root,
            )
        seen_procedures.add(procedure_id)
        seen_assets: set[str] = set()
        for item in procedure.get("bench_asset_catalog", ()):
            asset_id = str(item.get("id", ""))
            if not asset_id or asset_id in seen_assets:
                _issue(
                    issues,
                    "error",
                    "duplicate_bench_asset_id",
                    "scripts/dr_anmar_procedures.py",
                    f"{procedure_id}: missing or duplicate bench asset ID {asset_id!r}",
                    repository_root,
                )
            seen_assets.add(asset_id)
            provider = str(item.get("provider", "nvidia_i4h"))
            for key in ("path", *related_keys):
                if not item.get(key):
                    continue
                try:
                    resolved = resolve_provider_asset(
                        provider,
                        str(item[key]),
                        roots,
                        require=False,
                    )
                except ValueError as error:
                    _issue(
                        issues,
                        "error",
                        "unsafe_runtime_asset",
                        "scripts/dr_anmar_procedures.py",
                        f"{procedure_id}/{asset_id}/{key}: {error}",
                        repository_root,
                    )
                    continue
                if provider != "nvidia_i4h" and not resolved.is_file():
                    _issue(
                        issues,
                        "error",
                        "missing_runtime_asset",
                        "scripts/dr_anmar_procedures.py",
                        f"{procedure_id}/{asset_id}/{key}: {resolved} is missing",
                        repository_root,
                    )
        for relative in procedure.get("required_repository_assets", ()):
            try:
                resolved = resolve_provider_asset(
                    "dr_anmar_repository",
                    str(relative),
                    roots,
                    require=False,
                )
            except ValueError as error:
                _issue(
                    issues,
                    "error",
                    "unsafe_required_asset",
                    "scripts/dr_anmar_procedures.py",
                    f"{procedure_id}: {error}",
                    repository_root,
                )
                continue
            if not resolved.is_file():
                _issue(
                    issues,
                    "error",
                    "missing_required_asset",
                    "scripts/dr_anmar_procedures.py",
                    f"{procedure_id}: {resolved} is missing",
                    repository_root,
                )
        for relative in procedure.get("required_nvidia_assets", ()):
            try:
                resolve_provider_asset(
                    "nvidia_i4h",
                    str(relative),
                    roots,
                    require=False,
                )
            except ValueError as error:
                _issue(
                    issues,
                    "error",
                    "unsafe_required_i4h_asset",
                    "scripts/dr_anmar_procedures.py",
                    f"{procedure_id}: {error}",
                    repository_root,
                )
    return tuple(issues)


def validate_catalog(
    repository_root: Path = REPOSITORY_ROOT,
    *,
    procedures: Sequence[Mapping[str, Any]] | None = None,
    i4h_content_root: Path | None = None,
) -> dict[str, Any]:
    """Run the simulator-independent catalog gate and return a JSON-ready report."""

    repository_root = repository_root.expanduser().resolve()
    policy = load_policy(repository_root)
    units = discover_asset_units(repository_root, policy=policy)
    issues: list[CatalogIssue] = []
    _validate_policy(policy, repository_root, issues)
    _validate_json_and_usd(repository_root, policy, issues)
    _validate_unit_metadata(units, policy, repository_root, issues)
    if procedures is not None:
        roots = provider_roots(
            repository_root,
            i4h_content_root=i4h_content_root,
            policy=policy,
        )
        issues.extend(
            validate_procedure_catalog(
                procedures,
                roots,
                repository_root=repository_root,
            )
        )
    issues.sort(key=lambda item: (item.severity, item.code, item.path, item.message))
    errors = sum(issue.severity == "error" for issue in issues)
    warnings = sum(issue.severity == "warning" for issue in issues)
    return {
        "schema": "dr.anmar.asset-catalog-validation.v1",
        "catalog_version": policy["catalog_version"],
        "passed": errors == 0,
        "asset_units": len(units),
        "entrypoints": sum(len(unit.entrypoints) for unit in units),
        "files": sum(unit.file_count for unit in units),
        "bytes": sum(unit.bytes for unit in units),
        "errors": errors,
        "warnings": warnings,
        "issues": [asdict(issue) for issue in issues],
    }


def build_lock(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Build a deterministic release lock for all bundled asset units."""

    repository_root = repository_root.expanduser().resolve()
    policy = load_policy(repository_root)
    upstream = i4h_provider(policy)
    units = discover_asset_units(
        repository_root,
        include_hashes=True,
        policy=policy,
    )
    return {
        "schema": LOCK_SCHEMA,
        "catalog_version": policy["catalog_version"],
        "i4h": {
            "release": upstream["release"],
            "catalog_commit": upstream["catalog_commit"],
            "asset_version": upstream["asset_version"],
            "content_hash": upstream["content_hash"],
        },
        "assets": [
            {
                "id": unit.asset_id,
                "entrypoints": list(unit.entrypoints),
                "file_count": unit.file_count,
                "bytes": unit.bytes,
                "sha256": unit.sha256,
            }
            for unit in units
        ],
    }


def verify_lock(
    lock: Mapping[str, Any],
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[str, ...]:
    """Compare a previously generated release lock with the current tree."""

    if lock.get("schema") != LOCK_SCHEMA:
        return ("Unsupported or missing asset-catalog lock schema.",)
    current = build_lock(repository_root)
    failures: list[str] = []
    if lock.get("catalog_version") != current["catalog_version"]:
        failures.append("Catalog version does not match the lock.")
    if lock.get("i4h") != current["i4h"]:
        failures.append("Pinned i4h provider does not match the lock.")
    expected_assets = {str(asset["id"]): asset for asset in lock.get("assets", ())}
    current_assets = {str(asset["id"]): asset for asset in current.get("assets", ())}
    for asset_id in sorted(expected_assets.keys() | current_assets.keys()):
        if expected_assets.get(asset_id) != current_assets.get(asset_id):
            failures.append(f"Asset unit changed: {asset_id}")
    return tuple(failures)


def _procedure_rooms() -> Sequence[Mapping[str, Any]]:
    from dr_anmar_procedures import PROCEDURE_ROOMS

    return PROCEDURE_ROOMS


def _json_dump(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dr.Anmar asset catalog, resolver, verifier, and release lock")
    parser.add_argument(
        "--root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="Dr.Anmar repository root",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="List local asset units")
    inventory.add_argument(
        "--hash",
        action="store_true",
        help="Include deterministic folder SHA-256 values",
    )

    verify = subparsers.add_parser("verify", help="Run the repository catalog gate")
    verify.add_argument(
        "--lock",
        type=Path,
        help="Also verify a generated asset-catalog lock",
    )

    resolve = subparsers.add_parser("resolve", help="Resolve one provider-relative path")
    resolve.add_argument("provider")
    resolve.add_argument("path")
    resolve.add_argument("--i4h-root", type=Path)
    resolve.add_argument("--require", action="store_true")

    lock = subparsers.add_parser("lock", help="Generate a deterministic release lock")
    lock.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = args.root.expanduser().resolve()
    if args.command == "inventory":
        units = discover_asset_units(
            repository_root,
            include_hashes=bool(args.hash),
        )
        _json_dump(
            {
                "schema": "dr.anmar.asset-catalog-inventory.v1",
                "asset_units": [asdict(unit) for unit in units],
            }
        )
        return 0
    if args.command == "verify":
        report = validate_catalog(
            repository_root,
            procedures=_procedure_rooms(),
        )
        if args.lock:
            lock = json.loads(args.lock.read_text(encoding="utf-8"))
            lock_failures = verify_lock(lock, repository_root)
            report["lock_passed"] = not lock_failures
            report["lock_failures"] = list(lock_failures)
            report["passed"] = bool(report["passed"] and not lock_failures)
        _json_dump(report)
        return 0 if report["passed"] else 1
    if args.command == "resolve":
        roots = provider_roots(
            repository_root,
            i4h_content_root=args.i4h_root,
        )
        print(
            resolve_provider_asset(
                args.provider,
                args.path,
                roots,
                require=bool(args.require),
            )
        )
        return 0
    if args.command == "lock":
        payload = build_lock(repository_root)
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
        print(output)
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
