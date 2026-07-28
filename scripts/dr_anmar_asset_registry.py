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
PORTFOLIO_RELATIVE_PATH = Path("physics_next/dr-anmar-assets.json")
POLICY_SCHEMA = "dr.anmar.asset-catalog-policy.v1"
PORTFOLIO_SCHEMA = "dr.anmar.asset-portfolio.v2"
LOCK_SCHEMA = "dr.anmar.asset-catalog-lock.v3"
USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc"})
DEFAULT_NON_USD_ENTRYPOINT_NAMES = frozenset({"asset_bundle.json"})
_USD_REFERENCE = re.compile(r"@([^@\r\n]+)@")
_HASH_PATTERN = re.compile(r"^[0-9a-f]{7,64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SKIP_HASH_NAMES = frozenset({".DS_Store", ".gitattributes", ".gitignore"})
_SKIP_HASH_DIRS = frozenset({".git", "__pycache__"})
MEMBER_MANIFEST_NAMES = frozenset({"asset_manifest.json", "visual_manifest.json"})
PORTFOLIO_PATH_FIELDS = frozenset(
    {
        "asset",
        "auxiliary_asset",
        "base_layer",
        "explicit_tetmesh",
        "geometry_layer",
        "gpu_report",
        "historical_native_evidence",
        "interaction_frames",
        "material_texture",
        "materials_layer",
        "native_evidence",
        "operating_scene",
        "payload_asset",
        "physics_layer",
        "physx_layer",
        "profile",
        "report",
        "rigid_proxy",
        "runtime",
        "task_contract",
        "training_contract",
        "workcell_asset",
    }
)
PORTFOLIO_REQUIRED_FIELDS = frozenset(
    {
        "id",
        "asset",
        "live_integration",
        "product_capability",
        "training_readiness",
        "software_evidence",
        "native_simulator_evidence",
        "real_world_evidence",
        "clinical_validation",
    }
)


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


def load_portfolio(repository_root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    """Load the authoritative product-facing asset portfolio."""

    portfolio_path = repository_root / PORTFOLIO_RELATIVE_PATH
    payload = json.loads(portfolio_path.read_text(encoding="utf-8"))
    if payload.get("schema") != PORTFOLIO_SCHEMA:
        raise ValueError(f"Unsupported asset portfolio schema in {portfolio_path}")
    if not isinstance(payload.get("assets"), list):
        raise TypeError(f"Asset portfolio entries must be a list in {portfolio_path}")
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


def resolve_repository_artifact(
    relative_path: str | os.PathLike[str],
    repository_root: Path = REPOSITORY_ROOT,
    *,
    require: bool = False,
) -> Path:
    """Resolve one normalized repository artifact and reject root escape."""

    repository_root = repository_root.expanduser().resolve()
    relative = _safe_relative_path(relative_path)
    candidate = (repository_root / relative).resolve()
    if not _contains(repository_root, candidate):
        raise ValueError(f"Repository artifact escapes the repository: {relative.as_posix()!r}")
    if require and not candidate.is_file():
        raise FileNotFoundError(f"Missing repository artifact {relative.as_posix()}: {candidate}")
    return candidate


def policy_release_path(
    policy: Mapping[str, Any],
    key: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    """Resolve one policy-controlled release artifact inside the repository."""

    value = policy.get("release", {}).get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Asset catalog policy is missing release.{key}")
    return resolve_repository_artifact(value, repository_root)


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


def _asset_unit_directories(
    search_root: Path,
    non_usd_entrypoint_names: frozenset[str],
) -> tuple[Path, ...]:
    candidates = {
        path.parent
        for path in search_root.rglob("*")
        if path.is_file()
        and (
            path.suffix.lower() in USD_SUFFIXES
            or path.name in non_usd_entrypoint_names
        )
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
    """Discover each local folder that owns a USD stage or data entrypoint."""

    repository_root = repository_root.expanduser().resolve()
    policy = load_policy(repository_root) if policy is None else policy
    ignored_names = frozenset(str(name) for name in policy.get("quality", {}).get("ignored_names", ()))
    non_usd_entrypoint_names = frozenset(
        str(name)
        for name in policy.get("quality", {}).get(
            "non_usd_entrypoint_names",
            DEFAULT_NON_USD_ENTRYPOINT_NAMES,
        )
    )
    units: list[AssetUnit] = []
    for provider_id, provider_root, search_root in _inventory_search_roots(repository_root, policy):
        provider = policy["providers"][provider_id]
        if not search_root.is_dir():
            continue
        for unit_root in _asset_unit_directories(search_root, non_usd_entrypoint_names):
            files = tuple(
                sorted(
                    (path for path in unit_root.rglob("*") if path.is_file() and path.name not in ignored_names),
                    key=lambda item: item.as_posix(),
                )
            )
            entrypoints = tuple(
                path.relative_to(unit_root).as_posix()
                for path in files
                if path.parent == unit_root
                and (
                    path.suffix.lower() in USD_SUFFIXES
                    or path.name in non_usd_entrypoint_names
                )
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


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_extension_root(manifest_path: Path) -> Path | None:
    """Return the local asset-extension root that owns one manifest."""

    for parent in manifest_path.resolve().parents:
        if parent.name == "orbit.surgical.assets":
            return parent
    return None


def _validate_manifest_file_receipt(
    *,
    manifest_path: Path,
    reference: str,
    receipt: Mapping[str, Any],
    base: Path,
    repository_root: Path,
    issues: list[CatalogIssue],
    role: str,
) -> None:
    try:
        relative = _safe_relative_path(reference)
    except ValueError as error:
        _issue(
            issues,
            "error",
            "unsafe_member_manifest_reference",
            manifest_path,
            f"{role} {reference!r}: {error}",
            repository_root,
        )
        return
    base = base.resolve()
    resolved = (base / relative).resolve()
    if not _contains(base, resolved) or not _contains(repository_root, resolved):
        _issue(
            issues,
            "error",
            "escaping_member_manifest_reference",
            manifest_path,
            f"{role} escapes its declared root: {reference}",
            repository_root,
        )
        return
    if not resolved.is_file():
        _issue(
            issues,
            "error",
            "missing_member_manifest_reference",
            manifest_path,
            f"{role} does not exist: {reference}",
            repository_root,
        )
        return

    expected_bytes = receipt.get("bytes")
    if expected_bytes is not None:
        if (
            not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes < 0
        ):
            _issue(
                issues,
                "error",
                "invalid_member_manifest_bytes",
                manifest_path,
                f"{role} has an invalid byte receipt: {reference}",
                repository_root,
            )
        elif resolved.stat().st_size != expected_bytes:
            _issue(
                issues,
                "error",
                "member_manifest_bytes_mismatch",
                manifest_path,
                (
                    f"{role} byte receipt does not match {reference}: "
                    f"{expected_bytes} != {resolved.stat().st_size}"
                ),
                repository_root,
            )

    expected_sha256 = receipt.get("sha256")
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        _issue(
            issues,
            "error",
            "invalid_member_manifest_sha256",
            manifest_path,
            f"{role} has no full lowercase SHA-256 receipt: {reference}",
            repository_root,
        )
    elif _sha256_of_file(resolved) != expected_sha256:
        _issue(
            issues,
            "error",
            "member_manifest_sha256_mismatch",
            manifest_path,
            f"{role} SHA-256 receipt does not match: {reference}",
            repository_root,
        )


def validate_member_manifest(
    manifest_path: Path,
    repository_root: Path = REPOSITORY_ROOT,
    *,
    payload: Mapping[str, Any] | None = None,
) -> tuple[CatalogIssue, ...]:
    """Validate dependency-complete member manifests and overlay receipts.

    Current Dr.Anmar packages use several deliberately distinct schemas.  The
    common integrity interface is a root ``members`` mapping with byte and
    SHA-256 receipts, so validation keys off that interface rather than one
    historical schema identifier.
    """

    repository_root = repository_root.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    issues: list[CatalogIssue] = []
    if manifest_path.name not in MEMBER_MANIFEST_NAMES:
        return ()
    if payload is None:
        try:
            candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            _issue(
                issues,
                "error",
                "invalid_member_manifest",
                manifest_path,
                str(error),
                repository_root,
            )
            return tuple(issues)
        payload = candidate if isinstance(candidate, Mapping) else None
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("members"), Mapping
    ):
        return ()

    manifest_root = manifest_path.parent.resolve()
    members = payload["members"]
    for reference, receipt in sorted(members.items(), key=lambda item: str(item[0])):
        if not isinstance(reference, str) or not isinstance(receipt, Mapping):
            _issue(
                issues,
                "error",
                "invalid_member_manifest_entry",
                manifest_path,
                "Manifest members must map normalized paths to receipt objects.",
                repository_root,
            )
            continue
        _validate_manifest_file_receipt(
            manifest_path=manifest_path,
            reference=reference,
            receipt=receipt,
            base=manifest_root,
            repository_root=repository_root,
            issues=issues,
            role="member",
        )

    local_entrypoints: list[str] = []
    primary_usd = payload.get("primary_usd")
    if isinstance(primary_usd, str) and primary_usd:
        local_entrypoints.append(primary_usd)
    entrypoints = payload.get("entrypoints")
    if isinstance(entrypoints, Mapping):
        local_entrypoints.extend(
            value
            for value in entrypoints.values()
            if isinstance(value, str) and value
        )
    for reference in sorted(set(local_entrypoints)):
        if reference not in members:
            _issue(
                issues,
                "error",
                "unreceipted_manifest_entrypoint",
                manifest_path,
                f"Entrypoint is not included in members: {reference}",
                repository_root,
            )

    base_physics = payload.get("base_physics_sha256")
    if isinstance(base_physics, Mapping):
        for reference, expected_sha256 in sorted(base_physics.items()):
            _validate_manifest_file_receipt(
                manifest_path=manifest_path,
                reference=str(reference),
                receipt={"sha256": expected_sha256},
                base=manifest_root,
                repository_root=repository_root,
                issues=issues,
                role="base physics dependency",
            )

    extension_root = _asset_extension_root(manifest_path)
    if extension_root is not None:
        generator = payload.get("generator")
        if isinstance(generator, Mapping):
            generator_path = generator.get("path") or generator.get(
                "repository_path"
            )
            if isinstance(generator_path, str) and generator_path:
                _validate_manifest_file_receipt(
                    manifest_path=manifest_path,
                    reference=generator_path,
                    receipt={"sha256": generator.get("sha256")},
                    base=extension_root,
                    repository_root=repository_root,
                    issues=issues,
                    role="generator",
                )

        preserved = payload.get("preserved_source_assets")
        if isinstance(preserved, Mapping):
            for reference, receipt in sorted(preserved.items()):
                if isinstance(receipt, Mapping):
                    _validate_manifest_file_receipt(
                        manifest_path=manifest_path,
                        reference=str(reference),
                        receipt=receipt,
                        base=extension_root,
                        repository_root=repository_root,
                        issues=issues,
                        role="preserved source dependency",
                    )

        base_assets = payload.get("base_assets")
        if isinstance(base_assets, Mapping):
            for receipt in base_assets.values():
                if not isinstance(receipt, Mapping):
                    continue
                reference = receipt.get("path")
                if isinstance(reference, str) and reference:
                    _validate_manifest_file_receipt(
                        manifest_path=manifest_path,
                        reference=reference,
                        receipt=receipt,
                        base=extension_root,
                        repository_root=repository_root,
                        issues=issues,
                        role="base asset dependency",
                    )

        base_dependency = payload.get("base_dependency")
        if isinstance(base_dependency, Mapping):
            reference = base_dependency.get("repository_path")
            if isinstance(reference, str) and reference:
                _validate_manifest_file_receipt(
                    manifest_path=manifest_path,
                    reference=reference,
                    receipt=base_dependency,
                    base=extension_root,
                    repository_root=repository_root,
                    issues=issues,
                    role="base dependency",
                )

        external_dependencies = payload.get("external_dependencies")
        if isinstance(external_dependencies, Mapping):
            for receipt in external_dependencies.values():
                if not isinstance(receipt, Mapping):
                    continue
                reference = receipt.get("repository_path")
                if isinstance(reference, str) and reference:
                    _validate_manifest_file_receipt(
                        manifest_path=manifest_path,
                        reference=reference,
                        receipt=receipt,
                        base=extension_root,
                        repository_root=repository_root,
                        issues=issues,
                        role="external overlay dependency",
                    )

    vendor_dependencies = payload.get("vendor_dependencies")
    if isinstance(vendor_dependencies, Mapping):
        for dependency in vendor_dependencies.values():
            if not isinstance(dependency, Mapping):
                continue
            destination = dependency.get("destination_root")
            dependency_members = dependency.get("members")
            if not isinstance(destination, str) or not isinstance(
                dependency_members, Mapping
            ):
                continue
            try:
                destination_root = (
                    manifest_root / _safe_relative_path(destination)
                ).resolve()
            except ValueError as error:
                _issue(
                    issues,
                    "error",
                    "unsafe_vendor_dependency_root",
                    manifest_path,
                    str(error),
                    repository_root,
                )
                continue
            for reference, receipt in sorted(dependency_members.items()):
                if isinstance(receipt, Mapping):
                    _validate_manifest_file_receipt(
                        manifest_path=manifest_path,
                        reference=str(reference),
                        receipt=receipt,
                        base=destination_root,
                        repository_root=repository_root,
                        issues=issues,
                        role="vendored dependency",
                    )

    return tuple(issues)


def _validate_policy(
    policy: Mapping[str, Any],
    repository_root: Path,
    issues: list[CatalogIssue],
) -> None:
    upstream = i4h_provider(policy)
    release = policy.get("release", {})
    if not isinstance(release, Mapping):
        _issue(
            issues,
            "error",
            "invalid_release_policy",
            POLICY_RELATIVE_PATH,
            "release must be a JSON object.",
            repository_root,
        )
        release = {}
    if release.get("clinical_validation") is not False:
        _issue(
            issues,
            "error",
            "unsafe_release_clinical_claim",
            POLICY_RELATIVE_PATH,
            "release.clinical_validation must be exactly false.",
            repository_root,
        )
    for requirement in ("lock_required", "generated_catalog_required"):
        if release.get(requirement) is not True:
            _issue(
                issues,
                "error",
                "missing_release_gate",
                POLICY_RELATIVE_PATH,
                f"release.{requirement} must be exactly true.",
                repository_root,
            )
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
    if upstream.get("license_review_required") is not True:
        _issue(
            issues,
            "error",
            "i4h_license_review",
            POLICY_RELATIVE_PATH,
            "The NVIDIA provider must retain license_review_required=true.",
            repository_root,
        )
    for provider_id, provider in policy.get("providers", {}).items():
        if not isinstance(provider, Mapping):
            continue
        if provider.get("clinical_validation") is not False:
            _issue(
                issues,
                "error",
                "unsafe_provider_clinical_claim",
                POLICY_RELATIVE_PATH,
                f"Provider {provider_id!r} must set clinical_validation=false.",
                repository_root,
            )
    for key in ("lock_path", "catalog_document_path"):
        try:
            policy_release_path(policy, key, repository_root)
        except ValueError as error:
            _issue(
                issues,
                "error",
                "unsafe_release_artifact",
                POLICY_RELATIVE_PATH,
                str(error),
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


def validate_portfolio(
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[CatalogIssue, ...]:
    """Validate every product-facing asset and its declared artifact closure."""

    repository_root = repository_root.expanduser().resolve()
    issues: list[CatalogIssue] = []
    try:
        portfolio = load_portfolio(repository_root)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        _issue(
            issues,
            "error",
            "invalid_asset_portfolio",
            PORTFOLIO_RELATIVE_PATH,
            str(error),
            repository_root,
        )
        return tuple(issues)

    seen_ids: set[str] = set()
    for index, entry in enumerate(portfolio["assets"]):
        location = f"{PORTFOLIO_RELATIVE_PATH.as_posix()}#/assets/{index}"
        if not isinstance(entry, Mapping):
            _issue(
                issues,
                "error",
                "invalid_portfolio_entry",
                location,
                "Portfolio entries must be JSON objects.",
                repository_root,
            )
            continue
        asset_id = str(entry.get("id", "")).strip()
        if not asset_id or asset_id in seen_ids:
            _issue(
                issues,
                "error",
                "duplicate_portfolio_id",
                location,
                f"Missing or duplicate portfolio ID: {asset_id!r}",
                repository_root,
            )
        seen_ids.add(asset_id)
        missing = sorted(PORTFOLIO_REQUIRED_FIELDS - entry.keys())
        if missing:
            _issue(
                issues,
                "error",
                "missing_portfolio_fields",
                location,
                f"{asset_id or index}: missing required fields {missing}",
                repository_root,
            )
        if entry.get("clinical_validation") is not False:
            _issue(
                issues,
                "error",
                "unsafe_clinical_validation_claim",
                location,
                f"{asset_id or index}: clinical_validation must be exactly false.",
                repository_root,
            )
        legacy_evidence_fields = sorted(
            {"native_gpu_qualification", "physical_qualification"} & entry.keys()
        )
        if legacy_evidence_fields:
            _issue(
                issues,
                "error",
                "legacy_qualification_language",
                location,
                f"{asset_id or index}: remove ambiguous fields {legacy_evidence_fields}.",
                repository_root,
            )
        for key in (
            "product_capability",
            "training_readiness",
            "software_evidence",
            "native_simulator_evidence",
            "real_world_evidence",
        ):
            value = entry.get(key)
            if not isinstance(value, str) or not value.strip():
                _issue(
                    issues,
                    "error",
                    "invalid_evidence_tier",
                    location,
                    f"{asset_id or index}/{key}: must be a non-empty string.",
                    repository_root,
                )
        if "available" not in str(entry.get("training_readiness", "")):
            _issue(
                issues,
                "error",
                "missing_training_availability",
                location,
                f"{asset_id or index}: training_readiness must state current availability.",
                repository_root,
            )
        if entry.get("task_contract") and entry.get("product_capability") != (
            "executable_training_workcell"
        ):
            _issue(
                issues,
                "error",
                "task_contract_without_training_workcell",
                location,
                f"{asset_id or index}: task-contract assets must be executable training workcells.",
                repository_root,
            )
        declared_artifacts = 0
        for key in sorted(PORTFOLIO_PATH_FIELDS):
            value = entry.get(key)
            if value is None:
                continue
            if not isinstance(value, str) or not value:
                _issue(
                    issues,
                    "error",
                    "invalid_portfolio_artifact",
                    location,
                    f"{asset_id or index}/{key}: artifact path must be a non-empty string.",
                    repository_root,
                )
                continue
            declared_artifacts += 1
            try:
                resolved = resolve_repository_artifact(value, repository_root)
            except ValueError as error:
                _issue(
                    issues,
                    "error",
                    "unsafe_portfolio_artifact",
                    location,
                    f"{asset_id or index}/{key}: {error}",
                    repository_root,
                )
                continue
            if not resolved.is_file():
                _issue(
                    issues,
                    "error",
                    "missing_portfolio_artifact",
                    location,
                    f"{asset_id or index}/{key}: {value} is missing.",
                    repository_root,
                )
        if declared_artifacts == 0:
            _issue(
                issues,
                "error",
                "empty_portfolio_artifact_closure",
                location,
                f"{asset_id or index}: no catalog artifacts are declared.",
                repository_root,
            )
    return tuple(issues)


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
                            "native_simulator_evidence",
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
                    if isinstance(payload, Mapping):
                        issues.extend(
                            validate_member_manifest(
                                path,
                                repository_root,
                                payload=payload,
                            )
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
                "Asset unit has no direct USD or registered data entrypoint.",
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
    require_release_artifacts: bool = False,
    lock_path: Path | None = None,
) -> dict[str, Any]:
    """Run the simulator-independent catalog gate and return a JSON-ready report."""

    repository_root = repository_root.expanduser().resolve()
    policy = load_policy(repository_root)
    units = discover_asset_units(repository_root, policy=policy)
    issues: list[CatalogIssue] = []
    _validate_policy(policy, repository_root, issues)
    portfolio_issues = validate_portfolio(repository_root)
    issues.extend(portfolio_issues)
    try:
        portfolio_asset_count = len(load_portfolio(repository_root)["assets"])
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        portfolio_asset_count = 0
    _validate_json_and_usd(repository_root, policy, issues)
    _validate_unit_metadata(units, policy, repository_root, issues)
    growth_budgets = policy.get("quality", {}).get("growth_budgets", {})
    maximum_unit_bytes = int(
        growth_budgets.get("maximum_asset_unit_bytes", 0) or 0
    )
    maximum_catalog_bytes = int(
        growth_budgets.get("maximum_catalog_bytes", 0) or 0
    )
    for unit in units:
        if maximum_unit_bytes and unit.bytes > maximum_unit_bytes:
            _issue(
                issues,
                "error",
                "asset_unit_growth_budget_exceeded",
                unit.relative_path,
                f"{unit.bytes} bytes exceeds the {maximum_unit_bytes}-byte asset-unit budget.",
                repository_root,
            )
    catalog_bytes = sum(unit.bytes for unit in units)
    if maximum_catalog_bytes and catalog_bytes > maximum_catalog_bytes:
        _issue(
            issues,
            "error",
            "catalog_growth_budget_exceeded",
            "config/dranmar_asset_catalog.json",
            f"{catalog_bytes} bytes exceeds the {maximum_catalog_bytes}-byte catalog budget.",
            repository_root,
        )
    from dr_anmar_multimodal_assets import validate_all_multimodal_bundles

    for multimodal_issue in validate_all_multimodal_bundles(repository_root):
        _issue(
            issues,
            "error",
            multimodal_issue.code,
            multimodal_issue.path,
            multimodal_issue.message,
            repository_root,
        )
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
    if require_release_artifacts:
        issues.extend(
            validate_release_artifacts(
                repository_root,
                lock_path=lock_path,
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
        "portfolio_assets": portfolio_asset_count,
        "entrypoints": sum(len(unit.entrypoints) for unit in units),
        "files": sum(unit.file_count for unit in units),
        "bytes": catalog_bytes,
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
    portfolio = load_portfolio(repository_root)
    lock: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "catalog_version": policy["catalog_version"],
        "clinical_validation": False,
        "i4h": {
            "release": upstream["release"],
            "catalog_commit": upstream["catalog_commit"],
            "asset_version": upstream["asset_version"],
            "content_hash": upstream["content_hash"],
        },
        "assets": [
            {
                "id": unit.asset_id,
                "provider": unit.provider,
                "relative_path": unit.relative_path,
                "entrypoints": list(unit.entrypoints),
                "metadata": list(unit.metadata),
                "license_path": unit.license_path,
                "file_count": unit.file_count,
                "bytes": unit.bytes,
                "sha256": unit.sha256,
                "clinical_validation": False,
            }
            for unit in units
        ],
        "portfolio": {
            "schema": portfolio["schema"],
            "asset_count": len(portfolio["assets"]),
            "assets": [
                {
                    "id": str(entry["id"]),
                    "asset": str(entry["asset"]),
                    "artifacts": {
                        key: str(entry[key])
                        for key in sorted(PORTFOLIO_PATH_FIELDS)
                        if isinstance(entry.get(key), str) and entry[key]
                    },
                    "product_capability": entry["product_capability"],
                    "training_readiness": entry["training_readiness"],
                    "software_evidence": entry["software_evidence"],
                    "native_simulator_evidence": entry["native_simulator_evidence"],
                    "real_world_evidence": entry["real_world_evidence"],
                    "clinical_validation": entry["clinical_validation"],
                }
                for entry in sorted(
                    portfolio["assets"],
                    key=lambda item: str(item["id"]),
                )
            ],
        },
    }
    lock["catalog_sha256"] = catalog_lock_digest(lock)
    return lock


def catalog_lock_digest(lock: Mapping[str, Any]) -> str:
    """Hash the canonical lock payload, excluding its self-digest field."""

    canonical = dict(lock)
    canonical.pop("catalog_sha256", None)
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _display_status(value: Any, *, maximum: int = 72) -> str:
    if isinstance(value, Mapping):
        value = value.get("status", "structured qualification")
    rendered = str(value).replace("|", "\\|").replace("_", " ")
    return rendered if len(rendered) <= maximum else f"{rendered[: maximum - 1]}…"


def render_catalog_document(lock: Mapping[str, Any]) -> str:
    """Render the canonical human-readable catalog from one release lock."""

    upstream = lock["i4h"]
    assets = lock["assets"]
    portfolio = lock["portfolio"]["assets"]
    lines = [
        "<!-- Generated by scripts/dr_anmar_asset_registry.py; do not edit by hand. -->",
        "",
        "# Dr.Anmar simulation asset catalog",
        "",
        "This catalog is a human-readable view of the deterministic release lock.",
        "It records software and asset provenance; it is not clinical-validation evidence.",
        "",
        "## Release identity",
        "",
        f"- Catalog schema: `{lock['schema']}`",
        f"- Catalog version: `{lock['catalog_version']}`",
        f"- Catalog SHA-256: `{lock['catalog_sha256']}`",
        f"- Local asset units: `{len(assets)}`",
        f"- Product portfolio assets: `{len(portfolio)}`",
        "- Clinical validation: `false`",
        "",
        "## NVIDIA Isaac for Healthcare provider",
        "",
        f"- Release: `{upstream['release']}`",
        f"- Source commit: `{upstream['catalog_commit']}`",
        f"- Asset version: `{upstream['asset_version']}`",
        f"- Content address: `{upstream['content_hash']}`",
        "",
        "Downloaded NVIDIA assets retain their provider-specific license terms.",
        "",
        "## Local dependency-complete asset units",
        "",
        "| Asset unit | Entrypoints | Files | Bytes | SHA-256 | License evidence |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for asset in assets:
        entrypoints = len(asset["entrypoints"])
        license_path = str(asset.get("license_path") or "missing").replace("|", "\\|")
        lines.append(
            f"| `{asset['id']}` | {entrypoints} | {asset['file_count']} | "
            f"{asset['bytes']} | `{asset['sha256']}` | `{license_path}` |"
        )
    lines.extend(
        [
            "",
            "## Product-facing portfolio",
            "",
            "These assets are available Dr.Anmar simulation-training capabilities.",
            "Repository verification, native-simulator evidence, real-world evidence,",
            "and clinical evidence are deliberately separate claims.",
            "",
            "| Asset | Product capability | Training readiness | Software evidence |",
            "| --- | --- | --- | --- |",
        ]
    )
    for asset in portfolio:
        lines.append(
            f"| `{asset['id']}` | {_display_status(asset['product_capability'])} | "
            f"{_display_status(asset['training_readiness'])} | "
            f"{_display_status(asset['software_evidence'])} |"
        )
    lines.extend(
        [
            "",
            "## Evidence boundaries",
            "",
            "| Asset | Native simulator evidence | Real-world evidence | Clinical evidence |",
            "| --- | --- | --- | --- |",
        ]
    )
    for asset in portfolio:
        lines.append(
            f"| `{asset['id']}` | "
            f"{_display_status(asset['native_simulator_evidence'])} | "
            f"{_display_status(asset['real_world_evidence'])} | "
            f"`{'not established' if asset['clinical_validation'] is False else 'established'}` |"
        )
    lines.extend(
        [
            "",
            "Regenerate this file and its lock only after asset-specific structural,",
            "native-simulator, and applicable real-world evidence has been reviewed.",
            "",
        ]
    )
    return "\n".join(lines)


def verify_lock(
    lock: Mapping[str, Any],
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[str, ...]:
    """Compare a previously generated release lock with the current tree."""

    failures: list[str] = []
    if lock.get("schema") != LOCK_SCHEMA:
        return ("Unsupported or missing asset-catalog lock schema.",)
    assets = lock.get("assets")
    if not isinstance(assets, list):
        return ("Asset-catalog lock assets must be a list.",)
    asset_ids: list[str] = []
    for asset in assets:
        if not isinstance(asset, Mapping) or not isinstance(asset.get("id"), str):
            failures.append("Asset-catalog lock contains an invalid asset entry.")
            continue
        asset_ids.append(str(asset["id"]))
        if not re.fullmatch(r"[0-9a-f]{64}", str(asset.get("sha256", ""))):
            failures.append(f"Asset unit has an invalid SHA-256: {asset['id']}")
    if len(asset_ids) != len(set(asset_ids)):
        failures.append("Asset-catalog lock contains duplicate asset IDs.")
    if lock.get("catalog_sha256") != catalog_lock_digest(lock):
        failures.append("Asset-catalog lock self-digest does not match.")
    if failures:
        return tuple(failures)

    current = build_lock(repository_root)
    if lock.get("catalog_version") != current["catalog_version"]:
        failures.append("Catalog version does not match the lock.")
    if lock.get("i4h") != current["i4h"]:
        failures.append("Pinned i4h provider does not match the lock.")
    if lock.get("portfolio") != current["portfolio"]:
        failures.append("Product asset portfolio changed.")
    expected_assets = {str(asset["id"]): asset for asset in assets}
    current_assets = {str(asset["id"]): asset for asset in current.get("assets", ())}
    for asset_id in sorted(expected_assets.keys() | current_assets.keys()):
        if expected_assets.get(asset_id) != current_assets.get(asset_id):
            failures.append(f"Asset unit changed: {asset_id}")
    return tuple(failures)


def validate_release_artifacts(
    repository_root: Path = REPOSITORY_ROOT,
    *,
    lock_path: Path | None = None,
) -> tuple[CatalogIssue, ...]:
    """Verify the checked-in lock and generated catalog against the asset tree."""

    repository_root = repository_root.expanduser().resolve()
    policy = load_policy(repository_root)
    issues: list[CatalogIssue] = []
    canonical_lock_path = policy_release_path(policy, "lock_path", repository_root)
    selected_lock_path = lock_path.expanduser().resolve() if lock_path is not None else canonical_lock_path
    try:
        lock = json.loads(selected_lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _issue(
            issues,
            "error",
            "missing_or_invalid_catalog_lock",
            selected_lock_path,
            str(error),
            repository_root,
        )
        return tuple(issues)
    for failure in verify_lock(lock, repository_root):
        _issue(
            issues,
            "error",
            "catalog_lock_mismatch",
            selected_lock_path,
            failure,
            repository_root,
        )

    catalog_path = policy_release_path(
        policy,
        "catalog_document_path",
        repository_root,
    )
    try:
        expected_document = render_catalog_document(lock)
    except (KeyError, TypeError, ValueError) as error:
        _issue(
            issues,
            "error",
            "invalid_catalog_document_source",
            selected_lock_path,
            str(error),
            repository_root,
        )
        return tuple(issues)
    try:
        actual_document = catalog_path.read_text(encoding="utf-8")
    except OSError as error:
        _issue(
            issues,
            "error",
            "missing_generated_catalog",
            catalog_path,
            str(error),
            repository_root,
        )
    else:
        if actual_document != expected_document:
            _issue(
                issues,
                "error",
                "stale_generated_catalog",
                catalog_path,
                "Generated catalog.md does not match the canonical release lock.",
                repository_root,
            )
    return tuple(issues)


def _procedure_rooms() -> Sequence[Mapping[str, Any]]:
    from dr_anmar_procedures import PROCEDURE_ROOMS

    return PROCEDURE_ROOMS


def _json_dump(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _atomic_write_text(path: Path, content: str) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


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
        help="Verify this lock instead of the policy-defined canonical lock",
    )
    verify.add_argument(
        "--skip-release-artifacts",
        action="store_true",
        help="Run structural checks without the canonical lock/catalog gate",
    )

    resolve = subparsers.add_parser("resolve", help="Resolve one provider-relative path")
    resolve.add_argument("provider")
    resolve.add_argument("path")
    resolve.add_argument("--i4h-root", type=Path)
    resolve.add_argument("--require", action="store_true")

    lock = subparsers.add_parser("lock", help="Generate a deterministic release lock")
    lock.add_argument("--output", type=Path)

    catalog = subparsers.add_parser(
        "catalog",
        help="Render the human-readable catalog from a release lock",
    )
    catalog.add_argument("--lock", type=Path)
    catalog.add_argument("--output", type=Path)
    catalog.add_argument(
        "--check",
        action="store_true",
        help="Fail if the output differs instead of writing it",
    )
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
            require_release_artifacts=not args.skip_release_artifacts,
            lock_path=args.lock,
        )
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
        policy = load_policy(repository_root)
        output = (
            args.output.expanduser()
            if args.output is not None
            else policy_release_path(policy, "lock_path", repository_root)
        )
        _atomic_write_text(
            output,
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )
        print(output)
        return 0
    if args.command == "catalog":
        policy = load_policy(repository_root)
        lock_path = (
            args.lock.expanduser()
            if args.lock is not None
            else policy_release_path(policy, "lock_path", repository_root)
        )
        output = (
            args.output.expanduser()
            if args.output is not None
            else policy_release_path(
                policy,
                "catalog_document_path",
                repository_root,
            )
        )
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        rendered = render_catalog_document(lock)
        if args.check:
            try:
                existing = output.read_text(encoding="utf-8")
            except OSError:
                return 1
            return 0 if existing == rendered else 1
        _atomic_write_text(output, rendered)
        print(output)
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
