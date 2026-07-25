# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed catalog access for DrAnmar simulation-ready assets.

The interface follows the compatible parts of NVIDIA's i4h asset-catalog
v0.7.0 contract:

* public catalog entries are relative to the catalog data root;
* a USD lookup resolves the complete containing directory, not one detached
  layer;
* deterministic directory identity hashes both POSIX relative paths and file
  bytes in sorted order;
* local roots are explicit and environment-overridable.

DrAnmar assets remain repository-local until they are published by a catalog
provider.  This module never silently redirects missing DrAnmar content to the
NVIDIA production bucket.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
from typing import Final, Iterable


CATALOG_SCHEMA: Final = "dr.anmar.sim-ready-asset-catalog.v1"
CATALOG_VERSION: Final = "0.1.0"
CATALOG_ROOT_ENV: Final = "DRANMAR_ASSET_DATA_ROOT"
CATALOG_CACHE_ENV: Final = "DRANMAR_ASSET_CACHE_DIR"
I4H_REFERENCE_RELEASE: Final = "v0.7.0"
I4H_REFERENCE_COMMIT: Final = "b0b7ad39f26490d58d12407cfa74b3c9ad861769"

_DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
_HASH_SKIP_NAMES = frozenset({".gitattributes", ".gitignore", ".DS_Store"})
_HASH_SKIP_DIRS = frozenset({".git", "__pycache__"})
_USD_REFERENCE_RE = re.compile(r"@([^@]+)@")


@dataclass(frozen=True)
class DrAnmarAssetDescriptor:
    """One stable, category-relative DrAnmar asset entry."""

    asset_id: str
    catalog_subpath: str
    primary_usd: str
    rigid_proxy_usd: str | None
    payload_usd: str | None
    interaction_frames: str = "interaction_frames.json"
    physics_profile: str = "physics_profile.json"
    asset_manifest: str = "asset_manifest.json"
    allow_catalog_dependencies: bool = False

    def members(self) -> tuple[str, ...]:
        return tuple(
            member
            for member in (
                self.primary_usd,
                self.rigid_proxy_usd,
                self.payload_usd,
                self.interaction_frames,
                self.physics_profile,
                self.asset_manifest,
            )
            if member
        )


DRANMAR_SIM_READY_ASSETS: Final[dict[str, DrAnmarAssetDescriptor]] = {
    "wound_preparation": DrAnmarAssetDescriptor(
        "dranmar-wound-preparation-robot-v1",
        "Props/SurgicalPreparation/WoundPreparationRobot",
        "dranmar_wound_preparation_tool_standalone.usda",
        "dranmar_wound_preparation_tool_rigid_proxy.usda",
        "dranmar_wound_preparation_tool_payload.usda",
    ),
    "atraumatic_exposure": DrAnmarAssetDescriptor(
        "dranmar-atraumatic-exposure-robot-v1",
        "Props/SurgicalExposure/AtraumaticExposureRobot",
        "dranmar_atraumatic_exposure_tool_standalone.usda",
        "dranmar_atraumatic_exposure_tool_rigid_proxy.usda",
        "dranmar_atraumatic_exposure_tool_payload.usda",
    ),
    "adaptive_hemostasis": DrAnmarAssetDescriptor(
        "dranmar-adaptive-hemostasis-robot-v1",
        "Props/SurgicalHemostasis/AdaptiveHemostasisRobot",
        "dranmar_adaptive_hemostasis_tool_standalone.usda",
        "dranmar_adaptive_hemostasis_tool_rigid_proxy.usda",
        "dranmar_adaptive_hemostasis_tool_payload.usda",
    ),
    "adaptive_anastomosis": DrAnmarAssetDescriptor(
        "dranmar-adaptive-anastomosis-robot-v1",
        "Props/SurgicalReconstruction/AdaptiveAnastomosisRobot",
        "dranmar_adaptive_anastomosis_tool_standalone.usda",
        "dranmar_adaptive_anastomosis_tool_rigid_proxy.usda",
        "dranmar_adaptive_anastomosis_tool_payload.usda",
    ),
    "adaptive_seal_divide": DrAnmarAssetDescriptor(
        "dranmar-adaptive-seal-divide-robot-v1",
        "Props/SurgicalDivision/AdaptiveSealDivideRobot",
        "dranmar_adaptive_seal_divide_tool_standalone.usda",
        "dranmar_adaptive_seal_divide_tool_rigid_proxy.usda",
        "dranmar_adaptive_seal_divide_tool_payload.usda",
    ),
    "safeplane_dissection": DrAnmarAssetDescriptor(
        "dranmar-safeplane-dissection-robot-v1",
        "Props/SurgicalDissection/SafePlaneDissectionRobot",
        "dranmar_safeplane_dissection_tool_standalone.usda",
        "dranmar_safeplane_dissection_tool_rigid_proxy.usda",
        "dranmar_safeplane_dissection_tool_payload.usda",
    ),
    "perfusion_viability": DrAnmarAssetDescriptor(
        "dranmar-perfusion-viability-robot-v1",
        "Props/SurgicalAssessment/PerfusionViabilityRobot",
        "dranmar_perfusion_viability_tool_standalone.usda",
        "dranmar_perfusion_viability_tool_rigid_proxy.usda",
        "dranmar_perfusion_viability_tool_payload.usda",
    ),
    "oncologic_resection": DrAnmarAssetDescriptor(
        "dranmar-oncosurgery-cell-v1",
        "Props/SurgicalOncology/OncoSurgeryCell",
        "dranmar_tumor_resection_tool_standalone.usda",
        "dranmar_tumor_resection_tool_rigid_proxy.usda",
        "dranmar_tumor_resection_tool_payload.usda",
    ),
    "autonomous_rescue_or": DrAnmarAssetDescriptor(
        "dranmar-autonomous-rescue-or-v0.3.0",
        "Environments/SurgicalAutonomy/AutonomousRescueOR",
        "dranmar_autonomous_rescue_or.usda",
        None,
        "dranmar_universal_tool_changer_payload.usda",
        physics_profile="physics_profile.json",
        allow_catalog_dependencies=True,
    ),
}


def asset_data_root(*, require: bool = True) -> Path:
    """Resolve the explicit catalog data root without remote fall-through."""

    override = os.environ.get(CATALOG_ROOT_ENV)
    candidate = (
        Path(override).expanduser() if override else _DEFAULT_DATA_ROOT
    )
    if candidate.is_dir():
        return candidate.resolve()
    if require:
        raise FileNotFoundError(
            f"DrAnmar asset data root does not exist: {candidate}. "
            f"Set {CATALOG_ROOT_ENV} to a catalog extraction root."
        )
    return candidate


def asset_directory(asset_name: str, *, require: bool = True) -> Path:
    """Return the dependency-complete directory for one catalog entry."""

    try:
        descriptor = DRANMAR_SIM_READY_ASSETS[asset_name]
    except KeyError as error:
        raise KeyError(
            f"Unknown DrAnmar asset {asset_name!r}; expected one of "
            f"{sorted(DRANMAR_SIM_READY_ASSETS)}"
        ) from error
    directory = asset_data_root(require=require) / descriptor.catalog_subpath
    if require and not directory.is_dir():
        raise FileNotFoundError(
            f"DrAnmar asset directory does not exist: {directory}"
        )
    return directory


def asset_path(
    asset_name: str,
    member: str = "primary_usd",
    *,
    require: bool = True,
) -> Path:
    """Resolve one declared member while retaining its dependency directory."""

    descriptor = DRANMAR_SIM_READY_ASSETS[asset_name]
    try:
        relative_name = getattr(descriptor, member)
    except AttributeError as error:
        raise KeyError(
            f"Asset member {member!r} is not declared by {asset_name!r}"
        ) from error
    if not relative_name:
        raise KeyError(
            f"Asset member {member!r} is not available for {asset_name!r}"
        )
    path = asset_directory(asset_name, require=require) / relative_name
    if require and not path.is_file():
        raise FileNotFoundError(f"DrAnmar catalog member is missing: {path}")
    return path


def iter_hashed_files(folder: Path) -> Iterable[tuple[str, Path]]:
    """Yield the same stable path ordering used for catalog identity."""

    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Asset folder does not exist: {folder}")
    entries: list[tuple[str, Path]] = []
    for root, directories, filenames in os.walk(folder):
        directories[:] = sorted(
            name for name in directories if name not in _HASH_SKIP_DIRS
        )
        for filename in sorted(filenames):
            if filename in _HASH_SKIP_NAMES:
                continue
            file_path = Path(root) / filename
            entries.append((file_path.relative_to(folder).as_posix(), file_path))
    yield from sorted(entries)


def sha256_of_folder(folder: Path) -> str:
    """Hash sorted POSIX relative paths plus bytes, as i4h v0.7.0 does."""

    digest = hashlib.sha256()
    for relative_path, file_path in iter_hashed_files(Path(folder)):
        digest.update(relative_path.encode("utf-8"))
        with file_path.open("rb") as stream:
            while chunk := stream.read(8192):
                digest.update(chunk)
    return digest.hexdigest()


def content_addressed_cache_path(
    asset_name: str,
    *,
    version: str = CATALOG_VERSION,
    digest: str | None = None,
) -> Path:
    """Return the non-mutating cache destination for a catalog snapshot."""

    if digest is None:
        digest = sha256_of_folder(asset_directory(asset_name))
    cache_root = Path(
        os.environ.get(
            CATALOG_CACHE_ENV,
            Path.home() / ".cache" / "dranmar-assets",
        )
    ).expanduser()
    return cache_root / version / digest


def validate_usd_dependency_closure(asset_name: str) -> dict[str, object]:
    """Fail closed on missing or escaping USDA layer/asset references."""

    directory = asset_directory(asset_name)
    missing: list[str] = []
    escaping: list[str] = []
    references_checked = 0
    external_catalog_references_checked = 0
    descriptor = DRANMAR_SIM_READY_ASSETS[asset_name]
    data_root = asset_data_root().resolve()
    for _, source in iter_hashed_files(directory):
        if source.suffix.lower() not in {".usd", ".usda"}:
            continue
        try:
            text = source.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Binary .usd layers require OpenUSD resolution at native
            # qualification time; they are still included in the folder hash.
            continue
        for reference in _USD_REFERENCE_RE.findall(text):
            if "://" in reference:
                continue
            references_checked += 1
            resolved = (source.parent / reference).resolve()
            try:
                resolved.relative_to(directory.resolve())
            except ValueError:
                if descriptor.allow_catalog_dependencies:
                    try:
                        resolved.relative_to(data_root)
                    except ValueError:
                        escaping.append(f"{source.name}: {reference}")
                        continue
                    external_catalog_references_checked += 1
                else:
                    escaping.append(f"{source.name}: {reference}")
                    continue
            if not resolved.exists():
                missing.append(f"{source.name}: {reference}")
    if missing or escaping:
        raise RuntimeError(
            f"USD dependency closure failed for {asset_name}: "
            f"missing={missing}, escaping={escaping}"
        )
    return {
        "asset_name": asset_name,
        "directory": str(directory),
        "references_checked": references_checked,
        "external_catalog_references_checked": (
            external_catalog_references_checked
        ),
        "missing": [],
        "escaping": [],
    }


try:
    from i4h_asset_helper import BaseI4HAssets as _BaseI4HAssets
except (ImportError, ModuleNotFoundError):
    _BaseI4HAssets = object


class DrAnmarSurgicalRobotAssets(_BaseI4HAssets):
    """I4H-compatible relative names for published DrAnmar catalog bundles."""

    WOUND_PREPARATION = (
        "Props/SurgicalPreparation/WoundPreparationRobot/"
        "dranmar_wound_preparation_tool_standalone.usda"
    )
    ATRAUMATIC_EXPOSURE = (
        "Props/SurgicalExposure/AtraumaticExposureRobot/"
        "dranmar_atraumatic_exposure_tool_standalone.usda"
    )
    ADAPTIVE_HEMOSTASIS = (
        "Props/SurgicalHemostasis/AdaptiveHemostasisRobot/"
        "dranmar_adaptive_hemostasis_tool_standalone.usda"
    )
    ADAPTIVE_ANASTOMOSIS = (
        "Props/SurgicalReconstruction/AdaptiveAnastomosisRobot/"
        "dranmar_adaptive_anastomosis_tool_standalone.usda"
    )
    ADAPTIVE_SEAL_DIVIDE = (
        "Props/SurgicalDivision/AdaptiveSealDivideRobot/"
        "dranmar_adaptive_seal_divide_tool_standalone.usda"
    )
    SAFEPLANE_DISSECTION = (
        "Props/SurgicalDissection/SafePlaneDissectionRobot/"
        "dranmar_safeplane_dissection_tool_standalone.usda"
    )
    PERFUSION_VIABILITY = (
        "Props/SurgicalAssessment/PerfusionViabilityRobot/"
        "dranmar_perfusion_viability_tool_standalone.usda"
    )
    ONCOLOGIC_RESECTION = (
        "Props/SurgicalOncology/OncoSurgeryCell/"
        "dranmar_tumor_resection_tool_standalone.usda"
    )
    AUTONOMOUS_RESCUE_OR = (
        "Environments/SurgicalAutonomy/AutonomousRescueOR/"
        "dranmar_autonomous_rescue_or.usda"
    )
