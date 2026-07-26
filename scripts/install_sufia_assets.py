# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Resumable installer for the official SuFIA-BC OpenUSD organ scenes."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ASSETS = (
    ("OR_scene_CTLiver-Prostate-Bladder.zip", 1232706747, "fd1062cae1c54bca896871a0d63c0d001527b4c7c59b99de14a8a6e397f36d3a"),
    ("OR_scene_MAISI_imagesTr_liver_27_relabel_resample1_syn_seed6_postprocess.zip", 943105783, "d0e39b2e90f32d283e10ec267626767a879403aba9f26e3cef4be3c95a02790a"),
    ("OR_scene_MAISI_s0253_ct_relabel_resample1_syn_seed6_postprocess.zip", 776424651, "57507972b7387fef078c82add69d053f03bedc841c543cd4e8cc8006f23746c4"),
    ("OR_scene_MAISI_s0702_ct_relabel_resample2_syn_seed6_postprocess.zip", 782768899, "5bcf7c80c4b0ab69142b3c9c2b83fafa35907089c48a16e3cfd8b562e284301a"),
    ("OR_scene_MAISI_s0994_ct_relabel_resample2_syn_seed6_postprocess.zip", 792621339, "6ea011f7cccbf4ecc8725cba753c4721b6db142e242be8910e5f35afc943ea27"),
    ("OR_scene_MAISI_s1269_ct_relabel_resample1_syn_seed6_postprocess.zip", 876671061, "f7432beae6d9246d7851e87b91005f6dc1f6a55114a7b90d5a0f85a1eb59b7af"),
    ("OR_scene_s1371.zip", 861846489, "287595605efd7ec431c315f75067b6dc479be63bb7e68c72021c6188bca34f59"),
)
BASE_URL = "https://github.com/orbit-surgical/orbit-surgical/releases/download/v0.1.0"
ROOT = Path(os.environ.get("DR_ANMAR_ROOT", Path.home() / ".local/share/dr-anmar")).expanduser()
ARCHIVES = ROOT / "downloads/sufia_bc"
INSTALL = ROOT / "assets/sufia_bc"
STATUS = ROOT / "run/sufia_assets_status.json"


def update(**values) -> None:
    current = {}
    if STATUS.exists():
        try:
            current = json.loads(STATUS.read_text())
        except (OSError, ValueError):
            pass
    current.update(values, updated_at=datetime.now(timezone.utc).isoformat())
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(current, indent=2) + "\n")
    temporary.replace(STATUS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_is_verified(path: Path, expected: int, expected_sha256: str) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == expected
        and sha256_file(path) == expected_sha256
    )


def quarantine(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    quarantined = path.with_name(f"{path.name}.invalid-{stamp}")
    path.replace(quarantined)
    return quarantined


def download(name: str, expected: int, expected_sha256: str, index: int) -> Path:
    final = ARCHIVES / name
    partial = final.with_suffix(final.suffix + ".part")
    if archive_is_verified(final, expected, expected_sha256):
        return final
    if final.exists():
        quarantine(final)
    existing = partial.stat().st_size if partial.exists() else 0
    if existing > expected:
        quarantine(partial)
        existing = 0
    request = urllib.request.Request(f"{BASE_URL}/{name}")
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    with urllib.request.urlopen(request, timeout=60) as response:
        range_honored = response.status == 206
        if existing and not range_honored:
            existing = 0
        mode = "ab" if existing else "wb"
        output = partial.open(mode)
        try:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                existing += len(chunk)
                update(
                    phase="downloading",
                    current=name,
                    asset_index=index,
                    asset_count=len(ASSETS),
                    current_bytes=existing,
                    current_total=expected,
                    downloaded_bytes=sum(
                        min((ARCHIVES / item).stat().st_size, size)
                        if (ARCHIVES / item).exists()
                        else 0
                        for item, size, _sha256 in ASSETS
                    )
                    + min(existing, expected),
                    total_bytes=sum(size for _, size, _sha256 in ASSETS),
                )
        finally:
            output.close()
    if partial.stat().st_size != expected:
        raise IOError(f"Size mismatch for {name}: {partial.stat().st_size} != {expected}")
    actual_sha256 = sha256_file(partial)
    if actual_sha256 != expected_sha256:
        quarantined = quarantine(partial)
        raise IOError(
            f"SHA-256 mismatch for {name}: {actual_sha256} != {expected_sha256}; "
            f"quarantined at {quarantined}"
        )
    partial.replace(final)
    return final


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        source.extractall(destination)


def install_archive(archive: Path, scene_dir: Path, expected_sha256: str) -> None:
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{scene_dir.name}.extract-", dir=scene_dir.parent)
    )
    try:
        safe_extract(archive, temporary)
        extracted_files = [path for path in temporary.rglob("*") if path.is_file()]
        receipt = {
            "schema": "dr.anmar.sufia-asset-installation.v1",
            "archive": archive.name,
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": expected_sha256,
            "extracted_file_count": len(extracted_files),
            "extracted_bytes": sum(path.stat().st_size for path in extracted_files),
            "installed_at": datetime.now(timezone.utc).isoformat(),
            "clinical_validation": False,
        }
        (temporary / ".installed.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if scene_dir.exists():
            shutil.rmtree(scene_dir)
        temporary.replace(scene_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    INSTALL.mkdir(parents=True, exist_ok=True)
    update(
        phase="starting",
        asset_count=len(ASSETS),
        total_bytes=sum(size for _, size, _sha256 in ASSETS),
        error=None,
    )
    try:
        for index, (name, expected, expected_sha256) in enumerate(ASSETS, 1):
            archive = download(name, expected, expected_sha256, index)
            scene_dir = INSTALL / name.removesuffix(".zip")
            marker = scene_dir / ".installed.json"
            installed = {}
            if marker.is_file():
                try:
                    installed = json.loads(marker.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    installed = {}
            if (
                installed.get("archive_sha256") != expected_sha256
                or installed.get("archive_bytes") != expected
            ):
                update(phase="extracting", current=name, asset_index=index, asset_count=len(ASSETS))
                install_archive(archive, scene_dir, expected_sha256)
        usd_files = sorted(str(path.relative_to(INSTALL)) for path in INSTALL.rglob("*.usd"))
        update(phase="ready", installed_scenes=len(ASSETS), usd_files=usd_files, completed_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        update(phase="error", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
