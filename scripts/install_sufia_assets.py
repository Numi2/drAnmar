# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Resumable installer for the official SuFIA-BC OpenUSD organ scenes."""

from __future__ import annotations

import json
import os
import shutil
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ASSETS = (
    ("OR_scene_CTLiver-Prostate-Bladder.zip", 1232706747),
    ("OR_scene_MAISI_imagesTr_liver_27_relabel_resample1_syn_seed6_postprocess.zip", 943105783),
    ("OR_scene_MAISI_s0253_ct_relabel_resample1_syn_seed6_postprocess.zip", 776424651),
    ("OR_scene_MAISI_s0702_ct_relabel_resample2_syn_seed6_postprocess.zip", 782768899),
    ("OR_scene_MAISI_s0994_ct_relabel_resample2_syn_seed6_postprocess.zip", 792621339),
    ("OR_scene_MAISI_s1269_ct_relabel_resample1_syn_seed6_postprocess.zip", 876671061),
    ("OR_scene_s1371.zip", 861846489),
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


def download(name: str, expected: int, index: int) -> Path:
    final = ARCHIVES / name
    partial = final.with_suffix(final.suffix + ".part")
    if final.exists() and final.stat().st_size == expected:
        return final
    existing = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(f"{BASE_URL}/{name}")
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    mode = "ab" if existing else "wb"
    with urllib.request.urlopen(request, timeout=60) as response, partial.open(mode) as output:
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
                    min((ARCHIVES / item).stat().st_size, size) if (ARCHIVES / item).exists() else 0
                    for item, size in ASSETS
                )
                + min(existing, expected),
                total_bytes=sum(size for _, size in ASSETS),
            )
    if partial.stat().st_size != expected:
        raise IOError(f"Size mismatch for {name}: {partial.stat().st_size} != {expected}")
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


def main() -> None:
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    INSTALL.mkdir(parents=True, exist_ok=True)
    update(phase="starting", asset_count=len(ASSETS), total_bytes=sum(size for _, size in ASSETS), error=None)
    try:
        for index, (name, expected) in enumerate(ASSETS, 1):
            archive = download(name, expected, index)
            scene_dir = INSTALL / name.removesuffix(".zip")
            marker = scene_dir / ".installed"
            if not marker.exists():
                update(phase="extracting", current=name, asset_index=index, asset_count=len(ASSETS))
                if scene_dir.exists():
                    shutil.rmtree(scene_dir)
                safe_extract(archive, scene_dir)
                marker.touch()
        usd_files = sorted(str(path.relative_to(INSTALL)) for path in INSTALL.rglob("*.usd"))
        update(phase="ready", installed_scenes=len(ASSETS), usd_files=usd_files, completed_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        update(phase="error", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
