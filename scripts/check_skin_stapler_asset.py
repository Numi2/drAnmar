#!/usr/bin/env python3
"""Verify the repository-local DrAnmar skin stapler import contract."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET = (
    ROOT
    / "source/extensions/orbit.surgical.assets/data"
    / "Props/SurgicalClosure/SkinStapler"
)
REPORT = ASSET / "integration_report.json"
REQUIRED_FRAMES = {
    "handle_grasp",
    "trigger_contact",
    "jaw_tip",
    "staple_exit",
    "placement_reference",
    "count_reference",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_hashes(mapping: dict[str, str]) -> None:
    for relative_path, expected in mapping.items():
        path = ASSET / relative_path
        require(path.is_file(), f"Missing skin stapler file: {relative_path}")
        require(
            sha256(path) == expected,
            f"Skin stapler hash mismatch: {relative_path}",
        )


def check_openusd(files: tuple[Path, ...]) -> None:
    for path in files:
        text = path.read_text(encoding="utf-8")
        require(
            "token inputs:varname" not in text,
            f"Invalid token primvar-reader input remains in {path.name}",
        )
    require(
        sum(
            path.read_text(encoding="utf-8").count('string inputs:varname = "st"')
            for path in files
        )
        == 11,
        "Expected exactly 11 corrected string primvar-reader inputs",
    )

    for command in ("usdcat", "usdchecker"):
        executable = shutil.which(command)
        if executable is None:
            continue
        for path in files:
            result = subprocess.run(
                [executable, str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            require(
                result.returncode == 0,
                f"{command} rejected {path.name}: {result.stderr[-1000:]}",
            )


def main() -> int:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    require(
        report["schema"] == "dr.anmar.skin-stapler-integration.v1",
        "Unsupported skin stapler integration-report schema",
    )
    require(
        report["catalog_subpath"] == "Props/SurgicalClosure/SkinStapler",
        "Unexpected skin stapler catalog path",
    )

    integrated = report["integrated_payload"]
    check_hashes(integrated["runtime_layers"])
    check_hashes(integrated["contracts"])

    files = tuple(ASSET.rglob("*"))
    payload_files = tuple(path for path in files if path.is_file())
    require(
        len(payload_files) == integrated["files_including_integration_metadata"],
        "Integrated skin stapler file count changed",
    )
    require(
        sum(path.stat().st_size for path in payload_files) == integrated["bytes"],
        "Integrated skin stapler byte count changed",
    )

    interaction = json.loads((ASSET / "interaction_frames.json").read_text(encoding="utf-8"))
    require(
        REQUIRED_FRAMES.issubset(interaction["frames"]),
        "Skin stapler interaction-frame contract is incomplete",
    )
    profile = json.loads((ASSET / "physics_profile.json").read_text(encoding="utf-8"))
    require(profile["clinical_status"].startswith("not_clinically_validated"), "Clinical boundary missing")
    require(
        set(profile["states"]) == {"loaded", "empty"},
        "Skin stapler loaded/empty state contract changed",
    )

    usd_files = tuple(ASSET / name for name in integrated["runtime_layers"])
    check_openusd(usd_files)
    print(
        "Skin stapler asset check passed: "
        f"{len(payload_files)} files, {integrated['bytes']} bytes, "
        "3 OpenUSD runtime layers."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
