#!/usr/bin/env python3
"""Static and package validation for the DrAnmar adaptive-anastomosis release."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile


CATALOG = Path("Props/SurgicalReconstruction/AdaptiveAnastomosisRobot")
PRIMARY_USDA_COUNT = 8
PRIMARY_GLB_COUNT = 19
PRIMARY_TEXTURE_COUNT = 6
EXPECTED_PHASES = [
    "inspect", "capture", "align", "mandrel", "approximate", "evert",
    "staple", "release_capture", "reinforce", "occlude", "pressurize",
    "verify", "complete", "abort",
]
FORBIDDEN_NAMES = {".DS_Store"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_glb(path: Path) -> None:
    data = path.read_bytes()
    if len(data) < 12 or data[:4] != b"glTF":
        raise AssertionError(f"Invalid GLB header: {path}")
    if int.from_bytes(data[4:8], "little") != 2:
        raise AssertionError(f"Unsupported GLB version: {path}")
    if int.from_bytes(data[8:12], "little") != len(data):
        raise AssertionError(f"GLB length mismatch: {path}")


def assert_png(path: Path) -> None:
    if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError(f"Invalid PNG header: {path}")


def assert_archive(path: Path) -> None:
    if not path.exists():
        raise AssertionError(f"Missing release archive: {path}")
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise AssertionError(f"CRC failure in {path.name}: {bad}")
        for name in archive.namelist():
            parts = Path(name).parts
            if (
                "__pycache__" in parts
                or name.endswith(".pyc")
                or any(part in FORBIDDEN_NAMES for part in parts)
            ):
                raise AssertionError(f"Forbidden archive entry in {path.name}: {name}")
    checksum = Path(str(path) + ".sha256")
    if not checksum.exists() or checksum.read_text().split()[0] != sha256(path):
        raise AssertionError(f"Checksum mismatch: {path}")


def validate(root: Path, *, installed: bool, require_usdchecker: bool, skip_archives: bool) -> dict[str, object]:
    root = root.resolve()
    if installed:
        asset_root = (
            root / "source/extensions/orbit.surgical.assets/data" / CATALOG
        )
        source_root = root
        mirror_root = None
    else:
        asset_root = root / "assets" / CATALOG
        source_root = root
        mirror_root = (
            root / "source/extensions/orbit.surgical.assets/data" / CATALOG
        )
    if not asset_root.is_dir():
        raise AssertionError(f"Missing catalog directory: {asset_root}")

    usda = sorted(asset_root.glob("*.usda"))
    glb = sorted((asset_root / "glb").glob("*.glb"))
    textures = sorted((asset_root / "textures").glob("*.png"))
    if len(usda) != PRIMARY_USDA_COUNT:
        raise AssertionError(f"Expected {PRIMARY_USDA_COUNT} USDA assets, got {len(usda)}")
    if len(glb) != PRIMARY_GLB_COUNT:
        raise AssertionError(f"Expected {PRIMARY_GLB_COUNT} GLBs, got {len(glb)}")
    if len(textures) != PRIMARY_TEXTURE_COUNT:
        raise AssertionError(
            f"Expected {PRIMARY_TEXTURE_COUNT} textures, got {len(textures)}"
        )
    for path in glb:
        assert_glb(path)
    for path in textures:
        assert_png(path)

    if installed:
        json_paths = list(asset_root.glob("*.json")) + [
            root / "physics_next/surgical-reconstruction/dranmar-adaptive-anastomosis-v1.json",
            root / "physics_next/dr-anmar-assets.json",
        ]
        python_paths = [
            root / "source/extensions/orbit.surgical.assets/orbit/surgical/assets/adaptive_anastomosis_scene_evidence.py",
            root / "source/extensions/orbit.surgical.assets/orbit/surgical/assets/adaptive_anastomosis_robot.py",
            root / "examples/franka_adaptive_anastomosis_scene.py",
            root / "scripts/generate_dranmar_adaptive_anastomosis_robot.py",
            root / "scripts/validate_dranmar_adaptive_anastomosis_robot.py",
        ]
        hygiene_roots = [
            asset_root,
            root / "docs/adaptive_anastomosis_robot",
        ]
    else:
        json_paths = list(source_root.rglob("*.json"))
        python_paths = list(source_root.rglob("*.py"))
        hygiene_roots = [source_root]
    for path in json_paths:
        json.loads(path.read_text(encoding="utf-8"))
    for path in python_paths:
        if not path.is_file():
            raise AssertionError(f"Missing Python source: {path}")
        if "__pycache__" not in path.parts:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
    forbidden = [
        path for hygiene_root in hygiene_roots for path in hygiene_root.rglob("*")
        if "__pycache__" in path.parts
        or path.suffix == ".pyc"
        or path.name in FORBIDDEN_NAMES
    ]
    if forbidden:
        raise AssertionError(f"Forbidden generated files: {forbidden[:5]}")

    task = json.loads(
        (asset_root / "adaptive_anastomosis_task_contract.json").read_text()
    )
    if task.get("phases") != EXPECTED_PHASES:
        raise AssertionError("Task contract phases do not match the canonical sequence")
    profile = json.loads((asset_root / "physics_profile.json").read_text())
    mechanism = profile["mechanism"]
    if mechanism["active_joint_count"] != 14:
        raise AssertionError("Mechanism must expose exactly 14 active joints")
    if len(mechanism["active_joint_names"]) != 14:
        raise AssertionError("Active joint list must contain exactly 14 names")
    if mechanism["fixed_joint_names"] != ["staple_anvil_mount_joint"]:
        raise AssertionError("Fixed anvil joint must be explicit and excluded from active joints")

    manifest = json.loads((asset_root / "asset_manifest.json").read_text())
    entries = manifest["files"]
    if manifest["file_count"] != len(entries):
        raise AssertionError("Manifest file_count mismatch")
    if not installed:
        for entry in entries:
            path = root / entry["path"]
            if not path.is_file():
                raise AssertionError(f"Missing manifest file: {entry['path']}")
            if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
                raise AssertionError(f"Manifest hash mismatch: {entry['path']}")
        if not mirror_root or not mirror_root.is_dir():
            raise AssertionError("Missing extension catalog mirror")
        primary_files = sorted(
            path.relative_to(asset_root)
            for path in asset_root.rglob("*")
            if path.is_file()
        )
        mirror_files = sorted(
            path.relative_to(mirror_root)
            for path in mirror_root.rglob("*")
            if path.is_file()
        )
        if primary_files != mirror_files:
            raise AssertionError("Catalog mirror file list mismatch")
        for relative in primary_files:
            if sha256(asset_root / relative) != sha256(mirror_root / relative):
                raise AssertionError(f"Catalog mirror hash mismatch: {relative}")

    usdchecker = shutil.which("usdchecker")
    if require_usdchecker and not usdchecker:
        raise AssertionError("usdchecker is required but not available")
    if usdchecker:
        for path in usda:
            completed = subprocess.run(
                [usdchecker, str(path)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if completed.returncode:
                raise AssertionError(
                    f"usdchecker failed for {path.name}:\n{completed.stdout}"
                )

    if not installed and not skip_archives:
        parent = root.parent
        archives = [
            parent / "dranmar_adaptive_anastomosis_robot_v0.1.0.zip",
            parent / "dranmar_adaptive_anastomosis_robot_catalog_v0.1.0.zip",
            parent / "dranmar_adaptive_anastomosis_robot_repo_overlay_v0.1.0.zip",
        ]
        for archive in archives:
            assert_archive(archive)

    return {
        "status": "passed",
        "mode": "installed" if installed else "package",
        "root": str(root),
        "primary_usda": len(usda),
        "primary_glb": len(glb),
        "primary_textures": len(textures),
        "manifest_entries": len(entries),
        "usdchecker": bool(usdchecker),
        "archives_checked": not installed and not skip_archives,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--installed", action="store_true")
    parser.add_argument("--require-usdchecker", action="store_true")
    parser.add_argument("--skip-archives", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(
        args.root,
        installed=args.installed,
        require_usdchecker=args.require_usdchecker,
        skip_archives=args.skip_archives,
    ), indent=2))


if __name__ == "__main__":
    main()
