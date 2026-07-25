#!/usr/bin/env python3
"""Validate the DrAnmar SafePlane dissection release or installed overlay."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

CATALOG_SUBPATH = Path("Props/SurgicalDissection/SafePlaneDissectionRobot")
PRIMARY_USDA_COUNT = 13
PRIMARY_GLB_COUNT = 25
PRIMARY_TEXTURE_COUNT = 11
PACKAGE_PNG_COUNT = 24
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PROCEDURE = [
    "inspect",
    "capture",
    "traction",
    "blunt",
    "hydro",
    "guarded_scissors",
    "low_energy",
    "irrigate_and_evacuate",
    "verify_connectivity",
    "release",
    "complete",
]


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eligible_manifest_files(root: Path) -> set[str]:
    mirror = (
        root / "source/extensions/orbit.surgical.assets/data" / CATALOG_SUBPATH
    )
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and path.name not in {
            ".DS_Store",
            "asset_manifest.json",
            "static_build_report.json",
        }
        and not path.is_relative_to(mirror)
    }


def installed_manifest_path(root: Path, relative: str) -> Path | None:
    asset_prefix = f"assets/{CATALOG_SUBPATH.as_posix()}/"
    if relative.startswith(asset_prefix):
        suffix = relative.removeprefix(asset_prefix)
        return (
            root
            / "source/extensions/orbit.surgical.assets/data"
            / CATALOG_SUBPATH
            / suffix
        )
    if relative.startswith(("source/", "physics_next/", "docs/", "examples/", "tests/")):
        return root / relative
    if relative in {
        "scripts/generate_dranmar_safeplane_dissection_robot.py",
        "scripts/requirements_safeplane_dissection_generation.txt",
        "scripts/validate_dranmar_safeplane_dissection_robot.py",
    }:
        return root / relative
    return None


def validate_manifest(root: Path, manifest_path: Path, package_mode: bool) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == "dranmar.asset-manifest.v1", "unexpected manifest schema")
    entries = manifest.get("files")
    require(isinstance(entries, list) and entries, "manifest has no file entries")
    paths = [entry.get("path") for entry in entries]
    require(len(paths) == len(set(paths)), "manifest contains duplicate paths")
    require(
        not any(
            not isinstance(path, str)
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or "__pycache__" in Path(path).parts
            or Path(path).suffix == ".pyc"
            for path in paths
        ),
        "manifest contains an unsafe or non-portable path",
    )
    checked = 0
    for entry in entries:
        path = root / entry["path"] if package_mode else installed_manifest_path(root, entry["path"])
        if path is None:
            continue
        require(path.is_file(), f"manifest file is missing: {entry['path']}")
        require(path.stat().st_size == entry["bytes"], f"manifest byte mismatch: {entry['path']}")
        require(sha256(path) == entry["sha256"], f"manifest hash mismatch: {entry['path']}")
        checked += 1
    if package_mode:
        require(set(paths) == eligible_manifest_files(root), "manifest coverage differs from payload")
    require(checked > 0, "manifest validation checked no files")
    return checked


def validate_glb(path: Path) -> None:
    data = path.read_bytes()
    require(len(data) >= 20, f"GLB is too short: {path}")
    magic, version, declared_length = struct.unpack_from("<4sII", data)
    require(magic == b"glTF", f"invalid GLB magic: {path}")
    require(version == 2, f"unsupported GLB version: {path}")
    require(declared_length == len(data), f"GLB length mismatch: {path}")
    offset = 12
    chunk_types: list[int] = []
    while offset < len(data):
        require(offset + 8 <= len(data), f"truncated GLB chunk header: {path}")
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        require(chunk_length % 4 == 0, f"unaligned GLB chunk: {path}")
        require(offset + chunk_length <= len(data), f"truncated GLB chunk: {path}")
        chunk_types.append(chunk_type)
        offset += chunk_length
    require(offset == len(data), f"GLB trailing bytes: {path}")
    require(chunk_types and chunk_types[0] == 0x4E4F534A, f"GLB lacks JSON chunk: {path}")
    json_length = struct.unpack_from("<I", data, 12)[0]
    json.loads(data[20 : 20 + json_length].decode("utf-8").rstrip(" \x00"))


def validate_png(path: Path) -> None:
    data = path.read_bytes()
    require(data.startswith(PNG_SIGNATURE), f"invalid PNG signature: {path}")
    require(len(data) >= 33 and data[12:16] == b"IHDR", f"PNG lacks IHDR: {path}")
    width, height = struct.unpack_from(">II", data, 16)
    require(width > 0 and height > 0, f"PNG has invalid dimensions: {path}")
    require(data[-8:-4] == b"IEND", f"PNG lacks terminal IEND: {path}")


def run_usdchecker(path: Path, checker: str) -> None:
    result = subprocess.run(
        [checker, str(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    require(result.returncode == 0, f"usdchecker failed for {path}:\n{result.stdout}")


def validate_archives(root: Path) -> int:
    parent = root.parent
    archives = (
        parent / "dranmar_safeplane_dissection_robot_v0.1.0.zip",
        parent / "dranmar_safeplane_dissection_robot_catalog_v0.1.0.zip",
        parent / "dranmar_safeplane_dissection_robot_repo_overlay_v0.1.0.zip",
    )
    for archive in archives:
        require(archive.is_file(), f"release archive is missing: {archive}")
        checksum = archive.with_suffix(archive.suffix + ".sha256")
        require(checksum.is_file(), f"archive checksum is missing: {checksum}")
        recorded = checksum.read_text(encoding="utf-8").split()[0]
        require(recorded == sha256(archive), f"archive checksum mismatch: {archive}")
        with zipfile.ZipFile(archive) as handle:
            require(handle.namelist(), f"archive is empty: {archive}")
            require(handle.testzip() is None, f"archive CRC failure: {archive}")
            for info in handle.infolist():
                require(
                    info.date_time == (2026, 1, 1, 0, 0, 0),
                    f"non-deterministic ZIP timestamp: {archive}:{info.filename}",
                )
                require(
                    not Path(info.filename).is_absolute()
                    and ".." not in Path(info.filename).parts,
                    f"unsafe archive entry: {archive}:{info.filename}",
                )
    release = parent / "dranmar_safeplane_dissection_robot_release_v0.1.0.json"
    require(release.is_file(), "release record is missing")
    payload = json.loads(release.read_text(encoding="utf-8"))
    require(payload.get("asset") == "dranmar-safeplane-dissection-robot-v1", "wrong release asset")
    for key, archive in zip(("development_package", "catalog_package", "repository_overlay"), archives):
        require(payload[key]["sha256"] == sha256(archive), f"release hash mismatch: {key}")
    return len(archives)


def resolve_layout(root: Path) -> tuple[bool, Path, Path]:
    package_asset = root / "assets" / CATALOG_SUBPATH
    installed_asset = (
        root / "source/extensions/orbit.surgical.assets/data" / CATALOG_SUBPATH
    )
    if package_asset.is_dir():
        return True, package_asset, installed_asset
    require(installed_asset.is_dir(), f"cannot locate catalog below {root}")
    return False, installed_asset, installed_asset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--require-usdchecker", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    package_mode, asset_root, mirror_root = resolve_layout(root)

    if package_mode:
        require(not list(root.rglob("*.pyc")), "packaged Python bytecode is present")
        require(not list(root.rglob("__pycache__")), "packaged __pycache__ is present")
        require(not list(root.rglob(".DS_Store")), "workstation metadata is present")

    usda = sorted(asset_root.glob("*.usda"))
    glbs = sorted((asset_root / "glb").glob("*.glb"))
    textures = sorted((asset_root / "textures").glob("*.png"))
    require(len(usda) == PRIMARY_USDA_COUNT, f"expected {PRIMARY_USDA_COUNT} USDA, got {len(usda)}")
    require(len(glbs) == PRIMARY_GLB_COUNT, f"expected {PRIMARY_GLB_COUNT} GLB, got {len(glbs)}")
    require(len(textures) == PRIMARY_TEXTURE_COUNT, f"expected {PRIMARY_TEXTURE_COUNT} textures, got {len(textures)}")
    if package_mode:
        require(len(list(root.rglob("*.png"))) == PACKAGE_PNG_COUNT, "unexpected package PNG count")
        asset_files = {
            path.relative_to(asset_root).as_posix(): sha256(path)
            for path in asset_root.rglob("*")
            if path.is_file()
        }
        mirror_files = {
            path.relative_to(mirror_root).as_posix(): sha256(path)
            for path in mirror_root.rglob("*")
            if path.is_file()
        }
        require(asset_files == mirror_files, "asset and extension-data mirrors differ")

    json_files = sorted(root.rglob("*.json")) if package_mode else (
        sorted(asset_root.rglob("*.json"))
        + [
            root / "physics_next/dr-anmar-assets.json",
            root / "physics_next/surgical-dissection/dranmar-safeplane-dissection-v1.json",
        ]
    )
    python_files = sorted(root.rglob("*.py")) if package_mode else [
        root / "examples/franka_safeplane_dissection_scene.py",
        root / "examples/validate_safeplane_dissection_runtime.py",
        root / "scripts/generate_dranmar_safeplane_dissection_robot.py",
        root / "scripts/validate_dranmar_safeplane_dissection_robot.py",
        root / "source/extensions/orbit.surgical.assets/orbit/surgical/assets/safeplane_dissection_robot.py",
        root / "tests/test_safeplane_dissection_robot.py",
    ]
    require(all(path.is_file() for path in json_files), "installed JSON surface is incomplete")
    require(all(path.is_file() for path in python_files), "installed Python surface is incomplete")
    for path in json_files:
        json.loads(path.read_text(encoding="utf-8"))
    for path in python_files:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    for path in glbs:
        validate_glb(path)
    for path in sorted(root.rglob("*.png")) if package_mode else sorted(asset_root.rglob("*.png")):
        validate_png(path)

    task = json.loads((asset_root / "safeplane_dissection_task_contract.json").read_text(encoding="utf-8"))
    profile = json.loads((asset_root / "physics_profile.json").read_text(encoding="utf-8"))
    topology = json.loads((asset_root / "dissection_topology.json").read_text(encoding="utf-8"))
    require(task.get("procedure") == PROCEDURE, "task procedure contract differs")
    require(profile["tool"]["joint_count"] == 17, "physics profile joint count differs")
    require(profile["tissue_substrate"]["adhesion_bridge_count"] == 28, "profile bridge count differs")
    bridges = topology.get("adhesion_bridges", [])
    require(len(bridges) == 28, "topology bridge count differs")
    classes = {
        name: sum(item["bridge_class"] == name for item in bridges)
        for name in ("loose_connective_fibre", "dense_fibrous_band", "vascularized_adhesion")
    }
    require(classes == {
        "loose_connective_fibre": 18,
        "dense_fibrous_band": 6,
        "vascularized_adhesion": 4,
    }, f"bridge-class distribution differs: {classes}")
    require(set(topology["protected_structures"]) == {"vessel", "nerve", "duct"}, "protected structures differ")

    checker = shutil.which("usdchecker")
    if args.require_usdchecker:
        require(checker is not None, "usdchecker is required but unavailable")
    if checker:
        for path in usda:
            run_usdchecker(path, checker)

    manifest_checked = validate_manifest(root, asset_root / "asset_manifest.json", package_mode)
    required = (
        "docs/safeplane_dissection_robot/VALIDATION.md",
        "docs/safeplane_dissection_robot/FRANKA_INTEGRATION.md",
        "docs/safeplane_dissection_robot/MECHANISM.md",
        "docs/safeplane_dissection_robot/PHYSICAL_DISSECTION.md",
        "docs/safeplane_dissection_robot/PROTECTED_STRUCTURE_SAFETY.md",
        "docs/safeplane_dissection_robot/HYDRODISSECTION_AND_ENERGY.md",
        "examples/validate_safeplane_dissection_runtime.py",
        "tests/test_safeplane_dissection_robot.py",
    )
    require(all((root / path).is_file() for path in required), "release documentation or qualification tooling is incomplete")

    qualification = asset_root / "qualification_report.json"
    if qualification.exists():
        report = json.loads(qualification.read_text(encoding="utf-8"))
        require(report.get("status") == "pass", "qualification report does not pass")
        matrix = report.get("matrix", [])
        require(
            {entry.get("representation") for entry in matrix} == {"standalone", "franka"}
            and all(entry.get("status") == "pass" for entry in matrix),
            "qualification matrix is incomplete",
        )

    archive_count = validate_archives(root) if package_mode else 0
    result = {
        "schema": "dranmar.safeplane-dissection-static-validation.v1",
        "status": "pass",
        "mode": "package" if package_mode else "installed_overlay",
        "primary_usda": len(usda),
        "primary_glb": len(glbs),
        "primary_textures": len(textures),
        "python_files_compiled": len(python_files),
        "json_files": len(json_files),
        "manifest_files_checked": manifest_checked,
        "archives_checked": archive_count,
        "usdchecker": bool(checker),
        "qualification_report": qualification.exists(),
        "clinical_validation": False,
        "medical_device": False,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
