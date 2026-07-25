#!/usr/bin/env python3
"""Validate a complete Dr.Anmar perfusion-viability development release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import zipfile
from pathlib import Path

CATALOG_PATH = Path("Props/SurgicalAssessment/PerfusionViabilityRobot")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


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


def validate_usda(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    require(text.startswith("#usda "), f"missing USDA header: {path}")
    require(text.count("{") == text.count("}"), f"unbalanced USDA blocks: {path}")
    require(
        not re.search(r'^\s*over\s+"[^"]+"\s*\{[^}\n]*\}\s*$', text, re.MULTILINE),
        f"one-line over declaration: {path}",
    )
    require(
        not re.search(r"customData\s*=\s*\{[^}\n]+\}", text),
        f"one-line customData dictionary: {path}",
    )
    require(
        not re.search(r"quat[fd]\s+\w+\s*=\s*\([^()]*,\s*\([^()]+\)\)", text),
        f"nested quaternion syntax: {path}",
    )


def validate_glb(path: Path) -> None:
    data = path.read_bytes()
    require(len(data) >= 20, f"GLB is too short: {path}")
    magic, version, declared_length = struct.unpack_from("<4sII", data)
    require(magic == b"glTF" and version == 2, f"invalid GLB header: {path}")
    require(declared_length == len(data), f"GLB length mismatch: {path}")
    json_length, json_type = struct.unpack_from("<II", data, 12)
    require(json_type == 0x4E4F534A, f"GLB lacks leading JSON chunk: {path}")
    json.loads(data[20 : 20 + json_length].decode("utf-8").rstrip(" \x00"))


def validate_png(path: Path) -> None:
    data = path.read_bytes()
    require(data.startswith(PNG_SIGNATURE), f"invalid PNG signature: {path}")
    require(len(data) >= 33 and data[12:16] == b"IHDR", f"PNG lacks IHDR: {path}")
    width, height = struct.unpack_from(">II", data, 16)
    require(width > 0 and height > 0, f"PNG has invalid dimensions: {path}")
    require(data[-8:-4] == b"IEND", f"PNG lacks terminal IEND: {path}")


def validate_archive(path: Path) -> int:
    require(path.is_file(), f"archive missing: {path}")
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None, f"archive CRC failure: {path}")
        names = archive.namelist()
        require(names and all(not name.startswith("/") for name in names), f"unsafe archive path: {path}")
        require(
            all(".." not in Path(name).parts for name in names),
            f"archive contains parent traversal: {path}",
        )
        return len(names)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--archives-dir",
        type=Path,
        help="also validate the three sibling archives and release record",
    )
    parser.add_argument(
        "--require-pxr",
        action="store_true",
        help="require all 14 USDA files to compose through pxr.Usd.Stage.Open",
    )
    args = parser.parse_args()
    root = args.package_root.resolve()
    manifest_path = root / "SHA256SUMS.json"
    require(manifest_path.is_file(), f"checksum manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "SHA256SUMS.json"
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    require(set(manifest) == actual_files, "checksum manifest inventory differs from package")
    for relative, expected in manifest.items():
        require(sha256(root / relative) == expected, f"checksum mismatch: {relative}")

    asset_root = root / "assets" / CATALOG_PATH
    mirror_root = (
        root
        / "source/extensions/orbit.surgical.assets/data"
        / CATALOG_PATH
    )
    primary_usda = sorted(asset_root.glob("*.usda"))
    mirrored_usda = sorted(mirror_root.glob("*.usda"))
    require(len(primary_usda) == len(mirrored_usda) == 7, "expected 14 mirrored USDA files")
    for primary, mirror in zip(primary_usda, mirrored_usda):
        require(primary.name == mirror.name, "mirrored USDA names differ")
        validate_usda(primary)
        validate_usda(mirror)
        require(primary.read_bytes() == mirror.read_bytes(), f"USDA mirror differs: {primary.name}")
    if args.require_pxr:
        try:
            from pxr import Usd
        except ImportError as exc:
            raise ValidationError("pxr is required but unavailable") from exc
        for path in primary_usda + mirrored_usda:
            require(Usd.Stage.Open(str(path)) is not None, f"OpenUSD parse failed: {path}")

    glb_files = sorted(asset_root.joinpath("glb").glob("*.glb"))
    png_files = sorted(asset_root.joinpath("textures").glob("*.png"))
    require(len(glb_files) == 23, "unexpected GLB count")
    require(len(png_files) == 8, "unexpected PNG count")
    for path in glb_files:
        validate_glb(path)
    for path in png_files:
        validate_png(path)
    json_files = sorted(root.rglob("*.json"))
    for path in json_files:
        json.loads(path.read_text(encoding="utf-8"))
    python_files = sorted(root.rglob("*.py"))
    for path in python_files:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")

    static_report = json.loads((root / "static_build_report.json").read_text(encoding="utf-8"))
    require(static_report.get("status") == "pass", "static build report does not pass")
    version = json.loads((asset_root / "asset_manifest.json").read_text(encoding="utf-8"))["version"]
    qualification = json.loads((asset_root / "qualification_report.json").read_text(encoding="utf-8"))
    require(qualification.get("version") == version, "qualification version differs")
    require(qualification.get("status") == "pass", "qualification does not pass")

    archives = {}
    if args.archives_dir:
        archive_dir = args.archives_dir.resolve()
        names = {
            "development": f"dranmar_perfusion_viability_robot_v{version}.zip",
            "catalog": f"dranmar_perfusion_viability_robot_catalog_v{version}.zip",
            "overlay": f"dranmar_perfusion_viability_robot_repo_overlay_v{version}.zip",
        }
        for label, name in names.items():
            path = archive_dir / name
            archives[label] = {
                "path": str(path),
                "entries": validate_archive(path),
                "sha256": sha256(path),
            }
            checksum_path = path.with_suffix(path.suffix + ".sha256")
            require(checksum_path.is_file(), f"archive checksum missing: {checksum_path}")
            require(
                archives[label]["sha256"] in checksum_path.read_text(encoding="utf-8"),
                f"archive checksum file differs: {checksum_path}",
            )
        release_path = archive_dir / f"dranmar_perfusion_viability_robot_release_v{version}.json"
        release = json.loads(release_path.read_text(encoding="utf-8"))
        require(release.get("version") == version, "release-record version differs")

    result = {
        "schema": "dr.anmar.perfusion-viability-release-validation.v1",
        "status": "pass",
        "version": version,
        "hashed_files": len(manifest),
        "mirrored_usda": len(primary_usda) + len(mirrored_usda),
        "glb": len(glb_files),
        "png": len(png_files),
        "json": len(json_files),
        "python_compiled": len(python_files),
        "pxr_composed_usda": (
            len(primary_usda) + len(mirrored_usda)
            if args.require_pxr
            else 0
        ),
        "archives": archives,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        raise SystemExit(f"validation failed: {exc}")
