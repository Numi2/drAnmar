#!/usr/bin/env python3
"""Validate the installed DrAnmar perfusion and viability robot overlay."""

from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = (
    ROOT
    / "source/extensions/orbit.surgical.assets/data"
    / "Props/SurgicalAssessment/PerfusionViabilityRobot"
)
TASK_PHASES = (
    "inspect",
    "rgb",
    "icg",
    "speckle",
    "thermal",
    "oxygenation",
    "doppler",
    "ultrasound",
    "fuse",
    "diagnose",
    "intervene",
    "rescan",
    "verify",
)
PYTHON_FILES = (
    ROOT / "examples/franka_perfusion_viability_scene.py",
    ROOT / "examples/validate_perfusion_viability_runtime.py",
    ROOT / "scripts/dranmar_asset_authoring.py",
    ROOT / "scripts/generate_dranmar_perfusion_viability_robot.py",
    ROOT / "scripts/validate_dranmar_perfusion_viability_robot.py",
    ROOT
    / "source/extensions/orbit.surgical.assets/orbit/surgical/assets"
    / "perfusion_viability_robot.py",
    ROOT / "tests/test_perfusion_viability_robot.py",
)
EXPECTED_COUNTS = {"usda": 7, "glb": 23, "png": 8}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def validate_usda(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    require(text.startswith("#usda "), f"missing USDA header: {path}")
    require(text.count("{") == text.count("}"), f"unbalanced USDA blocks: {path}")
    require(
        not re.search(r"^\s*over\s+\"[^\"]+\"\s*\{[^}\n]*\}\s*$", text, re.MULTILINE),
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
    if 'variantSet "sensor_state"' in text:
        require(
            text.count('            over "Links"') == 7,
            f"variant branches contain duplicate Links opinions: {path}",
        )


def validate_glb(path: Path) -> None:
    data = path.read_bytes()
    require(len(data) >= 20, f"GLB is too short: {path}")
    magic, version, declared_length = struct.unpack_from("<4sII", data)
    require(magic == b"glTF", f"invalid GLB magic: {path}")
    require(version == 2, f"unsupported GLB version: {path}")
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


def main() -> int:
    require(ASSET_ROOT.is_dir(), f"asset root is missing: {ASSET_ROOT}")
    usda_files = sorted(ASSET_ROOT.glob("*.usda"))
    glb_files = sorted((ASSET_ROOT / "glb").glob("*.glb"))
    png_files = sorted((ASSET_ROOT / "textures").glob("*.png"))
    require(len(usda_files) == EXPECTED_COUNTS["usda"], "unexpected primary USDA count")
    require(len(glb_files) == EXPECTED_COUNTS["glb"], "unexpected GLB count")
    require(len(png_files) == EXPECTED_COUNTS["png"], "unexpected texture count")

    for path in usda_files:
        validate_usda(path)
    for path in glb_files:
        validate_glb(path)
    for path in png_files:
        validate_png(path)

    json_files = sorted(ASSET_ROOT.glob("*.json")) + [
        ROOT / "physics_next/dr-anmar-assets.json",
        ROOT / "physics_next/surgical-assessment/dranmar-perfusion-viability-v1.json",
    ]
    for path in json_files:
        require(path.is_file(), f"JSON contract is missing: {path}")
        json.loads(path.read_text(encoding="utf-8"))

    for path in PYTHON_FILES:
        require(path.is_file(), f"Python surface is missing: {path}")
        compile(path.read_text(encoding="utf-8"), str(path), "exec")

    task = json.loads(
        (ASSET_ROOT / "perfusion_viability_task_contract.json").read_text(
            encoding="utf-8"
        )
    )
    profile = json.loads((ASSET_ROOT / "physics_profile.json").read_text(encoding="utf-8"))
    modalities = json.loads(
        (ASSET_ROOT / "sensor_modalities.json").read_text(encoding="utf-8")
    )
    graph = json.loads((ASSET_ROOT / "perfusion_network.json").read_text(encoding="utf-8"))
    require(tuple(task["phases"]) == TASK_PHASES, "task-phase sequence differs")
    require(task.get("research_only") is True, "task must remain research-only")
    require(profile["tool"]["joint_count"] == 12, "tool joint count differs")
    require(len(graph["regions"]) == 24, "perfusion region count differs")
    require(len(graph["nodes"]) == 60, "vascular node count differs")
    require(len(graph["edges"]) == 82, "vascular edge count differs")
    require(len(modalities["modalities"]) >= 8, "multimodal sensor contract is incomplete")

    qualification = ASSET_ROOT / "qualification_report.json"
    if qualification.exists():
        report = json.loads(qualification.read_text(encoding="utf-8"))
        matrix = report.get("matrix", [])
        require(report.get("status") == "pass", "qualification report does not pass")
        require(
            {entry.get("representation") for entry in matrix}
            == {"standalone", "franka"}
            and all(entry.get("status") == "pass" for entry in matrix),
            "native qualification matrix is incomplete",
        )

    result = {
        "schema": "dranmar.perfusion-viability-static-validation.v1",
        "status": "pass",
        "primary_usda": len(usda_files),
        "primary_glb": len(glb_files),
        "primary_textures": len(png_files),
        "json_contracts": len(json_files),
        "python_files_compiled": len(PYTHON_FILES),
        "qualification_report": qualification.exists(),
        "research_only": True,
        "clinical_validation": False,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
