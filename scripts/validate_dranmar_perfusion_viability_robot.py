#!/usr/bin/env python3
"""Validate the installed DrAnmar perfusion and viability robot overlay."""

from __future__ import annotations

import argparse
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
    ROOT / "scripts/validate_dranmar_perfusion_viability_release.py",
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-pxr",
        action="store_true",
        help="require every primary USDA to compose through pxr.Usd.Stage.Open",
    )
    args = parser.parse_args()
    require(ASSET_ROOT.is_dir(), f"asset root is missing: {ASSET_ROOT}")
    usda_files = sorted(ASSET_ROOT.glob("*.usda"))
    glb_files = sorted((ASSET_ROOT / "glb").glob("*.glb"))
    png_files = sorted((ASSET_ROOT / "textures").glob("*.png"))
    require(len(usda_files) == EXPECTED_COUNTS["usda"], "unexpected primary USDA count")
    require(len(glb_files) == EXPECTED_COUNTS["glb"], "unexpected GLB count")
    require(len(png_files) == EXPECTED_COUNTS["png"], "unexpected texture count")

    for path in usda_files:
        validate_usda(path)
    if args.require_pxr:
        try:
            from pxr import Usd
        except ImportError as exc:
            raise ValidationError("pxr is required but unavailable") from exc
        for path in usda_files:
            require(Usd.Stage.Open(str(path)) is not None, f"OpenUSD parse failed: {path}")
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
    require(
        task.get("intended_use") == "simulation_training",
        "task intended use must remain simulation training",
    )
    require(profile["tool"]["joint_count"] == 12, "tool joint count differs")
    require(len(graph["regions"]) == 24, "perfusion region count differs")
    require(len(graph["nodes"]) == 60, "vascular node count differs")
    require(len(graph["edges"]) == 82, "vascular edge count differs")
    require(len(modalities["modalities"]) >= 8, "multimodal sensor contract is incomplete")
    require(
        task.get("diagnostic_input_boundary", "").startswith(
            "inference consumes only registered observable"
        ),
        "blind diagnostic-input boundary is missing",
    )
    require(
        task.get("intervention_rule", "").startswith(
            "recovery advances continuously from physical evidence"
        ),
        "physical intervention evidence rule is missing",
    )
    require(
        modalities.get("runtime_quality_gates", {}).get("explicit_abstention") is True,
        "sensor abstention gate is missing",
    )
    require(
        modalities.get("consumables", {}).get(
            "empty_state_disables_dependent_measurement"
        )
        is True,
        "consumable depletion behavior is missing",
    )

    native_evidence = ASSET_ROOT / "native_simulator_evidence.json"
    if native_evidence.exists():
        report = json.loads(native_evidence.read_text(encoding="utf-8"))
        matrix = report.get("matrix", [])
        require(report.get("status") == "pass", "native simulator evidence does not pass")
        require(
            {entry.get("representation") for entry in matrix}
            == {"standalone", "franka"}
            and all(entry.get("status") == "pass" for entry in matrix),
            "native simulator evidence matrix is incomplete",
        )
        require(report.get("version") == "0.1.1", "native simulator evidence is not v0.1.1")
        require(
            all(entry.get("rendered_registered_camera_count") == 6 for entry in matrix),
            "rendered camera evidence is incomplete",
        )
        require(
            all(entry.get("surface_deformable_fixture_attachments") == 2 for entry in matrix),
            "deformable fixture evidence is incomplete",
        )
        franka = next(
            entry for entry in matrix if entry.get("representation") == "franka"
        )
        require(
            len(franka.get("loaded_arm_motion", {})) == 3,
            "loaded Franka sweep evidence is incomplete",
        )
        bench = report.get("bench_compositor", {})
        require(
            bench.get("status") == "pass"
            and bench.get("active_bench_assets") == ["perfusion_viability_robot"]
            and bench.get("featured_robot_system") == "perfusion_viability_robot",
            "isolated perfusion bench evidence is incomplete",
        )
        require(
            bench.get("featured_station_position_m") == [0.04, 0.04, 0.015]
            and bench.get("featured_substrate_position_m") == [0.04, 0.04, 0.001]
            and bench.get("single_featured_system_policy") is True
            and bench.get("single_active_camera_renderer") is True,
            "bench framing or latency policy differs from the recorded contract",
        )

    result = {
        "schema": "dranmar.perfusion-viability-static-validation.v1",
        "status": "pass",
        "primary_usda": len(usda_files),
        "primary_glb": len(glb_files),
        "primary_textures": len(png_files),
        "json_contracts": len(json_files),
        "python_files_compiled": len(PYTHON_FILES),
        "native_simulator_evidence": native_evidence.exists(),
        "pxr_required": args.require_pxr,
        "intended_use": "simulation_training",
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
