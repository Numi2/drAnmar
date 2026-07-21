# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Pinned native-provider adapter for SonoGym orthopedic ultrasound.

SonoGym owns the Isaac Lab environments, CT-derived patient data, ultrasound
generation, robot state, observations, rewards, and safety constraints.
Dr.Anmar only launches the upstream task through a small browser-control bridge
and adds clinician-facing curriculum, recording provenance, and evaluation.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any


APP_ROOT = Path(os.environ.get("DR_ANMAR_ROOT", Path.home() / ".local/share/dr-anmar")).expanduser()
SONOGYM_INSTALL_ROOT = Path(
    os.environ.get("DR_ANMAR_SONOGYM_INSTALL_ROOT", APP_ROOT / "sonogym")
).expanduser()
SONOGYM_ROOT = Path(
    os.environ.get("DR_ANMAR_SONOGYM_ROOT", SONOGYM_INSTALL_ROOT / "vendor/SonoGym")
).expanduser()
SONOGYM_PYTHON = Path(
    os.environ.get("DR_ANMAR_SONOGYM_PYTHON", SONOGYM_INSTALL_ROOT / "env_isaaclab/bin/python")
).expanduser()
SONOGYM_COMMIT = "e67be58334d1a5274f0913af36f56e4b0b7ffe5a"
SONOGYM_ASSETS_COMMIT = "b37b080a8673f856266a2306724e48d5e034521a"
SONOGYM_ISAACLAB_RELEASE = "2.1.0"


SONOGYM_TASKS: dict[str, dict[str, Any]] = {
    "l4_ultrasound_navigation": {
        "title": "L4 ultrasound navigation",
        "description": "Move the robotic probe to the transverse plane through the centre of the L4 vertebra.",
        "task": "Isaac-robot-US-guidance-v0",
        "procedure_id": "orthopedic-l4-ultrasound-navigation",
        "action_dimensions": 3,
        "category": "manual_practice",
    },
    "l4_surface_reconstruction": {
        "title": "L4 surface reconstruction",
        "description": "Sweep the probe to reconstruct the L4 bone surface from simulated ultrasound observations.",
        "task": "Isaac-robot-US-reconstruction-v0",
        "procedure_id": "orthopedic-l4-surface-reconstruction",
        "action_dimensions": 4,
        "category": "manual_practice",
    },
    "l4_ultrasound_guided_surgery": {
        "title": "L4 ultrasound-guided surgery",
        "description": "Coordinate ultrasound localization and a constrained orthopedic instrument trajectory toward L4.",
        "task": "Isaac-robot-US-guided-surgery-v0",
        "procedure_id": "orthopedic-l4-ultrasound-guided-surgery",
        "action_dimensions": 6,
        "category": "manual_practice",
    },
}


def _git_head(root: Path) -> str | None:
    try:
        value = (root / ".git/HEAD").read_text().strip()
        if value.startswith("ref: "):
            value = (root / ".git" / value[5:]).read_text().strip()
        return value
    except OSError:
        return None


def runtime_prerequisites() -> dict[str, dict[str, Any]]:
    package_root = SONOGYM_ROOT / "source/spinal_surgery/spinal_surgery"
    assets_root = package_root / "assets/data"
    models_root = SONOGYM_ROOT / "models"
    python_ready = SONOGYM_PYTHON.is_file() and os.access(SONOGYM_PYTHON, os.X_OK)
    source_head = _git_head(SONOGYM_ROOT)
    try:
        install_manifest = json.loads((SONOGYM_INSTALL_ROOT / "install_manifest.json").read_text())
    except (OSError, ValueError):
        install_manifest = {}
    assets_pinned = install_manifest.get("assets_commit") == SONOGYM_ASSETS_COMMIT
    return {
        "source_pin": {
            "ready": source_head == SONOGYM_COMMIT,
            "path": str(SONOGYM_ROOT),
            "expected_commit": SONOGYM_COMMIT,
            "installed_commit": source_head,
        },
        "isaaclab_runtime": {
            "ready": python_ready,
            "path": str(SONOGYM_PYTHON),
            "release": SONOGYM_ISAACLAB_RELEASE,
        },
        "orthopedic_assets": {
            "ready": (assets_root / "HumanModels").is_dir() and assets_pinned,
            "path": str(assets_root),
            "expected_commit": SONOGYM_ASSETS_COMMIT,
            "installed_commit": install_manifest.get("assets_commit"),
        },
        "ultrasound_models": {
            "ready": models_root.is_dir() and any(models_root.rglob("*.pth")) and assets_pinned,
            "path": str(models_root),
            "expected_commit": SONOGYM_ASSETS_COMMIT,
            "installed_commit": install_manifest.get("assets_commit"),
        },
    }


def workflow_modes() -> dict[str, Any]:
    prerequisites = runtime_prerequisites()
    missing = [
        {
            "source_pin": "Pinned SonoGym source",
            "isaaclab_runtime": f"Isaac Lab {SONOGYM_ISAACLAB_RELEASE} runtime",
            "orthopedic_assets": "SonoGym CT-derived orthopedic assets",
            "ultrasound_models": "SonoGym ultrasound models",
        }[key]
        for key, value in prerequisites.items()
        if not value["ready"]
    ]
    return {
        "default_mode": "l4_ultrasound_navigation",
        "metadata_ready": True,
        "source_commit": SONOGYM_COMMIT,
        "modes": [
            {
                "id": mode_id,
                **definition,
                "launchable": True,
                "requires_hardware": False,
                "requires_arguments": False,
                "requires_rti": False,
                "recommended": mode_id == "l4_ultrasound_navigation",
                "launch_ready": not missing,
                "missing_prerequisites": list(missing),
                "blocked_reason": None if not missing else "Install the pinned SonoGym runtime and assets.",
            }
            for mode_id, definition in SONOGYM_TASKS.items()
        ],
        "runtime_prerequisites": prerequisites,
    }


def platform_workflow() -> dict[str, Any]:
    prerequisites = runtime_prerequisites()
    runtime_ready = all(item["ready"] for item in prerequisites.values())
    return {
        "id": "sonogym_orthopedics",
        "title": "Orthopedic robotic ultrasound",
        "directory": str(SONOGYM_ROOT),
        "provides": [
            "orthopedic_ultrasound",
            "l4_navigation",
            "bone_reconstruction",
            "ultrasound_guided_surgery",
            "safe_rl",
        ],
        "doctor_summary": "Learn robotic lumbar ultrasound navigation, reconstruction, and image-guided intervention.",
        "doctor_default_mode": "l4_ultrasound_navigation",
        "installed": SONOGYM_ROOT.is_dir(),
        "source_ready": SONOGYM_ROOT.is_dir(),
        "runtime_validated": runtime_ready,
        "path": str(SONOGYM_ROOT),
        "inspect_command": "Read the pinned SonoGym README and task configurations",
        **workflow_modes(),
    }


def launch_command(*, mode_id: str, bridge: Path, port: int) -> list[str]:
    mode = SONOGYM_TASKS.get(mode_id)
    if mode is None:
        raise KeyError(mode_id)
    return [
        str(SONOGYM_PYTHON),
        str(bridge),
        "--sonogym-root",
        str(SONOGYM_ROOT),
        "--task",
        str(mode["task"]),
        "--procedure-id",
        str(mode["procedure_id"]),
        "--title",
        str(mode["title"]),
        "--port",
        str(port),
        "--headless",
    ]
