#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Fast startup check for the already-prepared Dr.Anmar OpenUSD catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.root.expanduser().resolve() / "scenes/openusd/manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 1
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != 7:
        return 1
    scene_ids = {scene.get("id") for scene in scenes if isinstance(scene, dict)}
    if len(scene_ids) != len(scenes) or None in scene_ids:
        return 1
    openusd_root = manifest_path.parent.resolve()
    required_keys = ("runtime_organ_usd", "environment_usd", "composed_usd")
    for scene in scenes:
        if not isinstance(scene, dict) or scene.get("error"):
            return 1
        for key in required_keys:
            value = scene.get(key)
            path = Path(value).expanduser().resolve() if value else None
            if not path or openusd_root not in path.parents or not path.is_file():
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
