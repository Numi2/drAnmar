# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Fast, non-simulation readiness report for the Dr.Anmar suite."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from dr_anmar_catalog import CATALOG, FAMILIES


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "source/extensions/orbit.surgical.assets/data"
REQUIRED_ASSETS = (
    "Robots/dVRK/PSM/psm_col.usd",
    "Robots/dVRK/ECM/ecm.usd",
    "Robots/STAR/star.usd",
    "Props/Surgical_block/block.usd",
    "Props/Surgical_needle/needle_sdf.usd",
    "Props/Table/table.usd",
)
PACKAGES = ("isaaclab", "isaacsim", "torch", "rl_games", "rsl_rl", "stable_baselines3", "skrl", "robomimic", "h5py")


def main() -> None:
    assets = {name: (ASSET_ROOT / name).is_file() for name in REQUIRED_ASSETS}
    packages = {name: importlib.util.find_spec(name) is not None for name in PACKAGES}
    report = {
        "ready": all(assets.values()) and all(packages.values()) and len(CATALOG) == 54,
        "task_families": len(FAMILIES),
        "registered_task_variants_expected": len(CATALOG),
        "assets": assets,
        "packages": packages,
        "browser_interactive_tasks": [task["id"] for task in CATALOG if task["recommended"]],
        "simulation_only": True,
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ready"] else 1)


if __name__ == "__main__":
    main()
