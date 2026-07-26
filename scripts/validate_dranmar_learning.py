#!/usr/bin/env python3
"""Fast static qualification for the DrAnmar Learning Path contract."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "source/extensions/orbit.surgical.tasks/orbit/surgical/tasks"
ASSET_ROOT = ROOT / "source/extensions/orbit.surgical.assets/orbit/surgical/assets"
MANIFEST = ROOT / "config/dranmar_learning_path.json"


def main() -> int:
    failures: list[str] = []
    manifest = json.loads(MANIFEST.read_text())
    stages = manifest.get("stages", [])
    if [stage.get("stage") for stage in stages] != list(range(1, len(stages) + 1)):
        failures.append("learning stages must be contiguous and ordered")
    if not stages or any(not stage.get("task", "").startswith("DrAnmar-") for stage in stages):
        failures.append("every stage must reference a DrAnmar task ID")

    python_files = sorted(TASK_ROOT.rglob("*.py")) + sorted(ASSET_ROOT.rglob("*.py"))
    for path in python_files:
        source = path.read_text()
        try:
            ast.parse(source)
        except SyntaxError as exc:
            failures.append(f"{path.relative_to(ROOT)}:{exc.lineno}: {exc.msg}")
        if "from isaaclab.utils import configclass" in source:
            failures.append(f"{path.relative_to(ROOT)} uses the removed configclass import")

    task_source = "\n".join(path.read_text() for path in TASK_ROOT.rglob("*.py"))
    required_fragments = (
        "DRANMAR_LEARNING_TASK_IDS",
        "clone_in_fabric=True",
        '\"Metrics/success_rate\"',
        "RslRlMLPModelCfg",
        "check_for_nan = True",
        "obs_normalization=True",
    )
    for fragment in required_fragments:
        if fragment not in task_source:
            failures.append(f"missing learning contract fragment: {fragment}")
    if "RslRlPpoActorCriticCfg" in task_source:
        failures.append("deprecated combined actor-critic configuration remains")

    declared_ids = {stage["task"] for stage in stages}
    legacy_registration_source = "\n".join(
        path.read_text() for path in TASK_ROOT.rglob("__init__.py")
    )
    for task_id in declared_ids:
        legacy_id = task_id.replace("DrAnmar-", "Isaac-", 1)
        if legacy_id not in legacy_registration_source:
            failures.append(f"stage task has no source registration: {task_id}")

    if failures:
        print("DrAnmar Learning Path validation: FAIL", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(
        "DrAnmar Learning Path validation: PASS "
        f"({len(stages)} stages, {len(python_files)} Python modules)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
