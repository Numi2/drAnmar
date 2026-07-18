# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical catalog for the complete ORBIT-Surgical digital-twin suite."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TaskFamily:
    slug: str
    name: str
    stem: str
    robot: str
    arms: int
    procedure: str
    training_object: str | None
    description: str


FAMILIES = (
    TaskFamily("psm-reach", "PSM precision reach", "Isaac-Reach-PSM", "dVRK PSM", 1, "reach", None, "Single patient-side manipulator pose tracking."),
    TaskFamily("ecm-reach", "ECM endoscope reach", "Isaac-Reach-ECM", "dVRK ECM", 1, "reach", None, "Endoscopic camera manipulator positioning."),
    TaskFamily("star-reach", "STAR precision reach", "Isaac-Reach-STAR", "STAR", 1, "reach", None, "Smart Tissue Autonomous Robot pose tracking."),
    TaskFamily("dual-psm-reach", "Dual PSM coordination", "Isaac-Reach-Dual-PSM", "dVRK PSM", 2, "dual reach", None, "Coordinated bimanual patient-side manipulation."),
    TaskFamily("dual-star-reach", "Dual STAR coordination", "Isaac-Reach-Dual-STAR", "STAR", 2, "dual reach", None, "Coordinated bimanual STAR pose tracking."),
    TaskFamily("block-lift", "PSM block lift", "Isaac-Lift-Block-PSM", "dVRK PSM", 1, "lift", "surgical block", "Grasp and lift a rigid training block."),
    TaskFamily("needle-lift", "PSM needle lift", "Isaac-Lift-Needle-PSM", "dVRK PSM", 1, "lift", "suture needle", "Grasp and lift a curved suture needle."),
    TaskFamily("block-handover", "Dual PSM block handover", "Isaac-Handover-Block-Dual-PSM", "dVRK PSM", 2, "handover", "surgical block", "Pass a training block between two instruments."),
    TaskFamily("needle-handover", "Dual PSM needle handover", "Isaac-Handover-Needle-Dual-PSM", "dVRK PSM", 2, "handover", "suture needle", "Pass a curved suture needle between two instruments."),
)

VARIANTS = (
    ("joint", "Joint position", "", False, False),
    ("joint-play", "Joint position · play", "-Play", True, False),
    ("ik-abs", "Absolute IK", "-IK-Abs", False, False),
    ("ik-abs-play", "Absolute IK · play", "-IK-Abs-Play", True, False),
    ("ik-rel", "Relative IK", "-IK-Rel", False, True),
    ("ik-rel-play", "Relative IK · play", "-IK-Rel-Play", True, True),
)


def build_catalog() -> list[dict]:
    tasks: list[dict] = []
    for family in FAMILIES:
        for variant_slug, variant_name, suffix, play, browser_control in VARIANTS:
            task_id = f"{family.stem}{suffix}-v0"
            item = asdict(family)
            item.update(
                {
                    "id": task_id,
                    "variant": variant_slug,
                    "variant_name": variant_name,
                    "play": play,
                    "browser_control": browser_control,
                    "recommended": variant_slug == "ik-rel",
                }
            )
            tasks.append(item)
    return tasks


CATALOG = build_catalog()
TASKS_BY_ID = {task["id"]: task for task in CATALOG}
PRIMARY_TASKS = [task for task in CATALOG if task["recommended"]]


if __name__ == "__main__":
    print(json.dumps({"families": len(FAMILIES), "tasks": len(CATALOG), "catalog": CATALOG}, indent=2))
