from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(
    str(ROOT / "scripts/dr_anmar_physx_contact_attribution.py")
)


def test_partner_classification_is_environment_invariant() -> None:
    classify = MODULE["_partner_category"]
    assert (
        classify(
            "Robot_2",
            (
                "/World/envs/env_47/Robot_1/psm_tool_gripper1_link/"
                "collisions_xform/collisions"
            ),
        )
        == "counterpart_jaws"
    )
    assert (
        classify(
            "Robot_2",
            "/World/envs/env_47/Robot_1/psm_tool_yaw_link/visuals",
        )
        == "counterpart_arm_or_wrist"
    )
    assert (
        classify(
            "Robot_2",
            "/World/envs/env_47/Table/Table/Table",
        )
        == "support_table"
    )


def test_collector_keeps_largest_non_object_event_per_jaw() -> None:
    collector = MODULE["PhysxJawContactAttributionCollector"](num_envs=64)
    reporter = (
        "/World/envs/env_47/Robot_2/psm_tool_gripper1_link/"
        "collisions_xform/collisions"
    )
    collector._record_pair(
        reporter,
        "/World/envs/env_47/Object",
        9.0,
    )
    collector._record_pair(
        reporter,
        "/World/envs/env_47/Table/Table/Table",
        0.02,
    )
    collector._record_pair(
        reporter,
        "/World/envs/env_47/Robot_1/psm_tool_yaw_link/visuals",
        0.01,
    )
    records = collector.events_for_environments([47])
    assert records[47] == [
        {
            "sensor_index": 2,
            "reporter_robot": "robot_2",
            "reporter_jaw": 1,
            "reporter_collider": (
                "{ENV}/Robot_2/psm_tool_gripper1_link/"
                "collisions_xform/collisions"
            ),
            "partner_collider": "{ENV}/Table/Table/Table",
            "partner_category": "support_table",
            "maximum_contact_impulse_ns": 0.02,
        }
    ]
