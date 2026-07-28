#!/usr/bin/env python3
"""Capture exact PhysX jaw-contact partners for qualification evidence.

The collector subscribes to native PhysX contact reports only while a play
benchmark is active. It never writes simulator state, policy observations,
actions, rewards, or terminations.
"""

from __future__ import annotations

import math
import re
from typing import Any


_JAW_COLLIDER = re.compile(
    r"^/World/envs/env_(?P<env>\d+)/"
    r"(?P<robot>Robot_[12])/"
    r"psm_tool_gripper(?P<jaw>[12])_link(?:/|$)"
)
_ENV_PREFIX = re.compile(r"^/World/envs/env_\d+")
_PARTNER_ROBOT = re.compile(r"^\{ENV\}/(?P<robot>Robot_[12])(?:/|$)")


def _normalized_collider_path(path: str) -> str:
    """Remove the concrete environment index from a collider path."""
    return _ENV_PREFIX.sub("{ENV}", path)


def _partner_category(reporter_robot: str, partner_path: str) -> str:
    """Classify one non-object contact partner without geometric guesses."""
    normalized = _normalized_collider_path(partner_path)
    if normalized.startswith("{ENV}/Table/"):
        return "support_table"
    if normalized.startswith("{ENV}/GroundPlane"):
        return "ground_plane"
    robot_match = _PARTNER_ROBOT.match(normalized)
    if robot_match is not None:
        partner_robot = robot_match.group("robot")
        relationship = (
            "counterpart" if partner_robot != reporter_robot else "same_robot"
        )
        if "/psm_tool_gripper" in normalized:
            return f"{relationship}_jaws"
        return f"{relationship}_arm_or_wrist"
    return "remaining_scene_geometry"


def _impulse_magnitude(impulse: Any) -> float:
    """Return the magnitude of a scalar or vector contact impulse."""
    try:
        components = tuple(float(component) for component in impulse)
    except TypeError:
        return abs(float(impulse))
    return math.sqrt(sum(component * component for component in components))


class PhysxJawContactAttributionCollector:
    """Collect maximum same-step non-object impulses by jaw and environment."""

    def __init__(self, num_envs: int):
        self._num_envs = int(num_envs)
        self._current_events: dict[tuple[int, int], dict[str, Any]] = {}
        self._subscription = None
        self._path_decoder = None
        self._active_event_types: set[Any] = set()

    def start(self) -> None:
        """Subscribe after the Isaac application and stage are initialized."""
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType
        from pxr import PhysicsSchemaTools

        self._path_decoder = PhysicsSchemaTools.intToSdfPath
        self._active_event_types = {
            ContactEventType.CONTACT_FOUND,
            ContactEventType.CONTACT_PERSIST,
        }
        self._subscription = (
            get_physx_simulation_interface().subscribe_contact_report_events(
                self._on_contact_report_event
            )
        )

    def close(self) -> None:
        """Release the native subscription before closing the environment."""
        self._subscription = None

    def begin_control_step(self) -> None:
        """Discard the preceding control step's contacts."""
        self._current_events.clear()

    def events_for_environments(
        self,
        environment_indices: list[int],
    ) -> dict[int, list[dict[str, Any]]]:
        """Return stable JSON-ready records for selected environments."""
        selected = set(environment_indices)
        records: dict[int, list[dict[str, Any]]] = {
            env_index: [] for env_index in environment_indices
        }
        for (env_index, _), event in sorted(self._current_events.items()):
            if env_index in selected:
                records[env_index].append(dict(event))
        return records

    def _on_contact_report_event(self, contact_headers, contact_data) -> None:
        assert self._path_decoder is not None
        for header in contact_headers:
            if header.type not in self._active_event_types:
                continue
            collider_0 = str(self._path_decoder(header.collider0))
            collider_1 = str(self._path_decoder(header.collider1))
            maximum_impulse = 0.0
            start = header.contact_data_offset
            stop = start + header.num_contact_data
            for index in range(start, stop):
                maximum_impulse = max(
                    maximum_impulse,
                    _impulse_magnitude(contact_data[index].impulse),
                )
            self._record_pair(collider_0, collider_1, maximum_impulse)
            self._record_pair(collider_1, collider_0, maximum_impulse)

    def _record_pair(
        self,
        reporter_path: str,
        partner_path: str,
        maximum_impulse: float,
    ) -> None:
        reporter = _JAW_COLLIDER.match(reporter_path)
        if reporter is None or "/Object" in partner_path:
            return
        env_index = int(reporter.group("env"))
        if not 0 <= env_index < self._num_envs:
            return
        robot = reporter.group("robot")
        jaw = int(reporter.group("jaw"))
        sensor_index = (0 if robot == "Robot_1" else 2) + jaw - 1
        key = (env_index, sensor_index)
        existing = self._current_events.get(key)
        if (
            existing is not None
            and existing["maximum_contact_impulse_ns"] >= maximum_impulse
        ):
            return
        self._current_events[key] = {
            "sensor_index": sensor_index,
            "reporter_robot": robot.lower(),
            "reporter_jaw": jaw,
            "reporter_collider": _normalized_collider_path(reporter_path),
            "partner_collider": _normalized_collider_path(partner_path),
            "partner_category": _partner_category(robot, partner_path),
            "maximum_contact_impulse_ns": maximum_impulse,
        }
