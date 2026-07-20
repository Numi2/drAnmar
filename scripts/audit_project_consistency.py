#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Fast source-level integrity checks for curriculum, tasks and procedure rooms."""

from __future__ import annotations

import math
import sys

from dr_anmar_catalog import TASKS_BY_ID
from dr_anmar_curriculum import COURSES
from dr_anmar_procedures import PROCEDURE_ROOMS


def duplicates(values: list[str]) -> list[str]:
    return sorted({value for value in values if values.count(value) > 1})


def main() -> int:
    problems: list[str] = []
    course_ids = [course["id"] for course in COURSES]
    lessons = [lesson for course in COURSES for lesson in course["lessons"]]
    lesson_ids = [lesson["id"] for lesson in lessons]
    room_ids = [room["id"] for room in PROCEDURE_ROOMS]
    rooms = {room["id"]: room for room in PROCEDURE_ROOMS}

    for label, values in (("course", course_ids), ("lesson", lesson_ids), ("procedure", room_ids)):
        for value in duplicates(values):
            problems.append(f"duplicate {label} id: {value}")

    for lesson in lessons:
        task = lesson.get("task")
        if lesson.get("mode") in {"live", "train"} and not task:
            problems.append(f"runnable lesson has no task: {lesson['id']}")
        if task and task not in TASKS_BY_ID:
            problems.append(f"lesson uses an unknown task: {lesson['id']} -> {task}")
        procedure_id = lesson.get("procedure_id")
        if procedure_id:
            room = rooms.get(procedure_id)
            if room is None:
                problems.append(f"lesson references an unknown procedure: {lesson['id']} -> {procedure_id}")
            elif task != room["task"]:
                problems.append(f"lesson/procedure task mismatch: {lesson['id']} -> {procedure_id}")

    referenced_rooms = {lesson.get("procedure_id") for lesson in lessons if lesson.get("procedure_id")}
    for room in PROCEDURE_ROOMS:
        if room["task"] not in TASKS_BY_ID:
            problems.append(f"procedure uses an unknown task: {room['id']} -> {room['task']}")
        if room["id"] not in referenced_rooms:
            problems.append(f"procedure has no curriculum lesson: {room['id']}")
        if not room.get("steps"):
            problems.append(f"procedure has no guided steps: {room['id']}")
        for waypoint in room.get("waypoints", []):
            if len(waypoint) != 3 or not all(math.isfinite(float(value)) for value in waypoint):
                problems.append(f"procedure contains an invalid waypoint: {room['id']}")

    if problems:
        print("Project consistency audit failed:", file=sys.stderr)
        for problem in sorted(set(problems)):
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(
        f"Project consistency: {len(COURSES)} courses, {len(lessons)} lessons, "
        f"{len(PROCEDURE_ROOMS)} procedure rooms, all references valid"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
