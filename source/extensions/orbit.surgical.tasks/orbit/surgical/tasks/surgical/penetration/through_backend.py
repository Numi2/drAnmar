# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Native curved-needle passage geometry layered over qualified entry mechanics."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .backend import (
    CouplingWrench,
    DrAnmarNativeTissueEntryBackend,
    NeedlePose,
    TissueEntryBackendMetadata,
)


DRANMAR_NATIVE_THROUGH_REVISION = "dranmar-native-tissue-through-v1"


@dataclass(frozen=True)
class ThroughTissueSceneState:
    representation: str
    representation_switch_count: int
    exit_event_count: int
    surface_displacement_m: float
    local_strain: float
    embedded_arc_length_m: float
    exposed_arc_length_m: float
    exposed_fraction: float
    exit_position_m: tuple[float, float, float]
    finite: bool


@dataclass
class _ThroughState:
    exit_event_count: int = 0
    tip_was_outside: bool = False
    embedded_arc_length_m: float = 0.0
    exposed_arc_length_m: float = 0.0
    exit_position_m: tuple[float, float, float] = (0.0, 0.0, -0.003)


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class DrAnmarNativeTissueThroughBackend(DrAnmarNativeTissueEntryBackend):
    """Track the curved centerline through the slab and own the exit event."""

    through_sample_count = 129

    def __init__(self, num_scenes: int, *, integration_step_s: float = 0.002) -> None:
        super().__init__(num_scenes, integration_step_s=integration_step_s)
        self._through = [_ThroughState() for _ in range(num_scenes)]
        self._metadata = TissueEntryBackendMetadata(
            provider="dranmar_native_through",
            revision=DRANMAR_NATIVE_THROUGH_REVISION,
            implementation_sha256=_source_sha256(),
            integration_step_s=self.integration_step_s,
        )

    @property
    def scene_state(self) -> tuple[ThroughTissueSceneState, ...]:
        entry_states = super().scene_state
        total_length = math.pi * self.curvature_radius_m
        return tuple(
            ThroughTissueSceneState(
                representation=entry.representation,
                representation_switch_count=entry.representation_switch_count,
                exit_event_count=through.exit_event_count,
                surface_displacement_m=entry.surface_displacement_m,
                local_strain=entry.local_strain,
                embedded_arc_length_m=through.embedded_arc_length_m,
                exposed_arc_length_m=through.exposed_arc_length_m,
                exposed_fraction=through.exposed_arc_length_m / total_length,
                exit_position_m=through.exit_position_m,
                finite=entry.finite,
            )
            for entry, through in zip(entry_states, self._through, strict=True)
        )

    @classmethod
    def _arc_points(cls, pose: NeedlePose) -> list[tuple[float, float, float]]:
        points: list[tuple[float, float, float]] = []
        for index in range(cls.through_sample_count):
            # The authored sharp tip is at -pi/2 and advances along local +X.
            # Traverse from that tip toward the needle body in the decreasing
            # angular direction so the centerline trails, rather than leads,
            # the sharp point through the tissue.
            angle = -0.5 * math.pi - math.pi * index / (cls.through_sample_count - 1)
            local = (
                cls.curvature_radius_m * math.cos(angle),
                cls.curvature_radius_m * math.sin(angle),
                0.0,
            )
            offset = cls._rotate_xyzw(pose.quaternion_xyzw, local)
            points.append(tuple(pose.position[axis] + offset[axis] for axis in range(3)))
        return points

    def _update_through_geometry(
        self, index: int, pose: NeedlePose, punctured: bool
    ) -> None:
        state = self._through[index]
        if not punctured:
            state.exit_event_count = 0
            state.tip_was_outside = False
            state.embedded_arc_length_m = 0.0
            state.exposed_arc_length_m = 0.0
            state.exit_position_m = (0.0, 0.0, -0.003)
            return
        points = self._arc_points(pose)
        bottom_z = self.surface_z_m - self.tissue_thickness_m
        segment_length = math.pi * self.curvature_radius_m / (len(points) - 1)
        embedded = 0.0
        exposed = 0.0
        leading_outside = True
        exit_position = state.exit_position_m
        crossing_found = False
        for first, second in zip(points[:-1], points[1:], strict=True):
            midpoint_z = 0.5 * (first[2] + second[2])
            if bottom_z <= midpoint_z <= self.surface_z_m:
                embedded += segment_length
            if leading_outside and midpoint_z < bottom_z:
                exposed += segment_length
            else:
                leading_outside = False
            if not crossing_found and first[2] < bottom_z <= second[2]:
                span = second[2] - first[2]
                ratio = (bottom_z - first[2]) / span if abs(span) > 1.0e-12 else 0.0
                exit_position = tuple(
                    first[axis] + ratio * (second[axis] - first[axis])
                    for axis in range(3)
                )
                crossing_found = True
        tip_outside = points[0][2] < bottom_z
        if tip_outside and not state.tip_was_outside:
            state.exit_event_count += 1
        state.tip_was_outside = tip_outside
        state.embedded_arc_length_m = embedded
        state.exposed_arc_length_m = exposed
        state.exit_position_m = exit_position

    def step(
        self,
        tip_poses: Sequence[NeedlePose],
        arc_poses: Sequence[NeedlePose],
        punctured: Sequence[bool],
        *,
        dt_s: float = 0.02,
    ) -> tuple[CouplingWrench, ...]:
        response = super().step(tip_poses, arc_poses, punctured, dt_s=dt_s)
        for index, (pose, is_punctured) in enumerate(
            zip(arc_poses, punctured, strict=True)
        ):
            self._update_through_geometry(index, pose, bool(is_punctured))
        return response


def create_tissue_through_backend(
    num_scenes: int, *, integration_step_s: float = 0.002
) -> DrAnmarNativeTissueThroughBackend:
    return DrAnmarNativeTissueThroughBackend(
        num_scenes, integration_step_s=integration_step_s
    )
