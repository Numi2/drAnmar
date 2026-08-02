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


DRANMAR_NATIVE_THROUGH_REVISION = "dranmar-native-tissue-through-v3-cross-slab"

# Tissue coordinates are relative to the midpoint between the two authored
# slabs.  Keep these bounds synchronized with PenetrationSceneCfg: the 3 mm
# wound gap is deliberately not tissue and therefore cannot own an event.
LEFT_SLAB_X_BOUNDS_M = (-0.035, -0.002)
RIGHT_SLAB_X_BOUNDS_M = (0.002, 0.035)


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
    entry_slab: str
    exit_slab: str
    cross_slab_route_valid: bool
    invalid_exit_route: bool
    finite: bool


@dataclass
class _ThroughState:
    exit_event_count: int = 0
    tip_has_entered: bool = False
    previous_tip_z_m: float | None = None
    embedded_arc_length_m: float = 0.0
    exposed_arc_length_m: float = 0.0
    exit_position_m: tuple[float, float, float] = (0.0, 0.0, -0.003)
    entry_slab: str = "none"
    exit_slab: str = "none"
    invalid_exit_route: bool = False


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
                entry_slab=through.entry_slab,
                exit_slab=through.exit_slab,
                cross_slab_route_valid=(
                    through.entry_slab == "left" and through.exit_slab == "right"
                ),
                invalid_exit_route=through.invalid_exit_route,
                finite=entry.finite,
            )
            for entry, through in zip(entry_states, self._through, strict=True)
        )

    @classmethod
    def _arc_points(cls, pose: NeedlePose) -> list[tuple[float, float, float]]:
        points: list[tuple[float, float, float]] = []
        for index in range(cls.through_sample_count):
            # The authored sharp tip is at -pi/2 and points along local -X.
            # Traverse the rendered semicircle from the tip toward the swage.
            angle = -0.5 * math.pi + math.pi * index / (cls.through_sample_count - 1)
            local = (
                cls.curvature_radius_m * math.cos(angle),
                cls.curvature_radius_m * math.sin(angle),
                0.0,
            )
            offset = cls._rotate_xyzw(pose.quaternion_xyzw, local)
            points.append(tuple(pose.position[axis] + offset[axis] for axis in range(3)))
        return points

    def _update_through_geometry(
        self,
        index: int,
        tip_pose: NeedlePose,
        arc_pose: NeedlePose,
        punctured: bool,
    ) -> None:
        state = self._through[index]
        if not punctured:
            state.exit_event_count = 0
            state.tip_has_entered = False
            state.previous_tip_z_m = None
            state.embedded_arc_length_m = 0.0
            state.exposed_arc_length_m = 0.0
            state.exit_position_m = (0.0, 0.0, -0.003)
            state.entry_slab = "none"
            state.exit_slab = "none"
            state.invalid_exit_route = False
            return
        if state.entry_slab == "none":
            state.entry_slab = self._classify_slab(tip_pose.position[0])
        points = self._arc_points(arc_pose)
        bottom_z = self.surface_z_m - self.tissue_thickness_m
        segment_length = math.pi * self.curvature_radius_m / (len(points) - 1)
        embedded = 0.0
        exposed = 0.0
        leading_exposed = True
        exit_position = state.exit_position_m
        crossing_found = False
        for first, second in zip(points[:-1], points[1:], strict=True):
            midpoint_z = 0.5 * (first[2] + second[2])
            if bottom_z <= midpoint_z <= self.surface_z_m:
                embedded += segment_length
            # Exposure is the contiguous leading arc above the operative top
            # after re-emergence.  Arc elsewhere above the slab is not a free
            # graspable leading section.
            if leading_exposed and midpoint_z > self.surface_z_m:
                exposed += segment_length
            else:
                leading_exposed = False
            if not crossing_found and first[2] >= self.surface_z_m > second[2]:
                span = second[2] - first[2]
                ratio = (
                    (self.surface_z_m - first[2]) / span
                    if abs(span) > 1.0e-12
                    else 0.0
                )
                exit_position = tuple(
                    first[axis] + ratio * (second[axis] - first[axis])
                    for axis in range(3)
                )
                crossing_found = True
        tip_z = points[0][2]
        if tip_z < self.surface_z_m - 1.0e-5:
            state.tip_has_entered = True
        tip_reemerged = state.tip_has_entered and tip_z >= self.surface_z_m
        if tip_reemerged and state.exit_event_count == 0:
            state.exit_slab = self._classify_slab(exit_position[0])
            valid_route = state.entry_slab == "left" and state.exit_slab == "right"
            if valid_route:
                state.exit_event_count += 1
            else:
                state.invalid_exit_route = True
        state.previous_tip_z_m = tip_z
        state.embedded_arc_length_m = embedded
        state.exposed_arc_length_m = exposed
        state.exit_position_m = exit_position

    @staticmethod
    def _classify_slab(x_m: float) -> str:
        if LEFT_SLAB_X_BOUNDS_M[0] <= x_m <= LEFT_SLAB_X_BOUNDS_M[1]:
            return "left"
        if RIGHT_SLAB_X_BOUNDS_M[0] <= x_m <= RIGHT_SLAB_X_BOUNDS_M[1]:
            return "right"
        if LEFT_SLAB_X_BOUNDS_M[1] < x_m < RIGHT_SLAB_X_BOUNDS_M[0]:
            return "wound_gap"
        return "outside"

    def step(
        self,
        tip_poses: Sequence[NeedlePose],
        arc_poses: Sequence[NeedlePose],
        punctured: Sequence[bool],
        *,
        dt_s: float = 0.02,
    ) -> tuple[CouplingWrench, ...]:
        response = super().step(tip_poses, arc_poses, punctured, dt_s=dt_s)
        for index, (tip_pose, arc_pose, is_punctured) in enumerate(
            zip(tip_poses, arc_poses, punctured, strict=True)
        ):
            self._update_through_geometry(
                index, tip_pose, arc_pose, bool(is_punctured)
            )
        return response


def create_tissue_through_backend(
    num_scenes: int, *, integration_step_s: float = 0.002
) -> DrAnmarNativeTissueThroughBackend:
    return DrAnmarNativeTissueThroughBackend(
        num_scenes, integration_step_s=integration_step_s
    )
