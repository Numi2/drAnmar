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


DRANMAR_NATIVE_THROUGH_REVISION = "dranmar-native-tissue-through-v12-fem-flap-route"

# Tissue coordinates are relative to the midpoint between the two authored
# FEM flaps.  These are the authored TetMesh extents, not a legacy collision
# proxy.  The 3.127 mm open incision is not tissue and cannot own an entry or
# exit event; it is also too narrow to be treated as instrument access.
LEFT_SLAB_X_BOUNDS_M = (-0.035, -0.00155)
RIGHT_SLAB_X_BOUNDS_M = (0.00157713832065, 0.035)


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
    trailing_exposed_arc_length_m: float
    trailing_grasp_position_m: tuple[float, float, float]
    trailing_grasp_over_wound_gap: bool
    tract_support_active: bool
    tract_support_event_count: int
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
    trailing_exposed_arc_length_m: float = 0.0
    trailing_grasp_position_m: tuple[float, float, float] = (0.0, 0.0, 0.003)
    trailing_grasp_over_wound_gap: bool = False
    tract_support_active: bool = False
    tract_support_event_count: int = 0
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
                trailing_exposed_arc_length_m=through.trailing_exposed_arc_length_m,
                trailing_grasp_position_m=through.trailing_grasp_position_m,
                trailing_grasp_over_wound_gap=through.trailing_grasp_over_wound_gap,
                tract_support_active=through.tract_support_active,
                tract_support_event_count=through.tract_support_event_count,
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
            state.trailing_exposed_arc_length_m = 0.0
            state.trailing_grasp_position_m = (0.0, 0.0, 0.003)
            state.trailing_grasp_over_wound_gap = False
            state.tract_support_active = False
            state.tract_support_event_count = 0
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
        trailing_exposed = 0.0
        leading_exposed = True
        candidate_exit_position = state.exit_position_m
        crossing_found = False
        for first, second in zip(points[:-1], points[1:], strict=True):
            midpoint_x = 0.5 * (first[0] + second[0])
            midpoint_z = 0.5 * (first[2] + second[2])
            midpoint_slab = self._classify_slab(midpoint_x)
            if (
                midpoint_slab in {"left", "right"}
                and bottom_z <= midpoint_z <= self.surface_z_m
            ):
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
                candidate_exit_position = tuple(
                    first[axis] + ratio * (second[axis] - first[axis])
                    for axis in range(3)
                )
                crossing_found = True
        trailing_segments: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
        for first, second in reversed(list(zip(points[:-1], points[1:], strict=True))):
            midpoint_x = 0.5 * (first[0] + second[0])
            midpoint_z = 0.5 * (first[2] + second[2])
            midpoint_slab = self._classify_slab(midpoint_x)
            segment_is_in_tissue = (
                midpoint_slab in {"left", "right"}
                and bottom_z <= midpoint_z <= self.surface_z_m
            )
            if segment_is_in_tissue:
                break
            trailing_exposed += segment_length
            trailing_segments.append((first, second))
        # Classify the incision for route accounting only.  The authored FEM
        # gap is narrower than the distal jaws, so it must never be selected as
        # a regrasp access path.  Regrasp only the contiguous trailing arc on
        # the operative side, exactly as a needle driver would above tissue.
        final_drive_regrasp = state.tract_support_event_count >= 3
        candidates = trailing_segments
        if candidates:
            if final_drive_regrasp:
                # The final giver grasp must stay above the tissue so the jaw
                # can sweep the remaining arc without entering the right slab.
                first, second = max(
                    candidates,
                    key=lambda segment: 0.5 * (segment[0][2] + segment[1][2]),
                )
            else:
                divisor = 2 + min(state.tract_support_event_count, 5)
                first, second = candidates[len(candidates) // divisor]
            state.trailing_grasp_position_m = tuple(
                0.5 * (first[axis] + second[axis]) for axis in range(3)
            )
            state.trailing_grasp_over_wound_gap = False
        tip_z = points[0][2]
        if tip_z < self.surface_z_m - 1.0e-5:
            state.tip_has_entered = True
        tip_reemerged = state.tip_has_entered and tip_z >= self.surface_z_m
        if tip_reemerged and state.exit_event_count == 0:
            state.exit_slab = self._classify_slab(candidate_exit_position[0])
            valid_route = state.entry_slab == "left" and state.exit_slab == "right"
            if valid_route:
                state.exit_event_count += 1
                # Exit is an event coordinate, not a live intersection that
                # follows the needle after it has emerged and is presented or
                # pulled away.  Freeze it at the first authorized top crossing.
                state.exit_position_m = candidate_exit_position
            else:
                state.invalid_exit_route = True
        state.previous_tip_z_m = tip_z
        state.embedded_arc_length_m = embedded
        state.exposed_arc_length_m = exposed
        state.trailing_exposed_arc_length_m = trailing_exposed

    def request_tract_support(self, scene_ids: Sequence[int]) -> tuple[bool, ...]:
        """Authorize at most four tissue-held regrips of a genuinely embedded arc."""

        results: list[bool] = []
        for index in scene_ids:
            entry = self._scenes[index]
            state = self._through[index]
            eligible = (
                entry.punctured
                and entry.switch_count == 1
                and state.exit_event_count == 0
                and not state.invalid_exit_route
                # The instrument-width incision leaves shorter tissue
                # purchase on the 7 mm-radius needle.
                and state.embedded_arc_length_m >= 0.0015
                and state.trailing_exposed_arc_length_m >= 0.002
                and state.trailing_grasp_position_m[2]
                >= self.surface_z_m + 0.0015
                and state.tract_support_event_count < 4
            )
            if eligible:
                state.tract_support_active = True
                state.tract_support_event_count += 1
            results.append(eligible)
        return tuple(results)

    def release_tract_support(self, scene_ids: Sequence[int]) -> None:
        for index in scene_ids:
            self._through[index].tract_support_active = False

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
