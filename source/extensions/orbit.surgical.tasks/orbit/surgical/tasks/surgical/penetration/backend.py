# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic Dr.Anmar tissue-entry mechanics.

This module is deliberately dependency-free.  It implements the entry-only
mechanics required by the policy: a deformable pre-puncture surface response,
a one-way tip-to-arc representation switch commanded by the environment, and
bounded post-puncture shaft/arc resistance.  It does not model a persistent
tract, cutting, thread passage, exit, or topology change.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


DRANMAR_NATIVE_ENTRY_REVISION = "dranmar-native-tissue-entry-v1"


@dataclass(frozen=True)
class NeedlePose:
    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class CouplingWrench:
    force_n: tuple[float, float, float]
    torque_nm: tuple[float, float, float]


@dataclass(frozen=True)
class TissueEntryBackendMetadata:
    provider: str
    revision: str
    implementation_sha256: str
    integration_step_s: float
    clinical_validation: bool = False


@dataclass(frozen=True)
class TissueEntrySceneState:
    representation: str
    representation_switch_count: int
    surface_displacement_m: float
    local_strain: float
    lateral_displacement_m: tuple[float, float]
    contact_position_m: tuple[float, float, float]
    finite: bool


@dataclass
class _SceneState:
    punctured: bool = False
    switch_count: int = 0
    surface_displacement_m: float = 0.0
    anchor_xy: tuple[float, float] | None = None
    lateral_displacement_m: tuple[float, float] = (0.0, 0.0)
    contact_position_m: tuple[float, float, float] = (0.0, 0.0, 0.012)
    finite: bool = True


class DrAnmarTissueEntryBackend(Protocol):
    """Stable ABI between policy code and Dr.Anmar tissue mechanics."""

    @property
    def num_scenes(self) -> int: ...

    @property
    def metadata(self) -> TissueEntryBackendMetadata: ...

    @property
    def scene_state(self) -> tuple[TissueEntrySceneState, ...]: ...

    def step(
        self,
        tip_poses: Sequence[NeedlePose],
        arc_poses: Sequence[NeedlePose],
        punctured: Sequence[bool],
        *,
        dt_s: float = 0.02,
    ) -> tuple[CouplingWrench, ...]: ...

    def close(self) -> None: ...


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class DrAnmarNativeTissueEntryBackend:
    """Small batched viscoelastic entry model for the v1 policy.

    The coupon surface is 3 mm above the backend origin.  Before puncture, a
    Kelvin-Voigt foundation plus bounded nonlinear compression blocks the tip
    and records local surface displacement.  After the environment emits its
    force-gated puncture event, the model switches exactly once to bounded arc
    and shaft resistance.  The environment remains the sole event authority.
    """

    surface_z_m = 0.003
    tissue_thickness_m = 0.006
    relaxation_time_s = 0.008
    compression_stiffness_n_m = 18.0
    compression_cubic_n_m3 = 1.5e6
    normal_damping_n_s_m = 0.04
    lateral_stiffness_n_m = 12.0
    lateral_damping_n_s_m = 0.02
    postpuncture_bias_n = 0.012
    postpuncture_stiffness_n_m = 8.0
    curvature_radius_m = 0.010504226244065092
    arc_sample_count = 17

    def __init__(self, num_scenes: int, *, integration_step_s: float = 0.002) -> None:
        if not 1 <= int(num_scenes) <= 12:
            raise ValueError("entry policy supports 1..12 native tissue scenes")
        if integration_step_s <= 0.0 or not math.isfinite(integration_step_s):
            raise ValueError("integration step must be finite and positive")
        self.integration_step_s = float(integration_step_s)
        self._scenes = [_SceneState() for _ in range(int(num_scenes))]
        self._closed = False
        self._metadata = TissueEntryBackendMetadata(
            provider="dranmar_native_entry",
            revision=DRANMAR_NATIVE_ENTRY_REVISION,
            implementation_sha256=_source_sha256(),
            integration_step_s=self.integration_step_s,
        )

    @property
    def num_scenes(self) -> int:
        return len(self._scenes)

    @property
    def metadata(self) -> TissueEntryBackendMetadata:
        return self._metadata

    @property
    def scene_state(self) -> tuple[TissueEntrySceneState, ...]:
        return tuple(
            TissueEntrySceneState(
                representation="arc" if scene.punctured else "tip",
                representation_switch_count=scene.switch_count,
                surface_displacement_m=scene.surface_displacement_m,
                local_strain=scene.surface_displacement_m / self.tissue_thickness_m,
                lateral_displacement_m=scene.lateral_displacement_m,
                contact_position_m=scene.contact_position_m,
                finite=scene.finite,
            )
            for scene in self._scenes
        )

    @staticmethod
    def _finite_pose(pose: NeedlePose) -> bool:
        return all(
            math.isfinite(value)
            for value in (
                *pose.position,
                *pose.quaternion_xyzw,
                *pose.linear_velocity,
                *pose.angular_velocity,
            )
        )

    @staticmethod
    def _cross(
        left: tuple[float, float, float], right: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        return (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        )

    @classmethod
    def _rotate_xyzw(
        cls,
        quaternion: tuple[float, float, float, float],
        vector: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        qx, qy, qz, qw = quaternion
        qv = (qx, qy, qz)
        cross = cls._cross(qv, vector)
        nested = cls._cross(qv, cross)
        return tuple(
            vector[index] + 2.0 * (qw * cross[index] + nested[index])
            for index in range(3)
        )

    @classmethod
    def _deepest_arc_contact(
        cls, pose: NeedlePose
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        """Return the deepest sampled half-circle point, velocity, and lever arm."""

        deepest: tuple[float, float, float] | None = None
        deepest_offset = (0.0, 0.0, 0.0)
        for index in range(cls.arc_sample_count):
            angle = -0.5 * math.pi + math.pi * index / (cls.arc_sample_count - 1)
            local = (
                cls.curvature_radius_m * math.cos(angle),
                cls.curvature_radius_m * math.sin(angle),
                0.0,
            )
            offset = cls._rotate_xyzw(pose.quaternion_xyzw, local)
            point = tuple(pose.position[axis] + offset[axis] for axis in range(3))
            if deepest is None or point[2] < deepest[2]:
                deepest = point
                deepest_offset = offset
        assert deepest is not None
        rotational_velocity = cls._cross(pose.angular_velocity, deepest_offset)
        velocity = tuple(
            pose.linear_velocity[index] + rotational_velocity[index] for index in range(3)
        )
        return deepest, velocity, deepest_offset

    def step(
        self,
        tip_poses: Sequence[NeedlePose],
        arc_poses: Sequence[NeedlePose],
        punctured: Sequence[bool],
        *,
        dt_s: float = 0.02,
    ) -> tuple[CouplingWrench, ...]:
        if self._closed:
            raise RuntimeError("native tissue-entry backend is closed")
        if len(tip_poses) != self.num_scenes or len(arc_poses) != self.num_scenes:
            raise ValueError("pose count must match native tissue scene count")
        if len(punctured) != self.num_scenes:
            raise ValueError("puncture-state count must match native tissue scene count")
        if dt_s <= 0.0 or not math.isfinite(dt_s):
            raise ValueError("advance duration must be finite and positive")

        response: list[CouplingWrench] = []
        alpha = 1.0 - math.exp(-float(dt_s) / self.relaxation_time_s)
        for scene, tip, arc, requested_puncture in zip(
            self._scenes, tip_poses, arc_poses, punctured, strict=True
        ):
            finite = self._finite_pose(tip) and self._finite_pose(arc)
            scene.finite = finite
            if not finite:
                response.append(
                    CouplingWrench((math.nan, math.nan, math.nan), (math.nan, math.nan, math.nan))
                )
                continue

            requested = bool(requested_puncture)
            if requested and not scene.punctured:
                scene.punctured = True
                scene.switch_count += 1
            elif not requested and scene.punctured:
                # A false state after puncture denotes an environment reset,
                # not a reverse representation transition.
                scene.punctured = False
                scene.switch_count = 0
                scene.surface_displacement_m = 0.0
                scene.anchor_xy = None
                scene.lateral_displacement_m = (0.0, 0.0)
                scene.contact_position_m = (0.0, 0.0, 0.012)

            if scene.punctured:
                contact_position, contact_velocity, lever_arm = self._deepest_arc_contact(arc)
            else:
                contact_position = tip.position
                contact_velocity = tip.linear_velocity
                lever_arm = (0.0, 0.0, 0.0)
            indentation = max(0.0, self.surface_z_m - contact_position[2])
            scene.contact_position_m = contact_position
            target_displacement = min(indentation, self.tissue_thickness_m)
            scene.surface_displacement_m += alpha * (
                target_displacement - scene.surface_displacement_m
            )
            if indentation <= 0.0:
                scene.anchor_xy = None
                scene.lateral_displacement_m = (0.0, 0.0)
                response.append(CouplingWrench((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
                continue

            if scene.anchor_xy is None:
                scene.anchor_xy = contact_position[:2]
            dx = contact_position[0] - scene.anchor_xy[0]
            dy = contact_position[1] - scene.anchor_xy[1]
            scene.lateral_displacement_m = (dx, dy)
            fx = -self.lateral_stiffness_n_m * dx - self.lateral_damping_n_s_m * contact_velocity[0]
            fy = -self.lateral_stiffness_n_m * dy - self.lateral_damping_n_s_m * contact_velocity[1]
            downward_speed = max(0.0, -contact_velocity[2])
            if scene.punctured:
                fz = (
                    self.postpuncture_bias_n
                    + self.postpuncture_stiffness_n_m * indentation
                    + self.normal_damping_n_s_m * downward_speed
                )
            else:
                d = scene.surface_displacement_m
                fz = (
                    self.compression_stiffness_n_m * d
                    + self.compression_cubic_n_m3 * d * d * d
                    + self.normal_damping_n_s_m * downward_speed
                )
            force = (fx, fy, fz)
            torque = self._cross(lever_arm, force) if scene.punctured else (0.0, 0.0, 0.0)
            response.append(CouplingWrench(force, torque))
        return tuple(response)

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "DrAnmarNativeTissueEntryBackend":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def create_tissue_entry_backend(
    num_scenes: int,
    *,
    integration_step_s: float = 0.002,
) -> DrAnmarTissueEntryBackend:
    return DrAnmarNativeTissueEntryBackend(
        num_scenes,
        integration_step_s=integration_step_s,
    )
