# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Small deterministic mechanics used by the Dr.Anmar OpenUSD workstation.

The models in this module are deliberately simulator-agnostic.  OpenUSD owns
the rendered geometry while these classes own the mutable surface topology,
elastic displacement, suture constraints, and research telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _point_segment_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    length_squared = float(np.dot(segment, segment))
    if length_squared < 1e-12:
        return np.linalg.norm(points - start[None, :], axis=1)
    projection = np.clip(((points - start[None, :]) @ segment) / length_squared, 0.0, 1.0)
    closest = start[None, :] + projection[:, None] * segment[None, :]
    return np.linalg.norm(points - closest, axis=1)


@dataclass
class SurfaceMeshModel:
    """Mutable surface mesh with bounded deformation and real face removal."""

    original_points: np.ndarray
    face_counts: np.ndarray
    face_indices: np.ndarray
    current_points: np.ndarray = field(init=False)
    active_faces: np.ndarray = field(init=False)
    face_centroids: np.ndarray = field(init=False)
    revision: int = 0
    max_displacement_local: float = 0.0

    def __post_init__(self) -> None:
        self.original_points = np.asarray(self.original_points, dtype=np.float32).reshape(-1, 3).copy()
        self.face_counts = np.asarray(self.face_counts, dtype=np.int32).reshape(-1).copy()
        self.face_indices = np.asarray(self.face_indices, dtype=np.int32).reshape(-1).copy()
        if int(self.face_counts.sum()) != len(self.face_indices):
            raise ValueError("OpenUSD face counts do not match the face-index array")
        self.current_points = self.original_points.copy()
        self.active_faces = np.ones(len(self.face_counts), dtype=np.bool_)
        starts = np.concatenate(([0], np.cumsum(self.face_counts[:-1], dtype=np.int64)))
        gathered = self.original_points[self.face_indices]
        self.face_centroids = np.add.reduceat(gathered, starts, axis=0) / self.face_counts[:, None]

    @property
    def removed_faces(self) -> int:
        return int(len(self.active_faces) - np.count_nonzero(self.active_faces))

    def reset(self) -> None:
        self.current_points[:] = self.original_points
        self.active_faces[:] = True
        self.revision = 0
        self.max_displacement_local = 0.0

    def deform(
        self,
        center_local: np.ndarray,
        displacement_local: np.ndarray,
        radius_local: float,
        max_displacement_local: float,
        compliance: float = 0.72,
    ) -> bool:
        """Apply a smooth, capped displacement around a grasp/contact point."""
        center = np.asarray(center_local, dtype=np.float32)
        displacement = np.asarray(displacement_local, dtype=np.float32)
        if radius_local <= 0.0 or float(np.linalg.norm(displacement)) < 1e-8:
            return False
        distance = np.linalg.norm(self.original_points - center[None, :], axis=1)
        selected = distance < radius_local
        if not np.any(selected):
            # Make low-resolution anatomy respond at its closest vertex too.
            selected[int(np.argmin(distance))] = True
        normalized = np.clip(1.0 - distance[selected] / radius_local, 0.0, 1.0)
        weights = normalized * normalized * (3.0 - 2.0 * normalized)
        self.current_points[selected] += displacement[None, :] * weights[:, None] * compliance
        offsets = self.current_points[selected] - self.original_points[selected]
        magnitudes = np.linalg.norm(offsets, axis=1)
        over_limit = magnitudes > max_displacement_local
        if np.any(over_limit):
            offsets[over_limit] *= (max_displacement_local / magnitudes[over_limit])[:, None]
            self.current_points[selected] = self.original_points[selected] + offsets
        self.max_displacement_local = max(
            self.max_displacement_local,
            float(np.linalg.norm(self.current_points - self.original_points, axis=1).max(initial=0.0)),
        )
        self.revision += 1
        return True

    def recover(self, fraction: float = 0.045) -> bool:
        """Relax displaced vertices toward their undeformed surface."""
        delta = self.original_points - self.current_points
        maximum = float(np.linalg.norm(delta, axis=1).max(initial=0.0))
        if maximum < 1e-6:
            if maximum:
                self.current_points[:] = self.original_points
            return False
        self.current_points += delta * float(np.clip(fraction, 0.0, 1.0))
        self.revision += 1
        return True

    def cut_segment(self, start_local: np.ndarray, end_local: np.ndarray, radius_local: float) -> int:
        """Remove faces intersected by a swept cutting segment."""
        if not np.any(self.active_faces):
            return 0
        distances = _point_segment_distance(
            self.face_centroids,
            np.asarray(start_local, dtype=np.float32),
            np.asarray(end_local, dtype=np.float32),
        )
        remove = self.active_faces & (distances <= radius_local)
        removed = int(np.count_nonzero(remove))
        if removed:
            self.active_faces[remove] = False
            self.revision += 1
        return removed

    def active_topology(self) -> tuple[np.ndarray, np.ndarray]:
        """Return face arrays with cut faces omitted."""
        repeated_faces = np.repeat(np.arange(len(self.face_counts), dtype=np.int32), self.face_counts)
        index_mask = self.active_faces[repeated_faces]
        return self.face_counts[self.active_faces], self.face_indices[index_mask]


@dataclass
class SutureThreadModel:
    """Position-based suture strand with tissue pins and a cinch constraint."""

    node_count: int = 48
    segment_length_m: float = 0.0032
    damping: float = 0.965
    max_tissue_anchors: int = 2
    required_anchors_for_knot: int = 2
    points: np.ndarray = field(init=False)
    previous: np.ndarray = field(init=False)
    fixed: dict[int, np.ndarray] = field(default_factory=dict)
    tissue_anchor_indices: list[int] = field(default_factory=list)
    initialized: bool = False
    tension_n: float = 0.0
    peak_tension_n: float = 0.0
    knot_formed: bool = False
    knot_tightness: float = 0.0
    over_tension_events: int = 0
    over_tension_active: bool = False
    last_anchor_world: np.ndarray | None = None
    anchor_spacings_m: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.points = np.zeros((self.node_count, 3), dtype=np.float32)
        self.previous = self.points.copy()

    def reset(self) -> None:
        self.points.fill(0.0)
        self.previous.fill(0.0)
        self.fixed.clear()
        self.tissue_anchor_indices.clear()
        self.initialized = False
        self.tension_n = 0.0
        self.peak_tension_n = 0.0
        self.knot_formed = False
        self.knot_tightness = 0.0
        self.over_tension_events = 0
        self.over_tension_active = False
        self.last_anchor_world = None
        self.anchor_spacings_m.clear()

    def initialize(self, needle_world: np.ndarray) -> None:
        needle = np.asarray(needle_world, dtype=np.float32)
        tail = needle + np.asarray((0.0, -self.segment_length_m * (self.node_count - 1) * 0.78, 0.018), dtype=np.float32)
        alpha = np.linspace(0.0, 1.0, self.node_count, dtype=np.float32)[:, None]
        self.points[:] = tail[None, :] * (1.0 - alpha) + needle[None, :] * alpha
        self.previous[:] = self.points
        self.fixed = {0: tail.copy(), self.node_count - 1: needle.copy()}
        self.initialized = True

    def add_tissue_anchor(self, world_position: np.ndarray) -> bool:
        if not self.initialized or len(self.tissue_anchor_indices) >= self.max_tissue_anchors:
            return False
        fractions = np.linspace(0.16, 0.82, max(2, self.max_tissue_anchors), dtype=np.float32)
        fraction = float(fractions[len(self.tissue_anchor_indices)])
        index = int(round((self.node_count - 1) * fraction))
        if index in self.fixed:
            return False
        position = np.asarray(world_position, dtype=np.float32).copy()
        if self.last_anchor_world is not None:
            self.anchor_spacings_m.append(float(np.linalg.norm(position - self.last_anchor_world)))
        self.last_anchor_world = position.copy()
        self.fixed[index] = position
        self.points[index] = position
        self.previous[index] = position
        self.tissue_anchor_indices.append(index)
        return True

    def _apply_constraints(self) -> float:
        fixed_indices = sorted(self.fixed)
        raw_stretch = 0.0
        for left, right in zip(fixed_indices[:-1], fixed_indices[1:]):
            available = (right - left) * self.segment_length_m
            routed_span = float(np.linalg.norm(self.fixed[right] - self.fixed[left]))
            raw_stretch = max(raw_stretch, max(0.0, routed_span / max(available, 1e-8) - 1.0))
        for _ in range(7):
            for index in range(self.node_count - 1):
                delta = self.points[index + 1] - self.points[index]
                length = float(np.linalg.norm(delta))
                if length < 1e-8:
                    continue
                correction = delta * ((length - self.segment_length_m) / length)
                left_fixed = index in self.fixed
                right_fixed = index + 1 in self.fixed
                if left_fixed and not right_fixed:
                    self.points[index + 1] -= correction
                elif right_fixed and not left_fixed:
                    self.points[index] += correction
                elif not left_fixed and not right_fixed:
                    self.points[index] += correction * 0.5
                    self.points[index + 1] -= correction * 0.5
            if self.knot_formed:
                near = max(2, self.node_count // 4)
                far = min(self.node_count - 3, (self.node_count * 3) // 4)
                delta = self.points[far] - self.points[near]
                length = float(np.linalg.norm(delta))
                target = 0.0022
                if length > target:
                    correction = delta * ((length - target) / max(length, 1e-8)) * 0.5
                    if near not in self.fixed:
                        self.points[near] += correction
                    if far not in self.fixed:
                        self.points[far] -= correction
            for index, position in self.fixed.items():
                self.points[index] = position
        return raw_stretch

    def update(self, needle_world: np.ndarray, dt: float) -> None:
        needle = np.asarray(needle_world, dtype=np.float32)
        if not self.initialized:
            self.initialize(needle)
        self.fixed[self.node_count - 1] = needle.copy()
        dt = float(np.clip(dt, 1.0 / 240.0, 1.0 / 15.0))
        velocity = (self.points - self.previous) * self.damping
        self.previous[:] = self.points
        self.points += velocity + np.asarray((0.0, 0.0, -1.4), dtype=np.float32)[None, :] * (dt * dt)
        for index, position in self.fixed.items():
            self.points[index] = position
        stretch = self._apply_constraints()
        self.tension_n = float(np.clip(stretch * 1.8, 0.0, 6.0))
        self.peak_tension_n = max(self.peak_tension_n, self.tension_n)
        over_tension = self.tension_n > 1.5
        if over_tension and not self.over_tension_active:
            self.over_tension_events += 1
        self.over_tension_active = over_tension
        if len(self.tissue_anchor_indices) >= self.required_anchors_for_knot and not self.knot_formed:
            entry = self.fixed[self.tissue_anchor_indices[0]]
            loop_closed = float(np.linalg.norm(needle - entry)) <= 0.014
            routed_length = float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())
            if loop_closed and routed_length >= self.segment_length_m * (self.node_count - 1) * 0.82:
                self.knot_formed = True
        if self.knot_formed:
            self.knot_tightness = float(np.clip(max(self.knot_tightness, self.tension_n / 0.8), 0.0, 1.0))

    @property
    def stitch_count(self) -> int:
        return len(self.tissue_anchor_indices) // 2

    @property
    def mean_anchor_spacing_m(self) -> float:
        return float(np.mean(self.anchor_spacings_m)) if self.anchor_spacings_m else 0.0

    @property
    def spacing_variation_m(self) -> float:
        return float(np.std(self.anchor_spacings_m)) if len(self.anchor_spacings_m) > 1 else 0.0
