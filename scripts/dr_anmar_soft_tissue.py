# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic surgical interaction mechanics for the Dr.Anmar workstation.

Isaac Lab remains the robot, rigid-contact, and sensor authority.  These models
provide a bounded reduced-order tissue volume, needle puncture/friction,
topology-changing cutting, and a position-based suture strand when a room does
not ship a validated native deformable asset.  Parameters are research defaults
and must be calibrated against the intended bench material before comparison to
clinical force or failure thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _point_segment_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    length_squared = float(np.dot(segment, segment))
    if length_squared < 1e-12:
        return np.linalg.norm(points - start[None, :], axis=1)
    projection = np.clip(((points - start[None, :]) @ segment) / length_squared, 0.0, 1.0)
    closest = start[None, :] + projection[:, None] * segment[None, :]
    return np.linalg.norm(points - closest, axis=1)


def _safe_unit(vector: np.ndarray, fallback: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    length = float(np.linalg.norm(value))
    if length < 1e-9:
        return np.asarray(fallback, dtype=np.float32)
    return value / length


@dataclass(frozen=True)
class TissueMaterial:
    """Research-grade constitutive parameters expressed in SI units."""

    id: str
    youngs_modulus_pa: float
    poisson_ratio: float
    damping_ratio: float
    static_friction: float
    dynamic_friction: float
    puncture_force_n: float
    puncture_hysteresis_n: float
    needle_drag_n: float
    recovery_half_life_s: float
    yield_strain: float
    tear_strain: float
    cut_toughness_n: float
    anchor_pullout_force_n: float
    safe_force_n: float

    @property
    def safe_torque_nm(self) -> float:
        """Conservative tool-tip torque envelope derived from a 30 mm lever arm."""
        return self.safe_force_n * 0.03


TISSUE_MATERIALS: dict[str, TissueMaterial] = {
    "liver": TissueMaterial(
        "liver_research_default", 18_000.0, 0.47, 0.24, 0.34, 0.22, 0.55, 0.16, 0.18, 0.62, 0.32, 0.58, 0.42, 1.25, 2.0
    ),
    "gallbladder": TissueMaterial(
        "gallbladder_research_default", 38_000.0, 0.46, 0.20, 0.30, 0.18, 0.42, 0.12, 0.13, 0.48, 0.22, 0.42, 0.28, 0.85, 1.4
    ),
    "bladder": TissueMaterial(
        "bladder_research_default", 52_000.0, 0.48, 0.26, 0.32, 0.20, 0.62, 0.18, 0.20, 0.78, 0.38, 0.66, 0.48, 1.45, 2.2
    ),
    "vessel": TissueMaterial(
        "vessel_research_default", 120_000.0, 0.47, 0.18, 0.28, 0.16, 0.72, 0.20, 0.22, 0.44, 0.24, 0.48, 0.36, 0.72, 1.25
    ),
    "generic": TissueMaterial(
        "soft_tissue_research_default", 28_000.0, 0.47, 0.23, 0.32, 0.20, 0.58, 0.16, 0.18, 0.64, 0.32, 0.58, 0.42, 1.10, 2.0
    ),
}


def tissue_material_for_name(name: str) -> TissueMaterial:
    lowered = str(name).lower()
    for key in ("gallbladder", "bladder", "vessel", "liver"):
        if key in lowered:
            return TISSUE_MATERIALS[key]
    return TISSUE_MATERIALS["generic"]


@dataclass
class SurfaceSutureAnchor:
    """A suture bite bound to a weighted neighborhood of tissue vertices."""

    anchor_id: int
    kind: str
    vertex_indices: np.ndarray
    weights: np.ndarray
    rest_point_local: np.ndarray
    normal_local: np.ndarray
    bite_depth_m: float = 0.0
    active: bool = True

    def position(self, points: np.ndarray) -> np.ndarray:
        return np.sum(points[self.vertex_indices] * self.weights[:, None], axis=0).astype(np.float32)


@dataclass
class StitchConstraintState:
    """Persistent state for one entry/exit tissue bite pair."""

    anchor_pair: tuple[int, int]
    initial_gap_local: float
    current_gap_local: float
    closure_ratio: float = 0.0
    retained_closure: float = 0.0
    damage: float = 0.0
    failed: bool = False


@dataclass
class SurfaceMeshModel:
    """Reduced-order volume-preserving tissue with mutable OpenUSD topology.

    This is deliberately a fallback model: local contact displacement is coupled
    over mesh edges, a closed surface retains volume, attachment regions resist
    gross translation, and cut faces are both removed and opened.  It is not a
    claim of patient-specific finite-element validation.
    """

    original_points: np.ndarray
    face_counts: np.ndarray
    face_indices: np.ndarray
    material: TissueMaterial = TISSUE_MATERIALS["generic"]
    current_points: np.ndarray = field(init=False)
    active_faces: np.ndarray = field(init=False)
    face_centroids: np.ndarray = field(init=False)
    triangles: np.ndarray = field(init=False)
    edges: np.ndarray = field(init=False)
    rest_edge_lengths: np.ndarray = field(init=False)
    attachment_weights: np.ndarray = field(init=False)
    rest_volume_local: float = field(init=False, default=0.0)
    revision: int = 0
    max_displacement_local: float = 0.0
    max_strain: float = 0.0
    strain_energy_proxy_j: float = 0.0
    volume_ratio: float = 1.0
    cut_resistance_n: float = 0.0
    cut_energy_proxy_j: float = 0.0
    opened_faces: int = 0
    tear_events: int = 0
    suture_anchors: dict[int, SurfaceSutureAnchor] = field(default_factory=dict)
    stitch_constraints: dict[tuple[int, int], StitchConstraintState] = field(default_factory=dict)
    stitch_failures: int = 0
    suture_force_n: float = 0.0
    max_suture_force_n: float = 0.0
    closure_gap_local: float = 0.0
    closure_ratio: float = 0.0
    retained_closure: float = 0.0
    puncture_site_count: int = 0
    _tear_active: bool = False

    def __post_init__(self) -> None:
        self.original_points = np.asarray(self.original_points, dtype=np.float32).reshape(-1, 3).copy()
        self.face_counts = np.asarray(self.face_counts, dtype=np.int32).reshape(-1).copy()
        self.face_indices = np.asarray(self.face_indices, dtype=np.int32).reshape(-1).copy()
        if int(self.face_counts.sum()) != len(self.face_indices):
            raise ValueError("OpenUSD face counts do not match the face-index array")
        if len(self.original_points) < 4:
            raise ValueError("A tissue volume needs at least four vertices")
        self.current_points = self.original_points.copy()
        self.active_faces = np.ones(len(self.face_counts), dtype=np.bool_)
        starts = np.concatenate(([0], np.cumsum(self.face_counts[:-1], dtype=np.int64)))
        gathered = self.original_points[self.face_indices]
        self.face_centroids = np.add.reduceat(gathered, starts, axis=0) / self.face_counts[:, None]
        triangles: list[tuple[int, int, int]] = []
        edges: set[tuple[int, int]] = set()
        offset = 0
        for count in self.face_counts:
            polygon = self.face_indices[offset : offset + int(count)]
            offset += int(count)
            if len(polygon) >= 3:
                triangles.extend((int(polygon[0]), int(polygon[index]), int(polygon[index + 1])) for index in range(1, len(polygon) - 1))
            for left, right in zip(polygon, np.roll(polygon, -1)):
                if int(left) != int(right):
                    edges.add(tuple(sorted((int(left), int(right)))))
        self.triangles = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
        self.edges = np.asarray(sorted(edges), dtype=np.int32).reshape(-1, 2)
        self.rest_edge_lengths = np.linalg.norm(
            self.original_points[self.edges[:, 1]] - self.original_points[self.edges[:, 0]], axis=1
        ) if len(self.edges) else np.zeros(0, dtype=np.float32)
        self.rest_volume_local = self._signed_volume(self.original_points)
        extents = np.ptp(self.original_points, axis=0)
        attachment_axis = int(np.argmax(extents))
        coordinate = self.original_points[:, attachment_axis]
        span = max(float(np.ptp(coordinate)), 1e-6)
        # The least mobile 15% of the longest organ axis approximates its hilum/
        # bed attachment without assuming a particular imported coordinate frame.
        normalized = (coordinate - float(coordinate.min())) / span
        self.attachment_weights = np.clip((0.18 - normalized) / 0.18, 0.0, 1.0).astype(np.float32) ** 2

    def _signed_volume(self, points: np.ndarray) -> float:
        if not len(self.triangles):
            return 0.0
        tri = points[self.triangles]
        signed = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0
        return float(signed)

    def _update_mechanics(self) -> None:
        if len(self.edges):
            lengths = np.linalg.norm(
                self.current_points[self.edges[:, 1]] - self.current_points[self.edges[:, 0]], axis=1
            )
            strains = np.abs(lengths - self.rest_edge_lengths) / np.maximum(self.rest_edge_lengths, 1e-8)
            current_max_strain = float(strains.max(initial=0.0))
            self.max_strain = max(self.max_strain, current_max_strain)
            mean_squared_strain = float(np.mean(strains * strains)) if len(strains) else 0.0
            characteristic_volume = max(abs(self.rest_volume_local), 1e-9)
            self.strain_energy_proxy_j = 0.5 * self.material.youngs_modulus_pa * mean_squared_strain * characteristic_volume
            tear_active = current_max_strain >= self.material.tear_strain
            if tear_active and not self._tear_active:
                self.tear_events += 1
            self._tear_active = tear_active
        current_volume = self._signed_volume(self.current_points)
        if abs(self.rest_volume_local) > 1e-9:
            self.volume_ratio = float(abs(current_volume / self.rest_volume_local))

    def _couple_edges(self, selected: np.ndarray, strength: float) -> None:
        if not len(self.edges) or strength <= 0.0:
            return
        offsets = self.current_points - self.original_points
        for _ in range(2):
            accumulated = np.zeros_like(offsets)
            counts = np.zeros((len(offsets), 1), dtype=np.float32)
            left, right = self.edges[:, 0], self.edges[:, 1]
            np.add.at(accumulated, left, offsets[right])
            np.add.at(accumulated, right, offsets[left])
            np.add.at(counts, left, 1.0)
            np.add.at(counts, right, 1.0)
            neighborhood = accumulated / np.maximum(counts, 1.0)
            blend = np.where(selected[:, None], strength, strength * 0.18)
            offsets = offsets * (1.0 - blend) + neighborhood * blend
        offsets *= 1.0 - self.attachment_weights[:, None] * 0.82
        self.current_points[:] = self.original_points + offsets

    def _preserve_volume(self, strength: float = 0.28) -> None:
        if self.removed_faces or abs(self.rest_volume_local) <= 1e-9:
            return
        current = self._signed_volume(self.current_points)
        if abs(current) <= 1e-10:
            return
        ratio = float(np.clip(abs(self.rest_volume_local / current) ** (1.0 / 3.0), 0.94, 1.06))
        centroid = np.average(self.current_points, axis=0, weights=1.0 - 0.7 * self.attachment_weights)
        correction = centroid[None, :] + (self.current_points - centroid[None, :]) * ratio
        movable = (1.0 - self.attachment_weights)[:, None]
        self.current_points += (correction - self.current_points) * movable * float(np.clip(strength, 0.0, 1.0))

    @property
    def removed_faces(self) -> int:
        return int(len(self.active_faces) - np.count_nonzero(self.active_faces))

    def reset(self) -> None:
        self.current_points[:] = self.original_points
        self.active_faces[:] = True
        self.revision = 0
        self.max_displacement_local = 0.0
        self.max_strain = 0.0
        self.strain_energy_proxy_j = 0.0
        self.volume_ratio = 1.0
        self.cut_resistance_n = 0.0
        self.cut_energy_proxy_j = 0.0
        self.opened_faces = 0
        self.tear_events = 0
        self.suture_anchors.clear()
        self.stitch_constraints.clear()
        self.stitch_failures = 0
        self.suture_force_n = 0.0
        self.max_suture_force_n = 0.0
        self.closure_gap_local = 0.0
        self.closure_ratio = 0.0
        self.retained_closure = 0.0
        self.puncture_site_count = 0
        self._tear_active = False

    def bind_suture_anchor(
        self,
        anchor_id: int,
        point_local: np.ndarray,
        kind: str,
        normal_local: np.ndarray | None = None,
        bite_depth_m: float = 0.0,
        radius_local: float = 0.018,
        world_scale: float = 1.0,
    ) -> SurfaceSutureAnchor:
        """Bind a thread node to tissue material coordinates around a puncture site."""
        point = np.asarray(point_local, dtype=np.float32)
        distances = np.linalg.norm(self.current_points - point[None, :], axis=1)
        selected = np.flatnonzero(distances <= max(float(radius_local), 1e-6))
        if len(selected) < 6:
            selected = np.argsort(distances)[: min(12, len(distances))]
        if len(selected) > 28:
            selected = selected[np.argsort(distances[selected])[:28]]
        local_distances = distances[selected]
        sigma = max(float(radius_local) * 0.42, 1e-6)
        weights = np.exp(-0.5 * (local_distances / sigma) ** 2).astype(np.float32)
        weights /= max(float(weights.sum()), 1e-8)
        normal = _safe_unit(
            np.asarray(normal_local, dtype=np.float32)
            if normal_local is not None
            else point - np.mean(self.current_points, axis=0)
        )
        binding = SurfaceSutureAnchor(
            anchor_id=int(anchor_id),
            kind=str(kind),
            vertex_indices=np.asarray(selected, dtype=np.int32),
            weights=weights,
            rest_point_local=point.copy(),
            normal_local=normal,
            bite_depth_m=max(0.0, float(bite_depth_m)),
        )
        self.suture_anchors[int(anchor_id)] = binding
        self.puncture_site_count += 1

        # Leave a small, persistent puncture dimple.  The later stitch
        # constraint pulls the two bound neighborhoods together, while this
        # indentation preserves the visible entry/exit tract.
        scale = max(float(world_scale), 1e-6)
        dimple_local = min(0.0018 / scale, (0.00045 + binding.bite_depth_m * 0.10) / scale)
        mobility = 1.0 - self.attachment_weights[selected]
        self.current_points[selected] -= (
            normal[None, :] * dimple_local * (weights / max(float(weights.max()), 1e-8))[:, None] * mobility[:, None]
        )
        self._preserve_volume(strength=0.12)
        self._update_mechanics()
        self.revision += 1
        return binding

    def release_suture_anchor(self, anchor_id: int) -> None:
        binding = self.suture_anchors.get(int(anchor_id))
        if binding is not None:
            binding.active = False

    def suture_anchor_position(self, anchor_id: int) -> np.ndarray | None:
        binding = self.suture_anchors.get(int(anchor_id))
        if binding is None or not binding.active:
            return None
        return binding.position(self.current_points)

    def apply_suture_constraints(
        self,
        anchor_pairs: list[tuple[int, int]],
        tension_n: float,
        knot_security: float,
        dt_s: float,
        world_scale: float = 1.0,
    ) -> dict[str, Any]:
        """Approximate tissue edges and retain closure while a stitch is secured.

        The solver binds each entry and exit to a material-coordinate vertex
        neighborhood.  Thread tension draws the neighborhoods together; a
        secure knot retains the achieved approximation.  Excess load damages
        the bite and can pull the exit anchor through the tissue.
        """
        valid_pairs = [
            (int(entry), int(exit_))
            for entry, exit_ in anchor_pairs
            if self.suture_anchor_position(entry) is not None and self.suture_anchor_position(exit_) is not None
        ]
        self.suture_force_n = max(0.0, float(tension_n))
        self.max_suture_force_n = max(self.max_suture_force_n, self.suture_force_n)
        failed_anchor_ids: list[int] = []
        changed = False
        gaps: list[float] = []
        closures: list[float] = []
        retained: list[float] = []
        scale = max(float(world_scale), 1e-6)
        dt = float(np.clip(dt_s, 1.0 / 300.0, 0.1))

        for pair in valid_pairs:
            entry_binding = self.suture_anchors[pair[0]]
            exit_binding = self.suture_anchors[pair[1]]
            entry_position = entry_binding.position(self.current_points)
            exit_position = exit_binding.position(self.current_points)
            delta = exit_position - entry_position
            gap_local = float(np.linalg.norm(delta))
            if gap_local < 1e-8:
                continue
            state = self.stitch_constraints.get(pair)
            if state is None:
                state = StitchConstraintState(pair, gap_local, gap_local)
                self.stitch_constraints[pair] = state

            pullout = max(self.material.anchor_pullout_force_n, 1e-6)
            load_ratio = self.suture_force_n / pullout
            load_drive = float(np.clip(load_ratio / 0.72, 0.0, 1.0))
            knot_drive = float(np.clip(knot_security, 0.0, 1.0))
            live_closure = float(np.clip(load_drive * 0.72 + knot_drive * 0.48, 0.0, 1.0))
            state.retained_closure = max(state.retained_closure, live_closure * knot_drive)
            closure_drive = max(live_closure, state.retained_closure)
            minimum_gap_local = 0.0012 / scale
            target_gap_local = max(minimum_gap_local, state.initial_gap_local * (1.0 - 0.84 * closure_drive))
            excess_gap = max(0.0, gap_local - target_gap_local)
            response = 1.0 - np.exp(-dt * (5.0 + 9.0 * closure_drive))
            closure_step = min(excess_gap * 0.5 * response, 0.0012 / scale)

            if closure_step > 1e-8:
                direction = delta / gap_local
                for binding, sign in ((entry_binding, 1.0), (exit_binding, -1.0)):
                    normalized_weights = binding.weights / max(float(binding.weights.max()), 1e-8)
                    mobility = 1.0 - self.attachment_weights[binding.vertex_indices]
                    self.current_points[binding.vertex_indices] += (
                        direction[None, :]
                        * sign
                        * closure_step
                        * normalized_weights[:, None]
                        * mobility[:, None]
                    )
                selected = np.zeros(len(self.current_points), dtype=np.bool_)
                selected[entry_binding.vertex_indices] = True
                selected[exit_binding.vertex_indices] = True
                self._couple_edges(selected, strength=0.045 + 0.035 * closure_drive)
                changed = True

            overload = max(0.0, load_ratio - 0.82)
            state.damage = float(np.clip(state.damage + overload * dt * 0.42, 0.0, 1.2))
            if state.damage >= 1.0 and not state.failed:
                state.failed = True
                failed_anchor = pair[1]
                self.release_suture_anchor(failed_anchor)
                failed_anchor_ids.append(failed_anchor)
                self.stitch_failures += 1
                self.tear_events += 1

            updated_entry = entry_binding.position(self.current_points)
            updated_exit = exit_binding.position(self.current_points)
            state.current_gap_local = float(np.linalg.norm(updated_exit - updated_entry))
            state.closure_ratio = float(
                np.clip(1.0 - state.current_gap_local / max(state.initial_gap_local, 1e-8), 0.0, 1.0)
            )
            gaps.append(state.current_gap_local)
            closures.append(state.closure_ratio)
            retained.append(state.retained_closure)

        if changed:
            max_offset = 0.024 / scale
            offsets = self.current_points - self.original_points
            magnitudes = np.linalg.norm(offsets, axis=1)
            over_limit = magnitudes > max_offset
            if np.any(over_limit):
                offsets[over_limit] *= (max_offset / magnitudes[over_limit])[:, None]
                self.current_points[:] = self.original_points + offsets
            self._preserve_volume(strength=0.15)
            self.max_displacement_local = max(
                self.max_displacement_local,
                float(np.linalg.norm(self.current_points - self.original_points, axis=1).max(initial=0.0)),
            )
            self._update_mechanics()
            self.revision += 1

        self.closure_gap_local = float(np.mean(gaps)) if gaps else 0.0
        self.closure_ratio = float(np.mean(closures)) if closures else 0.0
        self.retained_closure = float(np.mean(retained)) if retained else 0.0
        return {
            "changed": changed,
            "failed_anchor_ids": failed_anchor_ids,
            "closure_gap_m": self.closure_gap_local * scale,
            "closure_ratio": self.closure_ratio,
            "retained_closure": self.retained_closure,
        }

    def deform(
        self,
        center_local: np.ndarray,
        displacement_local: np.ndarray,
        radius_local: float,
        max_displacement_local: float,
        compliance: float = 0.72,
    ) -> bool:
        """Apply material-scaled contact displacement and couple neighboring nodes."""
        center = np.asarray(center_local, dtype=np.float32)
        displacement = np.asarray(displacement_local, dtype=np.float32)
        if radius_local <= 0.0 or float(np.linalg.norm(displacement)) < 1e-8:
            return False
        distance = np.linalg.norm(self.original_points - center[None, :], axis=1)
        selected = distance < radius_local
        if not np.any(selected):
            selected[int(np.argmin(distance))] = True
        normalized = np.clip(1.0 - distance[selected] / radius_local, 0.0, 1.0)
        weights = normalized * normalized * (3.0 - 2.0 * normalized)
        stiffness_scale = float(np.clip(28_000.0 / max(self.material.youngs_modulus_pa, 1.0), 0.32, 1.8))
        effective_compliance = float(np.clip(compliance * stiffness_scale, 0.08, 1.4))
        self.current_points[selected] += displacement[None, :] * weights[:, None] * effective_compliance
        self._couple_edges(selected, strength=0.10 + 0.08 * self.material.damping_ratio)
        offsets = self.current_points - self.original_points
        magnitudes = np.linalg.norm(offsets, axis=1)
        over_limit = magnitudes > max_displacement_local
        if np.any(over_limit):
            offsets[over_limit] *= (max_displacement_local / magnitudes[over_limit])[:, None]
            self.current_points[:] = self.original_points + offsets
        self._preserve_volume()
        self.max_displacement_local = max(self.max_displacement_local, float(magnitudes.max(initial=0.0)))
        self._update_mechanics()
        self.revision += 1
        return True

    def recover(self, fraction: float | None = None, dt_s: float = 0.02) -> bool:
        """Relax elastically while retaining opened incision topology."""
        delta = self.original_points - self.current_points
        maximum = float(np.linalg.norm(delta, axis=1).max(initial=0.0))
        if maximum < 1e-6:
            if maximum:
                self.current_points[:] = self.original_points
            return False
        if fraction is None:
            half_life = max(self.material.recovery_half_life_s, 1e-4)
            fraction = 1.0 - 2.0 ** (-max(float(dt_s), 0.0) / half_life)
        suture_protection = np.zeros(len(self.current_points), dtype=np.float32)
        for anchor in self.suture_anchors.values():
            if not anchor.active:
                continue
            normalized_weights = anchor.weights / max(float(anchor.weights.max()), 1e-8)
            suture_protection[anchor.vertex_indices] = np.maximum(
                suture_protection[anchor.vertex_indices], normalized_weights
            )
        movable = (
            (1.0 - 0.88 * self.attachment_weights)
            * (1.0 - 0.90 * suture_protection)
        )[:, None]
        self.current_points += delta * float(np.clip(fraction, 0.0, 1.0)) * movable
        self._preserve_volume(strength=0.20)
        self._update_mechanics()
        self.revision += 1
        return True

    def cut_segment(self, start_local: np.ndarray, end_local: np.ndarray, radius_local: float) -> int:
        """Separate intersected faces and open the wound around the swept blade."""
        if not np.any(self.active_faces):
            return 0
        start = np.asarray(start_local, dtype=np.float32)
        end = np.asarray(end_local, dtype=np.float32)
        segment = end - start
        segment_length = float(np.linalg.norm(segment))
        distances = _point_segment_distance(self.face_centroids, start, end)
        remove = self.active_faces & (distances <= radius_local)
        removed = int(np.count_nonzero(remove))
        if not removed:
            self.cut_resistance_n = 0.0
            return 0
        affected_indices: list[int] = []
        offset = 0
        for face_index, count in enumerate(self.face_counts):
            count = int(count)
            if remove[face_index]:
                affected_indices.extend(int(value) for value in self.face_indices[offset : offset + count])
            offset += count
        affected = np.unique(np.asarray(affected_indices, dtype=np.int32))
        center = (start + end) * 0.5
        radial = center - np.mean(self.original_points, axis=0)
        opening_axis = _safe_unit(np.cross(_safe_unit(segment), _safe_unit(radial)), (1.0, 0.0, 0.0))
        side = np.sign((self.original_points[affected] - center[None, :]) @ opening_axis)
        side[side == 0.0] = 1.0
        opening = min(max(radius_local * 0.55, 2e-5), 0.012)
        self.current_points[affected] += side[:, None] * opening_axis[None, :] * opening
        self.active_faces[remove] = False
        self.opened_faces += removed
        self.cut_resistance_n = float(np.clip(self.material.cut_toughness_n * (0.35 + removed / 18.0), 0.0, self.material.safe_force_n * 1.5))
        self.cut_energy_proxy_j += self.cut_resistance_n * segment_length
        self._update_mechanics()
        self.revision += 1
        return removed

    def active_topology(self) -> tuple[np.ndarray, np.ndarray]:
        repeated_faces = np.repeat(np.arange(len(self.face_counts), dtype=np.int32), self.face_counts)
        index_mask = self.active_faces[repeated_faces]
        return self.face_counts[self.active_faces], self.face_indices[index_mask]

    def snapshot(self, world_scale: float = 1.0) -> dict[str, Any]:
        current_displacement = float(
            np.linalg.norm(self.current_points - self.original_points, axis=1).max(initial=0.0)
        ) * float(world_scale)
        stress_proxy = self.material.youngs_modulus_pa * self.max_strain
        return {
            "active": True,
            "model": "surface_bound_suture_tissue_v3",
            "authority": "isaac_contacts_plus_openusd_reduced_order_fallback",
            "material_profile": self.material.id,
            "calibration_status": "research_defaults_unvalidated",
            "current_displacement_m": round(current_displacement, 6),
            "max_displacement_m": round(self.max_displacement_local * float(world_scale), 6),
            "max_edge_strain": round(self.max_strain, 5),
            "stress_proxy_pa": round(stress_proxy, 1),
            "strain_energy_proxy_j": round(self.strain_energy_proxy_j, 7),
            "volume_ratio": round(self.volume_ratio, 5),
            "attachment_load_proxy_n": round(stress_proxy * current_displacement * 2e-5, 4),
            "tear_events": self.tear_events,
            "puncture_sites": self.puncture_site_count,
            "bound_suture_anchors": sum(int(anchor.active) for anchor in self.suture_anchors.values()),
            "stitch_constraints": sum(int(not stitch.failed) for stitch in self.stitch_constraints.values()),
            "stitch_failures": self.stitch_failures,
            "suture_force_n": round(self.suture_force_n, 4),
            "peak_suture_force_n": round(self.max_suture_force_n, 4),
            "closure_gap_m": round(self.closure_gap_local * float(world_scale), 6),
            "closure_ratio": round(self.closure_ratio, 4),
            "retained_closure": round(self.retained_closure, 4),
            "surface_revision": self.revision,
        }


@dataclass
class NeedleTissueInteractionModel:
    """Force-gated needle contact, puncture hysteresis, drag, and arc guidance."""

    material: TissueMaterial = TISSUE_MATERIALS["generic"]
    contact_band_m: float = 0.0025
    state: str = "free"
    punctured: bool = False
    entry_direction: np.ndarray | None = None
    penetration_depth_m: float = 0.0
    max_penetration_depth_m: float = 0.0
    interaction_force_n: float = 0.0
    peak_force_n: float = 0.0
    interaction_torque_nm: float = 0.0
    peak_torque_nm: float = 0.0
    rotation_scale: float = 1.0
    puncture_count: int = 0
    exit_count: int = 0
    puncture_work_j: float = 0.0
    curvature_alignment: float = 1.0
    safe_envelope_active: bool = False
    safe_envelope_events: int = 0
    _safe_envelope_was_active: bool = False

    def reset(self) -> None:
        self.state = "free"
        self.punctured = False
        self.entry_direction = None
        self.penetration_depth_m = 0.0
        self.max_penetration_depth_m = 0.0
        self.interaction_force_n = 0.0
        self.peak_force_n = 0.0
        self.interaction_torque_nm = 0.0
        self.peak_torque_nm = 0.0
        self.rotation_scale = 1.0
        self.puncture_count = 0
        self.exit_count = 0
        self.puncture_work_j = 0.0
        self.curvature_alignment = 1.0
        self.safe_envelope_active = False
        self.safe_envelope_events = 0
        self._safe_envelope_was_active = False

    def update(
        self,
        clearance_m: float | None,
        outward: np.ndarray | None,
        translation: np.ndarray,
        rotation: np.ndarray,
        dt_s: float,
        max_depth_m: float,
    ) -> np.ndarray:
        adjusted = np.asarray(translation, dtype=np.float32).copy()
        rotation_effort = float(np.linalg.norm(rotation))
        self.safe_envelope_active = False
        self.rotation_scale = 1.0
        if clearance_m is None or outward is None:
            if self.punctured:
                self.exit_count += 1
            self.state = "free"
            self.punctured = False
            self.entry_direction = None
            self.interaction_force_n = 0.0
            self.interaction_torque_nm = 0.0
            self.penetration_depth_m = 0.0
            self._safe_envelope_was_active = False
            return adjusted
        normal = _safe_unit(outward)
        normal_command = float(np.dot(adjusted, normal))
        inward_command = max(0.0, -normal_command)
        outward_command = max(0.0, normal_command)
        indentation = max(0.0, self.contact_band_m - float(clearance_m))
        command_speed_m_s = inward_command * 0.025
        elastic_force = indentation * self.material.youngs_modulus_pa * 0.010
        damping_force = command_speed_m_s * self.material.damping_ratio * 10.0
        if not self.punctured:
            self.state = "contact" if float(clearance_m) <= self.contact_band_m else "free"
            self.interaction_force_n = elastic_force + damping_force if self.state == "contact" else 0.0
            if (
                self.state == "contact"
                and inward_command > 0.04
                and self.interaction_force_n >= self.material.puncture_force_n
            ):
                self.punctured = True
                self.state = "punctured"
                self.puncture_count += 1
                self.entry_direction = _safe_unit(adjusted, tuple(-normal))
                self.interaction_force_n = max(self.interaction_force_n, self.material.puncture_force_n)
            elif self.state == "contact" and inward_command > 0.0:
                # Before puncture the tip indents the tissue instead of passing
                # through it at full command velocity.
                transmission = float(np.clip(1.0 - self.interaction_force_n / max(self.material.puncture_force_n, 1e-6), 0.08, 0.72))
                adjusted -= normal * normal_command * (1.0 - transmission)
        if self.punctured:
            self.penetration_depth_m = max(0.0, -float(clearance_m))
            self.max_penetration_depth_m = max(self.max_penetration_depth_m, self.penetration_depth_m)
            depth_fraction = float(np.clip(self.penetration_depth_m / max(max_depth_m, 1e-6), 0.0, 1.0))
            required_rotation = 0.10 + 0.42 * depth_fraction
            self.curvature_alignment = float(np.clip(rotation_effort / required_rotation, 0.0, 1.0))
            curvature_penalty = (1.0 - self.curvature_alignment) * 0.55 * depth_fraction
            self.interaction_force_n = (
                self.material.needle_drag_n
                + self.material.dynamic_friction * (self.material.puncture_force_n - self.material.puncture_hysteresis_n)
                + curvature_penalty
            )
            if inward_command > 0.0:
                resistance = float(np.clip(self.interaction_force_n / max(self.material.safe_force_n, 1e-6), 0.0, 0.82))
                adjusted -= normal * normal_command * resistance
            if self.penetration_depth_m >= max_depth_m and inward_command > 0.0:
                adjusted -= normal * float(np.dot(adjusted, normal))
                self.safe_envelope_active = True
            if outward_command > 0.0 and float(clearance_m) > self.contact_band_m + 0.0035:
                self.punctured = False
                self.state = "exited"
                self.exit_count += 1
                self.entry_direction = None
                self.penetration_depth_m = 0.0
        lever_arm_m = 0.008
        if self.punctured:
            lever_arm_m += 0.022 * (1.0 - self.curvature_alignment)
        command_torque_nm = min(rotation_effort, 1.0) * self.material.puncture_force_n * 0.006
        self.interaction_torque_nm = self.interaction_force_n * lever_arm_m + command_torque_nm
        self.safe_envelope_active = self.safe_envelope_active or (
            self.interaction_force_n >= self.material.safe_force_n
            or self.interaction_torque_nm >= self.material.safe_torque_nm
        )
        if self.interaction_force_n >= self.material.safe_force_n:
            adjusted *= float(np.clip(self.material.safe_force_n / max(self.interaction_force_n, 1e-6), 0.18, 1.0))
        if self.interaction_torque_nm >= self.material.safe_torque_nm:
            self.rotation_scale = float(
                np.clip(self.material.safe_torque_nm / max(self.interaction_torque_nm, 1e-9), 0.25, 1.0)
            )
        if self.safe_envelope_active and not self._safe_envelope_was_active:
            self.safe_envelope_events += 1
        self._safe_envelope_was_active = self.safe_envelope_active
        self.peak_force_n = max(self.peak_force_n, self.interaction_force_n)
        self.peak_torque_nm = max(self.peak_torque_nm, self.interaction_torque_nm)
        self.puncture_work_j += self.interaction_force_n * command_speed_m_s * max(float(dt_s), 0.0)
        return adjusted

    def snapshot(self) -> dict[str, Any]:
        return {
            "active": True,
            "state": self.state,
            "punctured": self.punctured,
            "penetration_depth_m": round(self.penetration_depth_m, 6),
            "max_penetration_depth_m": round(self.max_penetration_depth_m, 6),
            "interaction_force_n": round(self.interaction_force_n, 4),
            "peak_force_n": round(self.peak_force_n, 4),
            "interaction_torque_nm": round(self.interaction_torque_nm, 6),
            "peak_torque_nm": round(self.peak_torque_nm, 6),
            "rotation_scale": round(self.rotation_scale, 4),
            "puncture_threshold_n": self.material.puncture_force_n,
            "puncture_count": self.puncture_count,
            "exit_count": self.exit_count,
            "puncture_work_j": round(self.puncture_work_j, 7),
            "curvature_alignment": round(self.curvature_alignment, 4),
            "safe_force_n": self.material.safe_force_n,
            "safe_torque_nm": round(self.material.safe_torque_nm, 6),
            "safe_envelope_active": self.safe_envelope_active,
            "safe_envelope_events": self.safe_envelope_events,
            "material_profile": self.material.id,
            "calibration_status": "research_defaults_unvalidated",
        }


@dataclass
class SutureThreadModel:
    """Position-based suture with surface friction, tissue failure, and knot hold."""

    node_count: int = 48
    segment_length_m: float = 0.0032
    damping: float = 0.965
    max_tissue_anchors: int = 2
    required_anchors_for_knot: int = 2
    linear_stiffness_n: float = 2.4
    tensile_limit_n: float = 3.8
    anchor_pullout_force_n: float = 1.1
    support_plane_z_m: float | None = None
    collision_radius_m: float = 0.00045
    support_friction: float = 0.76
    points: np.ndarray = field(init=False)
    previous: np.ndarray = field(init=False)
    fixed: dict[int, np.ndarray] = field(default_factory=dict)
    tissue_anchor_indices: list[int] = field(default_factory=list)
    anchor_damage: dict[int, float] = field(default_factory=dict)
    anchor_kinds: dict[int, str] = field(default_factory=dict)
    anchor_bite_depth_m: dict[int, float] = field(default_factory=dict)
    anchor_slip_m: dict[int, float] = field(default_factory=dict)
    initialized: bool = False
    tension_n: float = 0.0
    peak_tension_n: float = 0.0
    strain: float = 0.0
    slack_m: float = 0.0
    knot_formed: bool = False
    knot_tightness: float = 0.0
    knot_security: float = 0.0
    over_tension_events: int = 0
    over_tension_active: bool = False
    tensile_overload_s: float = 0.0
    tissue_tear_events: int = 0
    anchor_pullouts: int = 0
    thread_broken: bool = False
    last_anchor_world: np.ndarray | None = None
    last_added_anchor_index: int | None = None
    anchor_spacings_m: list[float] = field(default_factory=list)
    closure_gap_m: float = 0.0
    closure_ratio: float = 0.0
    retained_closure: float = 0.0
    surface_coupling_force_n: float = 0.0
    failure_reason: str = ""
    knot_guide_targets: dict[int, np.ndarray] = field(default_factory=dict)
    knot_guide_weights: dict[int, float] = field(default_factory=dict)
    retained_crossings: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    instrument_contact_centers: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    instrument_contact_radius_m: float = 0.0024
    knot_throw_count: int = 0
    knot_slippage_m: float = 0.0
    knot_center_world: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.points = np.zeros((self.node_count, 3), dtype=np.float32)
        self.previous = self.points.copy()

    def reset(self) -> None:
        self.points.fill(0.0)
        self.previous.fill(0.0)
        self.fixed.clear()
        self.tissue_anchor_indices.clear()
        self.anchor_damage.clear()
        self.anchor_kinds.clear()
        self.anchor_bite_depth_m.clear()
        self.anchor_slip_m.clear()
        self.initialized = False
        self.tension_n = self.peak_tension_n = self.strain = self.slack_m = 0.0
        self.knot_formed = False
        self.knot_tightness = self.knot_security = 0.0
        self.over_tension_events = 0
        self.over_tension_active = False
        self.tensile_overload_s = 0.0
        self.tissue_tear_events = self.anchor_pullouts = 0
        self.thread_broken = False
        self.last_anchor_world = None
        self.last_added_anchor_index = None
        self.anchor_spacings_m.clear()
        self.closure_gap_m = self.closure_ratio = self.retained_closure = 0.0
        self.surface_coupling_force_n = 0.0
        self.failure_reason = ""
        self.knot_guide_targets.clear()
        self.knot_guide_weights.clear()
        self.retained_crossings.clear()
        self.instrument_contact_centers = np.empty((0, 3), dtype=np.float32)
        self.knot_throw_count = 0
        self.knot_slippage_m = 0.0
        self.knot_center_world = None

    @property
    def rest_length_m(self) -> float:
        return self.segment_length_m * (self.node_count - 1)

    def initialize(self, needle_world: np.ndarray) -> None:
        needle = np.asarray(needle_world, dtype=np.float32)
        tail = needle + np.asarray((0.0, -min(self.rest_length_m * 0.60, 0.095), 0.0), dtype=np.float32)
        if self.support_plane_z_m is not None:
            tail[2] = max(float(self.support_plane_z_m) + self.collision_radius_m, min(float(needle[2]), 0.0015))
        alpha = np.linspace(0.0, 1.0, self.node_count, dtype=np.float32)[:, None]
        self.points[:] = tail[None, :] * (1.0 - alpha) + needle[None, :] * alpha
        # Lay the initial slack in a visible, table-supported S curve instead
        # of allowing all excess length to collapse into the support plane.
        slack_amplitude = min(0.018, max(0.006, self.rest_length_m * 0.11))
        self.points[:, 0] += np.sin(alpha[:, 0] * np.pi * 2.0) * slack_amplitude
        self.previous[:] = self.points
        self.fixed = {0: tail.copy(), self.node_count - 1: needle.copy()}
        self.initialized = True

    def set_tail_world(self, tail_world: np.ndarray) -> None:
        """Move the free strand end with a grasping second instrument."""
        if not self.initialized or self.thread_broken:
            return
        target = np.asarray(tail_world, dtype=np.float32).copy()
        previous_target = self.fixed.get(0, self.points[0]).copy()
        delta = target - previous_target
        self.fixed[0] = target
        self.points[0] = target
        self.previous[0] += delta

    def set_instrument_contacts(
        self,
        centers_world: list[np.ndarray] | tuple[np.ndarray, ...],
        radius_m: float = 0.0024,
    ) -> None:
        """Set tool-tip collision proxies used by the strand solver."""
        valid = [np.asarray(center, dtype=np.float32).reshape(3) for center in centers_world if center is not None]
        self.instrument_contact_centers = (
            np.stack(valid).astype(np.float32) if valid else np.empty((0, 3), dtype=np.float32)
        )
        self.instrument_contact_radius_m = max(float(radius_m), self.collision_radius_m * 2.0)

    def set_knot_route(
        self,
        route_points: np.ndarray,
        crossing_pairs: list[tuple[int, int]],
        completed_turn_count: int,
        completed_throw_count: int,
        cinch_progress: float,
        center_world: np.ndarray,
    ) -> None:
        """Apply progressive contact guides to the real strand.

        The guides act like contact with the opposing strand.  They do not
        replace the PBD points or create a second render curve: segment length,
        gravity, tool contacts, self-collision, retained crossings and endpoint
        motion continue to determine the visible thread.
        """
        if not self.initialized or self.thread_broken:
            return
        route = np.asarray(route_points, dtype=np.float32).reshape(-1, 3)
        start_index = max(4, self.node_count // 6)
        maximum_points = max(0, self.node_count - start_index - 5)
        if len(route) > maximum_points:
            sample_indices = np.linspace(0, len(route) - 1, maximum_points, dtype=np.int32)
            route = route[sample_indices]
            remapped_pairs: list[tuple[int, int]] = []
            for left, right in crossing_pairs:
                remapped_pairs.append(
                    (
                        int(round(left * max(maximum_points - 1, 1) / max(len(route_points) - 1, 1))),
                        int(round(right * max(maximum_points - 1, 1) / max(len(route_points) - 1, 1))),
                    )
                )
            crossing_pairs = remapped_pairs

        target_indices = {start_index + offset for offset in range(len(route))}
        for index in list(self.knot_guide_weights):
            if index not in target_indices:
                weight = self.knot_guide_weights[index] * 0.78
                if weight < 0.025:
                    self.knot_guide_weights.pop(index, None)
                    self.knot_guide_targets.pop(index, None)
                else:
                    self.knot_guide_weights[index] = weight

        for offset, target in enumerate(route):
            index = start_index + offset
            previous_target = self.knot_guide_targets.get(index)
            self.knot_guide_targets[index] = (
                target.copy()
                if previous_target is None
                else previous_target * 0.62 + target * 0.38
            )
            self.knot_guide_weights[index] = min(
                1.0,
                self.knot_guide_weights.get(index, 0.0) + 0.085,
            )

        retained_limit = max(0, min(int(completed_turn_count), len(crossing_pairs)))
        minimum_separation = self.collision_radius_m * 2.15
        for left, right in crossing_pairs[:retained_limit]:
            first = start_index + int(left)
            second = start_index + int(right)
            if first == second or first < 0 or second >= self.node_count:
                continue
            key = (min(first, second), max(first, second))
            if key in self.retained_crossings:
                continue
            offset = self.knot_guide_targets.get(second, self.points[second]) - self.knot_guide_targets.get(
                first, self.points[first]
            )
            distance = float(np.linalg.norm(offset))
            if distance < 1e-7:
                offset = np.asarray((0.0, 0.0, minimum_separation), dtype=np.float32)
            elif distance < minimum_separation:
                offset = offset * (minimum_separation / distance)
            self.retained_crossings[key] = np.asarray(offset, dtype=np.float32)

        self.knot_center_world = np.asarray(center_world, dtype=np.float32).copy()
        self.knot_throw_count = max(self.knot_throw_count, int(completed_throw_count))
        self.knot_formed = self.knot_formed or self.knot_throw_count >= 1
        completed_ratio = float(np.clip(self.knot_throw_count / 3.0, 0.0, 1.0))
        cinch = float(np.clip(cinch_progress, 0.0, 1.0))
        self.knot_tightness = max(self.knot_tightness, completed_ratio * 0.78 + cinch * 0.22)
        slip_penalty = float(np.clip(self.knot_slippage_m / 0.008, 0.0, 1.0))
        self.knot_security = float(
            np.clip(max(self.knot_security, self.knot_tightness * (1.0 - 0.55 * slip_penalty)), 0.0, 1.0)
        )

    def _apply_knot_guides(self) -> None:
        for index, target in self.knot_guide_targets.items():
            if index in self.fixed or index < 0 or index >= self.node_count:
                continue
            weight = float(np.clip(self.knot_guide_weights.get(index, 0.0), 0.0, 1.0))
            self.points[index] += (target - self.points[index]) * (0.045 + 0.16 * weight)

    def _apply_retained_crossings(self) -> None:
        for (first, second), target_offset in self.retained_crossings.items():
            if first >= self.node_count or second >= self.node_count:
                continue
            error = (self.points[second] - self.points[first]) - target_offset
            first_fixed = first in self.fixed
            second_fixed = second in self.fixed
            if first_fixed and not second_fixed:
                self.points[second] -= error * 0.42
            elif second_fixed and not first_fixed:
                self.points[first] += error * 0.42
            elif not first_fixed and not second_fixed:
                self.points[first] += error * 0.21
                self.points[second] -= error * 0.21

    def _apply_self_collision(self) -> int:
        if not self.knot_guide_targets and not self.retained_crossings:
            return 0
        minimum_distance = self.collision_radius_m * 2.15
        delta = self.points[:, None, :] - self.points[None, :, :]
        distance_sq = np.sum(delta * delta, axis=2)
        mask = np.triu(np.ones((self.node_count, self.node_count), dtype=bool), k=3)
        mask &= distance_sq < minimum_distance * minimum_distance
        pairs = np.argwhere(mask)
        contacts = 0
        for first, second in pairs[:384]:
            separation = self.points[second] - self.points[first]
            distance = float(np.linalg.norm(separation))
            if distance < 1e-8:
                separation = np.asarray((0.0, 0.0, minimum_distance), dtype=np.float32)
                distance = minimum_distance
            correction = separation * ((minimum_distance - distance) / distance)
            first_fixed = int(first) in self.fixed
            second_fixed = int(second) in self.fixed
            if first_fixed and not second_fixed:
                self.points[second] += correction
            elif second_fixed and not first_fixed:
                self.points[first] -= correction
            elif not first_fixed and not second_fixed:
                self.points[first] -= correction * 0.5
                self.points[second] += correction * 0.5
            contacts += 1
        return contacts

    def _apply_instrument_contacts(self) -> int:
        if not len(self.instrument_contact_centers):
            return 0
        contact_radius = self.instrument_contact_radius_m + self.collision_radius_m
        contacts = 0
        for center in self.instrument_contact_centers:
            offsets = self.points - center[None, :]
            distances = np.linalg.norm(offsets, axis=1)
            for index in np.flatnonzero(distances < contact_radius):
                index = int(index)
                if index in self.fixed:
                    continue
                distance = float(distances[index])
                direction = (
                    offsets[index] / distance
                    if distance >= 1e-8
                    else np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
                )
                self.points[index] = center + direction * contact_radius
                contacts += 1
        return contacts

    def add_tissue_anchor(
        self,
        world_position: np.ndarray,
        kind: str = "entry",
        bite_depth_m: float = 0.0,
    ) -> bool:
        if not self.initialized or len(self.tissue_anchor_indices) >= self.max_tissue_anchors:
            self.last_added_anchor_index = None
            return False
        fractions = np.linspace(0.16, 0.82, max(2, self.max_tissue_anchors), dtype=np.float32)
        index = int(round((self.node_count - 1) * float(fractions[len(self.tissue_anchor_indices)])))
        if index in self.fixed:
            self.last_added_anchor_index = None
            return False
        position = np.asarray(world_position, dtype=np.float32).copy()
        if self.last_anchor_world is not None:
            self.anchor_spacings_m.append(float(np.linalg.norm(position - self.last_anchor_world)))
        self.last_anchor_world = position.copy()
        self.fixed[index] = position
        self.points[index] = position
        self.previous[index] = position
        self.tissue_anchor_indices.append(index)
        self.anchor_damage[index] = 0.0
        self.anchor_kinds[index] = str(kind)
        self.anchor_bite_depth_m[index] = max(0.0, float(bite_depth_m))
        self.anchor_slip_m[index] = 0.0
        self.last_added_anchor_index = index
        return True

    def update_tissue_anchor(
        self,
        anchor_index: int,
        world_position: np.ndarray,
        static_friction: float,
        dynamic_friction: float,
        dt_s: float,
    ) -> None:
        """Move a pinned thread node with tissue and accumulate frictional slip."""
        index = int(anchor_index)
        if index not in self.tissue_anchor_indices:
            return
        target = np.asarray(world_position, dtype=np.float32)
        previous = self.fixed.get(index, target)
        load_ratio = self.tension_n / max(self.anchor_pullout_force_n, 1e-6)
        static_limit = float(np.clip(static_friction, 0.05, 1.0))
        if load_ratio > static_limit:
            sliding_load = load_ratio - static_limit
            slip_rate = sliding_load * (1.0 - float(np.clip(dynamic_friction, 0.0, 0.95))) * 0.0018
            self.anchor_slip_m[index] = self.anchor_slip_m.get(index, 0.0) + slip_rate * max(float(dt_s), 0.0)
        self.fixed[index] = target.copy()
        # Avoid injecting surface-authoring motion as artificial strand
        # velocity.  The tissue-bound node follows its material coordinates.
        self.points[index] = target
        self.previous[index] += target - previous

    def detach_tissue_anchor(self, anchor_index: int, reason: str = "anchor_pullout") -> None:
        index = int(anchor_index)
        if index not in self.tissue_anchor_indices:
            return
        self.fixed.pop(index, None)
        self.tissue_anchor_indices.remove(index)
        self.anchor_damage.pop(index, None)
        self.anchor_kinds.pop(index, None)
        self.anchor_bite_depth_m.pop(index, None)
        self.anchor_slip_m.pop(index, None)
        self.anchor_pullouts += 1
        self.tissue_tear_events += 1
        self.failure_reason = str(reason)

    @property
    def active_anchor_pairs(self) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        anchors = list(self.tissue_anchor_indices)
        for index in range(0, len(anchors) - 1, 2):
            entry, exit_ = anchors[index], anchors[index + 1]
            if self.anchor_kinds.get(entry) == "entry" and self.anchor_kinds.get(exit_) == "exit":
                pairs.append((entry, exit_))
        return pairs

    def record_surface_coupling(self, snapshot: dict[str, Any]) -> None:
        self.closure_gap_m = max(0.0, float(snapshot.get("closure_gap_m", 0.0)))
        self.closure_ratio = float(np.clip(snapshot.get("closure_ratio", 0.0), 0.0, 1.0))
        self.retained_closure = float(np.clip(snapshot.get("retained_closure", 0.0), 0.0, 1.0))
        self.surface_coupling_force_n = self.tension_n if self.active_anchor_pairs else 0.0

    def _apply_constraints(self, iterations: int = 8) -> float:
        fixed_indices = sorted(self.fixed)
        raw_stretch = 0.0
        for left, right in zip(fixed_indices[:-1], fixed_indices[1:]):
            available = (right - left) * self.segment_length_m
            routed_span = float(np.linalg.norm(self.fixed[right] - self.fixed[left]))
            raw_stretch = max(raw_stretch, max(0.0, routed_span / max(available, 1e-8) - 1.0))
        for _ in range(max(1, int(iterations))):
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
            if self.knot_formed and not self.retained_crossings:
                near = max(2, self.node_count // 4)
                far = min(self.node_count - 3, (self.node_count * 3) // 4)
                delta = self.points[far] - self.points[near]
                length = float(np.linalg.norm(delta))
                target = 0.0022 + (1.0 - self.knot_security) * 0.004
                if length > target:
                    correction = delta * ((length - target) / max(length, 1e-8)) * 0.5
                    if near not in self.fixed:
                        self.points[near] += correction
                    if far not in self.fixed:
                        self.points[far] -= correction
            self._apply_knot_guides()
            self._apply_retained_crossings()
            self._apply_self_collision()
            self._apply_instrument_contacts()
            for index, position in self.fixed.items():
                self.points[index] = position
        residual_lengths = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        residual_stretch = max(
            0.0,
            float(np.max(residual_lengths, initial=0.0)) / max(self.segment_length_m, 1e-8) - 1.0,
        )
        raw_stretch = max(raw_stretch, residual_stretch)
        return raw_stretch

    def project_knot_constraints(self, iterations: int = 4) -> None:
        """Project the latest wrap/contact state before authoring the USD curve."""
        if not self.initialized:
            return
        before = self.points.copy()
        projected_strain = self._apply_constraints(iterations)
        self._apply_support_contact()
        displacement = self.points - before
        self.previous += displacement
        self.strain = max(self.strain, projected_strain)
        if not self.thread_broken:
            self.tension_n = float(np.clip(self.strain * self.linear_stiffness_n, 0.0, 8.0))
            self.peak_tension_n = max(self.peak_tension_n, self.tension_n)
        if self.retained_crossings and self.knot_center_world is not None:
            crossing_points = [
                (self.points[first] + self.points[second]) * 0.5
                for first, second in self.retained_crossings
                if first < self.node_count and second < self.node_count
            ]
            if crossing_points:
                knot_midpoint = np.mean(np.stack(crossing_points), axis=0)
                self.knot_slippage_m = max(
                    self.knot_slippage_m,
                    float(np.linalg.norm(knot_midpoint - self.knot_center_world)),
                )
                slip_penalty = float(np.clip(self.knot_slippage_m / 0.008, 0.0, 1.0))
                self.knot_security = float(
                    np.clip(self.knot_tightness * (1.0 - 0.55 * slip_penalty), 0.0, 1.0)
                )

    def _apply_support_contact(self) -> int:
        """Keep the free strand above the instrument table and damp sliding."""
        if self.support_plane_z_m is None:
            return 0
        support_height = float(self.support_plane_z_m) + self.collision_radius_m
        contacts = self.points[:, 2] < support_height
        for index in self.fixed:
            contacts[index] = False
        if not bool(np.any(contacts)):
            return 0
        velocity_xy = self.points[contacts, :2] - self.previous[contacts, :2]
        self.points[contacts, 2] = support_height
        self.previous[contacts, 2] = support_height
        retained_velocity = 1.0 - float(np.clip(self.support_friction, 0.0, 0.98))
        self.previous[contacts, :2] = self.points[contacts, :2] - velocity_xy * retained_velocity
        return int(np.count_nonzero(contacts))

    def update(self, needle_world: np.ndarray, dt: float) -> None:
        needle = np.asarray(needle_world, dtype=np.float32)
        if not self.initialized:
            self.initialize(needle)
        if self.thread_broken:
            self.fixed.pop(self.node_count - 1, None)
        else:
            self.fixed[self.node_count - 1] = needle.copy()
        dt = float(np.clip(dt, 1.0 / 240.0, 1.0 / 15.0))
        velocity = (self.points - self.previous) * self.damping
        self.previous[:] = self.points
        self.points += velocity + np.asarray((0.0, 0.0, -1.4), dtype=np.float32)[None, :] * (dt * dt)
        for index, position in self.fixed.items():
            self.points[index] = position
        self.strain = self._apply_constraints()
        self._apply_support_contact()
        # Projection onto the table can perturb adjacent segment lengths, so
        # settle once more before the final contact projection.
        self.strain = max(self.strain, self._apply_constraints())
        self._apply_support_contact()
        self.tension_n = 0.0 if self.thread_broken else float(np.clip(self.strain * self.linear_stiffness_n, 0.0, 8.0))
        routed_length = float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())
        self.slack_m = max(0.0, self.rest_length_m - routed_length)
        self.peak_tension_n = max(self.peak_tension_n, self.tension_n)
        over_tension = self.tension_n > min(1.5, self.anchor_pullout_force_n)
        if over_tension and not self.over_tension_active:
            self.over_tension_events += 1
        self.over_tension_active = over_tension
        for index in list(self.tissue_anchor_indices):
            excess = max(0.0, self.tension_n - self.anchor_pullout_force_n * 0.72)
            self.anchor_damage[index] = float(np.clip(self.anchor_damage.get(index, 0.0) + excess * dt / max(self.anchor_pullout_force_n, 1e-6), 0.0, 1.2))
            if self.anchor_damage[index] >= 1.0:
                self.detach_tissue_anchor(index, "thread_load_anchor_pullout")
        if self.tension_n >= self.tensile_limit_n:
            self.tensile_overload_s += dt
        else:
            self.tensile_overload_s = max(0.0, self.tensile_overload_s - dt * 2.0)
        if self.tensile_overload_s >= 0.30:
            self.thread_broken = True
            self.tension_n = 0.0
        if len(self.tissue_anchor_indices) >= self.required_anchors_for_knot and not self.knot_formed:
            entry = self.fixed[self.tissue_anchor_indices[0]]
            loop_closed = float(np.linalg.norm(needle - entry)) <= 0.014
            if loop_closed and routed_length >= self.rest_length_m * 0.82:
                self.knot_formed = True
        if self.knot_formed:
            self.knot_tightness = float(np.clip(max(self.knot_tightness, self.tension_n / 0.8), 0.0, 1.0))
            load_penalty = max(0.0, self.tension_n - 1.0) * 0.08
            self.knot_security = float(np.clip(max(self.knot_security, self.knot_tightness * 0.88) - load_penalty * dt, 0.0, 1.0))

    @property
    def stitch_count(self) -> int:
        return len(self.active_anchor_pairs)

    @property
    def mean_anchor_spacing_m(self) -> float:
        return float(np.mean(self.anchor_spacings_m)) if self.anchor_spacings_m else 0.0

    @property
    def spacing_variation_m(self) -> float:
        return float(np.std(self.anchor_spacings_m)) if len(self.anchor_spacings_m) > 1 else 0.0

    @property
    def mean_bite_depth_m(self) -> float:
        return float(np.mean(list(self.anchor_bite_depth_m.values()))) if self.anchor_bite_depth_m else 0.0

    @property
    def total_anchor_slip_m(self) -> float:
        return float(sum(self.anchor_slip_m.values()))

    def snapshot(self) -> dict[str, Any]:
        return {
            "active": True,
            "visible": self.initialized,
            "tension_n": round(self.tension_n, 4),
            "peak_tension_n": round(self.peak_tension_n, 4),
            "strain": round(self.strain, 5),
            "slack_m": round(self.slack_m, 5),
            "tissue_anchors": len(self.tissue_anchor_indices),
            "entry_anchors": sum(kind == "entry" for kind in self.anchor_kinds.values()),
            "exit_anchors": sum(kind == "exit" for kind in self.anchor_kinds.values()),
            "stitch_count": self.stitch_count,
            "mean_bite_depth_m": round(self.mean_bite_depth_m, 5),
            "mean_bite_spacing_m": round(self.mean_anchor_spacing_m, 5),
            "spacing_variation_m": round(self.spacing_variation_m, 5),
            "anchor_slip_m": round(self.total_anchor_slip_m, 6),
            "over_tension_events": self.over_tension_events,
            "tensile_overload_s": round(self.tensile_overload_s, 4),
            "tissue_tear_events": self.tissue_tear_events,
            "anchor_pullouts": self.anchor_pullouts,
            "anchor_damage_max": round(max(self.anchor_damage.values(), default=0.0), 4),
            "thread_broken": self.thread_broken,
            "knot_formed": self.knot_formed,
            "knot_tightness": round(self.knot_tightness, 4),
            "knot_security": round(self.knot_security, 4),
            "knot_throw_count": self.knot_throw_count,
            "retained_crossings": len(self.retained_crossings),
            "knot_slippage_m": round(self.knot_slippage_m, 6),
            "closure_gap_m": round(self.closure_gap_m, 6),
            "closure_ratio": round(self.closure_ratio, 4),
            "retained_closure": round(self.retained_closure, 4),
            "surface_coupling_force_n": round(self.surface_coupling_force_n, 4),
            "failure_reason": self.failure_reason,
            "model": "surface_coupled_position_based_suture_v4",
            "calibration_status": "research_defaults_unvalidated",
        }


def interaction_force_snapshot(
    needle: NeedleTissueInteractionModel | None,
    tissue: SurfaceMeshModel | None,
    thread: SutureThreadModel | None,
) -> dict[str, Any]:
    """Combine fallback interaction loads without replacing native sensor force."""
    components = {
        "needle_n": float(needle.interaction_force_n) if needle else 0.0,
        "thread_n": float(thread.tension_n) if thread else 0.0,
        "cut_n": float(tissue.cut_resistance_n) if tissue else 0.0,
        "attachment_n": 0.0,
    }
    if tissue:
        current_displacement = float(
            np.linalg.norm(tissue.current_points - tissue.original_points, axis=1).max(initial=0.0)
        )
        components["attachment_n"] = float(
            np.clip(tissue.material.youngs_modulus_pa * tissue.max_strain * current_displacement * 2e-5, 0.0, 8.0)
        )
    total = float(np.sqrt(sum(value * value for value in components.values())))
    safe_force = tissue.material.safe_force_n if tissue else needle.material.safe_force_n if needle else 2.0
    torque = float(needle.interaction_torque_nm) if needle else 0.0
    safe_torque = tissue.material.safe_torque_nm if tissue else needle.material.safe_torque_nm if needle else 0.06
    return {
        "active": any(value > 0.0 for value in components.values()),
        "resultant_proxy_n": round(total, 4),
        "components": {key: round(value, 4) for key, value in components.items()},
        "safe_force_n": safe_force,
        "resultant_proxy_torque_nm": round(torque, 6),
        "safe_torque_nm": round(safe_torque, 6),
        "safe_envelope_active": total >= safe_force or torque >= safe_torque,
        "authority": "native_contact_sensor_preferred_fallback_coupling_only",
        "calibration_status": "research_defaults_unvalidated",
    }
