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
        self._tear_active = False

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
        movable = (1.0 - 0.88 * self.attachment_weights)[:, None]
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
            "model": "reduced_order_volume_preserving_tissue_v2",
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
    """Position-based suture with tension, slack, anchor damage, and knot hold."""

    node_count: int = 48
    segment_length_m: float = 0.0032
    damping: float = 0.965
    max_tissue_anchors: int = 2
    required_anchors_for_knot: int = 2
    linear_stiffness_n: float = 2.4
    tensile_limit_n: float = 3.8
    anchor_pullout_force_n: float = 1.1
    points: np.ndarray = field(init=False)
    previous: np.ndarray = field(init=False)
    fixed: dict[int, np.ndarray] = field(default_factory=dict)
    tissue_anchor_indices: list[int] = field(default_factory=list)
    anchor_damage: dict[int, float] = field(default_factory=dict)
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
    tissue_tear_events: int = 0
    anchor_pullouts: int = 0
    thread_broken: bool = False
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
        self.anchor_damage.clear()
        self.initialized = False
        self.tension_n = self.peak_tension_n = self.strain = self.slack_m = 0.0
        self.knot_formed = False
        self.knot_tightness = self.knot_security = 0.0
        self.over_tension_events = 0
        self.over_tension_active = False
        self.tissue_tear_events = self.anchor_pullouts = 0
        self.thread_broken = False
        self.last_anchor_world = None
        self.anchor_spacings_m.clear()

    @property
    def rest_length_m(self) -> float:
        return self.segment_length_m * (self.node_count - 1)

    def initialize(self, needle_world: np.ndarray) -> None:
        needle = np.asarray(needle_world, dtype=np.float32)
        tail = needle + np.asarray((0.0, -self.rest_length_m * 0.78, 0.018), dtype=np.float32)
        alpha = np.linspace(0.0, 1.0, self.node_count, dtype=np.float32)[:, None]
        self.points[:] = tail[None, :] * (1.0 - alpha) + needle[None, :] * alpha
        self.previous[:] = self.points
        self.fixed = {0: tail.copy(), self.node_count - 1: needle.copy()}
        self.initialized = True

    def add_tissue_anchor(self, world_position: np.ndarray) -> bool:
        if not self.initialized or len(self.tissue_anchor_indices) >= self.max_tissue_anchors:
            return False
        fractions = np.linspace(0.16, 0.82, max(2, self.max_tissue_anchors), dtype=np.float32)
        index = int(round((self.node_count - 1) * float(fractions[len(self.tissue_anchor_indices)])))
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
        self.anchor_damage[index] = 0.0
        return True

    def _apply_constraints(self) -> float:
        fixed_indices = sorted(self.fixed)
        raw_stretch = 0.0
        for left, right in zip(fixed_indices[:-1], fixed_indices[1:]):
            available = (right - left) * self.segment_length_m
            routed_span = float(np.linalg.norm(self.fixed[right] - self.fixed[left]))
            raw_stretch = max(raw_stretch, max(0.0, routed_span / max(available, 1e-8) - 1.0))
        for _ in range(8):
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
                target = 0.0022 + (1.0 - self.knot_security) * 0.004
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
                self.fixed.pop(index, None)
                self.tissue_anchor_indices.remove(index)
                self.anchor_pullouts += 1
                self.tissue_tear_events += 1
                self.anchor_damage.pop(index, None)
        if self.tension_n >= self.tensile_limit_n:
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
        return len(self.tissue_anchor_indices) // 2

    @property
    def mean_anchor_spacing_m(self) -> float:
        return float(np.mean(self.anchor_spacings_m)) if self.anchor_spacings_m else 0.0

    @property
    def spacing_variation_m(self) -> float:
        return float(np.std(self.anchor_spacings_m)) if len(self.anchor_spacings_m) > 1 else 0.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "active": True,
            "visible": self.initialized,
            "tension_n": round(self.tension_n, 4),
            "peak_tension_n": round(self.peak_tension_n, 4),
            "strain": round(self.strain, 5),
            "slack_m": round(self.slack_m, 5),
            "tissue_anchors": len(self.tissue_anchor_indices),
            "stitch_count": self.stitch_count,
            "mean_bite_spacing_m": round(self.mean_anchor_spacing_m, 5),
            "spacing_variation_m": round(self.spacing_variation_m, 5),
            "over_tension_events": self.over_tension_events,
            "tissue_tear_events": self.tissue_tear_events,
            "anchor_pullouts": self.anchor_pullouts,
            "anchor_damage_max": round(max(self.anchor_damage.values(), default=0.0), 4),
            "thread_broken": self.thread_broken,
            "knot_formed": self.knot_formed,
            "knot_tightness": round(self.knot_tightness, 4),
            "knot_security": round(self.knot_security, 4),
            "model": "position_based_suture_with_anchor_failure_v2",
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
