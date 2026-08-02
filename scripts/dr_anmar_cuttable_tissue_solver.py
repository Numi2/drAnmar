#!/usr/bin/env python3
"""Cutting-ready nonlinear tissue and scalpel-contact reference mechanics.

This module deliberately stops before fracture.  It qualifies the continuum
and contact mechanics that a later arbitrary-cutting backend must preserve:
finite-strain elasticity, stress relaxation, pre-tensioned attachments,
continuous rounded-edge contact, two-way force, and deterministic receipts.

The implementation is a compact Total-Lagrangian tetrahedral reference, not a
production real-time solver and not a calibrated model of human tissue.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_PATH = REPOSITORY_ROOT / "physics_next/tissues/dr-anmar-cuttable-tissue-v1.json"


@dataclass(frozen=True)
class ScalpelPose:
    center_m: tuple[float, float, float]
    tangent: tuple[float, float, float] = (0.0, 1.0, 0.0)
    velocity_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class CuttableTissueReceipt:
    schema: str
    profile_id: str
    profile_sha256: str
    point_count: int
    tetrahedron_count: int
    steps: int
    finite: bool
    inverted_tetrahedra_peak: int
    minimum_jacobian: float
    maximum_anchor_drift_m: float
    maximum_volume_error_fraction: float
    maximum_contact_penetration_m: float
    off_grid_contact_coverage_fraction: float
    off_grid_force_variation_fraction: float
    peak_scalpel_force_n: float
    peak_scalpel_tangential_force_n: float
    force_at_hold_start_n: float
    force_at_hold_end_n: float
    force_relaxation_fraction: float
    peak_surface_displacement_m: float
    recovery_residual_m: float
    fracture_enabled: bool
    fracture_event_count: int
    deterministic_trace_sha256: str
    qualified: bool
    failed_gates: tuple[str, ...]
    evidence_boundary: str

    def payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failed_gates"] = list(self.failed_gates)
        return payload


def load_profile(path: Path = DEFAULT_PROFILE_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalized(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("Direction must be finite and non-zero")
    return value / norm


def build_regular_tetrahedral_coupon(
    profile: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Build a connected conforming coupon with six tets around each cell diagonal."""

    geometry = profile["geometry"]
    counts = tuple(int(geometry[f"cells_{axis}"]) for axis in "xyz")
    if any(value <= 0 for value in counts):
        raise ValueError("Coupon cell counts must be positive")
    widths = (
        float(geometry["width_m"]),
        float(geometry["depth_m"]),
        float(geometry["thickness_m"]),
    )
    axes = [
        np.linspace(-width / 2.0, width / 2.0, count + 1, dtype=np.float64)
        for width, count in zip(widths, counts, strict=True)
    ]
    points = np.asarray(
        [(x, y, z) for z in axes[2] for y in axes[1] for x in axes[0]],
        dtype=np.float64,
    )
    nx, ny, nz = counts

    def index(x_index: int, y_index: int, z_index: int) -> int:
        return z_index * (ny + 1) * (nx + 1) + y_index * (nx + 1) + x_index

    pattern = (
        (0, 1, 3, 7),
        (0, 3, 2, 7),
        (0, 2, 6, 7),
        (0, 6, 4, 7),
        (0, 4, 5, 7),
        (0, 5, 1, 7),
    )
    tetrahedra: list[tuple[int, int, int, int]] = []
    for z_index in range(nz):
        for y_index in range(ny):
            for x_index in range(nx):
                corners = (
                    index(x_index, y_index, z_index),
                    index(x_index + 1, y_index, z_index),
                    index(x_index, y_index + 1, z_index),
                    index(x_index + 1, y_index + 1, z_index),
                    index(x_index, y_index, z_index + 1),
                    index(x_index + 1, y_index, z_index + 1),
                    index(x_index, y_index + 1, z_index + 1),
                    index(x_index + 1, y_index + 1, z_index + 1),
                )
                tetrahedra.extend(tuple(corners[local] for local in tet) for tet in pattern)
    tets = np.asarray(tetrahedra, dtype=np.int64)
    signed = np.linalg.det(
        np.stack(
            (
                points[tets[:, 1]] - points[tets[:, 0]],
                points[tets[:, 2]] - points[tets[:, 0]],
                points[tets[:, 3]] - points[tets[:, 0]],
            ),
            axis=2,
        )
    )
    negative = signed < 0.0
    if np.any(negative):
        swapped = tets[negative, 0].copy()
        tets[negative, 0] = tets[negative, 1]
        tets[negative, 1] = swapped
    if np.any(np.abs(signed) <= 1.0e-16):
        raise ValueError("Coupon tetrahedralization produced a degenerate element")
    return points, tets


class CuttableTissueReferenceSolver:
    """Small deterministic Total-Lagrangian tetrahedral reference solver."""

    def __init__(self, profile: dict[str, Any]):
        self.profile = profile
        self.rest, self.tets = build_regular_tetrahedral_coupon(profile)
        self.position = self.rest.copy()
        self.velocity = np.zeros_like(self.position)
        geometry = profile["geometry"]
        prestrain = float(geometry["prestrain_x"])
        self.position[:, 0] *= 1.0 + prestrain
        self.initial_position = self.position.copy()
        half_width = float(geometry["width_m"]) * (1.0 + prestrain) / 2.0
        band = float(geometry["attachment_band_m"])
        self.fixed = np.abs(self.position[:, 0]) >= half_width - band
        self.fixed_position = self.position[self.fixed].copy()

        x0 = self.rest[self.tets[:, 0]]
        dm = np.stack(
            (
                self.rest[self.tets[:, 1]] - x0,
                self.rest[self.tets[:, 2]] - x0,
                self.rest[self.tets[:, 3]] - x0,
            ),
            axis=2,
        )
        self.dm_inverse = np.linalg.inv(dm)
        self.rest_volume = np.linalg.det(dm) / 6.0
        if np.any(self.rest_volume <= 0.0):
            raise ValueError("Reference tetrahedra must have positive volume")
        rows = self.dm_inverse
        self.shape_gradients = np.empty((len(self.tets), 4, 3), dtype=np.float64)
        self.shape_gradients[:, 1:, :] = rows
        self.shape_gradients[:, 0, :] = -np.sum(rows, axis=1)

        density = float(profile["material"]["density_kg_m3"])
        self.mass = np.zeros(len(self.rest), dtype=np.float64)
        for local in range(4):
            np.add.at(self.mass, self.tets[:, local], density * self.rest_volume / 4.0)
        if np.any(self.mass <= 0.0):
            raise ValueError("Every tissue node must have positive lumped mass")

        top = np.isclose(self.rest[:, 2], np.max(self.rest[:, 2]), atol=1.0e-12)
        self.top_nodes = np.flatnonzero(top)
        self.top_triangles = self._top_surface_triangles()
        self.prony_history = np.zeros((len(self.tets), 3, 3), dtype=np.float64)
        self.previous_elastic_stress = np.zeros_like(self.prony_history)
        self.fracture_work_j = np.zeros(len(self.tets), dtype=np.float64)
        self.damage = np.zeros(len(self.tets), dtype=np.float64)
        self.fracture_event_count = 0
        self.steps = 0

    def _top_surface_triangles(self) -> np.ndarray:
        faces: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        counts: dict[tuple[int, int, int], int] = {}
        for a, b, c, d in self.tets:
            for face in ((b, c, d), (a, d, c), (a, b, d), (a, c, b)):
                key = tuple(sorted(int(index) for index in face))
                counts[key] = counts.get(key, 0) + 1
                faces.setdefault(key, tuple(int(index) for index in face))
        maximum_z = float(np.max(self.rest[:, 2]))
        top_faces = [
            faces[key]
            for key, count in counts.items()
            if count == 1 and np.allclose(self.rest[list(key), 2], maximum_z, atol=1.0e-12)
        ]
        if not top_faces:
            raise ValueError("Coupon must expose a triangulated top collision surface")
        return np.asarray(sorted(top_faces), dtype=np.int64)

    def _deformation(self) -> tuple[np.ndarray, np.ndarray]:
        x0 = self.position[self.tets[:, 0]]
        ds = np.stack(
            (
                self.position[self.tets[:, 1]] - x0,
                self.position[self.tets[:, 2]] - x0,
                self.position[self.tets[:, 3]] - x0,
            ),
            axis=2,
        )
        deformation = np.einsum("tij,tjk->tik", ds, self.dm_inverse)
        return deformation, np.linalg.det(deformation)

    def _internal_force(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        material = self.profile["material"]
        youngs = float(material["youngs_modulus_pa"])
        poisson = float(material["poisson_ratio"])
        mu = youngs / (2.0 * (1.0 + poisson))
        lam = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        deformation, jacobian = self._deformation()
        safe_j = np.maximum(jacobian, 1.0e-9)
        inverse_transpose = np.swapaxes(np.linalg.inv(deformation), 1, 2)
        elastic_stress = (
            mu * (deformation - inverse_transpose)
            + lam * np.log(safe_j)[:, None, None] * inverse_transpose
        )

        fraction = float(material["prony_relaxation_fraction"])
        time_constant = float(material["prony_time_constant_s"])
        decay = math.exp(-dt / time_constant)
        self.prony_history = decay * self.prony_history + fraction * (
            elastic_stress - self.previous_elastic_stress
        )
        self.previous_elastic_stress = elastic_stress
        stress = (1.0 - fraction) * elastic_stress + self.prony_history

        local_force = -self.rest_volume[:, None, None] * np.einsum(
            "tij,tkj->tki", stress, self.shape_gradients
        )
        force = np.zeros_like(self.position)
        for local in range(4):
            np.add.at(force, self.tets[:, local], local_force[:, local])
        return force, jacobian

    def _scalpel_contact(
        self,
        pose: ScalpelPose,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        contact = self.profile["scalpel_contact"]
        center = np.asarray(pose.center_m, dtype=np.float64)
        tangent = _normalized(np.asarray(pose.tangent, dtype=np.float64))
        blade_velocity = np.asarray(pose.velocity_m_s, dtype=np.float64)
        half_length = float(contact["edge_length_m"]) / 2.0
        radius = float(contact["edge_radius_m"])
        if abs(float(tangent[2])) > 1.0e-6:
            raise ValueError("Reference top-surface contact requires an in-plane scalpel edge")
        segment_count = int(contact["edge_quadrature_segments"])
        if segment_count < 2:
            raise ValueError("Scalpel edge quadrature requires at least two segments")
        segment_length = 2.0 * half_length / segment_count
        offsets = -half_length + (np.arange(segment_count) + 0.5) * segment_length
        axis_points = center + offsets[:, None] * tangent

        triangles = self.position[self.top_triangles]
        p0 = triangles[:, 0, :2]
        edge_0 = triangles[:, 1, :2] - p0
        edge_1 = triangles[:, 2, :2] - p0
        denominator = edge_0[:, 0] * edge_1[:, 1] - edge_0[:, 1] * edge_1[:, 0]
        query = axis_points[:, None, :2] - p0[None, :, :]
        bary_1 = (
            query[:, :, 0] * edge_1[None, :, 1] - query[:, :, 1] * edge_1[None, :, 0]
        ) / denominator[None, :]
        bary_2 = (
            edge_0[None, :, 0] * query[:, :, 1] - edge_0[None, :, 1] * query[:, :, 0]
        ) / denominator[None, :]
        bary_0 = 1.0 - bary_1 - bary_2
        inside = (bary_0 >= -1.0e-10) & (bary_1 >= -1.0e-10) & (bary_2 >= -1.0e-10)
        valid = np.any(inside, axis=1)
        triangle_slot = np.argmax(inside, axis=1)
        barycentric = np.stack(
            (
                bary_0[np.arange(segment_count), triangle_slot],
                bary_1[np.arange(segment_count), triangle_slot],
                bary_2[np.arange(segment_count), triangle_slot],
            ),
            axis=1,
        )
        selected_triangles = self.top_triangles[triangle_slot]
        selected_position = self.position[selected_triangles]
        selected_velocity = self.velocity[selected_triangles]
        surface_point = np.einsum("si,sij->sj", barycentric, selected_position)
        surface_velocity = np.einsum("si,sij->sj", barycentric, selected_velocity)
        surface_normal = np.cross(
            selected_position[:, 1] - selected_position[:, 0],
            selected_position[:, 2] - selected_position[:, 0],
        )
        surface_normal /= np.maximum(np.linalg.norm(surface_normal, axis=1)[:, None], 1.0e-12)
        surface_normal = np.where(surface_normal[:, 2, None] < 0.0, -surface_normal, surface_normal)
        signed_gap = np.sum((axis_points - surface_point) * surface_normal, axis=1)
        penetration = np.maximum(0.0, radius - signed_gap)
        active = valid & (penetration > 0.0)
        force = np.zeros_like(self.position)
        reaction = np.zeros(3, dtype=np.float64)
        if not np.any(active):
            return force, reaction, 0.0

        normals = -surface_normal[active]
        relative_velocity = surface_velocity[active] - blade_velocity
        normal_velocity = np.sum(relative_velocity * normals, axis=1)
        normal_pressure = float(contact["normal_stiffness_pa_m"]) * penetration[active] - float(
            contact["normal_damping_pa_s_m"]
        ) * np.minimum(normal_velocity, 0.0)
        normal_pressure = np.maximum(0.0, normal_pressure)
        effective_gap = np.clip(signed_gap[active], 0.0, radius)
        strip_width = 2.0 * np.sqrt(np.maximum(0.0, radius * radius - effective_gap**2))
        normal_magnitude = normal_pressure * strip_width * segment_length
        nodal_force = normal_magnitude[:, None] * normals

        tangential_velocity = relative_velocity - normal_velocity[:, None] * normals
        tangential_speed = np.linalg.norm(tangential_velocity, axis=1)
        regularization = float(contact["friction_regularization_m_s"])
        friction_scale = np.tanh(tangential_speed / regularization)
        tangential_direction = tangential_velocity / np.maximum(tangential_speed[:, None], 1.0e-12)
        nodal_force -= (float(contact["dynamic_friction"]) * normal_magnitude * friction_scale)[
            :, None
        ] * tangential_direction
        active_triangles = selected_triangles[active]
        active_barycentric = barycentric[active]
        for local in range(3):
            np.add.at(
                force,
                active_triangles[:, local],
                active_barycentric[:, local, None] * nodal_force,
            )
        reaction = -np.sum(nodal_force, axis=0)
        return force, reaction, float(np.max(penetration[active]))

    def step(
        self,
        dt: float,
        scalpel: ScalpelPose | None = None,
    ) -> dict[str, float | int | bool | np.ndarray]:
        if bool(self.profile["fracture"]["enabled"]):
            raise RuntimeError("Reference milestone must fail closed with fracture disabled")
        force, jacobian = self._internal_force(dt)
        reaction = np.zeros(3, dtype=np.float64)
        penetration = 0.0
        if scalpel is not None:
            contact_force, reaction, penetration = self._scalpel_contact(scalpel)
            force += contact_force
        damping = math.exp(-float(self.profile["material"]["velocity_damping_per_s"]) * dt)
        self.velocity += dt * force / self.mass[:, None]
        self.velocity *= damping
        self.position += dt * self.velocity
        self.position[self.fixed] = self.fixed_position
        self.velocity[self.fixed] = 0.0
        self.steps += 1
        return {
            "reaction_n": reaction,
            "contact_penetration_m": penetration,
            "minimum_jacobian": float(np.min(jacobian)),
            "inverted_tetrahedra": int(np.count_nonzero(jacobian <= 0.0)),
            "finite": bool(
                np.isfinite(self.position).all()
                and np.isfinite(self.velocity).all()
                and np.isfinite(reaction).all()
            ),
        }

    def current_volume_m3(self) -> float:
        _, jacobian = self._deformation()
        return float(np.sum(self.rest_volume * jacobian))

    def maximum_anchor_drift_m(self) -> float:
        return float(
            np.max(np.linalg.norm(self.position[self.fixed] - self.fixed_position, axis=1))
        )


def _smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def _smoothstep_rate(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return 6.0 * value * (1.0 - value)


def _off_grid_contact_sweep(
    solver: CuttableTissueReferenceSolver,
) -> tuple[float, float]:
    geometry = solver.profile["geometry"]
    contact = solver.profile["scalpel_contact"]
    cell_width = (
        float(geometry["width_m"])
        * (1.0 + float(geometry["prestrain_x"]))
        / int(geometry["cells_x"])
    )
    surface_z = float(np.max(solver.position[:, 2]))
    edge_z = surface_z + float(contact["edge_radius_m"]) - 0.0001
    sample_count = 9
    forces: list[float] = []
    for offset in np.linspace(-cell_width / 2.0, cell_width / 2.0, sample_count):
        _, reaction, _ = solver._scalpel_contact(ScalpelPose((float(offset), 0.0, edge_z)))
        forces.append(float(np.linalg.norm(reaction)))
    coverage = sum(force > 1.0e-12 for force in forces) / sample_count
    mean_force = float(np.mean(forces))
    variation = (max(forces) - min(forces)) / max(mean_force, 1.0e-12)
    return coverage, variation


def run_intact_scalpel_qualification(
    profile: dict[str, Any] | None = None,
    *,
    profile_path: Path = DEFAULT_PROFILE_PATH,
) -> CuttableTissueReceipt:
    profile = profile or load_profile(profile_path)
    solver = CuttableTissueReferenceSolver(profile)
    off_grid_coverage, off_grid_variation = _off_grid_contact_sweep(solver)
    settings = profile["solver"]
    dt = float(settings["time_step_s"])
    surface_z = float(np.max(solver.initial_position[:, 2]))
    radius = float(profile["scalpel_contact"]["edge_radius_m"])
    clearance = float(settings["initial_clearance_m"])
    indentation = float(settings["commanded_indentation_m"])
    start_z = surface_z + radius + clearance
    contact_z = surface_z + radius - indentation

    phases = (
        ("settle", float(settings["settle_s"])),
        ("approach", float(settings["approach_s"])),
        ("hold", float(settings["hold_s"])),
        ("slide", float(settings["slide_s"])),
        ("retract", float(settings["retract_s"])),
        ("recovery", float(settings["recovery_s"])),
    )
    trace: list[tuple[float, ...]] = []
    force_samples: dict[str, list[float]] = {name: [] for name, _ in phases}
    max_penetration = 0.0
    min_jacobian = math.inf
    inverted_peak = 0
    finite = True
    peak_displacement = 0.0
    peak_tangential_force = 0.0
    elapsed = 0.0
    total_steps = 0

    for phase, duration in phases:
        steps = max(1, int(round(duration / dt)))
        for phase_step in range(steps):
            fraction = phase_step / max(steps - 1, 1)
            if phase == "approach":
                z = start_z + (contact_z - start_z) * _smoothstep(fraction)
                velocity_z = (contact_z - start_z) / duration * _smoothstep_rate(fraction)
                scalpel = ScalpelPose((0.0, 0.0, z), velocity_m_s=(0.0, 0.0, velocity_z))
            elif phase == "hold":
                scalpel = ScalpelPose((0.0, 0.0, contact_z))
            elif phase == "slide":
                slide = float(settings["commanded_slide_m"])
                y = slide * _smoothstep(fraction)
                velocity_y = slide / duration * _smoothstep_rate(fraction)
                scalpel = ScalpelPose((0.0, y, contact_z), velocity_m_s=(0.0, velocity_y, 0.0))
            elif phase == "retract":
                z = contact_z + (start_z - contact_z) * _smoothstep(fraction)
                velocity_z = (start_z - contact_z) / duration * _smoothstep_rate(fraction)
                scalpel = ScalpelPose(
                    (0.0, float(settings["commanded_slide_m"]), z),
                    velocity_m_s=(0.0, 0.0, velocity_z),
                )
            else:
                scalpel = None
            sample = solver.step(dt, scalpel)
            reaction = np.asarray(sample["reaction_n"], dtype=np.float64)
            force_magnitude = float(np.linalg.norm(reaction))
            peak_tangential_force = max(peak_tangential_force, abs(float(reaction[1])))
            force_samples[phase].append(force_magnitude)
            max_penetration = max(max_penetration, float(sample["contact_penetration_m"]))
            min_jacobian = min(min_jacobian, float(sample["minimum_jacobian"]))
            inverted_peak = max(inverted_peak, int(sample["inverted_tetrahedra"]))
            finite = finite and bool(sample["finite"])
            displacement = np.linalg.norm(
                solver.position[solver.top_nodes] - solver.initial_position[solver.top_nodes],
                axis=1,
            )
            peak_displacement = max(peak_displacement, float(np.max(displacement)))
            if total_steps % 20 == 0:
                trace.append(
                    (
                        round(elapsed, 8),
                        round(force_magnitude, 8),
                        round(float(sample["contact_penetration_m"]), 9),
                        round(float(sample["minimum_jacobian"]), 8),
                        round(float(np.max(displacement)), 9),
                    )
                )
            elapsed += dt
            total_steps += 1

    rest_volume = float(np.sum(solver.rest_volume))
    volume_error = abs(solver.current_volume_m3() - rest_volume) / rest_volume
    free = ~solver.fixed
    recovery = float(
        np.max(np.linalg.norm(solver.position[free] - solver.initial_position[free], axis=1))
    )
    hold = force_samples["hold"]
    window = max(1, len(hold) // 10)
    hold_start = float(np.mean(hold[:window]))
    hold_end = float(np.mean(hold[-window:]))
    relaxation = max(0.0, (hold_start - hold_end) / max(hold_start, 1.0e-12))
    trace_sha = hashlib.sha256(json.dumps(trace, separators=(",", ":")).encode("utf-8")).hexdigest()
    profile_sha = hashlib.sha256(
        json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    limits = profile["qualification"]
    gates = {
        "finite_state": finite,
        "no_inverted_tetrahedra": inverted_peak == 0,
        "minimum_jacobian": min_jacobian >= float(settings["minimum_jacobian"]),
        "anchor_drift": solver.maximum_anchor_drift_m() <= float(limits["maximum_anchor_drift_m"]),
        "volume_error": volume_error <= float(limits["maximum_volume_error_fraction"]),
        "contact_penetration": max_penetration <= float(limits["maximum_contact_penetration_m"]),
        "off_grid_contact_coverage": off_grid_coverage
        >= float(limits["minimum_off_grid_contact_coverage_fraction"]),
        "off_grid_force_variation": off_grid_variation
        <= float(limits["maximum_off_grid_force_variation_fraction"]),
        "force_relaxation_min": relaxation >= float(limits["minimum_force_relaxation_fraction"]),
        "force_relaxation_max": relaxation <= float(limits["maximum_force_relaxation_fraction"]),
        "recovery": recovery <= float(limits["maximum_recovery_residual_m"]),
        "fracture_fail_closed": solver.fracture_event_count == 0,
    }
    failed = tuple(name for name, passed in gates.items() if not passed)
    all_force = [value for values in force_samples.values() for value in values]
    return CuttableTissueReceipt(
        schema="dr.anmar.cuttable-tissue-intact-contact-receipt.v1",
        profile_id=str(profile["id"]),
        profile_sha256=profile_sha,
        point_count=len(solver.rest),
        tetrahedron_count=len(solver.tets),
        steps=solver.steps,
        finite=finite,
        inverted_tetrahedra_peak=inverted_peak,
        minimum_jacobian=min_jacobian,
        maximum_anchor_drift_m=solver.maximum_anchor_drift_m(),
        maximum_volume_error_fraction=volume_error,
        maximum_contact_penetration_m=max_penetration,
        off_grid_contact_coverage_fraction=off_grid_coverage,
        off_grid_force_variation_fraction=off_grid_variation,
        peak_scalpel_force_n=max(all_force),
        peak_scalpel_tangential_force_n=peak_tangential_force,
        force_at_hold_start_n=hold_start,
        force_at_hold_end_n=hold_end,
        force_relaxation_fraction=relaxation,
        peak_surface_displacement_m=peak_displacement,
        recovery_residual_m=recovery,
        fracture_enabled=bool(profile["fracture"]["enabled"]),
        fracture_event_count=solver.fracture_event_count,
        deterministic_trace_sha256=trace_sha,
        qualified=not failed,
        failed_gates=failed,
        evidence_boundary=str(profile["evidence_boundary"]["claims"]),
    )


__all__ = [
    "CuttableTissueReceipt",
    "CuttableTissueReferenceSolver",
    "DEFAULT_PROFILE_PATH",
    "ScalpelPose",
    "build_regular_tetrahedral_coupon",
    "load_profile",
    "run_intact_scalpel_qualification",
]
