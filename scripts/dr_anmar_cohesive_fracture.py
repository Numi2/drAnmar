#!/usr/bin/env python3
"""Cohesive-fracture and blade-front reference mechanics for DrAnmar tissue.

This module is a deterministic constitutive/topology oracle. It makes every
internal tetrahedral face eligible for irreversible mixed-mode separation and
seeds only a surface-connected front intersected by the swept finite blade.
The production dynamic solver remains fail-closed until discontinuous degrees
of freedom, two-sided collision, CUDA parity, and specimen calibration pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dr_anmar_cuttable_tissue_solver import (
    DEFAULT_PROFILE_PATH,
    ScalpelPose,
    build_regular_tetrahedral_coupon,
    load_profile,
)


@dataclass
class CohesiveState:
    maximum_effective_separation_m: float = 0.0
    damage: float = 0.0
    dissipated_energy_j_m2: float = 0.0
    seeded: bool = False
    failed: bool = False


@dataclass(frozen=True)
class CohesiveResponse:
    traction_on_positive_face_pa: tuple[float, float, float]
    damage: float
    effective_separation_m: float
    initiation_separation_m: float
    final_separation_m: float
    mixed_mode_fracture_energy_j_m2: float
    mode_ii_fraction: float


@dataclass(frozen=True)
class CohesiveInterface:
    index: int
    nodes: tuple[int, int, int]
    tetrahedra: tuple[int, int]
    centroid_m: tuple[float, float, float]
    normal: tuple[float, float, float]
    area_m2: float
    touches_top: bool


@dataclass(frozen=True)
class CohesiveFractureReceipt:
    schema: str
    profile_id: str
    profile_sha256: str
    tetrahedron_count: int
    internal_interface_count: int
    eligible_interface_fraction: float
    interface_graph_connected: bool
    mode_i_energy_relative_error: float
    mode_ii_energy_relative_error: float
    mixed_mode_energy_relative_error: float
    maximum_damage_regression: float
    unseeded_damage: float
    post_failure_compression_traction_pa: float
    rate_strengthening_traction_pa: float
    off_grid_blade_seed_coverage_fraction: float
    maximum_remote_seed_events: int
    buried_sweep_seed_events: int
    buried_sweep_candidates_rejected: int
    stationary_overlap_seed_events: int
    deterministic_trace_sha256: str
    dynamic_solver_fracture_enabled: bool
    qualified: bool
    failed_gates: tuple[str, ...]
    biomechanical_validation: bool
    clinical_validation: bool

    def payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failed_gates"] = list(self.failed_gates)
        return payload


class MixedModeCohesiveLaw:
    """Finite-separation bilinear cohesive law with irreversible damage."""

    def __init__(self, profile: dict[str, Any]):
        fracture = profile["fracture"]
        self.penalty = float(fracture["penalty_stiffness_pa_m"])
        self.compression_penalty = float(fracture["compression_stiffness_pa_m"])
        self.normal_peak = float(fracture["mode_i_peak_traction_pa"])
        self.shear_peak = float(fracture["mode_ii_peak_traction_pa"])
        self.mode_i_energy = float(fracture["mode_i_fracture_energy_j_m2"])
        self.mode_ii_energy = float(fracture["mode_ii_fracture_energy_j_m2"])
        self.bk_exponent = float(fracture["benzeggagh_kenane_exponent"])
        self.viscosity = float(fracture["cohesive_viscosity_pa_s_m"])
        if (
            min(
                self.penalty,
                self.compression_penalty,
                self.normal_peak,
                self.shear_peak,
                self.mode_i_energy,
                self.mode_ii_energy,
            )
            <= 0.0
        ):
            raise ValueError("Cohesive stiffness, strength, and energy must be positive")

    def envelope(
        self,
        opening_m: float,
        shear_m: float,
    ) -> tuple[float, float, float, float]:
        effective = math.hypot(max(opening_m, 0.0), shear_m)
        if effective <= 1.0e-15:
            return (
                self.normal_peak / self.penalty,
                2.0 * self.mode_i_energy / self.normal_peak,
                0.0,
                self.mode_i_energy,
            )
        normal_fraction = max(opening_m, 0.0) / effective
        shear_fraction = shear_m / effective
        peak = 1.0 / math.sqrt(
            (normal_fraction / self.normal_peak) ** 2 + (shear_fraction / self.shear_peak) ** 2
        )
        mode_ii_fraction = shear_fraction * shear_fraction
        fracture_energy = (
            self.mode_i_energy
            + (self.mode_ii_energy - self.mode_i_energy) * mode_ii_fraction**self.bk_exponent
        )
        initiation = peak / self.penalty
        final = 2.0 * fracture_energy / peak
        if final <= initiation:
            raise ValueError("Cohesive fracture energy must exceed elastic initiation energy")
        return initiation, final, mode_ii_fraction, fracture_energy

    def evaluate(
        self,
        jump_m: np.ndarray,
        normal: np.ndarray,
        relative_velocity_m_s: np.ndarray,
        dt: float,
        state: CohesiveState,
        *,
        seeded: bool,
    ) -> CohesiveResponse:
        normal = np.asarray(normal, dtype=np.float64)
        normal /= max(float(np.linalg.norm(normal)), 1.0e-15)
        jump = np.asarray(jump_m, dtype=np.float64)
        velocity = np.asarray(relative_velocity_m_s, dtype=np.float64)
        signed_normal = float(np.dot(jump, normal))
        opening = max(signed_normal, 0.0)
        shear_vector = jump - signed_normal * normal
        shear = float(np.linalg.norm(shear_vector))
        effective = math.hypot(opening, shear)
        initiation, final, mode_ii_fraction, fracture_energy = self.envelope(opening, shear)

        previous_damage = state.damage
        state.seeded = state.seeded or seeded
        state.maximum_effective_separation_m = max(state.maximum_effective_separation_m, effective)
        if state.seeded and state.maximum_effective_separation_m > initiation:
            maximum = state.maximum_effective_separation_m
            target_damage = final * (maximum - initiation) / (maximum * (final - initiation))
            state.damage = max(state.damage, min(1.0, target_damage))
        state.failed = state.damage >= 1.0 - 1.0e-12
        damage_increment = state.damage - previous_damage
        if damage_increment > 0.0:
            state.dissipated_energy_j_m2 += (
                0.5 * self.penalty * effective * effective * damage_increment
            )

        traction = np.zeros(3, dtype=np.float64)
        if signed_normal < 0.0:
            traction += -self.compression_penalty * signed_normal * normal
        if effective > 1.0e-15 and not state.failed:
            direction = (opening * normal + shear_vector) / effective
            cohesive_magnitude = (1.0 - state.damage) * self.penalty * effective
            effective_rate = float(np.dot(velocity, direction))
            viscous_magnitude = (1.0 - state.damage) * self.viscosity * max(effective_rate, 0.0)
            traction -= (cohesive_magnitude + viscous_magnitude) * direction
        return CohesiveResponse(
            traction_on_positive_face_pa=tuple(float(value) for value in traction),
            damage=state.damage,
            effective_separation_m=effective,
            initiation_separation_m=initiation,
            final_separation_m=final,
            mixed_mode_fracture_energy_j_m2=fracture_energy,
            mode_ii_fraction=mode_ii_fraction,
        )


def build_cohesive_interfaces(
    points: np.ndarray,
    tetrahedra: np.ndarray,
) -> tuple[list[CohesiveInterface], list[set[int]]]:
    owners: dict[tuple[int, int, int], list[int]] = {}
    for tetrahedron_index, tetrahedron in enumerate(tetrahedra):
        a, b, c, d = (int(value) for value in tetrahedron)
        for face in ((b, c, d), (a, d, c), (a, b, d), (a, c, b)):
            owners.setdefault(tuple(sorted(face)), []).append(tetrahedron_index)
    top_z = float(np.max(points[:, 2]))
    interfaces: list[CohesiveInterface] = []
    for nodes, adjacent in sorted(owners.items()):
        if len(adjacent) == 1:
            continue
        if len(adjacent) != 2:
            raise ValueError("Non-manifold tetrahedral face cannot form one cohesive interface")
        triangle = points[list(nodes)]
        cross = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        area = 0.5 * float(np.linalg.norm(cross))
        if area <= 1.0e-16:
            raise ValueError("Degenerate cohesive interface")
        normal = cross / (2.0 * area)
        interfaces.append(
            CohesiveInterface(
                index=len(interfaces),
                nodes=nodes,
                tetrahedra=(adjacent[0], adjacent[1]),
                centroid_m=tuple(float(value) for value in np.mean(triangle, axis=0)),
                normal=tuple(float(value) for value in normal),
                area_m2=area,
                touches_top=bool(np.any(np.isclose(triangle[:, 2], top_z, atol=1.0e-12))),
            )
        )
    edge_owners: dict[tuple[int, int], list[int]] = {}
    for interface in interfaces:
        a, b, c = interface.nodes
        for edge in ((a, b), (b, c), (c, a)):
            edge_owners.setdefault(tuple(sorted(edge)), []).append(interface.index)
    adjacency = [set() for _ in interfaces]
    for incident in edge_owners.values():
        for left in incident:
            adjacency[left].update(right for right in incident if right != left)
    return interfaces, adjacency


def _point_triangle_distance(point: np.ndarray, triangle: np.ndarray) -> float:
    a, b, c = triangle
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.linalg.norm(ap))
    bp = point - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return float(np.linalg.norm(bp))
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return float(np.linalg.norm(ap - (d1 / (d1 - d3)) * ab))
    cp = point - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return float(np.linalg.norm(cp))
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return float(np.linalg.norm(ap - (d2 / (d2 - d6)) * ac))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        edge = c - b
        return float(np.linalg.norm(bp - ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * edge))
    normal = np.cross(ab, ac)
    return abs(float(np.dot(ap, normal))) / max(float(np.linalg.norm(normal)), 1.0e-15)


def _segment_segment_distance(
    start_a: np.ndarray,
    end_a: np.ndarray,
    start_b: np.ndarray,
    end_b: np.ndarray,
) -> float:
    direction_a = end_a - start_a
    direction_b = end_b - start_b
    offset = start_a - start_b
    aa = float(np.dot(direction_a, direction_a))
    bb = float(np.dot(direction_b, direction_b))
    ab = float(np.dot(direction_a, direction_b))
    ao = float(np.dot(direction_a, offset))
    bo = float(np.dot(direction_b, offset))
    if aa <= 1.0e-20 and bb <= 1.0e-20:
        return float(np.linalg.norm(start_a - start_b))
    if aa <= 1.0e-20:
        parameter_a = 0.0
        parameter_b = np.clip(bo / bb, 0.0, 1.0)
    elif bb <= 1.0e-20:
        parameter_b = 0.0
        parameter_a = np.clip(-ao / aa, 0.0, 1.0)
    else:
        denominator = aa * bb - ab * ab
        parameter_a = (
            np.clip((ab * bo - ao * bb) / denominator, 0.0, 1.0) if denominator > 1.0e-20 else 0.0
        )
        parameter_b = (ab * parameter_a + bo) / bb
        if parameter_b < 0.0:
            parameter_b = 0.0
            parameter_a = np.clip(-ao / aa, 0.0, 1.0)
        elif parameter_b > 1.0:
            parameter_b = 1.0
            parameter_a = np.clip((ab - ao) / aa, 0.0, 1.0)
    closest_a = start_a + parameter_a * direction_a
    closest_b = start_b + parameter_b * direction_b
    return float(np.linalg.norm(closest_a - closest_b))


def _triangle_triangle_distance(left: np.ndarray, right: np.ndarray) -> float:
    distances = [_point_triangle_distance(vertex, right) for vertex in left]
    distances.extend(_point_triangle_distance(vertex, left) for vertex in right)
    left_edges = ((0, 1), (1, 2), (2, 0))
    right_edges = ((0, 1), (1, 2), (2, 0))
    distances.extend(
        _segment_segment_distance(left[a], left[b], right[c], right[d])
        for a, b in left_edges
        for c, d in right_edges
    )
    return min(distances)


def swept_blade_triangles(
    start: ScalpelPose,
    end: ScalpelPose,
    edge_length_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    start_center = np.asarray(start.center_m, dtype=np.float64)
    end_center = np.asarray(end.center_m, dtype=np.float64)
    start_tangent = np.asarray(start.tangent, dtype=np.float64)
    end_tangent = np.asarray(end.tangent, dtype=np.float64)
    start_tangent /= max(float(np.linalg.norm(start_tangent)), 1.0e-15)
    end_tangent /= max(float(np.linalg.norm(end_tangent)), 1.0e-15)
    half = edge_length_m / 2.0
    start_left, start_right = (
        start_center - half * start_tangent,
        start_center + half * start_tangent,
    )
    end_left, end_right = end_center - half * end_tangent, end_center + half * end_tangent
    return (
        np.asarray((start_left, start_right, end_right)),
        np.asarray((start_left, end_right, end_left)),
    )


class BladeSeededCohesiveFront:
    def __init__(self, points: np.ndarray, tetrahedra: np.ndarray, profile: dict[str, Any]):
        self.points = points
        self.interfaces, self.adjacency = build_cohesive_interfaces(points, tetrahedra)
        self.seeded: set[int] = set()
        self.seed_distance = float(profile["fracture"]["blade_seed_distance_m"])
        self.minimum_seed_speed = float(profile["fracture"]["minimum_seeded_separation_rate_m_s"])
        self.edge_length = float(profile["scalpel_contact"]["edge_length_m"])

    def seed_sweep(self, start: ScalpelPose, end: ScalpelPose) -> tuple[set[int], int]:
        commanded_speed = max(
            float(np.linalg.norm(np.asarray(start.velocity_m_s, dtype=np.float64))),
            float(np.linalg.norm(np.asarray(end.velocity_m_s, dtype=np.float64))),
        )
        if commanded_speed < self.minimum_seed_speed:
            return set(), 0
        swept = swept_blade_triangles(start, end, self.edge_length)
        candidates: set[int] = set()
        for interface in self.interfaces:
            triangle = self.points[list(interface.nodes)]
            distance = min(_triangle_triangle_distance(triangle, blade) for blade in swept)
            if distance <= self.seed_distance:
                candidates.add(interface.index)
        if self.seeded:
            connected = {
                candidate
                for candidate in candidates
                if self.adjacency[candidate].intersection(self.seeded) or candidate in self.seeded
            }
        else:
            connected = {
                candidate for candidate in candidates if self.interfaces[candidate].touches_top
            }
        remote = len(candidates - connected)
        new = connected - self.seeded
        self.seeded.update(connected)
        return new, remote


def _interface_graph_connected(adjacency: list[set[int]]) -> bool:
    if not adjacency:
        return False
    visited = {0}
    frontier = [0]
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency[current] - visited:
            visited.add(neighbor)
            frontier.append(neighbor)
    return len(visited) == len(adjacency)


def _monotonic_work(
    law: MixedModeCohesiveLaw,
    direction: np.ndarray,
    *,
    samples: int = 8193,
) -> tuple[float, float, list[tuple[float, float, float]]]:
    direction = direction / np.linalg.norm(direction)
    normal = np.asarray((0.0, 0.0, 1.0))
    opening_fraction = max(float(direction[2]), 0.0)
    shear_fraction = float(np.linalg.norm(direction[:2]))
    _, final, _, expected = law.envelope(opening_fraction, shear_fraction)
    separations = np.linspace(0.0, final, samples)
    state = CohesiveState()
    resistance: list[float] = []
    trace: list[tuple[float, float, float]] = []
    for separation in separations:
        response = law.evaluate(
            separation * direction,
            normal,
            np.zeros(3),
            1.0,
            state,
            seeded=True,
        )
        traction = np.asarray(response.traction_on_positive_face_pa)
        value = max(0.0, -float(np.dot(traction, direction)))
        resistance.append(value)
        if len(resistance) % 128 == 1:
            trace.append((float(separation), value, state.damage))
    work = float(np.trapezoid(np.asarray(resistance), separations))
    return work, expected, trace


def run_cohesive_fracture_qualification(
    profile: dict[str, Any] | None = None,
    *,
    profile_path: Path = DEFAULT_PROFILE_PATH,
) -> CohesiveFractureReceipt:
    profile = profile or load_profile(profile_path)
    points, tetrahedra = build_regular_tetrahedral_coupon(profile)
    interfaces, adjacency = build_cohesive_interfaces(points, tetrahedra)
    law = MixedModeCohesiveLaw(profile)
    traces: list[tuple[float, float, float]] = []
    energy_errors: list[float] = []
    for direction in (
        np.asarray((0.0, 0.0, 1.0)),
        np.asarray((1.0, 0.0, 0.0)),
        np.asarray((math.sqrt(0.5), 0.0, math.sqrt(0.5))),
    ):
        work, expected, trace = _monotonic_work(law, direction)
        energy_errors.append(abs(work - expected) / expected)
        traces.extend(trace)

    state = CohesiveState()
    normal = np.asarray((0.0, 0.0, 1.0))
    _, final, _, _ = law.envelope(1.0, 0.0)
    damage_samples: list[float] = []
    for separation in (*np.linspace(0.0, 0.8 * final, 128), *np.linspace(0.8 * final, 0.0, 128)):
        law.evaluate(
            np.asarray((0.0, 0.0, separation)),
            normal,
            np.zeros(3),
            1.0,
            state,
            seeded=True,
        )
        damage_samples.append(state.damage)
    damage_regression = max(
        0.0,
        max(
            (damage_samples[index] - damage_samples[index + 1])
            for index in range(len(damage_samples) - 1)
        ),
    )

    unseeded = CohesiveState()
    law.evaluate(
        np.asarray((0.0, 0.0, 1.1 * final)),
        normal,
        np.zeros(3),
        1.0,
        unseeded,
        seeded=False,
    )
    failed = CohesiveState()
    law.evaluate(
        np.asarray((0.0, 0.0, 1.1 * final)),
        normal,
        np.zeros(3),
        1.0,
        failed,
        seeded=True,
    )
    compression = law.evaluate(
        np.asarray((0.0, 0.0, -1.0e-5)),
        normal,
        np.zeros(3),
        1.0,
        failed,
        seeded=True,
    )
    compression_traction = float(
        np.dot(np.asarray(compression.traction_on_positive_face_pa), normal)
    )

    rate_state = CohesiveState()
    initiation, _, _, _ = law.envelope(1.0, 0.0)
    quasistatic = law.evaluate(
        np.asarray((0.0, 0.0, 0.5 * initiation)),
        normal,
        np.zeros(3),
        1.0e-4,
        rate_state,
        seeded=True,
    )
    dynamic = law.evaluate(
        np.asarray((0.0, 0.0, 0.5 * initiation)),
        normal,
        np.asarray((0.0, 0.0, 0.01)),
        1.0e-4,
        CohesiveState(),
        seeded=True,
    )
    rate_strengthening = max(
        0.0,
        -float(np.dot(np.asarray(dynamic.traction_on_positive_face_pa), normal))
        + float(np.dot(np.asarray(quasistatic.traction_on_positive_face_pa), normal)),
    )

    geometry = profile["geometry"]
    half_width = float(geometry["width_m"]) / 2.0
    attachment = float(geometry["attachment_band_m"])
    top_z = float(np.max(points[:, 2]))
    offsets = np.linspace(-half_width + 1.5 * attachment, half_width - 1.5 * attachment, 9)
    seeded_counts: list[int] = []
    remote_counts: list[int] = []
    for offset in offsets:
        front = BladeSeededCohesiveFront(points, tetrahedra, profile)
        new, remote = front.seed_sweep(
            ScalpelPose((float(offset), 0.0, top_z + 0.001)),
            ScalpelPose((float(offset), 0.0, top_z - 0.0015), velocity_m_s=(0.0, 0.0, -0.005)),
        )
        seeded_counts.append(len(new))
        remote_counts.append(remote)
    coverage = sum(count > 0 for count in seeded_counts) / len(seeded_counts)
    buried_front = BladeSeededCohesiveFront(points, tetrahedra, profile)
    buried_seeded, buried_rejected = buried_front.seed_sweep(
        ScalpelPose((0.0, 0.0, top_z - 0.004)),
        ScalpelPose((0.0, 0.0, top_z - 0.006), velocity_m_s=(0.0, 0.0, -0.005)),
    )
    stationary_front = BladeSeededCohesiveFront(points, tetrahedra, profile)
    stationary_seeded, _ = stationary_front.seed_sweep(
        ScalpelPose((0.0, 0.0, top_z + 0.001)),
        ScalpelPose((0.0, 0.0, top_z - 0.0015)),
    )
    trace_sha = hashlib.sha256(
        json.dumps(traces, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    profile_sha = hashlib.sha256(
        json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    limits = profile["fracture"]["qualification"]
    gates = {
        "all_internal_faces_eligible": bool(profile["fracture"]["all_internal_faces_eligible"]),
        "interface_graph_connected": _interface_graph_connected(adjacency),
        "mode_i_energy": energy_errors[0] <= float(limits["maximum_energy_relative_error"]),
        "mode_ii_energy": energy_errors[1] <= float(limits["maximum_energy_relative_error"]),
        "mixed_mode_energy": energy_errors[2] <= float(limits["maximum_energy_relative_error"]),
        "damage_irreversible": damage_regression <= float(limits["maximum_damage_regression"]),
        "unseeded_damage_blocked": unseeded.damage == 0.0,
        "post_failure_compression": compression_traction
        >= float(limits["minimum_post_failure_compression_traction_pa"]),
        "rate_strengthening": rate_strengthening > 0.0,
        "off_grid_blade_seed_coverage": coverage
        >= float(limits["minimum_off_grid_blade_seed_coverage_fraction"]),
        "remote_seed_blocked": max(remote_counts, default=0) == 0,
        "buried_sweep_blocked": not buried_seeded and buried_rejected > 0,
        "stationary_overlap_blocked": not stationary_seeded,
        "dynamic_solver_fail_closed": not bool(profile["fracture"]["enabled"]),
    }
    failed_gates = tuple(name for name, passed in gates.items() if not passed)
    return CohesiveFractureReceipt(
        schema="dr.anmar.cohesive-fracture-reference-receipt.v1",
        profile_id=str(profile["id"]),
        profile_sha256=profile_sha,
        tetrahedron_count=len(tetrahedra),
        internal_interface_count=len(interfaces),
        eligible_interface_fraction=1.0,
        interface_graph_connected=_interface_graph_connected(adjacency),
        mode_i_energy_relative_error=energy_errors[0],
        mode_ii_energy_relative_error=energy_errors[1],
        mixed_mode_energy_relative_error=energy_errors[2],
        maximum_damage_regression=damage_regression,
        unseeded_damage=unseeded.damage,
        post_failure_compression_traction_pa=compression_traction,
        rate_strengthening_traction_pa=rate_strengthening,
        off_grid_blade_seed_coverage_fraction=coverage,
        maximum_remote_seed_events=max(remote_counts, default=0),
        buried_sweep_seed_events=len(buried_seeded),
        buried_sweep_candidates_rejected=buried_rejected,
        stationary_overlap_seed_events=len(stationary_seeded),
        deterministic_trace_sha256=trace_sha,
        dynamic_solver_fracture_enabled=bool(profile["fracture"]["enabled"]),
        qualified=not failed_gates,
        failed_gates=failed_gates,
        biomechanical_validation=False,
        clinical_validation=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = run_cohesive_fracture_qualification(
        load_profile(args.profile), profile_path=args.profile
    )
    encoded = json.dumps(receipt.payload(), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if receipt.qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
