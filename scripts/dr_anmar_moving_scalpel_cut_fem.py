#!/usr/bin/env python3
"""Moving-scalpel cohesive front coupled to live embedded FEM release.

The embedded discontinuity is latent: duplicated nodes remain fully tied until
the finite blade sweep supplies the mixed-mode fracture energy for their cut
cells. Accepted cells release irreversibly, acquire unilateral compression,
and expose two deforming collision sheets. Slow cutting is solved quasi-
statically by dynamic relaxation; pseudo-time is not reported as physical time.
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

from dr_anmar_cuttable_tissue_solver import ScalpelPose, load_profile
from dr_anmar_dynamic_curved_cut_fem import (
    _build_settled_mesh,
    _level_gradient,
    _probe_collision,
    load_curved_profile,
)
from dr_anmar_persistent_cut_topology import (
    PersistentCutCellField,
    WorkChannels,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOVING_PROFILE_PATH = (
    REPOSITORY_ROOT / "physics_next/tissues/dr-anmar-moving-scalpel-cut-v1.json"
)


@dataclass(frozen=True)
class MovingScalpelCutReceipt:
    schema: str
    profile_id: str
    profile_sha256: str
    base_profile_sha256: str
    embedded_profile_sha256: str
    path_segment_count: int
    pseudo_dynamic_step_count: int
    fracture_event_count: int
    released_pair_count: int
    total_pair_count: int
    released_pair_fraction: float
    monotonic_release: bool
    maximum_release_ahead_of_blade_m: float
    first_release_segment: int
    last_release_segment: int
    repeated_path_additional_events: int
    crossing_path_additional_events: int
    intersection_cell_count: int
    subcritical_fracture_events: int
    stationary_fracture_events: int
    fracture_work_j: float
    adhesion_work_j: float
    wear_work_j: float
    viscous_work_j: float
    friction_work_j: float
    peak_cutting_force_n: float
    finite: bool
    inversion_observation_count: int
    minimum_jacobian: float
    mass_relative_error: float
    mean_wound_gap_m: float
    maximum_wound_gap_m: float
    positive_wound_area_m2: float
    negative_wound_area_m2: float
    opposed_area_relative_error: float
    two_sided_collision_coverage_fraction: float
    maximum_probe_surface_crossing_m: float
    event_trace_sha256: str
    deterministic_event_replay: bool
    qualified: bool
    failed_gates: tuple[str, ...]
    real_time_transient: bool
    biomechanical_validation: bool
    clinical_validation: bool

    def payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failed_gates"] = list(self.failed_gates)
        return payload


def load_moving_profile(path: Path = DEFAULT_MOVING_PROFILE_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _curve_y(x: np.ndarray | float, curved: dict[str, Any]) -> np.ndarray | float:
    cut = curved["implicit_cut"]
    return float(cut["center_y_m"]) + float(cut["amplitude_y_m"]) * np.sin(
        math.pi * np.asarray(x) / float(cut["wavelength_x_m"])
    )


def _path_poses(moving: dict[str, Any], curved: dict[str, Any]) -> list[ScalpelPose]:
    path = moving["path"]
    x_values = np.linspace(float(path["start_x_m"]), float(path["end_x_m"]), int(path["segments"]) + 1)
    centers = np.column_stack((x_values, _curve_y(x_values, curved), np.zeros_like(x_values)))
    tangent = tuple(float(value) for value in path["blade_tangent"])
    speed = float(path["command_speed_m_s"])
    poses: list[ScalpelPose] = []
    for index, center in enumerate(centers):
        if index == 0:
            direction = centers[1] - centers[0]
        else:
            direction = centers[index] - centers[index - 1]
        direction /= np.linalg.norm(direction)
        velocity = tuple(float(value) for value in speed * direction)
        poses.append(ScalpelPose(tuple(float(value) for value in center), tangent=tangent, velocity_m_s=velocity))
    return poses


def _crossing_poses(moving: dict[str, Any]) -> list[ScalpelPose]:
    path = moving["path"]
    y_values = np.linspace(-0.010, 0.010, int(path["segments"]) + 1)
    speed = float(path["command_speed_m_s"])
    tangent = tuple(float(value) for value in path["blade_tangent"])
    return [
        ScalpelPose((0.0, float(y), 0.0), tangent=tangent, velocity_m_s=(0.0, speed, 0.0))
        for y in y_values
    ]


def _work_channels(base: dict[str, Any], moving: dict[str, Any], *, ratio: float | None = None) -> WorkChannels:
    path = moving["path"]
    critical = float(base["fracture"]["mode_i_fracture_energy_j_m2"])
    return WorkChannels(
        fracture_j_m2=(float(path["fracture_work_ratio"]) if ratio is None else ratio) * critical,
        adhesion_j_m2=float(path["adhesion_energy_j_m2"]),
        wear_j_m2=float(path["wear_energy_j_m2"]),
        viscous_j_m2=float(path["viscous_energy_j_m2"]),
        friction_j_m2=float(path["friction_energy_j_m2"]),
    )


class MovingScalpelCutFEM:
    def __init__(self, base: dict[str, Any], curved: dict[str, Any], moving: dict[str, Any]):
        self.base = base
        self.curved = curved
        self.moving = moving
        self.mesh = _build_settled_mesh(base, curved)
        self.position = self.mesh.position
        self.velocity = self.mesh.velocity
        self.released = np.zeros(len(self.mesh.gap_plus_nodes), dtype=bool)
        self.node_to_pair = {
            int(node): pair
            for pair, nodes in enumerate(zip(self.mesh.gap_plus_nodes, self.mesh.gap_minus_nodes, strict=True))
            for node in nodes
        }
        self.field = PersistentCutCellField(base)
        self.release_history: list[int] = [0]
        self.release_ahead_history: list[float] = []
        self.event_trace: list[tuple[int, int, int, str]] = []
        self.minimum_jacobian = math.inf
        self.inversion_observations = 0
        self.finite = True
        self.steps = 0

    def _cell_key(self, point: np.ndarray) -> tuple[int, int, int]:
        index = np.floor((point - self.field.minimum) / self.field.cell_size).astype(int)
        index = np.clip(index, 0, self.field.counts - 1)
        return tuple(int(value) for value in index)

    def _sync_release_from_field(self, blade_x: float) -> int:
        before = int(np.count_nonzero(self.released))
        for pair, point in enumerate(self.mesh.gap_rest_points):
            cell = self.field.cells.get(self._cell_key(point))
            if cell is not None and any(patch.fractured for patch in cell.patches):
                self.released[pair] = True
        newly_released = int(np.count_nonzero(self.released)) - before
        released_points = self.mesh.gap_rest_points[self.released]
        if len(released_points):
            self.release_ahead_history.append(max(0.0, float(np.max(released_points[:, 0]) - blade_x)))
        self.release_history.append(int(np.count_nonzero(self.released)))
        return newly_released

    def advance_blade(self, segment: int, start: ScalpelPose, end: ScalpelPose, work: WorkChannels) -> int:
        new_patches = self.field.apply_sweep(start, end, work)
        newly_released = self._sync_release_from_field(float(end.center_m[0]))
        self.event_trace.append((segment, len(new_patches), newly_released, self.field.topology_sha256()))
        return newly_released

    def _deformation(self) -> tuple[np.ndarray, np.ndarray]:
        local = self.position[self.mesh.tetrahedra]
        ds = np.stack((local[:, 1] - local[:, 0], local[:, 2] - local[:, 0], local[:, 3] - local[:, 0]), axis=2)
        deformation = np.einsum("tij,tjk->tik", ds, self.mesh.dm_inverse)
        return deformation, np.linalg.det(deformation)

    def step(self, dt: float, blade_center: np.ndarray | None = None) -> None:
        material = self.base["material"]
        fracture = self.base["fracture"]
        solver_cfg = self.moving["quasi_static_solver"]
        youngs = float(material["youngs_modulus_pa"])
        poisson = float(material["poisson_ratio"])
        mu = youngs / (2.0 * (1.0 + poisson))
        lam = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        deformation, jacobian = self._deformation()
        inverse_transpose = np.swapaxes(np.linalg.inv(deformation), 1, 2)
        elastic = mu * (deformation - inverse_transpose) + lam * np.log(
            np.maximum(jacobian, 1.0e-9)
        )[:, None, None] * inverse_transpose
        fraction = float(material["prony_relaxation_fraction"])
        decay = math.exp(-dt / float(material["prony_time_constant_s"]))
        self.mesh.prony_history = decay * self.mesh.prony_history + fraction * (
            elastic - self.mesh.previous_elastic_stress
        )
        self.mesh.previous_elastic_stress = elastic
        stress = (1.0 - fraction) * elastic + self.mesh.prony_history
        local_force = -self.mesh.rest_volume[:, None, None] * np.einsum(
            "tij,tkj->tki", stress, self.mesh.shape_gradients
        )
        force = np.zeros_like(self.position)
        for local in range(4):
            np.add.at(force, self.mesh.tetrahedra[:, local], local_force[:, local])

        plus = self.mesh.gap_plus_nodes
        minus = self.mesh.gap_minus_nodes
        jump = self.position[plus] - self.position[minus]
        relative_velocity = self.velocity[plus] - self.velocity[minus]
        pair_force = np.zeros_like(jump)
        tied = ~self.released
        penalty = float(fracture["penalty_stiffness_pa_m"])
        viscosity = float(fracture["cohesive_viscosity_pa_s_m"])
        pair_force[tied] = self.mesh.gap_area[tied, None] * (
            -penalty * jump[tied] - viscosity * relative_velocity[tied]
        )
        if np.any(self.released):
            normal = self.mesh.gap_normals[self.released]
            normal_gap = np.sum(jump[self.released] * normal, axis=1)
            normal_speed = np.sum(relative_velocity[self.released] * normal, axis=1)
            compression = np.where(
                normal_gap < 0.0,
                -float(fracture["compression_stiffness_pa_m"]) * normal_gap
                - viscosity * np.minimum(normal_speed, 0.0),
                0.0,
            )
            pair_force[self.released] = (
                self.mesh.gap_area[self.released, None] * compression[:, None] * normal
            )
        np.add.at(force, plus, pair_force)
        np.add.at(force, minus, -pair_force)

        if blade_center is not None and np.any(self.released):
            released_indices = np.flatnonzero(self.released)
            relative = self.mesh.gap_rest_points[released_indices, :2] - blade_center[None, :2]
            near = np.linalg.norm(relative, axis=1) <= float(solver_cfg["blade_wedge_half_width_m"])
            active = released_indices[near]
            if len(active):
                current_gap = np.sum(jump[active] * self.mesh.gap_normals[active], axis=1)
                traction = np.minimum(
                    float(solver_cfg["blade_wedge_peak_traction_pa"]),
                    float(solver_cfg["blade_wedge_stiffness_pa_m"])
                    * np.maximum(float(solver_cfg["blade_wedge_target_gap_m"]) - current_gap, 0.0),
                )
                wedge = traction[:, None] * self.mesh.gap_area[active, None] * self.mesh.gap_normals[active]
                np.add.at(force, plus[active], wedge)
                np.add.at(force, minus[active], -wedge)

        damping = math.exp(-float(solver_cfg["velocity_damping_per_s"]) * dt)
        self.velocity += dt * force / self.mesh.mass[:, None]
        self.velocity *= damping
        self.position += dt * self.velocity
        self.position[self.mesh.fixed] = self.mesh.fixed_position
        self.velocity[self.mesh.fixed] = 0.0
        self.steps += 1
        self.minimum_jacobian = min(self.minimum_jacobian, float(np.min(jacobian)))
        self.inversion_observations += int(np.count_nonzero(jacobian <= 0.0))
        self.finite = self.finite and bool(
            np.isfinite(self.position).all() and np.isfinite(self.velocity).all() and np.isfinite(force).all()
        )

    def released_wound_mesh(self):
        original = self.mesh.wound_triangles_by_side
        filtered: dict[int, np.ndarray] = {}
        for side, triangles in original.items():
            mask = []
            for triangle in triangles:
                pairs = [self.node_to_pair[int(node)] for node in triangle]
                mask.append(all(self.released[pair] for pair in pairs))
            filtered[side] = triangles[np.asarray(mask, dtype=bool)]
        saved = self.mesh.wound_triangles_by_side
        self.mesh.wound_triangles_by_side = filtered
        try:
            return self.mesh.wound_surface_mesh()
        finally:
            self.mesh.wound_triangles_by_side = saved


def _run_event_topology(base: dict[str, Any], moving: dict[str, Any], curved: dict[str, Any]):
    poses = _path_poses(moving, curved)
    field = PersistentCutCellField(base)
    work = _work_channels(base, moving)
    trace = []
    for index, (start, end) in enumerate(zip(poses[:-1], poses[1:], strict=True)):
        new = field.apply_sweep(start, end, work)
        trace.append((index, len(new), field.fracture_event_count, field.topology_sha256()))
    curved_events = field.fracture_event_count
    for start, end in zip(poses[:-1], poses[1:], strict=True):
        field.apply_sweep(start, end, work)
    repeated = field.fracture_event_count - curved_events
    before_crossing = field.fracture_event_count
    crossing = _crossing_poses(moving)
    for start, end in zip(crossing[:-1], crossing[1:], strict=True):
        field.apply_sweep(start, end, work)
    crossing_events = field.fracture_event_count - before_crossing
    return field, trace, repeated, crossing_events


def run_moving_scalpel_qualification(
    moving_profile: dict[str, Any] | None = None,
    *,
    moving_profile_path: Path = DEFAULT_MOVING_PROFILE_PATH,
) -> MovingScalpelCutReceipt:
    moving = moving_profile or load_moving_profile(moving_profile_path)
    base = load_profile(REPOSITORY_ROOT / moving["base_profile"])
    curved = load_curved_profile(REPOSITORY_ROOT / moving["embedded_profile"])
    base_sha = hashlib.sha256(json.dumps(base, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if base_sha != moving["base_profile_sha256"]:
        raise ValueError("Moving-scalpel profile is not bound to the qualified base tissue")
    curved_sha = hashlib.sha256(json.dumps(curved, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    solver = MovingScalpelCutFEM(base, curved, moving)
    poses = _path_poses(moving, curved)
    work = _work_channels(base, moving)
    dt = float(moving["quasi_static_solver"]["pseudo_time_step_s"])
    relax_steps = int(moving["quasi_static_solver"]["relaxation_steps_per_segment"])
    first_release = -1
    last_release = -1
    for segment, (start, end) in enumerate(zip(poses[:-1], poses[1:], strict=True)):
        released = solver.advance_blade(segment, start, end, work)
        if released:
            first_release = segment if first_release < 0 else first_release
            last_release = segment
        blade_center = np.asarray(end.center_m, dtype=np.float64)
        for _ in range(relax_steps):
            solver.step(dt, blade_center)
    for _ in range(int(moving["quasi_static_solver"]["post_cut_relaxation_steps"])):
        solver.step(dt)

    event_field, event_trace, repeated, crossing_events = _run_event_topology(base, moving, curved)
    _, replay_trace, replay_repeated, replay_crossing = _run_event_topology(base, moving, curved)
    deterministic = event_trace == replay_trace and repeated == replay_repeated and crossing_events == replay_crossing
    subcritical = PersistentCutCellField(base)
    subcritical.apply_sweep(poses[0], poses[1], _work_channels(base, moving, ratio=0.99))
    stationary = PersistentCutCellField(base)
    still = ScalpelPose(poses[0].center_m, tangent=poses[0].tangent, velocity_m_s=(0.0, 0.0, 0.0))
    stationary.apply_sweep(still, still, work)

    mesh = solver.released_wound_mesh()
    if not len(mesh.triangles):
        collision_coverage, maximum_crossing = 0.0, math.inf
        gaps = np.zeros(1)
    else:
        collision_coverage, maximum_crossing = _probe_collision(mesh, curved["wound_collision"])
        gaps = np.sum(
            (solver.position[solver.mesh.gap_plus_nodes[solver.released]] - solver.position[solver.mesh.gap_minus_nodes[solver.released]])
            * solver.mesh.gap_normals[solver.released], axis=1
        )
    area_error = abs(mesh.positive_area_m2 - mesh.negative_area_m2) / max(mesh.positive_area_m2, 1.0e-15)
    initial_mass = float(base["material"]["density_kg_m3"]) * float(np.sum(solver.mesh.rest_volume))
    mass_error = abs(float(np.sum(solver.mesh.mass)) - initial_mass) / initial_mass
    released_fraction = float(np.mean(solver.released))
    monotonic = all(right >= left for left, right in zip(solver.release_history[:-1], solver.release_history[1:], strict=True))
    max_ahead = max(solver.release_ahead_history, default=0.0)
    geometry = base["geometry"]
    cut_area = float(geometry["thickness_m"]) * (
        float(moving["path"]["end_x_m"]) - float(moving["path"]["start_x_m"])
    )
    fracture_work = work.fracture_j_m2 * cut_area
    adhesion_work = work.adhesion_j_m2 * cut_area
    wear_work = work.wear_j_m2 * cut_area
    viscous_work = work.viscous_j_m2 * cut_area
    friction_work = work.friction_j_m2 * cut_area
    path_length = sum(
        float(np.linalg.norm(np.asarray(end.center_m) - np.asarray(start.center_m)))
        for start, end in zip(poses[:-1], poses[1:], strict=True)
    )
    peak_force = (fracture_work + adhesion_work + wear_work + viscous_work + friction_work) / path_length
    trace_sha = hashlib.sha256(json.dumps(solver.event_trace, separators=(",", ":")).encode()).hexdigest()
    profile_sha = hashlib.sha256(json.dumps(moving, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    limits = moving["qualification"]
    gates = {
        "fracture_events": solver.field.fracture_event_count >= int(limits["minimum_fracture_events"]),
        "released_fraction": released_fraction >= float(limits["minimum_dynamic_released_pair_fraction"]),
        "release_locality": max_ahead <= float(limits["maximum_release_ahead_of_blade_m"]),
        "monotonic_release": monotonic,
        "repeat_idempotent": repeated <= int(limits["maximum_repeated_path_additional_events"]),
        "crossing_topology": crossing_events >= int(limits["minimum_crossing_path_additional_events"])
        and event_field.intersection_cell_count() >= int(limits["minimum_intersection_cells"]),
        "subcritical_block": subcritical.fracture_event_count == 0,
        "stationary_block": stationary.fracture_event_count == 0,
        "finite": solver.finite,
        "no_inversion": solver.inversion_observations == 0,
        "jacobian": solver.minimum_jacobian >= float(limits["minimum_jacobian"]),
        "mass": mass_error <= float(limits["maximum_mass_relative_error"]),
        "minimum_gap": float(np.mean(gaps)) >= float(limits["minimum_mean_wound_gap_m"]),
        "maximum_gap": float(np.mean(gaps)) <= float(limits["maximum_mean_wound_gap_m"]),
        "opposed_area": area_error <= float(limits["maximum_opposed_area_relative_error"]),
        "two_sided_collision": collision_coverage >= float(limits["minimum_two_sided_collision_coverage_fraction"]),
        "probe_crossing": maximum_crossing <= float(limits["maximum_probe_surface_crossing_m"]),
        "event_replay": deterministic,
    }
    failed = tuple(name for name, passed in gates.items() if not passed)
    return MovingScalpelCutReceipt(
        schema="dr.anmar.moving-scalpel-cut-receipt.v1", profile_id=str(moving["id"]),
        profile_sha256=profile_sha, base_profile_sha256=base_sha, embedded_profile_sha256=curved_sha,
        path_segment_count=len(poses) - 1, pseudo_dynamic_step_count=solver.steps,
        fracture_event_count=solver.field.fracture_event_count, released_pair_count=int(np.count_nonzero(solver.released)),
        total_pair_count=len(solver.released), released_pair_fraction=released_fraction,
        monotonic_release=monotonic, maximum_release_ahead_of_blade_m=max_ahead,
        first_release_segment=first_release, last_release_segment=last_release,
        repeated_path_additional_events=repeated, crossing_path_additional_events=crossing_events,
        intersection_cell_count=event_field.intersection_cell_count(),
        subcritical_fracture_events=subcritical.fracture_event_count,
        stationary_fracture_events=stationary.fracture_event_count,
        fracture_work_j=fracture_work, adhesion_work_j=adhesion_work, wear_work_j=wear_work,
        viscous_work_j=viscous_work, friction_work_j=friction_work, peak_cutting_force_n=peak_force,
        finite=solver.finite, inversion_observation_count=solver.inversion_observations,
        minimum_jacobian=solver.minimum_jacobian, mass_relative_error=mass_error,
        mean_wound_gap_m=float(np.mean(gaps)), maximum_wound_gap_m=float(np.max(gaps)),
        positive_wound_area_m2=mesh.positive_area_m2, negative_wound_area_m2=mesh.negative_area_m2,
        opposed_area_relative_error=area_error, two_sided_collision_coverage_fraction=collision_coverage,
        maximum_probe_surface_crossing_m=maximum_crossing, event_trace_sha256=trace_sha,
        deterministic_event_replay=deterministic, qualified=not failed, failed_gates=failed,
        real_time_transient=False, biomechanical_validation=False, clinical_validation=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_MOVING_PROFILE_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = run_moving_scalpel_qualification(load_moving_profile(args.profile), moving_profile_path=args.profile)
    encoded = json.dumps(receipt.payload(), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if receipt.qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
