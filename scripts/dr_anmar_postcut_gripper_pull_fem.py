#!/usr/bin/env python3
"""Post-cut bilateral gripper custody and lateral flap-pull qualification."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from dr_anmar_cuttable_tissue_solver import load_profile
from dr_anmar_dynamic_curved_cut_fem import _level_set, load_curved_profile
from dr_anmar_moving_scalpel_cut_fem import (
    MovingScalpelCutFEM,
    _path_poses,
    _work_channels,
    load_moving_profile,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_PATH = (
    REPOSITORY_ROOT / "physics_next/tissues/dr-anmar-postcut-gripper-pull-v1.json"
)


@dataclass(frozen=True)
class GripperContact:
    force: np.ndarray
    top_count: int
    bottom_count: int
    top_normal_force_n: float
    bottom_normal_force_n: float
    lateral_reaction_n: float
    maximum_penetration_m: float


@dataclass(frozen=True)
class PostCutGripperPullReceipt:
    schema: str
    profile_id: str
    profile_sha256: str
    moving_profile_sha256: str
    cut_fracture_event_count: int
    released_pair_count: int
    retained_anchor_node_count: int
    gripped_top_candidate_count: int
    gripped_bottom_candidate_count: int
    peak_top_contact_count: int
    peak_bottom_contact_count: int
    pull_bilateral_custody_fraction: float
    peak_top_normal_force_n: float
    peak_bottom_normal_force_n: float
    peak_lateral_reaction_n: float
    maximum_contact_penetration_m: float
    commanded_lateral_pull_m: float
    gripped_flap_lateral_displacement_m: float
    opposite_flap_lateral_displacement_m: float
    differential_flap_displacement_m: float
    pre_pull_local_wound_gap_m: float
    peak_pull_local_wound_gap_m: float
    local_wound_gap_increase_m: float
    maximum_anchor_drift_m: float
    minimum_jacobian: float
    inversion_observation_count: int
    topology_event_delta: int
    recovery_residual_m: float
    finite: bool
    trace_sha256: str
    deterministic_replay: bool
    qualified: bool
    failed_gates: tuple[str, ...]
    quasi_static: bool
    biomechanical_validation: bool
    clinical_validation: bool

    def payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failed_gates"] = list(self.failed_gates)
        return payload


def load_gripper_profile(path: Path = DEFAULT_PROFILE_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class BilateralFrictionGripper:
    """Two finite jaw planes with penalty normal contact and capped stick friction."""

    def __init__(
        self,
        solver: MovingScalpelCutFEM,
        profile: dict[str, Any],
    ) -> None:
        self.solver = solver
        self.cfg = profile["gripper"]
        rest = solver.mesh.rest
        signed = _level_set(rest, solver.curved["implicit_cut"])
        maximum_z = float(np.max(rest[:, 2]))
        minimum_z = float(np.min(rest[:, 2]))
        scale_x = 1.0 + float(solver.base["geometry"]["prestrain_x"])
        material_center = np.asarray(self.cfg["grasp_center_material_m"], dtype=np.float64)
        self.center_xy = np.asarray((material_center[0] * scale_x, material_center[1]))
        half = np.asarray(self.cfg["pad_half_extent_m"], dtype=np.float64)
        in_patch = (
            (np.abs(rest[:, 0] - material_center[0]) <= half[0] + 1.0e-12)
            & (np.abs(rest[:, 1] - material_center[1]) <= half[1] + 1.0e-12)
            & (signed > 1.0e-9)
        )
        self.top_nodes = np.flatnonzero(
            in_patch & np.isclose(rest[:, 2], maximum_z, atol=1.0e-12)
        )
        self.bottom_nodes = np.flatnonzero(
            in_patch & np.isclose(rest[:, 2], minimum_z, atol=1.0e-12)
        )
        if not len(self.top_nodes) or not len(self.bottom_nodes):
            raise ValueError("Gripper pads do not cover both tissue surfaces")
        self.maximum_z = maximum_z
        self.minimum_z = minimum_z
        clearance = float(self.cfg["open_clearance_m"])
        self.top_plane_z = maximum_z + clearance
        self.bottom_plane_z = minimum_z - clearance
        self.top_velocity = np.zeros(3, dtype=np.float64)
        self.bottom_velocity = np.zeros(3, dtype=np.float64)
        self.top_offsets = np.full((len(self.top_nodes), 2), np.nan, dtype=np.float64)
        self.bottom_offsets = np.full((len(self.bottom_nodes), 2), np.nan, dtype=np.float64)

    def set_pose(
        self,
        center_xy: np.ndarray,
        top_plane_z: float,
        bottom_plane_z: float,
        dt: float,
    ) -> None:
        center_xy = np.asarray(center_xy, dtype=np.float64)
        old_center = self.center_xy.copy()
        old_top = self.top_plane_z
        old_bottom = self.bottom_plane_z
        self.center_xy = center_xy
        self.top_plane_z = float(top_plane_z)
        self.bottom_plane_z = float(bottom_plane_z)
        lateral_velocity = (center_xy - old_center) / dt
        self.top_velocity = np.asarray(
            (lateral_velocity[0], lateral_velocity[1], (self.top_plane_z - old_top) / dt)
        )
        self.bottom_velocity = np.asarray(
            (lateral_velocity[0], lateral_velocity[1], (self.bottom_plane_z - old_bottom) / dt)
        )

    def _surface_contact(
        self,
        nodes: np.ndarray,
        offsets: np.ndarray,
        *,
        top: bool,
    ) -> tuple[np.ndarray, int, float, float]:
        position = self.solver.position[nodes]
        velocity = self.solver.velocity[nodes]
        half = np.asarray(self.cfg["pad_half_extent_m"], dtype=np.float64)
        inside = np.all(np.abs(position[:, :2] - self.center_xy) <= half + 1.0e-12, axis=1)
        if top:
            penetration = position[:, 2] - self.top_plane_z
            penetration_rate = velocity[:, 2] - self.top_velocity[2]
            normal_sign = -1.0
            jaw_velocity = self.top_velocity
        else:
            penetration = self.bottom_plane_z - position[:, 2]
            penetration_rate = self.bottom_velocity[2] - velocity[:, 2]
            normal_sign = 1.0
            jaw_velocity = self.bottom_velocity
        active = inside & (penetration > 0.0)
        offsets[~active] = np.nan
        newly_active = active & np.isnan(offsets[:, 0])
        offsets[newly_active] = position[newly_active, :2] - self.center_xy
        nodal = np.zeros((len(nodes), 3), dtype=np.float64)
        if not np.any(active):
            return nodal, 0, 0.0, 0.0
        normal = (
            float(self.cfg["normal_stiffness_n_m"]) * penetration[active]
            + float(self.cfg["normal_damping_n_s_m"])
            * np.maximum(penetration_rate[active], 0.0)
        )
        normal = np.maximum(normal, 0.0)
        nodal[active, 2] = normal_sign * normal
        target_xy = self.center_xy + offsets[active]
        tangent = (
            float(self.cfg["tangential_stiffness_n_m"])
            * (target_xy - position[active, :2])
            + float(self.cfg["tangential_damping_n_s_m"])
            * (jaw_velocity[None, :2] - velocity[active, :2])
        )
        magnitude = np.linalg.norm(tangent, axis=1)
        limit = float(self.cfg["friction_coefficient"]) * normal
        scale = np.minimum(1.0, limit / np.maximum(magnitude, 1.0e-12))
        nodal[active, :2] = tangent * scale[:, None]
        return (
            nodal,
            int(np.count_nonzero(active)),
            float(np.sum(normal)),
            float(np.max(penetration[active])),
        )

    def contact_force(self) -> GripperContact:
        force = np.zeros_like(self.solver.position)
        top_force, top_count, top_normal, top_penetration = self._surface_contact(
            self.top_nodes, self.top_offsets, top=True
        )
        bottom_force, bottom_count, bottom_normal, bottom_penetration = self._surface_contact(
            self.bottom_nodes, self.bottom_offsets, top=False
        )
        np.add.at(force, self.top_nodes, top_force)
        np.add.at(force, self.bottom_nodes, bottom_force)
        lateral_reaction = float(abs(np.sum(force[:, 1])))
        return GripperContact(
            force=force,
            top_count=top_count,
            bottom_count=bottom_count,
            top_normal_force_n=top_normal,
            bottom_normal_force_n=bottom_normal,
            lateral_reaction_n=lateral_reaction,
            maximum_penetration_m=max(top_penetration, bottom_penetration),
        )


def _complete_cut(solver: MovingScalpelCutFEM) -> None:
    moving = solver.moving
    poses = _path_poses(moving, solver.curved)
    work = _work_channels(solver.base, moving)
    dt = float(moving["quasi_static_solver"]["pseudo_time_step_s"])
    relax = int(moving["quasi_static_solver"]["relaxation_steps_per_segment"])
    for segment, (start, end) in enumerate(zip(poses[:-1], poses[1:], strict=True)):
        solver.advance_blade(segment, start, end, work)
        center = np.asarray(end.center_m, dtype=np.float64)
        for _ in range(relax):
            solver.step(dt, center)
    for _ in range(int(moving["quasi_static_solver"]["post_cut_relaxation_steps"])):
        solver.step(dt)


def _run_once(
    profile: dict[str, Any],
    capture: Callable[
        [str, int, int, MovingScalpelCutFEM, BilateralFrictionGripper, GripperContact],
        None,
    ] | None = None,
) -> dict[str, Any]:
    moving_path = REPOSITORY_ROOT / profile["moving_cut_profile"]
    moving_file_sha = hashlib.sha256(moving_path.read_bytes()).hexdigest()
    if moving_file_sha != profile["moving_cut_profile_file_sha256"]:
        raise ValueError("Post-cut gripper profile is not bound to the moving-cut profile")
    moving = load_moving_profile(moving_path)
    moving = copy.deepcopy(moving)
    moving["quasi_static_solver"]["velocity_damping_per_s"] = float(
        profile["quasi_static_schedule"]["velocity_damping_per_s"]
    )
    base = load_profile(REPOSITORY_ROOT / moving["base_profile"])
    curved = load_curved_profile(REPOSITORY_ROOT / moving["embedded_profile"])
    solver = MovingScalpelCutFEM(base, curved, moving)
    _complete_cut(solver)
    if not np.all(solver.released):
        raise RuntimeError("Gripper qualification requires the completed persistent cut")
    cut_events = solver.field.fracture_event_count
    gripper = BilateralFrictionGripper(solver, profile)
    schedule = profile["quasi_static_schedule"]
    limits = profile["qualification"]
    dt = float(schedule["pseudo_time_step_s"])
    open_top = gripper.maximum_z + float(gripper.cfg["open_clearance_m"])
    open_bottom = gripper.minimum_z - float(gripper.cfg["open_clearance_m"])
    closed_top = gripper.maximum_z - float(gripper.cfg["closed_indentation_per_jaw_m"])
    closed_bottom = gripper.minimum_z + float(gripper.cfg["closed_indentation_per_jaw_m"])
    start_center = gripper.center_xy.copy()
    pull_center = start_center + np.asarray((0.0, float(gripper.cfg["lateral_pull_m"])))

    tracked_nodes = np.unique(np.concatenate((gripper.top_nodes, gripper.bottom_nodes)))
    signed = _level_set(solver.mesh.rest, curved["implicit_cut"])
    material_center = np.asarray(gripper.cfg["grasp_center_material_m"], dtype=np.float64)
    half_x = float(gripper.cfg["pad_half_extent_m"][0])
    outer_surface = np.isclose(
        np.abs(solver.mesh.rest[:, 2]),
        float(base["geometry"]["thickness_m"]) / 2.0,
        atol=1.0e-12,
    )
    opposite_nodes = np.flatnonzero(
        outer_surface
        & (signed < -1.0e-9)
        & (np.abs(solver.mesh.rest[:, 0] - material_center[0]) <= half_x)
        & (np.abs(signed) <= 0.0045)
    )
    if not len(opposite_nodes):
        raise RuntimeError("No opposite-flap tracking nodes were found")
    local_pairs = np.abs(solver.mesh.gap_rest_points[:, 0] - material_center[0]) <= half_x
    baseline = solver.position.copy()
    pre_gap = float(np.mean(np.maximum(
        np.sum(
            (solver.position[solver.mesh.gap_plus_nodes[local_pairs]]
             - solver.position[solver.mesh.gap_minus_nodes[local_pairs]])
            * solver.mesh.gap_normals[local_pairs],
            axis=1,
        ),
        0.0,
    )))
    anchor_baseline = solver.position[solver.mesh.fixed].copy()
    trace: list[tuple[Any, ...]] = []
    peak_top_count = peak_bottom_count = 0
    peak_top_force = peak_bottom_force = peak_lateral = peak_penetration = 0.0
    pull_custody: list[bool] = []
    peak_pull_gap = pre_gap
    pull_gripped_y = pull_opposite_y = 0.0

    def advance(
        phase: str,
        step_index: int,
        count: int,
        center_start: np.ndarray,
        center_end: np.ndarray,
        top_start: float,
        top_end: float,
        bottom_start: float,
        bottom_end: float,
    ) -> None:
        nonlocal peak_top_count, peak_bottom_count, peak_top_force, peak_bottom_force
        nonlocal peak_lateral, peak_penetration, peak_pull_gap, pull_gripped_y, pull_opposite_y
        alpha = (step_index + 1) / count
        center = center_start + alpha * (center_end - center_start)
        top = top_start + alpha * (top_end - top_start)
        bottom = bottom_start + alpha * (bottom_end - bottom_start)
        gripper.set_pose(center, top, bottom, dt)
        contact = gripper.contact_force()
        solver.step(dt, external_force=contact.force)
        if capture is not None:
            capture(phase, step_index, count, solver, gripper, contact)
        peak_top_count = max(peak_top_count, contact.top_count)
        peak_bottom_count = max(peak_bottom_count, contact.bottom_count)
        peak_top_force = max(peak_top_force, contact.top_normal_force_n)
        peak_bottom_force = max(peak_bottom_force, contact.bottom_normal_force_n)
        peak_lateral = max(peak_lateral, contact.lateral_reaction_n)
        peak_penetration = max(peak_penetration, contact.maximum_penetration_m)
        local_gap = float(np.mean(np.maximum(
            np.sum(
                (solver.position[solver.mesh.gap_plus_nodes[local_pairs]]
                 - solver.position[solver.mesh.gap_minus_nodes[local_pairs]])
                * solver.mesh.gap_normals[local_pairs],
                axis=1,
            ),
            0.0,
        )))
        if phase == "pull":
            bilateral = (
                contact.top_count >= int(limits["minimum_top_contact_nodes"])
                and contact.bottom_count >= int(limits["minimum_bottom_contact_nodes"])
            )
            pull_custody.append(bilateral)
            peak_pull_gap = max(peak_pull_gap, local_gap)
            pull_gripped_y = float(np.mean(solver.position[tracked_nodes, 1] - baseline[tracked_nodes, 1]))
            pull_opposite_y = float(np.mean(solver.position[opposite_nodes, 1] - baseline[opposite_nodes, 1]))
        if step_index % 20 == 0 or step_index + 1 == count:
            trace.append((
                phase, step_index, contact.top_count, contact.bottom_count,
                round(contact.top_normal_force_n, 8), round(contact.bottom_normal_force_n, 8),
                round(contact.lateral_reaction_n, 8), round(local_gap, 9),
                round(float(np.min(solver._deformation()[1])), 7),
            ))

    close_steps = int(schedule["close_steps"])
    for step in range(close_steps):
        advance("close", step, close_steps, start_center, start_center, open_top, closed_top, open_bottom, closed_bottom)
    hold_steps = int(schedule["custody_hold_steps"])
    for step in range(hold_steps):
        advance("custody", step, hold_steps, start_center, start_center, closed_top, closed_top, closed_bottom, closed_bottom)
    pull_steps = int(schedule["pull_steps"])
    for step in range(pull_steps):
        advance("pull", step, pull_steps, start_center, pull_center, closed_top, closed_top, closed_bottom, closed_bottom)
    lateral_hold = int(schedule["lateral_hold_steps"])
    for step in range(lateral_hold):
        advance("lateral_hold", step, lateral_hold, pull_center, pull_center, closed_top, closed_top, closed_bottom, closed_bottom)
    release_steps = int(schedule["release_steps"])
    for step in range(release_steps):
        advance("release", step, release_steps, pull_center, pull_center, closed_top, open_top, closed_bottom, open_bottom)
    recovery_steps = int(schedule["recovery_steps"])
    for step in range(recovery_steps):
        advance("recovery", step, recovery_steps, pull_center, pull_center, open_top, open_top, open_bottom, open_bottom)

    gripped_displacement = pull_gripped_y
    opposite_displacement = pull_opposite_y
    differential = gripped_displacement - opposite_displacement
    recovery = float(np.linalg.norm(
        np.mean(solver.position[tracked_nodes] - baseline[tracked_nodes], axis=0)
    ))
    anchor_drift = float(np.max(np.linalg.norm(
        solver.position[solver.mesh.fixed] - anchor_baseline, axis=1
    )))
    trace_sha = hashlib.sha256(json.dumps(trace, separators=(",", ":")).encode()).hexdigest()
    custody_fraction = float(np.mean(pull_custody)) if pull_custody else 0.0
    gap_increase = peak_pull_gap - pre_gap
    gates = {
        "top_contact": peak_top_count >= int(limits["minimum_top_contact_nodes"]),
        "bottom_contact": peak_bottom_count >= int(limits["minimum_bottom_contact_nodes"]),
        "custody": custody_fraction >= float(limits["minimum_pull_custody_fraction"]),
        "top_force": peak_top_force >= float(limits["minimum_peak_normal_force_per_jaw_n"]),
        "bottom_force": peak_bottom_force >= float(limits["minimum_peak_normal_force_per_jaw_n"]),
        "lateral_reaction": peak_lateral >= float(limits["minimum_peak_lateral_reaction_n"]),
        "flap_motion": gripped_displacement >= float(limits["minimum_gripped_flap_lateral_displacement_m"]),
        "differential_motion": differential >= float(limits["minimum_differential_flap_displacement_m"]),
        "wound_gap": gap_increase >= float(limits["minimum_local_wound_gap_increase_m"]),
        "penetration": peak_penetration <= float(limits["maximum_contact_penetration_m"]),
        "anchors": anchor_drift <= float(limits["maximum_anchor_drift_m"]),
        "jacobian": solver.minimum_jacobian >= float(limits["minimum_jacobian"]),
        "inversions": solver.inversion_observations == 0,
        "topology": solver.field.fracture_event_count - cut_events <= int(limits["maximum_topology_event_delta"]),
        "recovery": recovery <= float(limits["maximum_recovery_residual_m"]),
        "finite": solver.finite,
    }
    return {
        "solver": solver,
        "gripper": gripper,
        "moving_file_sha": moving_file_sha,
        "cut_events": cut_events,
        "peak_top_count": peak_top_count,
        "peak_bottom_count": peak_bottom_count,
        "custody_fraction": custody_fraction,
        "peak_top_force": peak_top_force,
        "peak_bottom_force": peak_bottom_force,
        "peak_lateral": peak_lateral,
        "peak_penetration": peak_penetration,
        "gripped_displacement": gripped_displacement,
        "opposite_displacement": opposite_displacement,
        "differential": differential,
        "pre_gap": pre_gap,
        "peak_pull_gap": peak_pull_gap,
        "gap_increase": gap_increase,
        "anchor_drift": anchor_drift,
        "recovery": recovery,
        "trace_sha": trace_sha,
        "gates": gates,
    }


def run_postcut_gripper_pull_qualification(
    profile: dict[str, Any] | None = None,
    *,
    profile_path: Path = DEFAULT_PROFILE_PATH,
) -> PostCutGripperPullReceipt:
    profile = profile or load_gripper_profile(profile_path)
    first = _run_once(profile)
    replay = _run_once(profile)
    deterministic = first["trace_sha"] == replay["trace_sha"]
    gates = dict(first["gates"])
    gates["replay"] = deterministic
    failed = tuple(name for name, passed in gates.items() if not passed)
    solver: MovingScalpelCutFEM = first["solver"]
    gripper: BilateralFrictionGripper = first["gripper"]
    profile_sha = hashlib.sha256(
        json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return PostCutGripperPullReceipt(
        schema="dr.anmar.postcut-gripper-pull-receipt.v1",
        profile_id=str(profile["id"]), profile_sha256=profile_sha,
        moving_profile_sha256=first["moving_file_sha"],
        cut_fracture_event_count=first["cut_events"],
        released_pair_count=int(np.count_nonzero(solver.released)),
        retained_anchor_node_count=int(np.count_nonzero(solver.mesh.fixed)),
        gripped_top_candidate_count=len(gripper.top_nodes),
        gripped_bottom_candidate_count=len(gripper.bottom_nodes),
        peak_top_contact_count=first["peak_top_count"],
        peak_bottom_contact_count=first["peak_bottom_count"],
        pull_bilateral_custody_fraction=first["custody_fraction"],
        peak_top_normal_force_n=first["peak_top_force"],
        peak_bottom_normal_force_n=first["peak_bottom_force"],
        peak_lateral_reaction_n=first["peak_lateral"],
        maximum_contact_penetration_m=first["peak_penetration"],
        commanded_lateral_pull_m=float(profile["gripper"]["lateral_pull_m"]),
        gripped_flap_lateral_displacement_m=first["gripped_displacement"],
        opposite_flap_lateral_displacement_m=first["opposite_displacement"],
        differential_flap_displacement_m=first["differential"],
        pre_pull_local_wound_gap_m=first["pre_gap"],
        peak_pull_local_wound_gap_m=first["peak_pull_gap"],
        local_wound_gap_increase_m=first["gap_increase"],
        maximum_anchor_drift_m=first["anchor_drift"],
        minimum_jacobian=solver.minimum_jacobian,
        inversion_observation_count=solver.inversion_observations,
        topology_event_delta=solver.field.fracture_event_count - first["cut_events"],
        recovery_residual_m=first["recovery"], finite=solver.finite,
        trace_sha256=first["trace_sha"], deterministic_replay=deterministic,
        qualified=not failed, failed_gates=failed, quasi_static=True,
        biomechanical_validation=False, clinical_validation=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = run_postcut_gripper_pull_qualification(
        load_gripper_profile(args.profile), profile_path=args.profile
    )
    encoded = json.dumps(receipt.payload(), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if receipt.qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
