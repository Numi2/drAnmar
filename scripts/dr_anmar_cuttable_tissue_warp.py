#!/usr/bin/env python3
"""NVIDIA Warp kernels and parity qualification for cuttable tissue mechanics.

Warp is intentionally not added to the repository-validation environment. This
module runs with the Warp bundled by the isolated Isaac runtime or an explicit
parity environment. The CPU reference remains the mathematical oracle; CUDA is
not promoted until these kernels agree with it and a full native receipt passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from dr_anmar_cuttable_tissue_solver import (
    DEFAULT_PROFILE_PATH,
    CuttableTissueReferenceSolver,
    ScalpelPose,
    load_profile,
)


@wp.kernel
def _clear_force(
    force_x: wp.array(dtype=float),
    force_y: wp.array(dtype=float),
    force_z: wp.array(dtype=float),
):
    node = wp.tid()
    force_x[node] = 0.0
    force_y[node] = 0.0
    force_z[node] = 0.0


@wp.kernel
def _neo_hookean_prony_force(
    positions: wp.array(dtype=wp.vec3),
    tetrahedra: wp.array(dtype=wp.vec4i),
    inverse_rest: wp.array(dtype=wp.mat33),
    shape_gradients: wp.array(dtype=wp.vec3),
    rest_volume: wp.array(dtype=float),
    prony_history: wp.array(dtype=wp.mat33),
    previous_elastic_stress: wp.array(dtype=wp.mat33),
    force_x: wp.array(dtype=float),
    force_y: wp.array(dtype=float),
    force_z: wp.array(dtype=float),
    jacobian: wp.array(dtype=float),
    shear_modulus: float,
    lame_lambda: float,
    prony_fraction: float,
    prony_decay: float,
):
    tet_index = wp.tid()
    tet = tetrahedra[tet_index]
    x0 = positions[tet[0]]
    edge_1 = positions[tet[1]] - x0
    edge_2 = positions[tet[2]] - x0
    edge_3 = positions[tet[3]] - x0
    deformed = wp.mat33(
        edge_1[0],
        edge_2[0],
        edge_3[0],
        edge_1[1],
        edge_2[1],
        edge_3[1],
        edge_1[2],
        edge_2[2],
        edge_3[2],
    )
    deformation = deformed * inverse_rest[tet_index]
    determinant = wp.determinant(deformation)
    jacobian[tet_index] = determinant
    safe_j = wp.max(determinant, 1.0e-9)
    inverse_transpose = wp.transpose(wp.inverse(deformation))
    elastic = shear_modulus * (deformation - inverse_transpose)
    elastic = elastic + lame_lambda * wp.log(safe_j) * inverse_transpose
    history = prony_decay * prony_history[tet_index] + prony_fraction * (
        elastic - previous_elastic_stress[tet_index]
    )
    prony_history[tet_index] = history
    previous_elastic_stress[tet_index] = elastic
    stress = (1.0 - prony_fraction) * elastic + history
    volume = rest_volume[tet_index]
    for local in range(4):
        gradient = shape_gradients[tet_index * 4 + local]
        local_force = -volume * (stress * gradient)
        node = tet[local]
        wp.atomic_add(force_x, node, local_force[0])
        wp.atomic_add(force_y, node, local_force[1])
        wp.atomic_add(force_z, node, local_force[2])


@wp.kernel
def _surface_scalpel_contact(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    top_triangles: wp.array(dtype=wp.vec3i),
    force_x: wp.array(dtype=float),
    force_y: wp.array(dtype=float),
    force_z: wp.array(dtype=float),
    reaction_x: wp.array(dtype=float),
    reaction_y: wp.array(dtype=float),
    reaction_z: wp.array(dtype=float),
    maximum_penetration: wp.array(dtype=float),
    active_segments: wp.array(dtype=wp.int32),
    center: wp.vec3,
    tangent: wp.vec3,
    blade_velocity: wp.vec3,
    half_length: float,
    radius: float,
    normal_stiffness: float,
    normal_damping: float,
    dynamic_friction: float,
    friction_regularization: float,
    segment_count: int,
    triangle_count: int,
):
    segment = wp.tid()
    segment_length = 2.0 * half_length / float(segment_count)
    offset = -half_length + (float(segment) + 0.5) * segment_length
    axis_point = center + offset * tangent
    chosen = int(-1)
    chosen_b0 = float(0.0)
    chosen_b1 = float(0.0)
    chosen_b2 = float(0.0)
    for triangle_slot in range(triangle_count):
        triangle = top_triangles[triangle_slot]
        p0 = positions[triangle[0]]
        p1 = positions[triangle[1]]
        p2 = positions[triangle[2]]
        edge_0_x = p1[0] - p0[0]
        edge_0_y = p1[1] - p0[1]
        edge_1_x = p2[0] - p0[0]
        edge_1_y = p2[1] - p0[1]
        query_x = axis_point[0] - p0[0]
        query_y = axis_point[1] - p0[1]
        denominator = edge_0_x * edge_1_y - edge_0_y * edge_1_x
        bary_1 = (query_x * edge_1_y - query_y * edge_1_x) / denominator
        bary_2 = (edge_0_x * query_y - edge_0_y * query_x) / denominator
        bary_0 = 1.0 - bary_1 - bary_2
        if chosen < 0 and bary_0 >= -1.0e-10 and bary_1 >= -1.0e-10 and bary_2 >= -1.0e-10:
            chosen = triangle_slot
            chosen_b0 = bary_0
            chosen_b1 = bary_1
            chosen_b2 = bary_2
    if chosen < 0:
        return

    triangle = top_triangles[chosen]
    p0 = positions[triangle[0]]
    p1 = positions[triangle[1]]
    p2 = positions[triangle[2]]
    surface_point = chosen_b0 * p0 + chosen_b1 * p1 + chosen_b2 * p2
    surface_velocity = (
        chosen_b0 * velocities[triangle[0]]
        + chosen_b1 * velocities[triangle[1]]
        + chosen_b2 * velocities[triangle[2]]
    )
    surface_normal = wp.normalize(wp.cross(p1 - p0, p2 - p0))
    if surface_normal[2] < 0.0:
        surface_normal = -surface_normal
    signed_gap = wp.dot(axis_point - surface_point, surface_normal)
    penetration = wp.max(0.0, radius - signed_gap)
    if penetration <= 0.0:
        return

    contact_normal = -surface_normal
    relative_velocity = surface_velocity - blade_velocity
    normal_velocity = wp.dot(relative_velocity, contact_normal)
    normal_pressure = normal_stiffness * penetration
    normal_pressure = normal_pressure - normal_damping * wp.min(normal_velocity, 0.0)
    normal_pressure = wp.max(normal_pressure, 0.0)
    effective_gap = wp.clamp(signed_gap, 0.0, radius)
    strip_width = 2.0 * wp.sqrt(wp.max(0.0, radius * radius - effective_gap * effective_gap))
    normal_magnitude = normal_pressure * strip_width * segment_length
    sample_force = normal_magnitude * contact_normal

    tangential_velocity = relative_velocity - normal_velocity * contact_normal
    tangential_speed = wp.length(tangential_velocity)
    if tangential_speed > 1.0e-12:
        friction_scale = wp.tanh(tangential_speed / friction_regularization)
        sample_force = (
            sample_force
            - (dynamic_friction * normal_magnitude * friction_scale / tangential_speed)
            * tangential_velocity
        )

    barycentric = wp.vec3(chosen_b0, chosen_b1, chosen_b2)
    for local in range(3):
        node = triangle[local]
        local_force = barycentric[local] * sample_force
        wp.atomic_add(force_x, node, local_force[0])
        wp.atomic_add(force_y, node, local_force[1])
        wp.atomic_add(force_z, node, local_force[2])
    wp.atomic_add(reaction_x, 0, -sample_force[0])
    wp.atomic_add(reaction_y, 0, -sample_force[1])
    wp.atomic_add(reaction_z, 0, -sample_force[2])
    wp.atomic_max(maximum_penetration, 0, penetration)
    wp.atomic_add(active_segments, 0, 1)


@dataclass(frozen=True)
class WarpParityReceipt:
    schema: str
    profile_id: str
    profile_sha256: str
    warp_version: str
    device: str
    device_is_cuda: bool
    internal_states: int
    contact_poses: int
    maximum_internal_force_relative_l2_error: float
    maximum_jacobian_absolute_error: float
    maximum_contact_force_relative_l2_error: float
    maximum_contact_reaction_absolute_error_n: float
    maximum_contact_penetration_absolute_error_m: float
    qualified: bool
    failed_gates: tuple[str, ...]
    cuda_promotion_pending: bool
    clinical_validation: bool

    def payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failed_gates"] = list(self.failed_gates)
        return payload


def _device_name(device: str) -> str:
    wp.init()
    return str(wp.get_device(device))


def _force_arrays(node_count: int, device: str):
    return (
        wp.zeros(node_count, dtype=float, device=device),
        wp.zeros(node_count, dtype=float, device=device),
        wp.zeros(node_count, dtype=float, device=device),
    )


def _stack_force(force_x, force_y, force_z) -> np.ndarray:
    return np.stack(
        (force_x.numpy(), force_y.numpy(), force_z.numpy()),
        axis=1,
    ).astype(np.float64)


def warp_internal_force(
    solver: CuttableTissueReferenceSolver,
    *,
    dt: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    material = solver.profile["material"]
    youngs = float(material["youngs_modulus_pa"])
    poisson = float(material["poisson_ratio"])
    shear = youngs / (2.0 * (1.0 + poisson))
    lame = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    force_x, force_y, force_z = _force_arrays(len(solver.position), device)
    jacobian = wp.zeros(len(solver.tets), dtype=float, device=device)
    history = wp.zeros(len(solver.tets), dtype=wp.mat33, device=device)
    previous = wp.zeros(len(solver.tets), dtype=wp.mat33, device=device)
    wp.launch(
        _neo_hookean_prony_force,
        dim=len(solver.tets),
        inputs=[
            wp.array(solver.position.astype(np.float32), dtype=wp.vec3, device=device),
            wp.array(solver.tets.astype(np.int32), dtype=wp.vec4i, device=device),
            wp.array(solver.dm_inverse.astype(np.float32), dtype=wp.mat33, device=device),
            wp.array(
                solver.shape_gradients.reshape(-1, 3).astype(np.float32),
                dtype=wp.vec3,
                device=device,
            ),
            wp.array(solver.rest_volume.astype(np.float32), dtype=float, device=device),
            history,
            previous,
            force_x,
            force_y,
            force_z,
            jacobian,
            shear,
            lame,
            float(material["prony_relaxation_fraction"]),
            float(np.exp(-dt / float(material["prony_time_constant_s"]))),
        ],
        device=device,
    )
    wp.synchronize_device(device)
    return _stack_force(force_x, force_y, force_z), jacobian.numpy().astype(np.float64)


def warp_scalpel_contact(
    solver: CuttableTissueReferenceSolver,
    pose: ScalpelPose,
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    contact = solver.profile["scalpel_contact"]
    force_x, force_y, force_z = _force_arrays(len(solver.position), device)
    reaction_x = wp.zeros(1, dtype=float, device=device)
    reaction_y = wp.zeros(1, dtype=float, device=device)
    reaction_z = wp.zeros(1, dtype=float, device=device)
    maximum_penetration = wp.zeros(1, dtype=float, device=device)
    active_segments = wp.zeros(1, dtype=wp.int32, device=device)
    segment_count = int(contact["edge_quadrature_segments"])
    tangent = np.asarray(pose.tangent, dtype=np.float64)
    tangent /= np.linalg.norm(tangent)
    wp.launch(
        _surface_scalpel_contact,
        dim=segment_count,
        inputs=[
            wp.array(solver.position.astype(np.float32), dtype=wp.vec3, device=device),
            wp.array(solver.velocity.astype(np.float32), dtype=wp.vec3, device=device),
            wp.array(solver.top_triangles.astype(np.int32), dtype=wp.vec3i, device=device),
            force_x,
            force_y,
            force_z,
            reaction_x,
            reaction_y,
            reaction_z,
            maximum_penetration,
            active_segments,
            wp.vec3(*map(float, pose.center_m)),
            wp.vec3(*map(float, tangent)),
            wp.vec3(*map(float, pose.velocity_m_s)),
            float(contact["edge_length_m"]) / 2.0,
            float(contact["edge_radius_m"]),
            float(contact["normal_stiffness_pa_m"]),
            float(contact["normal_damping_pa_s_m"]),
            float(contact["dynamic_friction"]),
            float(contact["friction_regularization_m_s"]),
            segment_count,
            len(solver.top_triangles),
        ],
        device=device,
    )
    wp.synchronize_device(device)
    reaction = np.asarray(
        [reaction_x.numpy()[0], reaction_y.numpy()[0], reaction_z.numpy()[0]],
        dtype=np.float64,
    )
    return (
        _stack_force(force_x, force_y, force_z),
        reaction,
        float(maximum_penetration.numpy()[0]),
        int(active_segments.numpy()[0]),
    )


def _relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(candidate - reference) / max(np.linalg.norm(reference), 1.0e-12))


def run_warp_parity(
    profile: dict[str, Any] | None = None,
    *,
    profile_path: Path = DEFAULT_PROFILE_PATH,
    device: str = "cpu",
) -> WarpParityReceipt:
    profile = profile or load_profile(profile_path)
    _device_name(device)
    dt = float(profile["solver"]["time_step_s"])
    internal_errors: list[float] = []
    jacobian_errors: list[float] = []
    for state_index in range(2):
        reference_solver = CuttableTissueReferenceSolver(profile)
        if state_index == 1:
            free = ~reference_solver.fixed
            x = reference_solver.position[:, 0]
            y = reference_solver.position[:, 1]
            reference_solver.position[free, 2] -= (
                0.0002
                * np.cos(np.pi * x[free] / float(profile["geometry"]["width_m"]))
                * np.cos(np.pi * y[free] / float(profile["geometry"]["depth_m"]))
            )
        warp_force, warp_jacobian = warp_internal_force(reference_solver, dt=dt, device=device)
        reference_force, reference_jacobian = reference_solver._internal_force(dt)
        internal_errors.append(_relative_l2(warp_force, reference_force))
        jacobian_errors.append(float(np.max(np.abs(warp_jacobian - reference_jacobian))))

    contact_errors: list[float] = []
    reaction_errors: list[float] = []
    penetration_errors: list[float] = []
    contact_solver = CuttableTissueReferenceSolver(profile)
    geometry = profile["geometry"]
    contact = profile["scalpel_contact"]
    cell_width = (
        float(geometry["width_m"])
        * (1.0 + float(geometry["prestrain_x"]))
        / int(geometry["cells_x"])
    )
    surface_z = float(np.max(contact_solver.position[:, 2]))
    edge_z = surface_z + float(contact["edge_radius_m"]) - 0.0001
    offsets = (-cell_width / 2.0, 0.0, cell_width / 3.0)
    for offset in offsets:
        pose = ScalpelPose((offset, 0.0, edge_z), velocity_m_s=(0.0, 0.002, -0.001))
        reference_force, reference_reaction, reference_penetration = (
            contact_solver._scalpel_contact(pose)
        )
        warp_force, warp_reaction, warp_penetration, active = warp_scalpel_contact(
            contact_solver, pose, device=device
        )
        if active <= 0:
            raise RuntimeError("Warp contact kernel missed a qualified off-grid blade pose")
        contact_errors.append(_relative_l2(warp_force, reference_force))
        reaction_errors.append(float(np.max(np.abs(warp_reaction - reference_reaction))))
        penetration_errors.append(abs(warp_penetration - reference_penetration))

    limits = profile["warp_parity"]
    metrics = {
        "internal_force": max(internal_errors),
        "contact_force": max(contact_errors),
        "penetration": max(penetration_errors),
    }
    gates = {
        "internal_force_parity": metrics["internal_force"]
        <= float(limits["maximum_internal_force_relative_l2_error"]),
        "contact_force_parity": metrics["contact_force"]
        <= float(limits["maximum_contact_force_relative_l2_error"]),
        "contact_penetration_parity": metrics["penetration"]
        <= float(limits["maximum_contact_penetration_absolute_error_m"]),
    }
    failed = tuple(name for name, passed in gates.items() if not passed)
    profile_sha = hashlib.sha256(
        json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    device_name = _device_name(device)
    is_cuda = "cuda" in device_name.lower()
    return WarpParityReceipt(
        schema="dr.anmar.cuttable-tissue-warp-parity-receipt.v1",
        profile_id=str(profile["id"]),
        profile_sha256=profile_sha,
        warp_version=str(wp.__version__),
        device=device_name,
        device_is_cuda=is_cuda,
        internal_states=len(internal_errors),
        contact_poses=len(contact_errors),
        maximum_internal_force_relative_l2_error=max(internal_errors),
        maximum_jacobian_absolute_error=max(jacobian_errors),
        maximum_contact_force_relative_l2_error=max(contact_errors),
        maximum_contact_reaction_absolute_error_n=max(reaction_errors),
        maximum_contact_penetration_absolute_error_m=max(penetration_errors),
        qualified=not failed,
        failed_gates=failed,
        cuda_promotion_pending=not is_cuda,
        clinical_validation=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = run_warp_parity(
        load_profile(args.profile), profile_path=args.profile, device=args.device
    )
    encoded = json.dumps(receipt.payload(), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if receipt.qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
