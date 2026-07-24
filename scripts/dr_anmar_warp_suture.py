#!/usr/bin/env python3
"""GPU-native XPBD dynamics for the Dr.Anmar surgical suture.

This backend replaces the microscopic rigid-body chain for high-contact tasks.
It uses NVIDIA Warp kernels directly (Warp 1.10 removed the former ``warp.sim``
module) and keeps the authored OpenUSD asset and JSON material profile as the
identity and calibration contract.

The implementation deliberately uses exhaustive segment-capsule pairs for the
360-segment research asset.  At this resolution there are fewer than 65k
non-local pairs, which is small on the target RTX 4090 and avoids broad-phase
false negatives while the knotting backend is being qualified.  Pair
corrections are gathered through a fixed CSR map, so the solver does not depend
on nondeterministic floating-point atomics.

This is an engineering simulation model.  It is not clinically validated.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from dr_anmar_suture_model import DEFAULT_PROFILE_PATH, derive, load_profile

try:
    import warp as wp
except ImportError as error:  # pragma: no cover - exercised on GPU deployment
    raise SystemExit(
        "NVIDIA Warp is required. Run this with the pinned Isaac Lab Python."
    ) from error


REPORT_SCHEMA = "dr.anmar.warp-suture-qualification.v1"
BACKEND_ID = "dr-anmar-warp-xpbd-segment-capsule-v1"


@wp.func
def _closest_segment_parameters(
    first_start: wp.vec3,
    first_end: wp.vec3,
    second_start: wp.vec3,
    second_end: wp.vec3,
) -> wp.vec2:
    """Closest parameters on two finite segments (Ericson clamp cases)."""

    first = first_end - first_start
    second = second_end - second_start
    offset = first_start - second_start
    first_squared = wp.dot(first, first)
    second_squared = wp.dot(second, second)
    first_offset = wp.dot(first, offset)
    second_offset = wp.dot(second, offset)
    epsilon = 1.0e-20
    first_amount = float(0.0)
    second_amount = float(0.0)

    if first_squared <= epsilon and second_squared <= epsilon:
        return wp.vec2(0.0, 0.0)
    if first_squared <= epsilon:
        second_amount = wp.clamp(second_offset / second_squared, 0.0, 1.0)
        return wp.vec2(0.0, second_amount)
    if second_squared <= epsilon:
        first_amount = wp.clamp(-first_offset / first_squared, 0.0, 1.0)
        return wp.vec2(first_amount, 0.0)

    cross = wp.dot(first, second)
    denominator = first_squared * second_squared - cross * cross
    if denominator > epsilon:
        first_amount = wp.clamp(
            (cross * second_offset - first_offset * second_squared) / denominator,
            0.0,
            1.0,
        )
    second_amount = (cross * first_amount + second_offset) / second_squared
    if second_amount < 0.0:
        second_amount = 0.0
        first_amount = wp.clamp(-first_offset / first_squared, 0.0, 1.0)
    elif second_amount > 1.0:
        second_amount = 1.0
        first_amount = wp.clamp(
            (cross - first_offset) / first_squared,
            0.0,
            1.0,
        )
    return wp.vec2(first_amount, second_amount)


@wp.func
def _constitutive_force(
    strain: float,
    axial_rigidity: float,
    yield_strain: float,
    failure_strain: float,
    failure_force: float,
    post_yield_exponent: float,
) -> float:
    extension = wp.max(strain, 0.0)
    yield_force = wp.min(
        axial_rigidity * yield_strain,
        failure_force * 0.7,
    )
    if extension <= yield_strain:
        return wp.min(axial_rigidity * extension, yield_force)
    normalized = wp.clamp(
        (extension - yield_strain) / (failure_strain - yield_strain),
        0.0,
        1.0,
    )
    return yield_force + (failure_force - yield_force) * wp.pow(
        normalized,
        post_yield_exponent,
    )


@wp.kernel
def _integrate(
    positions: wp.array(dtype=wp.vec3),
    previous_positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    inverse_masses: wp.array(dtype=float),
    gravity: wp.vec3,
    damping: float,
    maximum_displacement: float,
    dt: float,
):
    particle = wp.tid()
    position = positions[particle]
    previous_positions[particle] = position
    if inverse_masses[particle] <= 0.0:
        velocities[particle] = wp.vec3(0.0, 0.0, 0.0)
        return
    velocity = velocities[particle] * damping
    displacement = velocity * dt + gravity * (dt * dt)
    displacement_length = wp.length(displacement)
    if displacement_length > maximum_displacement:
        displacement = displacement * (maximum_displacement / displacement_length)
    positions[particle] = position + displacement


@wp.kernel
def _project_stretch_color(
    positions: wp.array(dtype=wp.vec3),
    inverse_masses: wp.array(dtype=float),
    rest_lengths: wp.array(dtype=float),
    axial_multipliers: wp.array(dtype=float),
    failure_forces: wp.array(dtype=float),
    active_segments: wp.array(dtype=wp.int32),
    lambdas: wp.array(dtype=float),
    color: int,
    axial_rigidity: float,
    yield_strain: float,
    failure_strain: float,
    post_yield_exponent: float,
    dt: float,
):
    segment = wp.tid()
    if segment % 2 != color or active_segments[segment] == 0:
        return
    first_index = segment
    second_index = segment + 1
    first = positions[first_index]
    second = positions[second_index]
    difference = second - first
    length = wp.length(difference)
    if length <= 1.0e-12:
        return
    rest = rest_lengths[segment]
    constraint = length - rest
    strain = wp.max(length / rest - 1.0, 0.0)
    force = _constitutive_force(
        strain,
        axial_rigidity,
        yield_strain,
        failure_strain,
        failure_forces[segment],
        post_yield_exponent,
    )
    stiffness = axial_rigidity / rest
    if strain > 1.0e-8 and force > 1.0e-8:
        stiffness = force / (strain * rest)
    stiffness = stiffness * axial_multipliers[segment]
    compliance = 1.0 / wp.max(stiffness, 1.0)
    alpha = compliance / (dt * dt)
    first_weight = inverse_masses[first_index]
    second_weight = inverse_masses[second_index]
    denominator = first_weight + second_weight + alpha
    if denominator <= 0.0:
        return
    old_lambda = lambdas[segment]
    delta_lambda = (-constraint - alpha * old_lambda) / denominator
    direction = difference / length
    positions[first_index] = first - direction * (first_weight * delta_lambda)
    positions[second_index] = second + direction * (second_weight * delta_lambda)
    lambdas[segment] = old_lambda + delta_lambda


@wp.func
def _project_one_stretch(
    segment: int,
    positions: wp.array(dtype=wp.vec3),
    inverse_masses: wp.array(dtype=float),
    rest_lengths: wp.array(dtype=float),
    axial_multipliers: wp.array(dtype=float),
    failure_forces: wp.array(dtype=float),
    active_segments: wp.array(dtype=wp.int32),
    lambdas: wp.array(dtype=float),
    axial_rigidity: float,
    yield_strain: float,
    failure_strain: float,
    post_yield_exponent: float,
    dt: float,
):
    if active_segments[segment] == 0:
        return
    first_index = segment
    second_index = segment + 1
    first = positions[first_index]
    second = positions[second_index]
    difference = second - first
    length = wp.length(difference)
    if length <= 1.0e-12:
        return
    rest = rest_lengths[segment]
    constraint = length - rest
    strain = wp.max(length / rest - 1.0, 0.0)
    force = _constitutive_force(
        strain,
        axial_rigidity,
        yield_strain,
        failure_strain,
        failure_forces[segment],
        post_yield_exponent,
    )
    stiffness = axial_rigidity / rest
    if strain > 1.0e-8 and force > 1.0e-8:
        stiffness = force / (strain * rest)
    stiffness = stiffness * axial_multipliers[segment]
    alpha = (1.0 / wp.max(stiffness, 1.0)) / (dt * dt)
    first_weight = inverse_masses[first_index]
    second_weight = inverse_masses[second_index]
    denominator = first_weight + second_weight + alpha
    if denominator <= 0.0:
        return
    old_lambda = lambdas[segment]
    delta_lambda = (-constraint - alpha * old_lambda) / denominator
    direction = difference / length
    positions[first_index] = first - direction * (first_weight * delta_lambda)
    positions[second_index] = second + direction * (second_weight * delta_lambda)
    lambdas[segment] = old_lambda + delta_lambda


@wp.kernel
def _project_stretch_serial(
    positions: wp.array(dtype=wp.vec3),
    inverse_masses: wp.array(dtype=float),
    rest_lengths: wp.array(dtype=float),
    axial_multipliers: wp.array(dtype=float),
    failure_forces: wp.array(dtype=float),
    active_segments: wp.array(dtype=wp.int32),
    lambdas: wp.array(dtype=float),
    segment_count: int,
    sweeps: int,
    axial_rigidity: float,
    yield_strain: float,
    failure_strain: float,
    post_yield_exponent: float,
    dt: float,
):
    """Forward/backward chain solve without inter-block synchronization error."""

    if wp.tid() != 0:
        return
    for _sweep in range(sweeps):
        for segment in range(segment_count):
            _project_one_stretch(
                segment,
                positions,
                inverse_masses,
                rest_lengths,
                axial_multipliers,
                failure_forces,
                active_segments,
                lambdas,
                axial_rigidity,
                yield_strain,
                failure_strain,
                post_yield_exponent,
                dt,
            )
        for reverse_index in range(segment_count):
            segment = segment_count - 1 - reverse_index
            _project_one_stretch(
                segment,
                positions,
                inverse_masses,
                rest_lengths,
                axial_multipliers,
                failure_forces,
                active_segments,
                lambdas,
                axial_rigidity,
                yield_strain,
                failure_strain,
                post_yield_exponent,
                dt,
            )


@wp.kernel
def _project_stretch_global(
    positions: wp.array(dtype=wp.vec3),
    inverse_masses: wp.array(dtype=float),
    rest_lengths: wp.array(dtype=float),
    axial_multipliers: wp.array(dtype=float),
    failure_forces: wp.array(dtype=float),
    active_segments: wp.array(dtype=wp.int32),
    lambdas: wp.array(dtype=float),
    directions: wp.array(dtype=wp.vec3),
    diagonal: wp.array(dtype=float),
    off_diagonal: wp.array(dtype=float),
    right_hand_side: wp.array(dtype=float),
    c_prime: wp.array(dtype=float),
    d_prime: wp.array(dtype=float),
    delta_lambdas: wp.array(dtype=float),
    segment_count: int,
    particle_mass: float,
    axial_rigidity: float,
    yield_strain: float,
    failure_strain: float,
    post_yield_exponent: float,
    dt: float,
):
    """Solve the coupled axial XPBD system as one tridiagonal block."""

    if wp.tid() != 0:
        return
    for segment in range(segment_count):
        first = positions[segment]
        second = positions[segment + 1]
        difference = second - first
        length = wp.length(difference)
        if active_segments[segment] == 0 or length <= 1.0e-12:
            directions[segment] = wp.vec3(1.0, 0.0, 0.0)
            diagonal[segment] = 1.0
            right_hand_side[segment] = 0.0
        else:
            direction = difference / length
            directions[segment] = direction
            rest = rest_lengths[segment]
            constraint = length - rest
            strain = wp.max(length / rest - 1.0, 0.0)
            force = _constitutive_force(
                strain,
                axial_rigidity,
                yield_strain,
                failure_strain,
                failure_forces[segment],
                post_yield_exponent,
            )
            stiffness = axial_rigidity / rest
            if strain > 1.0e-8 and force > 1.0e-8:
                stiffness = force / (strain * rest)
            stiffness = stiffness * axial_multipliers[segment]
            alpha = (
                (1.0 / wp.max(stiffness, 1.0))
                / (dt * dt)
                * particle_mass
            )
            # A two-ended straight chain has one redundant axial constraint.
            # Keep the nondimensional compliance above FP32 epsilon so the
            # tridiagonal system remains positive definite on consumer GPUs.
            alpha = wp.max(alpha, 1.0e-5)
            first_weight = inverse_masses[segment] * particle_mass
            second_weight = inverse_masses[segment + 1] * particle_mass
            diagonal[segment] = first_weight + second_weight + alpha
            right_hand_side[segment] = (
                -constraint - alpha * lambdas[segment]
            )

    for segment in range(segment_count - 1):
        value = float(0.0)
        if (
            active_segments[segment] != 0
            and active_segments[segment + 1] != 0
        ):
            value = (
                -inverse_masses[segment + 1]
                * particle_mass
                * wp.dot(directions[segment], directions[segment + 1])
            )
        off_diagonal[segment] = value

    denominator = diagonal[0]
    if wp.abs(denominator) < 1.0e-8:
        denominator = 1.0
    c_prime[0] = off_diagonal[0] / denominator
    d_prime[0] = right_hand_side[0] / denominator
    for segment in range(1, segment_count):
        denominator = (
            diagonal[segment]
            - off_diagonal[segment - 1] * c_prime[segment - 1]
        )
        if wp.abs(denominator) < 1.0e-8:
            denominator = 1.0
        if segment < segment_count - 1:
            c_prime[segment] = off_diagonal[segment] / denominator
        else:
            c_prime[segment] = 0.0
        d_prime[segment] = (
            right_hand_side[segment]
            - off_diagonal[segment - 1] * d_prime[segment - 1]
        ) / denominator

    delta_lambdas[segment_count - 1] = d_prime[segment_count - 1]
    for reverse_index in range(1, segment_count):
        segment = segment_count - 1 - reverse_index
        delta_lambdas[segment] = (
            d_prime[segment]
            - c_prime[segment] * delta_lambdas[segment + 1]
        )
    for segment in range(segment_count):
        lambdas[segment] = lambdas[segment] + delta_lambdas[segment]

    for particle in range(segment_count + 1):
        correction = wp.vec3(0.0, 0.0, 0.0)
        if particle > 0 and active_segments[particle - 1] != 0:
            correction = (
                correction
                + directions[particle - 1] * delta_lambdas[particle - 1]
            )
        if particle < segment_count and active_segments[particle] != 0:
            correction = (
                correction - directions[particle] * delta_lambdas[particle]
            )
        positions[particle] = (
            positions[particle]
            + correction * (inverse_masses[particle] * particle_mass)
        )


@wp.kernel
def _project_bend_color(
    positions: wp.array(dtype=wp.vec3),
    inverse_masses: wp.array(dtype=float),
    bend_multipliers: wp.array(dtype=float),
    active_segments: wp.array(dtype=wp.int32),
    lambdas: wp.array(dtype=wp.vec3),
    color: int,
    bend_compliance: float,
    dt: float,
):
    bend = wp.tid()
    if bend % 3 != color:
        return
    if active_segments[bend] == 0 or active_segments[bend + 1] == 0:
        return
    first_index = bend
    center_index = bend + 1
    last_index = bend + 2
    first_weight = inverse_masses[first_index]
    center_weight = inverse_masses[center_index]
    last_weight = inverse_masses[last_index]
    multiplier = bend_multipliers[bend]
    alpha = bend_compliance / (wp.max(multiplier, 1.0e-6) * dt * dt)
    denominator = first_weight + 4.0 * center_weight + last_weight + alpha
    if denominator <= 0.0:
        return
    first = positions[first_index]
    center = positions[center_index]
    last = positions[last_index]
    constraint = first - center * 2.0 + last
    old_lambda = lambdas[bend]
    delta_lambda = (-constraint - old_lambda * alpha) / denominator
    positions[first_index] = first + delta_lambda * first_weight
    positions[center_index] = center - delta_lambda * (2.0 * center_weight)
    positions[last_index] = last + delta_lambda * last_weight
    lambdas[bend] = old_lambda + delta_lambda


@wp.kernel
def _solve_contact_pairs(
    positions: wp.array(dtype=wp.vec3),
    previous_positions: wp.array(dtype=wp.vec3),
    inverse_masses: wp.array(dtype=float),
    pair_first: wp.array(dtype=wp.int32),
    pair_second: wp.array(dtype=wp.int32),
    pair_active: wp.array(dtype=wp.int32),
    pair_penetrating: wp.array(dtype=wp.int32),
    pair_distance: wp.array(dtype=float),
    delta_first_start: wp.array(dtype=wp.vec3),
    delta_first_end: wp.array(dtype=wp.vec3),
    delta_second_start: wp.array(dtype=wp.vec3),
    delta_second_end: wp.array(dtype=wp.vec3),
    physical_diameter: float,
    detection_distance: float,
    friction: float,
):
    pair = wp.tid()
    first_segment = pair_first[pair]
    second_segment = pair_second[pair]
    first_start = positions[first_segment]
    first_end = positions[first_segment + 1]
    second_start = positions[second_segment]
    second_end = positions[second_segment + 1]
    parameters = _closest_segment_parameters(
        first_start,
        first_end,
        second_start,
        second_end,
    )
    first_amount = parameters[0]
    second_amount = parameters[1]
    first_closest = first_start + (first_end - first_start) * first_amount
    second_closest = second_start + (second_end - second_start) * second_amount
    separation = first_closest - second_closest
    distance = wp.length(separation)
    pair_distance[pair] = distance
    pair_active[pair] = 0
    pair_penetrating[pair] = 0
    zero = wp.vec3(0.0, 0.0, 0.0)
    delta_first_start[pair] = zero
    delta_first_end[pair] = zero
    delta_second_start[pair] = zero
    delta_second_end[pair] = zero
    if distance <= detection_distance:
        pair_active[pair] = 1
    penetration = physical_diameter - distance
    if penetration <= 0.0:
        return
    pair_penetrating[pair] = 1

    normal = wp.vec3(0.0, 1.0, 0.0)
    if distance > 1.0e-10:
        normal = separation / distance
    else:
        first_direction = first_end - first_start
        second_direction = second_end - second_start
        fallback = wp.cross(first_direction, second_direction)
        fallback_length = wp.length(fallback)
        if fallback_length > 1.0e-12:
            normal = fallback / fallback_length

    first_start_amount = 1.0 - first_amount
    second_start_amount = 1.0 - second_amount
    first_start_weight = inverse_masses[first_segment]
    first_end_weight = inverse_masses[first_segment + 1]
    second_start_weight = inverse_masses[second_segment]
    second_end_weight = inverse_masses[second_segment + 1]
    effective_weight = (
        first_start_weight * first_start_amount * first_start_amount
        + first_end_weight * first_amount * first_amount
        + second_start_weight * second_start_amount * second_start_amount
        + second_end_weight * second_amount * second_amount
    )
    if effective_weight <= 0.0:
        return

    normal_scale = penetration / effective_weight
    first_correction = normal * normal_scale
    second_correction = -first_correction

    first_previous = (
        previous_positions[first_segment] * first_start_amount
        + previous_positions[first_segment + 1] * first_amount
    )
    second_previous = (
        previous_positions[second_segment] * second_start_amount
        + previous_positions[second_segment + 1] * second_amount
    )
    relative_displacement = (
        (first_closest - first_previous) - (second_closest - second_previous)
    )
    tangential = relative_displacement - normal * wp.dot(
        relative_displacement,
        normal,
    )
    tangential_length = wp.length(tangential)
    if tangential_length > 1.0e-12:
        friction_distance = wp.min(tangential_length, friction * penetration)
        friction_correction = (
            -tangential
            * (friction_distance / tangential_length)
            / effective_weight
        )
        first_correction = first_correction + friction_correction
        second_correction = second_correction - friction_correction

    delta_first_start[pair] = (
        first_correction * (first_start_weight * first_start_amount)
    )
    delta_first_end[pair] = first_correction * (
        first_end_weight * first_amount
    )
    delta_second_start[pair] = second_correction * (
        second_start_weight * second_start_amount
    )
    delta_second_end[pair] = second_correction * (
        second_end_weight * second_amount
    )


@wp.kernel
def _gather_contact_corrections(
    positions: wp.array(dtype=wp.vec3),
    particle_pair_offsets: wp.array(dtype=wp.int32),
    particle_pair_slots: wp.array(dtype=wp.int32),
    pair_penetrating: wp.array(dtype=wp.int32),
    delta_first_start: wp.array(dtype=wp.vec3),
    delta_first_end: wp.array(dtype=wp.vec3),
    delta_second_start: wp.array(dtype=wp.vec3),
    delta_second_end: wp.array(dtype=wp.vec3),
    relaxation: float,
):
    particle = wp.tid()
    correction = wp.vec3(0.0, 0.0, 0.0)
    active_count = int(0)
    begin = particle_pair_offsets[particle]
    end = particle_pair_offsets[particle + 1]
    for entry in range(begin, end):
        encoded = particle_pair_slots[entry]
        pair = encoded // 4
        slot = encoded - pair * 4
        if pair_penetrating[pair] != 0:
            active_count = active_count + 1
            if slot == 0:
                correction = correction + delta_first_start[pair]
            elif slot == 1:
                correction = correction + delta_first_end[pair]
            elif slot == 2:
                correction = correction + delta_second_start[pair]
            else:
                correction = correction + delta_second_end[pair]
    if active_count > 0:
        positions[particle] = positions[particle] + correction * (
            relaxation / float(active_count)
        )


@wp.kernel
def _project_attachments(
    positions: wp.array(dtype=wp.vec3),
    inverse_masses: wp.array(dtype=float),
    attachment_active: wp.array(dtype=wp.int32),
    attachment_targets: wp.array(dtype=wp.vec3),
    attachment_compliance: wp.array(dtype=float),
    attachment_lambdas: wp.array(dtype=wp.vec3),
    dt: float,
):
    particle = wp.tid()
    if attachment_active[particle] == 0:
        return
    if attachment_compliance[particle] <= 0.0:
        positions[particle] = attachment_targets[particle]
        attachment_lambdas[particle] = wp.vec3(0.0, 0.0, 0.0)
        return
    weight = inverse_masses[particle]
    alpha = attachment_compliance[particle] / (dt * dt)
    denominator = weight + alpha
    if denominator <= 0.0:
        return
    constraint = positions[particle] - attachment_targets[particle]
    old_lambda = attachment_lambdas[particle]
    delta_lambda = (-constraint - old_lambda * alpha) / denominator
    positions[particle] = positions[particle] + delta_lambda * weight
    attachment_lambdas[particle] = old_lambda + delta_lambda


@wp.kernel
def _project_ground(
    positions: wp.array(dtype=wp.vec3),
    previous_positions: wp.array(dtype=wp.vec3),
    inverse_masses: wp.array(dtype=float),
    ground_height: float,
    radius: float,
    friction: float,
):
    particle = wp.tid()
    if inverse_masses[particle] <= 0.0:
        return
    position = positions[particle]
    minimum_height = ground_height + radius
    if position[1] >= minimum_height:
        return
    previous = previous_positions[particle]
    displacement = position - previous
    tangential = wp.vec3(displacement[0], 0.0, displacement[2])
    tangential_length = wp.length(tangential)
    penetration = minimum_height - position[1]
    if tangential_length > 1.0e-12:
        friction_distance = wp.min(tangential_length, friction * penetration)
        position = position - tangential * (friction_distance / tangential_length)
    position[1] = minimum_height
    positions[particle] = position


@wp.kernel
def _update_velocities(
    positions: wp.array(dtype=wp.vec3),
    previous_positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    maximum_speed: float,
    dt: float,
):
    particle = wp.tid()
    velocity = (positions[particle] - previous_positions[particle]) / dt
    speed = wp.length(velocity)
    if speed > maximum_speed:
        velocity = velocity * (maximum_speed / speed)
    velocities[particle] = velocity


@wp.kernel
def _update_contact_history(
    pair_first: wp.array(dtype=wp.int32),
    pair_second: wp.array(dtype=wp.int32),
    pair_active: wp.array(dtype=wp.int32),
    contact_dwell: wp.array(dtype=float),
    compacted_pairs: wp.array(dtype=wp.int32),
    minimum_knot_separation: int,
    compaction_dwell: float,
    dt: float,
):
    pair = wp.tid()
    if pair_second[pair] - pair_first[pair] < minimum_knot_separation:
        return
    dwell = contact_dwell[pair]
    if pair_active[pair] != 0:
        dwell = dwell + dt
    elif compacted_pairs[pair] == 0:
        dwell = wp.max(0.0, dwell - dt)
    contact_dwell[pair] = dwell
    if dwell >= compaction_dwell:
        compacted_pairs[pair] = 1


@wp.kernel
def _update_knot_strength(
    segment_pair_offsets: wp.array(dtype=wp.int32),
    segment_pair_indices: wp.array(dtype=wp.int32),
    compacted_pairs: wp.array(dtype=wp.int32),
    knot_strength: wp.array(dtype=float),
    knot_efficiency: float,
):
    segment = wp.tid()
    begin = segment_pair_offsets[segment]
    end = segment_pair_offsets[segment + 1]
    for entry in range(begin, end):
        if compacted_pairs[segment_pair_indices[entry]] != 0:
            knot_strength[segment] = wp.min(
                knot_strength[segment],
                knot_efficiency,
            )


@wp.kernel
def _update_segment_contact_history(
    segment_pair_offsets: wp.array(dtype=wp.int32),
    segment_pair_indices: wp.array(dtype=wp.int32),
    pair_first: wp.array(dtype=wp.int32),
    pair_second: wp.array(dtype=wp.int32),
    pair_active: wp.array(dtype=wp.int32),
    contact_dwell: wp.array(dtype=float),
    compacted_segments: wp.array(dtype=wp.int32),
    knot_strength: wp.array(dtype=float),
    minimum_knot_separation: int,
    compaction_dwell: float,
    knot_efficiency: float,
    dt: float,
):
    """Track a moving contact patch without requiring one pair to stay fixed."""

    segment = wp.tid()
    has_nonlocal_contact = int(0)
    begin = segment_pair_offsets[segment]
    end = segment_pair_offsets[segment + 1]
    for entry in range(begin, end):
        pair = segment_pair_indices[entry]
        if (
            pair_second[pair] - pair_first[pair] >= minimum_knot_separation
            and pair_active[pair] != 0
        ):
            has_nonlocal_contact = 1
    dwell = contact_dwell[segment]
    if has_nonlocal_contact != 0:
        dwell = dwell + dt
    elif compacted_segments[segment] == 0:
        dwell = wp.max(0.0, dwell - dt)
    contact_dwell[segment] = dwell
    if dwell >= compaction_dwell:
        compacted_segments[segment] = 1
        knot_strength[segment] = wp.min(
            knot_strength[segment],
            knot_efficiency,
        )


@wp.kernel
def _evaluate_breakage(
    positions: wp.array(dtype=wp.vec3),
    rest_lengths: wp.array(dtype=float),
    active_segments: wp.array(dtype=wp.int32),
    broken_segments: wp.array(dtype=wp.int32),
    strand_failed: wp.array(dtype=wp.int32),
    stretch_lambdas: wp.array(dtype=float),
    base_failure_force: float,
    knot_strength: wp.array(dtype=float),
    damage_strength: wp.array(dtype=float),
    measured_force: wp.array(dtype=float),
    axial_rigidity: float,
    yield_strain: float,
    failure_strain: float,
    post_yield_exponent: float,
    dt: float,
):
    if wp.tid() != 0:
        return
    segment_count = active_segments.shape[0]
    failure_candidate = int(-1)
    maximum_ratio = float(0.0)
    for segment in range(segment_count):
        if active_segments[segment] == 0:
            measured_force[segment] = 0.0
        else:
            length = wp.length(positions[segment + 1] - positions[segment])
            strain = wp.max(
                length / rest_lengths[segment] - 1.0,
                0.0,
            )
            failure_force = (
                base_failure_force
                * knot_strength[segment]
                * damage_strength[segment]
            )
            force = _constitutive_force(
                strain,
                axial_rigidity,
                yield_strain,
                failure_strain,
                failure_force,
                post_yield_exponent,
            )
            measured_force[segment] = force
            ratio = wp.max(
                strain / wp.max(failure_strain, 1.0e-8),
                force / wp.max(failure_force, 1.0e-8),
            )
            if ratio >= 1.0 and ratio > maximum_ratio:
                maximum_ratio = ratio
                failure_candidate = segment
    if strand_failed[0] == 0 and failure_candidate >= 0:
        active_segments[failure_candidate] = 0
        broken_segments[failure_candidate] = 1
        strand_failed[0] = 1


@wp.kernel
def _evaluate_attachment_failure(
    attachment_active: wp.array(dtype=wp.int32),
    attachment_break_force: wp.array(dtype=float),
    attachment_failed: wp.array(dtype=wp.int32),
    attachment_failure_force: wp.array(dtype=float),
    inverse_masses: wp.array(dtype=float),
    segment_force: wp.array(dtype=float),
    particle_mass: float,
):
    particle = wp.tid()
    if (
        attachment_active[particle] == 0
        or attachment_break_force[particle] <= 0.0
    ):
        return
    force = float(0.0)
    if particle > 0:
        force = wp.max(force, segment_force[particle - 1])
    if particle < segment_force.shape[0]:
        force = wp.max(force, segment_force[particle])
    if force >= attachment_break_force[particle]:
        attachment_active[particle] = 0
        attachment_failed[particle] = 1
        attachment_failure_force[particle] = force
        inverse_masses[particle] = 1.0 / particle_mass


@dataclass(frozen=True)
class WarpSutureConfig:
    particle_count: int
    segment_count: int
    spacing_m: float
    diameter_m: float
    particle_mass_kg: float
    axial_rigidity_n: float
    flexural_rigidity_n_m2: float
    yield_strain: float
    failure_strain: float
    failure_force_n: float
    post_yield_exponent: float
    self_friction: float
    surface_friction: float
    detection_distance_m: float
    knot_minimum_separation: int
    knot_compaction_dwell_s: float
    knot_strength_efficiency: float
    swage_segment_count: int
    swage_axial_multiplier: float
    swage_bend_multiplier: float
    swage_pullout_force_n: float

    @classmethod
    def from_profile(cls, profile: dict[str, Any]) -> "WarpSutureConfig":
        derived = derive(profile)
        self_friction = profile["contact"]["load_dependent_self_friction"]
        runtime = profile["runtime_detection"]
        swage = profile["swage"]
        return cls(
            particle_count=derived.segment_count + 1,
            segment_count=derived.segment_count,
            spacing_m=derived.segment_spacing_m,
            diameter_m=derived.diameter_m,
            particle_mass_kg=derived.mass_kg / (derived.segment_count + 1),
            axial_rigidity_n=derived.axial_rigidity_n,
            flexural_rigidity_n_m2=float(
                profile["material"]["flexural_rigidity_n_m2"]
            ),
            yield_strain=float(profile["tension"]["yield_strain"]),
            failure_strain=float(profile["tension"]["failure_strain"]),
            failure_force_n=float(profile["tension"]["straight_failure_load_n"]),
            post_yield_exponent=float(
                profile["tension"]["post_yield_shape_exponent"]
            ),
            self_friction=float(self_friction["low_load_coefficient"]),
            surface_friction=float(profile["contact"]["dynamic_friction"]),
            detection_distance_m=float(
                runtime["self_contact_centerline_distance_m"]
            ),
            knot_minimum_separation=int(
                runtime["knot_minimum_index_separation"]
            ),
            knot_compaction_dwell_s=float(
                runtime["knot_compaction_dwell_s"]
            ),
            knot_strength_efficiency=float(
                profile["knot"]["nominal_strength_efficiency"]
            ),
            swage_segment_count=derived.swage_segment_count,
            swage_axial_multiplier=float(swage["axial_stiffness_multiplier"]),
            swage_bend_multiplier=float(swage["bending_stiffness_multiplier"]),
            swage_pullout_force_n=float(swage["pullout_force_n_seed"]),
        )


def _contact_pairs(
    segment_count: int,
    *,
    excluded_index_distance: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    first: list[int] = []
    second: list[int] = []
    for left in range(segment_count):
        for right in range(left + excluded_index_distance + 1, segment_count):
            first.append(left)
            second.append(right)
    return np.asarray(first, dtype=np.int32), np.asarray(second, dtype=np.int32)


def _particle_pair_csr(
    particle_count: int,
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    entries: list[list[int]] = [[] for _ in range(particle_count)]
    for pair, (left, right) in enumerate(zip(first.tolist(), second.tolist())):
        entries[left].append(pair * 4)
        entries[left + 1].append(pair * 4 + 1)
        entries[right].append(pair * 4 + 2)
        entries[right + 1].append(pair * 4 + 3)
    offsets = [0]
    flat: list[int] = []
    for particle_entries in entries:
        flat.extend(particle_entries)
        offsets.append(len(flat))
    return np.asarray(offsets, dtype=np.int32), np.asarray(flat, dtype=np.int32)


def _segment_pair_csr(
    segment_count: int,
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    entries: list[list[int]] = [[] for _ in range(segment_count)]
    for pair, (left, right) in enumerate(zip(first.tolist(), second.tolist())):
        entries[left].append(pair)
        entries[right].append(pair)
    offsets = [0]
    flat: list[int] = []
    for segment_entries in entries:
        flat.extend(segment_entries)
        offsets.append(len(flat))
    return np.asarray(offsets, dtype=np.int32), np.asarray(flat, dtype=np.int32)


def _rotation_matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert one right-handed 3x3 frame to a normalized WXYZ quaternion."""

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            (
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ),
            dtype=np.float64,
        )
    else:
        diagonal = np.diag(matrix)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = math.sqrt(
                max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0)
            ) * 2.0
            quaternion = np.asarray(
                (
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ),
                dtype=np.float64,
            )
        elif axis == 1:
            scale = math.sqrt(
                max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0.0)
            ) * 2.0
            quaternion = np.asarray(
                (
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ),
                dtype=np.float64,
            )
        else:
            scale = math.sqrt(
                max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0.0)
            ) * 2.0
            quaternion = np.asarray(
                (
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ),
                dtype=np.float64,
            )
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        return np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32)
    return (quaternion / norm).astype(np.float32)


class WarpSuture:
    """Fixed-topology, GPU-native Dr.Anmar strand simulation."""

    def __init__(
        self,
        profile: dict[str, Any],
        positions_m: Sequence[Sequence[float]],
        *,
        device: str = "cuda:0",
        excluded_contact_index_distance: int = 2,
    ) -> None:
        self.profile = profile
        self.config = WarpSutureConfig.from_profile(profile)
        self.device = device
        positions = np.asarray(positions_m, dtype=np.float32)
        expected_shape = (self.config.particle_count, 3)
        if positions.shape != expected_shape:
            raise ValueError(
                f"expected particle positions {expected_shape}, got {positions.shape}"
            )
        segment_lengths = np.linalg.norm(
            positions[1:] - positions[:-1],
            axis=1,
        )
        if not np.all(np.isfinite(positions)) or np.any(segment_lengths <= 0.0):
            raise ValueError("initial strand positions must be finite and nondegenerate")

        first, second = _contact_pairs(
            self.config.segment_count,
            excluded_index_distance=excluded_contact_index_distance,
        )
        particle_offsets, particle_slots = _particle_pair_csr(
            self.config.particle_count,
            first,
            second,
        )
        segment_offsets, segment_pairs = _segment_pair_csr(
            self.config.segment_count,
            first,
            second,
        )
        self.pair_count = len(first)
        self._host_attachment_active = np.zeros(
            self.config.particle_count,
            dtype=np.int32,
        )
        self._host_attachment_targets = positions.copy()
        self._host_attachment_compliance = np.zeros(
            self.config.particle_count,
            dtype=np.float32,
        )
        self._host_attachment_break_force = np.zeros(
            self.config.particle_count,
            dtype=np.float32,
        )
        self._host_inverse_masses = np.full(
            self.config.particle_count,
            1.0 / self.config.particle_mass_kg,
            dtype=np.float32,
        )

        with wp.ScopedDevice(device):
            self.positions = wp.array(positions, dtype=wp.vec3, device=device)
            self.previous_positions = wp.array(
                positions,
                dtype=wp.vec3,
                device=device,
            )
            self.velocities = wp.zeros(
                self.config.particle_count,
                dtype=wp.vec3,
                device=device,
            )
            self.inverse_masses = wp.array(
                self._host_inverse_masses,
                dtype=float,
                device=device,
            )
            self.rest_lengths = wp.array(
                segment_lengths.astype(np.float32),
                dtype=float,
                device=device,
            )
            axial_multipliers = np.ones(
                self.config.segment_count,
                dtype=np.float32,
            )
            axial_multipliers[: self.config.swage_segment_count] = (
                self.config.swage_axial_multiplier
            )
            bend_multipliers = np.ones(
                self.config.segment_count - 1,
                dtype=np.float32,
            )
            bend_multipliers[: self.config.swage_segment_count] = (
                self.config.swage_bend_multiplier
            )
            self.axial_multipliers = wp.array(
                axial_multipliers,
                dtype=float,
                device=device,
            )
            self.bend_multipliers = wp.array(
                bend_multipliers,
                dtype=float,
                device=device,
            )
            self.active_segments = wp.ones(
                self.config.segment_count,
                dtype=wp.int32,
                device=device,
            )
            self.broken_segments = wp.zeros(
                self.config.segment_count,
                dtype=wp.int32,
                device=device,
            )
            self.strand_failed = wp.zeros(
                1,
                dtype=wp.int32,
                device=device,
            )
            self.stretch_lambdas = wp.zeros(
                self.config.segment_count,
                dtype=float,
                device=device,
            )
            self.stretch_directions = wp.zeros(
                self.config.segment_count,
                dtype=wp.vec3,
                device=device,
            )
            self.stretch_diagonal = wp.zeros(
                self.config.segment_count,
                dtype=float,
                device=device,
            )
            self.stretch_off_diagonal = wp.zeros(
                self.config.segment_count - 1,
                dtype=float,
                device=device,
            )
            self.stretch_right_hand_side = wp.zeros(
                self.config.segment_count,
                dtype=float,
                device=device,
            )
            self.stretch_c_prime = wp.zeros(
                self.config.segment_count,
                dtype=float,
                device=device,
            )
            self.stretch_d_prime = wp.zeros(
                self.config.segment_count,
                dtype=float,
                device=device,
            )
            self.stretch_delta_lambdas = wp.zeros(
                self.config.segment_count,
                dtype=float,
                device=device,
            )
            self.bend_lambdas = wp.zeros(
                self.config.segment_count - 1,
                dtype=wp.vec3,
                device=device,
            )
            self.measured_force = wp.zeros(
                self.config.segment_count,
                dtype=float,
                device=device,
            )
            self.knot_strength = wp.ones(
                self.config.segment_count,
                dtype=float,
                device=device,
            )
            self.damage_strength = wp.ones(
                self.config.segment_count,
                dtype=float,
                device=device,
            )
            self.pair_first = wp.array(first, dtype=wp.int32, device=device)
            self.pair_second = wp.array(second, dtype=wp.int32, device=device)
            self.pair_active = wp.zeros(
                self.pair_count,
                dtype=wp.int32,
                device=device,
            )
            self.pair_penetrating = wp.zeros(
                self.pair_count,
                dtype=wp.int32,
                device=device,
            )
            self.pair_distance = wp.zeros(
                self.pair_count,
                dtype=float,
                device=device,
            )
            self.contact_dwell = wp.zeros(
                self.pair_count,
                dtype=float,
                device=device,
            )
            self.compacted_pairs = wp.zeros(
                self.pair_count,
                dtype=wp.int32,
                device=device,
            )
            self.segment_contact_dwell = wp.zeros(
                self.config.segment_count,
                dtype=float,
                device=device,
            )
            self.compacted_segments = wp.zeros(
                self.config.segment_count,
                dtype=wp.int32,
                device=device,
            )
            self.delta_first_start = wp.zeros(
                self.pair_count,
                dtype=wp.vec3,
                device=device,
            )
            self.delta_first_end = wp.zeros(
                self.pair_count,
                dtype=wp.vec3,
                device=device,
            )
            self.delta_second_start = wp.zeros(
                self.pair_count,
                dtype=wp.vec3,
                device=device,
            )
            self.delta_second_end = wp.zeros(
                self.pair_count,
                dtype=wp.vec3,
                device=device,
            )
            self.particle_pair_offsets = wp.array(
                particle_offsets,
                dtype=wp.int32,
                device=device,
            )
            self.particle_pair_slots = wp.array(
                particle_slots,
                dtype=wp.int32,
                device=device,
            )
            self.segment_pair_offsets = wp.array(
                segment_offsets,
                dtype=wp.int32,
                device=device,
            )
            self.segment_pair_indices = wp.array(
                segment_pairs,
                dtype=wp.int32,
                device=device,
            )
            self.attachment_active = wp.array(
                self._host_attachment_active,
                dtype=wp.int32,
                device=device,
            )
            self.attachment_targets = wp.array(
                self._host_attachment_targets,
                dtype=wp.vec3,
                device=device,
            )
            self.attachment_compliance = wp.array(
                self._host_attachment_compliance,
                dtype=float,
                device=device,
            )
            self.attachment_lambdas = wp.zeros(
                self.config.particle_count,
                dtype=wp.vec3,
                device=device,
            )
            self.attachment_break_force = wp.array(
                self._host_attachment_break_force,
                dtype=float,
                device=device,
            )
            self.attachment_failed = wp.zeros(
                self.config.particle_count,
                dtype=wp.int32,
                device=device,
            )
            self.attachment_failure_force = wp.zeros(
                self.config.particle_count,
                dtype=float,
                device=device,
            )

    def _sync_attachments(self) -> None:
        wp.copy(
            self.attachment_active,
            wp.array(
                self._host_attachment_active,
                dtype=wp.int32,
                device="cpu",
            ),
        )
        wp.copy(
            self.attachment_targets,
            wp.array(
                self._host_attachment_targets,
                dtype=wp.vec3,
                device="cpu",
            ),
        )
        wp.copy(
            self.attachment_compliance,
            wp.array(
                self._host_attachment_compliance,
                dtype=float,
                device="cpu",
            ),
        )
        wp.copy(
            self.inverse_masses,
            wp.array(
                self._host_inverse_masses,
                dtype=float,
                device="cpu",
            ),
        )
        wp.copy(
            self.attachment_break_force,
            wp.array(
                self._host_attachment_break_force,
                dtype=float,
                device="cpu",
            ),
        )

    def set_attachment(
        self,
        particle: int,
        target_m: Sequence[float],
        *,
        compliance_m_n: float = 0.0,
        break_force_n: float | None = None,
    ) -> None:
        if not 0 <= particle < self.config.particle_count:
            raise IndexError(particle)
        target = np.asarray(target_m, dtype=np.float32)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("attachment target must be a finite three-vector")
        self._host_attachment_active[particle] = 1
        self._host_attachment_targets[particle] = target
        self._host_attachment_compliance[particle] = max(
            0.0,
            float(compliance_m_n),
        )
        self._host_attachment_break_force[particle] = (
            0.0 if break_force_n is None else max(0.0, float(break_force_n))
        )
        if compliance_m_n <= 0.0:
            self._host_inverse_masses[particle] = 0.0
        else:
            self._host_inverse_masses[particle] = (
                1.0 / self.config.particle_mass_kg
            )
        self._sync_attachments()

    def clear_attachment(self, particle: int) -> None:
        if not 0 <= particle < self.config.particle_count:
            raise IndexError(particle)
        self._host_attachment_active[particle] = 0
        self._host_attachment_break_force[particle] = 0.0
        self._host_inverse_masses[particle] = (
            1.0 / self.config.particle_mass_kg
        )
        self._sync_attachments()

    def set_needle_connection(
        self,
        suture_exit_m: Sequence[float],
        *,
        compliance_m_n: float = 0.0,
    ) -> None:
        """Attach the swaged strand endpoint to the DrAnmar Needle exit."""

        self.set_attachment(
            0,
            suture_exit_m,
            compliance_m_n=compliance_m_n,
            break_force_n=self.config.swage_pullout_force_n,
        )

    def set_instrument_grasp(
        self,
        particles: Iterable[int],
        target_m: Sequence[float],
        *,
        compliance_m_n: float = 0.0,
    ) -> None:
        """Create a transferable kinematic grasp over selected strand nodes."""

        particle_indices = [int(particle) for particle in particles]
        if not particle_indices:
            raise ValueError("an instrument grasp requires at least one particle")
        for particle in particle_indices:
            if not 0 <= particle < self.config.particle_count:
                raise IndexError(particle)
        target = np.asarray(target_m, dtype=np.float32)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("grasp target must be a finite three-vector")
        current = self.positions.numpy()[particle_indices]
        offsets = current - current.mean(axis=0)
        for particle, offset in zip(particle_indices, offsets):
            self._host_attachment_active[particle] = 1
            self._host_attachment_targets[particle] = target + offset
            self._host_attachment_compliance[particle] = max(
                0.0,
                float(compliance_m_n),
            )
            self._host_attachment_break_force[particle] = 0.0
            if compliance_m_n <= 0.0:
                self._host_inverse_masses[particle] = 0.0
            else:
                self._host_inverse_masses[particle] = (
                    1.0 / self.config.particle_mass_kg
                )
        self._sync_attachments()

    def release_instrument_grasp(self, particles: Iterable[int]) -> None:
        for particle in particles:
            self._host_attachment_active[int(particle)] = 0
            self._host_attachment_break_force[int(particle)] = 0.0
            self._host_inverse_masses[int(particle)] = (
                1.0 / self.config.particle_mass_kg
            )
        self._sync_attachments()

    def set_segment_damage_strength(
        self,
        strength_fractions: Sequence[float],
    ) -> None:
        values = np.asarray(strength_fractions, dtype=np.float32)
        if values.shape != (self.config.segment_count,):
            raise ValueError("damage strength array has the wrong shape")
        values = np.clip(values, 0.05, 1.0)
        wp.copy(
            self.damage_strength,
            wp.array(values, dtype=float, device="cpu"),
        )

    def step(
        self,
        *,
        dt_s: float = 1.0 / 240.0,
        substeps: int = 4,
        solver_iterations: int = 10,
        gravity_m_s2: Sequence[float] = (0.0, -9.81, 0.0),
        ground_height_m: float | None = None,
    ) -> None:
        if dt_s <= 0.0 or substeps <= 0 or solver_iterations <= 0:
            raise ValueError("time step, substeps, and iterations must be positive")
        sub_dt = float(dt_s) / int(substeps)
        maximum_displacement = self.config.diameter_m * 0.45
        maximum_speed = maximum_displacement / sub_dt
        damping = math.exp(
            -float(self.profile["material"]["linear_velocity_damping"])
            * sub_dt
        ) * 0.995
        gravity = wp.vec3(*map(float, gravity_m_s2))
        bend_compliance = (
            self.config.spacing_m**3
            / self.config.flexural_rigidity_n_m2
        )
        for _ in range(substeps):
            self.stretch_lambdas.zero_()
            self.bend_lambdas.zero_()
            self.attachment_lambdas.zero_()
            wp.launch(
                _integrate,
                dim=self.config.particle_count,
                inputs=[
                    self.positions,
                    self.previous_positions,
                    self.velocities,
                    self.inverse_masses,
                    gravity,
                    damping,
                    maximum_displacement,
                    sub_dt,
                ],
                device=self.device,
            )
            for _iteration in range(solver_iterations):
                wp.launch(
                    _project_stretch_global,
                    dim=1,
                    inputs=[
                        self.positions,
                        self.inverse_masses,
                        self.rest_lengths,
                        self.axial_multipliers,
                        self.failure_forces(),
                        self.active_segments,
                        self.stretch_lambdas,
                        self.stretch_directions,
                        self.stretch_diagonal,
                        self.stretch_off_diagonal,
                        self.stretch_right_hand_side,
                        self.stretch_c_prime,
                        self.stretch_d_prime,
                        self.stretch_delta_lambdas,
                        self.config.segment_count,
                        self.config.particle_mass_kg,
                        self.config.axial_rigidity_n,
                        self.config.yield_strain,
                        self.config.failure_strain,
                        self.config.post_yield_exponent,
                        sub_dt,
                    ],
                    device=self.device,
                )
                for color in range(3):
                    wp.launch(
                        _project_bend_color,
                        dim=self.config.segment_count - 1,
                        inputs=[
                            self.positions,
                            self.inverse_masses,
                            self.bend_multipliers,
                            self.active_segments,
                            self.bend_lambdas,
                            color,
                            bend_compliance,
                            sub_dt,
                        ],
                        device=self.device,
                    )
                wp.launch(
                    _solve_contact_pairs,
                    dim=self.pair_count,
                    inputs=[
                        self.positions,
                        self.previous_positions,
                        self.inverse_masses,
                        self.pair_first,
                        self.pair_second,
                        self.pair_active,
                        self.pair_penetrating,
                        self.pair_distance,
                        self.delta_first_start,
                        self.delta_first_end,
                        self.delta_second_start,
                        self.delta_second_end,
                        self.config.diameter_m,
                        self.config.detection_distance_m,
                        self.config.self_friction,
                    ],
                    device=self.device,
                )
                wp.launch(
                    _gather_contact_corrections,
                    dim=self.config.particle_count,
                    inputs=[
                        self.positions,
                        self.particle_pair_offsets,
                        self.particle_pair_slots,
                        self.pair_penetrating,
                        self.delta_first_start,
                        self.delta_first_end,
                        self.delta_second_start,
                        self.delta_second_end,
                        0.85,
                    ],
                    device=self.device,
                )
                wp.launch(
                    _project_attachments,
                    dim=self.config.particle_count,
                    inputs=[
                        self.positions,
                        self.inverse_masses,
                        self.attachment_active,
                        self.attachment_targets,
                        self.attachment_compliance,
                        self.attachment_lambdas,
                        sub_dt,
                    ],
                    device=self.device,
                )
                if ground_height_m is not None:
                    wp.launch(
                        _project_ground,
                        dim=self.config.particle_count,
                        inputs=[
                            self.positions,
                            self.previous_positions,
                            self.inverse_masses,
                            float(ground_height_m),
                            self.config.diameter_m * 0.5,
                            self.config.surface_friction,
                        ],
                        device=self.device,
                    )
                # Contact, bending, and kinematic projections can reintroduce
                # stretch error.  Close every nonlinear iteration with the
                # coupled axial solve so load is transmitted over the entire
                # strand rather than left as a local end-segment artifact.
                wp.launch(
                    _project_stretch_global,
                    dim=1,
                    inputs=[
                        self.positions,
                        self.inverse_masses,
                        self.rest_lengths,
                        self.axial_multipliers,
                        self.failure_forces(),
                        self.active_segments,
                        self.stretch_lambdas,
                        self.stretch_directions,
                        self.stretch_diagonal,
                        self.stretch_off_diagonal,
                        self.stretch_right_hand_side,
                        self.stretch_c_prime,
                        self.stretch_d_prime,
                        self.stretch_delta_lambdas,
                        self.config.segment_count,
                        self.config.particle_mass_kg,
                        self.config.axial_rigidity_n,
                        self.config.yield_strain,
                        self.config.failure_strain,
                        self.config.post_yield_exponent,
                        sub_dt,
                    ],
                    device=self.device,
                )
            # Refresh contact observables after the final axial projection.
            # Corrections are intentionally not gathered here; this launch is
            # the exact post-solve contact state consumed by knot history and
            # qualification metrics.
            wp.launch(
                _solve_contact_pairs,
                dim=self.pair_count,
                inputs=[
                    self.positions,
                    self.previous_positions,
                    self.inverse_masses,
                    self.pair_first,
                    self.pair_second,
                    self.pair_active,
                    self.pair_penetrating,
                    self.pair_distance,
                    self.delta_first_start,
                    self.delta_first_end,
                    self.delta_second_start,
                    self.delta_second_end,
                    self.config.diameter_m,
                    self.config.detection_distance_m,
                    self.config.self_friction,
                ],
                device=self.device,
            )
            wp.launch(
                _update_velocities,
                dim=self.config.particle_count,
                inputs=[
                    self.positions,
                    self.previous_positions,
                    self.velocities,
                    maximum_speed,
                    sub_dt,
                ],
                device=self.device,
            )
            wp.launch(
                _update_contact_history,
                dim=self.pair_count,
                inputs=[
                    self.pair_first,
                    self.pair_second,
                    self.pair_active,
                    self.contact_dwell,
                    self.compacted_pairs,
                    self.config.knot_minimum_separation,
                    self.config.knot_compaction_dwell_s,
                    sub_dt,
                ],
                device=self.device,
            )
            wp.launch(
                _update_knot_strength,
                dim=self.config.segment_count,
                inputs=[
                    self.segment_pair_offsets,
                    self.segment_pair_indices,
                    self.compacted_pairs,
                    self.knot_strength,
                    self.config.knot_strength_efficiency,
                ],
                device=self.device,
            )
            wp.launch(
                _update_segment_contact_history,
                dim=self.config.segment_count,
                inputs=[
                    self.segment_pair_offsets,
                    self.segment_pair_indices,
                    self.pair_first,
                    self.pair_second,
                    self.pair_active,
                    self.segment_contact_dwell,
                    self.compacted_segments,
                    self.knot_strength,
                    self.config.knot_minimum_separation,
                    self.config.knot_compaction_dwell_s,
                    self.config.knot_strength_efficiency,
                    sub_dt,
                ],
                device=self.device,
            )
            wp.launch(
                _evaluate_breakage,
                dim=1,
                inputs=[
                    self.positions,
                    self.rest_lengths,
                    self.active_segments,
                    self.broken_segments,
                    self.strand_failed,
                    self.stretch_lambdas,
                    self.config.failure_force_n,
                    self.knot_strength,
                    self.damage_strength,
                    self.measured_force,
                    self.config.axial_rigidity_n,
                    self.config.yield_strain,
                    self.config.failure_strain,
                    self.config.post_yield_exponent,
                    sub_dt,
                ],
                device=self.device,
            )
            wp.launch(
                _evaluate_attachment_failure,
                dim=self.config.particle_count,
                inputs=[
                    self.attachment_active,
                    self.attachment_break_force,
                    self.attachment_failed,
                    self.attachment_failure_force,
                    self.inverse_masses,
                    self.measured_force,
                    self.config.particle_mass_kg,
                ],
                device=self.device,
            )

    def failure_forces(self):
        """Return a lazily materialized per-segment base failure array."""

        if not hasattr(self, "_failure_forces"):
            values = np.full(
                self.config.segment_count,
                self.config.failure_force_n,
                dtype=np.float32,
            )
            self._failure_forces = wp.array(
                values,
                dtype=float,
                device=self.device,
            )
        return self._failure_forces

    def state(self) -> dict[str, np.ndarray]:
        wp.synchronize_device(self.device)
        attachment_active = self.attachment_active.numpy()
        attachment_failed = self.attachment_failed.numpy()
        failed_indices = np.flatnonzero(attachment_failed)
        for particle in failed_indices.tolist():
            self._host_attachment_active[particle] = 0
            self._host_attachment_break_force[particle] = 0.0
            self._host_inverse_masses[particle] = (
                1.0 / self.config.particle_mass_kg
            )
        return {
            "positions_m": self.positions.numpy(),
            "velocities_m_s": self.velocities.numpy(),
            "active_segments": self.active_segments.numpy(),
            "broken_segments": self.broken_segments.numpy(),
            "strand_failed": self.strand_failed.numpy(),
            "segment_force_n": self.measured_force.numpy(),
            "knot_strength": self.knot_strength.numpy(),
            "pair_active": self.pair_active.numpy(),
            "pair_penetrating": self.pair_penetrating.numpy(),
            "pair_distance_m": self.pair_distance.numpy(),
            "compacted_pairs": self.compacted_pairs.numpy(),
            "compacted_segments": self.compacted_segments.numpy(),
            "attachment_active": attachment_active,
            "attachment_failed": attachment_failed,
            "attachment_failure_force_n": (
                self.attachment_failure_force.numpy()
            ),
        }

    def segment_transforms(self) -> dict[str, np.ndarray]:
        """Return centers and parallel-transport frames for USD segment visuals."""

        wp.synchronize_device(self.device)
        positions = self.positions.numpy()
        directions = positions[1:] - positions[:-1]
        lengths = np.linalg.norm(directions, axis=1)
        if np.any(lengths <= 1.0e-12) or not np.all(np.isfinite(lengths)):
            raise RuntimeError("cannot map a degenerate Warp strand to USD")
        tangents = directions / lengths[:, None]
        centers = (positions[1:] + positions[:-1]) * 0.5
        quaternions = np.empty(
            (self.config.segment_count, 4),
            dtype=np.float32,
        )
        candidate_axes = np.eye(3, dtype=np.float64)
        initial_axis = candidate_axes[
            int(np.argmin(np.abs(candidate_axes @ tangents[0])))
        ]
        normal = initial_axis - tangents[0] * np.dot(
            initial_axis,
            tangents[0],
        )
        normal /= np.linalg.norm(normal)
        for segment, tangent in enumerate(tangents):
            transported = normal - tangent * np.dot(normal, tangent)
            transported_length = float(np.linalg.norm(transported))
            if transported_length <= 1.0e-8:
                fallback = candidate_axes[
                    int(np.argmin(np.abs(candidate_axes @ tangent)))
                ]
                transported = fallback - tangent * np.dot(fallback, tangent)
                transported_length = float(np.linalg.norm(transported))
            normal = transported / transported_length
            binormal = np.cross(tangent, normal)
            binormal /= np.linalg.norm(binormal)
            normal = np.cross(binormal, tangent)
            frame = np.column_stack((tangent, normal, binormal))
            quaternions[segment] = _rotation_matrix_to_quaternion_wxyz(frame)
        return {
            "centers_m": centers.astype(np.float32),
            "orientations_wxyz": quaternions,
            "lengths_m": lengths.astype(np.float32),
        }


def straight_centerline(config: WarpSutureConfig) -> np.ndarray:
    positions = np.zeros((config.particle_count, 3), dtype=np.float32)
    positions[:, 0] = np.arange(config.particle_count) * config.spacing_m
    return positions


def _resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    differences = points[1:] - points[:-1]
    lengths = np.linalg.norm(differences, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    samples = np.linspace(0.0, cumulative[-1], count)
    result = np.empty((count, 3), dtype=np.float64)
    for axis in range(3):
        result[:, axis] = np.interp(samples, cumulative, points[:, axis])
    return result.astype(np.float32)


def overhand_knot_centerline(config: WarpSutureConfig) -> np.ndarray:
    """Open trefoil centerline with long tails, resampled at strand resolution."""

    parameter = np.linspace(0.16, 2.0 * math.pi - 0.16, 1400)
    raw = np.column_stack(
        (
            np.sin(parameter) + 2.0 * np.sin(2.0 * parameter),
            np.cos(parameter) - 2.0 * np.cos(2.0 * parameter),
            -np.sin(3.0 * parameter),
        )
    )
    raw_length = np.linalg.norm(raw[1:] - raw[:-1], axis=1).sum()
    knot_length = 0.032
    curve = raw * (knot_length / raw_length)
    curve -= curve.mean(axis=0)
    start_tangent = curve[1] - curve[0]
    start_tangent /= np.linalg.norm(start_tangent)
    end_tangent = curve[-1] - curve[-2]
    end_tangent /= np.linalg.norm(end_tangent)
    total_length = config.segment_count * config.spacing_m
    tail_length = max(0.0, (total_length - knot_length) * 0.5)
    start_tail = np.linspace(
        curve[0] - start_tangent * tail_length,
        curve[0],
        900,
        endpoint=False,
    )
    end_tail = np.linspace(
        curve[-1],
        curve[-1] + end_tangent * tail_length,
        900,
    )[1:]
    return _resample_polyline(
        np.vstack((start_tail, curve, end_tail)),
        config.particle_count,
    )


def _finite_state(state: dict[str, np.ndarray]) -> bool:
    return all(np.all(np.isfinite(value)) for value in state.values())


def _run_deterministic_replay(
    profile: dict[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    config = WarpSutureConfig.from_profile(profile)
    initial = overhand_knot_centerline(config)
    outputs: list[np.ndarray] = []
    for _ in range(2):
        solver = WarpSuture(profile, initial, device=device)
        solver.set_attachment(0, initial[0])
        solver.set_attachment(config.particle_count - 1, initial[-1])
        for _step in range(12):
            solver.step(
                dt_s=1.0 / 240.0,
                substeps=4,
                solver_iterations=8,
                gravity_m_s2=(0.0, 0.0, 0.0),
            )
        outputs.append(solver.state()["positions_m"])
    return {
        "passed": bool(np.array_equal(outputs[0], outputs[1])),
        "maximum_absolute_difference_m": float(
            np.max(np.abs(outputs[0] - outputs[1]))
        ),
    }


def _run_straight_stability(
    profile: dict[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    config = WarpSutureConfig.from_profile(profile)
    initial = np.zeros((config.particle_count, 3), dtype=np.float32)
    initial[:, 1] = (
        0.20 - np.arange(config.particle_count) * config.spacing_m
    )
    solver = WarpSuture(profile, initial, device=device)
    solver.set_attachment(0, initial[0])
    for _step in range(120):
        solver.step(
            dt_s=1.0 / 240.0,
            substeps=4,
            solver_iterations=8,
        )
    state = solver.state()
    transforms = solver.segment_transforms()
    lengths = np.linalg.norm(
        state["positions_m"][1:] - state["positions_m"][:-1],
        axis=1,
    )
    maximum_strain = float(
        np.max(np.abs(lengths / config.spacing_m - 1.0))
    )
    broken = int(np.count_nonzero(state["broken_segments"]))
    penetrating = int(np.count_nonzero(state["pair_penetrating"]))
    quaternion_norm_error = float(
        np.max(
            np.abs(
                np.linalg.norm(
                    transforms["orientations_wxyz"],
                    axis=1,
                )
                - 1.0
            )
        )
    )
    passed = (
        _finite_state(state)
        and broken == 0
        and penetrating == 0
        and maximum_strain < 0.001
        and transforms["centers_m"].shape
        == (config.segment_count, 3)
        and quaternion_norm_error < 1.0e-6
    )
    return {
        "passed": bool(passed),
        "maximum_segment_strain": maximum_strain,
        "broken_segment_count": broken,
        "penetrating_pair_count": penetrating,
        "configuration": "needle_attached_vertical_hanging_strand",
        "usd_segment_transform_count": int(
            transforms["centers_m"].shape[0]
        ),
        "maximum_quaternion_norm_error": quaternion_norm_error,
    }


def _run_knot_compaction(
    profile: dict[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    config = WarpSutureConfig.from_profile(profile)
    initial = overhand_knot_centerline(config)
    solver = WarpSuture(profile, initial, device=device)
    solver.set_attachment(0, initial[0])
    solver.set_attachment(config.particle_count - 1, initial[-1])
    first_direction = initial[0] - initial[1]
    first_direction /= np.linalg.norm(first_direction)
    last_direction = initial[-1] - initial[-2]
    last_direction /= np.linalg.norm(last_direction)
    start_time = time.perf_counter()
    steps = 120
    for step in range(steps):
        pull = 0.003 * (step + 1) / steps
        solver.set_attachment(0, initial[0] + first_direction * pull)
        solver.set_attachment(
            config.particle_count - 1,
            initial[-1] + last_direction * pull,
        )
        solver.step(
            dt_s=1.0 / 240.0,
            substeps=4,
            solver_iterations=10,
            gravity_m_s2=(0.0, 0.0, 0.0),
        )
    wp.synchronize_device(device)
    elapsed = time.perf_counter() - start_time
    state = solver.state()
    active_distances = state["pair_distance_m"][
        state["pair_active"].astype(bool)
    ]
    minimum_active_distance = (
        float(np.min(active_distances))
        if active_distances.size
        else math.inf
    )
    compacted = int(np.count_nonzero(state["compacted_pairs"]))
    compacted_segments = int(
        np.count_nonzero(state["compacted_segments"])
    )
    penetrating = int(np.count_nonzero(state["pair_penetrating"]))
    broken = int(np.count_nonzero(state["broken_segments"]))
    minimum_knot_strength = float(np.min(state["knot_strength"]))
    passed = (
        _finite_state(state)
        and compacted_segments > 0
        and broken == 0
        and minimum_knot_strength
        <= config.knot_strength_efficiency + 1.0e-6
        and minimum_active_distance >= config.diameter_m * 0.95
    )
    return {
        "passed": bool(passed),
        "compacted_pair_count": compacted,
        "compacted_segment_count": compacted_segments,
        "penetrating_pair_count": penetrating,
        "broken_segment_count": broken,
        "minimum_active_centerline_distance_m": minimum_active_distance,
        "minimum_knot_strength_fraction": minimum_knot_strength,
        "simulated_seconds": steps / 240.0,
        "wall_seconds": elapsed,
        "world_steps_per_second": steps / max(elapsed, 1.0e-9),
    }


def _run_instrument_transfer(
    profile: dict[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    config = WarpSutureConfig.from_profile(profile)
    initial = np.zeros((config.particle_count, 3), dtype=np.float32)
    initial[:, 1] = (
        0.20 - np.arange(config.particle_count) * config.spacing_m
    )
    solver = WarpSuture(profile, initial, device=device)
    solver.set_needle_connection(initial[0])
    grasp = (178, 179, 180)
    initial_center = initial[list(grasp)].mean(axis=0)
    first_target = initial_center + np.asarray(
        (0.0006, 0.0, 0.0),
        dtype=np.float32,
    )
    second_target = first_target + np.asarray(
        (0.0, 0.0, 0.0006),
        dtype=np.float32,
    )
    for step in range(24):
        amount = (step + 1) / 24.0
        target = initial_center + (first_target - initial_center) * amount
        solver.set_instrument_grasp(grasp, target)
        solver.step(
            dt_s=1.0 / 240.0,
            substeps=4,
            solver_iterations=8,
            gravity_m_s2=(0.0, 0.0, 0.0),
        )
    first_state = solver.state()
    first_center = first_state["positions_m"][list(grasp)].mean(axis=0)
    first_error = float(np.linalg.norm(first_center - first_target))

    for step in range(24):
        amount = (step + 1) / 24.0
        target = first_target + (second_target - first_target) * amount
        # Reassigning the same grasp nodes is the instrument-to-instrument
        # handoff boundary; strand-local offsets are retained.
        solver.set_instrument_grasp(grasp, target)
        solver.step(
            dt_s=1.0 / 240.0,
            substeps=4,
            solver_iterations=8,
            gravity_m_s2=(0.0, 0.0, 0.0),
        )
    second_state = solver.state()
    second_center = second_state["positions_m"][list(grasp)].mean(axis=0)
    second_error = float(np.linalg.norm(second_center - second_target))
    solver.release_instrument_grasp(grasp)
    for _step in range(12):
        solver.step(
            dt_s=1.0 / 240.0,
            substeps=4,
            solver_iterations=8,
            gravity_m_s2=(0.0, -9.81, 0.0),
        )
    final_state = solver.state()
    broken = int(np.count_nonzero(final_state["broken_segments"]))
    passed = (
        _finite_state(final_state)
        and broken == 0
        and first_error < 1.0e-7
        and second_error < 1.0e-7
        and solver._host_attachment_active[list(grasp)].sum() == 0
        and solver._host_attachment_active[0] == 1
    )
    return {
        "passed": bool(passed),
        "first_instrument_tracking_error_m": first_error,
        "second_instrument_tracking_error_m": second_error,
        "broken_segment_count": broken,
        "needle_attachment_retained": bool(
            solver._host_attachment_active[0] == 1
        ),
        "grasp_released": bool(
            solver._host_attachment_active[list(grasp)].sum() == 0
        ),
    }


def _run_overload_breakage(
    profile: dict[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    config = WarpSutureConfig.from_profile(profile)
    initial = straight_centerline(config)
    solver = WarpSuture(profile, initial, device=device)
    solver.set_attachment(0, initial[0])
    solver.set_attachment(config.particle_count - 1, initial[-1])
    peak_force = 0.0
    first_break_extension = None
    steps = 300
    for step in range(steps):
        strain = 0.26 * (step + 1) / steps
        target = initial[-1].copy()
        target[0] = config.segment_count * config.spacing_m * (1.0 + strain)
        solver.set_attachment(config.particle_count - 1, target)
        solver.step(
            dt_s=1.0 / 240.0,
            substeps=4,
            solver_iterations=10,
            gravity_m_s2=(0.0, 0.0, 0.0),
        )
        state = solver.state()
        peak_force = max(
            peak_force,
            float(np.max(state["segment_force_n"])),
        )
        if np.any(state["broken_segments"]):
            first_break_extension = strain
            break
    final_state = solver.state()
    broken = int(np.count_nonzero(final_state["broken_segments"]))
    passed = (
        _finite_state(final_state)
        and broken > 0
        and first_break_extension is not None
        and peak_force >= config.failure_force_n * 0.95
    )
    return {
        "passed": bool(passed),
        "broken_segment_count": broken,
        "first_break_global_extension_fraction": first_break_extension,
        "peak_measured_segment_force_n": peak_force,
        "target_failure_force_n": config.failure_force_n,
    }


def _run_needle_swage_pullout(
    profile: dict[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    config = WarpSutureConfig.from_profile(profile)
    initial = straight_centerline(config)
    solver = WarpSuture(profile, initial, device=device)
    solver.set_needle_connection(initial[0])
    solver.set_attachment(config.particle_count - 1, initial[-1])
    pullout_force = 0.0
    pullout_extension = None
    steps = 260
    for step in range(steps):
        strain = 0.20 * (step + 1) / steps
        target = initial[-1].copy()
        target[0] = config.segment_count * config.spacing_m * (1.0 + strain)
        solver.set_attachment(config.particle_count - 1, target)
        solver.step(
            dt_s=1.0 / 240.0,
            substeps=4,
            solver_iterations=10,
            gravity_m_s2=(0.0, 0.0, 0.0),
        )
        state = solver.state()
        if state["attachment_failed"][0] != 0:
            pullout_force = float(state["attachment_failure_force_n"][0])
            pullout_extension = strain
            break
    final_state = solver.state()
    suture_broken = int(np.count_nonzero(final_state["broken_segments"]))
    passed = (
        _finite_state(final_state)
        and final_state["attachment_failed"][0] != 0
        and final_state["attachment_active"][0] == 0
        and suture_broken == 0
        and final_state["strand_failed"][0] == 0
        and pullout_force >= config.swage_pullout_force_n
        and pullout_force < config.failure_force_n
    )
    return {
        "passed": bool(passed),
        "pullout_force_n": pullout_force,
        "target_pullout_force_n": config.swage_pullout_force_n,
        "pullout_global_extension_fraction": pullout_extension,
        "suture_broken_segment_count": suture_broken,
        "needle_attachment_released": bool(
            final_state["attachment_active"][0] == 0
        ),
    }


def qualify(
    profile: dict[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    wp.init()
    started = time.perf_counter()
    checks = {
        "deterministic_replay": _run_deterministic_replay(
            profile,
            device=device,
        ),
        "straight_strand_stability": _run_straight_stability(
            profile,
            device=device,
        ),
        "overhand_knot_compaction": _run_knot_compaction(
            profile,
            device=device,
        ),
        "instrument_grasp_transfer": _run_instrument_transfer(
            profile,
            device=device,
        ),
        "needle_swage_pullout": _run_needle_swage_pullout(
            profile,
            device=device,
        ),
        "excessive_load_breakage": _run_overload_breakage(
            profile,
            device=device,
        ),
    }
    config = WarpSutureConfig.from_profile(profile)
    passed = all(check["passed"] for check in checks.values())
    return {
        "schema": REPORT_SCHEMA,
        "backend": BACKEND_ID,
        "passed": passed,
        "device": device,
        "warp_version": wp.__version__,
        "profile_id": profile["id"],
        "profile_version": profile["version"],
        "clinical_validation": False,
        "pair_policy": {
            "kind": "exhaustive_nonlocal_segment_capsules",
            "segment_count": config.segment_count,
            "excluded_index_distance": 2,
            "pair_count": int(
                len(_contact_pairs(config.segment_count)[0])
            ),
            "correction_gather": "fixed_order_csr_without_float_atomics",
        },
        "solver_policy": {
            "method": "nonlinear_axial_xpbd_plus_discrete_bending",
            "internal_substeps": 4,
            "maximum_free_displacement_per_substep_diameters": 0.45,
            "self_contact": "exact_segment_segment_closest_points",
            "friction": "coulomb_tangential_position_limit",
            "failure": "xpbd_constraint_force_or_failure_strain",
        },
        "checks": checks,
        "wall_seconds": time.perf_counter() - started,
        "note": (
            "GPU engineering qualification only; physical bench, convergence, "
            "sim-to-real, and clinical validation remain required."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    profile = load_profile(args.profile)
    report = qualify(profile, device=args.device)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
