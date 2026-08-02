#!/usr/bin/env python3
"""NVIDIA Warp parity for the mixed-mode cohesive fracture oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from dr_anmar_cohesive_fracture import CohesiveState, MixedModeCohesiveLaw
from dr_anmar_cuttable_tissue_solver import DEFAULT_PROFILE_PATH, load_profile


@wp.kernel
def _mixed_mode_cohesive_response(
    jump: wp.array(dtype=wp.vec3),
    normal: wp.array(dtype=wp.vec3),
    relative_velocity: wp.array(dtype=wp.vec3),
    previous_maximum: wp.array(dtype=float),
    previous_damage: wp.array(dtype=float),
    seeded: wp.array(dtype=wp.int32),
    traction: wp.array(dtype=wp.vec3),
    damage: wp.array(dtype=float),
    maximum_separation: wp.array(dtype=float),
    initiation_separation: wp.array(dtype=float),
    final_separation: wp.array(dtype=float),
    fracture_energy: wp.array(dtype=float),
    mode_ii_fraction: wp.array(dtype=float),
    penalty: float,
    compression_penalty: float,
    normal_peak: float,
    shear_peak: float,
    mode_i_energy: float,
    mode_ii_energy: float,
    bk_exponent: float,
    viscosity: float,
):
    sample = wp.tid()
    unit_normal = wp.normalize(normal[sample])
    separation = jump[sample]
    signed_normal = wp.dot(separation, unit_normal)
    opening = wp.max(signed_normal, 0.0)
    shear_vector = separation - signed_normal * unit_normal
    shear = wp.length(shear_vector)
    effective = wp.sqrt(opening * opening + shear * shear)

    peak = normal_peak
    mode_fraction = float(0.0)
    energy = mode_i_energy
    if effective > 1.0e-15:
        normal_fraction = opening / effective
        shear_fraction = shear / effective
        peak = 1.0 / wp.sqrt(
            (normal_fraction / normal_peak) * (normal_fraction / normal_peak)
            + (shear_fraction / shear_peak) * (shear_fraction / shear_peak)
        )
        mode_fraction = shear_fraction * shear_fraction
        energy = mode_i_energy + (mode_ii_energy - mode_i_energy) * wp.pow(
            mode_fraction, bk_exponent
        )
    initiation = peak / penalty
    final = 2.0 * energy / peak
    maximum = wp.max(previous_maximum[sample], effective)
    current_damage = previous_damage[sample]
    if seeded[sample] > 0 and maximum > initiation:
        target = final * (maximum - initiation) / (maximum * (final - initiation))
        current_damage = wp.max(current_damage, wp.clamp(target, 0.0, 1.0))

    result = wp.vec3(0.0, 0.0, 0.0)
    if signed_normal < 0.0:
        result = result - compression_penalty * signed_normal * unit_normal
    if effective > 1.0e-15 and current_damage < 1.0 - 1.0e-12:
        direction = (opening * unit_normal + shear_vector) / effective
        cohesive = (1.0 - current_damage) * penalty * effective
        effective_rate = wp.dot(relative_velocity[sample], direction)
        viscous = (1.0 - current_damage) * viscosity * wp.max(effective_rate, 0.0)
        result = result - (cohesive + viscous) * direction

    traction[sample] = result
    damage[sample] = current_damage
    maximum_separation[sample] = maximum
    initiation_separation[sample] = initiation
    final_separation[sample] = final
    fracture_energy[sample] = energy
    mode_ii_fraction[sample] = mode_fraction


@dataclass(frozen=True)
class CohesiveWarpParityReceipt:
    schema: str
    profile_id: str
    profile_sha256: str
    warp_version: str
    device: str
    device_is_cuda: bool
    sample_count: int
    maximum_traction_relative_l2_error: float
    maximum_damage_absolute_error: float
    maximum_history_absolute_error_m: float
    maximum_envelope_separation_absolute_error_m: float
    maximum_fracture_energy_absolute_error_j_m2: float
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


def run_cohesive_warp_parity(
    profile: dict[str, Any] | None = None,
    *,
    profile_path: Path = DEFAULT_PROFILE_PATH,
    device: str = "cpu",
) -> CohesiveWarpParityReceipt:
    profile = profile or load_profile(profile_path)
    _device_name(device)
    law = MixedModeCohesiveLaw(profile)
    normal = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    normal_initiation, normal_final, _, _ = law.envelope(1.0, 0.0)
    _, shear_final, _, _ = law.envelope(0.0, 1.0)
    _, mixed_final, _, _ = law.envelope(np.sqrt(0.5), np.sqrt(0.5))
    samples = (
        (np.asarray((0.0, 0.0, 0.5 * normal_initiation)), np.zeros(3), 0.0, 0.0, 1),
        (np.asarray((0.0, 0.0, 0.7 * normal_final)), np.zeros(3), 0.0, 0.0, 1),
        (np.asarray((0.7 * shear_final, 0.0, 0.0)), np.zeros(3), 0.0, 0.0, 1),
        (
            0.7 * mixed_final * np.asarray((np.sqrt(0.5), 0.0, np.sqrt(0.5))),
            np.zeros(3),
            0.0,
            0.0,
            1,
        ),
        (np.asarray((0.0, 0.0, 1.1 * normal_final)), np.zeros(3), 0.0, 0.0, 0),
        (
            np.asarray((0.0, 0.0, 0.5 * normal_initiation)),
            np.asarray((0.0, 0.0, 0.01)),
            0.0,
            0.0,
            1,
        ),
        (np.asarray((0.0, 0.0, -1.0e-5)), np.zeros(3), 1.1 * normal_final, 1.0, 1),
    )
    jumps = np.asarray([sample[0] for sample in samples], dtype=np.float32)
    velocities = np.asarray([sample[1] for sample in samples], dtype=np.float32)
    previous_maximum = np.asarray([sample[2] for sample in samples], dtype=np.float32)
    previous_damage = np.asarray([sample[3] for sample in samples], dtype=np.float32)
    seeded = np.asarray([sample[4] for sample in samples], dtype=np.int32)
    normals = np.repeat(normal[None, :], len(samples), axis=0).astype(np.float32)

    arrays = {
        "traction": wp.empty(len(samples), dtype=wp.vec3, device=device),
        "damage": wp.empty(len(samples), dtype=float, device=device),
        "maximum": wp.empty(len(samples), dtype=float, device=device),
        "initiation": wp.empty(len(samples), dtype=float, device=device),
        "final": wp.empty(len(samples), dtype=float, device=device),
        "energy": wp.empty(len(samples), dtype=float, device=device),
        "mode": wp.empty(len(samples), dtype=float, device=device),
    }
    wp.launch(
        _mixed_mode_cohesive_response,
        dim=len(samples),
        inputs=[
            wp.array(jumps, dtype=wp.vec3, device=device),
            wp.array(normals, dtype=wp.vec3, device=device),
            wp.array(velocities, dtype=wp.vec3, device=device),
            wp.array(previous_maximum, dtype=float, device=device),
            wp.array(previous_damage, dtype=float, device=device),
            wp.array(seeded, dtype=wp.int32, device=device),
            arrays["traction"],
            arrays["damage"],
            arrays["maximum"],
            arrays["initiation"],
            arrays["final"],
            arrays["energy"],
            arrays["mode"],
            law.penalty,
            law.compression_penalty,
            law.normal_peak,
            law.shear_peak,
            law.mode_i_energy,
            law.mode_ii_energy,
            law.bk_exponent,
            law.viscosity,
        ],
        device=device,
    )
    wp.synchronize_device(device)

    reference = {key: [] for key in arrays}
    for jump, velocity, maximum, damage, is_seeded in samples:
        state = CohesiveState(
            maximum_effective_separation_m=maximum,
            damage=damage,
            seeded=bool(is_seeded),
            failed=damage >= 1.0 - 1.0e-12,
        )
        response = law.evaluate(
            jump,
            normal,
            velocity,
            1.0e-4,
            state,
            seeded=bool(is_seeded),
        )
        reference["traction"].append(response.traction_on_positive_face_pa)
        reference["damage"].append(response.damage)
        reference["maximum"].append(state.maximum_effective_separation_m)
        reference["initiation"].append(response.initiation_separation_m)
        reference["final"].append(response.final_separation_m)
        reference["energy"].append(response.mixed_mode_fracture_energy_j_m2)
        reference["mode"].append(response.mode_ii_fraction)

    warp_values = {key: value.numpy().astype(np.float64) for key, value in arrays.items()}
    reference_values = {
        key: np.asarray(value, dtype=np.float64) for key, value in reference.items()
    }
    traction_error = float(
        np.linalg.norm(warp_values["traction"] - reference_values["traction"])
        / max(float(np.linalg.norm(reference_values["traction"])), 1.0e-15)
    )
    damage_error = float(np.max(np.abs(warp_values["damage"] - reference_values["damage"])))
    history_error = float(np.max(np.abs(warp_values["maximum"] - reference_values["maximum"])))
    envelope_error = max(
        float(np.max(np.abs(warp_values["initiation"] - reference_values["initiation"]))),
        float(np.max(np.abs(warp_values["final"] - reference_values["final"]))),
    )
    energy_error = float(np.max(np.abs(warp_values["energy"] - reference_values["energy"])))
    limits = profile["warp_parity"]
    gates = {
        "traction": traction_error <= float(limits["maximum_cohesive_traction_relative_l2_error"]),
        "damage": damage_error <= float(limits["maximum_cohesive_damage_absolute_error"]),
        "history": history_error <= float(limits["maximum_cohesive_separation_absolute_error_m"]),
        "envelope": envelope_error <= float(limits["maximum_cohesive_separation_absolute_error_m"]),
    }
    failed = tuple(name for name, passed in gates.items() if not passed)
    profile_sha = hashlib.sha256(
        json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    device_name = _device_name(device)
    is_cuda = "cuda" in device_name.lower()
    return CohesiveWarpParityReceipt(
        schema="dr.anmar.cohesive-fracture-warp-parity-receipt.v1",
        profile_id=str(profile["id"]),
        profile_sha256=profile_sha,
        warp_version=str(wp.__version__),
        device=device_name,
        device_is_cuda=is_cuda,
        sample_count=len(samples),
        maximum_traction_relative_l2_error=traction_error,
        maximum_damage_absolute_error=damage_error,
        maximum_history_absolute_error_m=history_error,
        maximum_envelope_separation_absolute_error_m=envelope_error,
        maximum_fracture_energy_absolute_error_j_m2=energy_error,
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
    receipt = run_cohesive_warp_parity(
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
