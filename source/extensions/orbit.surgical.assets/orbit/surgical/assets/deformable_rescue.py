# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Contact-driven patient effects for the Autonomous Rescue OR.

The policy-facing API can request an intervention, but it cannot write bleeding,
closure, perfusion, or success values.  Those values are derived here from
monotonic post-physics observations submitted by the scene adapter.

This is an engineering simulation model.  The calibration values are explicit
research seeds and are not physiological or clinical reference values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import random
from types import MappingProxyType
from typing import Final, Mapping


MMHG_TO_PA: Final = 133.322
BLOOD_DENSITY_KG_M3: Final = 1060.0
M3_TO_ML: Final = 1_000_000.0
SUPPORTED_TARGETS: Final = frozenset(
    {
        "rescue_vessel",
        "bowel_anastomosis",
        "abdominal_wall",
        "occlusive_film",
    }
)


def _finite(value: float, name: str) -> float:
    rendered = float(value)
    if not math.isfinite(rendered):
        raise ValueError(f"{name} must be finite")
    return rendered


def _nonnegative(value: float, name: str) -> float:
    rendered = _finite(value, name)
    if rendered < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return rendered


def _fraction(value: float, name: str) -> float:
    rendered = _finite(value, name)
    if not 0.0 <= rendered <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return rendered


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _approach(current: float, target: float, dt_s: float, time_s: float) -> float:
    if time_s <= 0.0:
        return target
    blend = 1.0 - math.exp(-dt_s / time_s)
    return current + (target - current) * blend


@dataclass(frozen=True)
class PhysicsEvidenceFrame:
    """Raw post-physics measurements accepted from the scene adapter.

    The frame contains no success, control, seal, patency, or perfusion result.
    Separation and attachment counts must be measured from the live scene.
    """

    physics_step: int
    simulation_time_s: float
    dt_s: float
    station_id: str
    tool_id: str
    target_id: str
    left_normal_force_n: float
    right_normal_force_n: float
    separation_m: float
    tool_speed_m_s: float
    retained_attachment_count: int = 0
    patch_contact_point_count: int = 0
    leaked_particle_count: int = 0
    particle_volume_ml: float = 0.002

    def __post_init__(self) -> None:
        if self.physics_step < 0:
            raise ValueError("physics_step must be nonnegative")
        for name in ("station_id", "tool_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        if self.target_id not in SUPPORTED_TARGETS:
            raise ValueError(f"unsupported rescue target {self.target_id!r}")
        for name in (
            "simulation_time_s",
            "dt_s",
            "left_normal_force_n",
            "right_normal_force_n",
            "separation_m",
            "tool_speed_m_s",
            "particle_volume_ml",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be greater than zero")
        for name in (
            "retained_attachment_count",
            "patch_contact_point_count",
            "leaked_particle_count",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True)
class RescueEffectCalibration:
    """Versioned research seeds used by the contact-to-effect bridge."""

    contact_minimum_force_n: float = 0.12
    vessel_target_force_per_pad_n: float = 1.8
    vessel_soft_force_per_pad_n: float = 4.0
    vessel_hard_force_per_pad_n: float = 7.0
    vessel_maximum_asymmetry_n: float = 1.5
    vessel_closed_separation_m: float = 0.0044
    maximum_stable_tool_speed_m_s: float = 0.025
    contact_attack_time_s: float = 0.18
    contact_release_time_s: float = 0.10
    clip_maturation_time_s: float = 0.30
    patch_maturation_time_s: float = 0.75
    overload_damage_fraction_per_s: float = 0.035
    nominal_upstream_pressure_mmhg: float = 92.0
    downstream_pressure_mmhg: float = 8.0
    pressure_challenge_multiplier: float = 1.25
    maximum_verified_residual_flow_ml_s: float = 0.08
    minimum_verified_distal_perfusion_fraction: float = 0.52
    verification_window_s: float = 0.75
    minimum_verification_frames: int = 20
    baseline_blood_volume_ml: float = 5000.0
    particle_leak_blend: float = 0.25
    parameter_status: str = "provisional_engineering_seeds"

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name == "parameter_status":
                continue
            if name == "minimum_verification_frames":
                if value <= 0:
                    raise ValueError(f"{name} must be positive")
                continue
            _nonnegative(value, name)


@dataclass
class VesselRescueState:
    vessel_radius_m: float = 0.0022
    injury_fraction: float = 1.0
    discharge_coefficient: float = 0.68
    collateral_flow_fraction: float = 0.58
    transient_compression_fraction: float = 0.0
    retained_clip_fraction: float = 0.0
    patch_seal_fraction: float = 0.0
    overload_damage_fraction: float = 0.0
    residual_flow_ml_s: float = 0.0
    distal_perfusion_fraction: float = 1.0
    cumulative_blood_loss_ml: float = 0.0
    blood_volume_ml: float = 5000.0
    pressure_challenge_active: bool = False
    challenge_elapsed_s: float = 0.0
    challenge_frame_count: int = 0
    hemostasis_verified: bool = False
    last_physics_step: int = -1
    last_simulation_time_s: float = -1.0
    clip_contact_dwell_s: float = 0.0
    patch_contact_dwell_s: float = 0.0
    last_retained_attachment_count: int = 0


@dataclass
class RepairState:
    target_id: str
    initial_separation_m: float
    target_separation_m: float
    required_attachment_count: int
    approximation_fraction: float = 0.0
    retention_fraction: float = 0.0
    leak_rate_ml_s: float = 0.0
    overload_damage_fraction: float = 0.0
    last_physics_step: int = -1
    last_simulation_time_s: float = -1.0


@dataclass(frozen=True)
class RescueEffectsSnapshot:
    physics_step: int
    simulation_time_s: float
    vessel: Mapping[str, float | int | bool]
    repairs: Mapping[str, Mapping[str, float | int | bool]]
    evidence_frames: int
    rejected_frames: int


class _SceneAuthority:
    __slots__ = ()


class SceneEvidenceAdapter:
    """Environment-owned ingress for raw Isaac/PhysX observations."""

    __slots__ = ("_effects", "_authority")

    def __init__(
        self,
        effects: "ContactDrivenRescueEffects",
        authority: _SceneAuthority,
    ) -> None:
        self._effects = effects
        self._authority = authority

    def publish(self, frame: PhysicsEvidenceFrame) -> RescueEffectsSnapshot:
        return self._effects._ingest(frame, self._authority)


class ContactDrivenRescueEffects:
    """Own patient effects and reject action- or caller-authored outcomes."""

    def __init__(
        self,
        *,
        seed: int = 0,
        calibration: RescueEffectCalibration | None = None,
    ) -> None:
        self.calibration = calibration or RescueEffectCalibration()
        self._random = random.Random(seed)
        self._authority = _SceneAuthority()
        self._evidence_frames = 0
        self._rejected_frames = 0
        self._physics_step = -1
        self._simulation_time_s = 0.0
        self.vessel = self._new_vessel()
        self.repairs: dict[str, RepairState] = {
            "bowel_anastomosis": RepairState(
                "bowel_anastomosis", 0.012, 0.0012, 12
            ),
            "abdominal_wall": RepairState(
                "abdominal_wall", 0.030, 0.0020, 14
            ),
            "occlusive_film": RepairState(
                "occlusive_film", 0.006, 0.0006, 32
            ),
        }

    def _new_vessel(self) -> VesselRescueState:
        radius = self._random.uniform(0.0018, 0.0027)
        injury = self._random.uniform(0.72, 1.0)
        collateral = self._random.uniform(0.48, 0.68)
        return VesselRescueState(
            vessel_radius_m=radius,
            injury_fraction=injury,
            collateral_flow_fraction=collateral,
            blood_volume_ml=self.calibration.baseline_blood_volume_ml,
        )

    def create_scene_adapter(self) -> SceneEvidenceAdapter:
        """Create the adapter that must remain on the environment side."""

        return SceneEvidenceAdapter(self, self._authority)

    def reset(self, *, seed: int | None = None) -> RescueEffectsSnapshot:
        if seed is not None:
            self._random.seed(seed)
        self._evidence_frames = 0
        self._rejected_frames = 0
        self._physics_step = -1
        self._simulation_time_s = 0.0
        self.vessel = self._new_vessel()
        for state in self.repairs.values():
            state.approximation_fraction = 0.0
            state.retention_fraction = 0.0
            state.leak_rate_ml_s = 0.0
            state.overload_damage_fraction = 0.0
            state.last_physics_step = -1
            state.last_simulation_time_s = -1.0
        return self.snapshot()

    def start_pressure_challenge(self) -> None:
        self.vessel.pressure_challenge_active = True
        self.vessel.challenge_elapsed_s = 0.0
        self.vessel.challenge_frame_count = 0
        self.vessel.hemostasis_verified = False

    def stop_pressure_challenge(self) -> None:
        self.vessel.pressure_challenge_active = False
        self.vessel.challenge_elapsed_s = 0.0
        self.vessel.challenge_frame_count = 0
        self.vessel.hemostasis_verified = False

    def _ingest(
        self,
        frame: PhysicsEvidenceFrame,
        authority: _SceneAuthority,
    ) -> RescueEffectsSnapshot:
        if authority is not self._authority:
            self._rejected_frames += 1
            raise PermissionError("patient effects accept environment scene evidence only")
        if frame.physics_step <= self._physics_step:
            self._rejected_frames += 1
            raise ValueError("physics evidence must use a strictly increasing step")
        if frame.simulation_time_s <= self._simulation_time_s and self._physics_step >= 0:
            self._rejected_frames += 1
            raise ValueError("physics evidence must use increasing simulation time")

        self._physics_step = frame.physics_step
        self._simulation_time_s = frame.simulation_time_s
        self._evidence_frames += 1
        if frame.target_id == "rescue_vessel":
            self._update_vessel(frame)
        else:
            self._update_repair(frame)
        return self.snapshot()

    def _bilateral_contact_quality(
        self,
        frame: PhysicsEvidenceFrame,
        target_force_n: float,
        maximum_asymmetry_n: float,
    ) -> float:
        minimum_force = min(
            frame.left_normal_force_n,
            frame.right_normal_force_n,
        )
        if minimum_force < self.calibration.contact_minimum_force_n:
            return 0.0
        symmetry = 1.0 - _clamp(
            abs(frame.left_normal_force_n - frame.right_normal_force_n)
            / maximum_asymmetry_n
        )
        force = _clamp(minimum_force / target_force_n)
        speed = 1.0 - _clamp(
            frame.tool_speed_m_s
            / self.calibration.maximum_stable_tool_speed_m_s
        )
        return force * symmetry * speed

    def _update_vessel(self, frame: PhysicsEvidenceFrame) -> None:
        cfg = self.calibration
        state = self.vessel
        quality = self._bilateral_contact_quality(
            frame,
            cfg.vessel_target_force_per_pad_n,
            cfg.vessel_maximum_asymmetry_n,
        )
        gap_quality = 1.0 - _clamp(
            frame.separation_m / cfg.vessel_closed_separation_m
        )
        compression_target = quality * gap_quality
        release_time = (
            cfg.contact_attack_time_s
            if compression_target > state.transient_compression_fraction
            else cfg.contact_release_time_s
        )
        state.transient_compression_fraction = _clamp(
            _approach(
                state.transient_compression_fraction,
                compression_target,
                frame.dt_s,
                release_time,
            )
        )

        peak_force = max(frame.left_normal_force_n, frame.right_normal_force_n)
        if peak_force > cfg.vessel_soft_force_per_pad_n:
            overload = _clamp(
                (peak_force - cfg.vessel_soft_force_per_pad_n)
                / (
                    cfg.vessel_hard_force_per_pad_n
                    - cfg.vessel_soft_force_per_pad_n
                )
            )
            state.overload_damage_fraction = _clamp(
                state.overload_damage_fraction
                + overload * cfg.overload_damage_fraction_per_s * frame.dt_s
            )

        retained_increased = (
            frame.retained_attachment_count
            > state.last_retained_attachment_count
        )
        if frame.retained_attachment_count > 0 and quality > 0.6:
            state.clip_contact_dwell_s += frame.dt_s
        elif not retained_increased:
            state.clip_contact_dwell_s = max(
                0.0, state.clip_contact_dwell_s - frame.dt_s
            )
        clip_target = _clamp(
            frame.retained_attachment_count
            * state.clip_contact_dwell_s
            / cfg.clip_maturation_time_s
        )
        state.retained_clip_fraction = max(
            state.retained_clip_fraction,
            clip_target,
        )
        state.last_retained_attachment_count = frame.retained_attachment_count

        if frame.patch_contact_point_count >= 3 and quality > 0.45:
            state.patch_contact_dwell_s += frame.dt_s
        else:
            state.patch_contact_dwell_s = max(
                0.0, state.patch_contact_dwell_s - 2.0 * frame.dt_s
            )
        patch_coverage = _clamp(frame.patch_contact_point_count / 12.0)
        patch_target = patch_coverage * _clamp(
            state.patch_contact_dwell_s / cfg.patch_maturation_time_s
        )
        state.patch_seal_fraction = max(state.patch_seal_fraction, patch_target)

        control = 1.0
        for fraction in (
            state.transient_compression_fraction,
            state.retained_clip_fraction,
            state.patch_seal_fraction,
        ):
            control *= 1.0 - fraction
        effective_control = 1.0 - control
        pressure_mmhg = cfg.nominal_upstream_pressure_mmhg
        if state.pressure_challenge_active:
            pressure_mmhg *= cfg.pressure_challenge_multiplier
        pressure_pa = max(
            0.0,
            (pressure_mmhg - cfg.downstream_pressure_mmhg) * MMHG_TO_PA,
        )
        defect_area_m2 = (
            math.pi
            * state.vessel_radius_m**2
            * state.injury_fraction
            * (1.0 - effective_control) ** 2
        )
        model_flow_ml_s = (
            state.discharge_coefficient
            * defect_area_m2
            * math.sqrt(2.0 * pressure_pa / BLOOD_DENSITY_KG_M3)
            * M3_TO_ML
        )
        particle_flow_ml_s = (
            frame.leaked_particle_count
            * frame.particle_volume_ml
            / frame.dt_s
        )
        blend = cfg.particle_leak_blend if frame.leaked_particle_count else 0.0
        state.residual_flow_ml_s = (
            (1.0 - blend) * model_flow_ml_s + blend * particle_flow_ml_s
        )
        state.cumulative_blood_loss_ml += state.residual_flow_ml_s * frame.dt_s
        state.blood_volume_ml = max(
            0.0,
            state.blood_volume_ml - state.residual_flow_ml_s * frame.dt_s,
        )

        axial_occlusion = max(
            state.retained_clip_fraction,
            state.transient_compression_fraction * 0.75,
        )
        state.distal_perfusion_fraction = _clamp(
            state.collateral_flow_fraction
            + (1.0 - state.collateral_flow_fraction)
            * (1.0 - axial_occlusion)
            - 0.35 * state.overload_damage_fraction
        )
        if state.pressure_challenge_active:
            state.challenge_elapsed_s += frame.dt_s
            state.challenge_frame_count += 1
            stable = (
                state.challenge_elapsed_s >= cfg.verification_window_s
                and state.challenge_frame_count >= cfg.minimum_verification_frames
                and state.residual_flow_ml_s
                <= cfg.maximum_verified_residual_flow_ml_s
                and state.distal_perfusion_fraction
                >= cfg.minimum_verified_distal_perfusion_fraction
            )
            state.hemostasis_verified = stable
        else:
            state.hemostasis_verified = False
        state.last_physics_step = frame.physics_step
        state.last_simulation_time_s = frame.simulation_time_s

    def _update_repair(self, frame: PhysicsEvidenceFrame) -> None:
        state = self.repairs[frame.target_id]
        quality = self._bilateral_contact_quality(frame, 1.5, 1.2)
        geometric = 1.0 - _clamp(
            (frame.separation_m - state.target_separation_m)
            / max(
                state.initial_separation_m - state.target_separation_m,
                1.0e-9,
            )
        )
        state.approximation_fraction = quality * geometric
        state.retention_fraction = _clamp(
            frame.retained_attachment_count
            / max(state.required_attachment_count, 1)
        )
        unclosed = 1.0 - min(
            state.approximation_fraction,
            state.retention_fraction,
        )
        particle_leak = (
            frame.leaked_particle_count
            * frame.particle_volume_ml
            / frame.dt_s
        )
        state.leak_rate_ml_s = max(0.0, 2.5 * unclosed + particle_leak)
        peak_force = max(frame.left_normal_force_n, frame.right_normal_force_n)
        if peak_force > 6.0:
            state.overload_damage_fraction = _clamp(
                state.overload_damage_fraction
                + (peak_force - 6.0) * 0.005 * frame.dt_s
            )
        state.last_physics_step = frame.physics_step
        state.last_simulation_time_s = frame.simulation_time_s

    def policy_observation(self) -> Mapping[str, object]:
        """Return immutable outcomes; no mutation handles or authority objects."""

        snapshot = self.snapshot()
        return MappingProxyType(
            {
                "physics_step": snapshot.physics_step,
                "simulation_time_s": snapshot.simulation_time_s,
                "vessel": snapshot.vessel,
                "repairs": snapshot.repairs,
            }
        )

    def snapshot(self) -> RescueEffectsSnapshot:
        vessel = MappingProxyType(
            {
                "residual_flow_ml_s": self.vessel.residual_flow_ml_s,
                "distal_perfusion_fraction": self.vessel.distal_perfusion_fraction,
                "cumulative_blood_loss_ml": self.vessel.cumulative_blood_loss_ml,
                "blood_volume_ml": self.vessel.blood_volume_ml,
                "transient_compression_fraction": (
                    self.vessel.transient_compression_fraction
                ),
                "retained_clip_fraction": self.vessel.retained_clip_fraction,
                "patch_seal_fraction": self.vessel.patch_seal_fraction,
                "overload_damage_fraction": self.vessel.overload_damage_fraction,
                "pressure_challenge_active": (
                    self.vessel.pressure_challenge_active
                ),
                "hemostasis_verified": self.vessel.hemostasis_verified,
                "last_physics_step": self.vessel.last_physics_step,
            }
        )
        repairs = MappingProxyType(
            {
                name: MappingProxyType(
                    {
                        "approximation_fraction": state.approximation_fraction,
                        "retention_fraction": state.retention_fraction,
                        "leak_rate_ml_s": state.leak_rate_ml_s,
                        "overload_damage_fraction": (
                            state.overload_damage_fraction
                        ),
                        "last_physics_step": state.last_physics_step,
                    }
                )
                for name, state in self.repairs.items()
            }
        )
        return RescueEffectsSnapshot(
            physics_step=self._physics_step,
            simulation_time_s=self._simulation_time_s,
            vessel=vessel,
            repairs=repairs,
            evidence_frames=self._evidence_frames,
            rejected_frames=self._rejected_frames,
        )


__all__ = [
    "ContactDrivenRescueEffects",
    "PhysicsEvidenceFrame",
    "RepairState",
    "RescueEffectCalibration",
    "RescueEffectsSnapshot",
    "SceneEvidenceAdapter",
    "VesselRescueState",
]
