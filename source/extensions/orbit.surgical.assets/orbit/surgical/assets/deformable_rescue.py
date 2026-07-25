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
    Separation, target distance, and attachment counts must be measured from
    the live scene.
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
    target_distance_m: float
    retained_attachment_count: int = 0
    patch_contact_point_count: int = 0
    leaked_particle_count: int = 0
    particle_volume_ml: float = 0.002
    measured_cavity_pressure_kpa: float = 0.0
    measured_upstream_pressure_mmhg: float | None = None

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
            "target_distance_m",
            "particle_volume_ml",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        object.__setattr__(
            self,
            "measured_cavity_pressure_kpa",
            _finite(
                self.measured_cavity_pressure_kpa,
                "measured_cavity_pressure_kpa",
            ),
        )
        if self.measured_upstream_pressure_mmhg is not None:
            object.__setattr__(
                self,
                "measured_upstream_pressure_mmhg",
                _nonnegative(
                    self.measured_upstream_pressure_mmhg,
                    "measured_upstream_pressure_mmhg",
                ),
            )
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
    full_target_radius_m: float = 0.006
    maximum_target_radius_m: float = 0.025
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
    film_required_contact_points: int = 24
    film_target_pressure_kpa: float = 6.0
    film_pressure_tolerance_kpa: float = 3.0
    film_minimum_seal_quality: float = 0.80
    film_maximum_verified_leak_ml_s: float = 0.05
    film_verification_window_s: float = 1.0
    film_minimum_verification_frames: int = 20
    baseline_blood_volume_ml: float = 5000.0
    particle_leak_blend: float = 0.25
    parameter_status: str = "provisional_engineering_seeds"

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name == "parameter_status":
                continue
            if name in {
                "minimum_verification_frames",
                "film_required_contact_points",
                "film_minimum_verification_frames",
            }:
                if value <= 0:
                    raise ValueError(f"{name} must be positive")
                continue
            _nonnegative(value, name)
        if self.maximum_target_radius_m <= self.full_target_radius_m:
            raise ValueError(
                "maximum_target_radius_m must exceed full_target_radius_m"
            )


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
    measured_upstream_pressure_mmhg: float | None = None
    challenge_elapsed_s: float = 0.0
    challenge_frame_count: int = 0
    hemostasis_verified: bool = False
    last_physics_step: int = -1
    last_simulation_time_s: float = -1.0
    clip_contact_dwell_s: float = 0.0
    patch_contact_dwell_s: float = 0.0
    clip_maturity_fraction: float = 0.0
    patch_maturity_fraction: float = 0.0
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
    contact_coverage_fraction: float = 0.0
    measured_pressure_kpa: float = 0.0
    seal_quality: float = 0.0
    seal_verified: bool = False
    verification_elapsed_s: float = 0.0
    verification_frame_count: int = 0
    last_physics_step: int = -1
    last_simulation_time_s: float = -1.0


@dataclass(frozen=True)
class RescueEffectsSnapshot:
    physics_step: int
    simulation_time_s: float
    vessel: Mapping[str, float | int | bool | None]
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

    def finalize_interval(
        self,
        observed_target_ids: frozenset[str],
    ) -> RescueEffectsSnapshot:
        return self._effects._finalize_interval(
            observed_target_ids,
            self._authority,
        )


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
                "occlusive_film", 0.006, 0.0006, 8
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
            state.contact_coverage_fraction = 0.0
            state.measured_pressure_kpa = 0.0
            state.seal_quality = 0.0
            state.seal_verified = False
            state.verification_elapsed_s = 0.0
            state.verification_frame_count = 0
            state.last_physics_step = -1
            state.last_simulation_time_s = -1.0
        return self.snapshot()

    def _ingest(
        self,
        frame: PhysicsEvidenceFrame,
        authority: _SceneAuthority,
    ) -> RescueEffectsSnapshot:
        if authority is not self._authority:
            self._rejected_frames += 1
            raise PermissionError("patient effects accept environment scene evidence only")
        target_state = (
            self.vessel
            if frame.target_id == "rescue_vessel"
            else self.repairs[frame.target_id]
        )
        if frame.physics_step <= target_state.last_physics_step:
            self._rejected_frames += 1
            raise ValueError(
                "physics evidence must use a strictly increasing step per target"
            )
        if (
            frame.simulation_time_s <= target_state.last_simulation_time_s
            and target_state.last_physics_step >= 0
        ):
            self._rejected_frames += 1
            raise ValueError(
                "physics evidence must use increasing simulation time per target"
            )

        self._physics_step = max(self._physics_step, frame.physics_step)
        self._simulation_time_s = max(
            self._simulation_time_s,
            frame.simulation_time_s,
        )
        self._evidence_frames += 1
        if frame.target_id == "rescue_vessel":
            self._update_vessel(frame)
        elif frame.target_id == "occlusive_film":
            self._update_film(frame)
        else:
            self._update_repair(frame)
        return self.snapshot()

    def _finalize_interval(
        self,
        observed_target_ids: frozenset[str],
        authority: _SceneAuthority,
    ) -> RescueEffectsSnapshot:
        """Fail closed when an already-active repair loses scene evidence."""

        if authority is not self._authority:
            self._rejected_frames += 1
            raise PermissionError(
                "only the environment may finalize scene evidence"
            )
        if "rescue_vessel" not in observed_target_ids:
            raise ValueError(
                "every rescue interval requires fresh rescue_vessel evidence"
            )
        for target_id, state in self.repairs.items():
            if state.last_physics_step < 0 or target_id in observed_target_ids:
                continue
            state.approximation_fraction = 0.0
            state.retention_fraction = 0.0
            state.leak_rate_ml_s = 2.5
            state.contact_coverage_fraction = 0.0
            state.measured_pressure_kpa = 0.0
            state.seal_quality = 0.0
            state.seal_verified = False
            state.verification_elapsed_s = 0.0
            state.verification_frame_count = 0
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
        full_radius = self.calibration.full_target_radius_m
        maximum_radius = self.calibration.maximum_target_radius_m
        spatial = 1.0 - _clamp(
            (frame.target_distance_m - full_radius)
            / (maximum_radius - full_radius)
        )
        return force * symmetry * speed * spatial

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

        if frame.retained_attachment_count > 0 and quality > 0.6:
            state.clip_contact_dwell_s += frame.dt_s
            state.clip_maturity_fraction = max(
                state.clip_maturity_fraction,
                _clamp(
                    state.clip_contact_dwell_s / cfg.clip_maturation_time_s
                ),
            )
        elif frame.retained_attachment_count <= 0:
            state.clip_contact_dwell_s = 0.0
            state.clip_maturity_fraction = 0.0
        state.retained_clip_fraction = (
            state.clip_maturity_fraction
            if frame.retained_attachment_count > 0
            else 0.0
        )
        state.last_retained_attachment_count = frame.retained_attachment_count

        if frame.patch_contact_point_count >= 3 and quality > 0.45:
            state.patch_contact_dwell_s += frame.dt_s
            state.patch_maturity_fraction = max(
                state.patch_maturity_fraction,
                _clamp(
                    state.patch_contact_dwell_s / cfg.patch_maturation_time_s
                ),
            )
        patch_coverage = _clamp(frame.patch_contact_point_count / 12.0)
        state.patch_seal_fraction = min(
            state.patch_maturity_fraction,
            patch_coverage,
        )

        control = 1.0
        for fraction in (
            state.transient_compression_fraction,
            state.retained_clip_fraction,
            state.patch_seal_fraction,
        ):
            control *= 1.0 - fraction
        effective_control = 1.0 - control
        state.measured_upstream_pressure_mmhg = (
            frame.measured_upstream_pressure_mmhg
        )
        pressure_mmhg = (
            frame.measured_upstream_pressure_mmhg
            if frame.measured_upstream_pressure_mmhg is not None
            else cfg.nominal_upstream_pressure_mmhg
        )
        state.pressure_challenge_active = bool(
            frame.measured_upstream_pressure_mmhg is not None
            and frame.measured_upstream_pressure_mmhg
            >= (
                cfg.nominal_upstream_pressure_mmhg
                * cfg.pressure_challenge_multiplier
            )
        )
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
            state.challenge_elapsed_s = 0.0
            state.challenge_frame_count = 0
            state.hemostasis_verified = False
        state.last_physics_step = frame.physics_step
        state.last_simulation_time_s = frame.simulation_time_s

    def _update_repair(self, frame: PhysicsEvidenceFrame) -> None:
        state = self.repairs[frame.target_id]
        geometric = 1.0 - _clamp(
            (frame.separation_m - state.target_separation_m)
            / max(
                state.initial_separation_m - state.target_separation_m,
                1.0e-9,
            )
        )
        state.approximation_fraction = geometric
        state.retention_fraction = _clamp(
            frame.retained_attachment_count
            / max(state.required_attachment_count, 1)
        )
        retained_closure = (
            min(state.approximation_fraction, state.retention_fraction)
            * (1.0 - state.overload_damage_fraction)
        )
        unclosed = 1.0 - retained_closure
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

    def _update_film(self, frame: PhysicsEvidenceFrame) -> None:
        """Derive film integrity only from live bonds, contact, and pressure."""

        cfg = self.calibration
        state = self.repairs["occlusive_film"]
        state.approximation_fraction = 1.0 - _clamp(
            (frame.separation_m - state.target_separation_m)
            / max(
                state.initial_separation_m - state.target_separation_m,
                1.0e-9,
            )
        )
        state.retention_fraction = _clamp(
            frame.retained_attachment_count
            / max(state.required_attachment_count, 1)
        )
        state.contact_coverage_fraction = _clamp(
            frame.patch_contact_point_count
            / cfg.film_required_contact_points
        )
        state.measured_pressure_kpa = frame.measured_cavity_pressure_kpa
        pressure_quality = 1.0 - _clamp(
            abs(
                frame.measured_cavity_pressure_kpa
                + cfg.film_target_pressure_kpa
            )
            / cfg.film_pressure_tolerance_kpa
        )
        particle_leak = (
            frame.leaked_particle_count
            * frame.particle_volume_ml
            / frame.dt_s
        )
        retained_seal = min(
            state.approximation_fraction,
            state.retention_fraction,
            state.contact_coverage_fraction,
        )
        state.leak_rate_ml_s = max(
            0.0,
            2.5 * (1.0 - retained_seal) + particle_leak,
        )
        state.seal_quality = min(retained_seal, pressure_quality)
        stable = (
            state.seal_quality >= cfg.film_minimum_seal_quality
            and state.leak_rate_ml_s <= cfg.film_maximum_verified_leak_ml_s
        )
        if stable:
            state.verification_elapsed_s += frame.dt_s
            state.verification_frame_count += 1
        else:
            state.verification_elapsed_s = 0.0
            state.verification_frame_count = 0
        state.seal_verified = (
            state.verification_elapsed_s >= cfg.film_verification_window_s
            and state.verification_frame_count
            >= cfg.film_minimum_verification_frames
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
                "measured_upstream_pressure_mmhg": (
                    self.vessel.measured_upstream_pressure_mmhg
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
                        "contact_coverage_fraction": (
                            state.contact_coverage_fraction
                        ),
                        "measured_pressure_kpa": state.measured_pressure_kpa,
                        "seal_quality": state.seal_quality,
                        "seal_verified": state.seal_verified,
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
