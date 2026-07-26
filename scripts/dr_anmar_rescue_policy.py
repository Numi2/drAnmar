# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Executable rescue-policy and patient-effect rollout contracts."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from dr_anmar_rescue_dataset import (
    OBSERVATION_KEYS,
    rescue_policy_observation_shapes,
)


POLICY_RUNTIME_VERSION = "dr-anmar-rescue-policy-v1"


def infer_rescue_phase_code(
    telemetry: Mapping[str, Any],
    *,
    nearest_target_distance_m: float | None,
    effective_hold_observed: bool,
    success: bool,
) -> int:
    """Infer a causal, patient-evidence phase for the next policy action."""

    contact = telemetry.get("measured_contact", {})
    vessel = telemetry.get("vessel", {})
    authority = bool(telemetry.get("sensor_authority_available", False))
    bilateral_contact = (
        authority
        and float(contact.get("left_normal_force_n", 0.0)) > 0.0
        and float(contact.get("right_normal_force_n", 0.0)) > 0.0
    )
    compression = float(
        vessel.get("transient_compression_fraction", 0.0)
    )
    if success:
        return 7
    if effective_hold_observed:
        return 6
    if compression >= 0.02:
        return 5
    if bilateral_contact:
        return 4
    if (
        nearest_target_distance_m is not None
        and nearest_target_distance_m <= 0.04
    ):
        return 1
    return 0


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RescuePolicyRuntime:
    """Strict adapter around a Robomimic rollout policy.

    The adapter owns action validation and clipping. It deliberately has no
    reference to the patient runtime: policies can command robot actions only,
    while patient effects remain derived from post-physics evidence.
    """

    def __init__(
        self,
        policy: Any,
        *,
        checkpoint_path: Path,
        checkpoint_digest: str,
        action_dim: int,
        device: str,
    ) -> None:
        self.policy = policy
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_sha256 = checkpoint_digest
        self.action_dim = int(action_dim)
        self.device = device
        self.steps = 0
        self.last_inference_ms: float | None = None
        self.last_action_norm: float | None = None
        self.clipped_action_values = 0

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        *,
        action_dim: int,
        try_cuda: bool = True,
    ) -> "RescuePolicyRuntime":
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"rescue policy checkpoint not found: {path}")
        try:
            import robomimic.utils.file_utils as FileUtils
            import robomimic.utils.torch_utils as TorchUtils
        except ImportError as exc:
            raise RuntimeError(
                "Robomimic is required to execute a rescue checkpoint"
            ) from exc

        device = TorchUtils.get_torch_device(try_to_use_cuda=try_cuda)
        policy, checkpoint = FileUtils.policy_from_checkpoint(
            ckpt_path=str(path),
            device=device,
            verbose=True,
        )
        cls.validate_checkpoint_contract(checkpoint, action_dim=action_dim)
        runtime = cls(
            policy,
            checkpoint_path=path,
            checkpoint_digest=checkpoint_sha256(path),
            action_dim=action_dim,
            device=str(device),
        )
        runtime.reset()
        return runtime

    @staticmethod
    def validate_checkpoint_contract(
        checkpoint: Mapping[str, Any],
        *,
        action_dim: int,
    ) -> None:
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Robomimic returned invalid checkpoint metadata")
        if action_dim % 7:
            raise ValueError(
                "rescue checkpoint action dimension must be divisible by seven"
            )
        shape_metadata = checkpoint.get("shape_metadata")
        if not isinstance(shape_metadata, Mapping):
            raise ValueError(
                "rescue checkpoint is missing Robomimic shape_metadata"
            )
        checkpoint_action_dim = int(shape_metadata.get("ac_dim", -1))
        if checkpoint_action_dim != action_dim:
            raise ValueError(
                "rescue checkpoint action dimension mismatch: "
                f"{checkpoint_action_dim} versus {action_dim}"
            )
        all_shapes = shape_metadata.get("all_shapes")
        if not isinstance(all_shapes, Mapping):
            raise ValueError(
                "rescue checkpoint is missing observation shapes"
            )
        expected = rescue_policy_observation_shapes(action_dim // 7)
        checkpoint_keys = tuple(
            shape_metadata.get("all_obs_keys", all_shapes.keys())
        )
        if checkpoint_keys != OBSERVATION_KEYS:
            raise ValueError(
                "rescue checkpoint observation keys do not match the live "
                f"contract: {checkpoint_keys}"
            )
        for name, shape in expected.items():
            if name not in all_shapes:
                raise ValueError(
                    f"rescue checkpoint is missing observation {name}"
                )
            checkpoint_shape = tuple(int(value) for value in all_shapes[name])
            if checkpoint_shape != shape:
                raise ValueError(
                    f"rescue checkpoint observation {name} has shape "
                    f"{checkpoint_shape}; expected {shape}"
                )

    def reset(self) -> None:
        start_episode = getattr(self.policy, "start_episode", None)
        if callable(start_episode):
            start_episode()
        self.steps = 0
        self.last_inference_ms = None
        self.last_action_norm = None
        self.clipped_action_values = 0

    def act(
        self,
        observation: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        keys = tuple(observation)
        if keys != OBSERVATION_KEYS:
            missing = sorted(set(OBSERVATION_KEYS) - set(keys))
            extra = sorted(set(keys) - set(OBSERVATION_KEYS))
            raise ValueError(
                "rescue policy observation contract mismatch: "
                f"missing={missing}, extra={extra}, order={keys}"
            )
        prepared: dict[str, np.ndarray] = {}
        for name in OBSERVATION_KEYS:
            values = np.asarray(observation[name], dtype=np.float32)
            if values.ndim != 1 or not values.size:
                raise ValueError(
                    f"rescue policy observation {name} must be a non-empty vector"
                )
            if not np.isfinite(values).all():
                raise ValueError(
                    f"rescue policy observation {name} is non-finite"
                )
            prepared[name] = np.ascontiguousarray(values)

        started = time.perf_counter()
        raw_action = self.policy(prepared)
        inference_ms = (time.perf_counter() - started) * 1000.0
        action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if action.shape != (self.action_dim,):
            raise ValueError(
                "rescue policy action shape mismatch: "
                f"{action.shape} versus ({self.action_dim},)"
            )
        if not np.isfinite(action).all():
            raise ValueError("rescue policy produced NaN or infinity")
        clipped = np.clip(action, -1.0, 1.0)
        self.clipped_action_values += int(np.count_nonzero(clipped != action))
        self.steps += 1
        self.last_inference_ms = inference_ms
        self.last_action_norm = float(np.linalg.norm(clipped))
        return clipped.astype(np.float32, copy=False)

    def snapshot(self) -> dict[str, Any]:
        return {
            "runtime_version": POLICY_RUNTIME_VERSION,
            "checkpoint": self.checkpoint_path.name,
            "checkpoint_sha256": self.checkpoint_sha256,
            "device": self.device,
            "steps": self.steps,
            "last_inference_ms": self.last_inference_ms,
            "last_action_norm": self.last_action_norm,
            "clipped_action_values": self.clipped_action_values,
        }


@dataclass
class RescueOutcomeMonitor:
    """Detect a genuine temporary-compression rescue from patient effects."""

    required_effective_steps: int = 20
    maximum_steps: int = 1500
    minimum_compression_fraction: float = 0.10
    release_compression_fraction: float = 0.02
    maximum_overload_fraction: float = 0.02
    minimum_distal_perfusion_fraction: float = 0.25
    minimum_mean_arterial_pressure_mmhg: float = 50.0
    minimum_spo2_fraction: float = 0.85
    minimum_global_perfusion_fraction: float = 0.50
    steps: int = 0
    consecutive_effective_steps: int = 0
    effective_hold_observed: bool = False
    release_observed: bool = False
    peak_compression_fraction: float = 0.0
    peak_overload_fraction: float = 0.0
    minimum_perfusion_fraction: float = 1.0
    minimum_observed_map_mmhg: float = float("inf")
    minimum_observed_spo2_fraction: float = 1.0
    minimum_observed_global_perfusion_fraction: float = 1.0
    sensor_authority_steps: int = 0
    success: bool = False
    timed_out: bool = False

    def reset(self) -> None:
        defaults = type(self)(
            required_effective_steps=self.required_effective_steps,
            maximum_steps=self.maximum_steps,
            minimum_compression_fraction=(
                self.minimum_compression_fraction
            ),
            release_compression_fraction=(
                self.release_compression_fraction
            ),
            maximum_overload_fraction=self.maximum_overload_fraction,
            minimum_distal_perfusion_fraction=(
                self.minimum_distal_perfusion_fraction
            ),
            minimum_mean_arterial_pressure_mmhg=(
                self.minimum_mean_arterial_pressure_mmhg
            ),
            minimum_spo2_fraction=self.minimum_spo2_fraction,
            minimum_global_perfusion_fraction=(
                self.minimum_global_perfusion_fraction
            ),
        )
        self.__dict__.update(defaults.__dict__)

    def update(self, telemetry: Mapping[str, Any]) -> dict[str, Any]:
        self.steps += 1
        contact = telemetry.get("measured_contact", {})
        vessel = telemetry.get("vessel", {})
        vital_signs = telemetry.get("vital_signs", {})
        authority = bool(telemetry.get("sensor_authority_available", False))
        left_force = float(contact.get("left_normal_force_n", 0.0))
        right_force = float(contact.get("right_normal_force_n", 0.0))
        compression = float(
            vessel.get("transient_compression_fraction", 0.0)
        )
        overload = float(vessel.get("overload_damage_fraction", 0.0))
        perfusion = float(vessel.get("distal_perfusion_fraction", 1.0))
        map_mmhg = float(
            vital_signs.get("mean_arterial_pressure_mmhg", 100.0)
        )
        spo2 = float(vital_signs.get("spo2_fraction", 1.0))
        global_perfusion = float(
            vital_signs.get("global_perfusion_fraction", 1.0)
        )
        values = np.asarray(
            (
                left_force,
                right_force,
                compression,
                overload,
                perfusion,
                map_mmhg,
                spo2,
                global_perfusion,
            ),
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError("rescue outcome telemetry is non-finite")

        self.sensor_authority_steps += int(authority)
        self.peak_compression_fraction = max(
            self.peak_compression_fraction,
            compression,
        )
        self.peak_overload_fraction = max(
            self.peak_overload_fraction,
            overload,
        )
        self.minimum_perfusion_fraction = min(
            self.minimum_perfusion_fraction,
            perfusion,
        )
        self.minimum_observed_map_mmhg = min(
            self.minimum_observed_map_mmhg,
            map_mmhg,
        )
        self.minimum_observed_spo2_fraction = min(
            self.minimum_observed_spo2_fraction,
            spo2,
        )
        self.minimum_observed_global_perfusion_fraction = min(
            self.minimum_observed_global_perfusion_fraction,
            global_perfusion,
        )
        effective = (
            authority
            and left_force > 0.0
            and right_force > 0.0
            and compression >= self.minimum_compression_fraction
            and overload <= self.maximum_overload_fraction
            and perfusion >= self.minimum_distal_perfusion_fraction
        )
        self.consecutive_effective_steps = (
            self.consecutive_effective_steps + 1 if effective else 0
        )
        self.effective_hold_observed = (
            self.effective_hold_observed
            or self.consecutive_effective_steps
            >= self.required_effective_steps
        )
        released_now = bool(telemetry.get("release_observed", False)) and (
            compression < self.release_compression_fraction
        )
        self.release_observed = (
            self.release_observed
            or (self.effective_hold_observed and released_now)
        )
        self.success = (
            self.effective_hold_observed
            and self.release_observed
            and self.peak_overload_fraction
            <= self.maximum_overload_fraction
            and self.minimum_perfusion_fraction
            >= self.minimum_distal_perfusion_fraction
            and self.minimum_observed_map_mmhg
            >= self.minimum_mean_arterial_pressure_mmhg
            and self.minimum_observed_spo2_fraction
            >= self.minimum_spo2_fraction
            and self.minimum_observed_global_perfusion_fraction
            >= self.minimum_global_perfusion_fraction
        )
        self.timed_out = self.steps >= self.maximum_steps and not self.success
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        result = asdict(self)
        if not np.isfinite(self.minimum_observed_map_mmhg):
            result["minimum_observed_map_mmhg"] = None
        result["terminal"] = self.success or self.timed_out
        result["status"] = (
            "completed"
            if self.success
            else "timed_out"
            if self.timed_out
            else "running"
        )
        return result
