# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Pure state and validation helpers for webcam surgical teleoperation.

The browser owns hand landmark processing.  This module receives only bounded,
calibrated pose offsets and resamples their remaining displacement at the
simulator rate.  It deliberately has no Isaac Sim dependency so the safety
contract can be tested on an ordinary development machine.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable


HAND_FRAME_TIMEOUT_S = 0.250
MAX_TRANSLATION_OFFSET_M = (0.12, 0.12, 0.12)
MAX_ROTATION_VECTOR_RAD = (0.80, 0.80, 0.80)


def _bounded_vector(
    value: Iterable[float],
    *,
    dimensions: int,
    limits: tuple[float, ...],
    label: str,
) -> list[float]:
    values = [float(component) for component in value]
    if len(values) != dimensions:
        raise ValueError(f"{label} must contain {dimensions} values")
    if not all(math.isfinite(component) for component in values):
        raise ValueError(f"{label} must contain only finite values")
    if any(abs(component) > limit for component, limit in zip(values, limits)):
        raise ValueError(f"{label} exceeds its safe workspace bound")
    return values


def validate_hand_frame(
    sequence: int,
    hands: Iterable[dict[str, Any]],
    *,
    arms: int,
) -> tuple[int, list[dict[str, Any]]]:
    """Validate a complete API frame without mutating runtime state."""

    if isinstance(sequence, bool) or int(sequence) != sequence or sequence < 0:
        raise ValueError("sequence must be a non-negative integer")
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw in hands:
        arm = raw.get("arm")
        if isinstance(arm, bool) or not isinstance(arm, int) or arm not in range(arms):
            raise ValueError(f"arm must be between 0 and {arms - 1}")
        if arm in seen:
            raise ValueError("each available instrument may appear once")
        tracked = raw.get("tracked")
        motion_engaged = raw.get("motion_engaged")
        if not isinstance(tracked, bool) or not isinstance(motion_engaged, bool):
            raise ValueError("tracked and motion_engaged must be booleans")
        translation = _bounded_vector(
            raw.get("translation_offset_m", ()),
            dimensions=3,
            limits=MAX_TRANSLATION_OFFSET_M,
            label="translation_offset_m",
        )
        rotation = _bounded_vector(
            raw.get("rotation_vector_rad", ()),
            dimensions=3,
            limits=MAX_ROTATION_VECTOR_RAD,
            label="rotation_vector_rad",
        )
        aperture = float(raw.get("aperture_normalized"))
        confidence = float(raw.get("confidence"))
        if not math.isfinite(aperture) or not 0.0 <= aperture <= 1.0:
            raise ValueError("aperture_normalized must be finite and between 0 and 1")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and between 0 and 1")
        normalized.append(
            {
                "arm": arm,
                "tracked": tracked,
                "motion_engaged": motion_engaged,
                "translation_offset_m": translation,
                "rotation_vector_rad": rotation,
                "aperture_normalized": aperture,
                "confidence": confidence,
            }
        )
        seen.add(arm)
    if not normalized:
        raise ValueError("hands must contain at least one instrument state")
    return int(sequence), normalized


def proportional_gripper_action(aperture_normalized: float) -> float:
    """Map closed=0/open=1 aperture onto the compatible -1/+1 action slot."""

    aperture = float(aperture_normalized)
    if not math.isfinite(aperture) or not 0.0 <= aperture <= 1.0:
        raise ValueError("aperture_normalized must be finite and between 0 and 1")
    return 2.0 * aperture - 1.0


def proportional_jaw_targets(
    aperture_normalized: float,
    *,
    close_rad: float,
    open_rad: float,
) -> tuple[float, float]:
    """Return symmetric physical jaw targets for a normalized aperture."""

    aperture = float(aperture_normalized)
    if not math.isfinite(aperture) or not 0.0 <= aperture <= 1.0:
        raise ValueError("aperture_normalized must be finite and between 0 and 1")
    jaw = float(close_rad) + aperture * (float(open_rad) - float(close_rad))
    return -jaw, jaw


@dataclass
class HandArmState:
    tracked: bool = False
    motion_engaged: bool = False
    reacquire_unclutched: bool = True
    target_offset: list[float] = field(default_factory=lambda: [0.0] * 6)
    consumed_offset: list[float] = field(default_factory=lambda: [0.0] * 6)
    aperture_normalized: float = 1.0
    confidence: float = 0.0
    last_frame_at: float = 0.0
    stale: bool = True

    def discard_motion(self, *, require_unclutched: bool) -> None:
        self.motion_engaged = False
        self.target_offset = [0.0] * 6
        self.consumed_offset = [0.0] * 6
        self.reacquire_unclutched = require_unclutched


class HandTeleopRuntime:
    """Latest-master-pose runtime with simulator-rate residual consumption."""

    def __init__(self, arms: int, timeout_s: float = HAND_FRAME_TIMEOUT_S) -> None:
        self.arms = int(arms)
        self.timeout_s = float(timeout_s)
        self.last_sequence = -1
        self.enabled = False
        self.arm_states = [HandArmState() for _ in range(self.arms)]

    def disable_motion(self, *, require_unclutched: bool = True) -> None:
        self.enabled = False
        for arm_state in self.arm_states:
            arm_state.discard_motion(require_unclutched=require_unclutched)

    def enable_motion(self) -> None:
        self.enabled = True
        for arm_state in self.arm_states:
            arm_state.discard_motion(require_unclutched=True)

    def submit(
        self,
        sequence: int,
        hands: Iterable[dict[str, Any]],
        *,
        now: float | None = None,
    ) -> None:
        sequence, normalized = validate_hand_frame(sequence, hands, arms=self.arms)
        if sequence <= self.last_sequence:
            raise ValueError("sequence must increase monotonically")
        timestamp = time.monotonic() if now is None else float(now)
        seen: set[int] = set()
        for hand in normalized:
            arm = hand["arm"]
            seen.add(arm)
            arm_state = self.arm_states[arm]
            arm_state.last_frame_at = timestamp
            arm_state.confidence = hand["confidence"]
            arm_state.stale = False
            if hand["tracked"]:
                arm_state.tracked = True
                arm_state.aperture_normalized = hand["aperture_normalized"]
                if arm_state.reacquire_unclutched:
                    arm_state.discard_motion(require_unclutched=hand["motion_engaged"])
                    if not hand["motion_engaged"]:
                        arm_state.reacquire_unclutched = False
                elif hand["motion_engaged"]:
                    arm_state.motion_engaged = True
                    arm_state.target_offset = (
                        list(hand["translation_offset_m"])
                        + list(hand["rotation_vector_rad"])
                    )
                else:
                    arm_state.discard_motion(require_unclutched=False)
            else:
                arm_state.tracked = False
                arm_state.stale = True
                arm_state.discard_motion(require_unclutched=True)

        # A valid frame is a complete observation.  An omitted arm is treated
        # exactly like tracking loss, but its last jaw aperture is held.
        for arm, arm_state in enumerate(self.arm_states):
            if arm not in seen:
                arm_state.tracked = False
                arm_state.stale = True
                arm_state.discard_motion(require_unclutched=True)
        self.last_sequence = sequence

    def consume(
        self,
        axis_scales: Iterable[Iterable[float]],
        *,
        now: float | None = None,
    ) -> list[list[float]]:
        """Consume residual offsets using the active native IK scale per axis."""

        timestamp = time.monotonic() if now is None else float(now)
        scales = [[abs(float(value)) for value in arm] for arm in axis_scales]
        if len(scales) != self.arms or any(len(arm) != 6 for arm in scales):
            raise ValueError("axis_scales must contain six values per instrument")
        if any(
            not math.isfinite(value) or value <= 0.0
            for arm in scales
            for value in arm
        ):
            raise ValueError("axis_scales must be finite positive values")

        self.expire_stale(now=timestamp)
        commands = [[0.0] * 6 for _ in range(self.arms)]
        for arm, arm_state in enumerate(self.arm_states):
            if not self.enabled or not arm_state.tracked or not arm_state.motion_engaged:
                continue
            for axis, scale in enumerate(scales[arm]):
                residual = arm_state.target_offset[axis] - arm_state.consumed_offset[axis]
                command = max(-1.0, min(1.0, residual / scale))
                commands[arm][axis] = command
                arm_state.consumed_offset[axis] += command * scale
        return commands

    def expire_stale(self, *, now: float | None = None) -> bool:
        """Discard stale motion independently of the simulator step rate."""

        timestamp = time.monotonic() if now is None else float(now)
        expired = False
        for arm_state in self.arm_states:
            if (
                arm_state.last_frame_at > 0.0
                and timestamp - arm_state.last_frame_at > self.timeout_s
            ):
                changed = (
                    arm_state.tracked
                    or arm_state.motion_engaged
                    or not arm_state.stale
                    or any(arm_state.target_offset)
                    or any(arm_state.consumed_offset)
                    or not arm_state.reacquire_unclutched
                )
                arm_state.tracked = False
                arm_state.stale = True
                arm_state.discard_motion(require_unclutched=True)
                expired = expired or changed
        return expired

    def snapshot(self, *, now: float | None = None) -> dict[str, Any]:
        timestamp = time.monotonic() if now is None else float(now)
        arms = []
        for arm, arm_state in enumerate(self.arm_states):
            age_ms = (
                round(max(0.0, timestamp - arm_state.last_frame_at) * 1000)
                if arm_state.last_frame_at > 0.0
                else None
            )
            arms.append(
                {
                    "arm": arm,
                    "tracked": arm_state.tracked,
                    "motion_engaged": arm_state.motion_engaged,
                    "reacquire_unclutched": arm_state.reacquire_unclutched,
                    "target_offset": list(arm_state.target_offset),
                    "consumed_offset": list(arm_state.consumed_offset),
                    "aperture_normalized": arm_state.aperture_normalized,
                    "confidence": arm_state.confidence,
                    "tracking_age_ms": age_ms,
                    "stale": arm_state.stale
                    or age_ms is None
                    or age_ms > round(self.timeout_s * 1000),
                }
            )
        return {
            "enabled": self.enabled,
            "sequence": self.last_sequence,
            "watchdog_ms": round(self.timeout_s * 1000),
            "arms": arms,
        }
