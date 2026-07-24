"""Research controller utilities for trigger-to-pusher target synchronization.

OpenUSD authors revolute-joint limits in degrees, while Isaac Lab articulation
tensor targets are radians. The public target helpers therefore return radians
for the trigger and metres for the pusher.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

TRIGGER_LIMIT_DEG = 28.0
TRIGGER_LIMIT_RAD = math.radians(TRIGGER_LIMIT_DEG)
PUSHER_TRAVEL_M = 0.009
FIRE_THRESHOLD_DEG = 24.0
FIRE_THRESHOLD_RAD = math.radians(FIRE_THRESHOLD_DEG)
REARM_THRESHOLD_DEG = 8.0
REARM_THRESHOLD_RAD = math.radians(REARM_THRESHOLD_DEG)
DEPLOYMENT_THRESHOLD_M = PUSHER_TRAVEL_M * FIRE_THRESHOLD_DEG / TRIGGER_LIMIT_DEG


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))


def trigger_fraction(trigger_position_rad: float) -> float:
    """Normalize an Isaac joint position in radians to ``[0, 1]``."""

    return clamp(trigger_position_rad / TRIGGER_LIMIT_RAD, 0.0, 1.0)


def pusher_target_from_trigger(trigger_position_rad: float) -> float:
    """Map trigger rotation in radians to provisional pusher travel in metres."""

    return trigger_fraction(trigger_position_rad) * PUSHER_TRAVEL_M


def trigger_radians_from_degrees(trigger_position_deg: float) -> float:
    """Convert and clamp a human-facing degree command to an Isaac radian target."""

    return math.radians(clamp(trigger_position_deg, 0.0, TRIGGER_LIMIT_DEG))


def synchronized_joint_targets(trigger_position_rad: float) -> dict[str, float]:
    """Return Isaac Lab position targets: trigger in radians, pusher in metres."""

    trigger = clamp(trigger_position_rad, 0.0, TRIGGER_LIMIT_RAD)
    return {
        "trigger_joint": trigger,
        "pusher_joint": pusher_target_from_trigger(trigger),
    }


def synchronized_joint_targets_deg(trigger_position_deg: float) -> dict[str, float]:
    """Human-facing degree wrapper around :func:`synchronized_joint_targets`."""

    return synchronized_joint_targets(trigger_radians_from_degrees(trigger_position_deg))


@dataclass(frozen=True)
class FireCycle:
    """Piecewise target profile for a single research firing cycle."""

    press_steps: int = 80
    hold_steps: int = 80
    release_steps: int = 80

    def __post_init__(self) -> None:
        if self.press_steps <= 0 or self.hold_steps < 0 or self.release_steps <= 0:
            raise ValueError("press_steps and release_steps must be positive; hold_steps must be non-negative")

    @property
    def total_steps(self) -> int:
        return self.press_steps + self.hold_steps + self.release_steps

    def target_at(self, step: int) -> dict[str, float]:
        if step < 0 or step >= self.total_steps:
            raise IndexError(f"step {step} is outside [0, {self.total_steps})")
        if step < self.press_steps:
            fraction = (step + 1) / self.press_steps
        elif step < self.press_steps + self.hold_steps:
            fraction = 1.0
        else:
            released = step - self.press_steps - self.hold_steps + 1
            fraction = 1.0 - released / self.release_steps
        return synchronized_joint_targets(TRIGGER_LIMIT_RAD * clamp(fraction, 0.0, 1.0))
