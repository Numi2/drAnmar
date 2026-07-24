"""Dependency-free geometry helpers for skin-stapler closure tasks.

The returned values are simulation metrics, not clinical outcome metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

Vec3 = tuple[float, float, float]


def _vec3(value: Sequence[float], *, name: str) -> Vec3:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    result = (float(value[0]), float(value[1]), float(value[2]))
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(value: Vec3, scalar: float) -> Vec3:
    return (value[0] * scalar, value[1] * scalar, value[2] * scalar)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(value: Vec3) -> float:
    return math.sqrt(_dot(value, value))


def _normalized(value: Sequence[float], *, name: str) -> Vec3:
    vector = _vec3(value, name=name)
    norm = _norm(vector)
    if norm <= 1e-12:
        raise ValueError(f"{name} must have non-zero length")
    return _scale(vector, 1.0 / norm)


@dataclass(frozen=True)
class ClosureLine:
    start_m: Vec3
    end_m: Vec3

    def __post_init__(self) -> None:
        start = _vec3(self.start_m, name="start_m")
        end = _vec3(self.end_m, name="end_m")
        if _norm(_sub(end, start)) <= 1e-9:
            raise ValueError("closure line must have non-zero length")
        object.__setattr__(self, "start_m", start)
        object.__setattr__(self, "end_m", end)

    @property
    def length_m(self) -> float:
        return _norm(_sub(self.end_m, self.start_m))

    @property
    def tangent(self) -> Vec3:
        delta = _sub(self.end_m, self.start_m)
        return _scale(delta, 1.0 / _norm(delta))

    def point_at(self, longitudinal_m: float) -> Vec3:
        """Return the point at a signed distance along the line tangent."""
        return _add(self.start_m, _scale(self.tangent, float(longitudinal_m)))

    def evenly_spaced_targets(self, spacing_m: float, *, include_endpoints: bool = True) -> list[Vec3]:
        """Generate deterministic placement targets along the closure line."""
        spacing = float(spacing_m)
        if not math.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("spacing_m must be finite and positive")
        length = self.length_m
        if include_endpoints:
            count = max(2, int(math.floor(length / spacing)) + 1)
            if count == 2:
                distances = (0.0, length)
            else:
                actual = length / (count - 1)
                distances = tuple(index * actual for index in range(count))
        else:
            count = int(math.floor(length / spacing))
            distances = tuple((index + 1) * spacing for index in range(count) if (index + 1) * spacing < length)
        return [self.point_at(distance) for distance in distances]


@dataclass(frozen=True)
class PlacementAssessment:
    longitudinal_m: float
    lateral_error_m: float
    orientation_error_deg: float
    within_line_extent: bool


def assess_placement(
    closure_line: ClosureLine,
    placement_position_m: Sequence[float],
    staple_crown_direction: Sequence[float],
) -> PlacementAssessment:
    """Assess a simulated staple pose relative to a closure line.

    The staple crown is nominally perpendicular to the closure-line tangent.
    Orientation error is reported in the range 0–90 degrees.
    """
    point = _vec3(placement_position_m, name="placement_position_m")
    tangent = closure_line.tangent
    relative = _sub(point, closure_line.start_m)
    longitudinal = _dot(relative, tangent)
    closest = closure_line.point_at(longitudinal)
    lateral = _norm(_sub(point, closest))
    crown = _normalized(staple_crown_direction, name="staple_crown_direction")
    parallel_component = min(max(abs(_dot(crown, tangent)), 0.0), 1.0)
    perpendicular_error = abs(90.0 - math.degrees(math.acos(parallel_component)))
    return PlacementAssessment(
        longitudinal_m=longitudinal,
        lateral_error_m=lateral,
        orientation_error_deg=perpendicular_error,
        within_line_extent=0.0 <= longitudinal <= closure_line.length_m,
    )


def spacing_errors_m(longitudinal_positions_m: Sequence[float], target_spacing_m: float) -> list[float]:
    target = float(target_spacing_m)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("target_spacing_m must be finite and positive")
    ordered = sorted(float(value) for value in longitudinal_positions_m)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("longitudinal positions must be finite")
    return [abs((b - a) - target) for a, b in zip(ordered, ordered[1:])]
