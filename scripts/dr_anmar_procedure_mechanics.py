# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable procedure-level engineering models for Dr.Anmar rooms.

The native Isaac/ORBIT task remains the robot and contact authority.  These
small deterministic models add observable procedure state that the upstream
benchmarks do not provide: tube insertion, closure quality, vascular control,
hemostasis, ultrasound targeting and complication recovery.  The outputs are
research telemetry, not clinically calibrated measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator < 1e-10:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(np.dot(point - start, segment) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * segment)))


def _safe_unit(vector: np.ndarray, fallback: tuple[float, float, float] = (1.0, 0.0, 0.0)) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    length = float(np.linalg.norm(value))
    if length < 1e-8:
        return np.asarray(fallback, dtype=np.float32)
    return value / length


@dataclass
class TubeInsertionModel:
    entry: np.ndarray
    target: np.ndarray
    lumen_radius_m: float = 0.008
    shunt_length_m: float = 0.105
    previous_tip: np.ndarray | None = None
    insertion_depth_m: float = 0.0
    radial_error_m: float = 0.0
    alignment_error_deg: float = 90.0
    wall_load_proxy_n: float = 0.0
    peak_wall_load_proxy_n: float = 0.0
    buckling_events: int = 0
    buckled: bool = False
    patency_percent: float = 0.0

    def __post_init__(self) -> None:
        self.entry = np.asarray(self.entry, dtype=np.float32)
        self.target = np.asarray(self.target, dtype=np.float32)
        self.axis = _safe_unit(self.target - self.entry)
        self.target_depth_m = max(0.025, float(np.linalg.norm(self.target - self.entry)))

    def reset(self) -> None:
        self.previous_tip = None
        self.insertion_depth_m = 0.0
        self.radial_error_m = 0.0
        self.alignment_error_deg = 90.0
        self.wall_load_proxy_n = 0.0
        self.peak_wall_load_proxy_n = 0.0
        self.buckling_events = 0
        self.buckled = False
        self.patency_percent = 0.0

    def update(self, tip: np.ndarray | None, grasped: bool) -> None:
        if tip is None:
            return
        tip = np.asarray(tip, dtype=np.float32)
        along = float(np.dot(tip - self.entry, self.axis))
        axis_point = self.entry + self.axis * along
        self.radial_error_m = float(np.linalg.norm(tip - axis_point))
        self.insertion_depth_m = float(np.clip(along, 0.0, self.target_depth_m)) if grasped else 0.0
        if self.previous_tip is not None:
            motion = tip - self.previous_tip
            motion_length = float(np.linalg.norm(motion))
            if motion_length > 1e-6:
                cosine = float(np.clip(np.dot(motion / motion_length, self.axis), -1.0, 1.0))
                self.alignment_error_deg = float(np.degrees(np.arccos(abs(cosine))))
        excess = max(0.0, self.radial_error_m - self.lumen_radius_m)
        angle_factor = max(0.0, self.alignment_error_deg - 12.0) / 50.0
        self.wall_load_proxy_n = (
            float(np.clip(excess * 135.0 + angle_factor * 0.8, 0.0, 4.0))
            if grasped and self.insertion_depth_m > 0.0
            else 0.0
        )
        self.peak_wall_load_proxy_n = max(self.peak_wall_load_proxy_n, self.wall_load_proxy_n)
        buckled = bool(grasped and self.insertion_depth_m > 0.004 and self.wall_load_proxy_n > 0.9)
        if buckled and not self.buckled:
            self.buckling_events += 1
        self.buckled = buckled
        depth_fraction = self.insertion_depth_m / self.target_depth_m
        load_penalty = float(np.clip(self.wall_load_proxy_n / 2.0, 0.0, 1.0))
        buckle_penalty = 0.35 if self.buckled else 0.0
        self.patency_percent = float(np.clip(100.0 * depth_fraction * (1.0 - 0.55 * load_penalty - buckle_penalty), 0.0, 100.0))
        self.previous_tip = tip.copy()

    def curve_points(self, tip: np.ndarray | None, nodes: int = 18) -> np.ndarray:
        tip = np.asarray(tip if tip is not None else self.entry - self.axis * 0.04, dtype=np.float32)
        tail = tip - self.axis * self.shunt_length_m
        alpha = np.linspace(0.0, 1.0, nodes, dtype=np.float32)[:, None]
        points = tail[None, :] * (1.0 - alpha) + tip[None, :] * alpha
        sag = np.sin(np.linspace(0.0, np.pi, nodes, dtype=np.float32))[:, None]
        points += sag * np.asarray((0.0, 0.0, -0.006), dtype=np.float32)[None, :]
        return points

    def snapshot(self) -> dict[str, Any]:
        return {
            "active": True,
            "insertion_depth_m": round(self.insertion_depth_m, 5),
            "target_depth_m": round(self.target_depth_m, 5),
            "radial_error_m": round(self.radial_error_m, 5),
            "alignment_error_deg": round(self.alignment_error_deg, 1),
            "wall_load_proxy_n": round(self.wall_load_proxy_n, 3),
            "peak_wall_load_proxy_n": round(self.peak_wall_load_proxy_n, 3),
            "buckled": self.buckled,
            "buckling_events": self.buckling_events,
            "patency_percent": round(self.patency_percent, 1),
        }


@dataclass
class ClosureQualityModel:
    target_stitches: int
    mode: str
    pressure_kpa: float = 0.0

    def snapshot(self, thread: Any | None, waypoint_count: int) -> dict[str, Any]:
        anchors = len(getattr(thread, "tissue_anchor_indices", [])) if thread is not None else 0
        stitches = int(getattr(thread, "stitch_count", anchors // 2)) if thread is not None else 0
        tension = float(getattr(thread, "tension_n", 0.0)) if thread is not None else 0.0
        tightness = float(getattr(thread, "knot_tightness", 0.0)) if thread is not None else 0.0
        spacing = float(getattr(thread, "mean_anchor_spacing_m", 0.0)) if thread is not None else 0.0
        variation = float(getattr(thread, "spacing_variation_m", 0.0)) if thread is not None else 0.0
        completion = float(np.clip(stitches / max(1, self.target_stitches), 0.0, 1.0))
        closure_gap = float(np.clip(0.012 * (1.0 - completion) + max(0.0, 0.20 - tightness) * 0.003, 0.0004, 0.016))
        narrowing = float(np.clip(max(0.0, tension - 0.65) * 22.0 + max(0.0, tightness - 0.92) * 35.0, 0.0, 100.0))
        if self.mode == "anastomosis" and waypoint_count >= 7:
            self.pressure_kpa = min(12.0, self.pressure_kpa + 0.18)
        leak_rate = float(np.clip(180.0 * (closure_gap / 0.012) ** 2 + narrowing * 0.45, 0.0, 260.0))
        return {
            "active": True,
            "mode": self.mode,
            "anchors": anchors,
            "stitch_count": stitches,
            "target_stitches": self.target_stitches,
            "mean_bite_spacing_m": round(spacing, 5),
            "spacing_variation_m": round(variation, 5),
            "closure_gap_m": round(closure_gap, 5),
            "lumen_narrowing_percent": round(narrowing, 1),
            "test_pressure_kpa": round(self.pressure_kpa, 2),
            "leak_rate_proxy_ml_min": round(leak_rate, 1),
            "over_tension_events": int(getattr(thread, "over_tension_events", 0)) if thread is not None else 0,
        }


@dataclass
class VascularControlModel:
    mode: str
    baseline_bleed_rate_ml_min: float = 240.0
    blood_loss_proxy_ml: float = 0.0
    rebleed: bool = False

    def reset(self) -> None:
        self.blood_loss_proxy_ml = 0.0
        self.rebleed = False

    def snapshot(
        self,
        waypoint_count: int,
        source_distance_m: float | None,
        grippers_closed: int,
        dt: float,
    ) -> dict[str, Any]:
        distance = float(source_distance_m if source_distance_m is not None else 1.0)
        proximity = float(np.clip(1.0 - distance / 0.045, 0.0, 1.0))
        if self.mode == "clip_divide":
            clips = int(waypoint_count >= 2) + int(waypoint_count >= 3)
            divided = waypoint_count >= 4
            safe_interval = bool(divided and clips == 2)
            residual_flow = 100.0 if clips == 0 else 45.0 if clips == 1 else 3.0
            if divided and clips < 2:
                residual_flow = 100.0
            return {
                "active": True,
                "mode": self.mode,
                "clips_placed": clips,
                "divided": divided,
                "division_inside_protected_interval": safe_interval,
                "residual_flow_percent": residual_flow,
                "clip_spacing_m": 0.018 if clips == 2 else 0.0,
            }
        suction = proximity * (0.50 if waypoint_count >= 2 else 0.20)
        compression = proximity * min(1.0, grippers_closed * 0.55)
        definitive = 0.88 if waypoint_count >= 4 else 0.0
        control = float(np.clip(max(suction + compression, definitive), 0.0, 0.98))
        bleed_rate = self.baseline_bleed_rate_ml_min * (1.0 - control)
        self.blood_loss_proxy_ml += bleed_rate * max(0.0, dt) / 60.0
        self.rebleed = bool(waypoint_count >= 4 and distance > 0.06 and definitive < 0.9)
        return {
            "active": True,
            "mode": self.mode,
            "source_distance_m": round(distance, 5),
            "bleed_rate_proxy_ml_min": round(bleed_rate, 1),
            "blood_loss_proxy_ml": round(self.blood_loss_proxy_ml, 2),
            "visibility_percent": round(float(np.clip(100.0 - bleed_rate * 0.30, 8.0, 100.0)), 1),
            "controlled": bleed_rate <= 35.0,
            "rebleed": self.rebleed,
        }


@dataclass
class UltrasoundAccessModel:
    target: np.ndarray
    protected_center: np.ndarray
    protected_radius_m: float = 0.012
    min_target_error_m: float = 1.0

    def __post_init__(self) -> None:
        self.target = np.asarray(self.target, dtype=np.float32)
        self.protected_center = np.asarray(self.protected_center, dtype=np.float32)

    def reset(self) -> None:
        self.min_target_error_m = 1.0

    def snapshot(self, tool: np.ndarray | None, waypoints: np.ndarray) -> dict[str, Any]:
        if tool is None:
            return {"active": True, "target_visible": False}
        tool = np.asarray(tool, dtype=np.float32)
        target_error = float(np.linalg.norm(tool - self.target))
        self.min_target_error_m = min(self.min_target_error_m, target_error)
        path_error = min(
            (_point_segment_distance(tool, start, end) for start, end in zip(waypoints[:-1], waypoints[1:])),
            default=target_error,
        )
        protected_clearance = float(np.linalg.norm(tool - self.protected_center) - self.protected_radius_m)
        confidence = float(np.clip(1.0 - target_error / 0.09, 0.0, 1.0))
        visibility = float(np.clip(1.0 - path_error / 0.035, 0.0, 1.0))
        return {
            "active": True,
            "target_visible": confidence >= 0.25,
            "target_confidence": round(confidence, 3),
            "needle_visibility": round(visibility, 3),
            "target_error_m": round(target_error, 5),
            "minimum_target_error_m": round(self.min_target_error_m, 5),
            "protected_clearance_m": round(protected_clearance, 5),
            "target_contact": target_error <= 0.008 and protected_clearance > 0.0,
        }


@dataclass
class ProcedureMechanics:
    kind: str
    waypoints: np.ndarray
    target_stitches: int = 1
    tube: TubeInsertionModel | None = field(init=False, default=None)
    closure: ClosureQualityModel | None = field(init=False, default=None)
    vascular: VascularControlModel | None = field(init=False, default=None)
    ultrasound: UltrasoundAccessModel | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.waypoints = np.asarray(self.waypoints, dtype=np.float32).reshape(-1, 3)
        if self.kind == "tube_insertion" and len(self.waypoints) >= 3:
            self.tube = TubeInsertionModel(self.waypoints[1], self.waypoints[-1])
        if self.kind in {"running_suture", "anastomosis", "knot_tying", "threading"}:
            self.closure = ClosureQualityModel(max(1, self.target_stitches), self.kind)
        if self.kind in {"clip_divide", "hemostasis"}:
            self.vascular = VascularControlModel(self.kind)
        if self.kind == "ultrasound_access" and len(self.waypoints):
            target = self.waypoints[-1]
            protected = target + np.asarray((0.018, -0.012, -0.004), dtype=np.float32)
            self.ultrasound = UltrasoundAccessModel(target, protected)

    def reset(self) -> None:
        if self.tube:
            self.tube.reset()
        if self.vascular:
            self.vascular.reset()
        if self.ultrasound:
            self.ultrasound.reset()
        if self.closure:
            self.closure.pressure_kpa = 0.0

    def update(
        self,
        tool_positions: dict[int, np.ndarray],
        grippers_open: list[bool],
        assisted_grasped: bool,
        waypoint_count: int,
        thread: Any | None,
        cut: dict[str, Any],
        dt: float,
        scenario_id: str,
    ) -> dict[str, dict[str, Any]]:
        primary = tool_positions.get(0)
        secondary = tool_positions.get(1)
        closed = sum(not value for value in grippers_open)
        result: dict[str, dict[str, Any]] = {}
        if self.tube:
            self.tube.update(primary, assisted_grasped)
            result["tube"] = self.tube.snapshot()
        if self.closure:
            result["closure"] = self.closure.snapshot(thread, waypoint_count)
        if self.vascular:
            source = self.waypoints[2] if len(self.waypoints) > 2 else self.waypoints[-1]
            distances = [float(np.linalg.norm(position - source)) for position in (primary, secondary) if position is not None]
            result["vascular"] = self.vascular.snapshot(waypoint_count, min(distances) if distances else None, closed, dt)
        if self.ultrasound:
            result["ultrasound"] = self.ultrasound.snapshot(primary, self.waypoints)
        if self.kind in {"dissection", "biopsy"}:
            progress = float(np.clip(waypoint_count / max(1, len(self.waypoints)), 0.0, 1.0))
            faces = int(cut.get("faces_removed", 0))
            result["dissection"] = {
                "active": True,
                "mode": self.kind,
                "plane_progress": round(progress, 3),
                "faces_separated": faces,
                "protected_contact": False,
                "margin_consistency_percent": round(100.0 * progress * (1.0 if faces else 0.35), 1),
                "specimen_released": bool(self.kind == "biopsy" and progress >= 0.85 and faces > 8),
            }
        if self.kind == "recovery":
            failure_active = scenario_id != "baseline"
            result["recovery"] = {
                "active": True,
                "failure_injected": failure_active,
                "failure_id": scenario_id,
                "object_reacquired": assisted_grasped,
                "recovery_progress": round(float(np.clip(waypoint_count / max(1, len(self.waypoints)), 0.0, 1.0)), 3),
                "stable_recovery": bool(assisted_grasped and waypoint_count >= max(1, len(self.waypoints) - 1)),
            }
        return result
