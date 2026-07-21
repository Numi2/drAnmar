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
    stable_time_s: float = 0.0
    placement_verified: bool = False

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
        self.stable_time_s = 0.0
        self.placement_verified = False

    def update(self, tip: np.ndarray | None, grasped: bool, dt: float) -> None:
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
        speed = float(np.linalg.norm(tip - self.previous_tip)) / max(float(dt), 1e-4) if self.previous_tip is not None else 0.0
        stable = bool(
            grasped
            and depth_fraction >= 0.88
            and self.radial_error_m <= self.lumen_radius_m
            and self.wall_load_proxy_n <= 0.55
            and not self.buckled
            and speed <= 0.018
        )
        self.stable_time_s = self.stable_time_s + max(0.0, float(dt)) if stable else max(0.0, self.stable_time_s - 2.0 * max(0.0, float(dt)))
        self.placement_verified = self.stable_time_s >= 0.65
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
            "stable_time_s": round(self.stable_time_s, 2),
            "placement_verified": self.placement_verified,
        }


@dataclass
class ClosureQualityModel:
    target_stitches: int
    mode: str
    target_throws: int = 3
    pressure_kpa: float = 0.0
    throw_count: int = 0
    alternating_crossings: int = 0
    last_tool_side: int = 0
    crossing_cooldown_s: float = 0.0
    stable_closure_time_s: float = 0.0

    def reset(self) -> None:
        self.pressure_kpa = 0.0
        self.throw_count = 0
        self.alternating_crossings = 0
        self.last_tool_side = 0
        self.crossing_cooldown_s = 0.0
        self.stable_closure_time_s = 0.0

    def snapshot(
        self,
        thread: Any | None,
        tool_positions: dict[int, np.ndarray],
        grippers_open: list[bool],
        dt: float,
    ) -> dict[str, Any]:
        anchors = len(getattr(thread, "tissue_anchor_indices", [])) if thread is not None else 0
        stitches = int(getattr(thread, "stitch_count", anchors // 2)) if thread is not None else 0
        tension = float(getattr(thread, "tension_n", 0.0)) if thread is not None else 0.0
        tightness = float(getattr(thread, "knot_tightness", 0.0)) if thread is not None else 0.0
        knot_security = float(getattr(thread, "knot_security", 0.0)) if thread is not None else 0.0
        tissue_tear_events = int(getattr(thread, "tissue_tear_events", 0)) if thread is not None else 0
        anchor_pullouts = int(getattr(thread, "anchor_pullouts", 0)) if thread is not None else 0
        spacing = float(getattr(thread, "mean_anchor_spacing_m", 0.0)) if thread is not None else 0.0
        variation = float(getattr(thread, "spacing_variation_m", 0.0)) if thread is not None else 0.0
        completion = float(np.clip(stitches / max(1, self.target_stitches), 0.0, 1.0))
        physical_closure_gap = float(getattr(thread, "closure_gap_m", 0.0)) if thread is not None else 0.0
        closure_gap = (
            float(np.clip(physical_closure_gap, 0.0002, 0.020))
            if physical_closure_gap > 0.0
            else float(np.clip(0.012 * (1.0 - completion) + max(0.0, 0.20 - tightness) * 0.003, 0.0004, 0.016))
        )
        closure_ratio = float(getattr(thread, "closure_ratio", 0.0)) if thread is not None else 0.0
        retained_closure = float(getattr(thread, "retained_closure", 0.0)) if thread is not None else 0.0
        bite_depth = float(getattr(thread, "mean_bite_depth_m", 0.0)) if thread is not None else 0.0
        anchor_slip = float(getattr(thread, "total_anchor_slip_m", 0.0)) if thread is not None else 0.0
        narrowing = float(np.clip(max(0.0, tension - 0.65) * 22.0 + max(0.0, tightness - 0.92) * 35.0, 0.0, 100.0))
        self.crossing_cooldown_s = max(0.0, self.crossing_cooldown_s - max(0.0, float(dt)))
        primary = tool_positions.get(0)
        secondary = tool_positions.get(1)
        both_closed = len(grippers_open) >= 2 and not grippers_open[0] and not grippers_open[1]
        if self.mode == "knot_tying" and primary is not None and secondary is not None and both_closed:
            relative = np.asarray(primary, dtype=np.float32) - np.asarray(secondary, dtype=np.float32)
            side = 1 if relative[0] >= 0.008 else -1 if relative[0] <= -0.008 else 0
            close_enough = float(np.linalg.norm(relative)) <= 0.095
            if side and close_enough and self.last_tool_side and side != self.last_tool_side and self.crossing_cooldown_s <= 0.0:
                self.throw_count = min(self.target_throws, self.throw_count + 1)
                self.alternating_crossings += 1
                self.crossing_cooldown_s = 0.32
            if side:
                self.last_tool_side = side
        secure_enough = knot_security >= 0.45 if self.mode in {"threading", "running_suture", "anastomosis", "knot_tying"} else True
        closure_stable = bool(
            completion >= 1.0
            and closure_gap <= 0.006
            and tension <= 0.95
            and secure_enough
            and tissue_tear_events == 0
            and not bool(getattr(thread, "thread_broken", False))
        )
        self.stable_closure_time_s = (
            self.stable_closure_time_s + max(0.0, float(dt))
            if closure_stable
            else max(0.0, self.stable_closure_time_s - max(0.0, float(dt)))
        )
        if self.mode == "anastomosis" and self.stable_closure_time_s >= 0.6:
            self.pressure_kpa = min(12.0, self.pressure_kpa + 2.4 * max(0.0, float(dt)))
        leak_rate = float(
            np.clip(180.0 * (closure_gap / 0.012) ** 2 + narrowing * 0.45 + tissue_tear_events * 28.0, 0.0, 320.0)
        )
        slippage = float(
            np.clip(
                1.0 - knot_security * 0.70 - self.throw_count / max(1, self.target_throws) * 0.30 + anchor_pullouts * 0.18,
                0.0,
                1.0,
            )
        )
        return {
            "active": True,
            "mode": self.mode,
            "anchors": anchors,
            "stitch_count": stitches,
            "target_stitches": self.target_stitches,
            "mean_bite_spacing_m": round(spacing, 5),
            "spacing_variation_m": round(variation, 5),
            "closure_gap_m": round(closure_gap, 5),
            "closure_ratio": round(closure_ratio, 4),
            "retained_closure": round(retained_closure, 4),
            "mean_bite_depth_m": round(bite_depth, 5),
            "anchor_slip_m": round(anchor_slip, 6),
            "lumen_narrowing_percent": round(narrowing, 1),
            "test_pressure_kpa": round(self.pressure_kpa, 2),
            "leak_rate_proxy_ml_min": round(leak_rate, 1),
            "over_tension_events": int(getattr(thread, "over_tension_events", 0)) if thread is not None else 0,
            "tissue_tear_events": tissue_tear_events,
            "anchor_pullouts": anchor_pullouts,
            "thread_broken": bool(getattr(thread, "thread_broken", False)) if thread is not None else False,
            "knot_security": round(knot_security, 3),
            "throw_count": self.throw_count,
            "target_throws": self.target_throws,
            "alternating_crossings": self.alternating_crossings,
            "slippage_proxy": round(slippage, 3),
            "stable_closure_time_s": round(self.stable_closure_time_s, 2),
            "pressure_test_ready": closure_stable,
        }


@dataclass
class VascularControlModel:
    mode: str
    waypoints: np.ndarray
    baseline_bleed_rate_ml_min: float = 240.0
    blood_loss_proxy_ml: float = 0.0
    rebleed: bool = False
    placed_sites: list[int] = field(default_factory=list)
    divided: bool = False
    division_inside_interval: bool = False
    off_target_deployments: int = 0
    protected_violations: int = 0
    definitive_control: bool = False
    localized_time_s: float = 0.0
    time_to_control_s: float | None = None
    stable_control_time_s: float = 0.0
    elapsed_s: float = 0.0
    previous_tools: dict[int, np.ndarray] = field(default_factory=dict)
    previous_open: list[bool] = field(default_factory=list)
    clip_retention: dict[int, float] = field(default_factory=dict)
    compression_force_proxy_n: float = 0.0
    peak_compression_force_proxy_n: float = 0.0
    overcompression_events: int = 0
    overcompression_active: bool = False
    vessel_damage: float = 0.0

    def __post_init__(self) -> None:
        self.waypoints = np.asarray(self.waypoints, dtype=np.float32).reshape(-1, 3)
        if self.mode == "clip_divide":
            self.clip_sites = [self.waypoints[min(index, len(self.waypoints) - 1)] for index in (1, 3)]
            self.division_center = (self.clip_sites[0] + self.clip_sites[1]) * 0.5
            self.source = self.division_center
        else:
            self.clip_sites = []
            self.source = self.waypoints[min(2, len(self.waypoints) - 1)]
            self.division_center = self.source

    def reset(self) -> None:
        self.blood_loss_proxy_ml = 0.0
        self.rebleed = False
        self.placed_sites.clear()
        self.divided = False
        self.division_inside_interval = False
        self.off_target_deployments = 0
        self.protected_violations = 0
        self.definitive_control = False
        self.localized_time_s = 0.0
        self.time_to_control_s = None
        self.stable_control_time_s = 0.0
        self.elapsed_s = 0.0
        self.previous_tools.clear()
        self.previous_open.clear()
        self.clip_retention.clear()
        self.compression_force_proxy_n = 0.0
        self.peak_compression_force_proxy_n = 0.0
        self.overcompression_events = 0
        self.overcompression_active = False
        self.vessel_damage = 0.0

    def snapshot(
        self,
        tool_positions: dict[int, np.ndarray],
        grippers_open: list[bool],
        dt: float,
    ) -> dict[str, Any]:
        self.elapsed_s += max(0.0, float(dt))
        positions = {arm: np.asarray(position, dtype=np.float32) for arm, position in tool_positions.items()}
        distances = [float(np.linalg.norm(position - self.source)) for position in positions.values()]
        distance = min(distances, default=1.0)
        proximity = float(np.clip(1.0 - distance / 0.045, 0.0, 1.0))
        closed_near_source = sum(
            arm < len(grippers_open)
            and not grippers_open[arm]
            and float(np.linalg.norm(position - self.source)) <= 0.032
            for arm, position in positions.items()
        )
        self.compression_force_proxy_n = float(np.clip(closed_near_source * proximity * 0.82, 0.0, 2.6))
        self.peak_compression_force_proxy_n = max(self.peak_compression_force_proxy_n, self.compression_force_proxy_n)
        overcompressed = self.compression_force_proxy_n > 1.35
        if overcompressed and not self.overcompression_active:
            self.overcompression_events += 1
        self.overcompression_active = overcompressed
        if overcompressed:
            self.vessel_damage = float(np.clip(self.vessel_damage + (self.compression_force_proxy_n - 1.35) * max(dt, 0.0) * 0.28, 0.0, 1.0))
        close_edges = [
            arm
            for arm, is_open in enumerate(grippers_open)
            if arm < len(self.previous_open) and self.previous_open[arm] and not is_open
        ]
        if self.mode == "clip_divide":
            for arm in close_edges:
                position = positions.get(arm)
                if position is None:
                    continue
                available = [site for site in range(2) if site not in self.placed_sites]
                nearest = min(available, key=lambda site: float(np.linalg.norm(position - self.clip_sites[site])), default=None)
                if nearest is not None and float(np.linalg.norm(position - self.clip_sites[nearest])) <= 0.020:
                    self.placed_sites.append(nearest)
                    self.clip_retention[nearest] = 1.0
                elif float(np.linalg.norm(position - self.source)) <= 0.055:
                    self.off_target_deployments += 1
            if len(self.placed_sites) >= 2 and not self.divided:
                for arm, position in positions.items():
                    previous = self.previous_tools.get(arm)
                    if previous is None or float(np.linalg.norm(position - previous)) < 0.001:
                        continue
                    center_distance = _point_segment_distance(self.division_center, previous, position)
                    exposure_held = any(
                        other != arm
                        and other < len(grippers_open)
                        and not grippers_open[other]
                        and float(np.linalg.norm(other_position - self.source)) <= 0.060
                        for other, other_position in positions.items()
                    )
                    if center_distance <= 0.010 and exposure_held:
                        self.divided = True
                        self.division_inside_interval = True
                        break
            elif len(self.placed_sites) < 2:
                for arm, position in positions.items():
                    previous = self.previous_tools.get(arm)
                    if previous is not None and _point_segment_distance(self.division_center, previous, position) <= 0.007:
                        if float(np.linalg.norm(position - previous)) >= 0.002:
                            self.protected_violations += 1
                            break
            clips = len(self.placed_sites)
            divided = self.divided
            safe_interval = self.division_inside_interval
            for site in list(self.placed_sites):
                disturbance = max(
                    (
                        float(np.linalg.norm(position - self.previous_tools[arm]))
                        for arm, position in positions.items()
                        if arm in self.previous_tools
                        and float(np.linalg.norm(position - self.clip_sites[site])) <= 0.025
                    ),
                    default=0.0,
                )
                if disturbance > 0.006:
                    self.clip_retention[site] = max(
                        0.0,
                        self.clip_retention.get(site, 1.0) - (disturbance - 0.006) * 3.5,
                    )
            retained = sum(self.clip_retention.get(site, 0.0) >= 0.65 for site in self.placed_sites)
            residual_flow = 100.0 if retained == 0 else 45.0 if retained == 1 else 3.0
            if divided and clips < 2:
                residual_flow = 100.0
            if divided and retained < 2:
                self.rebleed = True
                residual_flow = max(residual_flow, 72.0)
            output = {
                "active": True,
                "mode": self.mode,
                "clips_placed": clips,
                "divided": divided,
                "division_inside_protected_interval": safe_interval,
                "residual_flow_percent": residual_flow,
                "clip_spacing_m": round(float(np.linalg.norm(self.clip_sites[0] - self.clip_sites[1])), 5) if clips == 2 else 0.0,
                "off_target_deployments": self.off_target_deployments,
                "protected_violations": self.protected_violations,
                "clips_retained": retained,
                "clip_retention_min": round(min(self.clip_retention.values(), default=0.0), 3),
                "compression_force_proxy_n": round(self.compression_force_proxy_n, 3),
                "overcompression_events": self.overcompression_events,
                "vessel_damage_proxy": round(self.vessel_damage, 3),
                "rebleed": self.rebleed,
                "calibration_status": "research_defaults_unvalidated",
            }
        else:
            suction = proximity * 0.52
            compression = min(0.58, self.compression_force_proxy_n / 1.55)
            if distance <= 0.028:
                self.localized_time_s += max(0.0, float(dt))
            if any(
                arm in close_edges and float(np.linalg.norm(positions[arm] - self.source)) <= 0.020
                for arm in close_edges
                if arm in positions
            ) and self.localized_time_s >= 0.35:
                self.definitive_control = True
                self.rebleed = False
            if self.definitive_control and (
                (distance > 0.070 and self.stable_control_time_s < 0.8) or self.vessel_damage >= 0.55
            ):
                self.definitive_control = False
                self.rebleed = True
                self.stable_control_time_s = 0.0
            definitive = 0.90 if self.definitive_control else 0.0
            control = float(np.clip(max(suction + compression, definitive), 0.0, 0.98))
            bleed_rate = self.baseline_bleed_rate_ml_min * (1.0 - control) * (1.0 + self.vessel_damage * 0.75)
            self.blood_loss_proxy_ml += bleed_rate * max(0.0, dt) / 60.0
            controlled = bleed_rate <= 35.0
            self.stable_control_time_s = self.stable_control_time_s + max(0.0, float(dt)) if controlled else 0.0
            if controlled and self.time_to_control_s is None:
                self.time_to_control_s = self.elapsed_s
            output = {
                "active": True,
                "mode": self.mode,
                "source_distance_m": round(distance, 5),
                "bleed_rate_proxy_ml_min": round(bleed_rate, 1),
                "blood_loss_proxy_ml": round(self.blood_loss_proxy_ml, 2),
                "visibility_percent": round(float(np.clip(100.0 - bleed_rate * 0.30, 8.0, 100.0)), 1),
                "localized": self.localized_time_s >= 0.35,
                "time_to_localize_s": round(self.elapsed_s - self.localized_time_s + 0.35, 2) if self.localized_time_s >= 0.35 else None,
                "definitive_control": self.definitive_control,
                "time_to_control_s": round(self.time_to_control_s, 2) if self.time_to_control_s is not None else None,
                "stable_control_time_s": round(self.stable_control_time_s, 2),
                "controlled": controlled,
                "rebleed": self.rebleed,
                "compression_force_proxy_n": round(self.compression_force_proxy_n, 3),
                "peak_compression_force_proxy_n": round(self.peak_compression_force_proxy_n, 3),
                "overcompression_events": self.overcompression_events,
                "vessel_damage_proxy": round(self.vessel_damage, 3),
                "calibration_status": "research_defaults_unvalidated",
            }
        self.previous_tools = {arm: position.copy() for arm, position in positions.items()}
        self.previous_open = list(grippers_open)
        return output


@dataclass
class UltrasoundAccessModel:
    target: np.ndarray
    protected_center: np.ndarray
    protected_radius_m: float = 0.012
    min_target_error_m: float = 1.0
    scan_pose: np.ndarray | None = None
    previous_probe: np.ndarray | None = None
    stable_probe_time_s: float = 0.0
    protected_contacts: int = 0
    was_protected_contact: bool = False
    target_reached: bool = False
    withdrawn_on_path: bool = False

    def __post_init__(self) -> None:
        self.target = np.asarray(self.target, dtype=np.float32)
        self.protected_center = np.asarray(self.protected_center, dtype=np.float32)
        self.scan_pose = np.asarray(self.scan_pose if self.scan_pose is not None else self.target + (-0.035, -0.020, 0.025), dtype=np.float32)

    def reset(self) -> None:
        self.min_target_error_m = 1.0
        self.previous_probe = None
        self.stable_probe_time_s = 0.0
        self.protected_contacts = 0
        self.was_protected_contact = False
        self.target_reached = False
        self.withdrawn_on_path = False

    def snapshot(self, probe: np.ndarray | None, needle: np.ndarray | None, waypoints: np.ndarray, dt: float) -> dict[str, Any]:
        if probe is None or needle is None:
            return {"active": True, "target_visible": False}
        probe = np.asarray(probe, dtype=np.float32)
        needle = np.asarray(needle, dtype=np.float32)
        target_error = float(np.linalg.norm(needle - self.target))
        self.min_target_error_m = min(self.min_target_error_m, target_error)
        path_error = min(
            (_point_segment_distance(needle, start, end) for start, end in zip(waypoints[:-1], waypoints[1:])),
            default=target_error,
        )
        protected_clearance = float(np.linalg.norm(needle - self.protected_center) - self.protected_radius_m)
        probe_error = float(np.linalg.norm(probe - self.scan_pose))
        probe_speed = float(np.linalg.norm(probe - self.previous_probe)) / max(float(dt), 1e-4) if self.previous_probe is not None else 0.0
        probe_stable = probe_error <= 0.035 and probe_speed <= 0.028
        self.stable_probe_time_s = self.stable_probe_time_s + max(0.0, float(dt)) if probe_stable else max(0.0, self.stable_probe_time_s - 2.0 * max(0.0, float(dt)))
        confidence = float(np.clip(1.0 - probe_error / 0.075, 0.0, 1.0)) * float(np.clip(self.stable_probe_time_s / 0.45, 0.0, 1.0))
        visibility = float(np.clip(1.0 - path_error / 0.035, 0.0, 1.0)) * confidence
        protected_contact = protected_clearance <= 0.0
        if protected_contact and not self.was_protected_contact:
            self.protected_contacts += 1
        self.was_protected_contact = protected_contact
        target_contact = target_error <= 0.008 and protected_clearance > 0.0 and confidence >= 0.55
        self.target_reached = self.target_reached or target_contact
        if self.target_reached and target_error >= 0.030 and visibility >= 0.35 and not protected_contact:
            self.withdrawn_on_path = True
        self.previous_probe = probe.copy()
        return {
            "active": True,
            "target_visible": confidence >= 0.25,
            "target_confidence": round(confidence, 3),
            "probe_error_m": round(probe_error, 5),
            "probe_stable_time_s": round(self.stable_probe_time_s, 2),
            "needle_visibility": round(visibility, 3),
            "target_error_m": round(target_error, 5),
            "minimum_target_error_m": round(self.min_target_error_m, 5),
            "protected_clearance_m": round(protected_clearance, 5),
            "protected_contacts": self.protected_contacts,
            "target_contact": target_contact,
            "target_reached": self.target_reached,
            "withdrawn_on_path": self.withdrawn_on_path,
        }


@dataclass
class HoopThreadingModel:
    """Score a curved needle passing through a rigid training hoop."""

    approach: np.ndarray
    center: np.ndarray
    exit: np.ndarray
    inner_radius_m: float = 0.017
    tube_radius_m: float = 0.0025
    previous_points: np.ndarray | None = None
    approach_seen: bool = False
    pass_count: int = 0
    ring_contacts: int = 0
    contact_active: bool = False
    minimum_center_error_m: float = 1.0
    best_clearance_m: float = -1.0
    passed_cleanly: bool = False
    recovered: bool = False
    entered_endpoint_indices: set[int] = field(default_factory=set)
    passed_endpoint_indices: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.approach = np.asarray(self.approach, dtype=np.float32)
        self.center = np.asarray(self.center, dtype=np.float32)
        self.exit = np.asarray(self.exit, dtype=np.float32)
        self.normal = _safe_unit(self.exit - self.approach)

    def reset(self) -> None:
        self.previous_points = None
        self.approach_seen = False
        self.pass_count = 0
        self.ring_contacts = 0
        self.contact_active = False
        self.minimum_center_error_m = 1.0
        self.best_clearance_m = -1.0
        self.passed_cleanly = False
        self.recovered = False
        self.entered_endpoint_indices.clear()
        self.passed_endpoint_indices.clear()

    def _coordinates(self, point: np.ndarray) -> tuple[float, float]:
        offset = np.asarray(point, dtype=np.float32) - self.center
        axial = float(np.dot(offset, self.normal))
        radial = float(np.linalg.norm(offset - self.normal * axial))
        return axial, radial

    def snapshot(self, needle_points: np.ndarray | None) -> dict[str, Any]:
        points = np.asarray(needle_points, dtype=np.float32).reshape(-1, 3) if needle_points is not None else np.empty((0, 3), dtype=np.float32)
        if not len(points):
            return {
                "active": True,
                "needle_visible": False,
                "pass_count": self.pass_count,
                "ring_contacts": self.ring_contacts,
                "passed_cleanly": self.passed_cleanly,
            }

        coordinates = [self._coordinates(point) for point in points]
        closest_axial, closest_radial = min(coordinates, key=lambda value: abs(value[0]))
        center_error = float(np.hypot(closest_axial, closest_radial))
        self.minimum_center_error_m = min(self.minimum_center_error_m, center_error)
        clearance = self.inner_radius_m - closest_radial
        self.best_clearance_m = max(self.best_clearance_m, clearance)
        self.approach_seen = self.approach_seen or any(axial <= -0.008 for axial, _radial in coordinates)

        ring_contact = any(
            abs(axial) <= self.tube_radius_m * 1.45
            and abs(radial - (self.inner_radius_m + self.tube_radius_m)) <= self.tube_radius_m * 1.55
            for axial, radial in coordinates
        )
        if ring_contact and not self.contact_active:
            self.ring_contacts += 1
        self.contact_active = ring_contact

        crossing_detected = False
        for index, (axial, radial) in enumerate(coordinates):
            if axial <= -0.002 and radial <= self.inner_radius_m - 0.0008:
                self.entered_endpoint_indices.add(index)
            if (
                index in self.entered_endpoint_indices
                and index not in self.passed_endpoint_indices
                and axial >= 0.002
                and radial <= self.inner_radius_m - 0.0008
            ):
                self.passed_endpoint_indices.add(index)
                crossing_detected = True

        if self.previous_points is not None and self.approach_seen and not crossing_detected:
            for index, (previous, current) in enumerate(zip(self.previous_points, points)):
                previous_axial, _ = self._coordinates(previous)
                current_axial, _ = self._coordinates(current)
                if previous_axial >= -0.001 or current_axial <= 0.001:
                    continue
                fraction = -previous_axial / max(current_axial - previous_axial, 1e-8)
                intersection = previous + float(np.clip(fraction, 0.0, 1.0)) * (current - previous)
                _axial, radial = self._coordinates(intersection)
                if radial <= self.inner_radius_m - 0.0008:
                    self.passed_endpoint_indices.add(index)
                    crossing_detected = True
                    break
        if crossing_detected:
            self.pass_count += 1
            self.best_clearance_m = max(self.best_clearance_m, clearance)
            self.passed_cleanly = self.ring_contacts == 0
        if self.pass_count and any(axial >= 0.025 for axial, _radial in coordinates):
            self.recovered = True
        self.previous_points = points.copy()
        return {
            "active": True,
            "needle_visible": True,
            "approach_seen": self.approach_seen,
            "center_error_m": round(center_error, 5),
            "minimum_center_error_m": round(self.minimum_center_error_m, 5),
            "inner_radius_m": round(self.inner_radius_m, 5),
            "clearance_m": round(clearance, 5),
            "best_clearance_m": round(self.best_clearance_m, 5),
            "ring_contact": ring_contact,
            "ring_contacts": self.ring_contacts,
            "pass_count": self.pass_count,
            "passed_cleanly": self.passed_cleanly,
            "recovered": self.recovered,
            "needle_coordinates_m": [
                {"axial": round(axial, 5), "radial": round(radial, 5)}
                for axial, radial in coordinates
            ],
            "calibration_status": "training_phantom_engineering_geometry",
        }


@dataclass
class SurgeonsKnotModel:
    """Recognize and score the canonical 2-1-1 surgeon's-knot hand sequence.

    This is an observable dry-lab gesture model.  The OpenUSD/Isaac thread and
    instruments remain the geometry and dynamics authority; this model labels
    wraps, opposing throw direction, seating symmetry and post-seat slippage.
    """

    center: np.ndarray
    expected_wraps: tuple[int, int, int] = (2, 1, 1)
    accumulated_angle_rad: float = 0.0
    previous_angle_rad: float | None = None
    wrapping: bool = False
    throws: list[dict[str, Any]] = field(default_factory=list)
    maximum_seat_separation_m: float = 0.0
    slippage_m: float = 0.0
    seat_symmetry: float = 0.0
    demonstrated_turns: float = 0.0
    demonstrated_completed_throws: int = 0
    expert_manipulation_step: int = 0
    expert_step_progress: float = 0.0
    active_cinch_progress: float = 0.0
    primary_position: np.ndarray | None = None
    secondary_position: np.ndarray | None = None
    loop_side: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float32)

    def reset(self) -> None:
        self.accumulated_angle_rad = 0.0
        self.previous_angle_rad = None
        self.wrapping = False
        self.throws.clear()
        self.maximum_seat_separation_m = 0.0
        self.slippage_m = 0.0
        self.seat_symmetry = 0.0
        self.demonstrated_turns = 0.0
        self.demonstrated_completed_throws = 0
        self.expert_manipulation_step = 0
        self.expert_step_progress = 0.0
        self.active_cinch_progress = 0.0
        self.primary_position = None
        self.secondary_position = None
        self.loop_side = None

    @property
    def phase(self) -> str:
        return (
            "double_first_throw"
            if len(self.throws) == 0
            else "opposing_single_throw"
            if len(self.throws) == 1
            else "final_securing_throw"
            if len(self.throws) == 2
            else "secure_and_inspect"
        )

    def _finalize_throw(self, separation_m: float) -> None:
        if len(self.throws) >= len(self.expected_wraps):
            return
        turns = abs(self.accumulated_angle_rad) / (2.0 * np.pi)
        observed_wraps = max(1, int(round(turns)))
        direction = 1 if self.accumulated_angle_rad >= 0.0 else -1
        expected = self.expected_wraps[len(self.throws)]
        direction_ok = not self.throws or direction == -int(self.throws[-1]["direction"])
        self.throws.append(
            {
                "index": len(self.throws) + 1,
                "wraps": observed_wraps,
                "expected_wraps": expected,
                "direction": direction,
                "direction_label": "clockwise" if direction > 0 else "counter-clockwise",
                "wrap_count_ok": observed_wraps == expected,
                "direction_ok": direction_ok,
            }
        )
        self.maximum_seat_separation_m = max(self.maximum_seat_separation_m, separation_m)
        self.accumulated_angle_rad = 0.0
        self.previous_angle_rad = None
        self.wrapping = False

    def snapshot(
        self,
        tool_positions: dict[int, np.ndarray],
        grippers_open: list[bool],
        hoop_passed: bool,
        thread: Any | None,
        dt: float,
        expert_guidance_active: bool = False,
        expert_manipulation_step: int = 0,
        expert_step_progress: float = 0.0,
        expert_tail_captured: bool = False,
        expert_primary_arm: int | None = None,
    ) -> dict[str, Any]:
        primary_arm = expert_primary_arm if expert_primary_arm in {0, 1} else 0
        secondary_arm = 1 - primary_arm
        primary = tool_positions.get(primary_arm)
        secondary = tool_positions.get(secondary_arm)
        both_closed = bool(
            primary is not None
            and secondary is not None
            and len(grippers_open) >= 2
            and not grippers_open[primary_arm]
            and not grippers_open[secondary_arm]
        )
        separation = float(np.linalg.norm(primary - secondary)) if both_closed else 0.0
        midpoint = (primary + secondary) * 0.5 if both_closed else self.center
        field_error = float(np.linalg.norm(midpoint - self.center))
        if both_closed:
            self.primary_position = np.asarray(primary, dtype=np.float32).copy()
            self.secondary_position = np.asarray(secondary, dtype=np.float32).copy()

        if expert_guidance_active and hoop_passed and expert_tail_captured and both_closed:
            step = int(expert_manipulation_step)
            progress = float(np.clip(expert_step_progress, 0.0, 1.0))
            demonstrated_by_step = {
                2: 2.0 * progress,
                3: 2.0,
                4: 2.0 + progress,
                5: 3.0,
                6: 3.0 + progress,
                7: 4.0,
            }
            self.demonstrated_turns = max(
                self.demonstrated_turns,
                demonstrated_by_step.get(step, 0.0),
            )
            completed_throws = (
                3
                if step >= 8 or (step == 7 and progress >= 0.82)
                else 2
                if step >= 6 or (step == 5 and progress >= 0.82)
                else 1
                if step >= 4 or (step == 3 and progress >= 0.82)
                else 0
            )
            self.demonstrated_completed_throws = max(
                self.demonstrated_completed_throws,
                completed_throws,
            )
            self.expert_manipulation_step = step
            self.expert_step_progress = progress
            self.active_cinch_progress = progress if step in {3, 5, 7} else 0.0
            canonical_directions = (1, -1, 1)
            while len(self.throws) < self.demonstrated_completed_throws:
                throw_index = len(self.throws)
                expected = self.expected_wraps[throw_index]
                direction = canonical_directions[throw_index]
                self.throws.append(
                    {
                        "index": throw_index + 1,
                        "wraps": expected,
                        "expected_wraps": expected,
                        "direction": direction,
                        "direction_label": "clockwise" if direction > 0 else "counter-clockwise",
                        "wrap_count_ok": True,
                        "direction_ok": throw_index == 0 or direction == -canonical_directions[throw_index - 1],
                    }
                )
                self.maximum_seat_separation_m = max(self.maximum_seat_separation_m, separation)

        if hoop_passed and not expert_guidance_active and both_closed and field_error <= 0.10 and len(self.throws) < 3:
            relative = primary - secondary
            angle = float(np.arctan2(relative[1], relative[0]))
            if separation <= 0.058:
                if self.previous_angle_rad is not None:
                    delta = angle - self.previous_angle_rad
                    delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
                    self.accumulated_angle_rad += delta
                self.previous_angle_rad = angle
                self.wrapping = True
            elif self.wrapping and separation >= 0.066:
                minimum_turns = 1.45 if not self.throws else 0.65
                if abs(self.accumulated_angle_rad) >= minimum_turns * 2.0 * np.pi:
                    self._finalize_throw(separation)

        if self.throws and both_closed and separation >= 0.060:
            self.maximum_seat_separation_m = max(self.maximum_seat_separation_m, separation)
            self.slippage_m = max(self.slippage_m, max(0.0, self.maximum_seat_separation_m - separation))
            self.seat_symmetry = max(self.seat_symmetry, float(np.clip(1.0 - field_error / 0.025, 0.0, 1.0)))

        first_throw_double = bool(self.throws and self.throws[0]["wrap_count_ok"])
        directions_alternate = all(item["direction_ok"] for item in self.throws[1:])
        sequence_valid = bool(
            len(self.throws) >= 3
            and first_throw_double
            and all(item["wrap_count_ok"] for item in self.throws)
            and directions_alternate
        )
        tension = float(getattr(thread, "tension_n", 0.0)) if thread is not None else 0.0
        secure = bool(
            hoop_passed
            and sequence_valid
            and self.slippage_m <= 0.008
            and self.seat_symmetry >= 0.55
            and (not expert_guidance_active or expert_tail_captured)
        )
        return {
            "active": True,
            "phase": self.phase,
            "hoop_passed": hoop_passed,
            "expert_guidance_active": expert_guidance_active,
            "demonstration_turns": self.demonstrated_turns,
            "demonstration_completed_throws": self.demonstrated_completed_throws,
            "demonstration_step_progress": round(self.expert_step_progress, 3),
            "throw_count": len(self.throws),
            "target_throws": 3,
            "expected_sequence": "2-1-1",
            "pending_turns": round(abs(self.accumulated_angle_rad) / (2.0 * np.pi), 2),
            "pending_direction": "clockwise" if self.accumulated_angle_rad >= 0.0 else "counter-clockwise",
            "throws": list(self.throws),
            "first_throw_double": first_throw_double,
            "directions_alternate": directions_alternate,
            "sequence_valid": sequence_valid,
            "seat_symmetry": round(self.seat_symmetry, 3),
            "tension_n": round(tension, 4),
            "slippage_m": round(self.slippage_m, 5),
            "secure": secure,
            "calibration_status": "dry_lab_gesture_proxy_pending_clinician_validation",
        }

    def constraint_route(self) -> dict[str, Any]:
        """Build progressive contact targets consumed by the physical strand.

        These points are never rendered directly.  They describe the opposing
        strand contact path while the PBD suture remains the geometry and
        dynamics authority.
        """
        completed_turns = sum(int(item["wraps"]) for item in self.throws)
        active_turns = min(2.0, abs(self.accumulated_angle_rad) / (2.0 * np.pi))
        turns = max(float(completed_turns) + active_turns, self.demonstrated_turns)
        if turns < 0.08 or self.primary_position is None or self.secondary_position is None:
            return {
                "points": np.empty((0, 3), dtype=np.float32),
                "crossings": [],
                "completed_turns": 0,
                "completed_throws": 0,
                "cinch_progress": 0.0,
                "center": self.center.copy(),
            }

        # The expert tools orbit in the horizontal operative plane.  Preserve
        # that plane for the full knot so an active partial wrap cannot rotate
        # edge-on as the hands circle one another.
        if self.loop_side is None:
            radial = self.primary_position - self.secondary_position
            radial = np.asarray((radial[0], radial[1], 0.0), dtype=np.float32)
            radial_length = float(np.linalg.norm(radial))
            self.loop_side = (
                radial / radial_length
                if radial_length >= 1e-8
                else np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
            )
        axis = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
        side = self.loop_side
        up = np.cross(axis, side)
        up /= max(float(np.linalg.norm(up)), 1e-8)

        completed_throws = (
            self.demonstrated_completed_throws
            if self.demonstrated_turns > 0.0
            else len(self.throws)
        )
        route_center = (self.primary_position + self.secondary_position) * 0.5
        current_cinch_throw = (
            (self.expert_manipulation_step - 3) // 2
            if self.expert_manipulation_step in {3, 5, 7}
            else -1
        )
        remaining = min(turns, 4.0)
        route_parts: list[np.ndarray] = []
        crossing_pairs: list[tuple[int, int]] = []
        route_length = 0
        sequence = ((2.0, 1.0), (1.0, -1.0), (1.0, 1.0))
        for throw_index, (available_turns, direction) in enumerate(sequence):
            used_turns = min(remaining, available_turns)
            if used_turns <= 0.0:
                break
            if throw_index < completed_throws:
                seat = 1.0
            elif throw_index == current_cinch_throw:
                seat = self.active_cinch_progress
            else:
                seat = 0.0
            radius = float((1.0 - seat) * 0.0066 + seat * 0.0052)
            samples_per_turn = 20
            samples = max(3, int(np.ceil(used_turns * samples_per_turn)) + 1)
            theta = np.linspace(0.0, direction * used_turns * 2.0 * np.pi, samples, dtype=np.float32)
            normalized_turn = np.abs(theta) / (2.0 * np.pi)
            pitch = (normalized_turn - used_turns * 0.5) * 0.00145
            throw_center = route_center + axis * ((throw_index - 1.0) * 0.0020)
            segment = (
                throw_center[None, :]
                + side[None, :] * (np.cos(theta) * radius)[:, None]
                + up[None, :] * (np.sin(theta) * radius)[:, None]
                + axis[None, :] * pitch[:, None]
            ).astype(np.float32)
            if route_parts:
                previous = route_parts[-1][-1]
                # Keep a stable node-to-crossing map while the instruments
                # move; changing connector resolution would move retained
                # crossings onto unrelated strand particles.
                bridge = np.linspace(previous, segment[0], 4, dtype=np.float32)[1:-1]
                route_parts.append(bridge)
                route_length += len(bridge)
            segment_start = route_length
            route_parts.append(segment)
            for turn_index in range(int(np.floor(used_turns + 1e-4))):
                crossing_pairs.append(
                    (
                        segment_start + turn_index * samples_per_turn,
                        segment_start + (turn_index + 1) * samples_per_turn,
                    )
                )
            route_length += len(segment)
            remaining -= used_turns
        points = np.concatenate(route_parts, axis=0) if route_parts else np.empty((0, 3), dtype=np.float32)
        completed_turn_count = sum(self.expected_wraps[:completed_throws])
        return {
            "points": points,
            "crossings": crossing_pairs,
            "completed_turns": completed_turn_count,
            "completed_throws": completed_throws,
            "cinch_progress": self.active_cinch_progress,
            "center": route_center.astype(np.float32),
        }


@dataclass
class ProcedureMechanics:
    kind: str
    waypoints: np.ndarray
    target_stitches: int = 1
    target_throws: int = 3
    tube: TubeInsertionModel | None = field(init=False, default=None)
    closure: ClosureQualityModel | None = field(init=False, default=None)
    vascular: VascularControlModel | None = field(init=False, default=None)
    ultrasound: UltrasoundAccessModel | None = field(init=False, default=None)
    hoop: HoopThreadingModel | None = field(init=False, default=None)
    surgeons_knot: SurgeonsKnotModel | None = field(init=False, default=None)
    traction_time_s: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.waypoints = np.asarray(self.waypoints, dtype=np.float32).reshape(-1, 3)
        if self.kind == "tube_insertion" and len(self.waypoints) >= 3:
            self.tube = TubeInsertionModel(self.waypoints[1], self.waypoints[-1])
        if self.kind in {"running_suture", "anastomosis", "knot_tying", "threading"}:
            self.closure = ClosureQualityModel(max(1, self.target_stitches), self.kind, max(1, self.target_throws))
        if self.kind in {"clip_divide", "hemostasis"}:
            self.vascular = VascularControlModel(self.kind, self.waypoints)
        if self.kind == "ultrasound_access" and len(self.waypoints):
            target = self.waypoints[-1]
            protected = target + np.asarray((0.018, -0.012, -0.004), dtype=np.float32)
            scan_pose = self.waypoints[0] if len(self.waypoints) else target + (-0.035, -0.020, 0.025)
            self.ultrasound = UltrasoundAccessModel(target, protected, scan_pose=scan_pose)
        if self.kind == "hoop_threading" and len(self.waypoints) >= 3:
            self.hoop = HoopThreadingModel(self.waypoints[0], self.waypoints[1], self.waypoints[2])
            knot_center = self.waypoints[2] + np.asarray((0.035, 0.0, -0.025), dtype=np.float32)
            self.surgeons_knot = SurgeonsKnotModel(knot_center)

    def reset(self) -> None:
        self.traction_time_s = 0.0
        if self.tube:
            self.tube.reset()
        if self.vascular:
            self.vascular.reset()
        if self.ultrasound:
            self.ultrasound.reset()
        if self.hoop:
            self.hoop.reset()
        if self.surgeons_knot:
            self.surgeons_knot.reset()
        if self.closure:
            self.closure.reset()

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
        needle_points: np.ndarray | None = None,
        expert_guidance_active: bool = False,
        expert_manipulation_step: int = 0,
        expert_step_progress: float = 0.0,
        expert_tail_captured: bool = False,
        expert_primary_arm: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        primary = tool_positions.get(0)
        secondary = tool_positions.get(1)
        closed = sum(not value for value in grippers_open)
        result: dict[str, dict[str, Any]] = {}
        if self.tube:
            self.tube.update(primary, assisted_grasped, dt)
            result["tube"] = self.tube.snapshot()
        if self.closure:
            result["closure"] = self.closure.snapshot(thread, tool_positions, grippers_open, dt)
        if self.vascular:
            result["vascular"] = self.vascular.snapshot(tool_positions, grippers_open, dt)
        if self.ultrasound:
            result["ultrasound"] = self.ultrasound.snapshot(primary, secondary, self.waypoints, dt)
        if self.hoop:
            result["hoop"] = self.hoop.snapshot(needle_points)
        if self.surgeons_knot:
            result["surgeons_knot"] = self.surgeons_knot.snapshot(
                tool_positions,
                grippers_open,
                bool(self.hoop and self.hoop.pass_count >= 1),
                thread,
                dt,
                expert_guidance_active,
                expert_manipulation_step,
                expert_step_progress,
                expert_tail_captured,
                expert_primary_arm,
            )
        if self.kind in {"dissection", "biopsy"}:
            faces = int(cut.get("faces_removed", 0))
            cut_length = float(cut.get("length_m", 0.0))
            target_length = max(0.035, sum(float(np.linalg.norm(end - start)) for start, end in zip(self.waypoints[:-1], self.waypoints[1:])))
            topology_progress = float(np.clip(cut_length / target_length, 0.0, 1.0))
            protected_center = np.mean(self.waypoints, axis=0) + np.asarray((0.0, 0.028, -0.004), dtype=np.float32)
            clearances = [float(np.linalg.norm(position - protected_center) - 0.014) for position in tool_positions.values()]
            protected_clearance = min(clearances, default=1.0)
            protected_contact = protected_clearance <= 0.0
            path_progress = float(np.clip(waypoint_count / max(1, len(self.waypoints)), 0.0, 1.0))
            field_center = np.mean(self.waypoints, axis=0)
            traction_active = bool(
                secondary is not None
                and len(grippers_open) >= 2
                and not grippers_open[1]
                and float(np.linalg.norm(secondary - field_center)) <= 0.085
            )
            self.traction_time_s = self.traction_time_s + max(0.0, float(dt)) if traction_active else max(0.0, self.traction_time_s - max(0.0, float(dt)))
            exposure_factor = 1.0 if self.traction_time_s >= 0.25 else 0.55
            raw_progress = min(path_progress, topology_progress) if self.kind == "biopsy" else 0.25 * path_progress + 0.75 * topology_progress
            progress = raw_progress * exposure_factor
            result["dissection"] = {
                "active": True,
                "mode": self.kind,
                "plane_progress": round(progress, 3),
                "faces_separated": faces,
                "protected_contact": protected_contact,
                "protected_clearance_m": round(protected_clearance, 5),
                "traction_active": traction_active,
                "traction_time_s": round(self.traction_time_s, 2),
                "margin_consistency_percent": round(100.0 * progress * (1.0 if not protected_contact else 0.55), 1),
                "specimen_released": bool(self.kind == "biopsy" and progress >= 0.85 and faces > 8 and not protected_contact),
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
