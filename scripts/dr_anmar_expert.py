# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Closed-loop expert demonstrations for Dr.Anmar procedure rooms.

The controller emits the same relative-IK action vectors and gripper states as
the browser controls.  It deliberately remains above Isaac Lab: Isaac owns
robot dynamics, contacts, collisions and task state, while this module owns the
observable teaching sequence and its bounded procedural targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np


EXPERT_CONTROLLER_VERSION = "dr-anmar-expert-v4-normalized-relative-ik"
EXPERT_PHASES: tuple[dict[str, str], ...] = (
    {"id": "rest", "title": "Rest", "instruction": "Confirm the neutral pose, anatomy and operative camera."},
    {"id": "approach", "title": "Approach", "instruction": "Move above the first target with the jaws open."},
    {"id": "align", "title": "Align", "instruction": "Set depth, angle and working-axis alignment before contact."},
    {"id": "contact", "title": "Contact", "instruction": "Enter the interaction zone slowly and preserve the safety envelope."},
    {"id": "grasp", "title": "Grasp", "instruction": "Close deliberately and verify that the intended object or tissue follows."},
    {"id": "manipulate", "title": "Manipulate", "instruction": "Execute the room-specific path, handoff or tissue interaction."},
    {"id": "verify", "title": "Verify", "instruction": "Hold still while task, force and tissue evidence are inspected."},
    {"id": "recover", "title": "Recover", "instruction": "Withdraw to a stable pose without undoing the completed work."},
)
EXPERT_PHASE_IDS = tuple(phase["id"] for phase in EXPERT_PHASES)


@dataclass
class ExpertCommand:
    action: np.ndarray
    grippers_open: list[bool]
    phase_changed: bool = False
    completed: bool = False


@dataclass
class ExpertDemonstrationController:
    procedure_id: str
    guide_kind: str
    action_dim: int
    arms: int
    has_grippers: bool
    waypoints: np.ndarray
    status: str = "idle"
    phase_index: int = -1
    phase_ticks: int = 0
    target_ticks: int = 0
    target_index: int = 0
    manipulation_step: int = 0
    completed_phases: list[str] = field(default_factory=list)
    degraded_reasons: list[str] = field(default_factory=list)
    takeover_phase: str | None = None
    paused_reason: str | None = None
    phase_anchor_tools: dict[int, np.ndarray] = field(default_factory=dict)
    object_anchor: np.ndarray | None = None
    phase_started_at: float = field(default_factory=time.monotonic)
    paused_at: float | None = None
    primary_arm: int | None = None
    tail_captured: bool = False
    needle_point_index: int | None = None
    ring_handoff_step: int = 0
    ring_handoff_anchor: np.ndarray | None = None
    ring_handoff_complete: bool = False

    def __post_init__(self) -> None:
        self.waypoints = np.asarray(self.waypoints, dtype=np.float32).reshape(-1, 3)
        self.group_width = 7 if self.has_grippers else 6

    @property
    def phase(self) -> str | None:
        if 0 <= self.phase_index < len(EXPERT_PHASE_IDS):
            return EXPERT_PHASE_IDS[self.phase_index]
        return None

    @property
    def active(self) -> bool:
        return self.status in {"running", "paused"}

    @property
    def phase_elapsed_s(self) -> float:
        end = self.paused_at if self.paused_at is not None else time.monotonic()
        return max(0.0, end - self.phase_started_at)

    def start(self) -> None:
        self.status = "running"
        self.phase_index = 0
        self.phase_ticks = 0
        self.target_ticks = 0
        self.target_index = 0
        self.manipulation_step = 0
        self.completed_phases.clear()
        self.degraded_reasons.clear()
        self.takeover_phase = None
        self.paused_reason = None
        self.phase_started_at = time.monotonic()
        self.paused_at = None
        self.phase_anchor_tools.clear()
        self.object_anchor = None
        self.primary_arm = None
        self.tail_captured = False
        self.needle_point_index = None
        self.ring_handoff_step = 0
        self.ring_handoff_anchor = None
        self.ring_handoff_complete = False

    def pause(self, reason: str = "Doctor paused the expert demonstration for inspection.") -> None:
        if self.status == "running":
            self.status = "paused"
            self.paused_reason = reason
            self.paused_at = time.monotonic()

    def resume(self) -> None:
        if self.status == "paused":
            if self.paused_at is not None:
                self.phase_started_at += time.monotonic() - self.paused_at
            self.status = "running"
            self.paused_reason = None
            self.paused_at = None

    def take_over(self) -> None:
        if self.active:
            self.takeover_phase = self.phase
            self.status = "taken_over"
            self.paused_reason = None
            self.paused_at = None

    def cancel(self) -> None:
        if self.active:
            self.status = "cancelled"

    def snapshot(self, reference_demo: str | None = None) -> dict[str, Any]:
        phases = []
        for index, definition in enumerate(EXPERT_PHASES):
            phase_id = definition["id"]
            phase_status = (
                "complete"
                if phase_id in self.completed_phases
                else "active"
                if index == self.phase_index and self.active
                else "pending"
            )
            phases.append({**definition, "status": phase_status})
        return {
            "controller": EXPERT_CONTROLLER_VERSION,
            "available": bool(self.procedure_id),
            "status": self.status,
            "phase": self.phase,
            "phase_index": self.phase_index,
            "phase_ticks": self.phase_ticks,
            "manipulation_step": self.manipulation_step,
            "primary_arm": self.primary_arm,
            "tail_captured": self.tail_captured,
            "needle_point_index": self.needle_point_index,
            "ring_handoff_complete": self.ring_handoff_complete,
            "procedure_instruction": self._procedure_instruction(),
            "phase_elapsed_s": round(self.phase_elapsed_s, 2) if self.phase is not None else 0.0,
            "phases": phases,
            "completed_phases": list(self.completed_phases),
            "progress_percent": round(100 * len(self.completed_phases) / len(EXPERT_PHASES)),
            "paused_reason": self.paused_reason,
            "takeover_phase": self.takeover_phase,
            "clean_reference_eligible": self.status == "completed" and not self.degraded_reasons,
            "degraded_reasons": list(self.degraded_reasons),
            "reference_demo": reference_demo,
            "recording_contract": "actions + robot state + cameras + mechanics + eight phase labels",
        }

    def _procedure_instruction(self) -> str | None:
        if self.guide_kind != "hoop_threading" or self.phase != "manipulate":
            return None
        if self.manipulation_step == 1 and not self.ring_handoff_complete:
            return "The tip has crossed: the far-side instrument now grasps the needle, the first releases, and the receiver pulls the threaded needle clear."
        return {
            0: "Lead with the needle tip: approach, center it in the hoop, rotate the curved needle through, then clear the far side.",
            1: "Bring both instruments into the knot field and control one strand end with each gripper.",
            2: "First throw: rotate through two complete wraps in the same direction around the receiving instrument.",
            3: "Capture the tail and pull both ends apart evenly to seat the double first throw.",
            4: "Second throw: reverse the crossing direction and form one complete opposing wrap.",
            5: "Seat the opposing throw without releasing the tension retained by the first throw.",
            6: "Final throw: return to the original crossing direction and form one securing wrap.",
            7: "Draw both ends down symmetrically, then hold still while slippage and knot security are inspected.",
        }.get(self.manipulation_step)

    def _action(self) -> np.ndarray:
        return np.zeros(self.action_dim, dtype=np.float32)

    def _set_motion(
        self,
        action: np.ndarray,
        arm: int,
        current: np.ndarray | None,
        target: np.ndarray | None,
        rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> float | None:
        if arm >= self.arms or current is None or target is None:
            return None
        delta = np.asarray(target, dtype=np.float32) - np.asarray(current, dtype=np.float32)
        start = arm * self.group_width
        action[start : start + 3] = np.clip(delta * (0.82 / 0.01), -1.0, 1.0)
        action[start + 3 : start + 6] = np.clip(
            np.asarray(rotation, dtype=np.float32) / 0.05, -1.0, 1.0
        )
        return float(np.linalg.norm(delta))

    def _phase_target(self, object_position: np.ndarray | None) -> np.ndarray | None:
        if object_position is not None and self.guide_kind in {
            "pickup", "handover", "retraction", "reposition", "needle_pass", "hoop_threading", "tube_insertion", "recovery"
        }:
            return np.asarray(object_position, dtype=np.float32)
        if len(self.waypoints):
            return self.waypoints[0]
        return np.asarray(object_position, dtype=np.float32) if object_position is not None else None

    def _reached_or_timeout(self, distance: float | None, tolerance: float, timeout: int, label: str) -> bool:
        if distance is not None and distance <= tolerance:
            return True
        if self.phase_ticks >= timeout:
            if distance is None:
                reason = f"{label}: target pose unavailable"
            else:
                reason = f"{label}: bounded target timeout at {distance * 1000:.1f} mm"
            if reason not in self.degraded_reasons:
                self.degraded_reasons.append(reason)
            return True
        return False

    def _advance(self, tool_positions: dict[int, np.ndarray]) -> bool:
        current = self.phase
        if current and current not in self.completed_phases:
            self.completed_phases.append(current)
        self.phase_index += 1
        self.phase_ticks = 0
        self.target_ticks = 0
        self.target_index = 0
        self.manipulation_step = 0
        self.phase_anchor_tools = {arm: value.copy() for arm, value in tool_positions.items()}
        self.phase_started_at = time.monotonic()
        if self.phase_index >= len(EXPERT_PHASES):
            self.phase_index = len(EXPERT_PHASES) - 1
            self.status = "completed"
            return True
        return False

    def _follow_waypoints(
        self,
        action: np.ndarray,
        tool_positions: dict[int, np.ndarray],
        arm: int,
        targets: np.ndarray,
        rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> bool:
        if not len(targets):
            return True
        index = min(self.target_index, len(targets) - 1)
        distance = self._set_motion(action, arm, tool_positions.get(arm), targets[index], rotation)
        self.target_ticks += 1
        if (distance is not None and distance <= 0.011) or self.target_ticks >= 80:
            if self.target_ticks >= 80 and (distance is None or distance > 0.018):
                self.degraded_reasons.append(f"manipulate waypoint {index + 1}: bounded convergence timeout")
            self.target_index += 1
            self.target_ticks = 0
        return self.target_index >= len(targets)

    def _follow_needle_tip_through_hoop(
        self,
        action: np.ndarray,
        tools: dict[int, np.ndarray],
        arm: int,
        needle_points: np.ndarray | None,
    ) -> bool:
        """Drive the rigid needle's sharp endpoint through the hoop opening.

        Relative-IK still moves the instrument, but the closed-loop error is
        measured at the needle tip.  This keeps the jaws and shaft outside the
        ring instead of incorrectly steering the tool origin through it.
        """
        points = (
            np.asarray(needle_points, dtype=np.float32).reshape(-1, 3)
            if needle_points is not None
            else np.empty((0, 3), dtype=np.float32)
        )
        if not len(points) or not len(self.waypoints):
            if self.target_ticks >= 40:
                self.pause("Needle pose is unavailable. Reset the room before continuing the expert pass.")
            self.target_ticks += 1
            return False
        if self.target_index >= len(self.waypoints):
            return True
        if self.needle_point_index is None or self.needle_point_index >= len(points):
            self.needle_point_index = int(
                np.argmin(np.linalg.norm(points - self.waypoints[0][None, :], axis=1))
            )
        index = min(self.target_index, len(self.waypoints) - 1)
        needle_tip = points[self.needle_point_index]
        normal = self.waypoints[-1] - self.waypoints[0]
        normal /= max(float(np.linalg.norm(normal)), 1e-8)
        target = self.waypoints[index].copy()
        rotation = (0.0, 0.0, 0.0)
        path_ready = True
        if index == 1:
            # Stop just before the metal ring.  The next stage is a deliberate
            # wrist-driven needle arc, never a straight tool translation.
            target = self.waypoints[1] - normal * 0.006
            rotation = (0.010, 0.0, 0.0)
        elif index == 2:
            arc_progress = float(np.clip(self.target_ticks / 96.0, 0.0, 1.0))
            smooth = arc_progress * arc_progress * (3.0 - 2.0 * arc_progress)
            target = (
                self.waypoints[1]
                + normal * (-0.006 + 0.018 * smooth)
                + np.asarray((0.0, 0.0, 0.0045 * np.sin(np.pi * smooth)), dtype=np.float32)
            )
            rotation = (0.030, 0.0, 0.0)
            path_ready = arc_progress >= 1.0
        distance = self._set_motion(
            action,
            arm,
            needle_tip,
            target,
            rotation,
        )
        self.target_ticks += 1
        if path_ready and distance is not None and distance <= 0.0075:
            self.target_index += 1
            self.target_ticks = 0
        elif self.target_ticks >= 180:
            millimeters = distance * 1000.0 if distance is not None else float("nan")
            self.pause(
                f"Needle tip could not reach ring waypoint {index + 1} "
                f"({millimeters:.1f} mm remaining). Reset and retry rather than skipping the pass."
            )
        return self.target_index >= len(self.waypoints)

    def _threaded_ring_handoff(
        self,
        action: np.ndarray,
        tools: dict[int, np.ndarray],
        object_position: np.ndarray | None,
        grippers: list[bool],
        primary_arm: int,
        native_grasp_contact_active: list[bool],
    ) -> bool:
        """Transfer the passed needle to the far-side gripper and withdraw it."""
        if self.arms < 2 or object_position is None or len(self.waypoints) < 2:
            self.pause("The physical needle handoff pose is unavailable. Reset the room and retry.")
            return False
        receiver_arm = 1 - primary_arm
        normal = self.waypoints[-1] - self.waypoints[0]
        normal /= max(float(np.linalg.norm(normal)), 1e-8)
        self.target_ticks += 1
        if self.ring_handoff_step == 0:
            grippers[primary_arm] = False
            grippers[receiver_arm] = True
            receiver_target = np.asarray(object_position, dtype=np.float32) + normal * 0.009
            distance = self._set_motion(
                action,
                receiver_arm,
                tools.get(receiver_arm),
                receiver_target,
            )
            if distance is not None and distance <= 0.012:
                grippers[receiver_arm] = False
                self.ring_handoff_step = 1
                self.target_ticks = 0
            elif self.target_ticks >= 180:
                self.pause("The far-side instrument could not reach the passed needle. Reset and retry the handoff.")
            return False

        if self.ring_handoff_step == 1:
            # Both jaws remain closed until the worker confirms that the
            # receiver owns a real fixed grasp joint. Never drop the needle
            # between two merely visual gripper poses.
            grippers[primary_arm] = False
            grippers[receiver_arm] = False
            receiver_has_custody = (
                receiver_arm < len(native_grasp_contact_active)
                and native_grasp_contact_active[receiver_arm]
            )
            if receiver_has_custody:
                grippers[primary_arm] = True
                self.ring_handoff_anchor = (
                    tools[receiver_arm].copy()
                    if receiver_arm in tools
                    else np.asarray(object_position, dtype=np.float32).copy()
                )
                self.ring_handoff_step = 2
                self.target_ticks = 0
            elif self.target_ticks >= 80:
                self.pause("The receiver closed but did not acquire physical needle custody. The original holder remains closed.")
            return False

        grippers[primary_arm] = True
        grippers[receiver_arm] = False
        receiver_has_custody = (
            receiver_arm < len(native_grasp_contact_active)
            and native_grasp_contact_active[receiver_arm]
        )
        if not receiver_has_custody:
            self.pause("The receiving instrument lost physical needle custody during withdrawal.")
            return False
        withdrawal_target = (
            self.ring_handoff_anchor + normal * 0.038
            if self.ring_handoff_anchor is not None
            else None
        )
        distance = self._set_motion(
            action,
            receiver_arm,
            tools.get(receiver_arm),
            withdrawal_target,
        )
        if distance is not None and distance <= 0.013:
            self.primary_arm = receiver_arm
            self.ring_handoff_complete = True
            self.target_ticks = 0
            return True
        if self.target_ticks >= 180:
            self.pause("The receiving instrument could not withdraw the threaded needle clear of the ring.")
        return False

    def _handover(
        self,
        action: np.ndarray,
        tools: dict[int, np.ndarray],
        object_position: np.ndarray | None,
        grippers: list[bool],
    ) -> bool:
        if self.arms < 2:
            self.degraded_reasons.append("handover requires two instruments")
            return True
        primary_anchor = self.phase_anchor_tools.get(0, tools.get(0))
        secondary_anchor = self.phase_anchor_tools.get(1, tools.get(1))
        if self.manipulation_step == 0:
            target = primary_anchor + np.asarray((0.0, 0.0, 0.045), dtype=np.float32) if primary_anchor is not None else None
            distance = self._set_motion(action, 0, tools.get(0), target)
            if self._reached_or_timeout(distance, 0.012, 80, "handover presentation"):
                self.manipulation_step = 1
                self.phase_ticks = 0
        elif self.manipulation_step == 1:
            target = np.asarray(object_position, dtype=np.float32) + (0.0, 0.0, 0.004) if object_position is not None else None
            distance = self._set_motion(action, 1, tools.get(1), target)
            if self._reached_or_timeout(distance, 0.013, 80, "handover receiver approach"):
                grippers[1] = False
                self.manipulation_step = 2
                self.phase_ticks = 0
        elif self.manipulation_step == 2:
            grippers[1] = False
            if self.phase_ticks >= 4:
                grippers[0] = True
                self.manipulation_step = 3
                self.phase_ticks = 0
        else:
            grippers[0] = True
            grippers[1] = False
            target = secondary_anchor + np.asarray((0.040, 0.015, 0.045), dtype=np.float32) if secondary_anchor is not None else None
            distance = self._set_motion(action, 1, tools.get(1), target)
            return self._reached_or_timeout(distance, 0.014, 80, "handover receiver recovery")
        return False

    def _clip_divide(self, action: np.ndarray, tools: dict[int, np.ndarray], grippers: list[bool]) -> bool:
        if not len(self.waypoints):
            return True
        center = np.mean(self.waypoints, axis=0)
        if self.arms > 1:
            self._set_motion(action, 1, tools.get(1), center + np.asarray((0.0, 0.020, 0.012), dtype=np.float32))
            grippers[1] = False
        index = min(self.target_index, len(self.waypoints) - 1)
        distance = self._set_motion(action, 0, tools.get(0), self.waypoints[index])
        is_clip_site = index in {1, 3}
        if is_clip_site and self.manipulation_step == 0:
            grippers[0] = True
        if (distance is not None and distance <= 0.012) or self.target_ticks >= 80:
            if is_clip_site and self.manipulation_step == 0:
                grippers[0] = False
                self.manipulation_step = 1
                self.target_ticks = 0
            elif is_clip_site and self.manipulation_step == 1 and self.target_ticks >= 3:
                grippers[0] = True
                self.manipulation_step = 0
                self.target_index += 1
                self.target_ticks = 0
            elif not is_clip_site:
                self.target_index += 1
                self.target_ticks = 0
        self.target_ticks += 1
        return self.target_index >= len(self.waypoints)

    def _knot(self, action: np.ndarray, tools: dict[int, np.ndarray], grippers: list[bool]) -> bool:
        if self.arms < 2:
            return self._follow_waypoints(action, tools, 0, self.waypoints, (0.025, 0.0, 0.0))
        center = np.mean(self.waypoints, axis=0) if len(self.waypoints) else np.zeros(3, dtype=np.float32)
        side = 1.0 if self.manipulation_step % 2 == 0 else -1.0
        target0 = center + np.asarray((0.022 * side, -0.008, 0.018), dtype=np.float32)
        target1 = center + np.asarray((-0.022 * side, 0.008, 0.018), dtype=np.float32)
        d0 = self._set_motion(action, 0, tools.get(0), target0, (0.018 * side, 0.0, 0.0))
        d1 = self._set_motion(action, 1, tools.get(1), target1, (-0.018 * side, 0.0, 0.0))
        grippers[0] = grippers[1] = False
        self.target_ticks += 1
        if ((d0 is not None and d0 <= 0.014) and (d1 is not None and d1 <= 0.014)) or self.target_ticks >= 60:
            self.manipulation_step += 1
            self.target_ticks = 0
        return self.manipulation_step >= 6

    def _surgeons_knot(
        self,
        action: np.ndarray,
        tools: dict[int, np.ndarray],
        grippers: list[bool],
        thread_tail_position: np.ndarray | None,
        knot_secure: bool,
    ) -> bool:
        """Demonstrate the canonical 2-1-1 bimanual surgeon's-knot sequence."""
        if self.arms < 2:
            if "surgeon's knot requires two instruments" not in self.degraded_reasons:
                self.degraded_reasons.append("surgeon's knot requires two instruments")
            return True
        center = (
            self.waypoints[-1] + np.asarray((0.035, 0.0, -0.025), dtype=np.float32)
            if len(self.waypoints)
            else np.asarray((0.080, 0.0, 0.040), dtype=np.float32)
        )
        grippers[0] = grippers[1] = False
        primary_arm = self.primary_arm if self.primary_arm in {0, 1} else 0
        receiver_arm = 1 - primary_arm
        step = self.manipulation_step
        self.target_ticks += 1

        if step == 1:
            target0 = center + np.asarray((0.024, 0.0, 0.010), dtype=np.float32)
            if not self.tail_captured and thread_tail_position is None:
                if self.target_ticks >= 40:
                    self.pause("The physical suture tail is not visible. Reset the room before attempting the knot.")
                return False
            if not self.tail_captured and thread_tail_position is not None:
                tail_target = np.asarray(thread_tail_position, dtype=np.float32) + np.asarray(
                    (0.0, 0.0, 0.010), dtype=np.float32
                )
                self._set_motion(action, primary_arm, tools.get(primary_arm), target0)
                tail_distance = self._set_motion(
                    action,
                    receiver_arm,
                    tools.get(receiver_arm),
                    tail_target,
                )
                grippers[receiver_arm] = True
                if tail_distance is not None and tail_distance <= 0.014:
                    grippers[receiver_arm] = False
                    self.tail_captured = True
                    self.target_ticks = 0
                elif self.target_ticks >= 180:
                    grippers[receiver_arm] = True
                    self.pause("The receiving instrument missed the free suture tail. Reset and retry the handoff.")
                return False
            target1 = center + np.asarray((-0.024, 0.0, 0.010), dtype=np.float32)
            d0 = self._set_motion(action, primary_arm, tools.get(primary_arm), target0)
            d1 = self._set_motion(action, receiver_arm, tools.get(receiver_arm), target1)
            complete = bool(d0 is not None and d1 is not None and d0 <= 0.012 and d1 <= 0.012)
            if complete:
                self.manipulation_step = 2
                self.target_ticks = 0
            elif self.target_ticks >= 140:
                self.pause("Both strand ends could not be positioned in the knot field. Reset and retry.")
            return False

        wrap_specs = {
            2: (2.0, 1.0, 118),
            4: (1.0, -1.0, 76),
            6: (1.0, 1.0, 76),
        }
        if step in wrap_specs:
            turns, direction, duration = wrap_specs[step]
            progress = float(np.clip(self.target_ticks / max(duration, 1), 0.0, 1.0))
            angle = direction * turns * 2.0 * np.pi * progress
            radial = np.asarray((0.024 * np.cos(angle), 0.024 * np.sin(angle), 0.010), dtype=np.float32)
            self._set_motion(action, primary_arm, tools.get(primary_arm), center + radial, (0.0, 0.0, 0.018 * direction))
            self._set_motion(action, receiver_arm, tools.get(receiver_arm), center - radial + np.asarray((0.0, 0.0, 0.020), dtype=np.float32), (0.0, 0.0, -0.018 * direction))
            if self.target_ticks >= duration:
                self.manipulation_step += 1
                self.target_ticks = 0
            return False

        if step in {3, 5, 7}:
            target0 = center + np.asarray((0.043, 0.0, 0.010), dtype=np.float32)
            target1 = center + np.asarray((-0.043, 0.0, 0.010), dtype=np.float32)
            d0 = self._set_motion(action, primary_arm, tools.get(primary_arm), target0)
            d1 = self._set_motion(action, receiver_arm, tools.get(receiver_arm), target1)
            complete = bool(d0 is not None and d1 is not None and d0 <= 0.010 and d1 <= 0.010)
            if complete:
                self.manipulation_step += 1
                self.target_ticks = 0
            elif self.target_ticks >= 120:
                self.pause(f"Knot throw {(step - 1) // 2} did not seat symmetrically. Reset and retry.")
            return False
        if step >= 8:
            if knot_secure:
                return True
            if self.target_ticks >= 80:
                self.pause("The 2-1-1 knot did not satisfy the physical security checks. Inspect or reset the room.")
            return False
        return False

    def _ultrasound(self, action: np.ndarray, tools: dict[int, np.ndarray], grippers: list[bool]) -> bool:
        if self.arms < 2 or not len(self.waypoints):
            return self._follow_waypoints(action, tools, 0, self.waypoints)
        self._set_motion(action, 0, tools.get(0), self.waypoints[0])
        grippers[0] = False
        needle_path = self.waypoints[1:] if len(self.waypoints) > 1 else self.waypoints
        if self.manipulation_step == 0:
            complete = self._follow_waypoints(action, tools, 1, needle_path, (0.020, 0.0, 0.0))
            grippers[1] = False
            if complete:
                self.manipulation_step = 1
                self.target_index = max(0, len(needle_path) - 2)
                self.target_ticks = 0
        else:
            withdrawal = needle_path[: max(1, self.target_index + 1)][::-1]
            return self._follow_waypoints(action, tools, 1, withdrawal, (-0.015, 0.0, 0.0))
        return False

    def _manipulate(
        self,
        action: np.ndarray,
        tools: dict[int, np.ndarray],
        object_position: np.ndarray | None,
        grippers: list[bool],
        thread_tail_position: np.ndarray | None = None,
        needle_points: np.ndarray | None = None,
        hoop_passed: bool = False,
        knot_secure: bool = False,
        native_grasp_contact_active: list[bool] | None = None,
    ) -> bool:
        kind = self.guide_kind
        if kind == "pickup":
            anchor = self.phase_anchor_tools.get(0, tools.get(0))
            target = anchor + np.asarray((0.0, 0.0, 0.040), dtype=np.float32) if anchor is not None else None
            distance = self._set_motion(action, 0, tools.get(0), target)
            object_lift = (
                float(object_position[2] - self.object_anchor[2])
                if object_position is not None and self.object_anchor is not None
                else 0.0
            )
            if object_lift >= 0.025:
                return True
            return self._reached_or_timeout(distance, 0.014, 120, "pickup manipulation")
        if kind == "handover":
            return self._handover(action, tools, object_position, grippers)
        if kind == "needle_pass":
            if self.target_index < min(2, len(self.waypoints)):
                return self._follow_waypoints(action, tools, 0, self.waypoints[:2], (0.020, 0.0, 0.0)) and self._handover(
                    action, tools, object_position, grippers
                )
            return self._handover(action, tools, object_position, grippers)
        if kind == "hoop_threading":
            primary_arm = self.primary_arm if self.primary_arm in range(self.arms) else 0
            physical_grasps = native_grasp_contact_active or [False] * self.arms
            if (
                self.manipulation_step == 0
                and (primary_arm >= len(physical_grasps) or not physical_grasps[primary_arm])
            ):
                self.pause("The needle is not physically secured in the primary gripper. Reset and grasp it before the ring pass.")
                return False
            if self.manipulation_step == 0:
                grippers[primary_arm] = False
                path_complete = self._follow_needle_tip_through_hoop(
                    action,
                    tools,
                    primary_arm,
                    needle_points,
                )
                if path_complete and hoop_passed:
                    self.manipulation_step = 1
                    self.target_ticks = 0
                    self.target_index = 0
                elif path_complete:
                    self.target_ticks += 1
                    if self.target_ticks >= 80:
                        self.pause("The needle cleared the path but no physical hoop crossing was detected. Reset and retry.")
                return False
            if not self.ring_handoff_complete:
                if not self._threaded_ring_handoff(
                    action,
                    tools,
                    object_position,
                    grippers,
                    primary_arm,
                    physical_grasps,
                ):
                    return False
                primary_arm = self.primary_arm if self.primary_arm in range(self.arms) else primary_arm
            self.pause(
                "The native PhysX deformable-thread phase is not available. "
                "Projected knot routes and scripted knot completion are disabled."
            )
            return False
        if kind == "clip_divide":
            return self._clip_divide(action, tools, grippers)
        if kind == "hemostasis":
            target = self.waypoints[min(2, len(self.waypoints) - 1)] if len(self.waypoints) else None
            distance = self._set_motion(action, 0, tools.get(0), target)
            grippers[0] = self.phase_ticks < 20
            return self.phase_ticks >= 48 or (distance is not None and distance <= 0.012 and self.phase_ticks >= 32)
        if kind == "ultrasound_access":
            return self._ultrasound(action, tools, grippers)
        if kind == "knot_tying":
            return self._knot(action, tools, grippers)
        if kind in {"dissection", "biopsy"} and self.arms > 1 and len(self.waypoints):
            center = np.mean(self.waypoints, axis=0)
            self._set_motion(action, 1, tools.get(1), center + np.asarray((0.0, 0.035, 0.016), dtype=np.float32))
            grippers[1] = False
            grippers[0] = False
            return self._follow_waypoints(action, tools, 0, self.waypoints)
        if len(self.waypoints):
            rotation = (0.024, 0.0, 0.0) if kind in {"threading", "running_suture", "anastomosis", "needle_pass", "hoop_threading"} else (0.0, 0.0, 0.0)
            return self._follow_waypoints(action, tools, 0, self.waypoints, rotation)
        anchor = self.phase_anchor_tools.get(0, tools.get(0))
        offsets = {
            "pickup": (0.0, 0.0, 0.060),
            "retraction": (0.040, 0.015, 0.060),
            "reposition": (0.050, 0.020, 0.045),
        }
        target = anchor + np.asarray(offsets.get(kind, (0.035, 0.0, 0.035)), dtype=np.float32) if anchor is not None else None
        distance = self._set_motion(action, 0, tools.get(0), target)
        return self._reached_or_timeout(distance, 0.013, 120, f"{kind} manipulation")

    def step(
        self,
        tool_positions: dict[int, np.ndarray],
        object_position: np.ndarray | None,
        grippers_open: list[bool],
        safety_envelope_active: bool = False,
        thread_tail_position: np.ndarray | None = None,
        needle_points: np.ndarray | None = None,
        hoop_passed: bool = False,
        knot_secure: bool = False,
        native_grasp_contact_active: list[bool] | None = None,
    ) -> ExpertCommand:
        action = self._action()
        grippers = list(grippers_open)
        if self.status != "running":
            return ExpertCommand(action, grippers)
        if safety_envelope_active:
            self.pause("Research safety envelope reached. Inspect forces before resuming or take control.")
            return ExpertCommand(action, grippers)
        if not self.phase_anchor_tools:
            self.phase_anchor_tools = {arm: value.copy() for arm, value in tool_positions.items()}
        if self.object_anchor is None and object_position is not None:
            self.object_anchor = np.asarray(object_position, dtype=np.float32).copy()
        if self.guide_kind == "hoop_threading" and self.primary_arm is None and object_position is not None:
            candidates = [
                (float(np.linalg.norm(position - object_position)), arm)
                for arm, position in tool_positions.items()
            ]
            self.primary_arm = min(candidates)[1] if candidates else 0
        self.phase_ticks += 1
        phase = self.phase
        phase_changed = False
        completed = False
        primary_arm = self.primary_arm if self.guide_kind == "hoop_threading" and self.primary_arm is not None else 0
        primary = tool_positions.get(primary_arm)
        base = self._phase_target(object_position)
        if phase == "rest":
            if self.has_grippers:
                grippers = [True] * self.arms
            if self.phase_ticks >= 12 and self.phase_elapsed_s >= 1.0:
                completed = self._advance(tool_positions)
                phase_changed = True
        elif phase == "approach":
            target = base + np.asarray((0.0, 0.0, 0.045), dtype=np.float32) if base is not None else None
            distance = self._set_motion(action, primary_arm, primary, target)
            if self._reached_or_timeout(distance, 0.014, 120, "approach") and self.phase_elapsed_s >= 0.9:
                completed = self._advance(tool_positions)
                phase_changed = True
        elif phase == "align":
            target = base + np.asarray((0.0, 0.0, 0.014), dtype=np.float32) if base is not None else None
            distance = self._set_motion(action, primary_arm, primary, target, (0.0, 0.012, 0.0))
            if self._reached_or_timeout(distance, 0.010, 100, "align") and self.phase_elapsed_s >= 0.9:
                completed = self._advance(tool_positions)
                phase_changed = True
        elif phase == "contact":
            distance = self._set_motion(action, primary_arm, primary, base)
            if self._reached_or_timeout(distance, 0.009, 100, "contact") and self.phase_elapsed_s >= 0.8:
                completed = self._advance(tool_positions)
                phase_changed = True
        elif phase == "grasp":
            if self.has_grippers:
                grippers[primary_arm] = False
            physical_grasps = native_grasp_contact_active or [False] * self.arms
            grasp_confirmed = (
                primary_arm < len(physical_grasps)
                and physical_grasps[primary_arm]
            )
            if (
                (self.guide_kind != "hoop_threading" or grasp_confirmed)
                and self.phase_ticks >= 12
                and self.phase_elapsed_s >= 0.9
            ):
                completed = self._advance(tool_positions)
                phase_changed = True
            elif self.guide_kind == "hoop_threading" and self.phase_ticks >= 120:
                self.pause("The primary instrument closed but did not acquire physical needle custody. Reset and retry the grasp.")
        elif phase == "manipulate":
            if self._manipulate(
                action,
                tool_positions,
                object_position,
                grippers,
                thread_tail_position,
                needle_points,
                hoop_passed,
                knot_secure,
                native_grasp_contact_active,
            ) and self.phase_elapsed_s >= 1.4:
                completed = self._advance(tool_positions)
                phase_changed = True
        elif phase == "verify":
            if self.guide_kind == "reposition" and self.has_grippers and self.phase_ticks >= 4:
                grippers[0] = True
            if self.phase_ticks >= 30 and self.phase_elapsed_s >= 1.4:
                completed = self._advance(tool_positions)
                phase_changed = True
        elif phase == "recover":
            arm = primary_arm if self.guide_kind == "hoop_threading" else 1 if self.guide_kind in {"handover", "needle_pass"} and self.arms > 1 else 0
            anchor = self.phase_anchor_tools.get(arm, tool_positions.get(arm))
            target = anchor + np.asarray((0.015, 0.0, 0.030), dtype=np.float32) if anchor is not None else None
            current = tool_positions.get(arm)
            distance = self._set_motion(action, arm, current, target) if self.manipulation_step == 0 else None
            displacement = float(np.linalg.norm(current - anchor)) if current is not None and anchor is not None else 0.0
            if self.manipulation_step == 0 and (
                displacement >= 0.025 or (distance is not None and distance <= 0.014)
            ):
                self.manipulation_step = 1
                self.phase_ticks = 0
                self.phase_started_at = time.monotonic()
            elif self.manipulation_step == 0 and self.phase_ticks >= 120:
                self.degraded_reasons.append(f"recover: bounded target timeout at {distance * 1000:.1f} mm" if distance is not None else "recover: target pose unavailable")
                self.manipulation_step = 1
                self.phase_ticks = 0
                self.phase_started_at = time.monotonic()
            elif self.manipulation_step == 1 and self.phase_ticks >= 24 and self.phase_elapsed_s >= 1.4:
                completed = self._advance(tool_positions)
                phase_changed = True
        return ExpertCommand(action, grippers, phase_changed=phase_changed, completed=completed)
