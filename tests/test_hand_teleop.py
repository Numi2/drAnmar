# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dr_anmar_hand_teleop import (  # noqa: E402
    HandTeleopRuntime,
    proportional_gripper_action,
    proportional_jaw_targets,
    validate_hand_frame,
)
from dr_anmar_psm_native_adapter import native_ik_action_scales  # noqa: E402
from install_hand_control_assets import default_destination  # noqa: E402


def hand(
    arm: int = 0,
    *,
    tracked: bool = True,
    engaged: bool = False,
    translation: list[float] | None = None,
    rotation: list[float] | None = None,
    aperture: float = 0.5,
    confidence: float = 0.9,
) -> dict[str, object]:
    return {
        "arm": arm,
        "tracked": tracked,
        "motion_engaged": engaged,
        "translation_offset_m": translation or [0.0, 0.0, 0.0],
        "rotation_vector_rad": rotation or [0.0, 0.0, 0.0],
        "aperture_normalized": aperture,
        "confidence": confidence,
    }


class HandFrameValidationTests(unittest.TestCase):
    def test_rejects_duplicate_arm_and_non_finite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "appear once"):
            validate_hand_frame(1, [hand(), hand()], arms=2)
        invalid = hand()
        invalid["rotation_vector_rad"] = [0.0, math.nan, 0.0]
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_hand_frame(2, [invalid], arms=2)

    def test_rejects_out_of_bounds_without_mutating_runtime(self) -> None:
        runtime = HandTeleopRuntime(2)
        runtime.enable_motion()
        runtime.submit(10, [hand(0, engaged=False), hand(1, engaged=False)], now=1.0)
        before = runtime.snapshot(now=1.0)
        invalid = hand(0, engaged=True, translation=[0.121, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "workspace bound"):
            runtime.submit(11, [invalid], now=1.1)
        self.assertEqual(before, runtime.snapshot(now=1.0))

    def test_sequence_must_be_monotonic(self) -> None:
        runtime = HandTeleopRuntime(1)
        runtime.submit(4, [hand()], now=1.0)
        with self.assertRaisesRegex(ValueError, "monotonically"):
            runtime.submit(4, [hand()], now=1.1)


class ResamplingAndSafetyTests(unittest.TestCase):
    def armed_runtime(self) -> HandTeleopRuntime:
        runtime = HandTeleopRuntime(1)
        runtime.enable_motion()
        runtime.submit(1, [hand(engaged=False)], now=1.0)
        return runtime

    def test_cumulative_pose_is_consumed_across_slow_steps(self) -> None:
        runtime = self.armed_runtime()
        runtime.submit(
            2,
            [hand(engaged=True, translation=[0.025, 0.0, 0.0])],
            now=1.01,
        )
        scales = [[0.01, 0.01, 0.01, 0.05, 0.05, 0.05]]
        commands = [runtime.consume(scales, now=now)[0][0] for now in (1.02, 1.03, 1.04)]
        for actual, expected in zip(commands, [1.0, 1.0, 0.5]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(runtime.arm_states[0].consumed_offset[0], 0.025)

    def test_watchdog_discards_motion_but_holds_aperture(self) -> None:
        runtime = self.armed_runtime()
        runtime.submit(
            2,
            [hand(engaged=True, translation=[0.04, 0.0, 0.0], aperture=0.27)],
            now=1.01,
        )
        command = runtime.consume([[0.01] * 6], now=1.30)[0]
        self.assertEqual(command, [0.0] * 6)
        self.assertTrue(runtime.arm_states[0].reacquire_unclutched)
        self.assertAlmostEqual(runtime.arm_states[0].aperture_normalized, 0.27)
        self.assertEqual(runtime.arm_states[0].target_offset, [0.0] * 6)

    def test_reacquisition_requires_frozen_frame(self) -> None:
        runtime = self.armed_runtime()
        runtime.consume([[0.01] * 6], now=1.30)
        runtime.submit(2, [hand(engaged=True, translation=[0.02, 0.0, 0.0])], now=1.31)
        self.assertFalse(runtime.arm_states[0].motion_engaged)
        runtime.submit(3, [hand(engaged=False)], now=1.32)
        self.assertFalse(runtime.arm_states[0].reacquire_unclutched)
        runtime.submit(4, [hand(engaged=True, translation=[0.005, 0.0, 0.0])], now=1.33)
        self.assertAlmostEqual(runtime.consume([[0.01] * 6], now=1.34)[0][0], 0.5)

    def test_manual_takeover_disables_pending_hand_motion(self) -> None:
        runtime = self.armed_runtime()
        runtime.submit(2, [hand(engaged=True, translation=[0.02, 0.0, 0.0])], now=1.01)
        runtime.disable_motion()
        self.assertFalse(runtime.enabled)
        self.assertEqual(runtime.consume([[0.01] * 6], now=1.02)[0], [0.0] * 6)
        self.assertEqual(runtime.arm_states[0].target_offset, [0.0] * 6)


class NativeScaleAndGripperTests(unittest.TestCase):
    def test_asset_installer_uses_workstation_data_root(self) -> None:
        with patch.dict(
            os.environ,
            {"DR_ANMAR_ROOT": "/srv/dr-anmar"},
            clear=False,
        ):
            self.assertEqual(
                default_destination(),
                Path(
                    "/srv/dr-anmar/assets/hand-control/"
                    "mediapipe-tasks-vision-0.10.35"
                ),
            )

    def test_reads_active_nvidia_ik_term_scale(self) -> None:
        robot = object()
        ik_type = type("DifferentialInverseKinematicsAction", (), {})
        ik_term = ik_type()
        ik_term._asset = robot
        ik_term._scale = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
        manager = SimpleNamespace(
            active_terms=["arm"],
            get_term=lambda _name: ik_term,
        )
        base = SimpleNamespace(
            scene={"robot_1": robot},
            action_manager=manager,
        )
        self.assertEqual(
            native_ik_action_scales(SimpleNamespace(unwrapped=base), ["robot_1"]),
            [[0.01, 0.02, 0.03, 0.04, 0.05, 0.06]],
        )

    def test_proportional_gripper_endpoints_and_midpoint(self) -> None:
        self.assertEqual(proportional_gripper_action(0.0), -1.0)
        self.assertEqual(proportional_gripper_action(0.5), 0.0)
        self.assertEqual(proportional_gripper_action(1.0), 1.0)
        self.assertEqual(
            proportional_jaw_targets(0.0, close_rad=0.07, open_rad=0.5),
            (-0.07, 0.07),
        )
        midpoint = proportional_jaw_targets(0.5, close_rad=0.07, open_rad=0.5)
        self.assertAlmostEqual(midpoint[0], -0.285)
        self.assertAlmostEqual(midpoint[1], 0.285)
        self.assertEqual(
            proportional_jaw_targets(1.0, close_rad=0.07, open_rad=0.5),
            (-0.5, 0.5),
        )

    def test_policy_contract_stays_binary_and_cartesian_recording_stays_exact(self) -> None:
        adapter = (ROOT / "scripts/dr_anmar_psm_native_adapter.py").read_text(encoding="utf-8")
        workstation = (ROOT / "scripts/dr_anmar_workstation.py").read_text(encoding="utf-8")
        self.assertIn("torch.where(", adapter)
        self.assertIn("torch.full_like(raw_gripper, -1.0)", adapter)
        self.assertIn('frame["cartesian_actions"] = action_np.copy()', workstation)
        self.assertIn('frame["resolved_joint_targets"] = native_joint_targets_np.copy()', workstation)


if __name__ == "__main__":
    unittest.main()
