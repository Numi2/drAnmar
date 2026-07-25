from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import unittest


PACKAGE = Path(__file__).resolve().parents[1]
MODULE_PATH = PACKAGE / (
    "source/extensions/orbit.surgical.assets/orbit/surgical/assets/"
    "adaptive_hemostasis_robot.py"
)
SPEC = importlib.util.spec_from_file_location("adaptive_hemostasis_robot", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class _InvalidPrim:
    def IsValid(self):
        return False


class _Stage:
    def GetPrimAtPath(self, _path):
        return _InvalidPrim()

    def RemovePrim(self, _path):
        pass


class AdaptiveHemostasisTests(unittest.TestCase):
    def test_phase_contract_is_complete(self):
        expected = {
            "inspect", "clear", "compress", "temporary_control_check", "clip",
            "release_compression", "patch", "pressure_challenge", "verify",
            "complete", "abort",
        }
        self.assertEqual(set(MODULE.PHASE_TARGETS), expected)
        joint_names = set(MODULE.TOOL_JOINTS.values())
        for targets in MODULE.PHASE_TARGETS.values():
            self.assertEqual(set(targets), joint_names)
            self.assertTrue(all(math.isfinite(value) for value in targets.values()))

    def test_hemorrhage_ledger_conserves_volume(self):
        ledger = MODULE.HemorrhageLedger(initial_reservoir_ml=2.0, reservoir_ml=2.0)
        self.assertEqual(ledger.emit(1.0), 1.0)
        self.assertEqual(ledger.suction(0.25), 0.25)
        self.assertEqual(ledger.spill(0.25), 0.25)
        self.assertEqual(ledger.discard(0.25), 0.25)
        self.assertAlmostEqual(ledger.conservation_error_ml, 0.0)
        with self.assertRaises(ValueError):
            ledger.emit(float("nan"))

    def test_bleed_model_sealing_is_monotonic(self):
        model = MODULE.ReducedOrderBleedModel()
        open_flow = model.flow_ml_min()
        model.compression_fraction = 0.5
        compressed = model.flow_ml_min()
        model.clip_occlusion_fraction = 0.8
        clipped = model.flow_ml_min()
        model.patch_seal_fraction = 0.95
        patched = model.flow_ml_min()
        self.assertGreater(open_flow, compressed)
        self.assertGreater(compressed, clipped)
        self.assertGreater(clipped, patched)

    def test_suction_validates_and_accounts(self):
        ledger = MODULE.HemorrhageLedger(initial_reservoir_ml=1.0, reservoir_ml=1.0)
        ledger.emit(0.004)
        controller = MODULE.AnnularSuctionController((0.0, 0.0, 0.0))
        positions, velocities, mask = controller.update_positions_velocities(
            [[0.001, 0.0, 0.0], [0.02, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            0.01,
            ledger,
        )
        self.assertEqual(len(positions), 1)
        self.assertEqual(mask.tolist(), [True, False])
        self.assertAlmostEqual(ledger.suctioned_ml, 0.002)
        with self.assertRaises(ValueError):
            controller.update_positions_velocities([[0.0, 0.0]], [[0.0, 0.0]], 0.1)

    def test_compression_force_envelope(self):
        controller = MODULE.TemporaryCompressionController("/Tool", "/Vessel")
        self.assertEqual(controller.update_loads(1.8, 1.7)["mode"], "controlled")
        self.assertEqual(controller.update_loads(4.1, 1.8)["mode"], "soft_limit")
        self.assertEqual(controller.update_loads(7.1, 1.8)["mode"], "hard_release")

    def test_clip_and_patch_retention(self):
        stage = _Stage()
        clip = MODULE.ClipRetentionController()
        bond = clip.register({"clip_path": "/Clip", "attachment_paths": ["/a", "/b"]})
        self.assertFalse(clip.apply_load(bond, 2.8, stage=stage))
        self.assertTrue(clip.apply_load(bond, 2.81, stage=stage))
        patch = MODULE.HemostaticPatchBondController()
        patch_bond = MODULE.PatchBond("/Patch", ["/p"])
        self.assertFalse(patch.apply_load(patch_bond, 0.8, stage=stage))
        self.assertTrue(patch.apply_load(patch_bond, 0.81, stage=stage))
        cured = MODULE.PatchBond("/Patch2", ["/q"], cure_fraction=1.0)
        self.assertFalse(patch.apply_load(cured, 8.0, stage=stage))
        self.assertTrue(patch.apply_load(cured, 8.01, stage=stage))

    def test_pressure_challenge_and_verification(self):
        sequence = MODULE.AdaptiveHemostasisSequenceController()
        sequence.set_clip_occlusion(0.999)
        sequence.set_patch_seal(0.999)
        sequence.transition("pressure_challenge")
        self.assertEqual(sequence.bleed_model.pressure_pa, sequence.challenge_pressure_pa)
        sequence.transition("verify")
        for _ in range(51):
            result = sequence.update_verification(0.1)
        self.assertTrue(result["complete"])
        self.assertTrue(result["passed"])
        self.assertLessEqual(
            result["average_flow_ml_min"], sequence.verifier.maximum_flow_ml_min
        )
        sequence.transition("complete")
        self.assertEqual(sequence.bleed_model.pressure_pa, sequence.baseline_pressure_pa)

    def test_invalid_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            MODULE.ReducedOrderBleedModel(density_kg_m3=0.0)
        with self.assertRaises(ValueError):
            MODULE.SealVerificationController(observation_window_s=0.0)
        with self.assertRaises(KeyError):
            MODULE.phase_targets("invented")
        with self.assertRaises(KeyError):
            MODULE.frame_path("/Tool", "invented")


if __name__ == "__main__":
    unittest.main()
