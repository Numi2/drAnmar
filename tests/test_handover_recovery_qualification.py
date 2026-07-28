from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/qualify_dranmar_handover_recovery.py"
)
SPEC = importlib.util.spec_from_file_location(
    "qualify_dranmar_handover_recovery",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
QUALIFICATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALIFICATION)


def _run(seed: int, *, candidate: bool) -> dict:
    incumbent_successes = list(range(700))
    successes = list(range(964 if candidate else 700))
    lifted = list(range(1128 if candidate else 960))
    acquired = list(range(993 if candidate else 730))
    retry_counts = [0 if index < 700 else 1 for index in range(1200)]
    return {
        "_source": {
            "path": f"/evidence/{seed}-{'candidate' if candidate else 'base'}",
            "sha256": f"evidence-{seed}-{candidate}",
        },
        "kind": "held_out_play",
        "seed": seed,
        "num_envs": 1200,
        "completed_episodes": 1200,
        "frames_per_env": 2000,
        "episode_length_s": 40.0,
        "first_terminal_outcome_per_environment": True,
        "environment_outcomes": {
            "successful_indices": successes,
            "lifted_10mm_indices": lifted,
            "receiver_acquired_indices": acquired,
            "termination_indices": {},
        },
        "termination_term_counts": {
            "needle_dropped_after_pickup": 10 if not candidate else 8,
            "protected_surface_force": 1 if not candidate else 0,
            "premature_giver_release": 1 if not candidate else 0,
            "excessive_object_force": 0,
            "object_dropping": 0,
        },
        "checkpoint": {
            "sha256": QUALIFICATION.IMMUTABLE_BASE_SHA256
        },
        "runtime": {
            "source": {
                "dranmar_revision": "controller",
                "asset_revision": "assets",
            }
        },
        "pickup_recovery": (
            {
                "enabled": True,
                "head_checkpoint": {"sha256": "pickup"},
                "position_cap_m": 0.0025,
                "orientation_cap_deg": 2.0,
                "retry_count_by_environment": retry_counts,
                "first_attempt_action_mismatches": 0,
                "first_attempt_action_max_abs_difference": 0.0,
            }
            if candidate
            else {"enabled": False}
        ),
        "receiver_recovery": (
            {
                "enabled": True,
                "head_checkpoint": {"sha256": "receiver"},
                "position_cap_m": 0.0025,
                "orientation_cap_deg": 2.0,
                "retry_count_by_environment": retry_counts,
                "first_attempt_action_mismatches": 0,
                "first_attempt_action_max_abs_difference": 0.0,
            }
            if candidate
            else {"enabled": False}
        ),
        "exports": {
            "stateful_composite_jit": {
                "parity_checked": True,
                "action_mismatches": 0,
                "maximum_action_absolute_difference": 0.0,
            }
        },
        "_incumbent_successes": incumbent_successes,
    }


def _qualification_inputs() -> tuple[dict, dict]:
    incumbents = {}
    candidates = {}
    for seed in QUALIFICATION.QUALIFICATION_SEEDS:
        incumbents[seed] = _run(seed, candidate=False)
        candidates[seed] = [
            _run(seed, candidate=True),
            _run(seed, candidate=True),
        ]
    return incumbents, candidates


def test_complete_frozen_bundle_passes_all_promotion_gates() -> None:
    incumbents, candidates = _qualification_inputs()

    report = QUALIFICATION.qualify(incumbents, candidates)

    assert report["qualified"] is True
    assert report["aggregate"]["success_rate"] == 964 / 1200
    assert report["aggregate"]["lift_rate"] == 0.94
    assert all(gate["passed"] for gate in report["gates"])


def test_missing_cached_success_and_safety_regression_fail_closed() -> None:
    incumbents, candidates = _qualification_inputs()
    broken = deepcopy(candidates[17][0])
    broken["environment_outcomes"]["successful_indices"].remove(3)
    broken["termination_term_counts"]["needle_dropped_after_pickup"] = 11
    candidates[17][0] = broken

    report = QUALIFICATION.qualify(incumbents, candidates)

    assert report["qualified"] is False
    failed = {
        gate["id"] for gate in report["gates"] if not gate["passed"]
    }
    assert "seed_17_run_1_cached_first_attempt_successes" in failed
    assert (
        "seed_17_run_1_needle_dropped_after_pickup_nonincrease"
        in failed
    )


def test_skipped_composite_export_parity_fails_closed() -> None:
    incumbents, candidates = _qualification_inputs()
    candidates[17][0]["exports"]["stateful_composite_jit"][
        "parity_checked"
    ] = False

    report = QUALIFICATION.qualify(incumbents, candidates)

    assert report["qualified"] is False
    assert any(
        gate["id"] == "seed_17_run_1_export_parity"
        and not gate["passed"]
        for gate in report["gates"]
    )


def test_base_policy_lock_matches_fail_closed_qualification() -> None:
    lock_path = (
        Path(__file__).resolve().parents[1]
        / "docs/handover_recovery_80/base_policy.lock.json"
    )
    lock = json.loads(lock_path.read_text())

    assert (
        lock["base_checkpoint"]["sha256"]
        == QUALIFICATION.IMMUTABLE_BASE_SHA256
    )
    assert lock["controller"]["episode_length_s"] == 40.0
    assert lock["controller"]["physical_observation_values"] == 98
    assert lock["controller"]["physical_action_values"] == 14
    assert lock["qualification_seeds"] == [17, 2361, 4099]


def test_paired_baseline_funnel_is_internally_consistent() -> None:
    baseline_path = (
        Path(__file__).resolve().parents[1]
        / "docs/handover_recovery_80/paired_40s_baseline.json"
    )
    baseline = json.loads(baseline_path.read_text())
    aggregate = baseline["aggregate"]
    seeds = baseline["seeds"].values()

    assert aggregate["episodes"] == 3600
    assert aggregate["successful"] == sum(
        seed["successful"] for seed in seeds
    )
    assert aggregate["lifted_10mm"] == sum(
        seed["lifted_10mm"] for seed in seeds
    )
    assert aggregate["receiver_acquired"] == sum(
        seed["receiver_acquired"] for seed in seeds
    )
    assert aggregate["success_rate"] == (
        aggregate["successful"] / aggregate["episodes"]
    )
    assert aggregate["receiver_acquisition_given_lift"] == (
        aggregate["receiver_acquired"] / aggregate["lifted_10mm"]
    )
