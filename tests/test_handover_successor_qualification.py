from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/qualify_dranmar_handover_successor.py"
)
SPEC = importlib.util.spec_from_file_location(
    "qualify_dranmar_handover_successor",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
QUALIFICATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALIFICATION)


CHECKPOINT_SHA256 = "a" * 64
JIT_SHA256 = "b" * 64


def _run(seed: int, *, candidate: bool) -> dict:
    success_count = 964 if candidate else 700
    lift_count = 1128 if candidate else 960
    acquisition_count = 993 if candidate else 730
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
        "reset_rotation_randomization_deg": 0.0,
        "first_terminal_outcome_per_environment": True,
        "environment_outcomes": {
            "successful_indices": list(range(success_count)),
            "lifted_10mm_indices": list(range(lift_count)),
            "receiver_acquired_indices": list(
                range(acquisition_count)
            ),
        },
        "termination_term_counts": {
            "needle_dropped_after_pickup": 8 if candidate else 10,
            "receiver_retention_lost": 4 if candidate else 5,
            "protected_surface_force": 0,
            "premature_giver_release": 0,
            "excessive_object_force": 0,
            "object_dropping": 0,
        },
        "checkpoint": None if candidate else {"sha256": "incumbent"},
        "handover_successor": (
            {
                "enabled": True,
                "sha256": CHECKPOINT_SHA256,
                "runtime_heuristic_stack": False,
            }
            if candidate
            else None
        ),
        "pickup_recovery": {"enabled": False},
        "receiver_recovery": {"enabled": False},
        "runtime": {
            "source": {
                "dranmar_revision": "controller",
                "asset_revision": "assets",
            }
        },
    }


def _inputs() -> tuple[dict, dict]:
    incumbents = {}
    candidates = {}
    for seed in QUALIFICATION.QUALIFICATION_SEEDS:
        incumbents[seed] = _run(seed, candidate=False)
        candidates[seed] = [
            _run(seed, candidate=True),
            _run(seed, candidate=True),
        ]
    return incumbents, candidates


def _set_outcomes(
    payload: dict,
    *,
    success: int,
    lift: int = 1128,
    acquisition: int = 993,
) -> None:
    payload["environment_outcomes"] = {
        "successful_indices": list(range(success)),
        "lifted_10mm_indices": list(range(lift)),
        "receiver_acquired_indices": list(range(acquisition)),
    }


def test_standalone_successor_passes_original_absolute_gate() -> None:
    incumbents, candidates = _inputs()
    report = QUALIFICATION.qualify(
        incumbents,
        candidates,
        checkpoint_sha256=CHECKPOINT_SHA256,
        jit_sha256=JIT_SHA256,
    )
    assert report["qualified"] is True
    assert report["aggregate"]["success_rate"] == 964 / 1200
    assert all(gate["passed"] for gate in report["gates"])


def test_successor_gate_rejects_safety_or_checkpoint_drift() -> None:
    incumbents, candidates = _inputs()
    broken = deepcopy(candidates[17][0])
    broken["termination_term_counts"]["protected_surface_force"] = 1
    broken["handover_successor"]["sha256"] = "c" * 64
    candidates[17][0] = broken
    report = QUALIFICATION.qualify(
        incumbents,
        candidates,
        checkpoint_sha256=CHECKPOINT_SHA256,
        jit_sha256=JIT_SHA256,
    )
    failed = {
        gate["id"] for gate in report["gates"] if not gate["passed"]
    }
    assert report["qualified"] is False
    assert "seed_17_run_1_protected_surface_force_zero" in failed
    assert "single_frozen_successor_checkpoint" in failed


def test_final_gate_allows_two_point_per_seed_success_margin() -> None:
    incumbents, candidates = _inputs()
    for incumbent in incumbents.values():
        _set_outcomes(
            incumbent,
            success=968,
            lift=1128,
            acquisition=993,
        )
    low_seed = QUALIFICATION.QUALIFICATION_SEEDS[0]
    for candidate in candidates[low_seed]:
        _set_outcomes(
            candidate,
            success=944,
            lift=1104,
            acquisition=972,
        )
    for seed in QUALIFICATION.QUALIFICATION_SEEDS[1:]:
        for candidate in candidates[seed]:
            _set_outcomes(
                candidate,
                success=1000,
                lift=1140,
                acquisition=1008,
            )

    report = QUALIFICATION.qualify(
        incumbents,
        candidates,
        checkpoint_sha256=CHECKPOINT_SHA256,
        jit_sha256=JIT_SHA256,
    )
    assert report["qualified"] is True
    assert abs(
        report["per_seed"][str(low_seed)][
            "success_rate_difference_vs_incumbent"
        ]
        + 0.02
    ) < 1.0e-12

    for candidate in candidates[low_seed]:
        _set_outcomes(
            candidate,
            success=943,
            lift=1104,
            acquisition=972,
        )
    report = QUALIFICATION.qualify(
        incumbents,
        candidates,
        checkpoint_sha256=CHECKPOINT_SHA256,
        jit_sha256=JIT_SHA256,
    )
    failed = {
        gate["id"] for gate in report["gates"] if not gate["passed"]
    }
    assert (
        f"seed_{low_seed}_success_noninferior_to_"
        "incumbent_with_2pp_margin"
    ) in failed


def test_final_gate_requires_strict_aggregate_success_superiority() -> None:
    incumbents, candidates = _inputs()
    for seed in QUALIFICATION.QUALIFICATION_SEEDS:
        _set_outcomes(incumbents[seed], success=968)
        for candidate in candidates[seed]:
            _set_outcomes(candidate, success=968)
    report = QUALIFICATION.qualify(
        incumbents,
        candidates,
        checkpoint_sha256=CHECKPOINT_SHA256,
        jit_sha256=JIT_SHA256,
    )
    failed = {
        gate["id"] for gate in report["gates"] if not gate["passed"]
    }
    assert "aggregate_success_strictly_exceeds_incumbent" in failed


def test_final_gate_uses_corrected_statistical_adverse_event_test() -> None:
    incumbents, candidates = _inputs()
    candidates[17][0]["termination_term_counts"][
        "object_dropping"
    ] = 1
    report = QUALIFICATION.qualify(
        incumbents,
        candidates,
        checkpoint_sha256=CHECKPOINT_SHA256,
        jit_sha256=JIT_SHA256,
    )
    drop_gate = next(
        gate
        for gate in report["gates"]
        if gate["id"]
        == (
            "aggregate_drop_rate_has_no_"
            "statistically_supported_increase"
        )
    )
    assert drop_gate["passed"] is True
    assert drop_gate["holm_adjusted_p_value"] >= 0.05

    for seed in QUALIFICATION.QUALIFICATION_SEEDS:
        for candidate in candidates[seed]:
            candidate["termination_term_counts"][
                "object_dropping"
            ] = 40
    report = QUALIFICATION.qualify(
        incumbents,
        candidates,
        checkpoint_sha256=CHECKPOINT_SHA256,
        jit_sha256=JIT_SHA256,
    )
    drop_gate = next(
        gate
        for gate in report["gates"]
        if gate["id"]
        == (
            "aggregate_drop_rate_has_no_"
            "statistically_supported_increase"
        )
    )
    assert drop_gate["passed"] is False
    assert drop_gate["holm_adjusted_p_value"] < 0.05


def test_development_cli_loads_only_development_seeds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    arguments = ["--profile", "development"]
    expected = set(QUALIFICATION.DEVELOPMENT_SEEDS)
    for seed in QUALIFICATION.DEVELOPMENT_SEEDS:
        incumbent = tmp_path / f"incumbent-{seed}.json"
        candidate = tmp_path / f"candidate-{seed}.json"
        payload = {"kind": "held_out_play", "seed": seed}
        incumbent.write_text(json.dumps(payload))
        candidate.write_text(json.dumps(payload))
        arguments.extend(("--incumbent", f"{seed}={incumbent}"))
        arguments.extend(("--candidate", f"{seed}={candidate}"))

    checkpoint = tmp_path / "successor.pt"
    jit = tmp_path / "successor.jit.pt"
    output = tmp_path / "development.json"
    checkpoint.write_bytes(b"checkpoint")
    jit.write_bytes(b"jit")
    arguments.extend(
        (
            "--checkpoint",
            str(checkpoint),
            "--jit",
            str(jit),
            "--output",
            str(output),
        )
    )

    def fake_qualify(incumbents, candidates, **kwargs):
        assert set(incumbents) == expected
        assert set(candidates) == expected
        assert kwargs["seeds"] == QUALIFICATION.DEVELOPMENT_SEEDS
        assert kwargs["expected_num_envs"] == 600
        assert kwargs["profile"] == "development"
        return {"qualified": True}

    monkeypatch.setattr(QUALIFICATION, "qualify", fake_qualify)
    assert QUALIFICATION.main(arguments) == 0
    assert json.loads(output.read_text()) == {"qualified": True}
