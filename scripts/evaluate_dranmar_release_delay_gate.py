#!/usr/bin/env python3
"""Gate post-probe release timing with randomized outcome evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch

from train_dranmar_active_custody_controller import (
    _causal_features,
    _load_risk_checkpoint,
    _risk_cutpoints,
    _risk_probability,
    _risk_strata,
    _sha256,
    _source_revision,
    _termination_indicator,
)


DATASET_SCHEMA_VERSION = (
    "dranmar-receiver-active-custody-release-delay-dataset-1.0"
)
REPORT_SCHEMA_VERSION = "dranmar-active-custody-release-delay-gate-1.0"
DEVELOPMENT_SEEDS = {104729, 130363, 196613}
DELAY_ARMS = (0, 1, 3)
ONE_SIDED_95_Z = 1.6448536269514722
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_RELEASE_DELAY_SEMANTICS = {
    "0": "immediate_release_after_probe",
    "1": "one_frame_reclose_then_release",
    "3": "three_frame_reclose_then_release",
}
_UNSAFE_TERMINATIONS = {
    "excessive_object_force",
    "protected_surface_force",
}


def _release_compliance(
    assigned_delay: torch.Tensor,
    applied_frames: torch.Tensor,
) -> dict[str, int]:
    assigned_delay = assigned_delay.long()
    applied_frames = applied_frames.long()
    if (
        assigned_delay.ndim != 1
        or applied_frames.shape != assigned_delay.shape
        or set(assigned_delay.unique().tolist()) != set(DELAY_ARMS)
        or bool((applied_frames < 0).any())
        or bool((applied_frames > assigned_delay).any())
        or not torch.equal(
            applied_frames[assigned_delay == 0],
            torch.zeros_like(applied_frames[assigned_delay == 0]),
        )
        or not torch.equal(
            applied_frames[assigned_delay == 1],
            torch.ones_like(applied_frames[assigned_delay == 1]),
        )
        or bool((applied_frames[assigned_delay == 3] < 1).any())
    ):
        raise ValueError("release-delay application contract drifted")
    return {
        "fully_applied": int(
            (applied_frames == assigned_delay).sum().item()
        ),
        "terminally_truncated": int(
            (applied_frames < assigned_delay).sum().item()
        ),
    }


def _load_dataset(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "schema_version",
        "seed",
        "seed_stream_offset",
        "runtime_seed",
        "num_envs",
        "base_checkpoint_sha256",
        "receiver_candidate_checkpoint_sha256",
        "observation_dimension",
        "probe_frames",
        "probe_intervention",
        "environment_index",
        "pre_observation",
        "post_observation",
        "receiver_correction",
        "retry_count",
        "probe_survived",
        "eventual_full_success",
        "termination_names",
        "eventual_termination_flags",
        "randomization",
        "randomization_seed",
        "intervention_profile",
        "assigned_release_delay_frames",
        "assigned_action_probability",
        "release_delay_semantics",
        "applied_delay_frames",
        "applied_receiver_action",
        "applied_giver_action",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"incomplete release-delay dataset: {path}")
    if (
        payload["schema_version"] != DATASET_SCHEMA_VERSION
        or int(payload["seed"]) not in DEVELOPMENT_SEEDS
        or int(payload["runtime_seed"])
        != int(payload["seed"]) + int(payload["seed_stream_offset"])
        or int(payload["observation_dimension"]) != 98
        or int(payload["probe_frames"]) != 1
        or payload["probe_intervention"] != "giver_gripper_open_pulse"
        or payload["randomization"] != "seeded_hash_uniform_three_arm"
        or payload["intervention_profile"]
        != "post_probe_dual_grasp_release_delay"
        or dict(payload["release_delay_semantics"])
        != _RELEASE_DELAY_SEMANTICS
    ):
        raise ValueError(f"incompatible release-delay dataset: {path}")

    environment_index = payload["environment_index"].long()
    count = int(environment_index.numel())
    tensors = (
        "pre_observation",
        "post_observation",
        "receiver_correction",
        "retry_count",
        "probe_survived",
        "eventual_full_success",
        "eventual_termination_flags",
        "assigned_release_delay_frames",
        "assigned_action_probability",
        "applied_delay_frames",
        "applied_receiver_action",
        "applied_giver_action",
    )
    if (
        count == 0
        or environment_index.ndim != 1
        or int(environment_index.unique().numel()) != count
        or int(environment_index.min()) < 0
        or int(environment_index.max()) >= int(payload["num_envs"])
        or any(int(payload[name].shape[0]) != count for name in tensors)
        or payload["eventual_termination_flags"].ndim != 2
        or payload["eventual_termination_flags"].shape[1]
        != len(payload["termination_names"])
        or payload["applied_receiver_action"].shape != (count, 7)
        or payload["applied_giver_action"].shape != (count, 7)
        or not bool(payload["probe_survived"].bool().all())
    ):
        raise ValueError(f"release-delay outcome contract drifted: {path}")

    assigned_delay = payload["assigned_release_delay_frames"].long()
    _release_compliance(
        assigned_delay,
        payload["applied_delay_frames"],
    )
    probability = payload["assigned_action_probability"].float()
    if not torch.allclose(
        probability,
        torch.full_like(probability, 1.0 / 3.0),
        atol=1.0e-7,
        rtol=0.0,
    ):
        raise ValueError(f"release-delay propensity drifted: {path}")

    receiver_action = payload["applied_receiver_action"].float()
    giver_action = payload["applied_giver_action"].float()
    delayed = assigned_delay > 0
    if (
        not bool(torch.isfinite(receiver_action).all())
        or not bool(torch.isfinite(giver_action).all())
        or not torch.allclose(
            receiver_action[:, :6],
            torch.zeros_like(receiver_action[:, :6]),
        )
        or not torch.equal(
            receiver_action[:, 6],
            torch.full_like(receiver_action[:, 6], -1.0),
        )
        or not torch.allclose(
            giver_action[delayed, :6],
            torch.zeros_like(giver_action[delayed, :6]),
        )
        or not torch.equal(
            giver_action[delayed, 6],
            torch.full_like(giver_action[delayed, 6], -1.0),
        )
        or not torch.equal(
            giver_action[~delayed],
            torch.zeros_like(giver_action[~delayed]),
        )
    ):
        raise ValueError(f"release-delay action contract drifted: {path}")

    _causal_features(payload)
    _termination_indicator(payload, _UNSAFE_TERMINATIONS)
    return payload


def _difference_in_success(
    assigned_delay: torch.Tensor,
    success: torch.Tensor,
    *,
    treatment: int,
    control: int = 0,
) -> dict[str, float | int]:
    treatment_mask = assigned_delay == treatment
    control_mask = assigned_delay == control
    treatment_samples = int(treatment_mask.sum().item())
    control_samples = int(control_mask.sum().item())
    if treatment_samples < 2 or control_samples < 2:
        raise ValueError("release-delay comparison lost action support")
    treatment_rate = float(
        success[treatment_mask].float().mean().item()
    )
    control_rate = float(success[control_mask].float().mean().item())
    standard_error = math.sqrt(
        treatment_rate
        * (1.0 - treatment_rate)
        / treatment_samples
        + control_rate
        * (1.0 - control_rate)
        / control_samples
    )
    estimate = treatment_rate - control_rate
    return {
        "treatment_delay_frames": treatment,
        "control_delay_frames": control,
        "treatment_samples": treatment_samples,
        "control_samples": control_samples,
        "treatment_success_rate": treatment_rate,
        "control_success_rate": control_rate,
        "estimate": estimate,
        "standard_error": standard_error,
        "lower_one_sided_95": (
            estimate - ONE_SIDED_95_Z * standard_error
        ),
        "upper_one_sided_95": (
            estimate + ONE_SIDED_95_Z * standard_error
        ),
    }


def _arm_summary(
    assigned_delay: torch.Tensor,
    success: torch.Tensor,
    unsafe: torch.Tensor,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for delay in DELAY_ARMS:
        selected = assigned_delay == delay
        result[str(delay)] = {
            "samples": int(selected.sum().item()),
            "successful": int(success[selected].sum().item()),
            "success_rate": float(
                success[selected].float().mean().item()
            ),
            "unsafe": int(unsafe[selected].sum().item()),
            "unsafe_rate": float(
                unsafe[selected].float().mean().item()
            ),
        }
    return result


def _gate_decision(
    *,
    action_support: dict[str, dict[str, int]],
    aggregate_effects: dict[str, dict[str, float | int]],
    seed_effects: dict[str, dict[str, dict[str, float | int]]],
    risk_effects: dict[str, dict[str, dict[str, float | int]]],
    unsafe_events: int,
    minimum_action_support: int,
) -> dict[str, object]:
    support_gate_passed = all(
        count >= minimum_action_support
        for seed_counts in action_support.values()
        for count in seed_counts.values()
    )
    aggregate_harm_gate_passed = all(
        float(effect["upper_one_sided_95"]) < 0.0
        for effect in aggregate_effects.values()
    )
    seed_direction_gate_passed = all(
        float(effect["estimate"]) <= 0.0
        for effects in seed_effects.values()
        for effect in effects.values()
    )
    no_risk_stratum_benefit = all(
        float(effect["lower_one_sided_95"]) <= 0.0
        for effects in risk_effects.values()
        for effect in effects.values()
    )
    safety_gate_passed = unsafe_events == 0
    immediate_release_locked = bool(
        support_gate_passed
        and aggregate_harm_gate_passed
        and seed_direction_gate_passed
        and no_risk_stratum_benefit
        and safety_gate_passed
    )
    return {
        "support_gate_passed": support_gate_passed,
        "aggregate_delay_harm_gate_passed": (
            aggregate_harm_gate_passed
        ),
        "seed_direction_gate_passed": seed_direction_gate_passed,
        "no_risk_stratum_delay_benefit": no_risk_stratum_benefit,
        "safety_gate_passed": safety_gate_passed,
        "immediate_release_locked": immediate_release_locked,
        "delay_candidate_authorized": False,
        "neural_delay_selector_authorized": False,
        "decision": (
            "retain_immediate_release_after_survived_probe"
            if immediate_release_locked
            else "no_release_timing_promotion_more_evidence_required"
        ),
        "risk_model_role": (
            "risk_stratification_and_next_intervention_gate_only"
        ),
    }


def _run(args: argparse.Namespace) -> int:
    if args.minimum_action_support <= 0:
        raise ValueError("minimum action support must be positive")
    output = Path(args.output).expanduser().resolve()
    risk_path = Path(args.risk_checkpoint).expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"release-delay report already exists: {output}"
        )

    source_revision = _source_revision()
    paths = [
        Path(value).expanduser().resolve() for value in args.dataset
    ]
    payloads = [_load_dataset(path) for path in paths]
    stream_keys = [
        (
            int(payload["seed"]),
            int(payload["seed_stream_offset"]),
        )
        for payload in payloads
    ]
    if len(stream_keys) != len(set(stream_keys)):
        raise ValueError("release-delay datasets repeat a seed stream")
    if {seed for seed, _ in stream_keys} != DEVELOPMENT_SEEDS:
        raise ValueError("release-delay gate requires all development seeds")

    base_hashes = {
        str(payload["base_checkpoint_sha256"])
        for payload in payloads
    }
    candidate_hashes = {
        str(payload["receiver_candidate_checkpoint_sha256"])
        for payload in payloads
    }
    if (
        len(base_hashes) != 1
        or len(candidate_hashes) != 1
        or _SHA256_PATTERN.fullmatch(next(iter(base_hashes))) is None
        or _SHA256_PATTERN.fullmatch(next(iter(candidate_hashes)))
        is None
    ):
        raise ValueError("release-delay checkpoint provenance drifted")
    base_hash = next(iter(base_hashes))
    candidate_hash = next(iter(candidate_hashes))
    risk_checkpoint = _load_risk_checkpoint(
        risk_path,
        source_revision=source_revision,
        base_hash=base_hash,
        candidate_hash=candidate_hash,
    )

    features = torch.cat(
        [_causal_features(payload) for payload in payloads]
    )
    assigned_delay = torch.cat(
        [
            payload["assigned_release_delay_frames"].long()
            for payload in payloads
        ]
    )
    applied_frames = torch.cat(
        [payload["applied_delay_frames"].long() for payload in payloads]
    )
    success = torch.cat(
        [payload["eventual_full_success"].bool() for payload in payloads]
    )
    unsafe = torch.cat(
        [
            _termination_indicator(
                payload,
                _UNSAFE_TERMINATIONS,
            )
            for payload in payloads
        ]
    )
    seed_index = torch.cat(
        [
            torch.full(
                (payload["environment_index"].numel(),),
                int(payload["seed"]),
                dtype=torch.long,
            )
            for payload in payloads
        ]
    )
    risk = _risk_probability(risk_checkpoint, features)
    risk_cutpoints = _risk_cutpoints(risk)
    risk_strata = _risk_strata(risk, risk_cutpoints)

    action_support = {
        str(seed): {
            str(delay): int(
                (
                    (seed_index == seed)
                    & (assigned_delay == delay)
                )
                .sum()
                .item()
            )
            for delay in DELAY_ARMS
        }
        for seed in sorted(DEVELOPMENT_SEEDS)
    }
    aggregate_effects = {
        str(delay): _difference_in_success(
            assigned_delay,
            success,
            treatment=delay,
        )
        for delay in (1, 3)
    }
    seed_effects = {
        str(seed): {
            str(delay): _difference_in_success(
                assigned_delay[seed_index == seed],
                success[seed_index == seed],
                treatment=delay,
            )
            for delay in (1, 3)
        }
        for seed in sorted(DEVELOPMENT_SEEDS)
    }
    risk_effects = {
        str(stratum): {
            str(delay): _difference_in_success(
                assigned_delay[risk_strata == stratum],
                success[risk_strata == stratum],
                treatment=delay,
            )
            for delay in (1, 3)
        }
        for stratum in range(4)
    }
    decision = _gate_decision(
        action_support=action_support,
        aggregate_effects=aggregate_effects,
        seed_effects=seed_effects,
        risk_effects=risk_effects,
        unsafe_events=int(unsafe.sum().item()),
        minimum_action_support=args.minimum_action_support,
    )

    dataset_provenance = []
    for path, payload in zip(paths, payloads, strict=True):
        delay = payload["assigned_release_delay_frames"].long()
        compliance = _release_compliance(
            delay,
            payload["applied_delay_frames"],
        )
        dataset_provenance.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "seed": int(payload["seed"]),
                "seed_stream_offset": int(
                    payload["seed_stream_offset"]
                ),
                "runtime_seed": int(payload["runtime_seed"]),
                "samples": int(delay.numel()),
                "action_counts": {
                    str(value): int((delay == value).sum().item())
                    for value in DELAY_ARMS
                },
                "successful": int(
                    payload["eventual_full_success"].sum().item()
                ),
                **compliance,
            }
        )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_revision": source_revision,
        "risk_checkpoint": {
            "path": str(risk_path),
            "sha256": _sha256(risk_path),
            "source_revision": risk_checkpoint["source_revision"],
            "control_scope": risk_checkpoint["control_scope"],
        },
        "base_checkpoint_sha256": base_hash,
        "receiver_candidate_checkpoint_sha256": candidate_hash,
        "datasets": dataset_provenance,
        "statistical_contract": {
            "estimand": "intention_to_treat_success_rate_difference",
            "control_arm": "0_frame_immediate_release",
            "confidence": "one_sided_95_normal_difference",
            "terminal_truncation_retained": True,
            "minimum_action_support_per_seed": (
                args.minimum_action_support
            ),
        },
        "samples": int(assigned_delay.numel()),
        "action_support": action_support,
        "aggregate": {
            "arms": _arm_summary(
                assigned_delay,
                success,
                unsafe,
            ),
            "effects_vs_immediate": aggregate_effects,
            "application": _release_compliance(
                assigned_delay,
                applied_frames,
            ),
        },
        "by_seed_effects_vs_immediate": seed_effects,
        "risk_stratification": {
            "quartile_cutpoints": risk_cutpoints.tolist(),
            "quartiles": {
                str(stratum): {
                    "samples": int(
                        (risk_strata == stratum).sum().item()
                    ),
                    "mean_predicted_risk": float(
                        risk[risk_strata == stratum].mean().item()
                    ),
                    "arms": _arm_summary(
                        assigned_delay[risk_strata == stratum],
                        success[risk_strata == stratum],
                        unsafe[risk_strata == stratum],
                    ),
                    "effects_vs_immediate": risk_effects[
                        str(stratum)
                    ],
                }
                for stratum in range(4)
            },
        },
        "decision_gate": decision,
        "next_stage": {
            "experiment": (
                "randomized_high_risk_preemptive_retry_vs_"
                "immediate_release"
            ),
            "motion_scope": (
                "reuse_existing_bounded_retry_state_machine"
            ),
            "promotion_authorized": False,
            "reason": (
                "the risk model separates outcome risk, while added "
                "release dwell is causally harmful and supplies no "
                "promotable motion action"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["decision_gate"], indent=2, sort_keys=True))
    print(f"[DrAnmar] Release-delay gate: {output}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("--risk_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
    )
    parser.add_argument(
        "--minimum_action_support",
        type=int,
        default=100,
    )
    return parser


def main() -> int:
    return _run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
