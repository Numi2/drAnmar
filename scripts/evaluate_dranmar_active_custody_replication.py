#!/usr/bin/env python3
"""Evaluate a locked custody candidate on fresh randomized streams."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path

import torch

import train_dranmar_active_custody_controller as controller


REPORT_SCHEMA_VERSION = "dranmar-active-custody-controller-replication-1.0"


def _source_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_candidate(
    path: Path,
    *,
    source_revision: str,
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != controller.CHECKPOINT_SCHEMA_VERSION
        or payload.get("fresh_randomized_replication_authorized") is not True
        or payload.get("heldout_physics_evaluation_authorized") is not False
        or payload.get("main_policy_replacement_authorized") is not False
    ):
        raise ValueError("candidate is not authorized for fresh randomized replication")
    candidate_source = str(payload.get("source_revision", ""))
    if re.fullmatch(r"[0-9a-f]{40}", candidate_source) is None or not controller._is_ancestor(
        candidate_source, source_revision
    ):
        raise ValueError("candidate source is not an ancestor")
    policy = payload.get("constrained_replication_candidate")
    if (
        not isinstance(policy, dict)
        or policy.get("kind") != "highest_risk_quartile_opposite_pulse_else_no_op"
        or policy.get("active_action_id") != -1
        or policy.get("inactive_action_id") != 0
        or policy.get("requires_fresh_randomized_replication") is not True
        or not 0.0 < float(policy.get("risk_threshold", 0.0)) < 1.0
    ):
        raise ValueError("candidate constrained policy is incompatible")
    model = payload.get("model")
    if (
        not isinstance(model, dict)
        or model.get("kind") != "standardized_three_head_linear_outcome_network"
        or "success" not in model
        or "unsafe" not in model
    ):
        raise ValueError("candidate outcome model is incompatible")
    return payload


def _locked_policy_actions(
    risk: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    if risk.ndim != 1 or not 0.0 < threshold < 1.0:
        raise ValueError("locked risk-screen contract drifted")
    return torch.where(
        risk >= threshold,
        torch.zeros(risk.shape, dtype=torch.long),
        torch.full(risk.shape, controller.NO_OP_INDEX, dtype=torch.long),
    )


def _required_sample_plan(
    evaluation: dict[str, object],
) -> dict[str, int | float | None]:
    effect = evaluation["success_effect_vs_no_op"]
    estimate = float(effect["estimate"])
    standard_error = float(effect["standard_error"])
    samples = int(effect["samples"])
    if estimate <= 0.0 or standard_error <= 0.0:
        return {
            "observed_samples": samples,
            "observed_effect": estimate,
            "estimated_total_samples_for_positive_lcb": None,
            "estimated_additional_samples": None,
        }
    scale = (controller.ONE_SIDED_95_Z * standard_error / estimate) ** 2
    estimated_total = max(samples, math.ceil(1.2 * samples * scale))
    return {
        "observed_samples": samples,
        "observed_effect": estimate,
        "estimated_total_samples_for_positive_lcb": estimated_total,
        "estimated_additional_samples": max(0, estimated_total - samples),
    }


def _run(args: argparse.Namespace) -> int:
    if args.minimum_action_support <= 0:
        raise ValueError("minimum action support must be positive")
    output = Path(args.output).expanduser().resolve()
    candidate_path = Path(args.candidate).expanduser().resolve()
    risk_path = Path(args.risk_checkpoint).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"replication report already exists: {output}")

    torch.set_num_threads(1)
    source_revision = _source_revision()
    candidate = _load_candidate(
        candidate_path,
        source_revision=source_revision,
    )
    paths = [Path(value).expanduser().resolve() for value in args.dataset]
    payloads = [controller._load_dataset(path) for path in paths]
    stream_keys = [
        (int(payload["seed"]), int(payload["seed_stream_offset"])) for payload in payloads
    ]
    if len(stream_keys) != len(set(stream_keys)):
        raise ValueError("replication datasets repeat a seed stream")
    if {seed for seed, _ in stream_keys} != controller.DEVELOPMENT_SEEDS:
        raise ValueError("replication requires all development seeds")
    discovery_streams = {
        (int(item["seed"]), int(item["seed_stream_offset"]))
        for item in candidate["training_datasets"]
    }
    if discovery_streams.intersection(stream_keys):
        raise ValueError("replication reuses a discovery seed stream")
    discovery_hashes = {str(item["sha256"]) for item in candidate["training_datasets"]}
    replication_hashes = {controller._sha256(path) for path in paths}
    if discovery_hashes.intersection(replication_hashes):
        raise ValueError("replication reuses a discovery dataset")

    base_hash = str(candidate["base_checkpoint_sha256"])
    receiver_hash = str(candidate["receiver_candidate_checkpoint_sha256"])
    if any(
        payload["base_checkpoint_sha256"] != base_hash
        or payload["receiver_candidate_checkpoint_sha256"] != receiver_hash
        for payload in payloads
    ):
        raise ValueError("replication policy checkpoint provenance drifted")
    expected_risk_hash = str(candidate["risk_checkpoint"]["sha256"])
    if controller._sha256(risk_path) != expected_risk_hash:
        raise ValueError("replication risk checkpoint hash drifted")
    risk_checkpoint = controller._load_risk_checkpoint(
        risk_path,
        source_revision=source_revision,
        base_hash=base_hash,
        candidate_hash=receiver_hash,
    )

    features = torch.cat([controller._causal_features(payload) for payload in payloads])
    action_index = torch.cat(
        [controller._action_indices(payload["assigned_action_id"]) for payload in payloads]
    )
    propensity = torch.cat([payload["assigned_action_probability"].float() for payload in payloads])
    success = torch.cat([payload["eventual_full_success"].bool() for payload in payloads])
    unsafe = torch.cat(
        [
            controller._termination_indicator(
                payload,
                controller._UNSAFE_TERMINATIONS,
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
    risk = controller._risk_probability(risk_checkpoint, features)
    threshold = float(candidate["constrained_replication_candidate"]["risk_threshold"])
    policy_action = _locked_policy_actions(risk, threshold=threshold)
    success_probability = controller._outcome_probability(
        candidate["model"]["success"],
        features,
    )
    unsafe_probability = controller._outcome_probability(
        candidate["model"]["unsafe"],
        features,
    )

    action_support = {
        str(seed): {
            str(int(controller.ACTION_IDS[index])): int(
                ((seed_index == seed) & (action_index == index)).sum()
            )
            for index in range(3)
        }
        for seed in sorted(controller.DEVELOPMENT_SEEDS)
    }
    support_gate_passed = all(
        count >= args.minimum_action_support
        for seed_counts in action_support.values()
        for count in seed_counts.values()
    )
    seeds: dict[str, object] = {}
    for seed in sorted(controller.DEVELOPMENT_SEEDS):
        selected = seed_index == seed
        seeds[str(seed)] = controller._policy_evaluation(
            success_probability[selected],
            unsafe_probability[selected],
            policy_action[selected],
            action_index[selected],
            propensity[selected],
            success[selected],
            unsafe[selected],
        )
    aggregate = controller._policy_evaluation(
        success_probability,
        unsafe_probability,
        policy_action,
        action_index,
        propensity,
        success,
        unsafe,
    )
    safety_gate_passed = bool(
        aggregate["unsafe_effect_vs_no_op"]["upper_one_sided_95"] <= args.maximum_unsafe_ucb
        and all(
            value["unsafe_effect_vs_no_op"]["upper_one_sided_95"] <= args.maximum_unsafe_ucb
            for value in seeds.values()
        )
    )
    aggregate_gate_passed = bool(
        aggregate["success_effect_vs_no_op"]["lower_one_sided_95"] > args.minimum_effect_lcb
    )
    directional_seed_replication = all(
        value["success_effect_vs_no_op"]["estimate"] > 0.0 for value in seeds.values()
    )
    worst_seed_gate_passed = all(
        value["success_effect_vs_no_op"]["lower_one_sided_95"] > args.minimum_effect_lcb
        for value in seeds.values()
    )
    strict_gate_passed = bool(
        support_gate_passed
        and safety_gate_passed
        and aggregate_gate_passed
        and worst_seed_gate_passed
    )
    directional_gate_passed = bool(
        support_gate_passed
        and safety_gate_passed
        and aggregate_gate_passed
        and directional_seed_replication
    )
    if strict_gate_passed:
        status = "replicated_authorized_for_heldout_physics"
        reason = (
            "The locked candidate replicated with a positive one-sided 95% "
            "success-effect lower bound in aggregate and on every development "
            "seed, with no observed unsafe-event increase."
        )
    elif directional_gate_passed:
        status = "directionally_replicated_more_power_required"
        reason = (
            "The locked candidate replicated in aggregate and remained "
            "positive on every development seed, but at least one per-seed "
            "lower confidence bound still crosses zero. Collect only the "
            "precomputed additional sample budget before held-out rollout."
        )
    else:
        status = "fresh_replication_failed"
        reason = (
            "The locked candidate did not reproduce its aggregate and "
            "cross-seed direction on fresh randomized streams. Preserve the "
            "evidence and redesign the intervention; do not run held-out."
        )

    provenance = controller._dataset_provenance(paths, payloads)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_revision": source_revision,
        "status": status,
        "decision": {
            "strict_gate_passed": strict_gate_passed,
            "directional_gate_passed": directional_gate_passed,
            "heldout_physics_evaluation_authorized": strict_gate_passed,
            "main_policy_replacement_authorized": False,
            "real_robot_authorized": False,
            "reason": reason,
        },
        "candidate": {
            "path": str(candidate_path),
            "sha256": controller._sha256(candidate_path),
            "source_revision": candidate["source_revision"],
            "discovery_datasets_excluded": True,
        },
        "risk_checkpoint": {
            "path": str(risk_path),
            "sha256": expected_risk_hash,
        },
        "locked_policy": {
            "kind": ("highest_risk_quartile_opposite_pulse_else_no_op"),
            "risk_threshold": threshold,
            "active_action_id": -1,
            "inactive_action_id": 0,
            "fit_or_threshold_updates_on_replication_data": False,
        },
        "gates": {
            "minimum_action_support_per_seed_arm": (args.minimum_action_support),
            "minimum_success_effect_lower_bound": (args.minimum_effect_lcb),
            "maximum_unsafe_effect_upper_bound": (args.maximum_unsafe_ucb),
            "support_gate_passed": support_gate_passed,
            "safety_gate_passed": safety_gate_passed,
            "aggregate_gate_passed": aggregate_gate_passed,
            "directional_seed_replication": (directional_seed_replication),
            "worst_seed_gate_passed": worst_seed_gate_passed,
            "strict_gate_passed": strict_gate_passed,
        },
        "datasets": provenance,
        "action_support": action_support,
        "aggregate_evaluation": aggregate,
        "seed_evaluations": seeds,
        "additional_sample_plan_by_seed": {
            seed: _required_sample_plan(value) for seed, value in seeds.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Evaluate a locked custody candidate on fresh randomized streams")
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--risk_checkpoint", required=True)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum_action_support", type=int, default=100)
    parser.add_argument("--minimum_effect_lcb", type=float, default=0.0)
    parser.add_argument("--maximum_unsafe_ucb", type=float, default=0.0)
    return parser


if __name__ == "__main__":
    raise SystemExit(_run(_parser().parse_args()))
