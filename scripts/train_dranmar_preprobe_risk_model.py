#!/usr/bin/env python3
"""Train a cross-seed custody-risk model using only pre-probe state."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from analyze_dranmar_active_custody_probe import (
    _apply_platt_calibrator,
    _fit_linear_model,
    _fit_platt_calibrator,
    _linear_logits,
    _metrics,
    _role_invariant_observation,
    _sha256,
    _source_revision,
)
from train_dranmar_active_custody_controller import (
    _causal_features,
    _load_risk_checkpoint,
    _risk_probability,
)


SYMMETRIC_SCHEMA_VERSION = (
    "dranmar-receiver-active-custody-intervention-dataset-1.0"
)
RELEASE_SCHEMA_VERSION = (
    "dranmar-receiver-active-custody-release-delay-dataset-1.0"
)
CHECKPOINT_SCHEMA_VERSION = (
    "dranmar-active-custody-preprobe-risk-model-1.0"
)
REPORT_SCHEMA_VERSION = (
    "dranmar-active-custody-preprobe-risk-gate-1.0"
)
DEVELOPMENT_SEEDS = {104729, 130363, 196613}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _natural_arm_mask(payload: dict[str, object]) -> torch.Tensor:
    schema = payload.get("schema_version")
    if schema == SYMMETRIC_SCHEMA_VERSION:
        mask = payload["assigned_action_id"].long() == 0
    elif schema == RELEASE_SCHEMA_VERSION:
        mask = (
            payload["assigned_release_delay_frames"].long() == 0
        )
    else:
        raise ValueError("unsupported pre-probe training dataset schema")
    if mask.ndim != 1 or not bool(mask.any()):
        raise ValueError("pre-probe dataset has no natural-action support")
    return mask


def _preprobe_features(
    payload: dict[str, object],
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    pre = payload["pre_observation"].float()
    correction = payload["receiver_correction"].float()
    retry_count = payload["retry_count"].float().unsqueeze(-1)
    if (
        pre.ndim != 2
        or pre.shape[-1] != 98
        or correction.shape != (pre.shape[0], 6)
        or retry_count.shape != (pre.shape[0], 1)
    ):
        raise ValueError("pre-probe feature source shape drifted")
    features = torch.cat(
        (
            _role_invariant_observation(pre),
            correction,
            retry_count.clamp(max=5.0) / 5.0,
        ),
        dim=-1,
    )
    if not bool(torch.isfinite(features).all()):
        raise ValueError("pre-probe features contain non-finite values")
    return features if mask is None else features[mask]


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
    }
    if (
        not isinstance(payload, dict)
        or not required.issubset(payload)
        or payload["schema_version"]
        not in {SYMMETRIC_SCHEMA_VERSION, RELEASE_SCHEMA_VERSION}
        or int(payload["seed"]) not in DEVELOPMENT_SEEDS
        or int(payload["runtime_seed"])
        != int(payload["seed"]) + int(payload["seed_stream_offset"])
        or int(payload["observation_dimension"]) != 98
        or int(payload["probe_frames"]) != 1
        or payload["probe_intervention"] != "giver_gripper_open_pulse"
    ):
        raise ValueError(f"incompatible pre-probe dataset: {path}")
    count = int(payload["environment_index"].numel())
    tensors = (
        "pre_observation",
        "post_observation",
        "receiver_correction",
        "retry_count",
        "probe_survived",
        "eventual_full_success",
    )
    if (
        count == 0
        or payload["environment_index"].ndim != 1
        or int(payload["environment_index"].unique().numel()) != count
        or any(int(payload[name].shape[0]) != count for name in tensors)
        or not bool(payload["probe_survived"].bool().all())
    ):
        raise ValueError(f"pre-probe dataset contract drifted: {path}")
    mask = _natural_arm_mask(payload)
    if int(mask.sum().item()) < 2:
        raise ValueError(f"natural arm lacks support: {path}")
    _preprobe_features(payload, mask)
    _causal_features(payload)
    return payload


def _risk_quartiles(
    risk: torch.Tensor,
    success: torch.Tensor,
) -> dict[str, object]:
    cutpoints = torch.quantile(
        risk,
        torch.tensor((0.25, 0.5, 0.75), dtype=risk.dtype),
    )
    strata = torch.bucketize(risk, cutpoints)
    quartiles: dict[str, object] = {}
    failure_rates = []
    for stratum in range(4):
        selected = strata == stratum
        failure_rate = float(
            (~success[selected]).float().mean().item()
        )
        failure_rates.append(failure_rate)
        quartiles[str(stratum)] = {
            "samples": int(selected.sum().item()),
            "mean_predicted_risk": float(
                risk[selected].mean().item()
            ),
            "failures": int((~success[selected]).sum().item()),
            "failure_rate": failure_rate,
        }
    return {
        "cutpoints": cutpoints.tolist(),
        "quartiles": quartiles,
        "strictly_increasing_failure_rate": all(
            left < right
            for left, right in zip(
                failure_rates,
                failure_rates[1:],
            )
        ),
    }


def _run(args: argparse.Namespace) -> int:
    output = Path(args.output).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    teacher_path = (
        Path(args.teacher_checkpoint).expanduser().resolve()
    )
    if output.exists():
        raise FileExistsError(
            f"pre-probe risk report already exists: {output}"
        )
    if checkpoint.exists():
        raise FileExistsError(
            f"pre-probe risk checkpoint already exists: {checkpoint}"
        )
    if output == checkpoint:
        raise ValueError("pre-probe report and checkpoint must differ")
    if args.l2 <= 0.0 or args.calibration_l2 <= 0.0:
        raise ValueError("risk fit coefficients must be positive")
    if args.max_iterations <= 0:
        raise ValueError("maximum iterations must be positive")
    if not 0.5 <= args.minimum_seed_auc <= 1.0:
        raise ValueError("minimum seed AUC must be in [0.5, 1.0]")
    if not 0.5 <= args.minimum_aggregate_auc <= 1.0:
        raise ValueError("minimum aggregate AUC must be in [0.5, 1.0]")

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
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
        raise ValueError("pre-probe datasets repeat a seed stream")
    if {seed for seed, _ in stream_keys} != DEVELOPMENT_SEEDS:
        raise ValueError("pre-probe gate requires all development seeds")

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
        raise ValueError("pre-probe checkpoint provenance drifted")
    base_hash = next(iter(base_hashes))
    candidate_hash = next(iter(candidate_hashes))
    teacher_checkpoint = _load_risk_checkpoint(
        teacher_path,
        source_revision=source_revision,
        base_hash=base_hash,
        candidate_hash=candidate_hash,
    )

    masks = [_natural_arm_mask(payload) for payload in payloads]
    features = torch.cat(
        [
            _preprobe_features(payload, mask)
            for payload, mask in zip(payloads, masks, strict=True)
        ]
    )
    causal_features = torch.cat(
        [
            _causal_features(payload)[mask]
            for payload, mask in zip(payloads, masks, strict=True)
        ]
    )
    success = torch.cat(
        [
            payload["eventual_full_success"].bool()[mask]
            for payload, mask in zip(payloads, masks, strict=True)
        ]
    )
    seed_index = torch.cat(
        [
            torch.full(
                (int(mask.sum().item()),),
                int(payload["seed"]),
                dtype=torch.long,
            )
            for payload, mask in zip(payloads, masks, strict=True)
        ]
    )
    teacher_success_probability = 1.0 - _risk_probability(
        teacher_checkpoint,
        causal_features,
    )
    teacher_metrics = _metrics(
        teacher_success_probability,
        success,
        float(success.float().mean().item()),
    )

    raw_cross_fit_logits = torch.empty(
        success.shape,
        dtype=torch.float32,
    )
    calibrated_cross_fit_probability = torch.empty_like(
        raw_cross_fit_logits
    )
    cross_fit_baseline_probability = torch.empty_like(
        raw_cross_fit_logits
    )
    folds: dict[str, object] = {}
    for held_out_seed in sorted(DEVELOPMENT_SEEDS):
        test = seed_index == held_out_seed
        train = ~test
        inner_features = features[train]
        inner_success = success[train]
        inner_seed_index = seed_index[train]
        inner_oof_logits = torch.empty(
            inner_success.shape,
            dtype=torch.float32,
        )
        inner_fits: dict[str, object] = {}
        for inner_held_out_seed in sorted(
            DEVELOPMENT_SEEDS - {held_out_seed}
        ):
            inner_test = inner_seed_index == inner_held_out_seed
            inner_train = ~inner_test
            inner_state, inner_fit = _fit_linear_model(
                inner_features[inner_train],
                inner_success[inner_train],
                l2=args.l2,
                max_iterations=args.max_iterations,
            )
            inner_oof_logits[inner_test] = _linear_logits(
                inner_state,
                inner_features[inner_test],
            )
            inner_fits[str(inner_held_out_seed)] = inner_fit
        calibrator = _fit_platt_calibrator(
            inner_oof_logits,
            inner_success,
            l2=args.calibration_l2,
            max_iterations=args.max_iterations,
        )
        state, fit = _fit_linear_model(
            features[train],
            success[train],
            l2=args.l2,
            max_iterations=args.max_iterations,
        )
        logits = _linear_logits(state, features[test])
        raw_cross_fit_logits[test] = logits
        calibrated_cross_fit_probability[test] = (
            _apply_platt_calibrator(logits, calibrator)
        )
        baseline_probability = float(
            success[train].float().mean().item()
        )
        cross_fit_baseline_probability[test] = baseline_probability
        folds[str(held_out_seed)] = {
            "train_samples": int(train.sum().item()),
            "test_samples": int(test.sum().item()),
            "fit_uses_held_out_seed": False,
            "fit": fit,
            "nested_calibration": {
                "fit_uses_held_out_seed": False,
                "inner_fits": inner_fits,
                "calibrator": calibrator,
            },
            "metrics": _metrics(
                calibrated_cross_fit_probability[test],
                success[test],
                baseline_probability,
            ),
            "teacher_metrics": _metrics(
                teacher_success_probability[test],
                success[test],
                baseline_probability,
            ),
        }

    raw_cross_fit_probability = torch.sigmoid(
        raw_cross_fit_logits
    )
    raw_aggregate = _metrics(
        raw_cross_fit_probability,
        success,
        cross_fit_baseline_probability,
    )
    aggregate = _metrics(
        calibrated_cross_fit_probability,
        success,
        cross_fit_baseline_probability,
    )
    ranking_gate_passed = bool(
        aggregate["roc_auc"] >= args.minimum_aggregate_auc
        and all(
            fold["metrics"]["roc_auc"] >= args.minimum_seed_auc
            for fold in folds.values()
        )
    )
    calibration_gate_passed = bool(
        aggregate["brier"] < aggregate["baseline_brier"]
        and all(
            fold["metrics"]["brier"]
            < fold["metrics"]["baseline_brier"]
            for fold in folds.values()
        )
    )
    cross_fit_risk = 1.0 - calibrated_cross_fit_probability
    risk_quartiles = _risk_quartiles(cross_fit_risk, success)
    signal_gate_passed = bool(
        ranking_gate_passed
        and calibration_gate_passed
        and risk_quartiles["strictly_increasing_failure_rate"]
    )

    final_state, final_fit = _fit_linear_model(
        features,
        success,
        l2=args.l2,
        max_iterations=args.max_iterations,
    )
    final_calibrator = _fit_platt_calibrator(
        raw_cross_fit_logits,
        success,
        l2=args.calibration_l2,
        max_iterations=args.max_iterations,
    )
    final_probability = _apply_platt_calibrator(
        _linear_logits(final_state, features),
        final_calibrator,
    )
    activation_threshold = float(
        torch.quantile(1.0 - final_probability, 0.75).item()
    )

    dataset_provenance = []
    for path, payload, mask in zip(
        paths,
        payloads,
        masks,
        strict=True,
    ):
        dataset_provenance.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "schema_version": payload["schema_version"],
                "seed": int(payload["seed"]),
                "seed_stream_offset": int(
                    payload["seed_stream_offset"]
                ),
                "runtime_seed": int(payload["runtime_seed"]),
                "source_samples": int(
                    payload["environment_index"].numel()
                ),
                "natural_arm_samples": int(mask.sum().item()),
                "natural_arm_successful": int(
                    payload["eventual_full_success"][mask].sum().item()
                ),
            }
        )
    checkpoint_payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "source_revision": source_revision,
        "teacher_checkpoint_sha256": _sha256(teacher_path),
        "control_scope": "pre_probe_risk_stratification_only",
        "motion_control_authorized": False,
        "base_checkpoint_sha256": base_hash,
        "receiver_candidate_checkpoint_sha256": candidate_hash,
        "feature_contract": {
            "kind": (
                "role_invariant_pre_observation_plus_correction"
            ),
            "dimension": int(features.shape[-1]),
            "uses_probe_response": False,
            "uses_future_information": False,
        },
        "model": {
            "kind": "standardized_linear_logistic_risk_network",
            **final_state,
            "calibration": final_calibrator,
            "l2": args.l2,
            "fit": final_fit,
        },
        "trial_allocation": {
            "kind": "top_risk_quartile",
            "activation_threshold": activation_threshold,
            "maximum_activation_fraction": 0.25,
            "control_action": "normal_probe_then_immediate_release",
            "candidate_action": (
                "existing_bounded_retry_before_giver_probe"
            ),
            "candidate_motion_authorized": False,
        },
        "training_datasets": dataset_provenance,
        "cross_fit_gate": {
            "ranking_gate_passed": ranking_gate_passed,
            "calibration_gate_passed": calibration_gate_passed,
            "signal_gate_passed": signal_gate_passed,
            "raw_aggregate": raw_aggregate,
            "calibrated_aggregate": aggregate,
        },
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload, checkpoint)
    checkpoint_hash = _sha256(checkpoint)

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_revision": source_revision,
        "status": (
            "preprobe_risk_model_cross_seed"
            if signal_gate_passed
            else "preprobe_risk_model_gate_failed"
        ),
        "decision": {
            "signal_gate_passed": signal_gate_passed,
            "ranking_gate_passed": ranking_gate_passed,
            "calibration_gate_passed": calibration_gate_passed,
            "risk_quartile_gate_passed": risk_quartiles[
                "strictly_increasing_failure_rate"
            ],
            "authorized_for_fresh_randomized_trial_allocation": (
                signal_gate_passed
            ),
            "motion_control_authorized": False,
            "main_policy_replacement_authorized": False,
        },
        "samples": int(success.numel()),
        "failures": int((~success).sum().item()),
        "feature_contract": checkpoint_payload["feature_contract"],
        "teacher_checkpoint": {
            "path": str(teacher_path),
            "sha256": _sha256(teacher_path),
            "metrics": teacher_metrics,
        },
        "cross_fit": {
            "strategy": "leave_one_seed_out_nested_calibration",
            "raw_aggregate": raw_aggregate,
            "aggregate": aggregate,
            "folds": folds,
            "risk_quartiles": risk_quartiles,
        },
        "final_model": {
            "checkpoint": str(checkpoint),
            "sha256": checkpoint_hash,
            "activation_threshold": activation_threshold,
            "fit": final_fit,
        },
        "datasets": dataset_provenance,
        "next_stage": {
            "experiment": (
                "fresh_randomized_preprobe_retry_vs_normal_probe"
            ),
            "eligible_cohort": (
                "fixed_top_quartile_preprobe_risk_threshold"
            ),
            "behavior_propensity": 0.5,
            "promotion_authorized": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["decision"], indent=2, sort_keys=True))
    print(f"[DrAnmar] Pre-probe risk report: {output}")
    print(f"[DrAnmar] Pre-probe risk checkpoint: {checkpoint}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
    )
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument(
        "--calibration_l2",
        type=float,
        default=0.01,
    )
    parser.add_argument("--max_iterations", type=int, default=300)
    parser.add_argument("--minimum_seed_auc", type=float, default=0.70)
    parser.add_argument(
        "--minimum_aggregate_auc",
        type=float,
        default=0.75,
    )
    parser.add_argument("--seed", type=int, default=4104737)
    return parser


def main() -> int:
    return _run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
