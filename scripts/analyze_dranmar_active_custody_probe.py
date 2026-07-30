#!/usr/bin/env python3
"""Audit whether active-custody probe transitions contain cross-seed signal."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional


SCHEMA_VERSION = "dranmar-receiver-active-custody-probe-dataset-1.0"
REPORT_SCHEMA_VERSION = "dranmar-active-custody-signal-audit-1.1"
CHECKPOINT_SCHEMA_VERSION = "dranmar-active-custody-risk-model-1.0"
DEVELOPMENT_SEEDS = {104729, 130363, 196613}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _select_role(
    observation: torch.Tensor,
    robot_1_slice: slice,
    robot_2_slice: slice,
    role_is_robot_1: torch.Tensor,
) -> torch.Tensor:
    return torch.where(
        role_is_robot_1.unsqueeze(-1),
        observation[:, robot_1_slice],
        observation[:, robot_2_slice],
    )


def _role_invariant_observation(
    observation: torch.Tensor,
) -> torch.Tensor:
    giver_is_robot_1 = observation[:, 82] > 0.5
    receiver_is_robot_1 = ~giver_is_robot_1

    def receiver(robot_1_slice: slice, robot_2_slice: slice) -> torch.Tensor:
        return _select_role(
            observation,
            robot_1_slice,
            robot_2_slice,
            receiver_is_robot_1,
        )

    def giver(robot_1_slice: slice, robot_2_slice: slice) -> torch.Tensor:
        return _select_role(
            observation,
            robot_1_slice,
            robot_2_slice,
            giver_is_robot_1,
        )

    return torch.cat(
        (
            receiver(slice(0, 8), slice(16, 24)),
            receiver(slice(8, 16), slice(24, 32)),
            giver(slice(0, 8), slice(16, 24)),
            giver(slice(8, 16), slice(24, 32)),
            receiver(slice(32, 39), slice(39, 46)),
            receiver(slice(46, 53), slice(53, 60)),
            observation[:, 60:66],
            receiver(slice(66, 68), slice(68, 70)),
            giver(slice(66, 68), slice(68, 70)),
            observation[:, 70:77],
            observation[:, 77:82],
            receiver(slice(84, 91), slice(91, 98)),
            giver(slice(84, 91), slice(91, 98)),
        ),
        dim=-1,
    )


def _causal_features(payload: dict[str, object]) -> torch.Tensor:
    pre = payload["pre_observation"].float()
    post = payload["post_observation"].float()
    if (
        pre.ndim != 2
        or post.shape != pre.shape
        or pre.shape[-1] != 98
    ):
        raise ValueError("active-custody observation shape drifted")
    pre_role = _role_invariant_observation(pre)
    post_role = _role_invariant_observation(post)
    correction = payload["receiver_correction"].float()
    retry_count = payload["retry_count"].float().unsqueeze(-1)
    features = torch.cat(
        (
            pre_role,
            post_role - pre_role,
            correction,
            retry_count.clamp(max=5.0) / 5.0,
        ),
        dim=-1,
    )
    if not bool(torch.isfinite(features).all()):
        raise ValueError("active-custody features contain non-finite values")
    return features


def _roc_auc(probability: torch.Tensor, target: torch.Tensor) -> float:
    positive = probability[target]
    negative = probability[~target]
    if positive.numel() == 0 or negative.numel() == 0:
        raise ValueError("ROC AUC requires both outcomes")
    difference = positive.unsqueeze(1) - negative.unsqueeze(0)
    return float(
        (
            (difference > 0.0).float()
            + 0.5 * (difference == 0.0).float()
        )
        .mean()
        .item()
    )


def _metrics(
    probability: torch.Tensor,
    target: torch.Tensor,
    baseline_probability: float | torch.Tensor,
) -> dict[str, float | int]:
    target_float = target.float()
    clipped = probability.clamp(1.0e-6, 1.0 - 1.0e-6)
    if isinstance(baseline_probability, torch.Tensor):
        baseline = baseline_probability
    else:
        baseline = torch.full_like(probability, baseline_probability)
    if baseline.shape != probability.shape:
        raise ValueError("baseline probability shape drifted")
    failure = ~target
    audit_count = max(1, int(round(0.2 * probability.numel())))
    highest_risk = probability.argsort()[:audit_count]
    captured_failures = int(failure[highest_risk].sum().item())
    return {
        "samples": int(target.numel()),
        "successful": int(target.sum().item()),
        "success_rate": float(target_float.mean().item()),
        "roc_auc": _roc_auc(probability, target),
        "brier": float(
            functional.mse_loss(probability, target_float).item()
        ),
        "baseline_brier": float(
            functional.mse_loss(
                baseline,
                target_float,
            ).item()
        ),
        "log_loss": float(
            functional.binary_cross_entropy(
                clipped,
                target_float,
            ).item()
        ),
        "failures": int(failure.sum().item()),
        "failures_captured_in_highest_risk_20_percent": (
            captured_failures
        ),
        "failure_capture_rate_at_20_percent": (
            captured_failures / int(failure.sum().item())
            if bool(failure.any())
            else 0.0
        ),
    }


def _fit_linear_model(
    train_features: torch.Tensor,
    train_target: torch.Tensor,
    *,
    l2: float,
    max_iterations: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0).clamp_min(1.0e-6)
    normalized_train = (train_features - mean) / std
    model = nn.Linear(train_features.shape[-1], 1)
    nn.init.zeros_(model.weight)
    initial_logit = torch.logit(
        train_target.float().mean().clamp(0.01, 0.99)
    )
    nn.init.constant_(model.bias, float(initial_logit.item()))
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=max_iterations,
        tolerance_grad=1.0e-8,
        tolerance_change=1.0e-10,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        logit = model(normalized_train).squeeze(-1)
        loss = functional.binary_cross_entropy_with_logits(
            logit,
            train_target.float(),
        )
        loss = loss + l2 * model.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        train_probability = torch.sigmoid(
            model(normalized_train).squeeze(-1)
        )
    state = {
        "feature_mean": mean.detach().clone(),
        "feature_std": std.detach().clone(),
        "weight": model.weight.detach().squeeze(0).clone(),
        "bias": model.bias.detach().squeeze(0).clone(),
    }
    return state, {
        "train_brier": float(
            functional.mse_loss(
                train_probability,
                train_target.float(),
            ).item()
        ),
        "weight_l2_norm": float(model.weight.norm().item()),
    }


def _linear_logits(
    state: dict[str, torch.Tensor],
    features: torch.Tensor,
) -> torch.Tensor:
    normalized = (
        features - state["feature_mean"]
    ) / state["feature_std"]
    return normalized @ state["weight"] + state["bias"]


def _fit_platt_calibrator(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    l2: float,
    max_iterations: int,
) -> dict[str, float]:
    if (
        logits.ndim != 1
        or target.shape != logits.shape
        or not bool(target.any())
        or not bool((~target).any())
    ):
        raise ValueError("Platt calibration requires two-class 1D logits")
    log_slope = nn.Parameter(torch.zeros(()))
    intercept = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS(
        (log_slope, intercept),
        lr=1.0,
        max_iter=max_iterations,
        tolerance_grad=1.0e-8,
        tolerance_change=1.0e-10,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        calibrated_logit = (
            torch.exp(log_slope) * logits + intercept
        )
        loss = functional.binary_cross_entropy_with_logits(
            calibrated_logit,
            target.float(),
        )
        loss = loss + l2 * log_slope.square()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        slope = float(torch.exp(log_slope).item())
        fitted_intercept = float(intercept.item())
        probability = torch.sigmoid(
            slope * logits + fitted_intercept
        )
    return {
        "kind": "positive_slope_platt_scaling",
        "slope": slope,
        "intercept": fitted_intercept,
        "l2": l2,
        "train_brier": float(
            functional.mse_loss(
                probability,
                target.float(),
            ).item()
        ),
    }


def _apply_platt_calibrator(
    logits: torch.Tensor,
    calibrator: dict[str, float],
) -> torch.Tensor:
    return torch.sigmoid(
        calibrator["slope"] * logits + calibrator["intercept"]
    )


def _load_dataset(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "schema_version",
        "seed",
        "seed_stream_offset",
        "runtime_seed",
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
        or payload.get("schema_version") != SCHEMA_VERSION
        or not required.issubset(payload)
        or int(payload["seed"]) not in DEVELOPMENT_SEEDS
        or int(payload["runtime_seed"])
        != int(payload["seed"]) + int(payload["seed_stream_offset"])
        or int(payload["observation_dimension"]) != 98
        or int(payload["probe_frames"]) != 1
        or payload["probe_intervention"]
        != "giver_gripper_open_pulse"
    ):
        raise ValueError(f"incompatible active-custody dataset: {path}")
    count = int(payload["environment_index"].numel())
    if (
        count == 0
        or torch.unique(payload["environment_index"]).numel() != count
    ):
        raise ValueError(f"dataset environment indices drifted: {path}")
    for name in (
        "pre_observation",
        "post_observation",
        "receiver_correction",
        "retry_count",
        "probe_survived",
        "eventual_full_success",
    ):
        if int(payload[name].shape[0]) != count:
            raise ValueError(f"dataset sample count drifted for {name}: {path}")
    return payload


def _run(args: argparse.Namespace) -> int:
    output = Path(args.output).expanduser().resolve()
    checkpoint = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else output.with_suffix(".pt")
    )
    if output.exists():
        raise FileExistsError(f"signal report already exists: {output}")
    if checkpoint == output:
        raise ValueError("risk checkpoint and signal report must differ")
    if checkpoint.exists():
        raise FileExistsError(
            f"risk checkpoint already exists: {checkpoint}"
        )
    if args.l2 <= 0.0:
        raise ValueError("L2 coefficient must be positive")
    if args.calibration_l2 <= 0.0:
        raise ValueError("calibration L2 coefficient must be positive")
    if args.max_iterations <= 0:
        raise ValueError("maximum iterations must be positive")
    if not 0.5 <= args.minimum_seed_auc <= 1.0:
        raise ValueError("minimum seed AUC must be in [0.5, 1.0]")
    if not 0.5 <= args.minimum_aggregate_auc <= 1.0:
        raise ValueError("minimum aggregate AUC must be in [0.5, 1.0]")
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    paths = [Path(value).expanduser().resolve() for value in args.dataset]
    payloads = [_load_dataset(path) for path in paths]
    stream_keys = [
        (int(payload["seed"]), int(payload["seed_stream_offset"]))
        for payload in payloads
    ]
    if len(set(stream_keys)) != len(stream_keys):
        raise ValueError("active-custody datasets repeat a seed stream")
    observed_seeds = {seed for seed, _ in stream_keys}
    if observed_seeds != DEVELOPMENT_SEEDS:
        raise ValueError(
            "signal audit requires all three development seeds"
        )
    base_hashes = {
        str(payload["base_checkpoint_sha256"]) for payload in payloads
    }
    if len(base_hashes) != 1:
        raise ValueError("active-custody base checkpoint drifted")
    if _SHA256_PATTERN.fullmatch(next(iter(base_hashes))) is None:
        raise ValueError("active-custody base checkpoint hash is invalid")
    candidate_hashes = {
        str(payload["receiver_candidate_checkpoint_sha256"])
        for payload in payloads
    }
    if len(candidate_hashes) != 1:
        raise ValueError("active-custody receiver checkpoint drifted")
    if _SHA256_PATTERN.fullmatch(next(iter(candidate_hashes))) is None:
        raise ValueError(
            "active-custody receiver checkpoint hash is invalid"
        )

    features = torch.cat([_causal_features(payload) for payload in payloads])
    target = torch.cat(
        [payload["eventual_full_success"].bool() for payload in payloads]
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
    source_revision = _source_revision()
    raw_cross_fit_logits = torch.empty_like(
        target,
        dtype=torch.float32,
    )
    calibrated_cross_fit_probability = torch.empty_like(
        target,
        dtype=torch.float32,
    )
    cross_fit_baseline_probability = torch.empty_like(
        target,
        dtype=torch.float32,
    )
    folds: dict[str, object] = {}
    for held_out_seed in sorted(DEVELOPMENT_SEEDS):
        test = seed_index == held_out_seed
        train = ~test
        inner_features = features[train]
        inner_target = target[train]
        inner_seed_index = seed_index[train]
        inner_oof_logits = torch.empty_like(
            inner_target,
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
                inner_target[inner_train],
                l2=args.l2,
                max_iterations=args.max_iterations,
            )
            inner_logits = _linear_logits(
                inner_state,
                inner_features[inner_test],
            )
            inner_oof_logits[inner_test] = inner_logits
            inner_fits[str(inner_held_out_seed)] = {
                "train_samples": int(inner_train.sum().item()),
                "calibration_samples": int(
                    inner_test.sum().item()
                ),
                "fit": inner_fit,
            }
        calibrator = _fit_platt_calibrator(
            inner_oof_logits,
            inner_target,
            l2=args.calibration_l2,
            max_iterations=args.max_iterations,
        )
        outer_state, fit = _fit_linear_model(
            features[train],
            target[train],
            l2=args.l2,
            max_iterations=args.max_iterations,
        )
        raw_logits = _linear_logits(outer_state, features[test])
        raw_probability = torch.sigmoid(raw_logits)
        calibrated_probability = _apply_platt_calibrator(
            raw_logits,
            calibrator,
        )
        raw_cross_fit_logits[test] = raw_logits
        calibrated_cross_fit_probability[test] = (
            calibrated_probability
        )
        baseline_probability = float(target[train].float().mean().item())
        cross_fit_baseline_probability[test] = baseline_probability
        raw_metrics = _metrics(
            raw_probability,
            target[test],
            baseline_probability,
        )
        calibrated_metrics = _metrics(
            calibrated_probability,
            target[test],
            baseline_probability,
        )
        folds[str(held_out_seed)] = {
            "train_samples": int(train.sum().item()),
            "baseline_probability": baseline_probability,
            "fit": fit,
            "nested_calibration": {
                "fit_uses_held_out_seed": False,
                "inner_strategy": (
                    "leave_one_seed_out_within_outer_training_seeds"
                ),
                "inner_fits": inner_fits,
                "calibrator": calibrator,
            },
            "raw_metrics": raw_metrics,
            "metrics": calibrated_metrics,
            "ranking_preserved_by_positive_slope_calibration": bool(
                abs(
                    raw_metrics["roc_auc"]
                    - calibrated_metrics["roc_auc"]
                )
                < 1.0e-7
            ),
        }

    raw_cross_fit_probability = torch.sigmoid(raw_cross_fit_logits)
    raw_aggregate = _metrics(
        raw_cross_fit_probability,
        target,
        cross_fit_baseline_probability,
    )
    aggregate = _metrics(
        calibrated_cross_fit_probability,
        target,
        cross_fit_baseline_probability,
    )
    per_seed_ranking_pass = all(
        fold["metrics"]["roc_auc"] >= args.minimum_seed_auc
        for fold in folds.values()
    )
    per_seed_calibration_pass = all(
        fold["metrics"]["brier"]
        < fold["metrics"]["baseline_brier"]
        for fold in folds.values()
    )
    ranking_gate_passed = bool(
        aggregate["roc_auc"] >= args.minimum_aggregate_auc
        and per_seed_ranking_pass
    )
    calibration_gate_passed = bool(
        aggregate["brier"] < aggregate["baseline_brier"]
        and per_seed_calibration_pass
    )
    signal_gate_passed = bool(
        ranking_gate_passed and calibration_gate_passed
    )
    if signal_gate_passed:
        status = "temporal_risk_model_calibrated_cross_seed"
        reason = (
            "Nested cross-fitted ranking and calibration passed on every "
            "development seed. Preserve this as the leading custody-risk "
            "model and use it to stratify randomized bounded custody "
            "interventions before learning motion control."
        )
        model_disposition = (
            "preserved_as_leading_custody_risk_model"
        )
    elif ranking_gate_passed:
        status = "temporal_ranking_signal_calibration_gate_failed"
        reason = (
            "Ranking signal replicated across seeds, but calibrated "
            "probabilities did not beat the base-rate predictor on every "
            "seed. Preserve the ranking model for analysis and recalibrate "
            "before using probability thresholds."
        )
        model_disposition = (
            "preserved_as_cross_seed_ranking_model"
        )
    else:
        status = "temporal_ranking_signal_not_cross_seed"
        reason = (
            "Do not train a custody controller from these transitions; "
            "cross-seed ranking signal is insufficient."
        )
        model_disposition = "retained_as_nonpromoted_experiment"

    final_state, final_fit = _fit_linear_model(
        features,
        target,
        l2=args.l2,
        max_iterations=args.max_iterations,
    )
    final_calibrator = _fit_platt_calibrator(
        raw_cross_fit_logits,
        target,
        l2=args.calibration_l2,
        max_iterations=args.max_iterations,
    )
    dataset_provenance = [
        {
            "path": str(path),
            "sha256": _sha256(path),
            "seed": int(payload["seed"]),
            "seed_stream_offset": int(
                payload["seed_stream_offset"]
            ),
            "runtime_seed": int(payload["runtime_seed"]),
            "samples": int(payload["environment_index"].numel()),
            "successful": int(
                payload["eventual_full_success"].sum().item()
            ),
        }
        for path, payload in zip(paths, payloads, strict=True)
    ]
    risk_checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "source_revision": source_revision,
        "control_scope": "risk_stratification_only",
        "motion_control_authorized": False,
        "base_checkpoint_sha256": next(iter(base_hashes)),
        "receiver_candidate_checkpoint_sha256": next(
            iter(candidate_hashes)
        ),
        "feature_contract": {
            "kind": "causal_role_invariant_pre_plus_delta",
            "dimension": int(features.shape[-1]),
            "uses_future_information": False,
        },
        "model": {
            "kind": "standardized_linear_logistic_risk_network",
            **final_state,
            "calibration": final_calibrator,
            "l2": args.l2,
            "fit": final_fit,
        },
        "training_datasets": dataset_provenance,
        "cross_fit_gate": {
            "status": status,
            "ranking_gate_passed": ranking_gate_passed,
            "calibration_gate_passed": calibration_gate_passed,
            "signal_gate_passed": signal_gate_passed,
            "raw_aggregate": raw_aggregate,
            "calibrated_aggregate": aggregate,
        },
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(risk_checkpoint, checkpoint)
    checkpoint_hash = _sha256(checkpoint)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_revision": source_revision,
        "status": status,
        "decision": {
            "signal_gate_passed": signal_gate_passed,
            "ranking_gate_passed": ranking_gate_passed,
            "calibration_gate_passed": calibration_gate_passed,
            "model_disposition": model_disposition,
            "risk_model_authorized_for_stratification": bool(
                ranking_gate_passed
            ),
            "motion_control_authorized": False,
            "control_policy_authorized": False,
            "reason": reason,
        },
        "feature_contract": {
            "kind": "causal_role_invariant_pre_plus_delta",
            "dimension": int(features.shape[-1]),
            "uses_future_information": False,
            "label": "eventual_retained_full_success",
        },
        "model": {
            "kind": (
                "l2_regularized_logistic_risk_network_with_nested_"
                "platt_calibration"
            ),
            "l2": args.l2,
            "l2_penalty": "coefficient_times_weight_squared_sum",
            "calibration_l2": args.calibration_l2,
            "calibration_strategy": (
                "nested_leave_one_physics_seed_out_positive_slope_platt"
            ),
            "max_iterations": args.max_iterations,
            "seed": args.seed,
        },
        "gate": {
            "minimum_aggregate_auc": args.minimum_aggregate_auc,
            "minimum_seed_auc": args.minimum_seed_auc,
            "requires_brier_improvement_on_every_seed": True,
        },
        "risk_checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_hash,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "control_scope": "risk_stratification_only",
        },
        "datasets": dataset_provenance,
        "folds": folds,
        "raw_aggregate": raw_aggregate,
        "aggregate": aggregate,
        "deployment_fit": {
            "fit": final_fit,
            "calibration": final_calibrator,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a leave-one-seed-out signal audit on active-custody probes"
        )
    )
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--l2", type=float, default=1.0e-3)
    parser.add_argument("--calibration_l2", type=float, default=1.0e-4)
    parser.add_argument("--max_iterations", type=int, default=200)
    parser.add_argument("--minimum_aggregate_auc", type=float, default=0.6)
    parser.add_argument("--minimum_seed_auc", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=104729)
    return parser


def main(argv: list[str]) -> int:
    return _run(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
