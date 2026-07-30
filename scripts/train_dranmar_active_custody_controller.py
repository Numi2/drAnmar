#!/usr/bin/env python3
"""Train and cross-seed gate a bounded active-custody controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional


DATASET_SCHEMA_VERSION = "dranmar-receiver-active-custody-intervention-dataset-1.0"
RISK_SCHEMA_VERSION = "dranmar-active-custody-risk-model-1.0"
CHECKPOINT_SCHEMA_VERSION = "dranmar-active-custody-controller-candidate-1.0"
REPORT_SCHEMA_VERSION = "dranmar-active-custody-controller-gate-1.0"
DEVELOPMENT_SEEDS = {104729, 130363, 196613}
ACTION_IDS = torch.tensor((-1, 0, 1), dtype=torch.long)
NO_OP_INDEX = 1
ONE_SIDED_95_Z = 1.6448536269514722
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_ACTION_SEMANTICS = {
    -1: "opposite_force_centering_pulse",
    0: "exact_no_op_hold_closed",
    1: "force_centering_pulse",
}
_UNSAFE_TERMINATIONS = {
    "excessive_object_force",
    "protected_surface_force",
}


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


def _is_ancestor(revision: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, descendant],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


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

    def receiver(
        robot_1_slice: slice,
        robot_2_slice: slice,
    ) -> torch.Tensor:
        return _select_role(
            observation,
            robot_1_slice,
            robot_2_slice,
            receiver_is_robot_1,
        )

    def giver(
        robot_1_slice: slice,
        robot_2_slice: slice,
    ) -> torch.Tensor:
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
    if pre.ndim != 2 or post.shape != pre.shape or pre.shape[-1] != 98:
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


def _action_indices(action_ids: torch.Tensor) -> torch.Tensor:
    indices = action_ids.long() + 1
    if indices.ndim != 1 or bool((indices < 0).any()) or bool((indices > 2).any()):
        raise ValueError("intervention action id left {-1, 0, 1}")
    return indices


def _termination_indicator(
    payload: dict[str, object],
    names: set[str],
) -> torch.Tensor:
    termination_names = list(payload["termination_names"])
    missing = names - set(termination_names)
    if missing:
        raise ValueError(
            "intervention dataset is missing termination flags: " + ", ".join(sorted(missing))
        )
    indices = [termination_names.index(name) for name in sorted(names)]
    flags = payload["eventual_termination_flags"].bool()
    return flags[:, indices].any(dim=-1)


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
        "intervention_frames",
        "intervention_action_limit",
        "intervention_action_semantics",
        "assigned_action_id",
        "assigned_action_probability",
        "applied_receiver_action",
        "force_centering_direction",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"incomplete intervention dataset: {path}")
    try:
        action_semantics = {
            int(key): str(value)
            for key, value in dict(payload["intervention_action_semantics"]).items()
        }
    except (TypeError, ValueError):
        raise ValueError(f"intervention action semantics drifted: {path}") from None
    if (
        payload["schema_version"] != DATASET_SCHEMA_VERSION
        or int(payload["seed"]) not in DEVELOPMENT_SEEDS
        or int(payload["runtime_seed"]) != int(payload["seed"]) + int(payload["seed_stream_offset"])
        or int(payload["observation_dimension"]) != 98
        or int(payload["probe_frames"]) != 1
        or payload["probe_intervention"] != "giver_gripper_open_pulse"
        or payload["randomization"] != "seeded_hash_uniform_three_arm"
        or int(payload["intervention_frames"]) != 1
        or action_semantics != _ACTION_SEMANTICS
    ):
        raise ValueError(f"incompatible intervention dataset: {path}")

    environment_index = payload["environment_index"].long()
    count = int(environment_index.numel())
    if (
        count == 0
        or environment_index.ndim != 1
        or int(environment_index.unique().numel()) != count
        or int(environment_index.min()) < 0
        or int(environment_index.max()) >= int(payload["num_envs"])
    ):
        raise ValueError(f"dataset environment indices drifted: {path}")
    tensors = (
        "pre_observation",
        "post_observation",
        "receiver_correction",
        "retry_count",
        "probe_survived",
        "eventual_full_success",
        "eventual_termination_flags",
        "assigned_action_id",
        "assigned_action_probability",
        "applied_receiver_action",
        "force_centering_direction",
    )
    if any(int(payload[name].shape[0]) != count for name in tensors):
        raise ValueError(f"dataset sample count drifted: {path}")
    if (
        payload["eventual_termination_flags"].ndim != 2
        or payload["eventual_termination_flags"].shape[1] != len(payload["termination_names"])
        or payload["applied_receiver_action"].shape != (count, 7)
        or not bool(payload["probe_survived"].bool().all())
    ):
        raise ValueError(f"intervention outcome contract drifted: {path}")

    action_id = payload["assigned_action_id"].long()
    if not torch.equal(action_id.unique().sort().values, ACTION_IDS):
        raise ValueError(f"dataset lacks three-arm action support: {path}")
    probability = payload["assigned_action_probability"].float()
    if not torch.allclose(
        probability,
        torch.full_like(probability, 1.0 / 3.0),
        atol=1.0e-7,
        rtol=0.0,
    ):
        raise ValueError(f"dataset behavior propensity drifted: {path}")
    action_limit = float(payload["intervention_action_limit"])
    if not 0.0 < action_limit <= 0.0025:
        raise ValueError(f"intervention action limit drifted: {path}")
    action = payload["applied_receiver_action"].float()
    direction = payload["force_centering_direction"].float()
    expected_translation = action_id.float() * direction * action_limit
    if (
        not bool(torch.isfinite(action).all())
        or not bool(torch.isfinite(direction).all())
        or not bool((direction.abs() == 1.0).all())
        or not torch.allclose(
            action[:, 2],
            expected_translation,
            atol=1.0e-7,
            rtol=0.0,
        )
        or not torch.allclose(
            action[:, :2],
            torch.zeros_like(action[:, :2]),
        )
        or not torch.allclose(
            action[:, 3:6],
            torch.zeros_like(action[:, 3:6]),
        )
        or not torch.equal(
            action[:, 6],
            torch.full_like(action[:, 6], -1.0),
        )
    ):
        raise ValueError(f"applied intervention action drifted: {path}")
    _causal_features(payload)
    _termination_indicator(payload, _UNSAFE_TERMINATIONS)
    return payload


def _load_risk_checkpoint(
    path: Path,
    *,
    source_revision: str,
    base_hash: str,
    candidate_hash: str,
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != RISK_SCHEMA_VERSION
        or payload.get("control_scope") != "risk_stratification_only"
        or payload.get("motion_control_authorized") is not False
        or not payload.get("cross_fit_gate", {}).get("signal_gate_passed")
        or payload.get("base_checkpoint_sha256") != base_hash
        or payload.get("receiver_candidate_checkpoint_sha256") != candidate_hash
        or payload.get("feature_contract", {}).get("kind") != "causal_role_invariant_pre_plus_delta"
    ):
        raise ValueError("risk checkpoint is not source-compatible")
    risk_source = str(payload.get("source_revision", ""))
    if re.fullmatch(r"[0-9a-f]{40}", risk_source) is None or not _is_ancestor(
        risk_source, source_revision
    ):
        raise ValueError("risk checkpoint source is not an ancestor")
    model = payload.get("model")
    if not isinstance(model, dict):
        raise ValueError("risk checkpoint model is missing")
    required = {
        "feature_mean",
        "feature_std",
        "weight",
        "bias",
        "calibration",
    }
    if not required.issubset(model):
        raise ValueError("risk checkpoint model is incomplete")
    return payload


def _risk_probability(
    checkpoint: dict[str, object],
    features: torch.Tensor,
) -> torch.Tensor:
    model = checkpoint["model"]
    mean = model["feature_mean"].float()
    std = model["feature_std"].float()
    weight = model["weight"].float()
    bias = model["bias"].float()
    if (
        mean.shape != (features.shape[-1],)
        or std.shape != mean.shape
        or weight.shape != mean.shape
        or bias.numel() != 1
    ):
        raise ValueError("risk checkpoint feature dimension drifted")
    logit = ((features - mean) / std) @ weight + bias
    calibration = model["calibration"]
    success_probability = torch.sigmoid(
        float(calibration["slope"]) * logit + float(calibration["intercept"])
    )
    return 1.0 - success_probability


def _risk_strata(
    risk: torch.Tensor,
    cutpoints: torch.Tensor,
) -> torch.Tensor:
    if cutpoints.shape != (3,):
        raise ValueError("risk quartile cutpoints drifted")
    return torch.bucketize(risk, cutpoints)


def _risk_cutpoints(risk: torch.Tensor) -> torch.Tensor:
    return torch.quantile(
        risk,
        torch.tensor((0.25, 0.5, 0.75), dtype=risk.dtype),
    )


def _balanced_weights(
    seed_index: torch.Tensor,
    strata: torch.Tensor,
    action_index: torch.Tensor,
) -> torch.Tensor:
    weights = torch.zeros(action_index.shape, dtype=torch.float32)
    for seed in seed_index.unique():
        for stratum in strata.unique():
            for action in range(3):
                cell = (seed_index == seed) & (strata == stratum) & (action_index == action)
                count = int(cell.sum())
                if count:
                    weights[cell] = 1.0 / count
    if not bool((weights > 0.0).all()):
        raise ValueError("risk-stratified action cell lost support")
    return weights * (weights.numel() / weights.sum())


def _fit_outcome_network(
    features: torch.Tensor,
    action_index: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    l2: float,
    max_iterations: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp_min(1.0e-6)
    normalized = (features - mean) / std
    if bool(target.all()) or not bool(target.any()):
        constant = target.float().mean()
        state = {
            "feature_mean": mean.detach().clone(),
            "feature_std": std.detach().clone(),
            "weight": torch.zeros(
                (3, features.shape[-1]),
                dtype=features.dtype,
            ),
            "bias": torch.zeros(3, dtype=features.dtype),
            "constant_probability": constant.detach().clone(),
        }
        return state, {
            "logged_action_brier": float(
                functional.mse_loss(
                    torch.full_like(target.float(), constant),
                    target.float(),
                ).item()
            ),
            "weight_l2_norm": 0.0,
            "constant_outcome": float(constant.item()),
        }
    model = nn.Linear(features.shape[-1], 3)
    nn.init.zeros_(model.weight)
    with torch.no_grad():
        for action in range(3):
            selected = action_index == action
            rate = target[selected].float().mean().clamp(0.01, 0.99)
            model.bias[action] = torch.logit(rate)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=max_iterations,
        tolerance_grad=1.0e-8,
        tolerance_change=1.0e-10,
        line_search_fn="strong_wolfe",
    )
    rows = torch.arange(features.shape[0])

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        selected_logit = model(normalized)[rows, action_index]
        losses = functional.binary_cross_entropy_with_logits(
            selected_logit,
            target.float(),
            reduction="none",
        )
        loss = (weights * losses).sum() / weights.sum()
        loss = loss + l2 * model.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        probability = torch.sigmoid(model(normalized))
        logged_probability = probability[rows, action_index]
        brier = functional.mse_loss(
            logged_probability,
            target.float(),
        )
    return {
        "feature_mean": mean.detach().clone(),
        "feature_std": std.detach().clone(),
        "weight": model.weight.detach().clone(),
        "bias": model.bias.detach().clone(),
    }, {
        "logged_action_brier": float(brier.item()),
        "weight_l2_norm": float(model.weight.norm().item()),
    }


def _outcome_probability(
    state: dict[str, torch.Tensor],
    features: torch.Tensor,
) -> torch.Tensor:
    if "constant_probability" in state:
        return torch.full(
            (features.shape[0], 3),
            float(state["constant_probability"].item()),
            dtype=features.dtype,
        )
    normalized = (features - state["feature_mean"]) / state["feature_std"]
    return torch.sigmoid(normalized @ state["weight"].transpose(0, 1) + state["bias"])


def _activation_threshold(
    probability: torch.Tensor,
    *,
    minimum_advantage: float,
    activation_cap: float,
) -> float:
    no_op = probability[:, NO_OP_INDEX]
    best_probability, best_action = probability.max(dim=-1)
    advantage = best_probability - no_op
    eligible = advantage[(best_action != NO_OP_INDEX) & (advantage >= minimum_advantage)]
    maximum_active = max(
        1,
        int(math.floor(activation_cap * probability.shape[0])),
    )
    if eligible.numel() <= maximum_active:
        return minimum_advantage
    return float(eligible.sort(descending=True).values[maximum_active - 1].item())


def _controller_actions(
    probability: torch.Tensor,
    *,
    advantage_threshold: float,
) -> torch.Tensor:
    no_op = probability[:, NO_OP_INDEX]
    best_probability, best_action = probability.max(dim=-1)
    activate = (best_action != NO_OP_INDEX) & ((best_probability - no_op) >= advantage_threshold)
    return torch.where(
        activate,
        best_action,
        torch.full_like(best_action, NO_OP_INDEX),
    )


def _dr_value(
    probability: torch.Tensor,
    target_action: torch.Tensor,
    logged_action: torch.Tensor,
    propensity: torch.Tensor,
    outcome: torch.Tensor,
) -> torch.Tensor:
    rows = torch.arange(probability.shape[0])
    target_prediction = probability[rows, target_action]
    logged_prediction = probability[rows, logged_action]
    correction = (
        (logged_action == target_action).float()
        / propensity
        * (outcome.float() - logged_prediction)
    )
    return target_prediction + correction


def _estimate(values: torch.Tensor) -> dict[str, float | int]:
    if values.ndim != 1 or values.numel() < 2:
        raise ValueError("confidence estimate requires at least two samples")
    mean = float(values.mean().item())
    standard_error = float(values.std(unbiased=True).item() / math.sqrt(values.numel()))
    return {
        "samples": int(values.numel()),
        "estimate": mean,
        "standard_error": standard_error,
        "lower_one_sided_95": mean - ONE_SIDED_95_Z * standard_error,
        "upper_one_sided_95": mean + ONE_SIDED_95_Z * standard_error,
    }


def _policy_evaluation(
    success_probability: torch.Tensor,
    unsafe_probability: torch.Tensor,
    policy_action: torch.Tensor,
    logged_action: torch.Tensor,
    propensity: torch.Tensor,
    success: torch.Tensor,
    unsafe: torch.Tensor,
) -> dict[str, object]:
    no_op_action = torch.full_like(policy_action, NO_OP_INDEX)
    policy_success = _dr_value(
        success_probability,
        policy_action,
        logged_action,
        propensity,
        success,
    )
    no_op_success = _dr_value(
        success_probability,
        no_op_action,
        logged_action,
        propensity,
        success,
    )
    policy_unsafe = _dr_value(
        unsafe_probability,
        policy_action,
        logged_action,
        propensity,
        unsafe,
    )
    no_op_unsafe = _dr_value(
        unsafe_probability,
        no_op_action,
        logged_action,
        propensity,
        unsafe,
    )
    action_counts = {
        str(int(ACTION_IDS[index])): int((policy_action == index).sum()) for index in range(3)
    }
    return {
        "policy_success_value": _estimate(policy_success),
        "no_op_success_value": _estimate(no_op_success),
        "success_effect_vs_no_op": _estimate(policy_success - no_op_success),
        "policy_unsafe_value": _estimate(policy_unsafe),
        "no_op_unsafe_value": _estimate(no_op_unsafe),
        "unsafe_effect_vs_no_op": _estimate(policy_unsafe - no_op_unsafe),
        "policy_action_counts": action_counts,
        "policy_non_no_op_fraction": float((policy_action != NO_OP_INDEX).float().mean().item()),
    }


def _arm_summary(
    action_index: torch.Tensor,
    success: torch.Tensor,
    unsafe: torch.Tensor,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for index, action_id in enumerate(ACTION_IDS.tolist()):
        selected = action_index == index
        result[str(action_id)] = {
            "samples": int(selected.sum()),
            "success_rate": float(success[selected].float().mean().item()),
            "unsafe_rate": float(unsafe[selected].float().mean().item()),
        }
    return result


def _stratified_arm_summary(
    seed_index: torch.Tensor,
    strata: torch.Tensor,
    action_index: torch.Tensor,
    success: torch.Tensor,
    unsafe: torch.Tensor,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for seed in sorted(DEVELOPMENT_SEEDS):
        seed_result: dict[str, object] = {}
        for stratum in range(4):
            selected = (seed_index == seed) & (strata == stratum)
            seed_result[str(stratum)] = {
                "samples": int(selected.sum()),
                "arms": _arm_summary(
                    action_index[selected],
                    success[selected],
                    unsafe[selected],
                ),
            }
        result[str(seed)] = seed_result
    return result


def _dataset_provenance(
    paths: list[Path],
    payloads: list[dict[str, object]],
) -> list[dict[str, object]]:
    result = []
    for path, payload in zip(paths, payloads, strict=True):
        action = payload["assigned_action_id"].long()
        result.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "seed": int(payload["seed"]),
                "seed_stream_offset": int(payload["seed_stream_offset"]),
                "runtime_seed": int(payload["runtime_seed"]),
                "samples": int(action.numel()),
                "successful": int(payload["eventual_full_success"].sum().item()),
                "action_counts": {
                    str(action_id): int((action == action_id).sum())
                    for action_id in ACTION_IDS.tolist()
                },
            }
        )
    return result


def _run(args: argparse.Namespace) -> int:
    if args.l2 <= 0.0 or args.max_iterations <= 0:
        raise ValueError("outcome model fit parameters must be positive")
    if not 0.0 < args.activation_cap <= 0.5:
        raise ValueError("activation cap must be in (0, 0.5]")
    if not 0.0 <= args.minimum_advantage < 1.0:
        raise ValueError("minimum advantage must be in [0, 1)")
    if args.minimum_action_support <= 0:
        raise ValueError("minimum action support must be positive")

    output = Path(args.output).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    risk_path = Path(args.risk_checkpoint).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"controller report already exists: {output}")
    if checkpoint_path.exists():
        raise FileExistsError(f"controller checkpoint already exists: {checkpoint_path}")
    if output == checkpoint_path:
        raise ValueError("controller report and checkpoint must differ")

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    source_revision = _source_revision()
    paths = [Path(value).expanduser().resolve() for value in args.dataset]
    payloads = [_load_dataset(path) for path in paths]
    stream_keys = [
        (int(payload["seed"]), int(payload["seed_stream_offset"])) for payload in payloads
    ]
    if len(stream_keys) != len(set(stream_keys)):
        raise ValueError("intervention datasets repeat a seed stream")
    if {seed for seed, _ in stream_keys} != DEVELOPMENT_SEEDS:
        raise ValueError("controller gate requires all development seeds")
    base_hashes = {str(payload["base_checkpoint_sha256"]) for payload in payloads}
    candidate_hashes = {
        str(payload["receiver_candidate_checkpoint_sha256"]) for payload in payloads
    }
    if (
        len(base_hashes) != 1
        or len(candidate_hashes) != 1
        or _SHA256_PATTERN.fullmatch(next(iter(base_hashes))) is None
        or _SHA256_PATTERN.fullmatch(next(iter(candidate_hashes))) is None
    ):
        raise ValueError("intervention checkpoint provenance drifted")
    base_hash = next(iter(base_hashes))
    candidate_hash = next(iter(candidate_hashes))
    risk_checkpoint = _load_risk_checkpoint(
        risk_path,
        source_revision=source_revision,
        base_hash=base_hash,
        candidate_hash=candidate_hash,
    )

    features = torch.cat([_causal_features(payload) for payload in payloads])
    action_index = torch.cat(
        [_action_indices(payload["assigned_action_id"]) for payload in payloads]
    )
    propensity = torch.cat([payload["assigned_action_probability"].float() for payload in payloads])
    success = torch.cat([payload["eventual_full_success"].bool() for payload in payloads])
    unsafe = torch.cat(
        [_termination_indicator(payload, _UNSAFE_TERMINATIONS) for payload in payloads]
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
    pooled_risk_cutpoints = _risk_cutpoints(risk)
    pooled_strata = _risk_strata(risk, pooled_risk_cutpoints)

    action_support = {
        str(seed): {
            str(int(ACTION_IDS[index])): int(((seed_index == seed) & (action_index == index)).sum())
            for index in range(3)
        }
        for seed in sorted(DEVELOPMENT_SEEDS)
    }
    support_gate_passed = all(
        count >= args.minimum_action_support
        for seed_counts in action_support.values()
        for count in seed_counts.values()
    )

    cross_fit_success_probability = torch.empty(
        (features.shape[0], 3),
        dtype=torch.float32,
    )
    cross_fit_unsafe_probability = torch.empty_like(cross_fit_success_probability)
    cross_fit_policy_action = torch.empty(
        features.shape[0],
        dtype=torch.long,
    )
    cross_fit_replication_action = torch.empty(
        features.shape[0],
        dtype=torch.long,
    )
    folds: dict[str, object] = {}
    for held_out_seed in sorted(DEVELOPMENT_SEEDS):
        test = seed_index == held_out_seed
        train = ~test
        train_cutpoints = _risk_cutpoints(risk[train])
        train_strata = _risk_strata(risk[train], train_cutpoints)
        weights = _balanced_weights(
            seed_index[train],
            train_strata,
            action_index[train],
        )
        success_state, success_fit = _fit_outcome_network(
            features[train],
            action_index[train],
            success[train],
            weights,
            l2=args.l2,
            max_iterations=args.max_iterations,
        )
        unsafe_state, unsafe_fit = _fit_outcome_network(
            features[train],
            action_index[train],
            unsafe[train],
            weights,
            l2=args.l2,
            max_iterations=args.max_iterations,
        )
        train_success_probability = _outcome_probability(
            success_state,
            features[train],
        )
        test_success_probability = _outcome_probability(
            success_state,
            features[test],
        )
        test_unsafe_probability = _outcome_probability(
            unsafe_state,
            features[test],
        )
        threshold = _activation_threshold(
            train_success_probability,
            minimum_advantage=args.minimum_advantage,
            activation_cap=args.activation_cap,
        )
        policy_action = _controller_actions(
            test_success_probability,
            advantage_threshold=threshold,
        )
        replication_action = torch.where(
            risk[test] >= train_cutpoints[2],
            torch.zeros(int(test.sum()), dtype=torch.long),
            torch.full(
                (int(test.sum()),),
                NO_OP_INDEX,
                dtype=torch.long,
            ),
        )
        cross_fit_success_probability[test] = test_success_probability
        cross_fit_unsafe_probability[test] = test_unsafe_probability
        cross_fit_policy_action[test] = policy_action
        cross_fit_replication_action[test] = replication_action
        folds[str(held_out_seed)] = {
            "train_samples": int(train.sum()),
            "test_samples": int(test.sum()),
            "fit_uses_held_out_seed": False,
            "risk_quartile_cutpoints": train_cutpoints.tolist(),
            "advantage_threshold": threshold,
            "success_fit": success_fit,
            "unsafe_fit": unsafe_fit,
            "evaluation": _policy_evaluation(
                test_success_probability,
                test_unsafe_probability,
                policy_action,
                action_index[test],
                propensity[test],
                success[test],
                unsafe[test],
            ),
            "constrained_replication_screen": _policy_evaluation(
                test_success_probability,
                test_unsafe_probability,
                replication_action,
                action_index[test],
                propensity[test],
                success[test],
                unsafe[test],
            ),
        }

    aggregate_evaluation = _policy_evaluation(
        cross_fit_success_probability,
        cross_fit_unsafe_probability,
        cross_fit_policy_action,
        action_index,
        propensity,
        success,
        unsafe,
    )
    replication_evaluation = _policy_evaluation(
        cross_fit_success_probability,
        cross_fit_unsafe_probability,
        cross_fit_replication_action,
        action_index,
        propensity,
        success,
        unsafe,
    )
    success_gate_passed = bool(
        aggregate_evaluation["success_effect_vs_no_op"]["lower_one_sided_95"]
        > args.minimum_effect_lcb
        and all(
            fold["evaluation"]["success_effect_vs_no_op"]["lower_one_sided_95"]
            > args.minimum_effect_lcb
            for fold in folds.values()
        )
    )
    safety_gate_passed = bool(
        aggregate_evaluation["unsafe_effect_vs_no_op"]["upper_one_sided_95"]
        <= args.maximum_unsafe_ucb
        and all(
            fold["evaluation"]["unsafe_effect_vs_no_op"]["upper_one_sided_95"]
            <= args.maximum_unsafe_ucb
            for fold in folds.values()
        )
    )
    activation_gate_passed = bool(
        0.0
        < aggregate_evaluation["policy_non_no_op_fraction"]
        <= args.maximum_cross_seed_activation
        and all(
            0.0
            < fold["evaluation"]["policy_non_no_op_fraction"]
            <= args.maximum_cross_seed_activation
            for fold in folds.values()
        )
    )
    gate_passed = bool(
        support_gate_passed
        and success_gate_passed
        and safety_gate_passed
        and activation_gate_passed
    )
    replication_safety_passed = bool(
        replication_evaluation["unsafe_effect_vs_no_op"]["upper_one_sided_95"]
        <= args.maximum_unsafe_ucb
        and all(
            fold["constrained_replication_screen"]["unsafe_effect_vs_no_op"]["upper_one_sided_95"]
            <= args.maximum_unsafe_ucb
            for fold in folds.values()
        )
    )
    replication_strict_gate_passed = bool(
        support_gate_passed
        and replication_safety_passed
        and replication_evaluation["success_effect_vs_no_op"]["lower_one_sided_95"]
        > args.minimum_effect_lcb
        and all(
            fold["constrained_replication_screen"]["success_effect_vs_no_op"]["lower_one_sided_95"]
            > args.minimum_effect_lcb
            for fold in folds.values()
        )
    )
    fresh_replication_authorized = bool(
        support_gate_passed
        and replication_safety_passed
        and replication_evaluation["success_effect_vs_no_op"]["lower_one_sided_95"]
        > args.minimum_effect_lcb
        and all(
            fold["constrained_replication_screen"]["success_effect_vs_no_op"]["estimate"] > 0.0
            for fold in folds.values()
        )
    )

    full_weights = _balanced_weights(
        seed_index,
        pooled_strata,
        action_index,
    )
    final_success_state, final_success_fit = _fit_outcome_network(
        features,
        action_index,
        success,
        full_weights,
        l2=args.l2,
        max_iterations=args.max_iterations,
    )
    final_unsafe_state, final_unsafe_fit = _fit_outcome_network(
        features,
        action_index,
        unsafe,
        full_weights,
        l2=args.l2,
        max_iterations=args.max_iterations,
    )
    full_success_probability = _outcome_probability(
        final_success_state,
        features,
    )
    final_threshold = _activation_threshold(
        full_success_probability,
        minimum_advantage=args.minimum_advantage,
        activation_cap=args.activation_cap,
    )
    provenance = _dataset_provenance(paths, payloads)
    risk_hash = _sha256(risk_path)
    if gate_passed:
        status = "authorized_for_heldout_physics_evaluation"
        disposition = "preserved_as_heldout_evaluation_candidate"
        reason = (
            "The conservative cross-fitted controller has a positive "
            "one-sided 95% success-effect lower bound on every development "
            "seed and in aggregate, with no upper-bound unsafe-event increase."
        )
    elif fresh_replication_authorized:
        status = "constrained_candidate_requires_fresh_replication"
        disposition = "preserved_as_locked_fresh_replication_candidate"
        reason = (
            "The unconstrained learned controller missed its strict gate. "
            "An exploratory constrained policy—opposite pulse only in the "
            "highest calibrated-risk quartile—has a positive aggregate "
            "one-sided 95% lower bound and a positive point estimate on every "
            "seed. Lock it now and test it on fresh randomized streams before "
            "any held-out policy rollout."
        )
    else:
        status = "retained_not_authorized"
        disposition = "preserved_as_nonpromoted_research_candidate"
        reason = (
            "At least one predeclared support, activation, success-effect, or "
            "unsafe-event gate failed. Preserve the candidate and causal "
            "dataset, but do not run it on held-out physics or replace the "
            "promoted actor."
        )
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "source_revision": source_revision,
        "status": status,
        "disposition": disposition,
        "control_scope": "active_custody_bounded_residual_candidate",
        "heldout_physics_evaluation_authorized": gate_passed,
        "fresh_randomized_replication_authorized": (fresh_replication_authorized),
        "main_policy_replacement_authorized": False,
        "real_robot_authorized": False,
        "base_checkpoint_sha256": base_hash,
        "receiver_candidate_checkpoint_sha256": candidate_hash,
        "risk_checkpoint": {
            "path": str(risk_path),
            "sha256": risk_hash,
            "source_revision": risk_checkpoint["source_revision"],
            "usage": "stratification_only",
        },
        "feature_contract": {
            "kind": "causal_role_invariant_pre_plus_delta",
            "dimension": int(features.shape[-1]),
            "uses_future_information": False,
            "requires_one_frame_giver_open_probe": True,
        },
        "action_contract": {
            "ids": ACTION_IDS.tolist(),
            "semantics": _ACTION_SEMANTICS,
            "frames": 1,
            "maximum_translation_action": 0.0025,
            "gripper_action": -1.0,
        },
        "model": {
            "kind": "standardized_three_head_linear_outcome_network",
            "success": final_success_state,
            "unsafe": final_unsafe_state,
            "advantage_threshold": final_threshold,
            "minimum_advantage": args.minimum_advantage,
            "activation_cap": args.activation_cap,
            "risk_quartile_cutpoints": pooled_risk_cutpoints,
            "success_fit": final_success_fit,
            "unsafe_fit": final_unsafe_fit,
        },
        "constrained_replication_candidate": {
            "kind": "highest_risk_quartile_opposite_pulse_else_no_op",
            "risk_threshold": float(pooled_risk_cutpoints[2].item()),
            "active_action_id": -1,
            "inactive_action_id": 0,
            "derived_after_exploratory_action_outcomes": True,
            "requires_fresh_randomized_replication": True,
            "fresh_randomized_replication_authorized": (fresh_replication_authorized),
            "strict_cross_seed_gate_passed": (replication_strict_gate_passed),
        },
        "training_datasets": provenance,
        "cross_fit_gate": {
            "passed": gate_passed,
            "support_gate_passed": support_gate_passed,
            "success_gate_passed": success_gate_passed,
            "safety_gate_passed": safety_gate_passed,
            "activation_gate_passed": activation_gate_passed,
            "aggregate_evaluation": aggregate_evaluation,
        },
        "constrained_replication_screen": {
            "strict_cross_seed_gate_passed": (replication_strict_gate_passed),
            "fresh_randomized_replication_authorized": (fresh_replication_authorized),
            "aggregate_evaluation": replication_evaluation,
        },
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    checkpoint_hash = _sha256(checkpoint_path)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_revision": source_revision,
        "status": status,
        "decision": {
            "gate_passed": gate_passed,
            "disposition": disposition,
            "heldout_physics_evaluation_authorized": gate_passed,
            "fresh_randomized_replication_authorized": (fresh_replication_authorized),
            "main_policy_replacement_authorized": False,
            "real_robot_authorized": False,
            "reason": reason,
        },
        "method": {
            "problem": "three-arm randomized contextual bandit",
            "outcome_model": ("risk-stratum-balanced three-head linear neural network"),
            "evaluation": (
                "leave-one-physics-seed-out doubly robust off-policy evaluation against exact no-op"
            ),
            "confidence": "one-sided 95% normal influence interval",
            "risk_model_usage": "stratification_only",
            "l2": args.l2,
            "max_iterations": args.max_iterations,
            "minimum_advantage": args.minimum_advantage,
            "activation_cap": args.activation_cap,
        },
        "gates": {
            "minimum_action_support_per_seed_arm": (args.minimum_action_support),
            "minimum_success_effect_lower_bound": (args.minimum_effect_lcb),
            "maximum_unsafe_effect_upper_bound": (args.maximum_unsafe_ucb),
            "maximum_cross_seed_activation": (args.maximum_cross_seed_activation),
            "support_gate_passed": support_gate_passed,
            "success_gate_passed": success_gate_passed,
            "safety_gate_passed": safety_gate_passed,
            "activation_gate_passed": activation_gate_passed,
            "gate_passed": gate_passed,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_hash,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "heldout_physics_evaluation_authorized": gate_passed,
            "fresh_randomized_replication_authorized": (fresh_replication_authorized),
        },
        "risk_checkpoint": {
            "path": str(risk_path),
            "sha256": risk_hash,
            "source_revision": risk_checkpoint["source_revision"],
        },
        "datasets": provenance,
        "action_support": action_support,
        "pooled_risk_quartile_cutpoints": pooled_risk_cutpoints.tolist(),
        "observed_arms": _arm_summary(
            action_index,
            success,
            unsafe,
        ),
        "observed_arms_by_seed_and_risk_quartile": (
            _stratified_arm_summary(
                seed_index,
                pooled_strata,
                action_index,
                success,
                unsafe,
            )
        ),
        "folds": folds,
        "aggregate_evaluation": aggregate_evaluation,
        "constrained_replication_candidate": {
            "policy": ("opposite pulse in highest calibrated-risk quartile; exact no-op otherwise"),
            "risk_threshold": float(pooled_risk_cutpoints[2].item()),
            "derived_after_exploratory_action_outcomes": True,
            "requires_fresh_randomized_replication": True,
            "strict_cross_seed_gate_passed": (replication_strict_gate_passed),
            "fresh_randomized_replication_authorized": (fresh_replication_authorized),
            "aggregate_evaluation": replication_evaluation,
        },
        "deployment_fit": {
            "success_fit": final_success_fit,
            "unsafe_fit": final_unsafe_fit,
            "advantage_threshold": final_threshold,
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
        description=("Train and cross-seed gate a bounded active-custody controller")
    )
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--risk_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--l2", type=float, default=1.0e-3)
    parser.add_argument("--max_iterations", type=int, default=200)
    parser.add_argument("--minimum_advantage", type=float, default=0.02)
    parser.add_argument("--activation_cap", type=float, default=0.25)
    parser.add_argument("--minimum_action_support", type=int, default=100)
    parser.add_argument("--minimum_effect_lcb", type=float, default=0.0)
    parser.add_argument("--maximum_unsafe_ucb", type=float, default=0.0)
    parser.add_argument(
        "--maximum_cross_seed_activation",
        type=float,
        default=0.4,
    )
    parser.add_argument("--seed", type=int, default=104729)
    return parser


if __name__ == "__main__":
    raise SystemExit(_run(_parser().parse_args()))
