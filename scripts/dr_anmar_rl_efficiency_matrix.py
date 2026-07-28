#!/usr/bin/env python3
"""Compare DrAnmar environment counts at an equal simulator-frame budget."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


class MatrixError(ValueError):
    """Evidence cannot support the requested equal-frame comparison."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MatrixError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise MatrixError(f"JSON root must be an object: {path}")
    return value


def _matrix_rows(config: dict[str, Any]) -> dict[int, int]:
    rows: dict[int, int] = {}
    rollout_steps = int(config["rollout_steps_per_environment"])
    fixed_frames = int(config["fixed_simulated_frames"])
    for row in config["environment_counts"]:
        num_envs = int(row["num_envs"])
        iterations = int(row["max_iterations"])
        if num_envs in rows:
            raise MatrixError(f"duplicate environment count {num_envs}")
        frames = num_envs * rollout_steps * iterations
        if frames != fixed_frames:
            raise MatrixError(
                f"{num_envs} environments produce {frames} frames, "
                f"not fixed budget {fixed_frames}"
            )
        rows[num_envs] = iterations
    return rows


def _first_threshold_iteration(
    history: list[float],
    threshold: float,
) -> int | None:
    for index, value in enumerate(history, start=1):
        if value >= threshold:
            return index
    return None


def analyze_matrix(
    config: dict[str, Any],
    evidence_documents: list[tuple[Path, dict[str, Any]]],
    qualification_documents: (
        list[tuple[Path, dict[str, Any]]] | None
    ) = None,
) -> dict[str, Any]:
    """Validate complete evidence and rank candidates without sample bias."""
    rows = _matrix_rows(config)
    seeds = [int(value) for value in config["seeds"]]
    expected = {
        (num_envs, seed)
        for num_envs in rows
        for seed in seeds
    }
    observed: dict[tuple[int, int], dict[str, Any]] = {}
    fixed_frames = int(config["fixed_simulated_frames"])
    rollout_steps = int(config["rollout_steps_per_environment"])
    threshold = float(config["training_success_threshold"])
    for path, evidence in evidence_documents:
        if evidence.get("kind") != "training":
            raise MatrixError(f"{path} is not training evidence")
        if evidence.get("task") != config["training_task"]:
            raise MatrixError(
                f"{path} task {evidence.get('task')!r} does not match "
                f"{config['training_task']!r}"
            )
        key = (int(evidence["num_envs"]), int(evidence["seed"]))
        if key not in expected:
            raise MatrixError(f"{path} has unexpected matrix cell {key}")
        if key in observed:
            raise MatrixError(f"duplicate matrix evidence for {key}")
        if int(evidence["simulated_frames"]) != fixed_frames:
            raise MatrixError(
                f"{path} used {evidence['simulated_frames']} frames; "
                f"required {fixed_frames}"
            )
        if int(evidence["rollout_steps_per_env"]) != rollout_steps:
            raise MatrixError(f"{path} changed rollout length")
        if not evidence.get("initial_policy_bundle", {}).get("bound"):
            raise MatrixError(f"{path} used an unbundled source checkpoint")
        if not evidence.get("output_policy_bundle"):
            raise MatrixError(f"{path} did not emit a serving policy bundle")
        observed[key] = evidence
    missing = sorted(expected - set(observed))
    if missing:
        raise MatrixError(f"missing matrix cells: {missing}")

    qualification_by_key: dict[
        tuple[str, str, int],
        tuple[Path, dict[str, Any]],
    ] = {}
    for path, evidence in qualification_documents or []:
        if evidence.get("kind") != "held_out_play":
            raise MatrixError(f"{path} is not held-out play evidence")
        checkpoint = evidence.get("checkpoint")
        if not isinstance(checkpoint, dict) or not checkpoint.get("sha256"):
            raise MatrixError(f"{path} has no checkpoint hash")
        key = (
            str(checkpoint["sha256"]),
            str(evidence.get("task")),
            int(evidence["seed"]),
        )
        if key in qualification_by_key:
            raise MatrixError(f"duplicate qualification evidence for {key}")
        qualification_by_key[key] = (path, evidence)

    qualification_cfg = config["qualification"]
    standard_task = str(qualification_cfg["task"])
    durability_task = str(qualification_cfg["durability_task"])

    def qualification_result(
        checkpoint_sha256: str,
        seed: int,
        task: str,
        minimum_success_rate: float,
    ) -> dict[str, Any] | None:
        item = qualification_by_key.get(
            (checkpoint_sha256, task, seed)
        )
        if item is None:
            return None
        path, evidence = item
        requested = int(evidence["requested_num_envs"])
        completed = int(evidence["completed_episodes"])
        completed_fraction = (
            completed / requested if requested else 0.0
        )
        safety_terminals = sum(
            int(evidence.get("termination_term_counts", {}).get(name, 0))
            for name in (
                "excessive_object_force",
                "protected_surface_force",
            )
        )
        safety_rate = (
            safety_terminals / completed if completed else 1.0
        )
        role_population = (
            evidence.get("first_episode_handover_diagnostics", {})
            .get("initial_giver_role_population", {})
        )
        both_roles = (
            int(role_population.get("robot_1", 0)) > 0
            and int(role_population.get("robot_2", 0)) > 0
        )
        role_imbalance = int(
            role_population.get("absolute_imbalance", requested)
        )
        role_balance_passed = role_imbalance <= int(
            qualification_cfg["maximum_initial_role_imbalance"]
        )
        success_rate = float(evidence["success_rate"])
        passed = (
            success_rate >= minimum_success_rate
            and completed_fraction
            >= float(
                qualification_cfg[
                    "minimum_completed_outcome_fraction"
                ]
            )
            and safety_rate
            <= float(
                qualification_cfg["maximum_safety_terminal_rate"]
            )
            and (
                both_roles
                or not bool(
                    qualification_cfg["require_both_giver_roles"]
                )
            )
            and role_balance_passed
        )
        return {
            "path": str(path),
            "task": task,
            "success_rate": success_rate,
            "completed_outcome_fraction": completed_fraction,
            "safety_terminal_rate": safety_rate,
            "both_giver_roles_present": both_roles,
            "initial_role_imbalance": role_imbalance,
            "role_balance_passed": role_balance_passed,
            "passed": passed,
        }

    candidates = []
    for num_envs in sorted(rows):
        seed_results = []
        for seed in seeds:
            evidence = observed[(num_envs, seed)]
            history = [
                float(value)
                for value in evidence["success"]["history"]
            ]
            threshold_iteration = _first_threshold_iteration(
                history,
                threshold,
            )
            total_fps = float(evidence["total_fps"])
            frames_to_threshold = (
                threshold_iteration * num_envs * rollout_steps
                if threshold_iteration is not None
                else None
            )
            estimated_time_to_threshold_s = (
                frames_to_threshold / total_fps
                if frames_to_threshold is not None and total_fps > 0.0
                else None
            )
            seed_results.append(
                {
                    "seed": seed,
                    "total_fps": total_fps,
                    "threshold_iteration": threshold_iteration,
                    "frames_to_threshold": frames_to_threshold,
                    "estimated_time_to_threshold_s": (
                        estimated_time_to_threshold_s
                    ),
                    "final_tail_mean": evidence["success"]["tail_mean"],
                    "checkpoint_sha256": evidence["checkpoint"]["sha256"],
                    "policy_bundle_contract_sha256": evidence[
                        "output_policy_bundle"
                    ]["contract_sha256"],
                    "standard_qualification": qualification_result(
                        evidence["checkpoint"]["sha256"],
                        seed,
                        standard_task,
                        float(
                            qualification_cfg[
                                "minimum_standard_success_rate"
                            ]
                        ),
                    ),
                    "durability_qualification": qualification_result(
                        evidence["checkpoint"]["sha256"],
                        seed,
                        durability_task,
                        float(
                            qualification_cfg[
                                "minimum_durability_success_rate"
                            ]
                        ),
                    ),
                }
            )
        threshold_times = [
            row["estimated_time_to_threshold_s"]
            for row in seed_results
            if row["estimated_time_to_threshold_s"] is not None
        ]
        all_seeds_reached = len(threshold_times) == len(seeds)
        qualifications_complete = all(
            row["standard_qualification"] is not None
            and row["durability_qualification"] is not None
            for row in seed_results
        )
        qualifications_passed = (
            qualifications_complete
            and all(
                row["standard_qualification"]["passed"]
                and row["durability_qualification"]["passed"]
                for row in seed_results
            )
        )
        candidates.append(
            {
                "num_envs": num_envs,
                "fixed_simulated_frames_per_seed": fixed_frames,
                "all_seeds_reached_training_threshold": all_seeds_reached,
                "qualifications_complete": qualifications_complete,
                "qualifications_passed": qualifications_passed,
                "median_total_fps": statistics.median(
                    row["total_fps"] for row in seed_results
                ),
                "median_estimated_time_to_threshold_s": (
                    statistics.median(threshold_times)
                    if all_seeds_reached
                    else None
                ),
                "seed_results": seed_results,
            }
        )
    qualification_evidence_supplied = bool(qualification_documents)
    eligible = [
        candidate
        for candidate in candidates
        if candidate["all_seeds_reached_training_threshold"]
        and (
            candidate["qualifications_passed"]
            if qualification_evidence_supplied
            else True
        )
    ]
    selected = (
        min(
            eligible,
            key=lambda row: row[
                "median_estimated_time_to_threshold_s"
            ],
        )["num_envs"]
        if eligible
        else None
    )
    return {
        "schema_version": "dranmar-rl-efficiency-result-1.0",
        "experiment_id": config["experiment_id"],
        "comparison_basis": "equal_simulated_frames",
        "fixed_simulated_frames_per_seed": fixed_frames,
        "seeds": seeds,
        "training_success_threshold": threshold,
        "candidates": candidates,
        (
            "selected_num_envs"
            if qualification_evidence_supplied
            else "provisional_fastest_num_envs"
        ): selected,
        "promotion_status": (
            (
                "qualified_candidate_selected"
                if qualification_evidence_supplied
                else "requires_standard_and_durability_qualification"
            )
            if selected is not None
            else (
                "no_candidate_passed_all_qualification_gates"
                if qualification_evidence_supplied
                else "no_candidate_reached_threshold_on_all_seeds"
            )
        ),
        "selection_rule": config["selection_rule"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank equal-frame DrAnmar RL training runs"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--evidence",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--qualification-evidence",
        type=Path,
        action="append",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        config = _read_json(args.config.resolve())
        evidence = [
            (path.resolve(), _read_json(path.resolve()))
            for path in args.evidence
        ]
        qualification_evidence = [
            (path.resolve(), _read_json(path.resolve()))
            for path in (args.qualification_evidence or [])
        ]
        result = analyze_matrix(
            config,
            evidence,
            qualification_evidence,
        )
    except (KeyError, TypeError, ValueError, MatrixError) as error:
        print(f"error: {error}")
        return 2
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
