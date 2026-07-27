#!/usr/bin/env python3
"""Track Dr.Anmar time to qualified task achievement (TQTA).

The tracker starts from a cryptographically bound task contract and stops only
when one frozen checkpoint satisfies the declared held-out competence,
physical-retention, and safety gates. It consumes the JSON evidence emitted by
dr_anmar_learning_benchmark.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "config/dranmar_learning_path.json"
SCHEMA_VERSION = "dranmar-tqta-1.0"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _source_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def task_gate(contract: dict[str, Any], task: str) -> dict[str, Any]:
    defaults = contract.get("defaults", {})
    stage = next(
        (
            item
            for item in contract.get("stages", [])
            if item.get("task") == task
        ),
        None,
    )
    if stage is None:
        raise ValueError(f"task is not declared in the learning contract: {task}")
    promotion = stage.get("promotion", {})
    threshold = promotion.get(
        "minimum_success_rate", defaults.get("success_threshold")
    )
    held_out_seeds = [int(seed) for seed in defaults.get("held_out_seeds", [])]
    required_passes = int(
        promotion.get("held_out_seed_passes", len(held_out_seeds))
    )
    if threshold is None or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"invalid success threshold for task: {task}")
    if not held_out_seeds or required_passes < 1:
        raise ValueError(f"task has no held-out seed gate: {task}")
    if required_passes > len(held_out_seeds):
        raise ValueError("held-out seed pass count exceeds declared seeds")
    hard_termination_terms = [
        str(name)
        for name in promotion.get(
            "hard_termination_terms_must_be_zero",
            [],
        )
    ]
    retention_comparison = promotion.get(
        "analytic_baseline_retention_comparison",
        "not_required",
    )
    if retention_comparison not in {
        "not_required",
        "not_higher_rate",
        "strictly_lower_rate_unless_baseline_zero",
    }:
        raise ValueError(
            f"invalid analytic baseline retention comparison for task: {task}"
        )
    required_policy_contract = promotion.get(
        "required_policy_contract",
        {},
    )
    if not isinstance(required_policy_contract, dict):
        raise ValueError(
            f"required policy contract must be an object for task: {task}"
        )
    return {
        "minimum_success_rate": float(threshold),
        "held_out_seeds": held_out_seeds,
        "held_out_seed_passes": required_passes,
        "require_complete_first_terminal_population": bool(
            promotion.get(
                "require_complete_first_terminal_population",
                False,
            )
        ),
        "hard_termination_terms_must_be_zero": hard_termination_terms,
        "require_matching_analytic_baseline": bool(
            promotion.get("require_matching_analytic_baseline", False)
        ),
        "candidate_success_must_not_trail_analytic_baseline": bool(
            promotion.get(
                "candidate_success_must_not_trail_analytic_baseline",
                False,
            )
        ),
        "analytic_baseline_retention_comparison": retention_comparison,
        "required_policy_contract": required_policy_contract,
        "require_training_play_source_parity": bool(
            promotion.get("require_training_play_source_parity", False)
        ),
    }


def new_tracker(
    *,
    task: str,
    contract_path: Path,
    started_at: datetime | None = None,
) -> dict[str, Any]:
    resolved_contract = contract_path.expanduser().resolve()
    contract = _load_json(resolved_contract)
    gate = task_gate(contract, task)
    start = started_at or _now()
    return {
        "schema_version": SCHEMA_VERSION,
        "tracker_id": str(uuid.uuid4()),
        "goal": "minimize_time_to_qualified_task_achievement",
        "task": task,
        "started_at": _iso(start),
        "qualified_at": None,
        "task_contract": {
            "path": str(resolved_contract),
            "sha256": _sha256(resolved_contract),
        },
        "source_revision_at_start": _source_revision(),
        "gate": gate,
        "resource_totals": {
            "training_wall_clock_seconds": 0.0,
            "gpu_device_hours": 0.0,
            "simulated_steps": 0,
            "successful_expert_demonstration_minutes": 0.0,
            "human_intervention_minutes": 0.0,
            "experiment_count": 0,
        },
        "evidence": [],
        "qualification": {
            "achieved": False,
            "checkpoint_sha256": None,
            "passing_held_out_seeds": [],
            "wall_clock_seconds_to_gate": None,
        },
    }


def _validate_tracker_contract(tracker: dict[str, Any]) -> None:
    if tracker.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported TQTA tracker schema")
    binding = tracker.get("task_contract", {})
    contract_path = Path(str(binding.get("path", "")))
    if not contract_path.is_file():
        raise ValueError(f"task contract is unavailable: {contract_path}")
    observed = _sha256(contract_path)
    if observed != binding.get("sha256"):
        raise ValueError(
            "task contract changed after TQTA start; create a new tracker"
        )
    contract = _load_json(contract_path)
    current_gate = task_gate(contract, str(tracker.get("task")))
    if current_gate != tracker.get("gate"):
        raise ValueError("task promotion gate changed after TQTA start")


def _evidence_record(path: Path, evidence: dict[str, Any]) -> dict[str, Any]:
    checkpoint = evidence.get("checkpoint") or {}
    runtime = evidence.get("runtime") or {}
    source = runtime.get("source") or {}
    handover = evidence.get("first_episode_handover_diagnostics") or {}
    retention = (
        handover.get("transport_retention_diagnostics") or {}
    ).get("overall") or {}
    completed = evidence.get("completed_episodes")
    sustained_losses = retention.get(
        "episodes_with_sustained_midair_loss_3_steps"
    )
    sustained_loss_rate = None
    if completed and sustained_losses is not None:
        sustained_loss_rate = float(sustained_losses) / int(completed)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "kind": evidence.get("kind"),
        "task": evidence.get("task"),
        "seed": evidence.get("seed"),
        "checkpoint_sha256": checkpoint.get("sha256"),
        "success_rate": evidence.get("success_rate"),
        "wall_time_s": evidence.get("wall_time_s"),
        "simulated_frames": evidence.get("simulated_frames"),
        "analytic_only": bool(evidence.get("analytic_only", False)),
        "num_envs": evidence.get("num_envs"),
        "completed_episodes": completed,
        "unresolved_episodes": evidence.get("unresolved_episodes"),
        "first_terminal_outcome_per_environment": evidence.get(
            "first_terminal_outcome_per_environment"
        ),
        "termination_term_counts": evidence.get(
            "termination_term_counts"
        )
        or {},
        "sustained_midair_loss_3_steps": sustained_losses,
        "sustained_midair_loss_3_steps_rate": sustained_loss_rate,
        "source_revision": source.get("dranmar_revision"),
        "policy_contract": {
            "policy_residual_scale": evidence.get(
                "policy_residual_scale"
            ),
            "policy_giver_residual_axes": evidence.get(
                "policy_giver_residual_axes"
            ),
            "policy_analytic_vertical_authority": evidence.get(
                "policy_analytic_vertical_authority"
            ),
            "policy_receiver_residual_enabled": evidence.get(
                "policy_receiver_residual_enabled"
            ),
        },
    }


def _recompute_totals(tracker: dict[str, Any]) -> None:
    evidence = tracker.get("evidence", [])
    training = [item for item in evidence if item.get("kind") == "training"]
    totals = tracker["resource_totals"]
    training_seconds = sum(
        float(item.get("wall_time_s") or 0.0) for item in training
    )
    totals["training_wall_clock_seconds"] = training_seconds
    totals["gpu_device_hours"] = training_seconds / 3600.0
    totals["simulated_steps"] = sum(
        int(item.get("simulated_frames") or 0) for item in training
    )
    totals["experiment_count"] = len(evidence)


def _recompute_qualification(
    tracker: dict[str, Any], *, observed_at: datetime
) -> None:
    gate = tracker["gate"]
    required_seeds = set(int(seed) for seed in gate["held_out_seeds"])
    threshold = float(gate["minimum_success_rate"])
    required_passes = int(gate["held_out_seed_passes"])
    training_checkpoints: dict[str, set[str | None]] = {}
    for item in tracker["evidence"]:
        checkpoint = item.get("checkpoint_sha256")
        if item.get("kind") == "training" and checkpoint:
            training_checkpoints.setdefault(str(checkpoint), set()).add(
                item.get("source_revision")
            )
    analytic_baselines = [
        item
        for item in tracker["evidence"]
        if item.get("kind") == "held_out_play"
        and item.get("analytic_only")
    ]

    def matching_baseline(item: dict[str, Any]) -> dict[str, Any] | None:
        for baseline in analytic_baselines:
            if (
                baseline.get("seed") == item.get("seed")
                and baseline.get("num_envs") == item.get("num_envs")
                and baseline.get("completed_episodes")
                == item.get("completed_episodes")
                and baseline.get("source_revision")
                == item.get("source_revision")
            ):
                return baseline
        return None

    def passes_physical_gate(item: dict[str, Any]) -> bool:
        if item.get("analytic_only"):
            return False
        if (
            gate["require_matching_analytic_baseline"]
            and not item.get("source_revision")
        ):
            return False
        if gate["require_complete_first_terminal_population"]:
            if not item.get("first_terminal_outcome_per_environment"):
                return False
            if (
                item.get("num_envs") is None
                or item.get("completed_episodes") != item.get("num_envs")
                or item.get("unresolved_episodes") != 0
            ):
                return False
        termination_counts = item.get("termination_term_counts") or {}
        if any(
            int(termination_counts.get(name, 0)) != 0
            for name in gate["hard_termination_terms_must_be_zero"]
        ):
            return False
        policy_contract = item.get("policy_contract") or {}
        if any(
            policy_contract.get(name) != expected
            for name, expected in gate["required_policy_contract"].items()
        ):
            return False

        baseline = matching_baseline(item)
        if gate["require_matching_analytic_baseline"] and baseline is None:
            return False
        if baseline is None:
            return True
        if gate["candidate_success_must_not_trail_analytic_baseline"]:
            baseline_success = baseline.get("success_rate")
            candidate_success = item.get("success_rate")
            if (
                baseline_success is None
                or candidate_success is None
                or float(candidate_success) < float(baseline_success)
            ):
                return False
        comparison = gate["analytic_baseline_retention_comparison"]
        if comparison != "not_required":
            baseline_rate = baseline.get(
                "sustained_midair_loss_3_steps_rate"
            )
            candidate_rate = item.get(
                "sustained_midair_loss_3_steps_rate"
            )
            if baseline_rate is None or candidate_rate is None:
                return False
            if comparison == "not_higher_rate":
                if float(candidate_rate) > float(baseline_rate):
                    return False
            elif (
                float(baseline_rate) > 0.0
                and float(candidate_rate) >= float(baseline_rate)
            ):
                return False
            elif (
                float(baseline_rate) == 0.0
                and float(candidate_rate) != 0.0
            ):
                return False
        return True

    candidates: dict[str, set[int]] = {}
    winner: tuple[str, list[int]] | None = None
    for item in tracker["evidence"]:
        if item.get("kind") != "held_out_play":
            continue
        checkpoint = item.get("checkpoint_sha256")
        seed = item.get("seed")
        success_rate = item.get("success_rate")
        checkpoint_key = str(checkpoint)
        training_revisions = training_checkpoints.get(checkpoint_key)
        if (
            training_revisions is None
            or seed is None
            or int(seed) not in required_seeds
            or success_rate is None
            or float(success_rate) < threshold
            or not passes_physical_gate(item)
            or (
                gate["require_training_play_source_parity"]
                and (
                    not item.get("source_revision")
                    or item.get("source_revision")
                    not in training_revisions
                )
            )
        ):
            continue
        seeds = candidates.setdefault(checkpoint_key, set())
        seeds.add(int(seed))
        if winner is None and len(seeds) >= required_passes:
            winner = (checkpoint_key, sorted(seeds))
    qualification = tracker["qualification"]
    if winner is None:
        qualification.update(
            {
                "achieved": False,
                "checkpoint_sha256": None,
                "passing_held_out_seeds": [],
                "wall_clock_seconds_to_gate": None,
            }
        )
        return

    checkpoint, seeds = winner
    if tracker.get("qualified_at") is None:
        tracker["qualified_at"] = _iso(observed_at)
    qualified_at = _parse_time(tracker["qualified_at"])
    elapsed = max(
        0.0,
        (qualified_at - _parse_time(tracker["started_at"])).total_seconds(),
    )
    qualification.update(
        {
            "achieved": True,
            "checkpoint_sha256": checkpoint,
            "passing_held_out_seeds": seeds,
            "wall_clock_seconds_to_gate": elapsed,
        }
    )


def ingest(
    tracker: dict[str, Any],
    evidence_paths: Iterable[Path],
    *,
    expert_minutes: float = 0.0,
    intervention_minutes: float = 0.0,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    _validate_tracker_contract(tracker)
    if expert_minutes < 0.0 or intervention_minutes < 0.0:
        raise ValueError("time contributions cannot be negative")

    existing_hashes = {item["sha256"] for item in tracker["evidence"]}
    evidence_paths = list(evidence_paths)
    if tracker["qualification"]["achieved"]:
        has_new_evidence = any(
            not path.expanduser().resolve().is_file()
            or _sha256(path.expanduser().resolve()) not in existing_hashes
            for path in evidence_paths
        )
        if has_new_evidence or expert_minutes or intervention_minutes:
            raise ValueError("TQTA tracker is already qualified and frozen")
        return tracker

    for original_path in evidence_paths:
        path = original_path.expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"evidence file not found: {path}")
        digest = _sha256(path)
        if digest in existing_hashes:
            continue
        evidence = _load_json(path)
        if evidence.get("kind") not in {"training", "held_out_play"}:
            raise ValueError(f"unsupported learning evidence kind: {path}")
        if evidence.get("task") != tracker.get("task"):
            raise ValueError(f"evidence task does not match tracker: {path}")
        record = _evidence_record(path, evidence)
        is_analytic_baseline = (
            record["kind"] == "held_out_play"
            and record["analytic_only"]
        )
        if not record["checkpoint_sha256"] and not is_analytic_baseline:
            raise ValueError(f"evidence has no frozen checkpoint hash: {path}")
        tracker["evidence"].append(record)
        existing_hashes.add(digest)

    totals = tracker["resource_totals"]
    totals["successful_expert_demonstration_minutes"] += expert_minutes
    totals["human_intervention_minutes"] += intervention_minutes
    _recompute_totals(tracker)
    _recompute_qualification(tracker, observed_at=observed_at or _now())
    return tracker


def _write_tracker(path: Path, tracker: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tracker, indent=2, sort_keys=True) + "\n")


def _report(tracker: dict[str, Any]) -> str:
    qualification = tracker["qualification"]
    totals = tracker["resource_totals"]
    status = "QUALIFIED" if qualification["achieved"] else "IN PROGRESS"
    wall = qualification["wall_clock_seconds_to_gate"]
    wall_text = f"{wall:.1f}s" if wall is not None else "pending"
    return (
        f"TQTA {status}: task={tracker['task']} wall={wall_text} "
        f"gpu_hours={totals['gpu_device_hours']:.4f} "
        f"steps={totals['simulated_steps']} "
        f"experiments={totals['experiment_count']}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track Dr.Anmar time to qualified task achievement"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start")
    start.add_argument("--task", required=True)
    start.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    start.add_argument("--tracker", type=Path, required=True)

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--tracker", type=Path, required=True)
    ingest_parser.add_argument("--expert-minutes", type=float, default=0.0)
    ingest_parser.add_argument("--intervention-minutes", type=float, default=0.0)
    ingest_parser.add_argument("evidence", nargs="+", type=Path)

    report = subparsers.add_parser("report")
    report.add_argument("--tracker", type=Path, required=True)
    report.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "start":
            tracker_path = args.tracker.expanduser().resolve()
            if tracker_path.exists():
                raise ValueError(f"tracker already exists: {tracker_path}")
            tracker = new_tracker(task=args.task, contract_path=args.contract)
            _write_tracker(tracker_path, tracker)
            print(_report(tracker))
            print(f"Tracker: {tracker_path}")
            return 0

        tracker_path = args.tracker.expanduser().resolve()
        tracker = _load_json(tracker_path)
        if args.command == "ingest":
            ingest(
                tracker,
                args.evidence,
                expert_minutes=args.expert_minutes,
                intervention_minutes=args.intervention_minutes,
            )
            _write_tracker(tracker_path, tracker)
            print(_report(tracker))
            return 0
        if args.json:
            print(json.dumps(tracker, indent=2, sort_keys=True))
        else:
            print(_report(tracker))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
