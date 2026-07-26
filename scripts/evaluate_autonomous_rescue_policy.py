# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Run seeded, contact-scored rollouts of the loaded rescue policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from collect_autonomous_rescue_experts import request_json


def wait_for_rollout(
    base_url: str,
    previous_demo: str | None,
    timeout_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    started = False
    terminal_policy: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        status = request_json(base_url, "/api/status")
        policy = status.get("rescue_policy", {})
        policy_status = str(policy.get("status", "not_configured"))
        recording = bool(status.get("recording", False))
        started = started or policy_status == "running" or recording
        if started and policy_status in {
            "completed",
            "timed_out",
            "error",
            "interrupted",
        }:
            terminal_policy = dict(policy)
        last_demo = status.get("last_demo")
        if (
            terminal_policy is not None
            and not recording
            and isinstance(last_demo, str)
            and last_demo != previous_demo
        ):
            return {
                "demo": last_demo,
                "status": terminal_policy["status"],
                "outcome": terminal_policy.get("outcome"),
                "checkpoint_sha256": terminal_policy.get(
                    "checkpoint_sha256"
                ),
            }
        time.sleep(0.25)
    raise TimeoutError(
        f"rescue policy did not save a rollout within {timeout_s:.1f}s"
    )


def stop_failed_rollout(base_url: str) -> None:
    try:
        status = request_json(base_url, "/api/status")
        if status.get("rescue_policy", {}).get("status") in {
            "starting",
            "running",
        }:
            request_json(base_url, "/api/rescue-policy/stop", {})
        if bool(status.get("recording", False)):
            request_json(base_url, "/api/record/stop", {})
    except RuntimeError:
        return


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one loaded Robomimic rescue checkpoint over seeded "
            "patient-contact scenarios."
        )
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:2361",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=17777)
    parser.add_argument(
        "--scenarios",
        default=(
            "baseline,camera_shift,low_light,glare,"
            "calibration_bias,sensor_dropout"
        ),
    )
    parser.add_argument("--episode-timeout-s", type=float, default=180.0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON report destination",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    if args.episode_timeout_s <= 0.0:
        raise ValueError("episode timeout must be positive")
    scenarios = [
        item.strip()
        for item in args.scenarios.split(",")
        if item.strip()
    ]
    if not scenarios:
        raise ValueError("at least one scenario is required")

    initial = request_json(args.url, "/api/status")
    if (
        initial.get("procedure", {}).get("id")
        != "dr-anmar-autonomous-rescue-or"
    ):
        raise RuntimeError(
            "the workstation is not running Autonomous Rescue OR"
        )
    if not initial.get("rescue_policy", {}).get("available"):
        raise RuntimeError(
            "the Autonomous Rescue OR room has no loaded checkpoint"
        )

    rollouts: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index in range(args.episodes):
        scenario = scenarios[index % len(scenarios)]
        seed = args.seed_start + index
        previous_demo = request_json(
            args.url,
            "/api/status",
        ).get("last_demo")
        try:
            request_json(
                args.url,
                "/api/scenario",
                {"scenario_id": scenario, "seed": seed},
            )
            request_json(args.url, "/api/rescue-policy/start", {})
            result = wait_for_rollout(
                args.url,
                previous_demo,
                args.episode_timeout_s,
            )
            result.update({"scenario": scenario, "seed": seed})
            rollouts.append(result)
            print(
                f"[{index + 1}/{args.episodes}] "
                f"{result['status']} {result['demo']} "
                f"scenario={scenario} seed={seed}",
                flush=True,
            )
        except (RuntimeError, TimeoutError) as error:
            stop_failed_rollout(args.url)
            errors.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "error": str(error),
                }
            )
            print(
                f"[{index + 1}/{args.episodes}] "
                f"error scenario={scenario} seed={seed}: {error}",
                flush=True,
            )
            if not args.continue_on_error:
                break

    successes = sum(item["status"] == "completed" for item in rollouts)
    report = {
        "requested": args.episodes,
        "completed_rollouts": len(rollouts),
        "patient_effect_successes": successes,
        "patient_effect_success_rate": (
            successes / len(rollouts) if rollouts else 0.0
        ),
        "rollouts": rollouts,
        "errors": errors,
        "policy_can_write_patient_outcome": False,
    }
    report_json = json.dumps(report, indent=2)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(report_json + "\n", encoding="utf-8")
        temporary.replace(output)
    print(report_json)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
