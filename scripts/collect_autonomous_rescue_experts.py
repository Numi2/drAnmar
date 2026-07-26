# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Collect seeded contact-driven rescue expert demonstrations."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request_json(
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = (
        json.dumps(payload).encode("utf-8")
        if payload is not None
        else None
    )
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    try:
        with urlopen(request, timeout=10.0) as response:
            value = json.load(response)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{path} returned HTTP {error.code}: {detail}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"could not reach Dr.Anmar at {base_url}: {error.reason}"
        ) from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} returned a non-object response")
    return value


def wait_for_episode(
    base_url: str,
    previous_demo: str | None,
    timeout_s: float,
) -> str:
    deadline = time.monotonic() + timeout_s
    started = False
    while time.monotonic() < deadline:
        status = request_json(base_url, "/api/status")
        expert = status.get("expert_demonstration", {})
        expert_status = str(expert.get("status", "idle"))
        recording = bool(status.get("recording", False))
        started = started or expert_status == "running" or recording
        last_demo = status.get("last_demo")
        if (
            started
            and expert_status == "completed"
            and not recording
            and isinstance(last_demo, str)
            and last_demo != previous_demo
        ):
            return last_demo
        if started and expert_status in {
            "paused",
            "cancelled",
            "taken_over",
        }:
            reason = expert.get("paused_reason") or ", ".join(
                expert.get("degraded_reasons", [])
            )
            raise RuntimeError(
                f"expert stopped in {expert_status}: "
                f"{reason or 'no effect evidence'}"
            )
        time.sleep(0.25)
    raise TimeoutError(
        f"rescue expert did not save an episode within {timeout_s:.1f}s"
    )


def stop_failed_episode(base_url: str) -> None:
    """Stop robot motion and finish any non-reference recording."""

    try:
        status = request_json(base_url, "/api/status")
        expert_status = str(
            status.get("expert_demonstration", {}).get(
                "status",
                "idle",
            )
        )
        if expert_status in {"running", "paused"}:
            request_json(base_url, "/api/expert/take-control", {})
        if bool(status.get("recording", False)):
            request_json(base_url, "/api/record/stop", {})
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if not bool(
                request_json(base_url, "/api/status").get(
                    "recording",
                    False,
                )
            ):
                return
            time.sleep(0.25)
    except RuntimeError:
        # Preserve the original collection failure as the actionable error.
        return


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the live Autonomous Rescue OR expert over seeded scenarios. "
            "The script changes scenario/reset state and starts robot motion; "
            "patient outcomes remain owned by scene evidence."
        )
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:2361",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=7777)
    parser.add_argument(
        "--scenarios",
        default=(
            "baseline,camera_shift,low_light,glare,"
            "calibration_bias,sensor_dropout"
        ),
    )
    parser.add_argument("--episode-timeout-s", type=float, default=180.0)
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
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
    procedure = initial.get("procedure", {})
    if procedure.get("id") != "dr-anmar-autonomous-rescue-or":
        raise RuntimeError(
            "the workstation is not running Autonomous Rescue OR"
        )
    saved = []
    failures = []
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
            request_json(args.url, "/api/expert/start", {})
            demonstration = wait_for_episode(
                args.url,
                previous_demo,
                args.episode_timeout_s,
            )
            saved.append(demonstration)
            print(
                f"[{index + 1}/{args.episodes}] "
                f"saved {demonstration} scenario={scenario} seed={seed}",
                flush=True,
            )
        except (RuntimeError, TimeoutError) as error:
            stop_failed_episode(args.url)
            failures.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "error": str(error),
                }
            )
            print(
                f"[{index + 1}/{args.episodes}] "
                f"failed scenario={scenario} seed={seed}: {error}",
                flush=True,
            )
            if not args.continue_on_failure:
                break

    print(
        json.dumps(
            {
                "requested": args.episodes,
                "saved": saved,
                "failures": failures,
                "policy_can_write_patient_outcome": False,
            },
            indent=2,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
