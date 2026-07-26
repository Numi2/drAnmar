# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""GR00T inference daemon for NVIDIA's native seven-action PSM."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import time
from pathlib import Path
from typing import Optional

import numpy as np
from common.config import get_policy_config
from common.health import PolicyHealth, serve_health
from common.io.policy import PolicyIOBase, camera_to_array
from dr_anmar_i4h.psm_data_config import DATA_CONFIG_NAME, PSM_ACTION_DIM, PSM_STATE_DIM, register

logger = logging.getLogger("dr_anmar_i4h.psm_infer")
ENV_ID = "surgical_lift_needle"


class PsmPolicyIO(PolicyIOBase):
    def latest_observation(self) -> Optional[dict]:
        with self._lock:
            if self._state is None or not self._camera_names.issubset(self._frames):
                return None
            room = camera_to_array(self._frames.get("room"))
            if room is None:
                return None
            joints = np.asarray(self._state.joint_positions, dtype=np.float64).copy()
            if joints.shape != (PSM_STATE_DIM,):
                raise RuntimeError(f"PSM state has shape {joints.shape}; expected ({PSM_STATE_DIM},)")
            return {
                "room": room,
                "joint_positions": joints,
                "state_ts": self._state.ts,
                "run_id": self._state.run_id,
                "episode_index": self._state.episode_index,
                "attempt_index": self._state.attempt_index,
            }


class PsmPolicyRunner:
    def __init__(
        self,
        *,
        model_path: str,
        task_description: str,
        denoising_steps: int,
        action_head_future_tokens: int | None,
    ) -> None:
        self.model_path = model_path
        self.task_description = task_description
        self.denoising_steps = denoising_steps
        self.action_head_future_tokens = action_head_future_tokens
        self.policy = None

    def ensure_loaded(self) -> None:
        if self.policy is not None:
            return
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"PSM checkpoint does not exist: {self.model_path}")
        from gr00t.experiment.data_config import DATA_CONFIG_MAP
        from gr00t.model.policy import Gr00tPolicy

        data_config = register()
        if DATA_CONFIG_MAP.get(DATA_CONFIG_NAME) is not data_config:
            raise RuntimeError("PSM data configuration registration is inconsistent")
        self.policy = Gr00tPolicy(
            model_path=self.model_path,
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            embodiment_tag="new_embodiment",
            denoising_steps=self.denoising_steps,
            action_head_future_tokens=self.action_head_future_tokens,
        )

    def infer(self, observation: dict, steps: int) -> np.ndarray:
        self.ensure_loaded()
        joints = np.asarray(observation["joint_positions"], dtype=np.float32)
        policy_observation = {
            "video.room": np.asarray(observation["room"], dtype=np.uint8)[np.newaxis, ...],
            "state.single_arm": joints[:6][np.newaxis, ...],
            "state.gripper": joints[6:8][np.newaxis, ...],
            "annotation.human.task_description": self.task_description,
        }
        chunk = self.policy.get_action(policy_observation)
        arm = _action_value(chunk, "single_arm")
        gripper = _action_value(chunk, "gripper")
        arm = np.asarray(arm, dtype=np.float32)
        gripper = np.asarray(gripper, dtype=np.float32)
        arm = arm.reshape(-1, arm.shape[-1])
        gripper = gripper.reshape(-1, gripper.shape[-1])
        if arm.shape[-1] != 6 or gripper.shape[-1] != 1:
            raise RuntimeError(
                f"GR00T PSM output has arm/gripper shapes {arm.shape}/{gripper.shape}; expected (*, 6)/(*, 1)"
            )
        action = np.concatenate((arm, gripper), axis=-1)
        if action.shape[-1] != PSM_ACTION_DIM or not np.isfinite(action).all():
            raise RuntimeError(f"GR00T PSM output is not a finite (*, {PSM_ACTION_DIM}) action")
        return action[: min(steps, len(action))]


def _action_value(chunk: dict, key: str):
    for candidate in (f"action.{key}", f"state.{key}", key):
        if candidate in chunk:
            return chunk[candidate]
    raise RuntimeError(f"GR00T output does not contain {key}; keys={sorted(chunk)}")


def add_args(parser: argparse.ArgumentParser) -> None:
    config = get_policy_config(ENV_ID)
    parser.add_argument("--task", default=config.task_description)
    parser.add_argument("--denoising-steps", type=int, default=config.denoising_steps or 4)
    parser.add_argument("--control-hz", type=float, default=config.control_hz or 60.0)
    parser.add_argument("--action-horizon", type=int, default=config.action_horizon or 16)
    parser.add_argument("--execution-steps", type=int, default=config.execution_steps or 16)
    parser.add_argument("--lazy-load", action="store_true")
    parser.add_argument("--warmup-timeout", type=float, default=0.0)
    parser.add_argument("--health-host", default="127.0.0.1")
    parser.add_argument("--health-port", type=int, default=None)


def run(args: argparse.Namespace) -> None:
    config = get_policy_config(ENV_ID)
    if args.execution_steps < 1 or args.execution_steps > args.action_horizon:
        raise SystemExit("execution-steps must be between 1 and action-horizon")
    model_path = _resolve_model_path(args.model_path, args.model_repo or config.required_model_repo)
    runner = PsmPolicyRunner(
        model_path=model_path,
        task_description=args.task,
        denoising_steps=args.denoising_steps,
        action_head_future_tokens=config.action_head_future_tokens,
    )
    health = PolicyHealth()
    health_port = args.health_port or config.required_health_port
    health_server = serve_health(health, host=args.health_host, port=health_port)
    stop = False

    def on_signal(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    try:
        if not args.lazy_load:
            health.set("loading")
            runner.ensure_loaded()
        period = 1.0 / args.control_hz
        with PsmPolicyIO(env_id=args.env) as io:
            health.set("waiting_for_samples")
            deadline = time.monotonic() + args.warmup_timeout if args.warmup_timeout > 0 else None
            while not stop and not io.wait_for_data(timeout=2.0):
                if deadline is not None and time.monotonic() >= deadline:
                    raise SystemExit("Timed out waiting for NVIDIA Arena camera and state samples")
            health.set("running")
            last_timestamp = -1
            while not stop:
                observation = io.latest_observation()
                if observation is None or observation["state_ts"] == last_timestamp:
                    time.sleep(0.005)
                    continue
                inference_timestamp = time.time_ns()
                actions = runner.infer(observation, args.action_horizon)
                io.publish_command(
                    actions[: args.execution_steps],
                    dt=period,
                    inference_ts=inference_timestamp,
                    run_id=observation.get("run_id"),
                    episode_index=observation.get("episode_index"),
                    attempt_index=observation.get("attempt_index"),
                )
                last_timestamp = observation["state_ts"]
    finally:
        health.set("stopping")
        health_server.shutdown()


def _resolve_model_path(explicit: str | None, repository: str) -> str:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        return str(path.resolve())
    from huggingface_hub import snapshot_download

    cache = os.environ.get("DR_ANMAR_PSM_MODEL_CACHE", os.path.expanduser("~/.cache/dr_anmar_psm"))
    return snapshot_download(repo_id=repository, cache_dir=cache)
