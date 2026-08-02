#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate analytical or recurrent-residual tissue entry on isolated episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument(
    "--task", default="DrAnmar-Penetrate-Tissue-Needle-PSM-IK-Rel-v0"
)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--episodes", type=int, default=48)
parser.add_argument("--num_envs", type=int, default=12)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--trace_interval", type=int, default=0)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_folder", type=Path)
parser.add_argument("--video_length", type=int, default=1800)
parser.add_argument("--video_width", type=int, default=960)
parser.add_argument("--video_height", type=int, default=720)
parser.add_argument("--video_frame_interval", type=int, default=5)
parser.add_argument("--video_fps", type=int, default=10)
parser.add_argument(
    "--giver_base_lift_m",
    type=float,
    help="Diagnostic-only giver base lift above the legacy reset pose.",
)
parser.add_argument(
    "--episode_length_s",
    type=float,
    help="Diagnostic-only episode horizon override.",
)
parser.add_argument(
    "--giver_joint_positions",
    help="Diagnostic-only comma-separated six-DOF giver reset posture.",
)
parser.add_argument(
    "--rcm_follow_gain",
    type=float,
    help="Diagnostic-only pre-contact RCM rotation coupling gain.",
)
parser.add_argument(
    "--ik_orientation_weight",
    type=float,
    help="Diagnostic-only giver IK orientation-row weight.",
)
parser.add_argument(
    "--receiver_ik_orientation_weight",
    type=float,
    help="Diagnostic-only receiver IK orientation-row weight.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import orbit.surgical.tasks  # noqa: E402, F401
from orbit.surgical.tasks.surgical.penetration.residual_model import (  # noqa: E402
    PenetrationAnalyticController,
    PulloutAnalyticController,
    ThroughPunctureAnalyticController,
)


DEPRECATED_MODEL_KEYS = (
    "stochastic",
    "init_noise_std",
    "noise_std_type",
    "state_dependent_std",
)


def _runner_cfg():
    cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    cfg.seed = args.seed
    cfg.device = args.device
    return cfg


def _sanitized_runner_dict(cfg) -> dict:
    result = cfg.to_dict()
    for model_name in ("actor", "critic"):
        for key in DEPRECATED_MODEL_KEYS:
            result[model_name].pop(key, None)
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if args.episodes <= 0 or args.num_envs <= 0:
        raise ValueError("episodes and num_envs must be positive")
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=True,
    )
    env_cfg.seed = args.seed
    if args.giver_base_lift_m is not None:
        if not 0.0 <= args.giver_base_lift_m <= 0.040:
            raise ValueError("giver_base_lift_m must be in [0.0, 0.040]")
        lift_m = args.giver_base_lift_m
        root_x, root_y, _ = env_cfg.scene.robot.init_state.pos
        env_cfg.scene.robot.init_state.pos = (
            root_x,
            root_y,
            0.04676338424909 + lift_m,
        )
        ranges = env_cfg.commands.entry_pose.ranges
        command_x = -0.00608844038348 - 0.4817536473274231 * lift_m
        command_y = -0.01046408122182 + 0.8763065934181213 * lift_m
        ranges.pos_x = (command_x, command_x)
        ranges.pos_y = (command_y, command_y)
    diagnostic_joint_positions = None
    if args.giver_joint_positions is not None:
        diagnostic_joint_positions = tuple(
            float(value) for value in args.giver_joint_positions.split(",")
        )
        if len(diagnostic_joint_positions) != 6:
            raise ValueError("giver_joint_positions must contain six values")
        joint_names = (
            "psm_yaw_joint",
            "psm_pitch_end_joint",
            "psm_main_insertion_joint",
            "psm_tool_roll_joint",
            "psm_tool_pitch_joint",
            "psm_tool_yaw_joint",
        )
        env_cfg.scene.robot.init_state.joint_pos.update(
            dict(zip(joint_names, diagnostic_joint_positions, strict=True))
        )
    if args.episode_length_s is not None:
        if args.episode_length_s <= 0.0:
            raise ValueError("episode_length_s must be positive")
        env_cfg.episode_length_s = args.episode_length_s
    if args.ik_orientation_weight is not None:
        if not 0.0 <= args.ik_orientation_weight <= 1.0:
            raise ValueError("ik_orientation_weight must be in [0.0, 1.0]")
        env_cfg.actions.body_action.controller.orientation_weight = (
            args.ik_orientation_weight
        )
    if args.receiver_ik_orientation_weight is not None:
        if not 0.0 <= args.receiver_ik_orientation_weight <= 1.0:
            raise ValueError("receiver_ik_orientation_weight must be in [0.0, 1.0]")
        env_cfg.actions.receiver_body_action.controller.orientation_weight = (
            args.receiver_ik_orientation_weight
        )
    render_mode = "rgb_array" if args.video else None
    if args.video:
        if args.num_envs != 1:
            raise ValueError("video recording requires --num_envs 1")
        if args.video_length <= 0:
            raise ValueError("video_length must be positive")
        if args.video_frame_interval <= 0 or args.video_fps <= 0:
            raise ValueError("video frame interval and fps must be positive")
        env_cfg.viewer.resolution = (args.video_width, args.video_height)
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (0.075, 0.16, 0.105)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.047)
    gym_env = gym.make(args.task, cfg=env_cfg, render_mode=render_mode)
    video_writer = None
    video_path = None
    if args.video:
        video_folder = (
            args.video_folder
            or args.report.resolve().parent / f"{args.report.stem}-video"
        ).resolve()
        video_folder.mkdir(parents=True, exist_ok=True)
        video_path = video_folder / f"{args.task}-seed{args.seed}.mp4"
        import imageio.v2 as imageio

        video_writer = imageio.get_writer(
            video_path,
            fps=args.video_fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )
    env = RslRlVecEnvWrapper(gym_env)
    runner = None
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    if checkpoint is None:
        if "Puncture-Pullout" in args.task:
            controller_type = PulloutAnalyticController
        elif "Through-Puncture" in args.task:
            controller_type = ThroughPunctureAnalyticController
        else:
            controller_type = PenetrationAnalyticController
        controller = controller_type().to(env.unwrapped.device)
        if args.rcm_follow_gain is not None:
            if not 0.0 <= args.rcm_follow_gain <= 1.5:
                raise ValueError("rcm_follow_gain must be in [0.0, 1.5]")
            if isinstance(controller, PulloutAnalyticController):
                entry_controller = controller.through_controller.entry_controller
            elif isinstance(controller, ThroughPunctureAnalyticController):
                entry_controller = controller.entry_controller
            else:
                entry_controller = controller
            entry_controller.rcm_follow_gain = args.rcm_follow_gain

        def policy(observation):
            return controller(observation["policy"])[0]

        policy_name = "analytical"
    else:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        agent_cfg = _runner_cfg()
        runner = OnPolicyRunner(
            env,
            _sanitized_runner_dict(agent_cfg),
            log_dir=None,
            device=agent_cfg.device,
        )
        runner.load(
            str(checkpoint),
            load_cfg={
                "actor": True,
                "critic": False,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            },
        )
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        policy_name = "recurrent_residual_ppo"

    observation = env.get_observations()
    completed = 0
    successes = 0
    hard_failures = 0
    timeouts = 0
    entry_errors: list[float] = []
    tangent_errors: list[float] = []
    plane_errors: list[float] = []
    embedded_depths: list[float] = []
    normalized_peak_forces: list[float] = []
    event_counts: list[int] = []
    exit_event_counts: list[int] = []
    right_underside_event_counts: list[int] = []
    receiver_pull_steps: list[int] = []
    receiver_curve_rotations: list[float] = []
    receiver_curve_center_errors: list[float] = []
    exit_errors: list[float] = []
    exposed_fractions: list[float] = []
    exposed_arc_lengths: list[float] = []
    backend_revisions: set[str] = set()
    backend_hashes: set[str] = set()
    failure_flags: dict[str, int] = {}
    control_steps = 0
    previous_trace_phase: int | None = None
    try:
        while completed < args.episodes and simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(observation)
                observation, _, dones, _ = env.step(actions)
                if runner is not None:
                    policy.reset(dones)
            control_steps += 1
            trace_state = getattr(env.unwrapped, "_dr_anmar_penetration_state", {})
            trace_phase = int(trace_state.get("phase", torch.tensor([-1]))[0])
            phase_changed = trace_phase != previous_trace_phase
            if (
                video_writer is not None
                and control_steps <= args.video_length
                and (
                    control_steps % args.video_frame_interval == 0
                    or phase_changed
                    or bool(torch.any(dones))
                )
            ):
                frame = gym_env.render()
                if isinstance(frame, list):
                    frame = frame[-1]
                video_writer.append_data(frame)
            if args.trace_interval and (
                control_steps % args.trace_interval == 0
                or phase_changed
                or bool(torch.any(dones))
            ):
                state = trace_state
                measurement = state.get("measurement", {})
                policy_observation = observation["policy"][0]
                print(
                    "[DR_ANMAR_TISSUE_TRACE] "
                    + json.dumps(
                        {
                            "step": control_steps,
                            "phase": int(state.get("phase", torch.tensor([-1]))[0]),
                            "entry_error_m": float(
                                measurement.get("entry_error", torch.tensor([float("nan")]))[0]
                            ),
                            "indentation_m": float(
                                measurement.get("indentation", torch.tensor([float("nan")]))[0]
                            ),
                            "tangent_error_deg": float(
                                measurement.get(
                                    "tangent_error", torch.tensor([float("nan")])
                                )[0]
                            ),
                            "plane_error_deg": float(
                                measurement.get(
                                    "plane_error", torch.tensor([float("nan")])
                                )[0]
                            ),
                            "exposed_fraction": float(
                                measurement.get("exposed_fraction", torch.tensor([0.0]))[0]
                            ),
                            "exit_error_m": float(
                                measurement.get("exit_error", torch.tensor([0.0]))[0]
                            ),
                            "right_underside_event_count": int(
                                state.get(
                                    "right_underside_event_count",
                                    torch.tensor([0]),
                                )[0]
                            ),
                            "tip_pos": [
                                float(value)
                                for value in measurement.get(
                                    "tip_pos", torch.zeros((1, 3))
                                )[0]
                            ],
                            "needle_pos_robot": [
                                float(value) for value in policy_observation[23:26]
                            ],
                            "entry_pos_robot": [
                                float(value) for value in policy_observation[36:39]
                            ],
                            "surface_normal_robot": [
                                float(value) for value in policy_observation[43:46]
                            ],
                            "giver_joint_positions": [
                                float(value) for value in policy_observation[:8]
                            ],
                            "giver_joint_positions_absolute": [
                                float(value)
                                for value in env.unwrapped.scene["robot"].data.joint_pos.torch[0]
                            ],
                            "giver_ee_position_robot": [
                                float(value) for value in policy_observation[16:19]
                            ],
                            "receiver_joint_positions": [
                                float(value) for value in policy_observation[86:94]
                            ],
                            "receiver_ee_pose_robot": [
                                float(value) for value in policy_observation[102:109]
                            ],
                            "receiver_guidance": [
                                float(value) for value in policy_observation[111:117]
                            ],
                            "exit_target": [
                                float(value)
                                for value in measurement.get(
                                    "exit_target", torch.zeros((1, 3))
                                )[0]
                            ],
                            "exit_position": [
                                float(value)
                                for value in measurement.get(
                                    "exit_position", torch.zeros((1, 3))
                                )[0]
                            ],
                            "receiver_distance_m": float(
                                measurement.get(
                                    "receiver_distance", torch.tensor([float("nan")])
                                )[0]
                            ),
                            "receiver_contacts": [
                                float(value)
                                for value in measurement.get(
                                    "receiver_jaw_forces", torch.zeros((1, 2))
                                )[0]
                            ],
                            "giver_tissue_force_n": float(
                                measurement.get(
                                    "giver_tissue_force",
                                    torch.tensor([float("nan")]),
                                )[0]
                            ),
                            "giver_shaft_wrist_force_n": float(
                                measurement.get(
                                    "giver_all_links_tissue_force",
                                    torch.tensor([float("nan")]),
                                )[0]
                            ),
                            "receiver_tissue_force_n": float(
                                measurement.get(
                                    "receiver_tissue_force",
                                    torch.tensor([float("nan")]),
                                )[0]
                            ),
                            "receiver_shaft_wrist_force_n": float(
                                measurement.get(
                                    "receiver_all_links_tissue_force",
                                    torch.tensor([float("nan")]),
                                )[0]
                            ),
                            "custody_owner": int(
                                state.get("custody_owner", torch.tensor([-1]))[0]
                            ),
                            "giver_regrasp_stage": int(
                                state.get(
                                    "giver_regrasp_stage", torch.tensor([-1])
                                )[0]
                            ),
                            "tract_support_active": bool(
                                state.get(
                                    "tract_support_active", torch.tensor([False])
                                )[0]
                            ),
                            "tract_support_event_count": int(
                                state.get(
                                    "tract_support_event_count", torch.tensor([0])
                                )[0]
                            ),
                            "drive_rotation_deg": float(
                                state.get(
                                    "drive_rotation_deg",
                                    torch.tensor([float("nan")]),
                                )[0]
                            ),
                            "receiver_curve_rotation_deg": float(
                                state.get(
                                    "receiver_curve_rotation_deg",
                                    torch.tensor([0.0]),
                                )[0]
                            ),
                            "receiver_curve_center_error_m": float(
                                state.get(
                                    "receiver_curve_center_error",
                                    torch.tensor([0.0]),
                                )[0]
                            ),
                            "action": [float(value) for value in actions[0]],
                        },
                        sort_keys=True,
                    )
                )
            previous_trace_phase = trace_phase
            if not bool(torch.any(dones)):
                continue
            termination_manager = env.unwrapped.termination_manager
            success_mask = termination_manager.get_term("success")
            hard_mask = termination_manager.get_term("hard_failure")
            timeout_mask = termination_manager.get_term("time_out")
            pullout = "Puncture-Pullout" in args.task
            through_puncture = "Through-Puncture" in args.task
            if pullout:
                receipt_attribute = "_dr_anmar_last_successful_pullout"
            elif through_puncture:
                receipt_attribute = "_dr_anmar_last_successful_through_puncture"
            else:
                receipt_attribute = "_dr_anmar_last_successful_entry"
            success_receipts = getattr(env.unwrapped, receipt_attribute, None)
            hard_receipts = getattr(
                env.unwrapped, "_dr_anmar_last_hard_failures", None
            )
            hard_evidence = getattr(
                env.unwrapped, "_dr_anmar_last_hard_failure_evidence", None
            )
            for env_index in torch.nonzero(dones, as_tuple=False).squeeze(-1).tolist():
                if completed >= args.episodes:
                    break
                completed += 1
                if bool(success_mask[env_index]):
                    successes += 1
                    receipt = success_receipts[env_index]
                    entry_errors.append(float(receipt["entry_error_m"]))
                    tangent_errors.append(float(receipt["tangent_error_deg"]))
                    plane_errors.append(float(receipt["plane_error_deg"]))
                    embedded_depths.append(
                        float(
                            receipt[
                                "embedded_arc_length_m"
                                if through_puncture or pullout
                                else "embedded_depth_m"
                            ]
                        )
                    )
                    normalized_peak_forces.append(
                        float(receipt["peak_force_n"])
                        / float(receipt["sampled_puncture_force_n"])
                    )
                    backend_revisions.add(str(receipt["backend_revision"]))
                    backend_hashes.add(str(receipt["backend_implementation_sha256"]))
                    if through_puncture or pullout:
                        event_counts.append(int(receipt["entry_event_count"]))
                        exit_event_counts.append(int(receipt["exit_event_count"]))
                        right_underside_event_counts.append(
                            int(receipt["right_underside_event_count"])
                        )
                        exit_errors.append(float(receipt["exit_error_m"]))
                        exposed_fractions.append(float(receipt["exposed_fraction"]))
                        exposed_arc_lengths.append(
                            float(receipt["exposed_arc_length_m"])
                        )
                        if pullout:
                            receiver_pull_steps.append(
                                int(receipt["receiver_pull_steps"])
                            )
                            receiver_curve_rotations.append(
                                float(receipt["receiver_curve_rotation_deg"])
                            )
                            receiver_curve_center_errors.append(
                                float(receipt["receiver_curve_center_error_m"])
                            )
                    else:
                        event_counts.append(int(receipt["event_count"]))
                elif bool(hard_mask[env_index]):
                    hard_failures += 1
                    flags = hard_receipts[env_index] if hard_receipts else ("unknown",)
                    if hard_evidence:
                        print(
                            "[DR_ANMAR_TISSUE_HARD_FAILURE] "
                            + json.dumps(
                                {
                                    "flags": list(flags),
                                    **hard_evidence[env_index],
                                },
                                sort_keys=True,
                            )
                        )
                    for flag in flags:
                        failure_flags[flag] = failure_flags.get(flag, 0) + 1
                elif bool(timeout_mask[env_index]):
                    timeouts += 1
                else:
                    failure_flags["unclassified_done"] = (
                        failure_flags.get("unclassified_done", 0) + 1
                    )
        if completed != args.episodes:
            raise RuntimeError(
                f"simulation stopped after {completed}/{args.episodes} episodes"
            )
        report = {
            "schema": "dr.anmar.tissue-entry-isolated-evaluation.v1",
            "task": args.task,
            "policy": policy_name,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "checkpoint_sha256": _sha256(checkpoint) if checkpoint else None,
            "seed": args.seed,
            "episodes": completed,
            "successes": successes,
            "success_rate": successes / completed,
            "hard_safety_failures": hard_failures,
            "timeouts": timeouts,
            "failure_flags": failure_flags,
            "control_steps": control_steps,
            "diagnostic_giver_base_lift_m": args.giver_base_lift_m,
            "diagnostic_giver_joint_positions": diagnostic_joint_positions,
            "diagnostic_rcm_follow_gain": args.rcm_follow_gain,
            "diagnostic_ik_orientation_weight": args.ik_orientation_weight,
            "diagnostic_receiver_ik_orientation_weight": (
                args.receiver_ik_orientation_weight
            ),
            "diagnostic_episode_length_s": args.episode_length_s,
            "entry_error_m_max": max(entry_errors, default=None),
            "entry_error_m_mean": (
                sum(entry_errors) / len(entry_errors) if entry_errors else None
            ),
            "tangent_error_deg_max": max(tangent_errors, default=None),
            "plane_error_deg_max": max(plane_errors, default=None),
            "embedded_depth_m_min": min(embedded_depths, default=None),
            "embedded_depth_m_max": max(embedded_depths, default=None),
            "normalized_peak_force_max": max(normalized_peak_forces, default=None),
            "normalized_peak_force_mean": (
                sum(normalized_peak_forces) / len(normalized_peak_forces)
                if normalized_peak_forces
                else None
            ),
            "exactly_one_event_per_success": bool(event_counts)
            and all(count == 1 for count in event_counts),
            "custody_model": "pregrasped_pose_coupling",
            "evidence_level": "simulator_engineering_only",
            "clinical_validation": False,
            "backend_revisions": sorted(backend_revisions),
            "backend_implementation_sha256": (
                next(iter(backend_hashes)) if len(backend_hashes) == 1 else None
            ),
        }
        if "Through-Puncture" in args.task or "Puncture-Pullout" in args.task:
            report.update(
                {
                    "schema": "dr.anmar.tissue-through-isolated-evaluation.v1",
                    "exit_error_m_max": max(exit_errors, default=None),
                    "exposed_fraction_min": min(exposed_fractions, default=None),
                    "exposed_arc_length_m_min": min(
                        exposed_arc_lengths, default=None
                    ),
                    "embedded_arc_length_m_min": min(
                        embedded_depths, default=None
                    ),
                    "embedded_arc_length_m_max": max(
                        embedded_depths, default=None
                    ),
                    "exactly_one_exit_event_per_success": bool(exit_event_counts)
                    and all(count == 1 for count in exit_event_counts),
                    "exactly_one_right_underside_event_per_success": bool(
                        right_underside_event_counts
                    )
                    and all(count == 1 for count in right_underside_event_counts),
                }
            )
        if "Puncture-Pullout" in args.task:
            report.update(
                {
                    "schema": "dr.anmar.tissue-puncture-pullout-isolated-evaluation.v1",
                    "custody_model": (
                        "bilateral_force_or_calibrated_geometry_then_receiver_pose_coupling"
                    ),
                    "complete_clearance_per_success": bool(exposed_fractions)
                    and all(value >= 0.995 for value in exposed_fractions)
                    and all(value <= 0.0001 for value in embedded_depths),
                    "receiver_pull_steps_min": min(
                        receiver_pull_steps, default=None
                    ),
                    "receiver_curve_rotation_deg_min": min(
                        receiver_curve_rotations, default=None
                    ),
                    "receiver_curve_center_error_m_max": max(
                        receiver_curve_center_errors, default=None
                    ),
                }
            )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_name(f".{args.report.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, args.report)
        print("[DR_ANMAR_TISSUE_ENTRY_EVALUATION] " + json.dumps(report, sort_keys=True))
        if video_path is not None:
            print(f"[DR_ANMAR_TISSUE_VIDEO] {video_path}")
        return 0
    finally:
        if video_writer is not None:
            video_writer.close()
        env.close()


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception:
        import traceback

        traceback.print_exc()
        os._exit(1)
    simulation_app.close()
    raise SystemExit(exit_code)
