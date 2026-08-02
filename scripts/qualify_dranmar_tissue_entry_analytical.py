#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Run the fixed-domain analytical tissue-entry gate before PPO training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument(
    "--task", default="DrAnmar-Penetrate-Tissue-Needle-PSM-IK-Rel-Play-v0"
)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--diagnose-hard-failure", action="store_true")
parser.add_argument(
    "--zero-controller",
    action="store_true",
    help="Diagnostic only: hold a zero task-space command after reset.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import orbit.surgical.tasks  # noqa: E402, F401
from orbit.surgical.tasks.surgical import mdp_common  # noqa: E402
from orbit.surgical.tasks.surgical.penetration.residual_model import (  # noqa: E402
    PenetrationAnalyticController,
)


def main() -> int:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)
    env_cfg.seed = 0
    if args.diagnose_hard_failure:
        env_cfg.terminations.hard_failure = None
    env = gym.make(args.task, cfg=env_cfg)
    controller = PenetrationAnalyticController().to(env.unwrapped.device)
    observation, _ = env.reset()
    reset_rejections = 0
    warmup_steps = 0
    reset_trace = []
    while warmup_steps < 128:
        warmup_steps += 1
        with torch.inference_mode():
            observation, _, terminated, truncated, _ = env.step(
                torch.zeros((1, 6), device=env.unwrapped.device)
            )
        state = env.unwrapped._dr_anmar_penetration_state
        if warmup_steps <= 40:
            reset_trace.append(
                {
                    "step": warmup_steps,
                    "settle": int(state["settle_control_steps"][0]),
                    "stage": int(state["grasp_attach_stage"][0]),
                    "jaw_n": state["jaw_forces"][0].detach().cpu().tolist(),
                    "needle_pos_w": env.unwrapped.scene["needle"].data.root_pos_w[0]
                    .detach()
                    .cpu()
                    .tolist(),
                }
            )
        if bool(terminated[0] or truncated[0]):
            reset_rejections += 1
            continue
        if int(state["settle_control_steps"][0]) == 0 and bool(
            state["custody_valid"][0]
        ):
            break
    else:
        state = env.unwrapped._dr_anmar_penetration_state
        print(
            "[DR_ANMAR_TISSUE_ENTRY_RESET_DIAGNOSTIC] "
            + json.dumps(
                {
                    "jaw_contact_forces_n": state["jaw_forces"][0]
                    .detach()
                    .cpu()
                    .tolist(),
                    "custody_valid": bool(state["custody_valid"][0]),
                    "grasp_position_error_m": float(state["grasp_position_error"][0]),
                    "grasp_angle_error_deg": float(state["grasp_angle_error_deg"][0]),
                    "settle_control_steps": int(state["settle_control_steps"][0]),
                    "hard_failures": sorted(state["gates"][0].hard_failures),
                    "reset_trace": reset_trace,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        raise RuntimeError("could not obtain settled bilateral needle custody in 128 steps")
    robot = env.unwrapped.scene["robot"]
    initial_joint_positions = robot.data.joint_pos[0, :6].detach().cpu().tolist()
    initial_needle_quaternion = observation["policy"][0, 26:30].detach().cpu().tolist()
    initial_end_effector_quaternion = observation["policy"][0, 19:23].detach().cpu().tolist()
    initial_end_effector_position = observation["policy"][0, 16:19].detach().cpu().tolist()
    initial_needle_tip_position_robot = observation["policy"][0, 23:26].detach().cpu().tolist()
    initial_entry_goal_robot = observation["policy"][0, 36:39].detach().cpu().tolist()
    completed_steps = 0
    terminated_by: list[str] = []
    max_rotation_action = 0.0
    translation_saturation_steps = 0
    rotation_saturation_steps = 0
    phase_names = ("approach", "align", "indent", "puncture", "stabilize")
    phase_first_control_step: dict[str, int] = {}
    phase_first_tip_world_m: dict[str, list[float]] = {}
    phase_first_ee_robot_m: dict[str, list[float]] = {}
    phase_visit_steps = {name: 0 for name in phase_names}
    min_entry_error_m = float(state["measurement"]["entry_error"][0])
    min_tangent_error_deg = float(state["measurement"]["tangent_error"][0])
    min_plane_error_deg = float(state["measurement"]["plane_error"][0])
    max_indentation_m = float(state["measurement"]["indentation"][0])
    action = torch.zeros((1, 6), device=env.unwrapped.device)
    try:
        for completed_steps in range(1, args.steps + 1):
            if not simulation_app.is_running():
                break
            with torch.inference_mode():
                if args.zero_controller:
                    action.zero_()
                else:
                    action, _, _ = controller(observation["policy"])
                max_rotation_action = max(
                    max_rotation_action, float(torch.max(torch.abs(action[:, 3:])))
                )
                translation_saturation_steps += int(
                    bool(torch.any(torch.abs(action[:, :3]) >= 0.999))
                )
                rotation_saturation_steps += int(
                    bool(torch.any(torch.abs(action[:, 3:]) >= 0.999))
                )
                observation, _, terminated, truncated, _ = env.step(action)
            step_state = env.unwrapped._dr_anmar_penetration_state
            terminated_now = bool(terminated[0] or truncated[0])
            live_receipt = getattr(env.unwrapped, "_dr_anmar_last_successful_entry", None)
            phase_index = (
                int(live_receipt[0]["phase"])
                if terminated_now and live_receipt
                else int(step_state["phase"][0])
            )
            phase_name = phase_names[phase_index]
            phase_visit_steps[phase_name] += 1
            phase_first_control_step.setdefault(phase_name, completed_steps)
            if phase_name not in phase_first_tip_world_m:
                phase_first_tip_world_m[phase_name] = (
                    step_state["measurement"]["tip_pos"][0].detach().cpu().tolist()
                )
                phase_ee_pos, _ = env.unwrapped.action_manager._terms[
                    "body_action"
                ]._compute_frame_pose()
                phase_first_ee_robot_m[phase_name] = (
                    phase_ee_pos[0].detach().cpu().tolist()
                )
            min_entry_error_m = min(
                min_entry_error_m,
                float(step_state["measurement"]["entry_error"][0]),
            )
            min_tangent_error_deg = min(
                min_tangent_error_deg,
                float(step_state["measurement"]["tangent_error"][0]),
            )
            min_plane_error_deg = min(
                min_plane_error_deg,
                float(step_state["measurement"]["plane_error"][0]),
            )
            max_indentation_m = max(
                max_indentation_m,
                float(step_state["measurement"]["indentation"][0]),
            )
            if terminated_now:
                terminated_by = [
                    name
                    for name in env.unwrapped.termination_manager.active_terms
                    if bool(env.unwrapped.termination_manager.get_term(name)[0])
                ]
                break

        state = env.unwrapped._dr_anmar_penetration_state
        action_term = env.unwrapped.action_manager._terms["body_action"]
        ik_ee_pos, _ = action_term._compute_frame_pose()
        ik_jacobian = action_term._compute_frame_jacobian()
        ik_singular_values = torch.linalg.svdvals(ik_jacobian[0])
        successful = "success" in terminated_by
        success_receipt = getattr(env.unwrapped, "_dr_anmar_last_successful_entry", None)
        evidence = success_receipt[0] if success_receipt else {
            "event_count": int(state["event_count"][0]),
            "representation_switch_count": int(state["representation_switch_count"][0]),
            "phase": int(state["phase"][0]),
            "hard_failures": tuple(sorted(state["gates"][0].hard_failures)),
            "entry_error_m": float(state["measurement"]["entry_error"][0]),
            "tangent_error_deg": float(state["measurement"]["tangent_error"][0]),
            "plane_error_deg": float(state["measurement"]["plane_error"][0]),
            "embedded_depth_m": float(state["measurement"]["embedded_depth"][0]),
            "peak_force_n": float(state["normal_force"][0]),
            "entry_position_m": tuple(
                float(value) for value in state["measurement"]["tip_pos"][0]
            ),
            "sampled_puncture_force_n": float(state["puncture_force_n"][0]),
            "accumulated_work_j": float(state["accumulated_work"][0]),
            "phase_sequence": tuple(state["gates"][0].phase_sequence),
            "custody_valid": bool(state["custody_valid"][0]),
            "custody_model": "pregrasped_pose_coupling",
            "rigid_needle_collisions_enabled": False,
            "backend_revision": state["backend_metadata"].revision,
            "backend_implementation_sha256": (
                state["backend_metadata"].implementation_sha256
            ),
        }
        fem_tissue = {}
        for tissue_name in ("tissue_left", "tissue_right"):
            tissue = env.unwrapped.scene[tissue_name]
            nodal_state = mdp_common.as_torch(tissue.data.nodal_state_w)
            default_state = mdp_common.as_torch(tissue.data.default_nodal_state_w)
            kinematic_target = mdp_common.as_torch(
                tissue.data.nodal_kinematic_target
            )
            fem_tissue[tissue_name] = {
                "node_count": int(nodal_state.shape[1]),
                "anchored_node_count": int(
                    torch.count_nonzero(kinematic_target[0, :, 3] == 0).item()
                ),
                "state_finite": bool(torch.isfinite(nodal_state).all()),
                "max_displacement_m": float(
                    torch.linalg.vector_norm(
                        nodal_state[0, :, :3] - default_state[0, :, :3], dim=-1
                    )
                    .amax()
                    .item()
                ),
            }
        report = {
            "schema": "dr.anmar.tissue-entry-analytical-gate.v1",
            "task": args.task,
            "completed_steps": completed_steps,
            "warmup_steps": warmup_steps,
            "reset_rejections": reset_rejections,
            "terminated_by": terminated_by,
            "successful": successful,
            "event_count": evidence["event_count"],
            "representation_switch_count": evidence["representation_switch_count"],
            "backend_revision": evidence["backend_revision"],
            "backend_implementation_sha256": evidence[
                "backend_implementation_sha256"
            ],
            "phase": evidence["phase"],
            "hard_failures": list(evidence["hard_failures"]),
            "entry_error_m": evidence["entry_error_m"],
            "tangent_error_deg": evidence["tangent_error_deg"],
            "plane_error_deg": evidence["plane_error_deg"],
            "embedded_depth_m": evidence["embedded_depth_m"],
            "peak_force_n": evidence["peak_force_n"],
            "entry_target_world_m": state["measurement"]["target"][0]
            .detach()
            .cpu()
            .tolist(),
            "needle_tip_world_m": list(evidence["entry_position_m"]),
            "custody_valid": evidence["custody_valid"],
            "custody_model": evidence["custody_model"],
            "rigid_needle_collisions_enabled": evidence[
                "rigid_needle_collisions_enabled"
            ],
            "phase_sequence": list(evidence["phase_sequence"]),
            "phase_first_control_step": phase_first_control_step,
            "phase_first_tip_world_m": phase_first_tip_world_m,
            "phase_first_ee_robot_m": phase_first_ee_robot_m,
            "phase_visit_steps": phase_visit_steps,
            "min_entry_error_m": min_entry_error_m,
            "min_tangent_error_deg": min_tangent_error_deg,
            "min_plane_error_deg": min_plane_error_deg,
            "max_indentation_m": max_indentation_m,
            "sampled_puncture_force_n": evidence["sampled_puncture_force_n"],
            "accumulated_work_j": evidence["accumulated_work_j"],
            "wrench_finite": bool(torch.isfinite(state["wrench"]).all()),
            "fem_tissue": fem_tissue,
            "max_rotation_action": max_rotation_action,
            "translation_saturation_steps": translation_saturation_steps,
            "rotation_saturation_steps": rotation_saturation_steps,
            "last_action": action[0].detach().cpu().tolist(),
            "ik_current_ee_position_robot_m": ik_ee_pos[0].detach().cpu().tolist(),
            "ik_desired_ee_position_robot_m": action_term._ik_controller.ee_pos_des[0]
            .detach()
            .cpu()
            .tolist(),
            "ik_processed_action": action_term._processed_actions[0]
            .detach()
            .cpu()
            .tolist(),
            "ik_jacobian_singular_values": ik_singular_values.detach().cpu().tolist(),
            "needle_quaternion_xyzw": observation["policy"][0, 26:30].detach().cpu().tolist(),
            "surface_normal_robot": observation["policy"][0, 43:46].detach().cpu().tolist(),
            "initial_joint_positions": initial_joint_positions,
            "initial_needle_quaternion_xyzw": initial_needle_quaternion,
            "initial_end_effector_quaternion_xyzw": initial_end_effector_quaternion,
            "initial_end_effector_position_robot_m": initial_end_effector_position,
            "initial_needle_tip_position_robot_m": initial_needle_tip_position_robot,
            "initial_entry_goal_robot_m": initial_entry_goal_robot,
            "final_end_effector_quaternion_xyzw": observation["policy"][0, 19:23]
            .detach()
            .cpu()
            .tolist(),
            "final_joint_positions": robot.data.joint_pos[0, :6].detach().cpu().tolist(),
            "final_joint_position_targets": robot.data.joint_pos_target[0, :6]
            .detach()
            .cpu()
            .tolist(),
            "final_applied_joint_effort": robot.data.applied_torque[0, :6]
            .detach()
            .cpu()
            .tolist(),
            "final_joint_velocity": robot.data.joint_vel[0, :6]
            .detach()
            .cpu()
            .tolist(),
            "jaw_net_contact_force_n": [
                float(mdp_common.contact_force_magnitude(env.unwrapped, sensor_name)[0])
                for sensor_name in (
                    "jaw_1_needle_contact",
                    "jaw_2_needle_contact",
                )
            ],
            "robot_contact_bodies": list(
                env.unwrapped.scene.sensors["robot_contacts"].body_names
            )
            if "robot_contacts" in env.unwrapped.scene.sensors
            else [],
            "robot_net_contact_forces_n": (
                torch.linalg.vector_norm(
                    mdp_common.as_torch(
                        env.unwrapped.scene.sensors["robot_contacts"].data.net_forces_w
                    )[0],
                    dim=-1,
                )
                .detach()
                .cpu()
                .tolist()
                if "robot_contacts" in env.unwrapped.scene.sensors
                else []
            ),
            "qualified_for_ppo": bool(
                successful
                and evidence["event_count"] == 1
                and evidence["representation_switch_count"] == 1
                and not evidence["hard_failures"]
            ),
            "evidence_level": "simulator_engineering_only",
            "clinical_validation": False,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_name(f".{args.report.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, args.report)
        print(
            "[DR_ANMAR_TISSUE_ENTRY_ANALYTICAL] " + json.dumps(report, sort_keys=True),
            flush=True,
        )
        return 0 if report["qualified_for_ppo"] else 1
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = main()
    if exit_code != 0:
        # Isaac's close path exits the process with status zero on this runtime,
        # which would silently open the PPO gate. The environment and receipt
        # are already closed/flushed by main, so preserve the gate status.
        os._exit(exit_code)
    simulation_app.close()
    raise SystemExit(0)
