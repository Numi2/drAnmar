# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Clinician-facing capability adapter for NVIDIA Isaac for Healthcare workflows.

Dr.Anmar owns study intent, pedagogy, provenance, and evidence packaging. The
underlying NVIDIA workflows continue to own sensor physics, policy runtimes,
hardware communication, synthetic data, and deployment infrastructure.
"""

from __future__ import annotations

import json
import hashlib
import os
import shutil
from pathlib import Path
from typing import Any


APP_ROOT = Path(os.environ.get("DR_ANMAR_ROOT", Path.home() / ".local/share/dr-anmar")).expanduser()
I4H_ROOT = Path(os.environ.get("DR_ANMAR_I4H_ROOT", APP_ROOT / "vendor/i4h-workflows")).expanduser()
I4H_RELEASE = os.environ.get("DR_ANMAR_I4H_RELEASE", "v0.6.0")
I4H_RELEASE_COMMIT = "8b03d55ecb647a43af54470b27bd09a239870aaf"
HOLOHUB_CLI_COMMIT = os.environ.get(
    "DR_ANMAR_HOLOHUB_CLI_COMMIT",
    "f7e791dac061e01c560d3a2c5b7da82350915b69",
)


def runtime_prerequisites() -> dict[str, Any]:
    docker = shutil.which("docker")
    rti_value = os.environ.get("RTI_LICENSE_FILE")
    rti_path = Path(rti_value).expanduser() if rti_value else I4H_ROOT / "rti/rti_license.dat"
    cli_pin_path = I4H_ROOT / "tools/utilities/cli/.cli_commit_hash"
    try:
        installed_cli_commit = cli_pin_path.read_text().strip()
    except OSError:
        installed_cli_commit = None
    try:
        workflow_head = (I4H_ROOT / ".git/HEAD").read_text().strip()
        if workflow_head.startswith("ref: "):
            workflow_head = (I4H_ROOT / ".git" / workflow_head[5:]).read_text().strip()
    except OSError:
        workflow_head = None
    return {
        "container_runtime": {"ready": bool(docker), "path": docker, "label": "Docker Engine"},
        "nvidia_gpu_device": {"ready": Path("/dev/nvidia0").exists(), "path": "/dev/nvidia0"},
        "rti_dds_license": {"ready": rti_path.is_file(), "path": str(rti_path)},
        "holohub_cli_pin": {
            "ready": installed_cli_commit == HOLOHUB_CLI_COMMIT,
            "expected_commit": HOLOHUB_CLI_COMMIT,
            "installed_commit": installed_cli_commit,
        },
        "workflow_release_pin": {
            "ready": workflow_head == I4H_RELEASE_COMMIT,
            "release": I4H_RELEASE,
            "expected_commit": I4H_RELEASE_COMMIT,
            "installed_commit": workflow_head,
        },
    }


WORKFLOW_BINDINGS = {
    "robotic_surgery": {
        "title": "Robotic surgery",
        "directory": "workflows/robotic_surgery",
        "provides": ["dvrk", "star", "surgical_tasks", "reinforcement_learning", "imitation_learning"],
        "doctor_summary": "Practise surgical subtasks, collect demonstrations, and evaluate supervised autonomy.",
        "doctor_default_mode": "lift_needle_organs",
    },
    "robotic_ultrasound": {
        "title": "Robotic ultrasound",
        "directory": "workflows/robotic_ultrasound",
        "provides": ["b_mode_ultrasound", "probe_pose", "acoustic_simulation", "holoscan", "pi0", "groot"],
        "doctor_summary": "Learn probe positioning and image acquisition with simulated B-mode feedback.",
        "doctor_default_mode": "teleop_with_ultrasound",
    },
    "telesurgery": {
        "title": "Telesurgery",
        "directory": "workflows/telesurgery",
        "provides": ["xr", "haptics", "video_streaming", "rti_dds", "hardware_in_the_loop"],
        "doctor_summary": "Study remote control, latency, haptic feedback, handover, and sim-to-real interfaces.",
        "doctor_default_mode": None,
    },
    "so_arm_starter": {
        "title": "SO-ARM + GR00T starter",
        "directory": "workflows/so_arm_starter",
        "provides": ["wrist_camera", "room_camera", "teleoperation", "lerobot", "groot"],
        "doctor_summary": "Learn the full collect, train, evaluate, and deploy loop on an accessible robot arm.",
        "doctor_default_mode": "sim_keyboard",
    },
    "rheo": {
        "title": "Rheo precision manipulation",
        "directory": "workflows/rheo",
        "provides": ["trocar_assembly", "bimanual_manipulation", "groot_n1_6", "online_rl", "cosmos_transfer_2_5"],
        "doctor_summary": "Use NVIDIA's trocar-assembly and precision-manipulation references as expert research starting points.",
        "doctor_default_mode": None,
        "expert_source_only": True,
    },
    "agentic": {
        "title": "Agentic data and policy pipeline",
        "directory": "workflows/agentic",
        "provides": ["teleoperation", "mimic", "vlm_annotation", "lerobot", "groot_n1_7", "openpi", "rollout_validation"],
        "doctor_summary": "Move a reviewed study from demonstrations through curation, policy adaptation, and closed-loop evaluation.",
        "doctor_default_mode": None,
        "expert_source_only": True,
    },
}


def _mode_category(mode_id: str, description: str) -> str:
    words = f"{mode_id} {description}".lower()
    if any(word in words for word in ("hardware-in-the-loop", "real_deploy", "teleop_real", "clarius", "realsense")):
        return "hardware"
    if mode_id.startswith("train"):
        return "training"
    if mode_id in {"visualization"}:
        return "visualization"
    if any(word in mode_id for word in ("policy", "play_rl", "evaluate")):
        return "policy"
    if mode_id in {"replay", "convert", "convert_hdf5"}:
        return "data"
    if mode_id in {"find_ports", "find_cameras", "calibrate_follower", "calibrate_leader", "download_model", "login_hf"}:
        return "setup"
    return "simulation"


def workflow_modes(workflow_id: str) -> dict[str, Any]:
    """Read launch choices from NVIDIA's pinned workflow metadata.

    The UI deliberately exposes only argument-free, non-privileged modes. This
    keeps device access, interactive authentication, and arbitrary command
    arguments outside the clinician-facing launch surface.
    """

    definition = WORKFLOW_BINDINGS.get(workflow_id)
    if definition is None:
        raise KeyError(workflow_id)
    workflow_root = I4H_ROOT / definition["directory"]
    metadata_path = workflow_root / "metadata.json"
    if not metadata_path.is_file():
        return {
            "default_mode": definition.get("doctor_default_mode"),
            "metadata_ready": False,
            "metadata_path": str(metadata_path),
            "modes": [],
            "expert_source_only": bool(definition.get("expert_source_only")),
            "blocked_reason": (
                "This v0.6 workflow has no HoloHub metadata contract; use its reviewed scripts outside the clinician launcher."
                if definition.get("expert_source_only")
                else None
            ),
        }
    try:
        metadata_bytes = metadata_path.read_bytes()
        metadata = json.loads(metadata_bytes)
    except (OSError, ValueError):
        return {
            "default_mode": definition.get("doctor_default_mode"),
            "metadata_ready": False,
            "metadata_path": str(metadata_path),
            "modes": [],
            "discovery_error": "Workflow metadata is unreadable; launch is disabled.",
        }
    official = metadata.get("workflow", {})
    official_modes = official.get("modes") if isinstance(official, dict) else None
    if not isinstance(official_modes, dict):
        return {
            "default_mode": None,
            "metadata_ready": False,
            "metadata_path": str(metadata_path),
            "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            "modes": [],
            "discovery_error": "Workflow metadata schema changed; launch is disabled until the adapter is reviewed.",
        }
    prerequisites = runtime_prerequisites()
    modes = []
    rejected_modes = []
    for mode_id, mode_definition in official_modes.items():
        if not isinstance(mode_id, str) or not mode_id or not isinstance(mode_definition, dict):
            rejected_modes.append(str(mode_id))
            continue
        description = str(mode_definition.get("description") or mode_id.replace("_", " ").title())
        run = mode_definition.get("run", {})
        if not isinstance(run, dict) or not isinstance(run.get("docker_run_args", []), list):
            rejected_modes.append(mode_id)
            continue
        docker_args = [str(value) for value in run.get("docker_run_args", [])]
        lower_description = description.lower()
        requires_arguments = "requires --run-args" in lower_description
        requires_hardware = any(
            value in lower_description
            for value in ("hardware-in-the-loop", "real-to-real", "real hardware", "clarius", "realsense")
        ) or any(value == "--privileged" or value.startswith("/dev") or "/dev:" in value for value in docker_args)
        interactive_auth = mode_id == "login_hf"
        category = _mode_category(mode_id, description)
        requires_rti = (
            workflow_id == "robotic_ultrasound" and category not in {"training", "data"}
        ) or (workflow_id == "so_arm_starter" and mode_id in {"sim_env", "policy"})
        launchable = not requires_arguments and not requires_hardware and not interactive_auth and category != "setup"
        missing_prerequisites = []
        if launchable and not prerequisites["container_runtime"]["ready"]:
            missing_prerequisites.append("Docker Engine")
        if launchable and requires_rti and not prerequisites["rti_dds_license"]["ready"]:
            missing_prerequisites.append("RTI Connext DDS license")
        if launchable:
            blocked_reason = None
        elif requires_hardware:
            blocked_reason = "Physical-device mode requires an approved hardware setup and safety review."
        elif requires_arguments:
            blocked_reason = "This advanced mode needs a reviewed file path or custom argument."
        elif interactive_auth:
            blocked_reason = "Interactive account login is kept outside the web launcher."
        else:
            blocked_reason = "Run this setup utility from the workstation terminal."
        modes.append(
            {
                "id": mode_id,
                "title": mode_id.replace("_", " ").title(),
                "description": description,
                "category": category,
                "launchable": launchable,
                "requires_hardware": requires_hardware,
                "requires_arguments": requires_arguments,
                "requires_rti": requires_rti,
                "blocked_reason": blocked_reason,
                "recommended": mode_id == definition.get("doctor_default_mode"),
                "launch_ready": launchable and not missing_prerequisites,
                "missing_prerequisites": missing_prerequisites,
                "runtime_validated": False,
            }
        )
    modes.sort(key=lambda item: (not item["recommended"], not item["launchable"], item["category"], item["title"]))
    return {
        "default_mode": definition.get("doctor_default_mode") or official.get("default_mode"),
        "official_default_mode": official.get("default_mode"),
        "metadata_ready": True,
        "metadata_path": str(metadata_path),
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "rejected_modes": rejected_modes,
        "discovery_error": f"{len(rejected_modes)} malformed modes were disabled" if rejected_modes else None,
        "modes": modes,
    }


MODALITY_CATALOG = (
    {
        "id": "stereo_endoscope",
        "title": "Stereo endoscope",
        "group": "Vision",
        "doctor_value": "Shows the same anatomy from two nearby viewpoints, like using both eyes for depth.",
        "outputs": ["left_rgb", "right_rgb", "calibration", "timestamps"],
        "provider": "Dr.Anmar surgical twin on Isaac Lab CameraCfg",
        "binding": "robotic_surgery",
        "native": True,
    },
    {
        "id": "depth_segmentation_pointcloud",
        "title": "Depth, segmentation, and point cloud",
        "group": "Vision",
        "doctor_value": "Separates appearance from geometry so a policy can reason about where tissue and tools are.",
        "outputs": ["depth_m", "semantic_id", "point_cloud_xyz_m", "camera_intrinsics"],
        "provider": "Isaac Lab camera and Replicator outputs",
        "binding": "robotic_surgery",
        "native": True,
    },
    {
        "id": "wrist_cameras",
        "title": "Instrument wrist cameras",
        "group": "Vision",
        "doctor_value": "Provides a close tool-centred view when the main endoscope is occluded or distant.",
        "outputs": ["wrist_1_rgb", "wrist_2_rgb", "camera_pose"],
        "provider": "Isaac Lab cameras attached to task-native tool-tip links",
        "binding": "robotic_surgery",
        "native": True,
    },
    {
        "id": "robot_anatomy_pose",
        "title": "Tool, robot, object, and anatomy pose",
        "group": "Robot state",
        "doctor_value": "Provides the exact simulated geometry behind what the camera shows.",
        "outputs": ["joint_state", "tool_pose", "object_pose", "anatomy_pose"],
        "provider": "Isaac Lab articulation and asset state",
        "binding": "robotic_surgery",
        "native": True,
    },
    {
        "id": "force_torque_deformation",
        "title": "Contact, force, torque, and deformation",
        "group": "Physical interaction",
        "doctor_value": "Reveals whether the robot completed the motion gently, not only whether it arrived.",
        "outputs": ["contact_force", "joint_torque", "nodal_displacement", "stress", "deformation_gradient"],
        "provider": "Isaac Lab contact, articulation, and deformable-object tensors",
        "binding": "robotic_surgery",
        "native": True,
    },
    {
        "id": "ultrasound",
        "title": "Simulated B-mode ultrasound",
        "group": "Medical imaging",
        "doctor_value": "Links probe motion to image quality and anatomy visibility without using a patient.",
        "outputs": ["b_mode", "probe_pose", "acoustic_parameters", "quality_metrics"],
        "provider": "Isaac for Healthcare robotic ultrasound workflow",
        "binding": "robotic_ultrasound",
        "native": False,
    },
    {
        "id": "operator_gaze_inputs",
        "title": "Operator gaze and inputs",
        "group": "Human factors",
        "doctor_value": "Captures what the operator attended to and how each command was issued.",
        "outputs": ["gaze_uv", "gaze_source", "input_source", "actions", "interventions"],
        "provider": "Dr.Anmar study schema with browser, XR, or external tracker adapters",
        "binding": "telesurgery",
        "native": True,
    },
    {
        "id": "procedure_annotations",
        "title": "Procedure phases and events",
        "group": "Clinical meaning",
        "doctor_value": "Turns a long trajectory into setup, approach, grasp, manipulation, and recovery examples.",
        "outputs": ["phase_code", "event_code", "annotations"],
        "provider": "Dr.Anmar clinician annotation vocabulary",
        "binding": "robotic_surgery",
        "native": True,
    },
    {
        "id": "haptic_xr",
        "title": "Haptic and XR teleoperation",
        "group": "Teleoperation",
        "doctor_value": "Lets experts demonstrate natural two-handed control while studying latency and handover.",
        "outputs": ["controller_pose", "buttons", "haptic_command", "latency", "handover_events"],
        "provider": "Isaac for Healthcare telesurgery workflow and RTI DDS",
        "binding": "telesurgery",
        "native": False,
    },
)


POLICY_STARTING_POINTS = (
    {
        "id": "behavior_cloning",
        "title": "Behavior Cloning",
        "analogy": "A resident learns by watching complete expert examples.",
        "inputs": ["selected study modalities", "robot state", "expert actions"],
        "provider": "ORBIT-Surgical / Dr.Anmar",
    },
    {
        "id": "pi0",
        "title": "π₀",
        "analogy": "Start from a general vision-language-action model, then adapt it to a defined procedure.",
        "inputs": ["language goal", "camera observations", "robot state"],
        "provider": "Isaac for Healthcare workflow integration",
    },
    {
        "id": "groot",
        "title": "GR00T N1.7",
        "analogy": "Use a pretrained robot foundation model as a starting point instead of learning everything from zero.",
        "inputs": ["language goal", "multi-camera video", "embodiment state/action mapping"],
        "provider": "NVIDIA Isaac GR00T through Isaac for Healthcare v0.6 agentic/SO-ARM workflows",
    },
    {
        "id": "reinforcement_learning",
        "title": "Reinforcement Learning",
        "analogy": "The robot practises repeatedly against an explicit simulator score.",
        "inputs": ["state or observations", "reward", "termination", "safety constraints"],
        "provider": "Isaac Lab / ORBIT-Surgical",
    },
)


def platform_payload(anatomy_root: Path) -> dict[str, Any]:
    workflows = []
    for workflow_id, definition in WORKFLOW_BINDINGS.items():
        path = I4H_ROOT / definition["directory"]
        launch = workflow_modes(workflow_id)
        workflows.append(
            {
                "id": workflow_id,
                **definition,
                "installed": path.is_dir(),
                "source_ready": path.is_dir(),
                "runtime_validated": False,
                "path": str(path),
                "inspect_command": (
                    f"Read {definition['directory']}/README.md"
                    if definition.get("expert_source_only")
                    else f"./i4h modes {workflow_id}"
                ),
                **launch,
            }
        )
    i4h_cli = I4H_ROOT / "i4h"
    return {
        "schema": "dr.anmar.isaac-healthcare-capabilities.v2",
        "strategy": "wrap_not_duplicate",
        "i4h_release": I4H_RELEASE,
        "i4h_release_commit": I4H_RELEASE_COMMIT,
        "i4h_root": str(I4H_ROOT),
        "i4h_cli_ready": i4h_cli.is_file(),
        "runtime_prerequisites": runtime_prerequisites(),
        "workflows": workflows,
        "modalities": [dict(item) for item in MODALITY_CATALOG],
        "policies": [dict(item) for item in POLICY_STARTING_POINTS],
        "maisi_anatomy_ready": anatomy_root.is_dir() and any(anatomy_root.iterdir()),
        "runtime_boundary": {
            "dr_anmar_owns": ["pedagogy", "study design", "annotations", "provenance", "evidence review"],
            "isaac_for_healthcare_owns": [
                "sensor physics",
                "medical workflow runtimes",
                "synthetic data",
                "policy integrations",
                "DDS and hardware-in-the-loop",
            ],
        },
    }


def study_manifest(
    *,
    study_id: str,
    title: str,
    clinical_question: str,
    task: str,
    modalities: list[str],
    policy: str,
    teleoperation: str,
    created_at: str,
    source_revision: str | None,
) -> dict[str, Any]:
    modality_map = {item["id"]: item for item in MODALITY_CATALOG}
    policy_map = {item["id"]: item for item in POLICY_STARTING_POINTS}
    selected = [modality_map[item] for item in modalities]
    workflows = sorted({item["binding"] for item in selected})
    return {
        "schema": "dr.anmar.multimodal-study.v1",
        "study_id": study_id,
        "title": title,
        "clinical_question": clinical_question,
        "simulation_only": True,
        "clinical_use": False,
        "created_at": created_at,
        "source_revision": source_revision,
        "task": task,
        "modalities": selected,
        "policy_starting_point": policy_map[policy],
        "teleoperation": teleoperation,
        "underlying_workflows": workflows,
        "workflow_inspection": [f"./i4h modes {workflow}" for workflow in workflows],
        "procedure_vocabulary": {
            "phases": ["setup", "approach", "grasp", "manipulation", "recovery"],
            "events": ["target_visible", "contact", "grasp", "task_complete", "handoff", "safety_review"],
        },
        "study_steps": [
            "Review the clinical question and selected signals.",
            "Calibrate cameras, robot, anatomy, and optional operator devices.",
            "Record complete expert demonstrations with phase and event annotations.",
            "Freeze a content-addressed dataset card.",
            "Train or adapt the selected policy starting point.",
            "Evaluate across anatomy, sensor, calibration, and supervision challenges.",
            "Review failures and interventions with clinicians before any next-stage research.",
        ],
    }
