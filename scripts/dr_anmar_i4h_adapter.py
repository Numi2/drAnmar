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
I4H_ROOT = Path(
    os.environ.get("DR_ANMAR_I4H_ROOT", APP_ROOT / "vendor/i4h-workflows-current")
).expanduser()
I4H_RELEASE = os.environ.get("DR_ANMAR_I4H_RELEASE", "v0.7.0")
I4H_RELEASE_COMMIT = os.environ.get(
    "DR_ANMAR_I4H_RELEASE_COMMIT",
    "9b526c6d107254727d3b113c612fb860fc65a5b2",
)
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
        "provides": [
            "trocar_assembly",
            "bimanual_manipulation",
            "surface_deformable_cloth",
            "newton_physx_backends",
            "xr_teleoperation",
            "groot",
            "online_rl",
        ],
        "doctor_summary": "Use NVIDIA's rigid and surface-deformable precision-manipulation references as expert research starting points.",
        "doctor_default_mode": None,
        "expert_source_only": True,
    },
    "agentic": {
        "title": "NVIDIA surgical Arena",
        "directory": "workflows/agentic",
        "provides": [
            "native_surgical_environments",
            "scripted_state_machines",
            "teleoperation",
            "mimic",
            "lerobot",
            "groot",
            "openpi",
        ],
        "doctor_summary": "Run NVIDIA's native surgical environments and use the same contracts for demonstrations, datasets, policies, and evaluation.",
        "doctor_default_mode": "surgical_reach_psm",
        "agentic_yaml_contract": True,
    },
    "catheter_navigation": {
        "title": "Endoluminal navigation",
        "directory": "workflows/catheter_navigation",
        "provides": ["fluoroscopy", "dsa", "xpbd_catheter", "vasculature_digital_twin", "ct_ingestion"],
        "doctor_summary": "Study patient-specific endovascular navigation with NVIDIA's fluoroscopy and catheter-physics workflow.",
        "doctor_default_mode": "interactive_viewport",
    },
}


def _yaml_section_scalar(text: str, section: str, key: str) -> str | None:
    """Read one scalar from NVIDIA's shallow environment YAML without copying it."""

    in_section = False
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        if indent == 0:
            in_section = stripped == f"{section}:"
            continue
        if in_section and indent == 2 and stripped.startswith(f"{key}:"):
            value = stripped.split(":", 1)[1].strip()
            if not value or value in {"null", "~"}:
                return None
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            return value
    return None


def agentic_runtime_prerequisites() -> dict[str, Any]:
    """Report only prerequisites needed by NVIDIA's v0.7 Agentic workflow."""

    workflow_root = I4H_ROOT / "workflows/agentic"
    uv_path = shutil.which("uv")
    if uv_path is None:
        user_uv = Path.home() / ".local/bin/uv"
        uv_path = str(user_uv) if user_uv.is_file() else None
    python_candidates = (
        workflow_root / "arena/.venv/bin/python",
        workflow_root / ".venv/bin/python",
    )
    workflow_python = next((path for path in python_candidates if path.is_file()), None)
    return {
        "uv": {"ready": uv_path is not None, "path": uv_path},
        "git": {"ready": bool(shutil.which("git")), "path": shutil.which("git")},
        "nvidia_gpu_device": {"ready": Path("/dev/nvidia0").exists(), "path": "/dev/nvidia0"},
        "agentic_python": {
            "ready": workflow_python is not None,
            "path": str(workflow_python) if workflow_python else None,
        },
    }


def agentic_workflow_modes() -> dict[str, Any]:
    """Discover NVIDIA v0.7 surgical environments from their source-of-truth YAMLs."""

    config_root = I4H_ROOT / "workflows/agentic/config/environments"
    prerequisites = agentic_runtime_prerequisites()
    missing = [
        label
        for key, label in (
            ("uv", "uv"),
            ("git", "git"),
            ("nvidia_gpu_device", "NVIDIA GPU"),
            ("agentic_python", "NVIDIA Agentic workflow setup"),
        )
        if not prerequisites[key]["ready"]
    ]
    modes = []
    digest = hashlib.sha256()
    for path in sorted(config_root.glob("surgical_*.yaml")) if config_root.is_dir() else []:
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        digest.update(path.name.encode("utf-8"))
        digest.update(source.encode("utf-8"))
        env_id = path.stem
        description = (
            _yaml_section_scalar(source, "arena", "description")
            or _yaml_section_scalar(source, "policy", "task_description")
            or env_id.replace("_", " ").title()
        )
        bridge_port = _yaml_section_scalar(source, "arena", "bridge_port")
        modes.append(
            {
                "id": env_id,
                "title": env_id.replace("_", " ").title(),
                "description": description,
                "category": "simulation",
                "launchable": True,
                "requires_hardware": False,
                "requires_arguments": False,
                "requires_rti": False,
                "blocked_reason": None,
                "recommended": env_id == "surgical_reach_psm",
                "launch_ready": not missing,
                "missing_prerequisites": list(missing),
                "runtime_validated": False,
                "provider_kind": "nvidia_agentic_state_machine",
                "upstream_environment": env_id,
                "robot": _yaml_section_scalar(source, "robot", "type"),
                "bridge_port": int(bridge_port) if bridge_port and bridge_port.isdigit() else None,
                "environment_contract": str(path),
                "environment_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            }
        )
    return {
        "default_mode": "surgical_reach_psm",
        "official_default_mode": None,
        "metadata_ready": bool(modes),
        "metadata_path": str(config_root),
        "metadata_sha256": digest.hexdigest() if modes else None,
        "rejected_modes": [],
        "discovery_error": None if modes else "No NVIDIA surgical environment YAMLs were found.",
        "modes": modes,
        "agentic_runtime_prerequisites": prerequisites,
        "upstream_contract": "workflows/agentic/config/environments/<env>.yaml",
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
    if definition.get("agentic_yaml_contract"):
        return agentic_workflow_modes()
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
                "This upstream workflow has no guarded launch contract; use its reviewed scripts outside the clinician launcher."
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


def workflow_inspection_command(workflow_id: str) -> str:
    definition = WORKFLOW_BINDINGS[workflow_id]
    if definition.get("agentic_yaml_contract"):
        return "workflows/agentic/arena/run.sh --list-envs"
    if definition.get("expert_source_only"):
        return f"Read {definition['directory']}/README.md"
    return f"./i4h modes {workflow_id}"


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
        "provider": "NVIDIA Isaac GR00T through Isaac for Healthcare v0.7 Agentic and SO-ARM workflows",
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
                "inspect_command": workflow_inspection_command(workflow_id),
                **launch,
            }
        )
    i4h_cli = I4H_ROOT / "i4h"
    return {
        "schema": "dr.anmar.isaac-healthcare-capabilities.v3",
        "strategy": "wrap_not_duplicate",
        "i4h_release": I4H_RELEASE,
        "i4h_release_commit": I4H_RELEASE_COMMIT,
        "i4h_root": str(I4H_ROOT),
        "i4h_cli_ready": i4h_cli.is_file(),
        "runtime_prerequisites": runtime_prerequisites(),
        "agentic_runtime_prerequisites": agentic_runtime_prerequisites(),
        "upstream_surgical_environments": agentic_workflow_modes()["modes"],
        "workflows": workflows,
        "modalities": [dict(item) for item in MODALITY_CATALOG],
        "policies": [dict(item) for item in POLICY_STARTING_POINTS],
        "maisi_anatomy_ready": anatomy_root.is_dir() and any(anatomy_root.iterdir()),
        "runtime_boundary": {
            "dr_anmar_owns": [
                "clinician pedagogy",
                "procedure composition",
                "recording and annotation",
                "evaluation and evidence review",
            ],
            "isaac_for_healthcare_owns": [
                "OpenUSD physical assets",
                "Isaac Lab actions, contacts, constraints, sensors and stepping",
                "native physics backends",
                "surgical environments and scripted state machines",
                "synthetic data",
                "policy, imitation and reinforcement learning infrastructure",
                "runtime and hardware-in-the-loop integration",
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
        "workflow_inspection": [workflow_inspection_command(workflow) for workflow in workflows],
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
