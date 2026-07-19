# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Clinician-facing composed operating-room presets for Dr.Anmar."""

from __future__ import annotations

from typing import Any


CT_MULTI_ORGAN = "OR_scene_CTLiver-Prostate-Bladder"
MAISI_LIVER_27 = "OR_scene_MAISI_imagesTr_liver_27_relabel_resample1_syn_seed6_postprocess"
MAISI_S0253 = "OR_scene_MAISI_s0253_ct_relabel_resample1_syn_seed6_postprocess"
MAISI_S0702 = "OR_scene_MAISI_s0702_ct_relabel_resample2_syn_seed6_postprocess"
MAISI_S0994 = "OR_scene_MAISI_s0994_ct_relabel_resample2_syn_seed6_postprocess"
MAISI_S1269 = "OR_scene_MAISI_s1269_ct_relabel_resample1_syn_seed6_postprocess"
SURGICAL_S1371 = "OR_scene_s1371"


def step(step_id: str, title: str, instruction: str, signal: str) -> dict[str, str]:
    return {"id": step_id, "title": title, "instruction": instruction, "signal": signal}


PROCEDURE_ROOMS: tuple[dict[str, Any], ...] = (
    {
        "id": "needle-pickup",
        "title": "Needle pickup",
        "category": "Needle skills",
        "difficulty": "Foundation",
        "task": "Isaac-Lift-Needle-PSM-IK-Rel-v0",
        "anatomy_scene": CT_MULTI_ORGAN,
        "anatomy_focus": "CT liver field",
        "robot": "dVRK PSM",
        "instrument": "Large needle driver",
        "objective": "Approach the curved needle, grasp its body, lift it clear of the tray, and hold a stable recovery pose.",
        "interaction": "ORBIT-Surgical rigid-body needle physics with an 18 mm jaw-capture zone, adaptive fine control, protected instrument surfaces, and a bounded needle-tip entry channel.",
        "fidelity": "native_object_physics",
        "guide_kind": "pickup",
        "proxy_organ": None,
        "steps": [
            step("approach", "Approach", "Bring the open jaws toward the middle third of the curved needle.", "tool motion"),
            step("grasp", "Grasp", "Close the gripper around the needle body; Space toggles the jaws.", "gripper close"),
            step("lift", "Lift", "Raise the grasped needle above its starting plane.", "needle elevation"),
            step("orient", "Orient", "Rotate the needle into a controlled, readable presentation.", "tool orientation"),
            step("recover", "Recover", "Hold the final pose without dropping the needle.", "stable hold"),
        ],
        "truth_note": "The needle and robot use simulator physics. Closing within 18 mm secures the needle until release. The instrument shaft stays outside an OpenUSD-derived surface while a grasped needle tip may enter a bounded 12 mm rigid rehearsal channel. The entry marker and depth are training aids, not deformable biomechanics or clinical validation.",
    },
    {
        "id": "needle-transfer",
        "title": "Needle handover",
        "category": "Needle skills",
        "difficulty": "Intermediate",
        "task": "Isaac-Handover-Needle-Dual-PSM-IK-Rel-v0",
        "anatomy_scene": CT_MULTI_ORGAN,
        "anatomy_focus": "CT liver field",
        "robot": "Dual dVRK PSM",
        "instrument": "Two needle drivers",
        "objective": "Pick up the curved needle, present it to the second instrument, transfer the grasp, and separate safely.",
        "interaction": "Native ORBIT-Surgical dual-arm needle rigid-body physics.",
        "fidelity": "native_object_physics",
        "guide_kind": "handover",
        "proxy_organ": None,
        "steps": [
            step("pickup", "Pick up", "Instrument 1 secures the middle third of the needle.", "first grasp"),
            step("present", "Present", "Move the needle into a stable shared workspace.", "object movement"),
            step("receive", "Receive", "Instrument 2 closes on the exposed needle body.", "second grasp"),
            step("release", "Release", "Instrument 1 opens only after the receiving grasp is stable.", "gripper exchange"),
            step("separate", "Separate", "Move both tools apart while retaining the needle.", "stable recovery"),
        ],
        "truth_note": "The transfer uses the upstream dual-PSM needle environment; success scoring remains a research measure.",
    },
    {
        "id": "suture-threading-path",
        "title": "Suture threading pathway",
        "category": "Needle skills",
        "difficulty": "Intermediate",
        "task": "Isaac-Handover-Needle-Dual-PSM-IK-Rel-v0",
        "anatomy_scene": CT_MULTI_ORGAN,
        "anatomy_focus": "Liver surface",
        "robot": "Dual dVRK PSM",
        "instrument": "Two needle drivers",
        "objective": "Orient and guide the needle through a visible sequence of entry, arc, and exit targets before recovery.",
        "interaction": "Physical needle manipulation with OpenUSD pathway markers, ordered waypoint scoring, and a bounded needle-tip entry channel.",
        "fidelity": "path_guided_rehearsal",
        "guide_kind": "threading",
        "proxy_organ": None,
        "steps": [
            step("pickup", "Pick up", "Secure the needle away from its tip.", "gripper close"),
            step("entry", "Align entry", "Place the needle tip inside the cyan entry target.", "waypoint 1"),
            step("arc", "Follow the arc", "Rotate through the amber middle target with small deliberate motion.", "waypoint 2"),
            step("exit", "Reach exit", "Bring the needle tip to the green exit target.", "waypoint 3"),
            step("recover", "Recover", "Move the needle clear and hold a stable final pose.", "stable hold"),
        ],
        "truth_note": "This room trains needle-path control with a bounded rigid puncture proxy. It does not yet model deformable tissue, thread tension, tearing, or clinically validated needle forces.",
    },
    {
        "id": "liver-cutting-path",
        "title": "Liver cutting-path rehearsal",
        "category": "Dissection planning",
        "difficulty": "Intermediate",
        "task": "Isaac-Reach-PSM-IK-Rel-v0",
        "anatomy_scene": CT_MULTI_ORGAN,
        "anatomy_focus": "Liver",
        "robot": "dVRK PSM",
        "instrument": "PSM training tool",
        "objective": "Trace a planned incision corridor across the liver surface while keeping the tool inside the ordered safety path.",
        "interaction": "OpenUSD liver context with ordered 3D path targets and trajectory capture.",
        "fidelity": "path_guided_rehearsal",
        "guide_kind": "cutting_path",
        "proxy_organ": None,
        "steps": [
            step("plan", "Inspect", "Identify the cyan start and the complete amber corridor before moving.", "camera review"),
            step("approach", "Approach", "Move above the start target and settle the tool orientation.", "waypoint 1"),
            step("trace", "Trace", "Follow every corridor target in order with controlled speed.", "ordered waypoints"),
            step("finish", "Finish", "Reach the green endpoint without skipping a target.", "final waypoint"),
            step("recover", "Recover", "Lift away from the planned incision corridor.", "tool clearance"),
        ],
        "truth_note": "This is incision-path and motion rehearsal. It does not claim topology-changing tissue cutting or validated force thresholds.",
    },
    {
        "id": "liver-retraction",
        "title": "Liver retraction practice",
        "category": "Tissue handling",
        "difficulty": "Intermediate",
        "task": "Isaac-Lift-Block-PSM-IK-Rel-v0",
        "anatomy_scene": CT_MULTI_ORGAN,
        "anatomy_focus": "Liver",
        "robot": "dVRK PSM",
        "instrument": "Atraumatic grasper control",
        "objective": "Grasp the liver-shaped training target, lift it into a retracted position, and maintain a stable exposure.",
        "interaction": "Native rigid-body grasp physics with a real OpenUSD liver mesh used as the visible target.",
        "fidelity": "rigid_tissue_proxy",
        "guide_kind": "retraction",
        "proxy_organ": "Liver_topo_blender",
        "steps": [
            step("inspect", "Inspect", "Choose a broad, accessible grasp region on the target.", "camera review"),
            step("approach", "Approach", "Bring the open jaws to the retraction target.", "tool motion"),
            step("grasp", "Grasp", "Close and confirm the target follows the tool.", "object coupling"),
            step("retract", "Retract", "Lift and move the target to expose the working area.", "object displacement"),
            step("hold", "Hold", "Maintain stable exposure before release.", "stable hold"),
        ],
        "truth_note": "The visible liver geometry is official OpenUSD anatomy; its current interaction is a rigid tissue-handling proxy, not deformable biomechanics.",
    },
    {
        "id": "gallbladder-reposition",
        "title": "Gallbladder repositioning",
        "category": "Tissue handling",
        "difficulty": "Intermediate",
        "task": "Isaac-Lift-Block-PSM-IK-Rel-v0",
        "anatomy_scene": CT_MULTI_ORGAN,
        "anatomy_focus": "Gallbladder",
        "robot": "dVRK PSM",
        "instrument": "Atraumatic grasper control",
        "objective": "Pick up the gallbladder-shaped target, reposition it to the indicated workspace, and release deliberately.",
        "interaction": "Native rigid-body grasp physics with the official OpenUSD gallbladder mesh as the visible target.",
        "fidelity": "rigid_tissue_proxy",
        "guide_kind": "reposition",
        "proxy_organ": "Gallbladder_topo_blender",
        "steps": [
            step("approach", "Approach", "Move to the broad body of the target with the gripper open.", "tool motion"),
            step("grasp", "Grasp", "Close gently and confirm target motion.", "object coupling"),
            step("lift", "Lift", "Clear the starting surface before translating.", "object elevation"),
            step("place", "Place", "Move into the green placement region and open the jaws.", "target displacement"),
            step("recover", "Recover", "Withdraw without disturbing the placed target.", "tool clearance"),
        ],
        "truth_note": "This uses official organ geometry over a rigid manipulation proxy; deformable tissue response remains a later validation item.",
    },
    {
        "id": "bladder-handover",
        "title": "Bladder-part bimanual relocation",
        "category": "Tissue handling",
        "difficulty": "Advanced",
        "task": "Isaac-Handover-Block-Dual-PSM-IK-Rel-v0",
        "anatomy_scene": CT_MULTI_ORGAN,
        "anatomy_focus": "Bladder",
        "robot": "Dual dVRK PSM",
        "instrument": "Two atraumatic grasper controls",
        "objective": "Move a bladder-shaped training target between instruments and place it in a new operative position.",
        "interaction": "Dual-arm rigid-body handover with official OpenUSD bladder geometry as the visible target.",
        "fidelity": "rigid_tissue_proxy",
        "guide_kind": "handover",
        "proxy_organ": "Bladder_topo_blender",
        "steps": [
            step("pickup", "Pick up", "Instrument 1 lifts the target clear of the table.", "first grasp"),
            step("present", "Present", "Stabilize the target in the shared workspace.", "object motion"),
            step("receive", "Receive", "Instrument 2 grasps before Instrument 1 releases.", "second grasp"),
            step("place", "Relocate", "Move the retained target to its indicated new position.", "object displacement"),
            step("recover", "Recover", "Release and separate both instruments.", "stable release"),
        ],
        "truth_note": "Organ appearance is OpenUSD anatomy; contact dynamics currently use the upstream rigid handover task.",
    },
    {
        "id": "synthetic-anatomy-navigation",
        "title": "Anatomy-variation navigation",
        "category": "Visual robustness",
        "difficulty": "Research challenge",
        "task": "Isaac-Reach-PSM-IK-Rel-v0",
        "anatomy_scene": SURGICAL_S1371,
        "anatomy_focus": "Multi-organ surgical anatomy",
        "robot": "dVRK PSM",
        "instrument": "PSM training tool",
        "objective": "Navigate an ordered target path in a distinct patient-anatomy geometry and record a comparable trajectory.",
        "interaction": "Official OpenUSD anatomy variation with native relative-IK tool control and 3D waypoint scoring.",
        "fidelity": "anatomy_context",
        "guide_kind": "navigation",
        "proxy_organ": None,
        "steps": [
            step("survey", "Survey", "Use the endoscope to identify anatomy and all visible targets.", "camera review"),
            step("approach", "Approach", "Move to the first cyan target.", "waypoint 1"),
            step("navigate", "Navigate", "Visit each amber target in order.", "ordered waypoints"),
            step("finish", "Finish", "Reach the green endpoint.", "final waypoint"),
            step("recover", "Recover", "Return to a safe observation pose.", "tool clearance"),
        ],
        "truth_note": "This scene evaluates navigation across anatomy variation; it is not a clinical procedure model.",
    },
)


PROCEDURES_BY_ID = {room["id"]: room for room in PROCEDURE_ROOMS}


def procedure_payload() -> dict[str, Any]:
    return {
        "schema": "dr.anmar.procedure-rooms.v1",
        "default": PROCEDURE_ROOMS[0]["id"],
        "rooms": [dict(room) for room in PROCEDURE_ROOMS],
        "fidelity_legend": {
            "native_object_physics": "Native ORBIT-Surgical grasp/manipulation physics.",
            "path_guided_rehearsal": "Interactive tool control with ordered 3D guidance; no tissue topology change.",
            "rigid_tissue_proxy": "Official organ appearance attached to an upstream rigid manipulation target.",
            "anatomy_context": "Official anatomy variation used as a visual and spatial context.",
        },
    }
