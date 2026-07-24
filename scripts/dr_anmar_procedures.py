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
        "id": "nvidia-native-surgical-bench",
        "title": "NVIDIA native suturing and instrument bench",
        "category": "Needle and instrument skills",
        "difficulty": "Foundation",
        "task": "Isaac-Handover-Needle-Dual-PSM-IK-Rel-v0",
        "anatomy_scene": "",
        "anatomy_focus": "NVIDIA surgical dry-lab bench",
        "robot": "Dual NVIDIA dVRK PSM",
        "instrument": "Two needle drivers, suture needle and surgical scissors",
        "objective": "Pick up and orient the curved needle, rehearse pad entry geometry, regrasp and hand it off, then practise physical scissors pickup and exchange.",
        "interaction": "NVIDIA Isaac for Healthcare v0.7 OpenUSD assets with native Isaac Lab articulation, rigid-body, collision and contact stepping.",
        "required_nvidia_assets": (
            "Robots/dVRK/PSM/psm.usd",
            "Props/SutureNeedle/needle_sdf.usd",
            "Props/SuturePad/suture_pad.usd",
            "Props/Table/table.usd",
            "Props/SurgicalInstruments/SurgicalScissors.usd",
        ),
        "nvidia_native_bench": True,
        "hide_anatomy": True,
        "show_waypoint_markers": False,
        "guide_kind": "native_suturing_bench",
        "lesson_steps": {
            "pickup": [
                step("approach", "Approach", "Bring open jaws to the middle third of the needle body.", "controlled approach"),
                step("grasp", "Grasp", "Close only after both jaws visibly straddle the needle.", "bilateral jaw contact"),
                step("lift", "Lift", "Raise the needle clear of the bench without changing its grasp point.", "needle elevation"),
                step("stabilize", "Stabilize", "Hold a readable needle-driving orientation without slip.", "stable custody"),
            ],
            "handover": [
                step("pickup", "Pick up", "Instrument 1 secures the middle third of the needle.", "sender custody"),
                step("present", "Present", "Expose a safe section of needle body in the shared workspace.", "usable presentation"),
                step("receive", "Receive", "Instrument 2 closes with bilateral jaw contact.", "dual custody"),
                step("release", "Release", "Instrument 1 opens only after receiver custody is stable.", "custody transfer"),
                step("separate", "Separate", "Move both instruments apart while Instrument 2 retains the needle.", "stable recovery"),
            ],
            "passing": [
                step("orient", "Orient", "Set a useful needle-driving angle before the pass.", "needle orientation"),
                step("pass", "Pass", "Move through the shared workspace with controlled curvature and clearance.", "controlled trajectory"),
                step("regrasp", "Regrasp", "Instrument 2 acquires an exposed section without colliding with Instrument 1.", "dual custody"),
                step("release", "Release", "Transfer custody only after the receiving grasp is stable.", "custody transfer"),
                step("present", "Present", "Finish ready for the next tissue bite.", "usable final orientation"),
            ],
            "recovery": [
                step("recognize", "Recognize", "Identify that the needle is displaced and stop uncertain motion.", "failure recognition"),
                step("reframe", "Restore view", "Adjust the camera until the needle and both tools are observable.", "target visibility"),
                step("reacquire", "Reacquire", "Approach slowly and close with bilateral jaw contact.", "physical regrasp"),
                step("clear", "Clear the field", "Lift the recovered needle away from the pad and other instrument.", "safe clearance"),
                step("stabilize", "Stabilize", "Return to a known, readable custody pose.", "stable recovery"),
            ],
        },
        "steps": [
            step("pickup", "Pick up", "Secure the middle third of the curved needle with Instrument 1.", "native bilateral jaw contact"),
            step("align", "Align at pad", "Present the needle tangent to the intended pad entry and follow its curvature without forcing the collider.", "needle and pad contact"),
            step("regrasp", "Regrasp", "Use Instrument 2 to take a useful exposed section of the needle body.", "dual custody"),
            step("handoff", "Hand off", "Open Instrument 1 only after Instrument 2 has a stable physical grasp.", "receiver custody"),
            step("exchange", "Exchange scissors", "Pick up, orient and transfer the scissors as a separate rigid instrument.", "scissors contact and pose"),
            step("recover", "Recover", "Return the needle and scissors to their separate table landing areas and separate both tools.", "stable placement"),
        ],
        "success_metrics": [
            "bilateral needle contact",
            "dual-custody overlap",
            "needle retention",
            "scissors rigid-body custody",
            "protected pad contact",
            "stable instrument placement",
        ],
    },
    {
        "id": "liver-retraction",
        "title": "Liver retraction practice",
        "category": "Tissue handling",
        "difficulty": "Intermediate",
        "task": "Isaac-Reach-PSM-IK-Rel-v0",
        "anatomy_scene": CT_MULTI_ORGAN,
        "anatomy_focus": "Liver",
        "robot": "dVRK PSM",
        "instrument": "Atraumatic grasper control",
        "objective": "Grasp the liver-shaped training target, lift it into a retracted position, and maintain a stable exposure.",
        "interaction": "Native PhysX FEM or coupled Newton VBD liver retraction with two-way instrument contact.",
        "guide_kind": "retraction",
        "steps": [
            step("inspect", "Inspect", "Choose a broad, accessible grasp region on the target.", "camera review"),
            step("approach", "Approach", "Bring the open jaws to the retraction target.", "tool motion"),
            step("grasp", "Grasp", "Close and confirm the target follows the tool.", "object coupling"),
            step("retract", "Retract", "Lift and move the target to expose the working area.", "object displacement"),
            step("hold", "Hold", "Maintain stable exposure before release.", "stable hold"),
        ],
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
        "guide_kind": "navigation",
        "steps": [
            step("survey", "Survey", "Use the endoscope to identify anatomy and all visible targets.", "camera review"),
            step("approach", "Approach", "Move to the first cyan target.", "waypoint 1"),
            step("navigate", "Navigate", "Visit each amber target in order.", "ordered waypoints"),
            step("finish", "Finish", "Reach the green endpoint.", "final waypoint"),
            step("recover", "Recover", "Return to a safe observation pose.", "tool clearance"),
        ],
    },
)


ADVANCED_PROCEDURE_ROOMS: tuple[dict[str, Any], ...] = (
    {
        "id": "dr-anmar-suturable-tissue",
        "title": "Dr.Anmar authored-assets operating room",
        "category": "Suturing and tissue mechanics",
        "difficulty": "Research foundation",
        "task": "Isaac-Reach-Dual-PSM-IK-Rel-v0",
        "anatomy_scene": "",
        "operating_room_environment": CT_MULTI_ORGAN,
        "anatomy_focus": "Dr.Anmar open-incision suturable tissue",
        "robot": "Dual dVRK PSM",
        "instrument": "Dr.Anmar Needle, Dr.Anmar 4-0 suture and two needle drivers",
        "objective": "Use the authored Dr.Anmar needle, 4-0 suture and open-incision tissue field to practise bimanual needle presentation, regrasping, handoff and planned wound-edge bites.",
        "interaction": "The Dr.Anmar OpenUSD room composes authored assets with native Isaac Lab PSM control, PhysX needle-suture dynamics and a stable OpenUSD tissue collision surface.",
        "guide_kind": "dr_anmar_wound_closure",
        "bimanual": True,
        "dr_anmar_needle_asset": True,
        "hide_anatomy": True,
        "show_waypoint_markers": False,
        "interactive_camera_width_px": 640,
        "interactive_camera_height_px": 426,
        "interactive_camera_eye_m": (0.30, 0.20, 0.23),
        "interactive_camera_target_m": (-0.050, -0.065, 0.035),
        "interactive_rgb_only": True,
        "interactive_multiview": True,
        "single_active_camera_renderer": True,
        "suture_native_segment_rendering": True,
        "suture_physics_lod": "interactive_90",
        "suture_telemetry_period_s": 1.0,
        "waypoints": (),
        "steps": [
            step("inspect", "Inspect wound", "Survey both real wound edges, tissue thickness and the open incision.", "wound geometry"),
            step("present", "Present the needle", "Use both instruments to orient and regrasp the authored needle above the wound field.", "stable needle custody"),
            step("plan", "Plan the bites", "Align the curved needle with symmetric entry and exit margins on the collision surface.", "needle trajectory and bite geometry"),
            step("handoff", "Regrasp and hand off", "Transfer the needle only after the receiving jaws establish stable physical custody.", "receiver custody"),
            step("qualify", "Qualify the path", "Record contact, custody and trajectory evidence; do not claim puncture until the topology backend is qualified.", "fail-closed evidence"),
        ],
        "success_metrics": [
            "bilateral needle contact",
            "stable regrasp",
            "handoff custody",
            "contact force",
            "needle path geometry",
        ],
    },
    {
        "id": "dr-anmar-hemostasis",
        "title": "DrAnmar vascular control laboratory",
        "category": "Vascular control and hemostasis",
        "difficulty": "Research foundation",
        "task": "Isaac-Reach-Dual-PSM-IK-Rel-v0",
        "anatomy_scene": "",
        "anatomy_focus": "DrAnmar hollow layered vessel",
        "robot": "Dual dVRK PSM",
        "instrument": "DrAnmar Vascular Clip and two needle drivers",
        "objective": "Manipulate the rigid clip, align it across the deformable vessel, study contact and placement, and collect force and deformation evidence.",
        "interaction": "Native PhysX owns intact vessel deformation, rigid clip motion, and two-way contact. Plastic clip forming, pressure-tight sealing, pulsatile flow, bleeding, rupture, and clinical hemostasis remain fail-closed.",
        "guide_kind": "dr_anmar_hemostasis",
        "bimanual": True,
        "hide_anatomy": True,
        "show_waypoint_markers": False,
        "waypoints": (),
        "steps": [
            step("inspect", "Inspect vessel", "Survey the lumen, wall, vessel axis, clip gap and clip orientation.", "geometry and pose"),
            step("pickup", "Pick up clip", "Secure the rigid clip without crossing or crushing the vessel.", "filtered jaw contact"),
            step("align", "Align clip", "Center the clip across the vessel with both arms clear of the wall.", "clip-to-vessel pose"),
            step("contact", "Establish contact", "Apply controlled contact and observe vessel deformation without claiming plastic closure.", "two-way contact"),
            step("qualify", "Record evidence", "Capture force, displacement, contact, damage-surrogate and replay evidence under the fail-closed boundary.", "research telemetry"),
        ],
        "success_metrics": [
            "clip pickup",
            "clip-to-vessel alignment",
            "bilateral tool contact",
            "vessel displacement",
            "contact force",
            "volume preservation",
            "stable recovery",
        ],
    },
    {
        "id": "nvidia-strand-ring-threading",
        "title": "Bimanual needle, strand and ring lab",
        "category": "Deformable-object skills",
        "difficulty": "Intermediate",
        "task": "Isaac-Thread-PSM-IK-Rel-v0",
        "anatomy_scene": "",
        "anatomy_focus": "SoftMimicGen dry-lab field",
        "robot": "Dual dVRK PSM",
        "instrument": "Two PSM forceps",
        "objective": "Practice two-handed needle transfer and ring passes with a native deformable strand attached to the needle.",
        "interaction": "Two independently controlled PSMs, rigid PhysX ring contact, native FEM strand self-contact and a physical needle-to-strand attachment.",
        "guide_kind": "softmimicgen_threading",
        "bimanual": True,
        "enable_strand_self_collision": True,
        "hide_anatomy": True,
        "show_waypoint_markers": False,
        "waypoints": (),
        "steps": [
            step("approach", "Approach", "Bring the left instrument to the needle body and the right instrument to the receiving side.", "bimanual alignment"),
            step("grasp", "Needle grasp", "Close the left jaws on the needle body without pinching the attached strand.", "needle contact"),
            step("pass", "Pass the ring", "Lead the curved needle and attached strand through the rigid ring.", "ring crossing"),
            step("handoff", "Handoff", "Establish the receiving grasp, release the first instrument, and pull the needle clear.", "physical transfer"),
            step("recover", "Set and recover", "Place the needle safely, inspect the strand path, and return both instruments to a stable pose.", "stable placement"),
        ],
        "success_metrics": ["two-instrument trajectory", "handoff contact sequence", "ring contact", "strand nodal state"],
    },
    {
        "id": "ultrasound-guided-access",
        "title": "Ultrasound-guided needle access",
        "category": "Image-guided intervention",
        "difficulty": "Advanced",
        "task": "Isaac-Reach-Dual-PSM-IK-Rel-v0",
        "anatomy_scene": MAISI_S0253,
        "anatomy_focus": "Synthetic abdominal target",
        "robot": "Dual dVRK PSM",
        "instrument": "Instrument 1 ultrasound probe · Instrument 2 access needle",
        "objective": "Acquire a simulated ultrasound target, plan an in-plane trajectory, and advance to the target while avoiding the protected vessel.",
        "interaction": "NVIDIA medical-ultrasound workflow output synchronized with physical probe and needle poses.",
        "external_provider": "nvidia_robotic_ultrasound",
        "provider_mode": "teleop_with_ultrasound",
        "guide_kind": "ultrasound_access",
        "waypoints": ((-0.050, -0.030, 0.075), (-0.026, -0.013, 0.058), (-0.004, 0.003, 0.047), (0.018, 0.017, 0.037)),
        "steps": [
            step("survey", "Survey", "Move the probe until the target is centered and confidence is stable.", "target confidence"),
            step("plan", "Plan trajectory", "Align the needle so its path remains visible in-plane.", "needle visibility"),
            step("advance", "Advance", "Move toward the target with short, controlled depth changes.", "target error"),
            step("confirm", "Confirm target", "Stop inside the target zone without entering the protected vessel.", "target contact"),
            step("recover", "Withdraw", "Withdraw along the same visible trajectory.", "safe recovery"),
        ],
        "success_metrics": ["target confidence", "needle visibility", "target error", "protected clearance", "trajectory length"],
    },
)


SONOGYM_ORTHOPEDIC_ROOMS: tuple[dict[str, Any], ...] = (
    {
        "id": "orthopedic-l4-ultrasound-navigation",
        "title": "L4 ultrasound navigation",
        "category": "Orthopedic ultrasound",
        "difficulty": "Foundation",
        "task": "Isaac-robot-US-guidance-v0",
        "anatomy_scene": "sonogym-lumbar-l4",
        "anatomy_focus": "CT-derived lumbar anatomy and L4 vertebra",
        "robot": "KUKA LBR with ultrasound probe",
        "instrument": "Robotic ultrasound probe",
        "objective": "Find the transverse ultrasound plane through the centre of L4 and hold a stable diagnostic view.",
        "interaction": "SonoGym native CT-derived ultrasound navigation, robot control, observations and reward.",
        "external_provider": "sonogym_orthopedics",
        "provider_mode": "l4_ultrasound_navigation",
        "guide_kind": "orthopedic_ultrasound_navigation",
        "waypoints": (),
        "steps": [
            step("survey", "Survey", "Sweep the lumbar surface and identify the L4 acoustic appearance.", "target visibility"),
            step("orient", "Orient", "Rotate into a transverse view without losing contact.", "probe orientation"),
            step("centre", "Centre L4", "Translate until the centre of L4 is aligned with the target plane.", "plane error"),
            step("hold", "Hold view", "Maintain the plane steadily before completing the attempt.", "view stability"),
        ],
        "success_metrics": ["plane error", "probe path length", "view stability", "episode reward"],
    },
    {
        "id": "orthopedic-l4-surface-reconstruction",
        "title": "L4 ultrasound surface reconstruction",
        "category": "Orthopedic ultrasound",
        "difficulty": "Intermediate",
        "task": "Isaac-robot-US-reconstruction-v0",
        "anatomy_scene": "sonogym-lumbar-l4",
        "anatomy_focus": "CT-derived lumbar anatomy and L4 surface",
        "robot": "KUKA LBR with ultrasound probe",
        "instrument": "Robotic ultrasound probe",
        "objective": "Acquire complementary ultrasound sweeps that reconstruct the L4 bone surface with efficient coverage.",
        "interaction": "SonoGym native reconstruction state, coverage observation, submodular reward and Isaac Lab stepping.",
        "external_provider": "sonogym_orthopedics",
        "provider_mode": "l4_surface_reconstruction",
        "guide_kind": "orthopedic_ultrasound_reconstruction",
        "waypoints": (),
        "steps": [
            step("localize", "Localize L4", "Acquire an initial view of the L4 target surface.", "initial localization"),
            step("sweep", "Build coverage", "Sweep through complementary views instead of repeating the same plane.", "surface coverage"),
            step("inspect", "Inspect gaps", "Use the reconstruction observation to find uncovered regions.", "uncovered surface"),
            step("complete", "Complete model", "Finish with stable, efficient coverage of the target surface.", "coverage efficiency"),
        ],
        "success_metrics": ["surface coverage", "uncovered points", "trajectory length", "episode reward"],
    },
    {
        "id": "orthopedic-l4-ultrasound-guided-surgery",
        "title": "L4 ultrasound-guided orthopedic trajectory",
        "category": "Orthopedic ultrasound",
        "difficulty": "Advanced research",
        "task": "Isaac-robot-US-guided-surgery-v0",
        "anatomy_scene": "sonogym-lumbar-l4",
        "anatomy_focus": "CT-derived lumbar anatomy and protected L4 target",
        "robot": "FR3 ultrasound robot + KUKA orthopedic instrument robot",
        "instrument": "Ultrasound probe and orthopedic drill trajectory",
        "objective": "Localize L4 with ultrasound, align the orthopedic trajectory, and advance while respecting SonoGym's safety cost.",
        "interaction": "SonoGym native dual-robot ultrasound-guided surgery environment with safe-action constraints.",
        "external_provider": "sonogym_orthopedics",
        "provider_mode": "l4_ultrasound_guided_surgery",
        "guide_kind": "orthopedic_ultrasound_guided_surgery",
        "waypoints": (),
        "steps": [
            step("localize", "Localize", "Acquire and stabilize the L4 ultrasound target.", "target visibility"),
            step("plan", "Plan", "Align the approach trajectory with the target and safety corridor.", "trajectory alignment"),
            step("advance", "Advance", "Use short, controlled actions while monitoring ultrasound and safety state.", "target error"),
            step("verify", "Verify", "Stop at the planned endpoint and inspect the final target and cost state.", "safe endpoint"),
        ],
        "success_metrics": ["target error", "trajectory length", "unsafe-action cost", "episode reward"],
    },
)


PROCEDURE_ROOMS = PROCEDURE_ROOMS + ADVANCED_PROCEDURE_ROOMS + SONOGYM_ORTHOPEDIC_ROOMS


PROCEDURE_SUITES: tuple[dict[str, Any], ...] = (
    {
        "id": "suturing-suite",
        "title": "Needle and thread skills",
        "description": "Progress from needle handling through transfer, regrasping and bimanual strand work.",
        "rooms": ["nvidia-native-surgical-bench", "dr-anmar-suturable-tissue", "nvidia-strand-ring-threading"],
    },
    {
        "id": "image-guided-suite",
        "title": "Image-guided intervention",
        "description": "Develop anatomy navigation, ultrasound targeting, access and complication recovery skills.",
        "rooms": ["synthetic-anatomy-navigation", "ultrasound-guided-access", "liver-retraction"],
    },
    {
        "id": "vascular-control-suite",
        "title": "Vascular control research",
        "description": "Develop clip handling, vessel contact, placement and fail-closed hemostasis evidence collection.",
        "rooms": ["dr-anmar-hemostasis"],
    },
    {
        "id": "orthopedic-ultrasound-suite",
        "title": "Orthopedic robotic ultrasound",
        "description": "Progress from L4 plane localization through surface reconstruction to ultrasound-guided orthopedic trajectory research.",
        "rooms": [
            "orthopedic-l4-ultrasound-navigation",
            "orthopedic-l4-surface-reconstruction",
            "orthopedic-l4-ultrasound-guided-surgery",
        ],
    },
)


PROCEDURES_BY_ID = {room["id"]: room for room in PROCEDURE_ROOMS}


def procedure_payload() -> dict[str, Any]:
    return {
        "schema": "dr.anmar.procedure-rooms.v2",
        "default": PROCEDURE_ROOMS[0]["id"],
        "rooms": [dict(room) for room in PROCEDURE_ROOMS],
        "suites": [dict(suite) for suite in PROCEDURE_SUITES],
    }
