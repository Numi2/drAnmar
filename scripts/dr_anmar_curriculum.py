# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Doctor-first curriculum and research references for Dr.Anmar Studio."""

from __future__ import annotations


COURSES = [
    {
        "id": "robot-foundations",
        "number": "01",
        "title": "Robot foundations",
        "short_title": "Foundations",
        "description": "Learn how a surgical robot sees, moves, and holds a tool before teaching it a procedure.",
        "lessons": [
            {
                "id": "psm-precision-reach",
                "title": "Move a patient-side instrument",
                "eyebrow": "Runnable lab · 12 min",
                "task": "Isaac-Reach-PSM-IK-Rel-v0",
                "video": None,
                "mode": "live",
                "summary": "Guide one PSM tool to a target pose. Think of this as hand-eye coordination for a robot.",
                "goal": "Reach the target smoothly while keeping movements small and controlled.",
                "concepts": ["6D pose", "inverse kinematics", "action"],
            },
            {
                "id": "ecm-camera-control",
                "title": "Position the endoscopic camera",
                "eyebrow": "Runnable lab · 10 min",
                "task": "Isaac-Reach-ECM-IK-Rel-v0",
                "video": None,
                "mode": "live",
                "summary": "Move the ECM camera to improve the operative view without moving the surgical tool.",
                "goal": "Center the target and keep it visible, like framing a subject with a camera.",
                "concepts": ["observation", "camera pose", "workspace"],
            },
            {
                "id": "dual-arm-coordination",
                "title": "Coordinate two instruments",
                "eyebrow": "Runnable lab · 15 min",
                "task": "Isaac-Reach-Dual-PSM-IK-Rel-v0",
                "video": None,
                "mode": "live",
                "summary": "Control two PSM instruments and switch between them while maintaining a shared plan.",
                "goal": "Place both tools without crossing or losing sight of either instrument.",
                "concepts": ["bimanual control", "coordination", "state"],
            },
        ],
    },
    {
        "id": "learning-from-demonstration",
        "number": "02",
        "title": "Learning from demonstrations",
        "short_title": "Demonstrations",
        "description": "Show the robot a good example, save every action, and understand how Behavior Cloning learns from it.",
        "lessons": [
            {
                "id": "needle-lift",
                "title": "Needle lift",
                "eyebrow": "Runnable lab + SuFIA-BC reference · 20 min",
                "task": "Isaac-Lift-Needle-PSM-IK-Rel-v0",
                "video": "/research/videos/lift.mp4",
                "mode": "live",
                "summary": "Watch an expert lift, perform your own attempt, and save it as training data.",
                "goal": "Approach the curved needle, close the gripper, and lift without losing the grasp.",
                "concepts": ["demonstration", "Behavior Cloning", "trajectory"],
            },
            {
                "id": "needle-handover",
                "title": "Needle handover",
                "eyebrow": "Runnable bimanual lab + SuFIA-BC reference · 25 min",
                "task": "Isaac-Handover-Needle-Dual-PSM-IK-Rel-v0",
                "video": "/research/videos/handover.mp4",
                "mode": "live",
                "summary": "Pass a curved needle between instruments while preserving a usable grasp.",
                "goal": "The receiving gripper secures the needle before the first instrument releases it.",
                "concepts": ["handover", "timing", "multi-arm policy"],
            },
            {
                "id": "block-transfer",
                "title": "Block transfer",
                "eyebrow": "Runnable approximation + SuFIA-BC reference · 20 min",
                "task": "Isaac-Handover-Block-Dual-PSM-IK-Rel-v0",
                "video": "/research/videos/transfer.mp4",
                "mode": "live",
                "summary": "Use the rigid block handover environment to practise the coordination pattern shown in the research task.",
                "goal": "Transfer the object cleanly and compare your motion with the research reference.",
                "concepts": ["generalization", "object state", "demonstration quality"],
            },
        ],
    },
    {
        "id": "reinforcement-learning",
        "number": "03",
        "title": "Reinforcement learning",
        "short_title": "RL policies",
        "description": "Let a policy practise in simulation and improve from a score, like coaching by outcomes instead of copying motions.",
        "lessons": [
            {
                "id": "rl-reach",
                "title": "Train a reaching policy",
                "eyebrow": "Runnable training lab · 30+ min",
                "task": "Isaac-Reach-PSM-IK-Rel-v0",
                "training_task": "Isaac-Reach-PSM-v0",
                "video": None,
                "mode": "train",
                "summary": "Create a small RL run where the policy earns a better score by reaching the target accurately.",
                "goal": "Understand observation, action, reward, episode, and checkpoint from one bounded run.",
                "concepts": ["reward", "episode", "checkpoint"],
            },
            {
                "id": "rl-needle-lift",
                "title": "Train needle lifting",
                "eyebrow": "Runnable training lab · advanced",
                "task": "Isaac-Lift-Needle-PSM-IK-Rel-v0",
                "training_task": "Isaac-Lift-Needle-PSM-v0",
                "video": "/research/videos/lift.mp4",
                "mode": "train",
                "summary": "Compare outcome-driven RL with copying expert needle-lift demonstrations.",
                "goal": "Identify which parts of a delicate grasp are easiest to specify as a score and which are easier to demonstrate.",
                "concepts": ["reward design", "sample efficiency", "policy"],
            },
        ],
    },
    {
        "id": "visual-policy-lab",
        "number": "04",
        "title": "Visual policy lab",
        "short_title": "Vision",
        "description": "Explore why camera placement and 3D representation change what a surgical policy can learn.",
        "lessons": [
            {
                "id": "camera-view-generalization",
                "title": "Same task, different camera",
                "eyebrow": "SuFIA-BC supplementary study · 15 min",
                "task": None,
                "video": "/research/videos/Lift_view_train.mp4",
                "video_set": [
                    {"label": "Training view", "src": "/research/videos/Lift_view_train.mp4"},
                    {"label": "New view 1", "src": "/research/videos/Lift_view_1.mp4"},
                    {"label": "New view 2", "src": "/research/videos/Lift_view_2.mp4"},
                ],
                "mode": "reference",
                "summary": "Compare the same needle-lift behavior from three viewpoints and see why a 2D image policy can be viewpoint-sensitive.",
                "goal": "Explain which visual clues remain stable when the camera moves.",
                "concepts": ["multi-view", "robustness", "visual observation"],
            },
            {
                "id": "point-cloud-policy",
                "title": "From pixels to a 3D point cloud",
                "eyebrow": "SuFIA-BC 3D policy lesson · 15 min",
                "task": None,
                "video": "/research/videos/Tissue_view_train.mp4",
                "mode": "reference",
                "summary": "A color image says what each pixel looks like; depth adds how far away it is. Together they form a 3D point cloud.",
                "goal": "Compare a multi-camera image policy with a 3D Diffusion Policy built from one endoscopic RGB-D view.",
                "concepts": ["RGB-D", "point cloud", "Diffusion Policy"],
            },
            {
                "id": "needle-generalization",
                "title": "Generalize to a new needle",
                "eyebrow": "SuFIA-BC supplementary study · 12 min",
                "task": None,
                "video": "/research/videos/needle_n1.mp4",
                "video_set": [
                    {"label": f"Needle {index}", "src": f"/research/videos/needle_n{index}.mp4"}
                    for index in range(1, 6)
                ],
                "mode": "reference",
                "summary": "Compare five needle instances to understand object generalization rather than memorizing one exact shape.",
                "goal": "Spot what the policy must treat as essential: graspable geometry, pose, and curvature.",
                "concepts": ["instance generalization", "domain variation", "robust policy"],
            },
        ],
    },
    {
        "id": "procedure-studies",
        "number": "05",
        "title": "Procedure studies",
        "short_title": "Procedures",
        "description": "Break research procedures into observable phases before building a safe supervised-autonomy task.",
        "lessons": [
            {
                "id": "shunt-insertion",
                "title": "Shunt insertion",
                "eyebrow": "ORBIT-Surgical paper lesson · reference only",
                "task": None,
                "video": None,
                "mode": "reference",
                "summary": "Study the benchmark procedure as phases: acquire the shunt, align it with the vessel, insert, and verify placement.",
                "goal": "Write down the observations, actions, success criteria, and hand-back conditions required for a supervised policy.",
                "concepts": ["task decomposition", "success criteria", "human hand-back"],
                "truth_note": "The public ORBIT-Surgical checkout does not include a runnable shunt-insertion environment. This is a paper and supplementary-material lesson, not a simulated live task.",
            },
            {
                "id": "tissue-retraction",
                "title": "Tissue retraction",
                "eyebrow": "SuFIA-BC research reference · organ scene",
                "task": None,
                "video": "/research/videos/tissue.mp4",
                "mode": "reference",
                "summary": "Observe how a visual policy grasps and retracts deformable tissue while preserving the operative view.",
                "goal": "Identify tissue state, safe grasp region, desired exposure, and release conditions.",
                "concepts": ["deformable tissue", "visual feedback", "safety boundary"],
            },
            {
                "id": "suture-pad",
                "title": "Suture pad interaction",
                "eyebrow": "SuFIA-BC research reference",
                "task": None,
                "video": "/research/videos/suture.mp4",
                "mode": "reference",
                "summary": "Study needle approach and contact around a suturing pad as an example of precise visual behavior.",
                "goal": "Separate the approach, contact, passage, and recovery phases.",
                "concepts": ["contact-rich task", "phase", "precision"],
            },
        ],
    },
]


POLICY_GUIDE = [
    {
        "id": "behavior-cloning",
        "name": "Behavior Cloning",
        "analogy": "Like a resident copying many expert demonstrations.",
        "learns_from": "Recorded observation → action pairs",
        "strength": "Natural motions and fast learning when demonstrations are consistent",
        "watch_for": "It can struggle when it reaches a situation not represented in the examples",
    },
    {
        "id": "reinforcement-learning",
        "name": "Reinforcement Learning",
        "analogy": "Like practising repeatedly with a precise score after every attempt.",
        "learns_from": "Rewards, penalties, and simulated trial-and-error",
        "strength": "Can discover strategies beyond the demonstrations",
        "watch_for": "A poorly designed score can teach technically successful but clinically undesirable behavior",
    },
    {
        "id": "visual-bc",
        "name": "Visual Behavior Cloning",
        "analogy": "The resident sees the camera view and learns which motion usually follows.",
        "learns_from": "Images or point clouds paired with expert actions",
        "strength": "Uses the same kind of visual input available during endoscopic procedures",
        "watch_for": "Camera, lighting, anatomy, and instrument variation must be represented",
    },
]


GLOSSARY = [
    ("Action", "The command sent to the simulated robot, such as a small tool movement or gripper change."),
    ("Behavior Cloning", "Learning to imitate expert actions from recorded examples."),
    ("Checkpoint", "A saved version of a policy during training; like saving progress at a particular practice session."),
    ("Demonstration", "A time-aligned recording of what the robot observed and what the expert commanded."),
    ("Digital twin", "A simulation of the robot, instruments, objects, anatomy, cameras, and physics used for safe experiments."),
    ("Episode", "One practice attempt, from reset until success, failure, or timeout."),
    ("Observation", "The information available to the policy: robot state, camera images, depth, object pose, or a combination."),
    ("Policy", "The learned decision-maker that converts an observation into the next action."),
    ("Point cloud", "A 3D collection of points produced from color plus depth; each point represents a visible surface location."),
    ("Reward", "The numeric coaching signal used by reinforcement learning."),
    ("Synthetic data", "Training examples generated in simulation, where geometry, state, and camera labels are known exactly."),
    ("Supervised autonomy", "The robot performs a bounded behavior while a clinician monitors, can interrupt, and retains authority."),
]


def curriculum_payload() -> dict:
    lessons = [lesson for course in COURSES for lesson in course["lessons"]]
    return {
        "courses": COURSES,
        "policies": POLICY_GUIDE,
        "glossary": [{"term": term, "definition": definition} for term, definition in GLOSSARY],
        "lesson_count": len(lessons),
        "runnable_count": sum(lesson["mode"] in {"live", "train"} for lesson in lessons),
        "simulation_only": True,
        "sources": {
            "orbit": "https://orbit-surgical.github.io/",
            "orbit_paper": "https://autolab.berkeley.edu/assets/publications/media/2024-ICRA-ORBIT-Surgical.pdf",
            "sufia_bc": "https://orbit-surgical.github.io/sufia-bc/",
            "sufia_bc_paper": "https://autolab.berkeley.edu/assets/publications/media/sufia_bc.pdf",
        },
    }
