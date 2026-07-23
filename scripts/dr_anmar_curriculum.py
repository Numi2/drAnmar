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
                "procedure_id": "needle-pickup",
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
                "procedure_id": "needle-transfer",
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
                "training_task": "Isaac-Reach-PSM-IK-Rel-v0",
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
                "training_task": "Isaac-Lift-Needle-PSM-IK-Rel-v0",
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
                "id": "tissue-retraction",
                "title": "Tissue retraction",
                "eyebrow": "SuFIA-BC research reference · organ scene",
                "task": "Isaac-Lift-Block-PSM-IK-Rel-v0",
                "procedure_id": "liver-retraction",
                "video": "/research/videos/tissue.mp4",
                "mode": "live",
                "summary": "Observe how a visual policy grasps and retracts deformable tissue while preserving the operative view.",
                "goal": "Identify tissue state, safe grasp region, desired exposure, and release conditions.",
                "concepts": ["deformable tissue", "visual feedback", "safety boundary"],
            },
            {
                "id": "suture-pad",
                "title": "NVIDIA native surgical skills bench",
                "eyebrow": "Runnable NVIDIA v0.7 dry-lab · 25 min",
                "task": "Isaac-Handover-Needle-Dual-PSM-IK-Rel-v0",
                "procedure_id": "nvidia-native-surgical-bench",
                "video": None,
                "mode": "live",
                "summary": "Use two native dVRK PSMs around NVIDIA's pad, needle and scissors with a clear adjustable overview camera.",
                "goal": "Acquire and orient the needle, rehearse pad entry geometry, complete a physical regrasp and handoff, then exchange the scissors.",
                "concepts": ["native contact", "needle presentation", "dual custody", "instrument exchange"],
            },
            {
                "id": "dr-anmar-suturable-tissue-live",
                "title": "DrAnmar wound-closure mechanics",
                "eyebrow": "Runnable research asset lab · 30 min",
                "task": "Isaac-Reach-Dual-PSM-IK-Rel-v0",
                "procedure_id": "dr-anmar-suturable-tissue",
                "video": None,
                "mode": "live",
                "summary": "Use the independent DrAnmar Needle and 4-0 suture around a deformable open-incision tissue asset while preserving the topology-changing capability boundary.",
                "goal": "Establish controlled tissue contact, approximate the wound edges, and collect replayable mechanics evidence without claiming unsupported puncture.",
                "concepts": ["deformable contact", "suture material history", "sim-to-real", "fail-closed qualification"],
            },
            {
                "id": "dr-anmar-hemostasis-live",
                "title": "DrAnmar vascular control mechanics",
                "eyebrow": "Runnable research asset lab · 30 min",
                "task": "Isaac-Reach-Dual-PSM-IK-Rel-v0",
                "procedure_id": "dr-anmar-hemostasis",
                "video": None,
                "mode": "live",
                "summary": "Manipulate the independent DrAnmar Vascular Clip around a hollow deformable vessel and record contact, pose and deformation evidence.",
                "goal": "Pick up and align the clip, establish controlled vessel contact, and preserve the plasticity, flow, leakage and clinical qualification boundary.",
                "concepts": ["vascular contact", "collision fidelity", "surrogate admissibility", "evidence gates"],
            },
        ],
    },
]


COURSES.extend(
    [
        {
            "id": "needle-thread-skills",
            "number": "06",
            "title": "Needle and thread skills",
            "short_title": "Thread skills",
            "description": "Build bimanual needle control, strand handling, ring passes and deliberate regrasping.",
            "lessons": [
                {
                    "id": "needle-hoop-threading-live",
                    "title": "Needle, strand and ring",
                    "eyebrow": "Bimanual sequence · 25 min",
                    "task": "Isaac-Thread-PSM-IK-Rel-v0",
                    "procedure_id": "nvidia-strand-ring-threading",
                    "video": None,
                    "mode": "live",
                    "summary": "Control a threaded curved needle with two instruments, pass it through the ring, and hand it off cleanly.",
                    "goal": "Complete a physical ring pass and handoff while maintaining strand control.",
                    "concepts": ["hand-eye coordination", "thread control", "handoff", "ring pass"],
                },
                {
                    "id": "needle-passing-regrasp-live",
                    "title": "Needle passing and regrasping",
                    "eyebrow": "Runnable bimanual orientation lab · 25 min",
                    "task": "Isaac-Handover-Needle-Dual-PSM-IK-Rel-v0",
                    "procedure_id": "needle-passing-regrasp",
                    "video": None,
                    "mode": "live",
                    "summary": "Pass a needle between instruments and deliberately rebuild a useful driving angle before the next bite.",
                    "goal": "Maintain custody, control the curved body, and finish in a repeatable needle-driving orientation.",
                    "concepts": ["needle orientation", "regrasp", "bimanual custody"],
                },
            ],
        },
        {
            "id": "clinical-procedure-labs",
            "number": "07",
            "title": "Guided access and recovery",
            "short_title": "Procedure labs",
            "description": "Practise image-guided access and deliberate recovery in the available rooms.",
            "lessons": [
                {
                    "id": "ultrasound-access-live",
                    "title": "Ultrasound-guided access",
                    "eyebrow": "NVIDIA medical-ultrasound lab · 30 min",
                    "task": "Isaac-Reach-Dual-PSM-IK-Rel-v0",
                    "procedure_id": "ultrasound-guided-access",
                    "video": None,
                    "mode": "live",
                    "summary": "Acquire a target, maintain in-plane needle visibility, and reach it while preserving vessel clearance.",
                    "goal": "Reach the target zone with high visibility and a safe protected-structure margin.",
                    "concepts": ["B-mode", "in-plane access", "target confidence"],
                },
                {
                    "id": "recovery-live",
                    "title": "Complication recovery",
                    "eyebrow": "Runnable randomized challenge · 25 min",
                    "task": "Isaac-Handover-Needle-Dual-PSM-IK-Rel-v0",
                    "procedure_id": "complication-recovery",
                    "video": None,
                    "mode": "live",
                    "summary": "Recognize a failure, stop unsafe movement, restore visibility, reacquire the object and hand back safely.",
                    "goal": "Demonstrate conservative recovery rather than continuing through uncertainty.",
                    "concepts": ["failure recognition", "recovery", "human hand-back"],
                },
            ],
        },
        {
            "id": "anatomy-manipulation-labs",
            "number": "08",
            "title": "Anatomy navigation",
            "short_title": "Anatomy skills",
            "description": "Adapt camera and tool motion across the installed anatomy variations.",
            "lessons": [
                {
                    "id": "anatomy-variation-live",
                    "title": "Navigate patient-shape variation",
                    "eyebrow": "Runnable MAISI/OpenUSD comparison · 20 min",
                    "task": "Isaac-Reach-PSM-IK-Rel-v0",
                    "procedure_id": "synthetic-anatomy-navigation",
                    "video": None,
                    "mode": "live",
                    "summary": "Repeat a controlled reach across the installed anatomy presets and compare how geometry changes the approach corridor.",
                    "goal": "Adapt the camera and approach plan rather than memorizing one scene geometry.",
                    "concepts": ["anatomy variation", "domain generalization", "approach planning"],
                },
            ],
        },
        {
            "id": "orthopedic-robotic-ultrasound",
            "number": "09",
            "title": "Orthopedic robotic ultrasound",
            "short_title": "Orthopedics",
            "description": "Learn lumbar probe navigation, ultrasound-based L4 reconstruction, and image-guided orthopedic trajectory research in the native SonoGym environments.",
            "lessons": [
                {
                    "id": "sonogym-l4-navigation-live",
                    "title": "Find the L4 transverse plane",
                    "eyebrow": "Native SonoGym manual lab · 25 min",
                    "task": "Isaac-robot-US-guidance-v0",
                    "procedure_id": "orthopedic-l4-ultrasound-navigation",
                    "video": None,
                    "mode": "live",
                    "summary": "Control a KUKA-mounted ultrasound probe across CT-derived lumbar anatomy and localize the transverse plane through the centre of L4.",
                    "goal": "Reach and hold the L4 target plane with a short, stable probe path.",
                    "concepts": ["probe navigation", "lumbar ultrasound", "target-plane localization", "robotic imaging"],
                },
                {
                    "id": "sonogym-l4-reconstruction-live",
                    "title": "Reconstruct the L4 surface",
                    "eyebrow": "Native SonoGym reconstruction lab · 30 min",
                    "task": "Isaac-robot-US-reconstruction-v0",
                    "procedure_id": "orthopedic-l4-surface-reconstruction",
                    "video": None,
                    "mode": "live",
                    "summary": "Plan complementary ultrasound sweeps while watching which simulated L4 surface regions are covered or still missing.",
                    "goal": "Build efficient surface coverage without repeatedly sampling the same view.",
                    "concepts": ["3D reconstruction", "coverage", "submodular reward", "view planning"],
                },
                {
                    "id": "sonogym-l4-guided-surgery-live",
                    "title": "Ultrasound-guided L4 trajectory",
                    "eyebrow": "Native SonoGym safe-surgery lab · 35 min",
                    "task": "Isaac-robot-US-guided-surgery-v0",
                    "procedure_id": "orthopedic-l4-ultrasound-guided-surgery",
                    "video": None,
                    "mode": "live",
                    "summary": "Coordinate ultrasound localization and an orthopedic instrument trajectory while monitoring target error and the environment's safety cost.",
                    "goal": "Reach the planned L4 endpoint using a controlled trajectory without entering the task's unsafe state.",
                    "concepts": ["ultrasound guidance", "dual-robot coordination", "safe RL", "orthopedic trajectory"],
                },
            ],
        },
    ]
)


# A lesson must teach a decision process, not only name a task.  These briefs
# are deliberately clinician-facing and simulator-specific: they explain what
# to observe, what to try, how to judge the attempt, and what to think about
# before moving on.  Detailed physics/procedure truth remains owned by the
# corresponding native room definition.
LESSON_GUIDES = {
    "psm-precision-reach": {
        "notice": "Watch how a small hand command changes both tool-tip position and orientation in the camera view.",
        "practice": "Move in one axis at a time, slow near the target, then settle without a corrective wobble.",
        "steps": ["Identify the tool tip and target before moving.", "Use coarse motion only in open space.", "Approach on one clear axis, then align the remaining axes.", "Hold the final pose and inspect the error."],
        "success_checks": ["The target is reached without contact or overshoot.", "The final pose is stable rather than oscillating.", "The path uses deliberate, economical corrections."],
        "reflection": "Which view or axis made depth hardest to judge?",
    },
    "ecm-camera-control": {
        "notice": "Notice how camera translation and rotation change scale, horizon, occlusion, and apparent tool motion.",
        "practice": "Frame the target with useful surrounding anatomy, then hold the view while the tool remains untouched.",
        "steps": ["Locate the target and both workspace boundaries.", "Translate until the target is near image centre.", "Adjust angle and distance for a clear working corridor.", "Stop and confirm the view stays stable."],
        "success_checks": ["The target is centred and remains visible.", "Useful context is preserved around the target.", "The camera does not collide with or displace the instruments."],
        "reflection": "Did a closer view improve precision or remove useful context?",
    },
    "dual-arm-coordination": {
        "notice": "Track both tips, their crossing risk, and which instrument currently has the clearest route.",
        "practice": "Use both keyboard hands together to place the instruments while preserving separation and visibility.",
        "steps": ["Survey both targets and choose an approach side for each arm.", "Move the first instrument into a safe staging pose.", "Advance the second while monitoring separation.", "Settle both tools without crossing shafts."],
        "success_checks": ["Both targets are reached.", "No tool-tool or protected-scene contact occurs.", "Neither instrument blocks the operative view."],
        "reflection": "When was simultaneous motion better than moving one arm at a time?",
    },
    "needle-lift": {
        "notice": "Look for jaw alignment on the needle body, physical custody after closure, and orientation during the lift.",
        "practice": "Approach the middle third, close once aligned, lift clear, orient, and recover without a drop.",
        "steps": ["Inspect the needle pose and choose a grasp on its curved body.", "Approach with open jaws and minimal lateral motion.", "Close only when the body lies between both jaws.", "Lift, orient for the next action, and hold a stable recovery pose."],
        "success_checks": ["The needle moves only after a confirmed physical grasp.", "It clears the tray without scraping or dropping.", "The final orientation is usable for a handoff or tissue bite."],
        "reflection": "Was failure caused by approach alignment, closure timing, or lift direction?",
    },
    "needle-handover": {
        "notice": "The critical moment is dual custody: the receiver must hold the needle before the sender releases.",
        "practice": "Present a safe section of the curved body, establish the receiving grasp, then release and separate.",
        "steps": ["Acquire and present the needle in the shared workspace.", "Bring the receiver to an exposed section of the body.", "Close the receiving jaws and confirm the needle is retained.", "Open the sender, separate both tools, and present the next driving angle."],
        "success_checks": ["A measurable dual-grasp overlap occurs before release.", "The needle never becomes free or contacts the table.", "The receiver finishes with a useful orientation."],
        "reflection": "Could the receiver approach without forcing the sender to compensate?",
    },
    "block-transfer": {
        "notice": "Use the rigid block to isolate timing, shared-workspace positioning, and release order from needle curvature.",
        "practice": "Lift, present, receive, release, and place the block using a clear two-arm rhythm.",
        "steps": ["Acquire the block with Instrument 1.", "Present it centrally with room for the second jaw.", "Secure Instrument 2 before releasing Instrument 1.", "Move to the destination and place with a controlled release."],
        "success_checks": ["The block remains physically held throughout transfer.", "The instruments do not collide.", "Placement finishes inside the target region."],
        "reflection": "Which part of the transfer should stay identical when the object changes?",
    },
    "rl-reach": {
        "notice": "Connect each policy input and action to the reward curve, episode outcome, and final reach error.",
        "practice": "Review the starter recipe, run a small Cartesian IK-relative training job, then compare early and later checkpoints.",
        "steps": ["Inspect the observation, action, termination, and reward definitions.", "Keep the first run small and reproducible.", "Watch reward trend, success rate, and action smoothness together.", "Load a checkpoint and evaluate it from unseen resets."],
        "success_checks": ["The run produces a checkpoint and reproducible manifest.", "Reach success improves without unstable action growth.", "You can explain what behaviour each reward term encourages."],
        "reflection": "Could the policy increase reward while behaving in a way you would reject?",
    },
    "rl-needle-lift": {
        "notice": "A lift score is incomplete unless grasp quality, orientation, contacts, force limits, and safe completion are represented.",
        "practice": "Compare the needle-lift reward with an expert trajectory before approving a bounded training recipe.",
        "steps": ["Inspect the Cartesian action and observation spaces.", "Map every reward term to visible behaviour.", "Identify missing safety and grasp-quality signals.", "Run only the reviewed recipe, then compare checkpoint behaviour with demonstrations."],
        "success_checks": ["Training and human demonstrations use the same Cartesian action meaning.", "Reported success includes stable physical needle custody.", "Evaluation exposes unsafe contacts and reward loopholes."],
        "reflection": "Which desired behaviour is easier to demonstrate than to express as a reward?",
    },
    "camera-view-generalization": {
        "notice": "Compare which needle, jaw, target, and depth cues stay reliable when the camera viewpoint changes.",
        "practice": "Play the three views from the beginning and record what changes versus what remains task-relevant.",
        "steps": ["Study the training view and name the visible decision cues.", "Switch to each new view at the same action phase.", "Identify cues that move, disappear, or become ambiguous.", "Choose observations or augmentation that could preserve the behaviour."],
        "success_checks": ["Stable task cues are separated from camera-specific appearance.", "At least one likely failure under viewpoint shift is identified.", "A concrete robustness experiment is proposed."],
        "reflection": "Which cue did the training camera make deceptively easy?",
    },
    "point-cloud-policy": {
        "notice": "Depth changes a colored pixel into a 3D surface point, but occlusion and missing depth still remain.",
        "practice": "Relate visible tissue and tool structures to the geometry a point-cloud policy can actually observe.",
        "steps": ["Identify ambiguous relationships in the RGB view.", "Ask which ambiguities depth resolves.", "Mark regions hidden from the endoscope.", "Compare single-view 3D input with multi-view image input."],
        "success_checks": ["You can state what RGB-D adds and what it cannot recover.", "Occluded anatomy is not treated as observed geometry.", "The chosen representation matches the task question."],
        "reflection": "Would another camera or a better 3D representation reduce the main uncertainty?",
    },
    "needle-generalization": {
        "notice": "Across needles, separate incidental appearance from pose, curvature, exposed body, and graspable geometry.",
        "practice": "Compare the same phase across all five examples and predict which instance is hardest before watching the result.",
        "steps": ["Choose one common action phase.", "Compare curvature, scale, pose, and visibility across instances.", "Predict the required grasp adjustment.", "Check whether the observed action supports the prediction."],
        "success_checks": ["The comparison uses task geometry rather than color alone.", "A plausible out-of-distribution failure is identified.", "A useful training variation is proposed."],
        "reflection": "What must the policy estimate explicitly instead of memorizing?",
    },
    "tissue-retraction": {
        "notice": "Judge the trade-off between improved exposure, grasp location, tissue load, occlusion, and elastic recovery.",
        "practice": "Choose an atraumatic region, establish contact, retract gradually, hold exposure, and release deliberately.",
        "steps": ["Inspect the anatomy and select a safe grasp region.", "Approach with the jaws aligned to the surface.", "Retract only far enough to reveal the target corridor.", "Hold, inspect load and visibility, then release under control."],
        "success_checks": ["The intended field becomes visible.", "Peak solver-derived load stays within the room's range.", "The organ recovers without a drop or abrupt release."],
        "reflection": "Did additional retraction still improve the view, or only increase load?",
    },
    "suture-pad": {
        "notice": "The needle, pad, scissors and PSMs are NVIDIA assets; Isaac Lab and PhysX own their motion and contact.",
        "practice": "Pick up and orient the needle, rehearse a tangent pad approach, transfer custody, then exchange and return the scissors.",
        "steps": ["Acquire the needle on its curved body.", "Align tangent to the intended pad entry without forcing the rigid surface.", "Establish receiver custody before sender release.", "Pick up, present and return the scissors to their table landing area."],
        "success_checks": ["The needle moves only through physical jaw contact.", "Dual custody occurs before release.", "The scissors are physically retained and returned without a drop.", "No puncture or cut is claimed from rigid contact alone."],
        "reflection": "Which skills can be evaluated with rigid contact, and which require a future native deformable or topology-changing backend?",
    },
    "dr-anmar-suturable-tissue-live": {
        "notice": "The live PhysX lane supports intact deformation and contact; arbitrary puncture, persistent tracts, thread passage, cutting, and clinical claims remain blocked.",
        "practice": "Inspect the wound, establish counter-traction, approximate the edges, present the DrAnmar Needle, and record force and deformation evidence.",
        "steps": ["Inspect both wound flaps and the open gap.", "Establish gentle bilateral contact and counter-traction.", "Approximate the edges without crushing or inversion.", "Present the needle and record the live suture and tissue telemetry."],
        "success_checks": ["The tissue remains finite and attached.", "Wound motion and tool contacts are recorded.", "The suture material-history telemetry remains available.", "No topology-changing capability is inferred from intact contact."],
        "reflection": "Which observed signal is real simulator evidence, and which remaining behavior still requires a topology-capable backend or bench data?",
    },
    "dr-anmar-hemostasis-live": {
        "notice": "The live room supports deformable-vessel and rigid-clip contact, not plastic clip forming, pressure-tight sealing, pulsatile flow, bleeding, rupture, or clinical hemostasis.",
        "practice": "Inspect the lumen and clip, pick up the clip, align it across the vessel, establish controlled contact, and record the state.",
        "steps": ["Confirm vessel axis, lumen and clip orientation.", "Acquire the clip with filtered jaw contact.", "Align the full-section serrated collider across the vessel.", "Establish controlled contact and capture pose, force and deformation evidence."],
        "success_checks": ["The clip remains under physical control.", "The vessel deforms without non-finite state.", "Clip pose and deformable telemetry are recorded.", "A critically damaged surrogate state is never called admissible."],
        "reflection": "What additional solver and bench evidence would be required before a contact result could support an occlusion or seal claim?",
    },
    "needle-hoop-threading-live": {
        "notice": "Coordinate both instruments around the needle, strand, and ring.",
        "practice": "Acquire the threaded needle, pass it through the ring, and transfer it between instruments.",
        "steps": ["Grasp the needle body without pinching the strand.", "Guide the curved needle and strand through the ring.", "Establish receiver custody before releasing the first instrument.", "Pull the needle clear and recover both tools."],
        "success_checks": ["The needle remains under instrument control.", "The strand stays attached during the ring pass and handoff.", "The needle and strand clear the ring."],
        "reflection": "Which instrument movement gave you the best control of strand slack?",
    },
    "needle-passing-regrasp-live": {
        "notice": "A successful handoff is not enough; the receiver must finish with a usable needle-driving angle.",
        "practice": "Orient, pass the target, establish receiver custody, release, and deliberately rebuild presentation.",
        "steps": ["Orient the needle into a readable presentation.", "Move its body through the ordered spatial target.", "Close the receiver on an exposed safe section.", "Release the sender and present the needle for the next bite."],
        "success_checks": ["The ordered target passage is completed.", "There is physical dual-grasp overlap with no drop.", "Final needle orientation is stable and repeatable."],
        "reflection": "Was the final grasp chosen for easy transfer or for the next surgical action?",
    },
    "ultrasound-access-live": {
        "notice": "Treat target confidence, needle visibility, trajectory, contact, and protected-structure clearance as simultaneous signals.",
        "practice": "Acquire and stabilize the ultrasound target, align an in-plane path, advance under continuous visibility, then verify.",
        "steps": ["Position the probe until the target and protected structures are clear.", "Plan a trajectory that keeps the needle in plane.", "Advance only while the tip remains visible.", "Stop at the target zone and verify clearance."],
        "success_checks": ["The target remains visible through the approach.", "The tip reaches the planned zone without unsafe-state entry.", "Protected-structure margin stays positive."],
        "reflection": "When visibility was lost, did you stop and reacquire or continue from assumption?",
    },
    "recovery-live": {
        "notice": "Recovery begins with recognizing uncertainty and stopping; speed is secondary to restoring a known safe state.",
        "practice": "Identify the injected failure, stop both tools, restore visibility, reacquire only if safe, then hand control back.",
        "steps": ["Name the failure from available evidence.", "Stop motion and move to a safe clearance pose.", "Restore camera view and reassess object/anatomy state.", "Recover the object conservatively or request human hand-back."],
        "success_checks": ["Unsafe movement stops promptly.", "Visibility and tool state are re-established before manipulation resumes.", "Recovery completes without compounding the original event."],
        "reflection": "What was the earliest trustworthy signal that normal execution should stop?",
    },
    "anatomy-variation-live": {
        "notice": "Patient-shape variation changes corridors, occlusions, target depth, and useful camera pose even when the task goal is identical.",
        "practice": "Survey each anatomy preset, plan a fresh route, visit targets in order, and compare trajectories.",
        "steps": ["Use the adjustable camera to survey the scene.", "Identify the target path and protected obstacles.", "Visit each ordered target with controlled tool clearance.", "Save the trajectory and repeat on another anatomy preset."],
        "success_checks": ["All ordered targets are reached.", "The path adapts to geometry rather than replaying fixed coordinates.", "Comparable trajectories are recorded for review."],
        "reflection": "Which geometric change most altered the approach plan?",
    },
    "sonogym-l4-navigation-live": {
        "notice": "Relate probe motion and contact to image change; the goal is a stable target plane, not merely a bright ultrasound frame.",
        "practice": "Scan systematically, localize L4, refine orientation, and hold the transverse target plane.",
        "steps": ["Begin with a broad, ordered scan rather than random motion.", "Use anatomical image changes to bracket the L4 region.", "Refine translation and angle in small increments.", "Hold the best plane and inspect target error."],
        "success_checks": ["The native SonoGym target plane is reached.", "Probe motion remains short and stable near the target.", "The target view persists during the hold."],
        "reflection": "Which image feature was most useful for deciding the next probe movement?",
    },
    "sonogym-l4-reconstruction-live": {
        "notice": "New information comes from complementary views; repeated passes over the same surface add little coverage.",
        "practice": "Plan distinct sweeps, monitor reconstructed coverage, and redirect the probe toward missing regions.",
        "steps": ["Inspect the initial uncovered surface state.", "Acquire one stable sweep across the target.", "Use the coverage feedback to choose a complementary path.", "Stop when additional motion adds little new surface."],
        "success_checks": ["Coverage rises across successive sweeps.", "Redundant sampling is limited.", "The reconstructed L4 surface is produced by the native SonoGym workflow."],
        "reflection": "Which sweep contributed the most new geometry, and why?",
    },
    "sonogym-l4-guided-surgery-live": {
        "notice": "Balance localization confidence, planned endpoint error, trajectory smoothness, and the environment's unsafe-state signal.",
        "practice": "Localize with ultrasound, plan the instrument corridor, advance in controlled increments, and stop on uncertainty.",
        "steps": ["Acquire and stabilize the L4 target view.", "Choose a trajectory using the native task state.", "Advance while maintaining localization confidence.", "Reach the planned endpoint or stop immediately on unsafe-state entry."],
        "success_checks": ["Endpoint error reaches the task threshold.", "No native unsafe-state condition is entered.", "Probe and instrument motion remain coordinated and reproducible."],
        "reflection": "Which uncertainty should trigger a pause before endpoint error starts increasing?",
    },
}


def _courses_with_guides() -> list[dict]:
    guided_courses = []
    for course in COURSES:
        guided_lessons = []
        for lesson in course["lessons"]:
            guide = LESSON_GUIDES.get(lesson["id"])
            if guide is None:
                raise RuntimeError(f"Guided lesson content is missing for {lesson['id']}")
            guided_lessons.append({**lesson, **guide})
        guided_courses.append({**course, "lessons": guided_lessons})
    return guided_courses


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
    courses = _courses_with_guides()
    lessons = [lesson for course in courses for lesson in course["lessons"]]
    return {
        "courses": courses,
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
            "sonogym": "https://sonogym.github.io/",
            "sonogym_code": "https://github.com/SonoGym/SonoGym",
        },
    }
