# Dr.Anmar

**Contact-driven surgical robotics for simulation, clinician demonstration,
robot learning, and patient-effect evaluation.**

[![Research software](https://img.shields.io/badge/status-research%20software-5b5bd6)](docs/EVIDENCE_LEVELS.md)
[![Isaac Sim](https://img.shields.io/badge/runtime-Isaac%20Sim-76b900)](https://developer.nvidia.com/isaac/sim)
[![Isaac Lab](https://img.shields.io/badge/robot%20learning-Isaac%20Lab-76b900)](https://isaac-sim.github.io/IsaacLab/)
[![OpenUSD](https://img.shields.io/badge/scenes-OpenUSD-5b5bd6)](https://openusd.org/)
[![Assets](https://img.shields.io/badge/assets-dr--assets-0ea5e9)](https://github.com/Numi2/dr-assets)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)

<p align="center">
  <a href="https://github.com/Numi2/dr-assets">
    <img src="https://raw.githubusercontent.com/Numi2/dr-assets/e0c8d0dee04ce241cc914fe2b152ad9ca15c4b1a/docs/media/hero-autonomous-rescue-or.png"
         width="100%"
         alt="Dr.Anmar Autonomous Rescue OR with patient, intervention stations, robot workcells, resuscitation, and physiological monitoring">
  </a>
</p>

<p align="center">
  <em>Autonomous Rescue OR — robot intervention, patient state, tools,
  resuscitation, and monitoring in one composable research scene.</em>
</p>

> A robot may command motion and intervention intent. It may not write the
> patient outcome.

Dr.Anmar owns the clinician-facing workflow, procedure rooms, robot-control
contracts, patient-effect architecture, demonstration pipeline, evaluation
surface, and evidence lifecycle. NVIDIA Isaac Sim, Isaac Lab, PhysX,
ORBIT-Surgical-derived foundations, and optional providers perform bounded
technical roles.

The result is a research platform in which robot behavior stays inspectable:
articulations and contacts advance in the simulator, post-physics evidence
drives patient effects, and learning algorithms receive reward only after the
environment computes the resulting benefit or harm.

> [!CAUTION]
> Dr.Anmar is research software for simulation, synthetic data, and evaluation.
> It is not clinically validated, is not a medical device, and must not control
> physical surgical hardware or be used for patient care.

## Robotics first

| Robot layer | Dr.Anmar responsibility |
| --- | --- |
| **Control** | Bounded Cartesian or joint commands, instrument selection, gripper state, immediate stop, operator takeover, and command provenance |
| **Simulation** | Articulations, rigid and deformable contact, attachments, particles, sensors, tool state, and procedure geometry |
| **Patient effects** | Bleeding, perfusion, repair integrity, fluid balance, ventilation, oxygenation, tissue damage, and physiological response |
| **Learning** | Causally aligned observations and actions, complete-episode datasets, policy loading, seeded rollout evaluation, and failure analysis |

The browser never declares contact, repair, division, perfusion, or success.
Those remain environment-owned outputs.

## Current robotic systems

Dr.Anmar currently integrates seven procedure-focused robot systems plus the
multi-arm OncoSurgery training cell. Each system ships with a standalone
mechanism, a composable robot payload, OpenUSD assets, interaction frames,
controller/task contracts, and a named evidence boundary.

| System | Robotic capability | Research surface |
| --- | --- | --- |
| [Wound preparation](docs/VALIDATION.md) | Contact-guarded debridement, irrigation, aspiration, interchangeable cartridges | Contact work, debris release, fluid accounting |
| [Atraumatic exposure](docs/atraumatic_exposure_robot/VALIDATION.md) | Bilateral distributed soft-tissue capture, lift, pitch, hold, and release | Force symmetry, visibility, capture retention |
| [Adaptive hemostasis](docs/adaptive_hemostasis_robot/VALIDATION.md) | Compression, irrigation, suction, clip delivery, patching, pressure verification | Flow reduction, retained repair, overload damage |
| [Adaptive anastomosis](docs/adaptive_anastomosis_robot/VALIDATION.md) | Alignment, approximation, eversion, stapling, reinforcement, leak and patency assessment | Closure geometry, retention, pressure decay |
| [Adaptive seal and divide](docs/adaptive_seal_divide_robot/VALIDATION.md) | Centering, compression, sealing, guarded division, irrigation and suction | Seal state, energy observables, blade interlock |
| [SafePlane dissection](docs/safeplane_dissection_robot/VALIDATION.md) | Distributed traction, blunt spreading, hydrodissection, guarded scissors and spatula | Protected-structure clearance and continuity |
| [Perfusion viability](docs/perfusion_viability_robot/VALIDATION.md) | Registered RGB, NIR/ICG, speckle, thermal, oxygenation, Doppler and ultrasound sensing | Multimodal fusion, fault diagnosis, abstention |
| [OncoSurgery cell](docs/SURGICAL_ONCOLOGY.md) | Three-station resection, margin sensing, specimen handling and cavity verification | Resection topology, protected structures, margin state |

### Procedure-scale systems

| Oncologic resection | SafePlane dissection |
| --- | --- |
| [![Three-arm oncologic resection cell](https://raw.githubusercontent.com/Numi2/dr-assets/e0c8d0dee04ce241cc914fe2b152ad9ca15c4b1a/docs/media/oncologic-resection.png)](https://github.com/Numi2/dr-assets/tree/main/data/Props/SurgicalOncology/OncoSurgeryCell) | [![Exploded SafePlane dissection mechanism](https://raw.githubusercontent.com/Numi2/dr-assets/e0c8d0dee04ce241cc914fe2b152ad9ca15c4b1a/docs/media/safeplane-dissection.png)](https://github.com/Numi2/dr-assets/tree/main/data/Props/SurgicalDissection/SafePlaneDissectionRobot) |
| Three coordinated stations for sensing, resection, margin assessment, and specimen handling. | Interchangeable traction, hydro, blunt, scissors, energy, and sensing components. |

| Adaptive hemostasis | Adaptive anastomosis |
| --- | --- |
| [![Adaptive hemostasis instrument](https://raw.githubusercontent.com/Numi2/dr-assets/e0c8d0dee04ce241cc914fe2b152ad9ca15c4b1a/docs/media/adaptive-hemostasis.png)](https://github.com/Numi2/dr-assets/tree/main/data/Props/SurgicalHemostasis/AdaptiveHemostasisRobot) | [![Adaptive anastomosis instrument](https://raw.githubusercontent.com/Numi2/dr-assets/e0c8d0dee04ce241cc914fe2b152ad9ca15c4b1a/docs/media/adaptive-anastomosis.png)](https://github.com/Numi2/dr-assets/tree/main/data/Props/SurgicalReconstruction/AdaptiveAnastomosisRobot) |
| Compression, clip, patch, suction, irrigation, and verification modes. | Alignment, approximation, staple formation, reinforcement, leak test, and patency. |

| Perfusion viability | Dynamic abdominal patient |
| --- | --- |
| [![Multimodal perfusion assessment instrument](https://raw.githubusercontent.com/Numi2/dr-assets/e0c8d0dee04ce241cc914fe2b152ad9ca15c4b1a/docs/media/perfusion-viability.png)](https://github.com/Numi2/dr-assets/tree/main/data/Props/SurgicalAssessment/PerfusionViabilityRobot) | [![Dynamic abdominal patient](https://raw.githubusercontent.com/Numi2/dr-assets/e0c8d0dee04ce241cc914fe2b152ad9ca15c4b1a/docs/media/dynamic-abdominal-patient.png)](https://github.com/Numi2/dr-assets/tree/main/data/Props/Patients/DynamicAbdominalPatient) |
| Registered multimodal sensing with explicit validity and abstention behavior. | Layered abdominal access, organs, pathology, respiration, wound margins, and physiology. |

Robotic motion media belongs beside the exact replay that produced it. The
front page therefore uses revision-stable system and mechanism views; procedure
video should come from complete replays that visibly include approach, contact,
patient effect, release, and recovery rather than phase labels alone.

## Contact-owned patient effects

The learning environment owns the transition from intervention to outcome.
Temporary effects require current evidence; retained repairs require persistent
attachment and integrity.

| Uncontrolled vessel | Temporary compression | Retained repair |
| --- | --- | --- |
| ![Uncontrolled vessel](https://raw.githubusercontent.com/Numi2/dr-assets/e0c8d0dee04ce241cc914fe2b152ad9ca15c4b1a/docs/media/effect-vessel-bleeding.png) | ![Compressed vessel](https://raw.githubusercontent.com/Numi2/dr-assets/e0c8d0dee04ce241cc914fe2b152ad9ca15c4b1a/docs/media/effect-vessel-compressed.png) | ![Retained vessel repair](https://raw.githubusercontent.com/Numi2/dr-assets/e0c8d0dee04ce241cc914fe2b152ad9ca15c4b1a/docs/media/effect-vessel-repaired.png) |
| Flow and blood loss remain active. | Benefit expires when bilateral contact disappears. | Benefit persists only while repair attachment and integrity remain valid. |

```mermaid
flowchart LR
    A["Clinician or policy<br/>motion + intervention intent"] --> B["Robot controller<br/>bounded commands"]
    B --> C["Isaac Sim + PhysX<br/>articulation + contact"]
    C --> D["Post-physics evidence<br/>force, geometry, flow,<br/>pressure, attachment, dwell"]
    D --> E["Dr.Anmar patient effects<br/>conservation + damage + repair"]
    E --> F["Patient state<br/>bleeding, MAP, perfusion,<br/>oxygenation, integrity"]
    F --> G["Transition reward<br/>improvement minus harm"]
    G --> A
    A -. "no outcome write path" .-> D
```

Examples of mutually supported evidence:

| Intervention | Environment-owned evidence | Failure remains visible |
| --- | --- | --- |
| Compress or clip | Bilateral force, symmetry, separation, placement, speed, retained attachment | Release, migration, overload, distal perfusion loss |
| Patch or anastomose | Distributed contact, closure gap, integrity, pressure hold, leaked particles | Delamination, residual leak, stenosis, rupture |
| Infuse | Plunger travel, outlet flow, reservoir loss, access attachment, line pressure | Disconnection, occlusion, overpressure, extravasation |
| Ventilate | Airway attachment, valve travel, delivered/leaked flow, pressure, oxygen fraction, chest excursion | Leak, unsafe pressure, inadequate delivery |

The complete mechanics and learning contracts live in the canonical
[Dr.Anmar asset catalog](https://github.com/Numi2/dr-assets).

## Executable Autonomous Rescue learning loop

Autonomous Rescue OR connects clinician demonstration to policy evaluation
without giving the policy patient-outcome controls:

```text
record complete expert episode
        ↓
pack causally aligned observations and actions
        ↓
train behavior cloning policy
        ↓
load immutable checkpoint into the live room
        ↓
run seeded patient-effect rollouts
        ↓
compare benefit, harm, release, and failure
```

```bash
./dr_anmar_rescue_il.sh policy-room /path/to/model_epoch_200.pth 2361
./dr_anmar_rescue_il.sh rollout 2361
./dr_anmar_rescue_il.sh evaluate-policy 2361 20 --continue-on-error
```

Robot, contact, vessel, vital-sign, fluid-balance, and causal camera signals
are observations. Patient-effect fields are excluded from the policy action
space, and train/validation masks are assigned at complete-episode boundaries.

## DrAnmar Learning Path

The reinforcement-learning path starts with measurable PSM pose control and
promotes policies through dual-tool coordination, contact-qualified lift, and
physical handover. The current handover incumbent is a frozen 98-observation,
14-action actor composed with a fixed pickup correction and a learned receiver
candidate-value head. It uses stable `DrAnmar-*` task IDs, GPU-native scene
cloning, seeded evaluation, live RAM/VRAM fitting, and typed benchmark evidence.

```bash
./dr_anmar_learning.sh validate
./dr_anmar_learning.sh smoke
./dr_anmar_learning.sh sweep
./dr_anmar_learning.sh tqta-start
./dr_anmar_learning.sh train
./dr_anmar_learning.sh tqta-report
./dr_anmar_learning.sh promoted-handover 1200 2000
```

See the complete [DrAnmar Learning Path](docs/DRANMAR_LEARNING_PATH.md) for task
stages, efficiency controls, promotion gates, and evidence boundaries.
The adopted
[robotic-surgery RL technical direction](docs/ROBOTIC_SURGERY_RL_TECHNICAL_DIRECTION.md)
minimizes **time to qualified task achievement**: wall-clock time from a frozen
task contract to the first checkpoint that passes held-out competence, safety,
and recovery gates.

### Force-aware puncture and pullout

![Successful DrAnmar needle puncture, receiver handover, and pullout](docs/media/dranmar-tissue-puncture-pullout-seed7001.gif)

This is a rendered replay of a successful isolated simulator episode, not an
illustration. The analytical controller for
`DrAnmar-Puncture-Pullout-Tissue-Needle-PSM-IK-Rel-Play-v0` advances the curved
needle through the native DrAnmar tissue interaction, exposes it beyond the
exit surface, transfers custody to the receiver PSM, and pulls it clear.

The recorded RTX 4090 seed `7001` produced exactly one entry event and one exit
event, `0.429 mm` maximum entry error, `0.073 mm` maximum exit error, `99.22%`
exposed needle arc, `1.008x` normalized peak puncture force, and zero hard safety
failures. The matching typed
[evaluation receipt](physics_next/receipts/tissue-puncture-pullout-analytical-rtx4090-seed7001.json)
and [benchmark contract](physics_next/benchmarks/needle-puncture-pullout-v1.json)
are committed with the task.

Tissue forces, puncture/exit events, phase validity, and success remain
environment-owned. Receiver custody uses bilateral force when available, with
a calibrated sustained closed-jaw geometry fallback before receiver-pose
coupling. These are simulator-engineering results only; they are not
biomechanical, clinical, or autonomous-surgery validation.

### Champion needle pickup and handover

<p align="center">
  <img src="docs/media/dranmar-champion-needle-pickup-handover.gif"
       width="600"
       alt="Dr.Anmar champion policy picking up a curved needle and handing it from one PSM instrument to the other in Isaac Sim and PhysX">
</p>

<p align="center">
  <em>One complete successful rollout from the frozen simulation champion:
  needle pickup, lift, presentation, receiver acquisition, giver release, and
  retained handover.</em>
</p>

This is the real promoted composite running the
`DrAnmar-Handover-Needle-Dual-PSM-IK-Rel-v0` task, not an authored animation.
The showcase was recorded on an NVIDIA RTX 4090 with seed `104729`; the episode
terminated in success after 746 control frames with no drop, premature release,
retention-loss, excessive-force, or protected-surface termination. The
[showcase evidence](docs/handover_recovery_80/champion_seed104729_showcase_evidence.json)
records the source revision, runtime, checkpoint hashes, controller settings,
episode trace, and terminal counts.

The champion is deliberately a provenance-locked **hybrid policy**, not a
claim that one end-to-end neural network learned the full handover:

| Component | Role in the rollout |
| --- | --- |
| Frozen base actor | An analytic phase controller produces the physical sequence, while a residual MLP consumes the 98-value handover observation and adds bounded corrections to the resulting 14-D dual-PSM action. The immutable checkpoint is identified by SHA-256 in the [promotion lock](docs/handover_recovery_80/promoted_policy.lock.json). |
| Pickup correction | A fixed post-reset pose correction, capped at `1.875 mm` translation and `1.5 deg` orientation, preserves the strongest verified needle-pickup behavior. |
| Receiver value head | A learned candidate-value model ranks the receiver's first acquisition correction. The selected correction is locally refined within `1.0 mm` and `1.0 deg`, with final caps of `2.5 mm` and `2.0 deg`. |
| Runtime bounds | Receiver retries, retention servo, and giver stabilization are disabled in the promoted configuration, so the displayed success is one bounded pickup-and-transfer attempt. |
| Physics authority | Isaac Sim and PhysX contacts own custody, drops, force failures, release validity, retention, and the final success terminal; the policy cannot write its own outcome. |

Across the locked development cohort, the composite succeeded in **1,292 of
1,800 episodes (71.78%)**. Needle lift reached **98.94%**; receiver acquisition
given lift reached **80.40%**; retention after acquisition reached **90.22%**.
The owner accepted it as the current simulation champion while explicitly
overriding the original 80% target. It remains a research baseline, not a
qualification claim.

### Current handover learning frontier

The promoted handover actor and the new custody-risk model solve different
problems and are meant to compose, not replace one another:

| Artifact | Output | Current evidence-backed status |
| --- | --- | --- |
| [Promoted handover actor](docs/handover_recovery_80/promoted_policy.lock.json) | Robot motion | **1,292 / 1,800** development successes (**71.78%**). This is the owner-promoted simulation incumbent; the original 80% development goal was explicitly overridden and no qualification claim is made. |
| One-decision receiver residual | Bounded motion correction | Not promoted. Its best held-out result was **+3 / 3,600** in aggregate while one seed regressed by 12; a fresh-stream update also lost to the incumbent on development seeds. |
| Calibrated active-custody risk model | Failure probability | Preserved as the leading risk model. Across three left-out physics seeds, AUC was **0.704–0.776** and nested cross-fitted Brier score improved to **0.07455** from a **0.08063** base-rate reference. |
| Counterfactual receiver trajectory | Bounded receiver XYZ scaling | Not promoted. Exact no-op replay passed, but uniform scale `0.6` reduced the activated cohort from **93 to 87** successes and added **4** receiver safety failures on the first prespecified seed. |
| [Recurrent hybrid full-action successor](docs/handover_successor/learning_report_31b7339.json) | Complete 14-D dual-arm action | Candidate only; not promoted. The final source-locked checkpoint learned from **8** exact incumbent episodes plus **8** exact safe DAgger episodes across two rounds. It combines GRU episode memory, binary gripper decisions, and learned negative-limit / precision / positive-limit motion modes, but its first network-only replay still ended in phase 0 with `protected_surface_force`. The 71.78% incumbent therefore remains unchanged. |

The actor still moves the robot. The risk model observes the one-frame
active-custody transition and estimates whether retention is likely to fail; it
does not emit actions and has no release or motion authority.

The causal trajectory screen is complete. Separate same-index Isaac processes
reproduced prebranch tensors and no-op terminal outcomes exactly, while
neighboring vectorized PhysX clones did not remain isolated after an
intervention. The only surviving speed candidate was negative, so no
behavior-cloning or PPO update was started from it and the promoted actor
remains unchanged.

The next learning stage is executable through
[`dr_anmar_handover_successor.py`](scripts/dr_anmar_handover_successor.py).
Bit-identical safe successes from the frozen incumbent can be admitted as
distillation demonstrations, which is how the hand-authored runtime is replaced
by one network without treating failed incumbent actions as expertise. Offline
action error is not a promotion metric: the first eight-episode clone failed
closed loop despite low held-out error. Two prespecified DAgger rounds then used
an oracle fraction of `0.9`. In each round, exactly four of eight screened seeds
produced safe terminal successes and were admitted; all retention, drop, or
force failures were quarantined. The candidate visits states under the
oracle/student mixture, while the frozen promoted composite supplies labels for
every visited state. A trajectory enters training only when two
single-environment replays are bit-identical, end in success, contain no safety
event, preserve source and checkpoint hashes, and cover all four
action-bearing phases.

The custody-risk model only allocates independent collection for failure states.
Better-than-incumbent actions there must still come from constrained trajectory
optimization or clinician teleoperation, then beat two bit-identical no-op
controls without any safety event. Accepted offline successes, accepted DAgger
episodes, and independently verified rescues train one compact
phase-conditioned recurrent network that emits the full 14-D action. Episode
order is preserved during training because grasp and release boundaries depend
on history that is not fully represented by one observation. Grippers are
binary decisions; each motion channel separately classifies negative
saturation, precision control, or positive saturation before emitting its
continuous precision value. This prevents regression from averaging a
safety-critical limit command with the following fine-control command.
Episode-level splitting prevents frame leakage, qualification seeds are
forbidden from training, and every checkpoint is candidate-only until live
seeded evaluation promotes it.

Incumbent-only DAgger is now stopped. It reduced held-out action MAE but did not
produce a safe autonomous phase-0 policy. The next useful information must be
independent safe expert action in student-visited phase-0 states and in the
quarantined retention failures, sourced from constrained trajectory
optimization or clinician teleoperation with immutable receipts. More cloning
of the incumbent would repeat its capability ceiling without resolving the
closed-loop error.

```text
record exact incumbent success twice → admit offline distillation demonstration
→ train candidate → record exact safe oracle-mixture replay twice
→ admit on-policy DAgger labels → retrain
or lock teacher proposal → record control/control/teacher → accept rescue
→ train full-action successor → compare against the frozen incumbent
```

The hand-authored recovery composition remains sealed as the 71.78% regression
baseline until a successor passes the live gate. It is the DAgger oracle during
training only; none of its recovery wrappers are present in the successor
runtime. It should be archived, not deleted, only after a learned checkpoint
demonstrably replaces it.

The new source-bound successor record, including every screened seed, admitted
dataset hash, candidate hash, and final failed gate, is the
[recurrent successor learning report](docs/handover_successor/learning_report_31b7339.json).
The earlier receiver-policy calibration record remains in the
[receiver policy learning report on the experiment branch](https://github.com/Numi2/drAnmar/blob/experiment/handover-attempt-rl/docs/handover_recovery_80/attempt_policy_learning_report.json).
## Doctor Studio

Doctor Studio is the clinician-facing workspace for live simulation,
teleoperation, demonstration recording, guidance, policy comparison, and
failure analysis.

| Live operating room | Skills Twin |
| --- | --- |
| ![Dr.Anmar live operating room](docs/screenshots/dr-anmar-live-operating-room-2026.png) | ![Dr.Anmar Skills Twin](docs/screenshots/dr-anmar-skills-twin-2026.png) |
| Robot control, cameras, guidance, immediate stop, recording, and room state. | Phase timing, trajectory inspection, replay comparison, and clinician-selected references. |

| Multimodal study lab | Policy lab |
| --- | --- |
| ![Dr.Anmar multimodal study lab](docs/screenshots/dr-anmar-multimodal-lab-2026.png) | ![Dr.Anmar policy lab](docs/screenshots/doctor-studio-policy-lab.png) |
| RGB, depth, segmentation, point clouds, wrist cameras, pose, torque, contact, and annotations. | Seeded evaluation, perturbation, failure review, and bounded policy comparison. |

Control and data contracts:

- [Keyboard and controller interaction](docs/KEYBOARD_CONTROLS.md)
- [Camera-native hand teleoperation](docs/WEBCAM_TELEOPERATION.md)
- [Multimodal studies](docs/MULTIMODAL_STUDIES.md)
- [Executable expert guidance](docs/EXECUTABLE_EXPERT_GUIDANCE.md)

## Platform architecture

```mermaid
flowchart TD
    A["Clinician / researcher"] --> B["Doctor Studio<br/>control, guidance, studies, review"]
    B --> C["Dr.Anmar hub<br/>identity, operator lease, lifecycle, provenance"]
    C --> D["Isaac worker<br/>task, robot, sensor, controller, recorder"]
    D --> E["Isaac Lab<br/>articulation + learning"]
    D --> F["PhysX<br/>rigid + deformable + particles"]
    D --> G["OpenUSD<br/>scenes + assets + variants"]
    D --> H["Evidence<br/>trajectories + metrics + dataset cards"]
```

Downloaded assets, checkpoints, demonstrations, logs, and runtime state remain
outside Git. The repository contains the code, contracts, authored assets,
documentation, and revision-bound evidence references needed to reproduce a
study.

## Evidence boundary

Dr.Anmar keeps five claims separate:

| Level | Establishes | Does not establish |
| --- | --- | --- |
| Product capability | A workflow is integrated and available | Numerical fidelity |
| Repository verification | Source, schemas, manifests, paths, and contracts are internally consistent | Native engine behavior |
| Native-simulator evidence | A named revision ran on a recorded simulator, stack, and GPU | Real-world transfer |
| Real-world evidence | Instrumented hardware or bench data support a specific correlation | Clinical effectiveness |
| Clinical evidence | A defined clinical study supports a specific claim | Claims outside that study |

Current parameters are engineering parameters unless a robot-specific artifact
records instrumented calibration. Evidence does not transfer automatically
between workcells, revisions, representations, simulator versions, GPUs, or
physics configurations.

Start with:

- [Current release readiness](docs/CURRENT_RELEASE_READINESS.md)
- [Revision-bound portfolio evidence](physics_next/benchmarks/dranmar-portfolio-evidence-index.json)
- [Evidence levels](docs/EVIDENCE_LEVELS.md)
- [Validation backlog](docs/VALIDATION_BACKLOG.md)

## Quick start

The asset catalog is a pinned submodule, so clone recursively:

```bash
git clone --recurse-submodules https://github.com/Numi2/drAnmar.git
cd drAnmar
cp .env.example .env
```

Set `ISAAC_PYTHON` in `.env` to the selected Isaac environment, then install
the local extensions and start Doctor Studio:

```bash
export IsaacLab_PATH=/absolute/path/to/IsaacLab
./orbitsurgical.sh
./dr_anmar_suite.sh start
```

Open [http://localhost:2360](http://localhost:2360).

```bash
./dr_anmar_suite.sh status
./dr_anmar_suite.sh logs
./dr_anmar_suite.sh restart
./dr_anmar_suite.sh stop
```

The simulator backend requires Linux x86-64 and a compatible NVIDIA GPU. The
browser, source, and documentation can be inspected on macOS and Windows.
Recorded native-simulator lanes include Isaac Sim 5.1 / Isaac Lab 2.3.2 and
Isaac Sim 6.0.1.0. The current isolated lane is defined by
`config/physics-next-lock.json`, including full source revisions and exact
Torch packages; use the exact stack named by the evidence artifact you are
reproducing.

See [SECURITY.md](SECURITY.md) before exposing Doctor Studio beyond a trusted
LAN or private VPN.

## Repository map

```text
web/                    Doctor Studio browser application
scripts/                Hub, workers, control adapters, generators
examples/               Native CUDA evidence programs
source/extensions/      Simulator tasks and robot integrations
source/standalone/      Teleoperation, data, training, policy workflows
source/extensions/orbit.surgical.assets/
                        Pinned canonical dr-assets repository
physics_next/           Next-generation mechanics and evidence contracts
docs/                   Architecture, mechanisms, operation, evidence
dr_anmar_*.sh           Service, runtime, asset, and training launchers
```

## Research navigation

- [Ownership and provenance](docs/OWNERSHIP.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Asset catalog](docs/ASSET_CATALOG.md)
- [Native suture mechanics](docs/DR_ANMAR_SUTURE_4_0.md)
- [Warp suture research backend](docs/DR_ANMAR_WARP_SUTURE.md)
- [Dynamic-patient boundary](docs/DYNAMIC_PATIENT_VALIDATION_BOUNDARY.md)
- [Multimodal study contract](docs/MULTIMODAL_STUDIES.md)
- [Contributing](CONTRIBUTING.md)

## Ownership, attribution, and license

Dr.Anmar owns the product workflow, procedure rooms, robot integrations,
patient-effect architecture, learning interface, and evidence lifecycle in this
repository. The compatibility namespace and identified task/robot foundations
retain their ORBIT-Surgical-derived BSD-3-Clause attribution. NVIDIA Isaac Sim,
Isaac Lab, PhysX, Isaac for Healthcare, and optional providers retain their own
licenses and are not bundled unless explicitly documented.

Dr.Anmar is distributed under the [BSD 3-Clause License](LICENSE). Publications
should report the Dr.Anmar revision, pinned `dr-assets` revision, simulator and
Isaac Lab versions, GPU/driver, scenario and seed, control policy, sensor
profile, and applicable evidence artifact.
