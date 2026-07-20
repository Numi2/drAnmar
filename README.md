# Dr.Anmar

Dr.Anmar is a clinician-facing surgical robotics learning and research studio built on
[ORBIT-Surgical](https://github.com/orbit-surgical/orbit-surgical), NVIDIA Isaac Sim, Isaac Lab, and
optional Isaac for Healthcare workflows. It turns a robotics simulator into one coherent place where a
doctor can learn robot control, practise complete simulated procedures, record demonstrations, inspect
performance, and understand how data becomes a behavior-cloning or reinforcement-learning policy.

Researchers use the same workstation to compose OpenUSD anatomy, instruments and tasks; collect synchronized
multimodal trajectories; reproduce controlled failures; compare policies; and preserve dataset, experiment and
runtime provenance. Dr.Anmar is the pedagogical and workflow layer—the NVIDIA and ORBIT runtimes remain the
simulation engines underneath it.

<p align="center">
  <img src="docs/screenshots/dr-anmar-live-controls-2026.gif" width="960" alt="Live Dr.Anmar OpenUSD operating room showing the simulated surgical instrument, anatomy, keyboard controls, camera views, gripper feedback, and guided supervision">
</p>

<p align="center"><strong>A surgical robotics lab that begins with the doctor—not the simulator manual</strong><br>
Live digital-twin control, procedural mechanics, training data, coaching, failure studies, and robot learning in one browser.</p>

Every visual below was captured from Dr.Anmar itself. No figures were copied from research papers.

> [!WARNING]
> Dr.Anmar is research software for simulation, synthetic data, and education. It is not a medical
> device, is not clinically validated, and must not be used for diagnosis, treatment, patient-specific
> planning, or control of physical surgical hardware.

## What Dr.Anmar offers

| Offering | What the doctor sees | What the researcher gets |
| --- | --- | --- |
| **Guided robotics curriculum** | 29 lessons that explain control, demonstration, vision, policies, procedures, safety and recovery in clinical language | A reproducible progression from observation to teleoperation, data collection, training and comparison |
| **Interactive surgical digital twin** | 19 procedure rooms with game-like keyboard control, immediate stop/takeover, camera presets, task guidance and seven anatomy choices | Nine native ORBIT-Surgical task families, 54 registered variants, versioned OpenUSD composition and simulator telemetry |
| **Suturing, cutting and tissue workflows** | Needle pickup/handoff, interrupted and running sutures, knot rehearsal, incision, organ retraction, shunt insertion, vascular control, dissection, biopsy and recovery rooms | Coupled tissue volume/attachment, puncture/drag, thread slack/tension/failure, cut opening/work, vessel compression/rebleed, shunt, ultrasound and procedure-state telemetry synchronized with action |
| **Demonstrations and Skills Twin** | Record a complete attempt, replay it, see phase-aware coaching, and compare against a clinician-selected reference | Checksummed trajectory/manifest pairs, task-native tool paths, multimodal observations, analysis and content-addressed dataset cards |
| **Failure Lab and policy evaluation** | Practise shifted viewpoints, low light, occlusion, target variation, calibration drift, tissue variation and safe hand-back | Seeded challenge matrices, interventions, native outcomes, safety events and immutable policy-evaluation cards |
| **Multimodal study builder** | Start from a clinical question and choose what the policy should perceive | Stereo RGB, depth, segmentation, point clouds, wrist cameras, pose, torque, contact, deformation, operator input and procedure annotations, with guarded NVIDIA workflow bindings |

<p align="center">
  <img src="docs/screenshots/dr-anmar-platform-tour-2026.gif" width="960" alt="Fresh tour of Dr.Anmar Learn, Skills Twin, Failure Lab, Multimodal Lab, Policy Lab, and Anatomy Library workspaces">
</p>

<p align="center"><strong>One connected workflow</strong><br>
Learn → operate → record → receive coaching → stress-test → build a study → train and compare.</p>

### Live surgical digital twin

The operating room composes a procedure, an official anatomy preset, the correct dVRK or STAR task, its
instrument controls, safety boundary, camera views, teaching steps and measurable completion signals. The
doctor changes rooms from a plain-language procedure menu; Dr.Anmar handles the Isaac worker lifecycle and
restores the room after bounded training or an external workflow.

<p align="center">
  <img src="docs/screenshots/dr-anmar-live-operating-room-2026.png" width="960" alt="Live Dr.Anmar liver-retraction operating room with OpenUSD anatomy, dVRK instrument, complete keyboard control dock, guidance, and camera HUD">
</p>

The current library includes foundations such as needle pickup and handoff; reconstruction work such as
single and running sutures, knot tying and anastomosis; vascular shunt, clipping and bleeding control;
ultrasound-guided access; organ retraction and repositioning; cutting, dissection and biopsy; and explicit
complication recovery. Their physics are transparent engineering models for research and teaching—not claims
of validated human tissue, force, diagnostic imaging or clinical skill assessment.

## Control pedagogy designed for doctors

Dr.Anmar exposes robotics progressively instead of presenting a wall of simulator controls:

1. **Observe** — see the complete task and learn which visual cues matter.
2. **Demonstrate** — perform the same task in the digital twin and record the whole trajectory.
3. **Train** — connect the demonstration to Behavior Cloning or run a deliberately bounded RL exercise.
4. **Compare** — change anatomy, viewpoint, or object pose and inspect where behavior changes.

<p align="center">
  <img src="docs/screenshots/doctor-studio-learning-loop.gif" width="960" alt="Dr.Anmar guiding a needle-lift lesson through Observe, Demonstrate, Train, and Compare">
</p>

<p align="center"><strong>Observe → Demonstrate → Train → Compare</strong><br>
One repeatable loop connects clinical intent, robot control, data collection, and policy evaluation.</p>

![Dr.Anmar guided needle-lift lesson with the four-step learning rail](docs/screenshots/doctor-studio-guided-learning.png)

### Surgical controls that behave like a game

The live workstation translates six-degree-of-freedom robot control into a control surface clinicians can
learn immediately:

- Hold to move; release to stop.
- Precision, Normal, and Fast speed modes make fine grasping and open-space travel equally accessible.
- Position and angle controls use spatial language—*toward patient*, *away*, *roll*, *pitch*, and *yaw*.
- Every visible action has an audited keyboard equivalent, with standard gamepad support for continuous input.
- `Enter` becomes a contextual approach → grasp → lift control, while six hold-to-move surgical combinations
  provide orbiting, curved needle driving, reversal, lift/retract, and lower/approach with one key each.
- A quick tap performs a bounded precision nudge; holding the same combination key gives continuous motion.
- Needle driving uses the actual OpenUSD surface direction, locks the entry vector at puncture, and reverses that
  vector during withdrawal so the same two keys remain intuitive across entry and exit.
- `Option` and `Shift` act as temporary precision and fast clutches; `Esc` always stops and restores manual control.
- Gripper and demonstration controls sit beside movement so practice naturally becomes training data.

The complete keyboard map and the rationale for each combined movement are documented in
[`docs/KEYBOARD_CONTROLS.md`](docs/KEYBOARD_CONTROLS.md).

<p align="center">
  <img src="docs/screenshots/keyboard-surgical-control-workflow.gif" width="960" alt="Live Dr.Anmar keyboard workflow grasping a curved needle, visibly entering and fully withdrawing from the anatomy surface while held, then presenting it to a second instrument before handoff">
</p>

<p align="center"><strong>Keyboard-first needle workflow in the live simulator</strong><br>
Target-guided pickup → grasp → close-up entry → full exit while held → dual-instrument pre-handoff.</p>

<p align="center">
  <img src="docs/screenshots/fast-needle-pickup-and-handoff.gif" width="960" alt="Live dual-arm Dr.Anmar workflow rapidly picking up a curved needle, presenting it to the receiving instrument, closing the receiving jaws, releasing the original holder, and carrying the retained needle toward the organ">
</p>

<p align="center"><strong>Fast dual-instrument pickup and completed handoff</strong><br>
Pickup → presentation → dual grasp → holder release → receiver recovery → organ approach, with the active keyboard controls visible throughout.</p>

The instrument shaft remains outside the anatomy surface while the grasped needle tip can follow a bounded
12 mm entry channel. Entry is force-gated, advancing resistance responds to needle-arc alignment, and the
suturing room models visible thread slack, tension, tissue-anchor damage, pullout, breakage and knot security.
Cutting rooms remove and open intersected OpenUSD faces while recording resistance/work; tissue handling adds
mesh-coupled deformation, approximate volume preservation, attachment and recovery. Native Isaac tensors stay
authoritative. The fallback parameters are explicit unvalidated research defaults, not human-tissue or clinical
force claims.

<p align="center">
  <img src="docs/screenshots/surgical-control-panel.png" width="400" alt="Dr.Anmar game-like surgical instrument controls for speed, position, angle, and gripper">
</p>

### Robot learning explained in clinical language

The Policy Lab compares Behavior Cloning, Reinforcement Learning, and Visual Behavior Cloning with simple
resident-training analogies. Training begins with a small, reviewable recipe so a new user learns what
observations, rewards, environments, iterations, logs, and checkpoints mean before scaling up.

![Dr.Anmar Policy Lab comparing robot-learning approaches and preparing a bounded training run](docs/screenshots/doctor-studio-policy-lab.png)

### Every practice attempt becomes structured training data

The Demonstrations workspace teaches what makes an example useful, records the complete behavior from
approach through safe recovery, and keeps synchronized observations, actions, joint motion, tool motion,
and object pose together for Behavior Cloning.

![Dr.Anmar Demonstrations workspace showing recording controls and example-quality guidance](docs/screenshots/doctor-studio-demonstrations.png)

<p align="center">
  <img src="docs/screenshots/dr-anmar-skills-twin-2026.png" width="960" alt="Fresh Dr.Anmar Surgical Skills Twin view showing a recorded attempt, telemetry, phase timeline, coaching, replay, and reference-path controls">
</p>

The Skills Twin turns one saved attempt into an inspectable coaching record: tool path, object lift, grasp drift,
corrections, recovery hold, available contact and tissue signals, native simulator outcome, phase timeline and
reference comparison. Scores and cues are deliberately labeled as research proxies pending clinician validation.

### Multimodal studies without infrastructure-first UX

The Multimodal Lab starts with a medical research question, then explains each sensor or state channel by the
decision it helps a policy make. It generates a study manifest and binds selected modalities to the appropriate
Dr.Anmar or NVIDIA workflow while keeping privileged hardware modes locked until their prerequisites exist.

<p align="center">
  <img src="docs/screenshots/dr-anmar-multimodal-lab-2026.png" width="960" alt="Fresh Dr.Anmar Multimodal Study Builder showing clinician explanations, robotic surgery and ultrasound workflows, and selectable stereo, depth, segmentation, point-cloud, wrist-camera, pose and physical-interaction signals">
</p>

## What is included

- Doctor Studio web interface with a live simulated endoscope and game-like PSM controls.
- A default operating-room showcase with the surgical robot, controls, room, and liver geometry.
- Nineteen composable operating rooms grouped into suturing/reconstruction, vascular access/control,
  gallbladder/dissection, image-guided intervention, anatomy navigation, tissue handling, and complication recovery.
- A runnable ORBIT-style vascular shunt room with flexible-tube geometry, lumen alignment, insertion depth,
  buckling, wall-load, and patency telemetry.
- Live suturing mechanics: constrained thread, tissue entry/exit pins, slack, strain, tension, anchor damage,
  pullout, breakage, knot tightness/security, multi-bite closure, anastomosis/leak-test state, coupled tissue
  response, reset, procedure scoring, and synchronized recording.
- Action-driven vascular control, hemostasis, bimanual procedural ultrasound, tissue-plane dissection,
  lesion excision, and failure recovery: clip and control events require actual jaw actions, probe and needle
  roles are separate, and dissection progress requires both topology change and counter-traction.
- Live incision mechanics that remove and separate swept OpenUSD faces, expose the incision bed, record cut
  resistance/work and topology revisions, and restore the exact original topology on reset.
- Reduced-order liver, gallbladder, and bladder tissue with mesh-coupled deformation, approximate volume
  preservation, attachment resistance, elastic recovery, and native ORBIT-Surgical gross-body dynamics.
- Force-gated needle puncture with hysteresis, drag, needle-arc alignment, depth and safe-force envelopes;
  vessel compression, clip retention, over-compression damage, flow, bleeding and rebleeding telemetry.
- Guided lessons, plain-language robotics explanations, progress tracking, and a robotics glossary.
- Demonstration recording and replay for behavior-cloning experiments.
- Surgical Skills Twin analysis with telemetry-derived coaching, phase timelines, subscores, and selected-attempt replay.
- Registered task-native dVRK/STAR tool-tip paths rendered as an optional phase-coloured guide inside the live
  OpenUSD operating room, with a legacy trajectory fallback.
- Needle-lift Failure Lab with reproducible camera and visual challenges, supervision state, and immediate doctor handoff.
- Synchronized endoscopic RGB, robot state, native simulator outcomes, and available contact-force evidence in new demonstrations.
- Clinician-selected reference demonstrations with normalized trajectory comparison and coaching.
- Synchronized 50 Hz robot state plus 5 Hz endoscopic RGB, metric depth, semantic IDs, camera intrinsics,
  native task outcomes, available contact forces, and deformable-tissue research telemetry.
- Stereo endoscope views, task-native instrument wrist cameras, camera-frame metric point clouds, joint torque,
  anatomy pose, operator gaze/input provenance, and procedure phase/event annotation.
- A clinician-facing Multimodal Lab that converts a research question into an exportable study manifest bound
  to NVIDIA's robotic-surgery, robotic-ultrasound, SO-ARM/GR00T, and telesurgery workflows.
- A guarded NVIDIA workflow runner with live mode discovery, plain-language prerequisites, job logs,
  provenance manifests, automatic lesson restoration, and privileged hardware modes locked out by default.
- Simulator-native target-pose, control-calibration, and multi-organ context challenges beside the camera and
  image stressors.
- Automated challenge summaries with per-scenario descriptive statistics, intervention rate, native success,
  safety-event rate, and 95% intervals where repeated rollouts exist.
- Content-addressed dataset cards that freeze demonstration and sidecar checksums, provenance, modalities,
  task context, references, duration, and intended research use into an exportable JSON record.

The multimodal architecture and study schema are described in
[`docs/MULTIMODAL_STUDIES.md`](docs/MULTIMODAL_STUDIES.md).
- Automated challenge matrices that replay one demonstration across selected scenarios and seeds while preserving every rollout.
- Versioned demonstration and experiment manifests recording task, scenario, seed, source revision, and training recipe.
- Nine ORBIT-Surgical task families and 54 registered control/play variants.
- RSL-RL, RL-Games, Stable-Baselines3, SKRL, and Robomimic workflows inherited from ORBIT-Surgical.
- A resumable installer for seven official ORBIT-Surgical v0.1.0 anatomy scene archives.
- An Isaac Sim 5.1 / Isaac Lab 2.3.2 compatibility port of the upstream environments.

Downloaded anatomy, demonstrations, checkpoints, logs, and runtime state are deliberately kept outside
the Git repository.

The first Skills Twin metrics are explicitly research coaching proxies rather than validated clinical
assessment instruments. Runtime, telemetry, performance, and clinician-study work intentionally deferred
from the fast implementation pass is tracked in [the validation backlog](docs/VALIDATION_BACKLOG.md).

## Requirements

The simulation runtime requires a compatible Linux x86-64 machine with an NVIDIA GPU. The Dr.Anmar
source folder can be inspected and managed on macOS or Windows, but Isaac Sim cannot run there as this
project's simulator backend.

The current validated baseline is:

- NVIDIA Isaac Sim 5.1
- Isaac Lab 2.3.2
- Python 3.10 or newer inside the Isaac environment

The July 20 Gilgamesh pass exercised all nine native interactive task families, audited seven runtime anatomy
layers plus seven complete OpenUSD room compositions, and ran representative ultrasound, suturing, cutting,
tissue-manipulation, recording and shared-control workflows on an RTX 4090. Exact evidence and remaining gates
are in [`docs/GILGAMESH_VALIDATION_2026-07-20.md`](docs/GILGAMESH_VALIDATION_2026-07-20.md).
The later coupled-physics and Isaac for Healthcare v0.6.0 pass is recorded in
[`docs/GILGAMESH_PHYSICS_I4H_V060_2026-07-20.md`](docs/GILGAMESH_PHYSICS_I4H_V060_2026-07-20.md).

Install Isaac Sim and Isaac Lab using their official instructions before continuing. NVIDIA components
are dependencies and are not redistributed by this repository.

## Quick start

Clone the repository outside the Isaac Lab checkout:

```bash
git clone https://github.com/Numi2/drAnmar.git
cd DrAnmar
cp .env.example .env
```

Edit `.env` so `ISAAC_PYTHON` points to the Python executable in your Isaac-enabled environment. Runtime
data defaults to `~/.local/share/dr-anmar` and can be moved by changing `DR_ANMAR_ROOT`.
For a shared workstation, set a long random `DR_ANMAR_ACCESS_TOKEN`; Doctor Studio then requires login and
shares the resulting secure session with its operating-room worker. Set `DR_ANMAR_COOKIE_SECURE=1` behind
HTTPS. `DR_ANMAR_SENSOR_PROFILE` selects `efficient`, `stereo`, or full `research` camera capture.

Install the ORBIT-Surgical extensions:

```bash
export IsaacLab_PATH=/absolute/path/to/IsaacLab
./orbitsurgical.sh
```

Start Doctor Studio:

```bash
./dr_anmar_suite.sh start
```

Optionally install the pinned Isaac for Healthcare v0.6.0 workflow source and compatible HoloHub CLI:

```bash
./scripts/install_i4h_workflows.sh
```

Official workflow containers also require Docker Engine with NVIDIA GPU container support. DDS-based modes
such as robotic ultrasound require a valid `RTI_LICENSE_FILE`. The Multimodal Lab reports these prerequisites
and keeps launch disabled until they are present; the normal Dr.Anmar operating room does not require them.

Open [http://localhost:2360](http://localhost:2360). Useful service commands are:

```bash
./dr_anmar_suite.sh status
./dr_anmar_suite.sh logs
./dr_anmar_suite.sh restart
./dr_anmar_suite.sh stop
```

The hub and worker listen on all network interfaces by default so another trusted device can reach the
workstation. Optional token authentication and a single-operator browser lease are built in, but authentication
is disabled until `DR_ANMAR_ACCESS_TOKEN` is configured. Keep the service on a trusted LAN or private VPN,
enable the token for shared deployments, terminate HTTPS in front of it, and never expose ports 2360 or 2361
directly to the public internet. See [SECURITY.md](SECURITY.md).

## Anatomy scenes

The optional installer downloads the seven official ORBIT-Surgical v0.1.0 OpenUSD anatomy archives
(about 6.3 GB compressed) into `DR_ANMAR_ROOT`, never into the repository:

```bash
"${ISAAC_PYTHON}" scripts/install_sufia_assets.py
```

The Anatomy Library will detect the extracted scenes automatically. Review the upstream project and
release terms before redistributing any downloaded assets; this repository does not vendor them.

## Command-line workflows

```bash
# Inventory environments and educational content
./dr_anmar.sh list
./dr_anmar.sh catalog
./dr_anmar.sh doctor

# Run a bounded GPU smoke session
./dr_anmar.sh smoke Isaac-Lift-Needle-PSM-IK-Rel-v0 120

# Start a training workflow (example)
./dr_anmar_train.sh rsl_rl Isaac-Lift-Needle-PSM-v0 --num_envs 256 --max_iterations 1000
```

The upstream teleoperation, state-machine, Robomimic, training, and policy-playback programs remain
under `source/standalone`.

## Repository layout

```text
web/                  Doctor Studio browser application
scripts/              Hub, workstation, curriculum, asset installer, and checks
source/extensions/    ORBIT-Surgical simulator extensions and robot assets
source/standalone/    Simulation, teleoperation, data, and learning workflows
docs/design/          Product design reference
dr_anmar_*.sh         Portable service and training launchers
```

## Open-source development

Run the public-release checks before opening a pull request:

```bash
python3 scripts/check_public_release.py
python3 scripts/audit_project_consistency.py
python3 scripts/audit_keyboard_controls.py
python3 scripts/check_web_syntax.py
python3 -m compileall -q scripts source
bash -n dr_anmar.sh dr_anmar_suite.sh dr_anmar_train.sh dr_anmar_workstation.sh orbitsurgical.sh
```

See the [complete engineering audit](docs/COMPLETE_AUDIT_2026-07-20.md),
[CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidance, and [NOTICE.md](NOTICE.md) for provenance.

## ORBIT-Surgical citation

If this project contributes to published research, cite the ORBIT-Surgical authors:

```bibtex
@article{yu2024orbit,
  title={ORBIT-Surgical: An Open-Simulation Framework for Learning Surgical Augmented Dexterity},
  author={Yu, Qinxi and Moghani, Masoud and Dharmarajan, Karthik and Schorp, Vincent and Panitch, William Chung-Ho and Liu, Jingzhou and Hari, Kush and Huang, Huang and Mittal, Mayank and Goldberg, Ken and others},
  journal={arXiv preprint arXiv:2404.16027},
  year={2024}
}
```

## License

Dr.Anmar and the included ORBIT-Surgical-derived source are distributed under the
[BSD 3-Clause License](LICENSE). NVIDIA Isaac Sim and other optional dependencies or downloaded assets
retain their own licenses and terms.
