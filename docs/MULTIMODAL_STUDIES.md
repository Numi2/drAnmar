# Multimodal studies in Dr.Anmar

Dr.Anmar owns the clinician-facing study, teaching, and evidence workflow. It connects that workflow to
NVIDIA Isaac for Healthcare and Isaac Lab providers without reimplementing their medical sensor physics,
policy runtimes, RTI DDS transport, or hardware-in-the-loop workflows. See
[`OWNERSHIP.md`](OWNERSHIP.md) for the product and provider boundary.

## Division of responsibility

Dr.Anmar owns:

- clinical questions and guided study recipes;
- plain-language explanations of observations, actions, policies, and failure modes;
- procedure phases and event annotations;
- clinician references, interventions, coaching, and evidence review;
- synchronized demonstration manifests, checksums, dataset cards, and study exports.

Isaac for Healthcare and Isaac Lab own:

- RGB, stereo, depth, segmentation, point-cloud, and deformable-object simulation;
- robotic surgery, robotic ultrasound, SO-ARM, and telesurgery reference workflows;
- MAISI anatomy generation and OpenUSD digital twins;
- π₀, GR00T, reinforcement-learning, and imitation-learning integrations;
- XR, haptic, RTI DDS, Holoscan, and physical hardware bridges.

## Current live surgical-twin channels

| Signal | Cadence | Dataset representation | Purpose |
|---|---:|---|---|
| Robot action, joint state, tool and object pose | 50 Hz | Numeric arrays | Control and kinematics |
| Anatomy showcase pose | 50 Hz | World-frame pose | Registration and provenance |
| Applied and computed joint torque | 50 Hz | Numeric arrays | Interaction-effort research |
| Stereo endoscope | 5 Hz dataset; live stream | Left and right RGB | Depth cues and robustness |
| Left-endoscope depth | 5 Hz | Metric depth map | Geometry |
| Semantic segmentation | 5 Hz | `uint32` ID map + label table | Tool/anatomy identity |
| Point cloud | 5 Hz | Fixed-grid camera-frame XYZ metres | 3D policy observations |
| Instrument wrist cameras | 5 Hz dataset; live stream | One or two RGB views | Tool-centred perception |
| Contact and tissue signals | up to 50 Hz | Force, displacement, deformation proxy, stress | Physical interaction research |
| Operator input source | 50 Hz | Enumerated code | Human-factors studies |
| Operator attention/gaze | 50 Hz | Normalized image coordinate + source | Attention studies |
| Procedure phase and event | 50 Hz + manifest events | Codes and human-readable log | Clinical meaning |

The browser pointer is explicitly recorded as an **attention proxy**, not eye tracking. Real eye-gaze data
must enter through an external eye tracker or XR adapter and retain its source label.

## Procedure annotation vocabulary

Phases:

1. Setup
2. Approach
3. Grasp
4. Manipulation
5. Recovery

Events:

- target visible;
- contact;
- grasp;
- task complete;
- human handoff;
- safety review.

The vocabulary is intentionally small so a clinician can annotate while operating. Specialty-specific
extensions should be versioned rather than silently changing these base labels.

## NVIDIA workflow bindings

- `robotic_surgery`: dVRK/STAR tasks, surgical sensors, demonstrations, RL, and imitation learning.
- `robotic_ultrasound`: B-mode sensor simulation, probe pose, acoustic configuration, π₀/GR00T, and Holoscan.
- `so_arm_starter`: room/wrist views, accessible teleoperation, LeRobot conversion, and GR00T onboarding.
- `telesurgery`: XR, haptics, low-latency video, RTI DDS, handover, and hardware-in-the-loop.
- `rheo`: expert-source trocar assembly, bimanual precision manipulation, surface-deformable cloth with
  Newton/PhysX backends, XR demonstration capture, GR00T adaptation, and online RL.
- `agentic`: NVIDIA's six native surgical Arena environments, scripted expert state machines,
  teleoperation recording, Mimic expansion, LeRobot conversion, GR00T/openpi adaptation, and rollout review.
- `catheter_navigation`: patient-specific endoluminal navigation, fluoroscopy/DSA, CT ingestion, and
  NVIDIA's X-PBD catheter simulation.

The adapter root is configured with `DR_ANMAR_I4H_ROOT`. If unset, Dr.Anmar looks in
`~/.local/share/dr-anmar/vendor/i4h-workflows-current`. The active path is a symlink to a versioned checkout,
so a previous qualified provider can be restored without modifying its files. Missing workflows remain
visible as defined connectors;
the UI must never fabricate sensor output when an official runtime is absent.

Dr.Anmar discovers legacy workflow launch modes from each pinned workflow's `metadata.json`. For Agentic
surgical environments, `workflows/agentic/config/environments/<env>.yaml` is the upstream source of truth.
The adapter reads those files without copying their robot, action, bridge-port, or task definitions. The
web launcher exposes only registered, non-hardware modes and checks the appropriate runtime before pausing
the interactive operating room. Each launch produces a log and `dr.anmar.healthcare-workflow-job.v1`
manifest, and the prior lesson is restored when the job exits.

Dr.Anmar pins Isaac for Healthcare workflows **v0.7.0** at
`9b526c6d107254727d3b113c612fb860fc65a5b2` and its compatible HoloHub CLI at
`f7e791dac061e01c560d3a2c5b7da82350915b69`. The adapter verifies both installed revisions instead of
silently consuming a moving upstream layout. Agentic's six surgical environments are launched through
NVIDIA's own Arena state-machine entrypoint; Dr.Anmar does not substitute a custom simulator, action space,
grasp controller, reward, or physics loop. Rheo remains expert-source-only until it has an equally narrow
upstream launch contract.

## Native physics boundary

Dr.Anmar has no surgical-interaction fallback. A room launches only when an active native worker owns its
complete physical state. PhysX owns rigid manipulation and the promoted liver-retraction deformable; the
official Isaac for Healthcare workflow owns ultrasound ray tracing and B-mode output; Newton VBD and the
topology-changing research backend remain unavailable to doctors until their room contracts are complete.
Missing tissue, strand, puncture, cut, flow or ultrasound capabilities keep the corresponding room closed.

## Study workflow

1. Write one clinical research question.
2. Choose only the signals needed to answer it.
3. Select a policy starting point and expert-control method.
4. Export the Dr.Anmar study manifest.
5. Inspect and configure the bound NVIDIA workflows.
6. Calibrate cameras, robot, anatomy, and operator devices.
7. Record complete, annotated demonstrations.
8. Freeze a content-addressed dataset card.
9. Train or adapt the selected policy.
10. Evaluate across sensor, anatomy, calibration, and supervision challenges.
11. Review failures, safety events, and interventions with clinicians.

When the starting point is Dr.Anmar's executable simulation expert, inspect `clean_reference_eligible`,
`degraded_reasons`, `behavior_cloning_reference_candidate` and `reference_review_status` before adding the
trajectory to a study. Eight completed phases alone are not evidence of task success or expert quality. The
controller and capture semantics are documented in
[`EXECUTABLE_EXPERT_GUIDANCE.md`](EXECUTABLE_EXPERT_GUIDANCE.md).

Every output remains for simulation, education, synthetic data, and preclinical research. It is not a
clinical decision, diagnostic output, or authorization to control a clinical robot.
