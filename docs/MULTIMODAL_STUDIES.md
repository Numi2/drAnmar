# Multimodal studies in Dr.Anmar

Dr.Anmar is the clinician-facing study, teaching, and evidence layer over NVIDIA Isaac for Healthcare.
It does not reimplement NVIDIA's medical sensor physics, policy runtimes, RTI DDS transport, or
hardware-in-the-loop workflows.

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
- `rheo`: expert-source trocar assembly, bimanual precision manipulation, GR00T adaptation, online RL,
  and Cosmos Transfer 2.5 references.
- `agentic`: expert-source teleoperation recording, mimic expansion, VLM annotation, LeRobot conversion,
  GR00T N1.7/openpi adaptation, and closed-loop rollout review.

The adapter root is configured with `DR_ANMAR_I4H_ROOT`. If unset, Dr.Anmar looks in
`~/.local/share/dr-anmar/vendor/i4h-workflows`. Missing workflows remain visible as defined connectors;
the UI must never fabricate sensor output when an official runtime is absent.

Dr.Anmar discovers launch modes from each pinned workflow's `metadata.json`. The web launcher exposes only
argument-free, non-privileged modes and checks the container runtime and RTI DDS license before pausing the
interactive operating room. Device access, interactive account login, and custom `--run-args` stay outside
the clinician-facing surface. Each launch produces a log and `dr.anmar.healthcare-workflow-job.v1` manifest,
and the prior lesson is restored when the job exits.

Dr.Anmar pins Isaac for Healthcare workflows **v0.6.0** at
`8b03d55ecb647a43af54470b27bd09a239870aaf` and its compatible HoloHub CLI at
`f7e791dac061e01c560d3a2c5b7da82350915b69`. The adapter verifies both installed revisions instead of
silently consuming a moving upstream layout. The four metadata-backed workflows retain the guarded web
launcher. Rheo and Agentic are intentionally source-only in the doctor interface because v0.6.0 does not
publish the same `metadata.json` launch contract for them; their reviewed scripts remain available to
research engineers without exposing arbitrary or hardware-affecting commands to clinicians.

## Coupled surgical-interaction fallback

Rooms without a validated native deformable asset use Dr.Anmar's second-generation reduced-order fallback:

- edge-coupled deformation, approximate volume preservation, organ-bed attachment and elastic recovery;
- force-gated puncture, hysteresis, needle drag, curvature-alignment resistance and a safe-force envelope;
- suture slack, strain, tension, tissue-anchor damage, pullout, strand breakage, knot tightness and security;
- cut-face removal plus visible wound opening, cut resistance and accumulated work;
- vessel compression, clip retention, over-compression damage, residual flow, bleeding and rebleeding.

Native Isaac contact, articulation, sensor and deformable tensors remain authoritative whenever present.
Fallback measurements are labeled `research_defaults_unvalidated` in live telemetry and demonstrations and
must not be interpreted as calibrated human biomechanics.

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

Every output remains for simulation, education, synthetic data, and preclinical research. It is not a
clinical decision, diagnostic output, or authorization to control a clinical robot.
