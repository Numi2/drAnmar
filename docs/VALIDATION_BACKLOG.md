# Dr.Anmar validation backlog

This ledger records completed engineering checks and the work still required for the Surgical Skills
Twin, Failure Lab, and Multimodal Lab. Completed runtime checks are evidence of software behavior in
simulation, not clinical validation.

## Release status

- Implementation status: Skills Twin, Failure Lab, RGB-D and semantic recording, native scene variation,
  clinician reference comparison and path guide, tissue/contact telemetry, and automated challenge-matrix
  slices complete in source.
- Runtime validation status: live Gilgamesh passes completed on 2026-07-19 and 2026-07-20. The current task,
  OpenUSD, control, recording, and representative-room evidence is in `GILGAMESH_VALIDATION_2026-07-20.md`;
  the remaining items below are still open.
- Clinical validation status: not started.
- Intended use remains simulation, education, synthetic data, and preclinical research only.

## Source audit hardening — 2026-07-20

The follow-up remediation pass closed the remaining source-level gaps identified by this ledger:

- browser mutations now use a 30-second single-operator lease shared by Doctor Studio and the embedded worker;
- optional token login uses a host-wide HTTP-only cookie, bounded login attempts, and a deployment-controlled
  HTTPS-only cookie flag;
- demonstrations are atomically saved, structurally inspected before replay/reference/dataset use, and report
  unreadable or too-short recordings without crashing the UI;
- demonstration, dataset-card, experiment, and policy-card enumeration is paginated; immutable hashes are cached
  by path, size, and modification time rather than trusted from stale manifests;
- recording has frame, duration, and uncompressed-byte ceilings; `efficient`, `stereo`, and `research` sensor
  profiles now control which Isaac camera sensors are instantiated and captured;
- procedure annotations carry monotonic event sequence numbers in both manifest and trajectory, so multiple
  between-frame events remain recoverable from the annotation ledger;
- new manifests record source, Python, Torch, CUDA, GPU, task-configuration and workflow-metadata provenance;
- the spawned anatomy prim world transform, task-native tip registration, explicit assisted-grasp state,
  tool/object distance, depth-validity, semantic-foreground and luminance signals are recorded;
- NVIDIA mode discovery validates the pinned metadata schema and fails closed on malformed or drifted modes;
- stereo drift, seeded camera dropout, and compliant-surface response variations are real challenge scenarios;
- immutable policy evaluation cards bind dataset, training run, checkpoint and completed challenge matrix hashes;
- external gaze/XR input fails closed without explicit study, consent-protocol and sensor-enable configuration;
- `docs/VALIDATION_GATES.json` separates source-ready work from host, licensed-runtime, hardware, biomechanical,
  and clinician-evidence blockers, and CI rejects unsupported validation claims.

These controls do not satisfy the biomechanical, hardware, or clinical evidence requests below. Gilgamesh was
unreachable during the initial source-only follow-up, but the later 2026-07-20 pass restored live host evidence;
only the gates explicitly demonstrated in the new runtime report should be treated as partially closed.

## Gilgamesh release-candidate evidence — 2026-07-20

### Coupled-physics and i4h v0.6 update

- Isaac for Healthcare source is now pinned to v0.6.0 at
  `8b03d55ecb647a43af54470b27bd09a239870aaf`; its compatible HoloHub CLI is pinned to
  `f7e791dac061e01c560d3a2c5b7da82350915b69`.
- The live adapter verified both revisions and retained guarded discovery for 8 surgical, 18 ultrasound and
  15 SO-ARM modes. Rheo and Agentic remain explicitly expert-source-only because they lack the same metadata
  launch contract.
- Coupled v2 tissue, needle, thread, cutting, vascular and force schemas passed deterministic execution in
  the Isaac Python environment and loaded live in the dual-PSM interrupted-stitch room.
- Complete trajectories and biomechanical calibration remain open. Exact evidence and boundaries are in
  `GILGAMESH_PHYSICS_I4H_V060_2026-07-20.md`.

- All nine native interactive task families completed 40 CUDA steps with finite reported joint state.
- Seven runtime anatomy layers and seven full room compositions passed hard scale and dependency gates.
- The suite, one-operator boundary, gripper/drive/camera controls, recording and analysis were exercised live.
- Ultrasound, suturing, cutting and liver-retraction rooms loaded with their intended procedural mechanics ready.
- Full trajectories in every room, isolated performance, licensed NVIDIA provider workflows, hardware,
  biomechanics and clinician evidence remain open. See `GILGAMESH_VALIDATION_2026-07-20.md`.

- Training now actually pauses the interactive Isaac worker and restores the exact prior procedure/anatomy
  composition; NVIDIA workflow resume uses the same full-context path.
- GPU-owning jobs, Failure Lab matrices, room switches and interactive state mutations are lifecycle-gated.
- Clean shutdown terminates managed GPU jobs; startup reconciles stale training, healthcare and matrix
  manifests and stops a matching orphan process group.
- OpenUSD startup uses a seven-scene file preflight instead of rebuilding geometry on every launch.
- Secondary cameras are JPEG-encoded only while subscribed; raw multimodal recording remains enabled.
- Demonstrations use atomic promotion, a data SHA-256, observed sampling rate and a bounded auto-save limit.
- CI now checks all keyboard controls plus curriculum/task/procedure consistency and Doctor Studio JavaScript.
- Full findings and remaining external gates are recorded in `COMPLETE_AUDIT_2026-07-20.md`.

## Gilgamesh runtime evidence — 2026-07-19

Validated on the RTX 4090 host:

- Doctor Studio and the default needle-lift relative-IK room started on ports 2360/2361 with the official
  CT-liver showcase, stereo-left, stereo-right, and wrist-1 live camera streams.
- A real browser pass loaded the operating room and Multimodal Lab without application console errors;
  the only initial error was a missing favicon, which is now answered with HTTP 204.
- The pinned `i4h-workflows` v0.5.0 source at revision
  `fb7727ef12e980022997fccb6cbca5621e4616e4` exposed robotic-surgery, robotic-ultrasound, SO-ARM,
  and telesurgery connectors. NVIDIA mode metadata was parsed directly rather than copied into Dr.Anmar.
- HoloHub CLI was pinned to the release-compatible revision
  `5c49897bd229d4ce46cbcd4a68c640f6258233f7`. This fixes upstream main-branch drift that removed
  `utilities/cli/holohub.py` after the i4h v0.5.0 release.
- The web runner rejected privileged Clarius hardware access and rejected container launch before pausing
  the operating room when Docker was absent. A pre-fix failure-path exercise also proved that job logs and
  manifests persist and the prior Dr.Anmar lesson automatically resumes after an official workflow exits.
- A study manifest was created and exported through the live hub.
- A 5-second multimodal recording saved 169 control/state frames and 22 vision frames. It contained left
  and right RGB, wrist RGB, finite metric depth, semantic IDs, fixed-grid camera-frame point clouds, joint
  state, applied/computed joint torque, world-frame robot/object/anatomy pose, simulator outcome, gaze/input
  provenance, and procedure phase/event codes.
- Stereo-left and stereo-right frames were numerically distinct; the wrist view was substantially distinct;
  all sampled depth and point-cloud values were valid in this capture; all recorded torque values were finite.
- Procedure events were preserved both as per-frame numeric codes and as two human-readable manifest entries.
- Isaac Lab 2.3 camera metadata changed from a dictionary to a per-environment list after rendering. The live
  capture exposed this compatibility fault; metadata normalization was added and the complete recording then
  saved successfully.
- All seven installed anatomy packages were rebuilt as separate metre-scale, Z-up OpenUSD compositions. Their
  room, ceiling, table, anatomy, and camera layers were opened in one Isaac audit pass: 14 stages, zero unresolved
  asset paths, and 14 authored cameras. The live needle-pickup worker then started with the sanitized CT anatomy
  and its matching repaired operating-room layer.
- The default liver context was moved out of the needle spawn volume, given an enabled convex collision mesh,
  and the wrist camera was changed to a live tool-following oblique view. A controlled browser-API attempt moved
  the PSM to 11.7 mm from the needle, closed the jaws, activated the limited grasp joint, and lifted the needle
  44.7 mm; opening and resetting removed the joint and restored the scene.
- The operating room gained zero-extra-sensor Operative, Close, and Overview stereo presets, a view-centred
  reticle, live target-direction guidance, gamepad camera/gripper bindings, and a strictly visual OpenUSD surgical
  drape. A browser pass switched presets, kept the stream live, exposed the active grasp state, and produced no
  application console warnings or errors.
- The jaw-capture zone was tightened to 18 mm and manual translation now feathers near the target. A closed-loop
  live attempt followed the new offset guidance, captured the needle at 13.4 mm, and lifted it 37.5 mm. The
  OpenUSD environment still opened as a metre-scale Z-up stage with 50 meshes and zero unresolved dependencies.
- Visible collidable anatomy now supplies a mesh-sampled control-space safety surface with a bounds fallback. The guard
  removes only the inward command component so withdrawal and tangential motion remain available; its activation
  state and anatomy clearance are exposed in the live API and clinician overlay.
- The anatomy guard now samples the actual visible OpenUSD mesh instead of relying only on its bounding volume.
  A live needle-pickup pass derived the official needle endpoints from mesh vertices, entered the rigid tissue
  proxy to 2.0 mm while the instrument tip remained about 2.3 mm outside, disabled the organ collider only for
  the latched needle-entry interval, and restored the protected shaft boundary. Operative and wrist streams both
  showed the needle occluded by the organ surface. Entry depth, tip clearance, and puncture state are recorded
  with each demonstration frame. The 12 mm cap remains an engineering rehearsal limit, not a clinical threshold.

Known host/runtime blockers from the same pass:

- Docker Engine and the NVIDIA container runtime are not installed for the `numi` account, and that account
  has no passwordless administrative access. Official i4h containers therefore remain blocked before launch.
- No RTI Connext DDS license is configured. Ultrasound, DDS policy pipelines, telesurgery, and related
  distributed modes must remain disabled until a valid license file is supplied.
- Co-resident Jetbot vision workloads used about 13.8 GB of GPU memory. Dr.Anmar rendered at roughly 2 FPS
  while those processes remained untouched. Sensor-throughput and long-recording performance results from
  this pass are not representative of an isolated 4090.
- Physical hardware, haptic, XR, external eye-tracker, Clarius, RealSense, and real-robot modes were not run.

## P0 — validate before calling this release runtime-ready

- Start the complete suite with Isaac Sim 5.1 and Isaac Lab 2.3.2 on the validated Linux/NVIDIA host.
- Confirm `dr_anmar_workstation.py` imports and starts for the default needle-lift relative-IK task.
- Exercise every new API through the hub and worker:
  - list and apply failure scenarios;
  - change manual and guided modes;
  - start selected demonstration replay;
  - take control during replay;
  - list demonstration analysis;
  - list persisted experiment manifests.
- Confirm the Doctor Studio loads Skills Twin and Failure Lab without browser-console errors.
- Confirm applying a challenge resets the environment once and preserves the selected task.
- Confirm the shifted-camera pose is valid for the default room and does not place the camera inside geometry.
- Confirm low-light, glare, partial-occlusion, and combined visual transforms update the streamed frame.
- Confirm manual movement interrupts supervised replay immediately and increments intervention count once.
- Confirm the explicit **Take control immediately** action stops replay and zeroes active drive commands.
- Record and stop a new demonstration; confirm both `.npz` and v2 `.json` manifest are written.
- Confirm the hub can read the worker's enriched demonstration list and selected analysis.
- Confirm newly recorded v2 demonstrations contain synchronized 360 × 240 endoscopic RGB at the declared
  5 Hz sampling rate and that timestamps align with the 50 Hz robot-state trajectory.
- Confirm depth is stored in metres as finite 360 × 240 float arrays, semantic IDs remain uint32 after nearest-
  neighbour resizing, the semantic label map is serializable, and camera intrinsics match the rendered sensor.
- Confirm the two endoscope cameras share intrinsics, use the intended 6 mm baseline, remain time-synchronized,
  and have correct left/right extrinsics after baseline and shifted-camera resets.
- Confirm each dynamically tool-following wrist camera resolves the actual `psm_tool_tip_link`, `endo360_needle`,
  or `ecm_end_link`, has the expected optical convention, clears surrounding geometry, and renders for single
  and dual robots.
- Independently unproject sampled depth and confirm every fixed-grid point-cloud XYZ is in metres in the declared
  left-camera optical frame, including invalid-depth encoding.
- Confirm applied/computed joint-torque arrays exist, use documented simulator units, align with joint ordering,
  and are never described as a wrist force-torque sensor.
- Confirm anatomy pose is the actual spawned prim transform for every anatomy preset; the initial implementation
  records the configured showcase transform and must not be treated as patient registration.
- Confirm simulator reward, termination, truncation, success, and contact-force tensors are converted without
  blocking GPU execution or introducing device synchronization stalls.
- Confirm deformable-object nodal displacement, deformation-gradient, and stress tensors exist for each tissue
  task, use the expected frames/units, and do not stall the GPU at 50 Hz recording cadence.
- Select a clinician reference, compare another attempt, and verify normalized action interpolation for
  demonstrations with different durations and action dimensions.
- Promote a clinician reference to the world-space path guide; verify the moving-body heuristic selects the
  actual tool tip, the points register after reset, remain visible from the endoscope, and hide immediately.
- Confirm the lateral and depth target scenarios call `write_root_pose_to_sim` and zero root velocity after the
  seeded environment reset without moving unrelated rigid objects.
- Confirm calibration bias rotates/scales both manual commands and automated replay exactly once.
- Confirm the multi-organ anatomy context reveals intended organ prims without exposing material/helper prims
  or changing collision behavior unexpectedly.
- Review every repaired room composition in a colour-managed Isaac renderer. Geometry and dependency integrity
  are validated, but the replacement PreviewSurface room finishes and illumination are not clinically reviewed.
- Confirm the Y-up centimetre source-room conversion and -0.95 m floor registration align the walls, ceiling,
  upstream ORBIT-Surgical table, each robot base, and all endoscope views across all seven anatomy variants.
- Confirm the upstream ORBIT-Surgical table/object collision and the single enabled Dr.Anmar anatomy collider do
  not introduce duplicate bodies, solver instability, or task-dependent contact artifacts.
- Drive every supported tool into every enabled organ from the top and sides; confirm the OpenUSD-derived virtual
  fixture activates before visible instrument penetration, preserves tangent/withdrawal motion, and does not block intended
  needle, cutting-path, retraction, or handover work. The current sampled surface is a usability boundary, not tissue
  deformation or a clinically validated forbidden-region model.
- Calibrate the liver, gallbladder, and bladder compliant-surface parameters against tissue-specific measurements,
  replace the rigid gross-motion core where a validated FEM/MPM body is available, and independently validate
  contact/strain telemetry before quantitative tissue-handling studies.
- Exercise complete suturing and incision trajectories with clinicians; validate thread material, puncture force,
  knot security, tissue tearing, cut width/depth, topology quality, and performance before making biomechanical claims.
- Run a small two-scenario, two-seed challenge matrix and confirm every rollout resets, replays, records,
  analyzes, and updates the durable matrix manifest before the next rollout begins.
- Confirm challenge summary means, 95% normal intervals, native-success rate, intervention rate, safety-event
  rate, and per-scenario groups exactly match an independent calculation. Treat intervals as descriptive only.
- Interrupt an automated matrix rollout with **Take control immediately** and confirm the matrix records the
  intervention and continues according to the intended study protocol.
- Start one bounded training recipe and confirm its experiment manifest advances from `starting` to `running`
  and then `complete` or `failed`.
- Freeze a dataset card from multiple demonstrations; independently recompute every SHA-256, verify duplicate
  content produces the same content-addressed ID, and confirm the exported JSON survives a hub restart.

## P1 — validate telemetry and coaching correctness

- Confirm `psm_tool_tip_link` and `endo360_needle` indices match the rendered end effector for every supported
  dVRK and STAR task; legacy demonstrations still use the maximum-moving-body fallback.
- Check tool-path units and magnitude against an independently calculated trajectory.
- Calibrate normalized action-similarity against clinician judgment; it is not yet a measure of surgical skill
  or procedural equivalence.
- Verify object-position keys for needle, block, and handover tasks.
- Calibrate the current 8 mm lift-evidence threshold separately for each task and object.
- Verify gripper-close event detection for single- and dual-arm action layouts.
- Validate phase-event ordering for demonstrations containing re-grasps, aborted attempts, or multiple lifts.
- Validate the relative tool-object drift proxy against explicit simulated grasp/contact state before calling it
  grasp slip; rotation about a stable grasp can currently contribute to the distance signal.
- Calibrate direction-correction, idle-time, recovery-hold, and smoothness proxies on a controlled set of
  novice and expert demonstrations.
- Confirm legacy v1 demonstrations remain downloadable and degrade gracefully when no analysis exists.
- Confirm very short, empty, corrupted, and partially written demonstrations do not break the UI.
- Add explicit task-success signals from the Isaac environment rather than treating object displacement as
  the final success measure.
- Replace maximum-moving-body path selection with task-specific tool-tip body registration.
- Validate contact force, deformation-gradient, stress, camera visibility, and grasp-slip metrics against
  independent ground truth for every supported task.
- Replace the current engineering advisories (2 N contact, 15 mm displacement, 0.50 deformation proxy) with
  task-specific, clinician- and biomechanical-engineer-reviewed research limits; they are not clinical limits.

## P1 — validate reproducibility and performance

- Confirm the recorded scenario seed deterministically reproduces the environment reset, native target jitter,
  camera pose, anatomy visibility, and every future randomizer on the supported Isaac Lab release.
- Measure storage growth and save latency from synchronized endoscopic RGB arrays across long demonstrations
  and large automated matrices.
- Decide whether RGB observations belong in the main `.npz`, a chunked Zarr/HDF5 dataset, or an external
  image/video stream before collecting large research datasets.
- Confirm dataset-card hashing and UI enumeration remain responsive for 100 large demonstrations; hashing is
  currently synchronous and should move to a background job if it delays the doctor-facing request.
- Measure render, GPU-memory, storage, and save-latency impact of stereo plus one or two wrist cameras. Add a
  doctor-selectable sensor budget if simultaneous cameras reduce the required control rate.
- Confirm procedure phase/event pulses are aligned to the correct trajectory frame and are not lost when events
  occur between sampling intervals.
- Validate external eye-tracker and XR gaze timestamps, coordinate conventions, calibration drift, and dropout.
  The browser pointer channel is an attention proxy and must never be reported as measured eye gaze.
- Verify input-source codes distinguish keyboard/pointer, gamepad, dVRK MTM, XR, and haptic devices throughout
  recording, replay, handover, and export.
- Recheck mode discovery when changing the pinned i4h or HoloHub revisions; metadata drift must fail closed
  rather than silently enabling an unknown hardware or custom-argument mode.
- After Docker/NVIDIA Container Toolkit and RTI licensing are configured, launch the official robotic-ultrasound
  workflow through the guarded runner; validate B-mode frames, probe pose, acoustic parameters, timestamps,
  and the Holoscan path before enabling live ultrasound controls.
- Validate optional XR and haptic adapters with end-to-end latency, packet loss, force scaling, saturation,
  emergency stop, human handover, RTI DDS licensing, and hardware-in-the-loop isolation.
- Define consent, privacy, retention, pseudonymization, and access controls before collecting identifiable gaze,
  operator video, voice, or real clinician performance data.
- Confirm live that a hub restart accurately marks an in-progress process-local matrix as interrupted and
  prevents its stale manifest from being mistaken for a completed result; source reconciliation is implemented.
- Record package versions, simulator build, GPU, task configuration, and policy checkpoint hashes in manifests.
- Measure CPU and GPU impact of Pillow-based stream perturbations at interactive and idle frame rates.
- Confirm experiment and demonstration enumeration remains responsive with 1,000+ manifests.
- Validate the implemented startup lifecycle reconciliation for a training process killed before its monitor
  thread updates the manifest, including PID-reuse protection through command-identity checks.
- Confirm concurrent browser sessions cannot unintentionally change each other's scenario or autonomy mode.

## P2 — clinical and educational validation

- Convene a clinician panel to define the observable phases and acceptable recovery for needle lift.
- Replace generic coaching thresholds with clinician-authored, task-specific criteria.
- Run a novice/intermediate/expert study and test construct validity of each automated metric.
- Compare automated scores with blinded GEARS ratings; report agreement rather than calling the proxy GEARS.
- Measure whether Skills Twin feedback improves subsequent performance, not merely whether users like it.
- Test whether the Failure Lab improves detection of uncertainty and appropriate hand-back behavior.
- Evaluate usability, workload, accessibility, simulator sickness, and learning retention.
- Validate specialty-specific tasks independently; do not transfer needle-lift thresholds to other procedures.

## Planned next engineering slice

- Have the Gilgamesh administrator install Docker Engine plus NVIDIA Container Toolkit, then configure a valid
  RTI Connext DDS license and rerun the guarded ultrasound launch.
- Launch `teleop_with_ultrasound`, confirm the B-mode and visualization processes become ready, and embed their
  clinician-facing video/status surface in Dr.Anmar rather than exposing container logs as the primary view.
- Add simulator-native tissue-stiffness, latency, object-scale, grasp-friction, stereo-calibration, and sensor-
  dropout randomizers to the automated challenge matrix.
- Add visibility/occlusion metrics and independently validated task-specific grasp/contact state.
- Add policy evaluation cards linked to immutable datasets, checkpoints, runtime revisions, and challenge matrices.

## 2026-07-20 keyboard-control live evidence

Captured on Gilgamesh in the running Isaac workstation rather than from mocked UI state:

- The workstation self-audit reported `51/51` visible controls mapped to keyboard shortcuts.
- Quick-tap semantic actions were recorded as `keyboard_smart_action`, and command lifetime followed the live
  simulator rate instead of expiring before a 2 Hz physics step.
- In the single-PSM needle room, a held needle reached a measured 2.64 mm bounded entry, then withdrew until
  puncture was false and the tip had a 49.17 mm positive clearance; assisted grasp remained active throughout.
- In the dual-PSM room, the holder retained the extracted needle while the receiving open gripper approached
  to 21.51 mm. Capture stopped before the receiving grasp or release, as intended for the pre-handoff view.
- The resulting GIF holds control highlights for readable frames and includes close, overview, instrument-select,
  entry, reverse, and pre-handoff states.
- A second live dual-PSM capture completed the transfer rather than stopping at pre-handoff: instrument 2 first
  acquired and lifted the needle, instrument 1 closed at 1.91 mm while both assisted grasps were active, then
  instrument 2 opened. Final simulator state reported assisted grasp `[true, false]`, grippers `[closed, open]`,
  and the receiving instrument retained the needle while separating 51.16 mm from the released holder.
- The completed-transfer showcase is a 10.2-second, 1152×648 GIF assembled from live Isaac frames. After release,
  the receiver remains grasped and carries the needle toward the organ until the simulator reports the bounded
  entry state. It keeps active instrument, smart action, movement, grasp, release, and camera feedback visible
  instead of overlaying staged labels.

## 2026-07-20 suturing, cutting, and organ-mechanics evidence

- The suturing room launched with the official CT liver OpenUSD scene, a visible 48-node constrained strand, and
  compliant surface authoring active. Untouched strand tension remained at 0.0 N after changing integration from
  wall-clock rendering time to the simulator's 20 ms step. Entry/exit pins, tension resistance, tissue deformation,
  loop closure and the persistent cinch constraint are implemented; a complete clinician-driven knot remains open.
- The incision room reported `topology_ready: true`. Its startup round-trip changed a live liver point, removed and
  read back at least one OpenUSD face, then restored the original vertex and face arrays before frame 1. A complete
  tool-driven incision from corridor start to recovery remains open for visual and topology-quality review.
- Liver retraction, gallbladder repositioning, and dual-arm bladder handover each resolved the correct visible organ,
  retained the matching OpenUSD operating room, and independently passed live surface-authoring round trips.
- A real dual-PSM API run approached the bladder from 67.7 mm to 5.3 mm, activated assisted capture, created a
  3.5 mm local jaw indentation, moved the organ 117.8 mm, switched to elastic recovery on release, and restored
  displacement and surface revision to zero on Reset.
- A 65-frame demonstration saved suture tension/pins/knot state, incision activity/length/removed faces/topology
  revision, and compliant-surface displacement/recovery/revision in the NPZ. The v2 JSON sidecar preserved the
  mechanics modality, final mechanics state, and task-analysis metrics.

Still open: full clinician-driven entry-through-exit knot completion; continuous tool-driven cut and incision-edge
review; tissue-specific FEM/MPM material calibration; validated needle, thread, cutting and organ contact forces;
tearing, bleeding, cautery and healing models; and clinical construct/usability studies.

Still required before research claims expand: calibrate the interactive surface, thread and incision models against
independent measurements, calibrate camera projection and depth cues, verify collision/contact telemetry under every
anatomy scene, and run clinician usability studies. This remains simulation training, not validated biomechanics or
clinical use.

## 2026-07-20 procedure-room expansion implementation

- Added runnable room definitions and clinician curricula for vascular shunt insertion, single interrupted stitch,
  running suturing, intracorporeal knot tying, needle passing/regrasping, anastomosis with pressure/leak testing,
  clip-ligate-divide, bleeding control, tissue-plane dissection, ultrasound-guided access, biopsy/lesion excision,
  and complication recovery. Existing anatomy navigation and organ-manipulation rooms remain available.
- Added reusable procedure mechanics for flexible-tube alignment/depth/buckling/wall load/patency; stitch count,
  spacing, closure gap, lumen narrowing and leak rate; clips, division and residual flow; suction/compression,
  blood-loss and rebleed; ultrasound target confidence, needle visibility and protected clearance; topology-changing
  dissection/excision; and randomized recovery progress.
- Added OpenUSD training geometry for the shunt/vessel, target vessels, visible clips, bleed source, anastomosis
  lumen ends, ultrasound target/protected vessel, and lesion target. Procedure state is displayed beside the live
  view and recorded as synchronized NPZ channels plus final-mechanics manifest state.
- Added task-specific research scores for shunt placement, closure/anastomosis, vascular control, hemostasis,
  ultrasound access, dissection/excision and complication recovery. These are explicitly engineering proxies.

Deferred validation gates for the expanded rooms:

- Drive every new room from entry through recovery on Gilgamesh and visually confirm procedure geometry, camera
  framing, ordered targets, dual-arm selection, reset behavior, overlays and recorded telemetry.
- Calibrate the flexible shunt against measured tube bending, friction, insertion force, buckling and flow/patency;
  validate different lumen diameters, curves, branching and pulsatile-flow cases.
- Calibrate needle penetration, multi-bite thread friction, tissue holding strength, closure gap, knot slippage,
  lumen constriction, pressure and leakage against independently measured phantoms.
- Validate clip deployment position, spacing, retention and burst pressure; replace residual-flow and bleeding
  proxies with a verified fluid/device model before any physiological claim.
- Validate dissection-plane separation, protected-structure collision, excision margins and specimen release against
  anatomy-specific task definitions; add thermal spread before enabling energy-device training claims.
- Launch and validate NVIDIA's official robotic-ultrasound containers after Docker, NVIDIA Container Toolkit and
  RTI licensing are present; compare the procedural B-mode proxy with official sensor output before combining data.
- Have specialty clinicians author acceptable phases, recovery actions, error taxonomies and task-specific scoring;
  then perform construct validity, inter-rater agreement, usability, workload and learning-transfer studies.

## 2026-07-20 procedure-mechanics polish

- Removed marker-only success shortcuts from shunt verification, knot tying, anastomosis pressure testing, clip
  placement, vessel division, hemostasis and ultrasound access. Progress now depends on the intended instrument
  interaction, stable dwell or topology change.
- Added PSM jaw control to the native dual-reach base used by vascular, dissection, biopsy and ultrasound rooms, so
  their close, compression and counter-traction actions are recorded in the same 14-channel action stream.
- Added stable shunt verification, off-target clip and protected-interval events, early-release rebleed, actual
  alternating two-arm knot throws, counter-traction-gated dissection progress, and protected-structure penalties.
- Converted ultrasound access to a bimanual room: Instrument 1 positions and stabilizes the probe while Instrument 2
  advances and withdraws the needle. Confidence now depends on probe pose and dwell; needle visibility, target
  contact, withdrawal and protected-vessel contacts are tracked separately.
- Added the new procedure events to demonstration files and task-specific scoring. They remain engineering proxies.

Focused gates still required after this polish:

- Clinician-check the spatial tolerance for clip deployment and division, the probe/needle role convention, knot
  crossing detection, dissection counter-traction radius, and each stability dwell on the live Gilgamesh controls.
- Confirm all added gripper action terms retain the intended PSM joint ordering after future Isaac Lab upgrades.
- Replace proximity and close-event procedure proxies with device-specific clip, suction, probe, scissors and
  dissector assets plus calibrated contact and material models before any claim of clinical fidelity.
