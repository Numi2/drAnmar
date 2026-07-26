# Dr.Anmar unresolved validation gates

This file contains unresolved gates only. Completed checks belong in
revision-bound evidence, the current release-readiness record, and Git history.
Passing a software or simulator gate does not establish biomechanical accuracy,
clinical validity, medical-device status, or suitability for patient care.

## P0 — native release-stack qualification

- Run NVIDIA Isaac Sim Asset Validator against every registered USD entrypoint
  on the exact supported Isaac Sim release.
- Execute the dynamic patient's explicit volume TetMeshes, surface
  deformables, contact sensors, wound-margin attachments, and respiratory
  targets in separate repeated native trials.
- Record finite nodal state, contact forces, attachment state, slip,
  deformation, strain/energy proxies, engine diagnostics, performance, reset
  repeatability, and failure behavior.
- Establish a stable volume-deformable configuration before enabling that
  route in the doctor-facing room. The recorded CUDA error 700 experiment
  remains a failed experiment, not evidence.
- Run the manual `NVIDIA native qualification` workflow on a locked
  `dranmar-nvidia` runner and retain its exact-stack artifact.

Exit gate: all registered stages pass the supported native validator and the
promoted patient mechanics have repeatable, revision-bound native artifacts.

## P1 — procedure-specific native evidence

- Execute each workcell's complete procedure controller and intended
  deformable/contact route on the exact supported simulator stack.
- Record parent and submodule revisions, asset hashes, controller phases,
  physics configuration, repeated-run criteria, raw measurements, and engine
  diagnostics.
- Record Autonomous Rescue OR multi-arm/deformable execution and verify that
  post-physics contact state remains authoritative for patient effects.
- Exercise complete suturing, incision-state release, retraction, handover,
  hemostasis, anastomosis, seal/divide, dissection, perfusion, and oncology
  workflows before promoting procedure-complete claims.
- Keep generated previews and generative video non-authoritative for contact,
  control, safety, scoring, patient effects, or task success.

Exit gate: each promoted workcell has its own reproducible native artifact;
generic loading, a rendered frame, or bench composition is not substituted.

## P1 — telemetry calibration and runtime reproducibility

- Compare RGB, metric depth, semantic IDs, intrinsics, world-frame extrinsics,
  timestamps, joint ordering/units, torque/force units, tool-tip registration,
  and anatomy transforms with independent native references.
- Validate contact force, deformation, stress, slip, visibility, procedure
  events, and task-success streams against simulator ground truth.
- Prove seeded resets reproduce anatomy, targets, cameras, physics
  randomization, and perturbations on the supported runtime.
- Independently recompute challenge statistics and dataset-card hashes; verify
  interruption, takeover, restart reconciliation, and content identity.
- Measure capture latency, storage growth, GPU/CPU cost, and manifest
  enumeration at research-dataset scale before selecting a long-term RGB-D
  container.
- Native-test the single-operator lease under concurrent browsers and direct
  clients, including expiry, reconnect, recording finalization, and emergency
  stop.
- Put identifiable video, voice, gaze, or clinician-performance recording
  behind an approved consent, privacy, retention, pseudonymization, access,
  deletion, and incident-response protocol.

Exit gate: retained datasets are synchronized, independently checked,
reproducible, bounded in cost, and traceable to exact simulator and operator
state.

## P1 — reduce native regression blast radius

- Extract camera/capture, simulator adapters, procedure orchestration, and
  patient-effect authority from `dr_anmar_workstation.py` behind narrow
  contracts while preserving one simulation clock and state-transition path.
- Add native integration tests at each extracted boundary.
- Retain the current scoped source lint/type baseline and ratchet new modules
  into it; do not claim inherited Isaac/ORBIT code is globally type-clean.
- Schedule or release-gate revision-bound NVIDIA jobs once a maintained native
  runner is available.

Exit gate: the workstation entrypoint is composition code and domain services
can be qualified independently.

## P2 — physical correlation

- Collect instrumented reference data for the exact tissue or phantom,
  grasper, needle, thread, cutter, clip, stapler, seal device, and fluid setup
  named by each intended claim.
- Fit constitutive, contact, friction, puncture, cutting, attachment, pullout,
  tearing, seal, leakage, and failure parameters to those measurements.
- Record nodal displacement, strain, energy, force, slip, attachment, topology,
  and failure telemetry across repeated trials.
- Establish mesh-convergence and time-step sensitivity.
- Predeclare tolerances and uncertainty before simulation-to-bench comparison.
- Do not describe pre-segmented continuity release as physical incision
  propagation, particle carriers as validated CFD/FSI, or distributed
  attachments as calibrated wound-edge grasping.

Exit gate: narrow quantitative correlation claims are supported for named
materials, instruments, states, and procedures. This does not establish
clinical validity.

## P2 — educational and clinical research

- Have specialty clinicians define observable phases, acceptable recovery,
  error taxonomies, and scoring thresholds.
- Test construct validity with novice, intermediate, and expert cohorts.
- Compare automated scores with blinded reference ratings and report
  agreement without renaming engineering proxies as clinical metrics.
- Measure learning transfer, usability, workload, accessibility, simulator
  sickness, and retention.
- Validate each specialty and procedure independently.
- Complete applicable regulatory, ethics, privacy, and clinical-review
  processes before any patient-care or medical-device claim.

Exit gate: educational or clinical claims are limited to the populations,
procedures, endpoints, and study designs that directly support them.
