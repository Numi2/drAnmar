# Dr.Anmar validation backlog

This file contains unresolved gates only. Completed checks and dated execution
history belong in revision-bound evidence artifacts and Git history. The latest
repository-wide assessment is
[`COMPLETE_REPOSITORY_AND_ASSET_AUDIT_2026-07-26.md`](COMPLETE_REPOSITORY_AND_ASSET_AUDIT_2026-07-26.md).

Passing a software or simulator gate does not establish biomechanical accuracy,
clinical validity, medical-device status, or suitability for patient care.

## P0 — restore release integrity

A public or physics-qualified release remains blocked until every item in this
section passes from a clean clone.

### OpenUSD and catalog

- Run the Isaac Sim Asset Validator on the supported release stack.
- Resolve the 14 missing material dependencies or declare, pin, install, and
  verify each external resolver dependency.

Exit gate: every cataloged USD layer opens and composes on the release
OpenUSD/Isaac stack, and the repository-local dependency walk is complete.

### Continuous integration

- Check out the asset submodule recursively and assert its expected revision.
- Run the complete non-Isaac Python suite, browser tests, source compilation,
  shell syntax, catalog verification, public-release check, and native OpenUSD
  parse on clean runners.
- Separate fast source checks from Linux/NVIDIA native qualification jobs.
- Pin third-party GitHub Actions to immutable commit SHAs.

Exit gate: hosted CI reproduces the complete local catalog and all applicable
non-Isaac checks are green.

### Public-source portability

- Replace machine-specific paths and private host defaults with documented
  environment variables or portable identifiers.
- Encode `assets/dr_anmar` as canonical source content while continuing to
  reject downloaded models, datasets, logs, checkpoints, caches, and runtime
  state.
- Run the public-release check in CI and test it against both a clean checkout
  and a populated external runtime-data directory.

Exit gate: `python3 scripts/check_public_release.py` passes without local
exceptions or private-machine assumptions.

### Evidence consistency

- Regenerate Dynamic Abdominal Patient evidence and correct the profile's
  explicit TetMesh count to match the anatomy manifest.
- Add parent revision, asset-submodule revision, generator hashes, and input
  hashes to retained evidence.
- Compare regenerated deterministic reports byte-for-byte in CI.
- Derive duplicated anatomy and topology totals from their authoritative
  manifests.

Exit gate: source, profile, manifest, portfolio, lock, documentation, and
retained reports agree at the exact recorded revisions.

### Public claims

- Limit the generic seven-system matrix claim to what it measures: workcell
  selection, simulator stepping, visible camera output, absence of reported
  fatal errors, and clean shutdown.
- Do not use the generic matrix as evidence of complete controllers, contact
  behavior, deformable cooking, physical cutting, or calibrated tissue
  response.
- Map every stronger capability statement to an asset-specific,
  content-addressed artifact with explicit exclusions.

Exit gate: every product-facing claim resolves to a revision-bound assertion in
a machine-readable evidence artifact.

## P1 — secure and reproduce deployment

- Default the hub and workstation to loopback.
- Require explicit non-loopback opt-in, a non-empty access token, authenticated
  mutation control independent of the `Origin` header, TLS termination, and a
  host firewall for remote use.
- Verify every downloaded SuFIA archive with SHA-256 before extraction.
- Pin Isaac Lab, CRESSim-MPM, and other Git dependencies to full commits.
- Lock Python dependencies and retain an installation receipt containing
  resolved packages, source revisions, simulator build, driver, and GPU stack.
- Add one documented, locked bootstrap command for the complete non-Isaac
  development and test environment.
- Scope lint and type checks to Dr.Anmar-authored modules, then ratchet the
  accepted baseline instead of claiming the inherited repository is globally
  clean.

Exit gate: a fresh supported host reproduces the recorded environment from
immutable inputs and exposes no unauthenticated mutation surface.

## P1 — obtain asset-specific native evidence

- Execute each workcell's complete procedure controller and intended
  deformable/contact route on the exact supported simulator stack.
- Record parent and submodule revisions, asset hashes, controller phases,
  physics configuration, repeated-run criteria, engine diagnostics, and raw
  measurements.
- Qualify Dynamic Abdominal Patient volume and surface deformables separately
  from structural TetMesh validity.
- Record Autonomous Rescue OR multi-arm/deformable execution and verify that
  post-physics contact state remains authoritative for patient effects.
- Exercise complete suturing, incision, retraction, handover, hemostasis,
  anastomosis, seal/divide, dissection, perfusion, and oncology workflows before
  promoting procedure-complete claims.
- Keep generated previews and provider visual predictions non-authoritative for
  contact, safety, scoring, or patient effects.

Exit gate: each promoted workcell has its own reproducible native artifact;
generic loading or composition smoke tests are not substituted.

## P1 — validate recording, telemetry, and reproducibility

- Verify RGB, depth, semantic IDs, camera intrinsics/extrinsics, timestamps,
  joint ordering, torque units, tool-tip registration, and anatomy transforms
  against independent references.
- Validate contact force, deformation, stress, slip, visibility, procedure
  events, and task-success channels against simulator ground truth.
- Replace maximum-moving-body and object-displacement fallbacks with
  task-specific tool-tip and success registrations.
- Prove seeded resets reproduce anatomy, targets, cameras, physics
  randomization, and future perturbations on the supported runtime.
- Independently recompute challenge-matrix statistics and dataset-card hashes;
  verify interruption, takeover, restart reconciliation, and content-addressed
  identity.
- Measure recording latency, storage growth, GPU/CPU cost, and manifest
  enumeration at research-dataset scale before choosing the long-term RGB-D
  container format.
- Validate concurrent-session isolation so one browser cannot change another
  operator's scenario or autonomy state.
- Define consent, privacy, retention, pseudonymization, and access controls
  before recording identifiable video, voice, gaze, or clinician-performance
  data.

Exit gate: retained datasets are synchronized, reproducible, bounded in cost,
and traceable to the exact simulator and operator state.

## P2 — qualify physical-correlation claims

- Collect instrumented reference data for the exact tissue or phantom,
  grasper, needle, thread, cutter, clip, stapler, seal device, and fluid setup
  named by each claim.
- Fit constitutive, contact, friction, puncture, cutting, attachment, pullout,
  tearing, seal, leakage, and failure parameters to those measurements.
- Record nodal displacement, strain, energy, force, slip, attachment, topology,
  and failure telemetry across repeated trials.
- Establish mesh-convergence and time-step sensitivity.
- Predeclare tolerances and uncertainty before comparing simulation with bench
  measurements.
- Do not describe pre-segmented continuity release as physical incision
  propagation, particle carriers as validated CFD/FSI, or distributed
  attachments as calibrated wound-edge grasping.

Exit gate: narrow quantitative correlation claims are supported for specific
materials, instruments, states, and procedures. This does not establish
clinical validity.

## P2 — educational and clinical research

- Have specialty clinicians define observable phases, acceptable recovery,
  error taxonomies, and task-specific scoring thresholds.
- Test construct validity with novice, intermediate, and expert cohorts.
- Compare automated scores with blinded reference ratings and report agreement
  without renaming engineering proxies as validated clinical metrics.
- Measure learning transfer, usability, workload, accessibility, simulator
  sickness, and retention.
- Validate every specialty and procedure independently; do not transfer
  thresholds between tasks without evidence.
- Complete the applicable regulatory, ethics, privacy, and clinical-review
  process before any patient-care or medical-device claim.

Exit gate: educational or clinical claims are limited to the populations,
procedures, endpoints, and study designs that actually support them.
