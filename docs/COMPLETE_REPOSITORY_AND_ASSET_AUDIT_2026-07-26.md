# Dr.Anmar complete repository and asset audit

Date: 26 July 2026

Audited parent revision: `d606aa60937ba9381941771792671a7ab61223b1`

Audited asset-submodule revision: `19c9e39c94e7c4f3e1a182881f6f976b49da0609`

Primary local checkout: the repository root

Native OpenUSD host: `numi`, NVIDIA RTX 4090

OpenUSD version used for the independent parse: `0.25.11`

## Executive verdict

Dr.Anmar is a substantial, unusually well-documented surgical-simulation research
platform. It has a coherent product boundary, a deterministic asset catalog,
strong solver-independent patient-state tests, extensive authored anatomy, and
much better evidence language than most early research repositories.

It is **not ready for a public release or a blanket “physics-qualified” claim at
the audited revisions**. Three release blockers are confirmed:

1. One catalog entrypoint, the dynamic abdominal patient's rigid proxy, fails a
   native OpenUSD parse.
2. The only GitHub Actions workflow is red because it does not check out the
   asset submodule.
3. The repository's own required public-release gate fails the current `main`.

Four additional high-priority risks need correction before the repository is
treated as a dependable multi-user research service:

- the generic seven-system native matrix is described more broadly than its
  assertions and explicit boundaries support;
- retained evidence and profile metadata have drifted from the current dynamic
  patient assets;
- the hub has a network-facing, authentication-optional default;
- two installation paths do not provide fully immutable, hash-verified
  dependency acquisition.

The current laparotomy is a credible **simulation-training representation**:
five bilateral, explicit TetMesh wound layers, a centered median opening,
distributed exposure-tool attachments, and an ordered incision-state
controller. It is not yet evidence of live incision propagation, calibrated
constitutive response, or calibrated grasper-to-wound-edge interaction.

No clinical, medical-device, patient-specific, or patient-care claim is
supported. The repository itself generally preserves this boundary correctly.

## Scope

The audit covered:

- the parent repository and the complete `dr-assets` submodule;
- all cataloged and repository-local USD entrypoints;
- GLB and image decoding;
- JSON and YAML structure;
- the Python and browser-control regression suites;
- the dynamic abdominal patient, laparotomy, physiology, fluids, and damage
  contracts;
- robot workcell evidence, portfolio metadata, and qualification language;
- the Autonomous Rescue OR contracts and qualification report;
- launchers, installers, release checks, CI, dependency pinning, and network
  defaults;
- a clean native OpenUSD parse and dependency walk on the configured Linux
  host; and
- repository scale, maintainability, and binary-asset growth.

This was an audit, not a remediation pass. Other than this report, no product
source, asset, evidence artifact, runtime state, or remote service was changed.

## Verification summary

| Gate | Result | Evidence |
| --- | --- | --- |
| Parent and submodule state | Pass | Both checkouts clean and revision-matched locally and on `numi` |
| Catalog verification | Pass locally | 28 asset units, 866 files, 138 entrypoints, 470,298,596 bytes, 0 warnings, 0 errors |
| Python tests | Pass | 226 passed; one NumPy deprecation warning |
| Browser-control tests | Pass | 14 passed |
| Python compilation | Pass | Repository Python sources compiled |
| Shell syntax | Pass | Launchers and shell scripts passed `bash -n` |
| Strict JSON | Pass with format note | 165 strict JSON documents parsed; 4 VS Code files are JSONC |
| YAML | Pass | 23 YAML documents parsed |
| GLB/image payloads | Pass | 272 GLBs and 132 asset-submodule images decoded with finite geometry/pixels; no load failures |
| Native OpenUSD layer parse | **Fail** | 177 of 178 USD/USDAs opened; one dynamic-patient rigid proxy failed |
| OpenUSD default prims | Pass for opened layers | All 177 opened layers declared a default prim |
| OpenUSD dependency closure | Conditional | 14 unresolved material paths across 11 source layers |
| Dynamic-patient source/physiology validator | Pass for executed scope | `passed=true`, `overall_qualified=false` |
| Public-release gate | **Fail** | Five reported violation classes |
| GitHub Actions | **Fail** | Current and seven preceding catalog runs failed |
| Git object integrity | Pass | No corrupt Git objects; only unreachable/dangling history objects |
| Broad lint/type exploration | Not a usable gate | Repository-wide Ruff is not green; Pyright reports 654 errors and 36 warnings, predominantly unscoped upstream/runtime API noise |

The native host was clean and synchronized when inspected. No Dr.Anmar hub or
worker process was left running.

## Findings

### F-01 — Critical: the dynamic-patient rigid proxy is invalid OpenUSD

Affected asset:

`source/extensions/orbit.surgical.assets/data/Props/Patients/DynamicAbdominalPatient/dranmar_dynamic_abdominal_patient_rigid_proxy.usda`

Native `Sdf.Layer.FindOrOpen` on OpenUSD 0.25.11 fails at line 53,358:

```text
Expected } at 'def Scope "PhysicsMaterials"'
in </DrAnmarDynamicAbdominalPatientRigidProxy>
```

This file is not an incidental debug export. It is:

- one of the three primary patient entrypoints;
- registered as the portfolio's rigid perception/planning proxy; and
- explicitly required by the dynamic-patient repository tests and validator.

The existing validator passes it because it checks balanced brace/bracket
counts, selected syntax patterns, default-prim text, and relative-file
existence. It explicitly does not run an OpenUSD parser. Balanced delimiter
counts are insufficient to establish valid USDA grammar.

Impact:

- any planner, perception lane, or composition that selects this proxy can fail
  before simulation initialization;
- catalog verification can report zero issues for a non-openable entrypoint;
- the repository does not meet its own portfolio rule that OpenUSD parsing is
  required before release.

Required correction:

1. Repair the generation source if it exists; otherwise add an authoritative
   generator or deterministic repair source instead of maintaining a
   hand-edited 4.5 MB USDA.
2. Regenerate the rigid proxy and its manifest/lock entries.
3. Add `Sdf.Layer.FindOrOpen` and `Usd.Stage.Open` checks for every catalog
   entrypoint.
4. Fail the release if a layer emits a parser error, has no default prim, or
   fails composition.
5. Run the NVIDIA Isaac Sim Asset Validator on the resulting stages in the
   supported runtime.

Removal condition: all 178 current USD/USDAs and every future catalog
entrypoint open and compose without parser errors on the release OpenUSD/Isaac
stack.

### F-02 — Critical: CI does not materialize the asset repository

The only workflow uses:

```yaml
- uses: actions/checkout@v4
```

but `source/extensions/orbit.surgical.assets` is a Git submodule. The workflow
does not set `submodules: recursive` and does not run an explicit submodule
checkout.

The current failing run is:

<https://github.com/Numi2/drAnmar/actions/runs/30182749224>

The clean runner saw only 5 asset units and 52 files, then reported 158 errors.
The same revision, with the submodule present, verifies as 28 units, 866 files,
and zero catalog issues. The current run and seven preceding runs all failed for
the same class of missing-content errors.

Impact:

- `main` has no trustworthy green continuous-integration signal;
- real asset regressions are buried inside setup failures;
- GitHub never executes the workflow's tests because catalog verification fails
  first; and
- the README's implication of automated repository verification is not true for
  the current hosted workflow.

Required correction:

1. Check out submodules recursively.
2. Assert the exact expected submodule revision before validation.
3. Run the full 226-test suite, browser tests, compile gate, shell gate,
   public-release gate, native OpenUSD parse, and catalog verification.
4. Separate fast source checks from Linux/Isaac native qualification jobs.
5. Pin third-party actions to immutable commit SHAs for the release workflow.

Removal condition: a clean GitHub runner reproduces the complete local catalog
and all non-Isaac gates pass.

### F-03 — High: the repository fails its own required public-release gate

`CONTRIBUTING.md` names `scripts/check_public_release.py` as a required check.
It fails the audited `main` for:

- machine-specific Linux home-directory paths in two documents and one
  benchmark JSON;
- a hard-coded private tailnet host in `dr_anmar_webcam.sh`; and
- committed `assets/`, which the gate classifies as a forbidden runtime
  directory.

The `assets/` finding is a policy conflict, not an instruction to delete the
folder: this repository intentionally catalogs `assets/dr_anmar` as source
content. The release gate must distinguish canonical source assets from
downloaded/runtime assets, or the canonical assets must move to a non-runtime
namespace.

Impact:

- the documented release process is currently impossible to satisfy;
- machine topology leaks into public artifacts; and
- the release gate is absent from CI, so the contradiction can persist.

Required correction:

- replace machine paths and the private IP with documented environment
  variables or sanitized placeholders;
- decide and encode the authoritative policy for `assets/dr_anmar`;
- add the gate to CI; and
- test both a clean source checkout and a populated runtime-data directory.

Removal condition: the public-release gate passes from a clean clone without
special local exceptions.

### F-04 — High: the seven-system native claim exceeds the generic matrix

The README says seven procedure-specific systems had native-simulator runs
recorded for exact revisions and stacks and that each provides a headless CUDA
native-simulator evidence program.

The referenced generic matrix does record a host, GPU, driver, and runtime
stack, but it does **not** record the parent or asset-submodule commit. Its
per-case assertions establish:

- process readiness;
- no reported fatal error;
- correct featured-system selection;
- a non-empty 960 × 640 JPEG; and
- clean worker shutdown.

The same artifact explicitly says the generic bench did not execute each
package's complete procedure controller or deformable-cooking route.

The portfolio reinforces that narrower boundary:

- wound preparation says a smoke was recorded but the current native matrix is
  not established;
- exposure, hemostasis, anastomosis, seal/divide, and safe-plane dissection say
  composition was observed but the current native matrix is not established;
- dynamic patient native CUDA execution is not recorded; and
- Autonomous Rescue OR v0.4.0 multi-arm/deformable execution is not recorded.

The perfusion system has a substantially stronger asset-specific evidence
artifact. The oncology asset also has bounded native evidence, with explicit
non-contact and no-contact-calibration limits. Those stronger artifacts should
be the model for each workcell.

Impact:

- readers can interpret “seven systems recorded” as controller and tissue-route
  execution when the matrix only proves loading, selection, stepping, and
  visibility;
- the phrase “exact revisions” is not supported by commit IDs in the matrix;
  and
- generic composition evidence can be mistaken for mechanism qualification.

Required correction:

- narrow the README to the exact generic matrix assertions; or
- produce one content-addressed evidence artifact per system, with parent and
  submodule SHAs, generator hashes, runtime versions, hardware, controller
  phases, contact/deformable measurements, engine diagnostics, repeated-run
  criteria, and explicit exclusions.

Removal condition: every product claim maps to a revision-bound assertion in
an asset-specific evidence artifact.

### F-05 — High: retained patient evidence and profile counts have drifted

Re-running the dynamic-patient validator produces a report that differs from
the committed benchmark artifact. The retained report records the primary
patient USDA as:

- 14,794 bytes;
- 70 brace pairs; and
- 34 flat quaternions.

The current source is:

- 15,610 bytes;
- 72 brace pairs; and
- 36 flat quaternions.

The executable profile also says there are 7 explicit TetMesh organ components,
while the current anatomy manifest and validator contain 9:

- 4,028 vertices; and
- 14,702 tetrahedra.

The separate laparotomy asset adds 10 bilateral wound-margin TetMesh bodies and
8,640 positive tetrahedra.

Impact:

- retained evidence is not a faithful record of the current source snapshot;
- human-facing counts can regress to the earlier seven-component release
  description; and
- no current cross-file invariant forces profile, manifest, report, lock, and
  source counts to agree.

Required correction:

1. Regenerate the validation report after every asset change.
2. Record parent/submodule revisions and input file hashes in the report.
3. Make CI compare regenerated output byte-for-byte with the retained artifact.
4. Derive summary counts from `anatomy_manifest.json`; do not duplicate them
   manually in the profile.

Removal condition: regenerated evidence is byte-identical and all profile,
manifest, portfolio, and documentation counts agree.

### F-06 — High: network-facing defaults are authentication-optional

Both `scripts/dr_anmar_hub.py` and
`scripts/dr_anmar_workstation.py` default `--host` to `0.0.0.0`.
`dr_anmar_suite.sh` starts the hub without overriding that host. The normal
worker wrapper is safer because `dr_anmar.sh` supplies `127.0.0.1`, but direct
workstation invocation remains network-facing.

When `DR_ANMAR_ACCESS_TOKEN` is empty,
`access_is_authorized()` returns `True`. The same-host Origin check and operator
lease are applied only when a mutating request supplies an Origin header.
Direct API clients can omit Origin, so neither control is an authorization
boundary.

`.env.example` says the token may be empty on a localhost-only setup, but the
default hub setup is not localhost-only. `SECURITY.md` is also stale: it says
the services provide no authentication even though optional token support now
exists.

Impact:

- another process or device on the reachable LAN/VPN can access state-changing
  research endpoints when the token is unset;
- the browser lease does not protect direct clients; and
- the documented deployment boundary does not match the default launch path.

Required correction:

- default both services to `127.0.0.1`;
- require an explicit non-loopback opt-in and a non-empty token for any
  non-loopback bind;
- enforce the operator/mutation policy independently of the Origin header;
- treat Origin as CSRF defense only;
- update `SECURITY.md` and `.env.example`; and
- require an authenticated TLS reverse proxy and host firewall for remote use.

Removal condition: an empty-token default launch is loopback-only, while a
non-loopback launch fails closed without authentication.

### F-07 — High: two dependency installers are not fully content-addressed

`scripts/install_sufia_assets.py` verifies each downloaded archive by byte
length only before extraction. File size is not an integrity or provenance
check.

`dr_anmar_physics_next.sh` pins the top-level Isaac Sim wheel version, but:

- installs Python dependencies without a lock or hashes;
- fetches and detaches the mutable
  `origin/release/3.0.0-beta2` branch rather than an asserted commit; and
- clones CRESSim-MPM from a tag without verifying the resulting commit.

Other repository installers already demonstrate the better pattern: immutable
commits, receipts, and content hashes.

Impact:

- an upstream replacement, mirror compromise, or dependency drift can change a
  supposedly reproducible runtime;
- archive corruption that preserves length is not detected; and
- evidence from one install cannot be reliably reconstructed later.

Required correction:

- add SHA-256 values for every SuFIA archive and verify before extraction;
- pin and assert full Git commits;
- lock Python dependencies and retain a signed/hashed installation receipt;
- write the resolved package list, GPU stack, and source SHAs to
  `runtime.json`; and
- fail rather than silently accepting a different dependency graph.

Removal condition: every downloaded or cloned input is verified against an
immutable identifier and the installed environment is reproducible from a
retained lock/receipt.

### F-08 — Medium: 14 material dependencies are not self-contained

The independent OpenUSD dependency walk found 14 unresolved material paths
across 11 source layers:

- old HTTP S3 material paths for the surgical block, surgical needle, and PSM;
- `OmniPBR.mdl` in the table, STAR, ECM, and PSM family; and
- the same transitive material dependency in the two composed NVIDIA
  needle/Dr.Anmar suture assets.

A fully configured Kit resolver may provide some of these MDLs. Plain OpenUSD
0.25.11 does not, and the old HTTP paths are not a durable offline dependency.
The current catalog policy permits HTTP, HTTPS, and Omniverse references and
skips resolution once a scheme is allowed. Binary USD files are also skipped by
the catalog's textual dependency scanner.

Impact:

- a source checkout is not a self-contained visual-material package;
- offline runs may render fallback materials or fail resolution; and
- catalog verification can be green without proving composed dependency
  closure.

Required correction:

- vendor redistribution-compatible local materials or replace them with local
  PreviewSurface equivalents;
- otherwise declare them as explicit provider dependencies and test the
  configured Kit resolver;
- run `UsdUtils.ComputeAllDependencies` plus the Isaac Sim Asset Validator in
  CI/native qualification; and
- reject obsolete HTTP references in release assets.

Removal condition: dependency closure is complete in the supported offline
package, or every external resolver dependency is pinned, licensed, installed,
and verified.

### F-09 — Medium: the development and quality environment is not reproducible

The root `pyproject.toml` contains build-system and tool configuration but no
`[project]` metadata or development dependency set. There is no root lockfile.
A fresh developer environment therefore cannot infer the dependencies needed
for the full test suite.

The full tests do pass when supplied explicitly with Python 3.10, pytest,
NumPy, h5py, trimesh, and Pillow. The hosted workflow installs only pytest and
runs three catalog files.

Broad Ruff and Pyright invocations are currently too noisy to serve as release
gates. Much of the output comes from inherited ORBIT/Isaac dynamic APIs and
missing runtime packages, but the absence of a scoped target means new
Dr.Anmar-authored defects cannot be distinguished from accepted upstream debt.

Required correction:

- add a locked developer/test environment;
- document one bootstrap command;
- scope lint/type gates to Dr.Anmar-authored modules first;
- explicitly exclude generated USD authors and external/inherited code where
  appropriate;
- ratchet the scoped baseline rather than claiming the full repository is type
  clean; and
- run the full non-Isaac suite in CI.

Removal condition: a clean clone can reproduce the full non-Isaac gates with
one documented, locked command.

### F-10 — Medium: the workstation is a 12,690-line runtime monolith

`scripts/dr_anmar_workstation.py` combines:

- application and API serving;
- a large embedded browser UI;
- robot/policy control;
- Isaac scene and asset composition;
- deformable/contact and patient-effect bridges;
- camera encoding and streaming;
- demonstration recording;
- Autonomous Rescue orchestration; and
- runtime status/evidence collection.

The file is not automatically defective, and the current tests exercise many
of its contracts. It nevertheless creates a large qualification and regression
blast radius: a UI, API, camera, asset, or rescue change can affect the same
process and module.

Required correction:

- extract transport/authentication, state models, camera/recording, asset
  adapters, patient-effect authority, and rescue orchestration behind narrow
  interfaces;
- keep one authoritative simulation clock and state transition path;
- add contract tests at each boundary; and
- preserve a thin executable composition root.

Removal condition: the workstation entrypoint becomes orchestration code while
domain services can be tested without importing the full Isaac/browser stack.

### F-11 — Medium: physical realism is structured but not qualified

The authored mechanics follow the relevant shape of NVIDIA's current
deformable guidance:

- volume deformables use explicit `UsdGeom.TetMesh` simulation state;
- surface routes use triangular `UsdGeom.Mesh` state;
- visual/collision representations are separated where supported;
- wound margins use distributed attachments; and
- the runtime avoids pretending that arbitrary fracture or live remeshing
  exists.

This is good architecture. It does not establish realistic force response.

The current repository explicitly records:

- provisional engineering material parameters;
- no fit to instrumented abdominal tissue;
- no fit to the Dr.Anmar grasper;
- no validated injury biomechanics;
- no validated fluid-structure interaction;
- no patient-specific calibration;
- dynamic-patient native volume deformables not qualified; and
- Autonomous Rescue v0.4.0 native multi-arm/deformable evidence not recorded.

The latest NVIDIA documentation also describes the deformable schema as
evolving and documents implementation limits such as collider and surface
kinematic restrictions. Relevant primary references:

- [Omni Physics Deformable Schema](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/deformables/omniphysics_deformable_schema.html)
- [Omni PhysX Deformable Bodies](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/109.0/dev_guide/deformables/deformable_bodies.html)
- [Isaac Sim Asset Validation](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/robot_setup/asset_validation.html)

To qualify volume deformation, incision behavior, and wound-edge grasping for a
specific training claim, Dr.Anmar needs:

1. revision-bound native execution of the exact USD and controller;
2. stable repeated contact trials, not only a rendered frame;
3. nodal displacement, strain, energy, contact-force, slip, attachment, and
   failure telemetry;
4. mesh-convergence and time-step sensitivity studies;
5. instrumented material and grasper/tissue reference data;
6. predeclared error tolerances and pass/fail thresholds; and
7. retained raw measurements, calibration method, and uncertainty.

Until those gates pass, the correct description is:

> A realistic-looking, mechanics-enabled, pre-segmented laparotomy
> simulation-training model with provisional parameters.

It is not yet a physically calibrated incision-propagation or wound-grasping
model.

### F-12 — Low: binary growth will make maintenance progressively expensive

The working checkout is approximately 1.2 GB. The asset submodule contributes
approximately 457 MB, repository Git metadata is approximately 375 MB, and
tracked screenshots contribute approximately 79 MB. Git LFS is not in use.

No individual tracked file exceeded the repository's 95 MB release threshold;
the largest observed asset was approximately 28.5 MB. This is not an immediate
hosting blocker, but generated GLBs, GIFs, and USD binaries will accumulate in
history.

Required correction:

- define which binary artifacts are canonical source, inspection output,
  evidence, or release payload;
- keep essential revision-bound evidence in Git;
- move reproducible heavy outputs to LFS or content-addressed releases; and
- add growth budgets per asset unit.

### F-13 — Low: one NumPy API is deprecated

The full test suite emits one warning because
`perfusion_viability_robot.py` selects `np.trapz`. NumPy recommends
`np.trapezoid`. This is not a current behavioral failure, but it should be
removed before a dependency update turns it into a break.

## What is strong now

The following work should be preserved:

1. **Evidence boundaries are explicit.** `docs/EVIDENCE_LEVELS.md` cleanly
   separates product capability, repository verification, native execution,
   real-world correlation, and clinical evidence.
2. **Clinical claims fail closed.** Portfolio entries and provider policy retain
   `clinical_validation: false`.
3. **The patient runtime is substantive.** Eight condition presets remained
   finite; blood volume is conserved; hemorrhage stops at available volume;
   hemostasis reduces modeled flow; and damage/intervention events persist.
4. **The laparotomy representation is anatomically coherent for its stated
   purpose.** It is centered, median, bilateral, full-thickness, camera-visible,
   and has no incorrect loose central tissue plug.
5. **The mechanics routes are honest.** Explicit TetMesh, surface, segmented,
   attachment, and reduced-order routes are separated rather than presented as
   one universal solver.
6. **The catalog is useful.** Local inventory, deterministic locking,
   ownership/provider separation, license fallback, and clinical boundaries are
   machine-readable.
7. **Visual payload integrity is excellent.** Every inspected GLB and image
   decoded, and tetrahedral checks found finite positive volumes.
8. **Autonomous Rescue has a sound authority model.** Policy intent is
   separated from post-physics, contact-owned patient effects, and the v0.4.0
   report correctly says native execution is not yet recorded.
9. **Fail-closed runtime patterns are present.** Requested deformable routes,
   resource preflight, evidence-free completion rejection, and patient-effect
   ownership are treated as safety contracts.

## Current qualification matrix

| Capability | Current status | Correct claim |
| --- | --- | --- |
| Local catalog structure and lock | Verified at audited revisions | Repository-verified locally |
| GLB/image readability | Verified | Inspection payloads decode |
| JSON/YAML/source compilation | Verified | Static repository checks pass |
| Dynamic-patient physiology invariants | Verified in solver-independent tests | Software-behavior evidence |
| Explicit tetrahedral positive volumes | Verified structurally | Mesh-structure evidence |
| Complete OpenUSD package | **Not verified** | One primary proxy is invalid |
| Generic seven-system bench composition | Recorded | Selection, stepping, visibility, clean shutdown |
| Seven complete procedure controllers | Not established by generic matrix | Requires asset-specific evidence |
| Dynamic-patient volume deformables | Not qualified | Native execution not recorded |
| Physical incision propagation | Not implemented/qualified | Ordered release of pre-authored continuity only |
| Wound-edge grasp force realism | Not qualified | Distributed attachment architecture with provisional values |
| PBD fluids as CFD/FSI | Not qualified | Particle carriers plus conservative reduced-order ledgers |
| Autonomous Rescue v0.4.0 native multi-arm/deformable runtime | Not recorded | Repository runtime implemented |
| Physical tissue correlation | Not established | Instrumented bench work required |
| Clinical validity or patient-care use | Not established | Research simulation only |

## Prioritized remediation plan

### Phase 0 — Restore release integrity

1. Fix and regenerate the dynamic-patient rigid proxy.
2. Add native OpenUSD layer/stage parsing for all entrypoints.
3. Fix GitHub submodule checkout.
4. Make the public-release gate pass and run it in CI.
5. Regenerate the dynamic-patient evidence and correct the nine-component
   TetMesh count.

Exit gate: clean clone, complete submodule, green non-Isaac CI, 178/178 current
USD layers open, retained reports reproduce exactly.

### Phase 1 — Align claims with evidence

1. Narrow the generic seven-system README wording.
2. Add parent/submodule SHAs and input hashes to every native artifact.
3. Build a per-workcell evidence matrix modeled after the stronger perfusion
   artifact.
4. Cross-check portfolio strings against retained evidence automatically.

Exit gate: every doctor-facing capability statement resolves to an exact
machine-readable artifact and explicit boundary.

### Phase 2 — Secure and reproduce deployment

1. Make loopback the default.
2. Require authentication for non-loopback binding and all mutating clients.
3. Hash every archive and pin every Git dependency to a full commit.
4. Add a locked development and runtime environment.
5. Resolve or explicitly provision the 14 material paths.

Exit gate: a fresh supported host can reproduce the runtime from immutable
inputs without exposing an unauthenticated mutation surface.

### Phase 3 — Qualify the intended surgical mechanics

1. Run native volume/surface deformable and attachment trials on the exact
   supported stack.
2. Record contact, strain, slip, attachment, incision-state, and failure
   telemetry.
3. Establish mesh/time-step convergence.
4. Collect instrumented tissue and grasper reference data.
5. Fit parameters, predeclare tolerances, repeat trials, and retain raw data.

Exit gate: narrow, quantitative physical-correlation claims can be made for
specific materials, instruments, states, and procedures. This still does not
establish clinical evidence.

### Phase 4 — Reduce regression cost

1. Decompose the workstation around one authoritative clock/state contract.
2. Introduce scoped lint/type gates.
3. Add binary growth budgets and artifact-retention rules.
4. Promote native tests to revision-bound scheduled/release jobs.

## Release decision

**Decision: hold public/qualified release at the audited revisions.**

The repository is suitable for continued internal simulation research and
engineering development if its existing non-clinical boundaries are retained.
It should not be labeled fully release-ready, physically qualified, or
clinically validated until Phase 0 is complete. Claims about realistic
deformation, incision, and grasping should remain explicitly experimental until
Phase 3 produces quantitative evidence.

## Reproduction commands

Representative commands used in this audit:

```bash
git status --short --branch
git submodule status
python3 scripts/dr_anmar_asset_registry.py verify
python3 scripts/check_public_release.py
python3 -m compileall -q scripts source examples tests
node --test tests/hand_control.test.mjs
python3 scripts/validate_dranmar_dynamic_abdominal_patient.py --output /tmp/dranmar-dynamic-audit.json
gh run view 30182749224 --repo Numi2/drAnmar --log-failed
```

The full Python suite was reproduced in an ephemeral Python 3.10 environment
with the explicit test dependencies because the repository has no root
development lock:

```bash
uv run --no-project --python 3.10 \
  --with pytest==8.3.5 \
  --with h5py \
  --with numpy \
  --with trimesh \
  --with pillow \
  pytest -q tests
```

Native OpenUSD verification used the Physics-next Python environment on `numi`
and applied `Sdf.Layer.FindOrOpen` plus
`UsdUtils.ComputeAllDependencies` to all 178 repository and submodule
USD/USDAs.
