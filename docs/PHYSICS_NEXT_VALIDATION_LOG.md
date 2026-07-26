# Physics-next validation log

This is the evidence log for the experimental surgical-physics lane. It does
does not convert simulation results into real-world or clinical claims.

## 2026-07-21 — Gilgamesh baseline

- Stable Dr.Anmar remained live on ports 2360/2361 during installation and
  asset preparation.
- Target: RTX 4090, NVIDIA driver 580.159.03.
- Isolated runtime target: Isaac Sim 6.0.1, Isaac Lab 3.0.0 beta 2, the
  release's managed PyTorch 2.10 CUDA 12.8 stack, plus Newton VBD.
- CRESSim-MPM is pinned at v2.2.0 / revision
  `09aa5009b8580351f516b6df7660e87821fc5eb6` for the later topology-change
  adapter. Cloning it does not mean cutting or puncture is validated.
- CT liver surface extraction produced 7,174 vertices and 14,344 triangles,
  with zero boundary and non-manifold edges.
- The 8 mm candidate contains 33,274 vertices and 165,031 consistently
  oriented tetrahedra. Its render-to-simulation nearest-node error is 0.400 mm
  at p95 and 4.750 mm maximum.
- The candidate is still blocked on anatomically reviewed attachments,
  material calibration, solver benchmarks, deterministic replay and clinician
  review.
- The backend-neutral `UsdGeom.TetMesh` contains the same 33,274 vertices and
  165,031 tetrahedra, with a SHA-256 of
  `064a5fc0a57d7233317c7768c558fca529a7aa70605f05282546b1d4d087d589`.
- The full beta2 extras set currently has upstream dependency-metadata
  conflicts between Isaac Sim, Isaac Lab, visualizers and RL packages. Dr.Anmar
  records this instead of treating `pip check` as a solver gate; isolated module
  probes and actual backend trajectories are the runtime authority. The mesh
  tool remains in its own environment.

### Newton VBD coupon result

- Executed twice through the kitless Newton 1.2.1 VBD runtime on the RTX 4090,
  with 10 substeps, 8 iterations and one captured CUDA graph per 100 Hz frame.
- Both 600-step trajectories remained finite and produced the identical final
  state hash, establishing a deterministic replay RMSE of exactly 0 for this
  coupon and configuration.
- Physics step time was 15.198 ms p50 and 17.074 ms p95.
- Peak global tetrahedral-volume error was 4.787%; no tetrahedra inverted.
- A driven 8 mm spherical rigid-contact probe generated 287 positive contact
  samples, 0.0136 mm peak penetration, a 1.019 N peak net elastic normal
  reaction and 0.0766 N s normal-force integral. Tangential friction is not
  included in that scalar reaction diagnostic.
- Recovery residual was 1.950 micrometres after release.
- This run exposed and corrected a backend-adapter error: the canonical
  dimensionless damping seed had initially been passed as Newton `k_damp=120`.
  The explicit Newton adapter now uses an unvalidated `k_damp=0.01` seed; it
  must still be calibrated against physical tissue or phantom data.
- All five canonical Newton coupon engineering gates pass. This promotes only
  the coupon result inside the benchmark comparison; it does not activate
  Newton in the stable runtime or validate patient-tissue biomechanics.
- The result is a procedural 702-node solver coupon. It does not validate or
  promote the patient-specific liver TetMesh.

### Patient liver Newton integration smoke

- Loaded all 33,274 vertices and 165,031 tetrahedra from the authored 8 mm
  patient liver candidate into Newton VBD; graph coloring completed in 0.817 s.
- A 12-step, 2-iteration geometric retraction displaced the tissue by 2.087 mm.
  All states remained finite, the final global volume error was 0.080%, and no
  tetrahedra inverted.
- Median solve time was 1.591 ms; this short, low-iteration smoke is not a
  canonical runtime benchmark.
- The coordinate-selected fixture and pull bands are intentionally labeled
  non-anatomical. The result proves asset/solver interoperability, not material,
  attachment or clinical fidelity, and `promotion_allowed` remains false.

### PhysX FEM activation boundary

Isaac Sim 6 / PhysX requires explicit acceptance of NVIDIA's Omniverse Kit
EULA on this installation. The wrapper exits with code 3 and a readable action
instead of prompting or accepting legal terms automatically. The PhysX
benchmark remains pending that explicit activation.

## Gates still required

- Run the same canonical coupon trajectory on PhysX FEM after explicit Kit
  EULA activation.
- Add the same rigid-tool fixture and force diagnostic to the PhysX coupon so
  the two backends can be compared directly.
- Author reviewed liver attachment and vascular regions instead of guessing
  them from coordinates.
- Calibrate indentation, relaxation, puncture, withdrawal, cutting and suture
  pullout curves against a documented phantom or ex-vivo dataset.
- Add two-way dVRK/tool coupling and quantify penetration, force and runtime.
- Integrate CRESSim-MPM behind the canonical state interface, then validate
  topology-changing cut and needle-tract behavior separately.
- Record a canonical patient-asset deterministic replay with reviewed boundary
  conditions, failure modes and clinician review.

All entries are simulation-training assets. Their repository and native-runtime
evidence does not establish real-world correlation or clinical evidence.

## 2026-07-26 — current locked laparotomy smoke

- Installed the isolated `core` profile from
  `config/physics-next-lock.json`: Isaac Sim 6.0.1.0, Isaac Lab revision
  `51104d55d46192f9c981f2b63007d5156e141cec`, Torch/Torchvision/Torchaudio
  2.11.0 CUDA 12.8, and CRESSim-MPM revision
  `09aa5009b8580351f516b6df7660e87821fc5eb6`.
- The pip runtime deliberately retains Isaac Sim's CUDA 13 libraries alongside
  Isaac Lab's CUDA 12.8 Torch libraries. Removing the CUDA 13 cuDNN package
  caused native startup to fail on `libcudnn.so.9`; restoring the upstream
  dual-library layout corrected startup.
- Dependency verification accepted exactly the six overrides declared by the
  pinned Isaac Lab `pyproject.toml` and rejected any missing or unexpected
  conflict. The content-addressed runtime receipt verifies the lock, package
  freeze, dependency report, and both source revisions.
- NVIDIA Asset Validator reported zero blockers and zero issues for the
  repaired rigid abdominal proxy under PhysicsRules and SimReadyAssetRules.
- The full dynamic-patient laparotomy scene activated ten explicit-TetMesh
  wound edges, completed 720 CUDA steps, and wrote a 900 by 900 camera capture.
  No CUDA or PhysX error was recorded in the run log.
- The 60 prepositioned wound-edge fixture bonds in that scene are a
  camera-presentation mechanism, not contact-qualified grasp evidence. This
  single smoke does not close repeatability, calibration, live topology
  change, physical incision, grasp, or clinical gates.

## 2026-07-24 — laparotomy sponge integration

- Added the independently authored Apache-2.0 asset at
  `Props/SurgicalCount/LaparotomySponge`, with coordinated unfolded
  surface-deformable and folded rigid representations.
- Corrected invalid quaternion syntax in all 32 loop-collider declarations and
  invalid one-line variant overrides before integration.
- Corrected semantic fallback behavior for Isaac Sim 5.1 when optional
  Replicator modules are absent.
- Replaced the old bend-stiffness override, which omitted thickness, with the
  selected runtime's thickness-aware derivation.
- Recomputed dry and wet effective densities from the exact connected
  0.210219025612 m² simulation surface so the integrated masses are 0.022 kg
  and 0.120 kg.
- Added distinct dry/wet rigid contact materials and extended body collision to
  full visual thickness while retaining a deliberate 1 mm lateral inset.
- Repository validation passes all 13 geometry, topology, collision, texture,
  reference, semantic, physics and licensing checks.
- On the RTX 4090 / driver 580.159.03, the folded dry and wet variants each
  completed 240 CUDA steps in Isaac Sim 5.1.0.0 / Isaac Lab 2.3.2 with finite
  state.
- After the operator explicitly accepted the installed Omniverse Kit licence,
  both dry and wet surface variants completed 240 CUDA steps in Isaac Sim
  6.0.1.0 / Isaac Lab 3.0.0 beta 2. Each run exposed all 1,027 nodes, applied
  the Omni Physics and PhysX surface-deformable schemas, kept self-collision
  enabled and completed with finite nodal state.

These values are active provisional research parameters. They are not
manufacturer measurements, clinical validation, or patient-care approval.

## 2026-07-24 — skin stapler integration

- Imported the user-authored Apache-2.0 v0.2.0 payload at
  `Props/SurgicalClosure/SkinStapler`: 36 geometry files, 14 GLB inspection
  exports, 11 textures, rigid and articulated staplers, a standalone formed
  staple, interaction frames and a physics profile.
- Corrected all 11 `UsdPrimvarReader_float2.inputs:varname` declarations from
  `token` to the shader-required `string` type. This changed no geometry,
  collision or physics values.
- All three integrated layers pass `usdcat`. Current `usdchecker` passes the
  default layers and the loaded/empty variants.
- Added I4H-compatible paths plus Isaac Lab 2.3.2 rigid and Isaac Lab 3.0
  articulated configuration factories, semantic labels, synchronized
  trigger/pusher control, simulated deployment bookkeeping and closure-task
  helpers.
- Added the loaded rigid representation as an optional, contact-instrumented
  prop in the main operating-room bench. It is disabled by default so the
  established bench composition remains unchanged until selected.
- Native Isaac Sim 5.1 and 6.0 CUDA qualification is still pending. Physical
  parameters remain provisional and unmeasured; tissue penetration, staple
  formation, closure strength, healing, sterility and clinical quality remain
  outside the model.
