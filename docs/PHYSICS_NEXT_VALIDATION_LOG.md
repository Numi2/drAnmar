# Physics-next validation log

This is the evidence log for the experimental surgical-physics lane. It does
not convert research simulation results into clinical claims.

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
  with 10 substeps, 10 iterations and one captured CUDA graph per 100 Hz frame.
- Both 600-step trajectories remained finite and produced the identical final
  state hash, establishing a deterministic replay RMSE of exactly 0 for this
  coupon and configuration.
- Physics step time was 17.732 ms p50 and 19.687 ms p95.
- Peak global tetrahedral-volume error was 4.916%; no tetrahedra inverted.
- Recovery residual was 0.461 micrometres after release.
- This run exposed and corrected a backend-adapter error: the canonical
  dimensionless damping seed had initially been passed as Newton `k_damp=120`.
  The explicit Newton adapter now uses an unvalidated `k_damp=0.01` seed; it
  must still be calibrated against physical tissue or phantom data.
- Four of five engineering gates pass. Rigid-tool contact penetration remains
  unmeasured, so `promotion_allowed` correctly remains false.
- The result is a procedural 702-node solver coupon. It does not validate or
  promote the patient-specific liver TetMesh.

### PhysX FEM activation boundary

Isaac Sim 6 / PhysX requires explicit acceptance of NVIDIA's Omniverse Kit
EULA on this installation. The wrapper exits with code 3 and a readable action
instead of prompting or accepting legal terms automatically. The PhysX
benchmark remains pending that explicit activation.

## Gates still required

- Run the same canonical coupon trajectory on PhysX FEM after explicit Kit
  EULA activation.
- Add rigid-tool contact to both solver coupons and measure penetration and
  force; Newton's finite-state, runtime, global-volume and deterministic-replay
  gates already pass.
- Author reviewed liver attachment and vascular regions instead of guessing
  them from coordinates.
- Calibrate indentation, relaxation, puncture, withdrawal, cutting and suture
  pullout curves against a documented phantom or ex-vivo dataset.
- Add two-way dVRK/tool coupling and quantify penetration, force and runtime.
- Integrate CRESSim-MPM behind the canonical state interface, then validate
  topology-changing cut and needle-tract behavior separately.
- Record patient-asset deterministic replay, failure modes and clinician review.

All entries remain simulation/research-only and not clinically validated.
