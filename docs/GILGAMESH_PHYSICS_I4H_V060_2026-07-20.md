# Gilgamesh coupled-physics and Isaac for Healthcare v0.6 evidence

Date: 2026-07-20  
Host: `gilgamesh-System-Product-Name`  
GPU: NVIDIA GeForce RTX 4090, 24 GB  
Scope: simulation, education, synthetic data and preclinical research only

## Isaac for Healthcare upgrade

- Upgraded the installed `i4h-workflows` checkout from v0.5.0 to **v0.6.0** at
  `8b03d55ecb647a43af54470b27bd09a239870aaf`.
- Installed the v0.6-compatible HoloHub CLI revision
  `f7e791dac061e01c560d3a2c5b7da82350915b69`.
- The live `/api/healthcare-platform` response reported both pins ready.
- Direct upstream metadata discovery reported 8 robotic-surgery, 18 robotic-ultrasound and 15 SO-ARM modes.
- Upstream `./i4h list` and `./i4h modes robotic_surgery` executed successfully. Rheo and Agentic are
  present in v0.6 source but lack the metadata contract used by the guarded web launcher, so Dr.Anmar exposes
  them as expert-source capabilities rather than fabricating one-click launch commands.

Docker Engine and an RTI DDS licence are still absent on this host. This blocks provider-container and DDS
runtime validation, but it does not block the existing Dr.Anmar operating rooms.

## Coupled surgical mechanics

The implementation now provides:

- tissue-specific research material profiles and OpenUSD physics friction binding;
- mesh-edge-coupled deformation, approximate volume preservation, attachment resistance and recovery;
- force-gated needle puncture, hysteresis, drag, curvature alignment, work, force/torque telemetry and translational/rotational safety attenuation;
- suture slack, strain, tension, anchor damage, pullout, breakage, knot tightness and knot security;
- face removal plus visible incision opening, cut resistance and accumulated work;
- vessel compression, clip retention, over-compression damage, residual flow, bleeding and rebleeding;
- a coupled fallback-force vector recorded alongside native Isaac contact and articulation signals.

A deterministic mechanics exercise in Gilgamesh's Isaac Python environment passed deformation, volume,
topology removal, cut resistance, puncture transition, thread initialization and coupled-force invariants.

## Live operating-room evidence

- Restarted the suite on ports 2360/2361 with the upgraded code.
- The default needle room produced live frames and exposed the new needle and coupled-force schemas.
- Switched through the normal hub lifecycle to `suture-threading-path` using the existing doctor workflow.
- The dual-PSM room produced live frames with its intended OpenUSD anatomy and reported:
  - `reduced_order_volume_preserving_tissue_v2`;
  - `position_based_suture_with_anchor_failure_v2`;
  - liver research material profile;
  - authoring-ready mutable topology;
  - volume, strain, stress, thread, needle and closure fields;
  - explicit `research_defaults_unvalidated` calibration status.
- No Dr.Anmar exception or traceback was found in the worker startup log. The only matched message was an
  upstream PhysX benchmark-plugin dependency warning.

## Remaining evidence gates

This pass proves source integration, deterministic model behavior, OpenUSD authoring startup and live schema
availability. It does **not** prove biomechanical validity. Still required:

- complete puncture, exit, suturing, knot, cut, retraction and vascular trajectories in every intended room;
- bench calibration for each tissue, needle, suture, clip and instrument combination;
- comparison with native volumetric-deformable assets where available;
- force/torque sensor calibration and haptic-device validation;
- clinician construct-validity, workload and learning-transfer studies;
- Docker/NVIDIA Container Toolkit and licensed RTI DDS validation for official provider workflows.
