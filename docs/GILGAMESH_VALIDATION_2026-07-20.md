# Gilgamesh runtime validation — 2026-07-20

This report records engineering evidence from the Dr.Anmar RTX 4090 deployment. It does not constitute
clinical, biomechanical, device, or training-effectiveness validation.

## Environment

- Host: Gilgamesh, Ubuntu 24.04.4, NVIDIA GeForce RTX 4090, driver 580.159.03.
- Runtime: Isaac Sim 5.1, Isaac Lab 2.3, CUDA 12.8, Torch 2.7.0.
- Dr.Anmar revision: `539df2008d159799321440d4654ab9f6f45bd8c8`.
- Doctor Studio: port 2360; Isaac worker: port 2361; sensor profile: `efficient`.

## Passed checks

### OpenUSD integrity and scale

- Seven dependency-clean runtime anatomy layers and seven complete room compositions opened through Isaac.
- All 14 stages reported Z-up, metre-scale metadata and zero unresolved dependencies.
- The audit found that the upstream anatomy layers mixed source scales despite declaring `metersPerUnit=1`.
  The versioned sanitizer now normalizes every asset against the correctly scaled CT reference envelope.
- Every runtime anatomy now has a 0.6855 m maximum extent. Every anatomy inside a composed surgical room has
  a 0.2399 m maximum extent after the intended 0.35 operating-field scale.
- Hard audit gates reject runtime anatomy above 0.70 m and composed anatomy above 0.25 m.

### Native task matrix

The following task families each completed 40 CUDA simulation steps. Every reported robot joint-position and
joint-velocity array remained finite; dual-arm tasks reported two robots.

- `Isaac-Reach-PSM-IK-Rel-v0`
- `Isaac-Reach-ECM-IK-Rel-v0`
- `Isaac-Reach-STAR-IK-Rel-v0`
- `Isaac-Reach-Dual-PSM-IK-Rel-v0`
- `Isaac-Reach-Dual-STAR-IK-Rel-v0`
- `Isaac-Lift-Block-PSM-IK-Rel-v0`
- `Isaac-Lift-Needle-PSM-IK-Rel-v0`
- `Isaac-Handover-Block-Dual-PSM-IK-Rel-v0`
- `Isaac-Handover-Needle-Dual-PSM-IK-Rel-v0`

### Live doctor workflow

- Doctor Studio listed all 19 procedure rooms and all seven installed anatomy presets.
- The default needle-pickup room loaded the repaired OpenUSD scene and a live endoscopic camera.
- Gripper close/open, bounded drive, explicit stop, and Close camera view were exercised through the live worker.
- A new demonstration was recorded and saved as
  `dr_anmar_lift_needle_psm_ik_rel_20260720_164313_401.npz`; the live analysis endpoint accepted it.
- Two operator identities exercised the shared-workstation lease: the owner received HTTP 200, the second
  operator received HTTP 423, and explicit release succeeded.
- The browser-rendered workstation reported `51/51 controls mapped to keyboard`; gripper state feedback changed
  from OPEN to CLOSED in the endoscopic HUD and control dock.

### Representative advanced rooms

- **Ultrasound-guided access:** dual dVRK PSM, normalized MAISI s0253 anatomy, live OpenUSD room, two grippers,
  and active procedural B-mode confidence, visibility, target-error, and protected-clearance telemetry.
- **Single interrupted stitch:** dual needle handover task with the visible thread model active.
- **Liver incision:** OpenUSD liver collision surface loaded; the reversible startup topology mutation/restoration
  check passed and `topology_ready=true`.
- **Liver retraction:** the organ proxy rendered, one collision mesh was enabled, and the bounded compliant OpenUSD
  surface reported `active=true` and `authoring_ready=true`.

## Performance boundary observed

Jetbot perception and vision services were intentionally left running. They occupied about 15.4 GB before
Dr.Anmar started; the live worker brought total GPU use to about 21.9/24.6 GB and rendered near 2 FPS. This proves
co-resident operation but is not an isolated performance result.

## Still open

- Run complete clinician-like trajectories, not only startup/state checks, in all 19 rooms and across all seven
  anatomy variants.
- Exercise actual puncture, stitch, knot, full cut, tissue recovery, shunt placement, bleeding control,
  dissection, biopsy, and complication-recovery completion criteria with independent result inspection.
- Run the stereo and research sensor profiles, long recordings, replay/intervention, challenge matrices,
  dataset cards, training, and policy-card promotion on an isolated GPU budget.
- Install Docker plus NVIDIA Container Toolkit and a valid RTI Connext DDS licence before validating NVIDIA's
  provider ultrasound workflow. The built-in deterministic procedural B-mode room is a research rehearsal,
  not a diagnostic ultrasound simulator.
- Obtain tissue, needle, suture, cutting, flow, force, hardware, clinician-construct, and learning-transfer
  evidence before making biomechanical, clinical, or educational-effectiveness claims.
