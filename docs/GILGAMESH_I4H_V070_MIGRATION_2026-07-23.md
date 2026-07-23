# Gilgamesh Isaac for Healthcare v0.7 upstream-first migration

Date: 2026-07-23

Host: `gilgamesh-System-Product-Name`

GPU: NVIDIA GeForce RTX 4090, 24 GB

Scope: simulation, education, synthetic data and preclinical research only

## Architecture

OpenUSD remains the scene and physical-asset authority. Isaac Lab and NVIDIA's native simulation backends
remain the authority for robot actions, contacts, constraints, sensors and stepping. Dr.Anmar provides the
clinician curriculum, procedure composition, recording, annotation, evaluation and lifecycle controls.
It does not replace an upstream action space, grasp controller, reward, state machine or physics backend.

## Immutable upstream pins

- Isaac for Healthcare workflows: `v0.7.0`
- Workflow commit: `9b526c6d107254727d3b113c612fb860fc65a5b2`
- Compatible HoloHub CLI commit: `f7e791dac061e01c560d3a2c5b7da82350915b69`
- Active provider link: `/home/numi/dr_anmar/vendor/i4h-workflows-current`
- Versioned v0.7 checkout: `/home/numi/dr_anmar/vendor/i4h-workflows-v0.7.0`
- Preserved v0.6 rollback checkout: `/home/numi/dr_anmar/vendor/i4h-workflows`

The installer verifies the release commit before executing or activating the checkout. Activation changes
only the provider symlink; it does not edit the preserved rollback checkout.

## Native surgical contracts

Dr.Anmar discovers these environments from NVIDIA's
`workflows/agentic/config/environments/<env>.yaml` files:

| Environment | Robot | Bridge port |
| --- | --- | ---: |
| `surgical_reach_psm` | PSM | 8870 |
| `surgical_reach_dual_psm` | dual PSM | 8871 |
| `surgical_reach_star` | STAR | 8872 |
| `surgical_lift_needle` | PSM | 8873 |
| `surgical_lift_needle_organs` | PSM | 8874 |
| `surgical_lift_block` | PSM | 8875 |

The environment YAML remains the source of truth for robot type, bridge port, task description, observation
and action schema. The guarded Dr.Anmar launcher invokes NVIDIA's own scripted expert entrypoint:

```bash
workflows/agentic/arena/run.sh \
  --env surgical_reach_psm \
  --state-machine \
  --episodes 1 \
  --num_envs 1 \
  --headless \
  --disable-cameras
```

## Installation and qualification log

- NVIDIA driver `580.159.03` detected.
- 1.2 TB free disk space detected.
- User-local `uv 0.11.31` installed because v0.7 Agentic setup requires `uv`.
- Dr.Anmar's setup wrapper confines `uv` caches and managed Python runtimes to the writable Dr.Anmar
  runtime root, then delegates directly to NVIDIA's upstream `workflows/agentic/setup.sh`.
- v0.7.0 cloned and verified at its immutable commit.
- HoloHub CLI cloned and verified at its immutable commit.
- Existing Dr.Anmar room on port 2396 remained running during source installation.
- Agentic Arena runtime and all six upstream surgical contracts are installed and adapter-ready.
- NVIDIA's Arena enumerated all six surgical environments plus its other registered environments.
- NVIDIA's policy router resolved all six surgical environments to its declared GR00T N1.5 stack.
- The `surgical_lift_needle` policy entrypoint passed NVIDIA's own dry-run and resolved its upstream
  `policy.surgical_baseline.infer.infer` module and shared environment configuration.
- All six surgical environments passed NVIDIA's own `--dry-run` registration check.
- `surgical_reach_psm` built its native scene, completed two PhysX steps and shut down cleanly.
- NVIDIA's scripted `surgical_reach_psm` expert completed 1/1 episode in 150 native steps with
  `success=True` and 0.0006 m final error.
- NVIDIA's scripted `surgical_lift_needle` expert completed its full 250-step phase sequence but failed:
  `success=False`, with `rise_m=-0.014`. Dr.Anmar does not replace that failed native grasp with an assisted
  attachment or custom success rule. Needle lift therefore remains unqualified on Gilgamesh.
- NVIDIA's full non-Cosmos Agentic setup completed. The common, Arena, GR00T N1.5/N1.6/N1.7,
  openpi π0, Mimic, dataset and annotator virtual environments are present.
- NVIDIA's end-to-end planner resolved `surgical_reach_psm` to GR00T N1.5 and produced the upstream
  record → Mimic → convert → summary plan without executing or downloading a model checkpoint.
- The existing Dr.Anmar room on port 2396 continued serving frames after both native qualification runs.

The remote evidence logs are content-addressed:

| Evidence | SHA-256 |
| --- | --- |
| `logs/i4h-v0.7.0-agentic-setup.log` | `3b1f69fb05f13fb70db1f1c78c879e9163ab59ae02c06738a496c62c246e3aa6` |
| `logs/i4h-v0.7.0-contract-qualification.log` | `29f42df0db8b544612ee2c8400c4d550a78a87fa2896a0d3f2568ffbc9b44f90` |
| `logs/i4h-v0.7.0-e2e-plan.log` | `ab3c79d655d2a39707e6a10fc6e313f538c0dd41a58a4db29becf6476ff74316` |
| `logs/i4h-v0.7.0-surgical-reach-psm-smoke.log` | `272d2a098333cd5d1255cc8c511079b69c543643110457f6e81238fcc6363474` |
| `logs/i4h-v0.7.0-surgical-reach-psm-state-machine.log` | `79b75a50f47f41bf6873d1dadd39146e3ce31cddcea303d1c0c806ab97a145b2` |
| `logs/i4h-v0.7.0-surgical-lift-needle-state-machine.log` | `8dcba1a660d90b80b2f6b360f186001af6a917a805c0afd3efbd69b3f5949eb2` |

## Native PSM control and dataset contract

The follow-up foundation qualification used
[`scripts/run_i4h_psm_foundation.sh`](../scripts/run_i4h_psm_foundation.sh) without modifying the pinned provider:

- NVIDIA's `DifferentialIKController` completed a 150-step `surgical_reach_psm` expert with
  `success=True`.
- The adapter recorded 150 seven-value PSM policy actions and 150 seven-value absolute target audits.
- Re-encoding the native joint targets through NVIDIA's joint-position scale/offset had a maximum error of
  `1.49e-08`, below the `2e-6` acceptance limit.
- NVIDIA's unmodified joint-position replay consumed all 150 frames.
- A camera-enabled repeat recorded 150 synchronized `480 × 640` room frames.
- NVIDIA's v0.7 dataset component converted it to one 150-frame LeRobot episode with seven action values,
  eight state values and H.264 room video.
- A separate two-step `surgical_lift_needle` smoke confirmed the full eight-value pose-and-gripper input
  becomes a seven-value PSM policy command with zero joint-target round-trip error. It was deliberately
  saved as an unsuccessful diagnostic and is not a qualified needle-lift demonstration.
- The pre-existing Dr.Anmar room on port 2396 remained reachable.

Evidence remains on Gilgamesh under `/home/numi/dr_anmar/validation/psm-foundation/`. Dr.Anmar now overlays the
missing `psm_singlecam` GR00T data configuration, a real PSM fine-tuning module, and a finite-action inference
daemon without modifying NVIDIA's checkout. The loader accepted the converted seven-action/eight-state dataset.
No checkpoint was trained from the diagnostic reach trajectory.

## Qualification boundary

Source discovery and immutable installation do not prove that an environment is runtime-qualified.
`surgical_reach_psm` is the clinician default because it passed native scene and state-machine execution.
The failed needle-lift result remains visible as upstream evidence and is not promoted as a validated expert
trajectory. The v0.7 scene-edit bridge is not yet the operating-room viewport backend. No trajectory is
accepted as a Behavior Cloning reference until it is recorded, replayed and reviewed.

Docker and an RTI DDS licence are not installed on Gilgamesh. This continues to block the official
containerized/DDS robotic-ultrasound modes, without blocking the normal Dr.Anmar operating room or the
Agentic surgical Arena setup.

No policy checkpoint was downloaded and no policy-training or demonstration-generation run was started in
this migration. Those are explicit research actions, not installation side effects.

The matching content-addressed NVIDIA asset provider was added in a follow-up without changing the native
Arena contracts. See [`I4H_ASSET_CATALOG_V070.md`](I4H_ASSET_CATALOG_V070.md).

## Rollback

To restore the preserved v0.6 provider without deleting v0.7:

```bash
ln -sfn /home/numi/dr_anmar/vendor/i4h-workflows \
  /home/numi/dr_anmar/vendor/i4h-workflows-current
```

Restart the Dr.Anmar hub after changing the provider link. The running room on port 2396 was launched before
this migration and is not mutated by the link change.
