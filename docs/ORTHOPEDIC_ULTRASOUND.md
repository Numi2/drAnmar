# Orthopedic robotic ultrasound

Dr.Anmar integrates [SonoGym](https://sonogym.github.io/) as a native provider for orthopedic ultrasound
research. SonoGym remains the authority for its CT-derived lumbar patient, robot assets, ultrasound generation,
task state, observations, rewards, safety cost, and Isaac Lab stepping. Dr.Anmar supplies the clinician-facing
course, browser transport, process lifecycle, provenance manifest, and later evaluation workflows.

## Included rooms

| Dr.Anmar room | Upstream Gym environment | Clinical learning intent |
| --- | --- | --- |
| L4 ultrasound navigation | `Isaac-robot-US-guidance-v0` | Find and hold the transverse plane through the centre of L4 |
| L4 ultrasound surface reconstruction | `Isaac-robot-US-reconstruction-v0` | Plan complementary sweeps and inspect simulated surface coverage |
| L4 ultrasound-guided orthopedic trajectory | `Isaac-robot-US-guided-surgery-v0` | Coordinate localization and a bounded orthopedic trajectory while monitoring the task safety cost |

## Runtime boundary

The browser sends a normalized action vector to `scripts/dr_anmar_sonogym_worker.py`. That process calls the
upstream Gym environment's `step()` method and returns its ultrasound or reconstruction observation as the live
view. The bridge does not change collision response, IK, target state, reward, termination, reconstruction,
ultrasound generation, or the guided-surgery safety constraint.

The task launches on worker port 2361, so Doctor Studio can use its existing operating-room frame. The hub
pauses the prior Dr.Anmar worker, records an experiment manifest, starts the pinned SonoGym task, and restores
the prior room when the task stops.

## Installation and provenance

Run:

```bash
./scripts/install_sonogym.sh
```

The installer keeps all large files under `DR_ANMAR_ROOT/sonogym` and pins:

- SonoGym source: `e67be58334d1a5274f0913af36f56e4b0b7ffe5a`
- SonoGym assets/models: `b37b080a8673f856266a2306724e48d5e034521a`
- Isaac Lab: `2.1.0`

The SonoGym source is MIT licensed. The downloaded asset/model dataset declares CC BY 4.0; preserve upstream
attribution in derived datasets, media, and redistributed asset bundles.

The optional LeRobot expert dataset is not downloaded because it is not required for manual practice or RL.
It can be added later for ACT or Diffusion Policy studies with a separate dataset provenance record.

## Current scientific boundary

SonoGym describes its current patient simulation as CT-derived 3D anatomy with model-based and learned
ultrasound. Its published future work includes better ultrasound quality and diversity, soft-tissue deformation,
larger patient populations, stronger cross-patient generalization, and clinical validation on real systems.
Dr.Anmar therefore presents these rooms as simulation and research exercises, not diagnostic training,
procedure planning, autonomous drilling, or evidence of clinical competence.

Sources: [project page](https://sonogym.github.io/), [source repository](https://github.com/SonoGym/SonoGym),
[paper](https://arxiv.org/abs/2507.01152).
