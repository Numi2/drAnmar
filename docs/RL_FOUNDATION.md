# Dr.Anmar robot-learning foundation

Dr.Anmar uses one PSM control and data path:

```text
doctor Cartesian command
→ NVIDIA DifferentialIKController
→ NVIDIA PSM joint target
→ canonical seven-value action per PSM
→ Isaac Lab HDF5
→ NVIDIA LeRobot converter
→ GR00T
```

There is no Dr.Anmar IK solver, robot controller, or second policy action space in this path. OpenUSD owns the
assets and scene; Isaac Lab owns the action manager, contacts, sensors, stepping and recording; NVIDIA's PSM
embodiment owns the robot and action configuration. Dr.Anmar supplies a thin clinician-input and dataset-contract
adapter.

## Native PSM control adapter

[`scripts/dr_anmar_psm_native_adapter.py`](../scripts/dr_anmar_psm_native_adapter.py) accepts either native
Cartesian command used by NVIDIA's PSM rooms:

```text
relative IK: Δx, Δy, Δz, Δroll, Δpitch, Δyaw, logical gripper

absolute IK:
x, y, z, quaternion w, x, y, z, logical gripper
```

`NativePsmControlAdapter.step()` validates that the active arm term is NVIDIA's
`DifferentialInverseKinematicsAction`, submits the command to `env.step()`, and returns:

- the original Cartesian command;
- the seven-value policy action;
- the six absolute arm-joint targets plus logical jaw aperture; and
- the untouched Isaac Lab step result.

The seven policy values are the six raw inputs accepted by NVIDIA's `JointPositionAction` plus the sign consumed
by its `BinaryJointAction`. The adapter derives them from the targets produced by NVIDIA IK by inverting the
PSM embodiment's own scale and default offset. It does not solve IK or set robot joints. A dual-PSM room
concatenates the same contract in scene order:

```text
left six joints, left gripper, right six joints, right gripper
```

## Recording contract

The guarded launcher [`scripts/run_i4h_psm_foundation.sh`](../scripts/run_i4h_psm_foundation.sh) delegates to the
pinned Isaac for Healthcare v0.7 Arena and installs a PSM-only recorder:

| HDF5 key | Single PSM | Dual PSM | Meaning |
| --- | ---: | ---: | --- |
| `actions` | `T × 7` | `T × 14` | Canonical policy action; directly replayable in NVIDIA joint-position mode |
| `processed_actions` | `T × 7` | `T × 14` | Same canonical action for NVIDIA dataset tooling |
| `obs/actions` | `T × 7` | `T × 14` | Same canonical action for the v0.7 LeRobot converter |
| `cartesian_actions` | `T × 7/8` | `T × 14/16` | Original doctor/state-machine Cartesian intent |
| `resolved_joint_targets` | `T × 7` | `T × 14` | Absolute IK joint targets plus logical jaw aperture |
| `obs/joint_pos` | `T × 8` | `T × 16` | Physical arm and two-jaw state |

The standard Isaac Lab replay path reads `actions`; the v0.7 generic LeRobot converter reads `obs/actions`.
Both therefore receive the same seven-value policy action. The Cartesian stream remains available for provenance
and controller analysis, but cannot be mistaken for policy joints.

Arena initially constructs its recorder before applying `--record-to`. The launcher gives that bootstrap recorder
an isolated writable directory, then NVIDIA's normal `setup_recording()` switches to the requested HDF5.

The browser workstation uses the same contract. A doctor still operates the relative Cartesian room, but every
recording now keeps both `cartesian_actions` and the post-IK native policy action. Successful recordings are
exported as `.hdf5` next to the human-readable manifest and `.npz`; failed task attempts remain available for
analysis but NVIDIA's converter correctly excludes them from expert training data.

## Run it

Record a camera-enabled native expert:

```bash
./scripts/run_i4h_psm_foundation.sh \
  --env surgical_reach_psm \
  --state-machine \
  --episodes 1 \
  --num_envs 1 \
  --headless \
  --enable_cameras \
  --rendering_mode performance \
  --record-to /path/to/psm-reach.hdf5
```

Replay through NVIDIA's joint-position environment:

```bash
$DR_ANMAR_I4H_ROOT/workflows/agentic/arena/run.sh \
  --env surgical_reach_psm \
  --replay /path/to/psm-reach.hdf5 \
  --episode-index 0 \
  --headless \
  --disable-cameras
```

Convert through NVIDIA's LeRobot component:

```bash
$DR_ANMAR_I4H_ROOT/workflows/agentic/dataset/run.sh \
  --env surgical_reach_psm \
  --hdf5-path /path/to/psm-reach.hdf5 \
  --repo-id local/dr-anmar-psm-reach \
  --overwrite
```

Validate, train, and serve the PSM GR00T policy overlay:

```bash
./scripts/run_i4h_psm_policy.sh validate-data \
  --dataset-path local/dr-anmar-psm-reach

./scripts/run_i4h_psm_policy.sh train \
  --dataset-path local/dr-anmar-psm-lift \
  --output-dir /path/to/checkpoints

./scripts/run_i4h_psm_policy.sh infer \
  --model-path /path/to/checkpoint
```

The overlay supplies the missing v0.7 surgical `train_module` and a non-placeholder inference daemon. It uses one
room camera, the eight physical PSM joint observations, and the seven-value action contract. It does not modify
the pinned NVIDIA checkout.

## Qualification boundary

The action/data path must pass all of these before a recording is accepted:

1. native IK rollout completes;
2. `actions`, `processed_actions` and `obs/actions` are finite `T × 7` or `T × 14` arrays;
3. each six-joint block reconstructs the native IK targets within `2e-6`;
4. every logical gripper channel is exactly `-1` or `+1`;
5. unmodified NVIDIA joint-position replay consumes every frame;
6. NVIDIA's converter writes the expected LeRobot action/state dimensions;
7. the GR00T PSM loader accepts the converted dataset before any GPU training begins.

This proves control/data compatibility, not clinical skill, reward quality, policy convergence or sim-to-real
validity. The upstream needle-lift expert is still inconsistent at physically grasping the thin curved needle,
and the handover MDP has not yet completed a physical five-phase handover rollout. Neither is called RL-ready
until repeated physical qualification passes.
