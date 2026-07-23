# Dr.Anmar robot-learning foundation

Dr.Anmar uses one PSM control and data path:

```text
doctor Cartesian command
→ NVIDIA DifferentialIKController
→ NVIDIA PSM joint target
→ canonical seven-value PSM policy action
→ Isaac Lab HDF5
→ NVIDIA LeRobot converter
→ GR00T
```

There is no Dr.Anmar IK solver, robot controller, or second policy action space in this path. OpenUSD owns the
assets and scene; Isaac Lab owns the action manager, contacts, sensors, stepping and recording; NVIDIA's PSM
embodiment owns the robot and action configuration. Dr.Anmar supplies a thin clinician-input and dataset-contract
adapter.

## Native PSM control adapter

[`scripts/dr_anmar_psm_native_adapter.py`](../scripts/dr_anmar_psm_native_adapter.py) accepts the native absolute-IK
command used by NVIDIA's PSM rooms:

```text
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
PSM embodiment's own scale and default offset. It does not solve IK or set robot joints.

## Recording contract

The guarded launcher [`scripts/run_i4h_psm_foundation.sh`](../scripts/run_i4h_psm_foundation.sh) delegates to the
pinned Isaac for Healthcare v0.7 Arena and installs a PSM-only recorder:

| HDF5 key | Shape | Meaning |
| --- | ---: | --- |
| `actions` | `T × 7` | Canonical policy action; directly replayable in NVIDIA joint-position mode |
| `processed_actions` | `T × 7` | Same canonical action for NVIDIA dataset tooling |
| `obs/actions` | `T × 7` | Same canonical action for the current v0.7 LeRobot converter |
| `cartesian_actions` | `T × 7/8` | Original absolute pose, with gripper when the source room exposes one |
| `resolved_joint_targets` | `T × 7` | Six absolute joint targets plus logical physical jaw aperture |
| `obs/joint_pos` | `T × 8` | Six arm joints plus both physical jaw joints |

The standard Isaac Lab replay path reads `actions`; the v0.7 generic LeRobot converter reads `obs/actions`.
Both therefore receive the same seven-value policy action. The Cartesian stream remains available for provenance
and controller analysis, but cannot be mistaken for policy joints.

Arena initially constructs its recorder before applying `--record-to`. The launcher gives that bootstrap recorder
an isolated writable directory, then NVIDIA's normal `setup_recording()` switches to the requested HDF5.

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

## Qualification boundary

The action/data path must pass all of these before a recording is accepted:

1. native IK rollout completes;
2. `actions`, `processed_actions` and `obs/actions` are finite `T × 7` arrays;
3. the six encoded joint inputs reconstruct the native IK targets within `2e-6`;
4. the logical gripper is exactly `-1` or `+1`;
5. unmodified NVIDIA joint-position replay consumes every frame;
6. NVIDIA's converter writes a seven-action/eight-state LeRobot dataset.

This proves control/data compatibility, not clinical skill, reward quality, policy convergence or sim-to-real
validity. Needle lift and handover still require clinically meaningful observations, rewards, terminations,
force limits, RCM constraints and held-out evaluation before they can be called RL-ready. GR00T fine-tuning also
remains unavailable for the v0.7 surgical YAMLs while their upstream `train_module` is `null`.
