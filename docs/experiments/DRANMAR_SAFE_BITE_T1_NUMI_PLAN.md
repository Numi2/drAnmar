# T1 Numi Validation Plan (Prepared, Not Executed)

This plan is deliberately unexecuted. No SSH connection, Isaac launch, GPU
allocation, policy rollout, or training run is part of the local T1 change.

## Approval boundary

The 2,400-environment run is a separate approval point. It must not start until
the 1,200-environment deformable workload has recorded finite state, correct
fixture counts, measured VRAM headroom, and stable throughput.

## Gate 1: clean revision and registration

Run from a clean, reviewed T1 revision on Gilgamesh:

```bash
./dr_anmar_learning.sh validate
./dr_anmar_learning.sh list
```

Required:

- all three T1 task IDs register;
- the pinned coupled MJWarp plus VBD solver imports;
- quaternion identity remains `(0, 0, 0, 1)`;
- each environment reports exactly 80 tissue fixture nodes; and
- no local-only or uncommitted source is used for evidence.

## Gate 2: native deformable smoke

Instantiate the T1 environment without training:

```bash
./dr_anmar_learning.sh probe \
  DrAnmar-Handover-Needle-Safe-Bite-T1-v0 \
  64 \
  40 \
  output/dranmar-learning/t1/probe-64
```

Required:

- finite rigid and deformable state for all frames;
- exactly one TetMesh per environment;
- no NaN state, and no inverted tetrahedra when the native qualification
  instrumentation is run;
- tissue outer edges remain fixed while interior nodes remain dynamic; and
- process/GPU memory and frames per second are recorded.

## Gate 3: analytic full-chain baseline

The first policy run uses no checkpoint and no learned residual:

```bash
DR_ANMAR_ISAAC_PYTHON="${DR_ANMAR_ISAAC_PYTHON:-${HOME}/dr_anmar/physics-next/env_isaaclab/bin/python}"
"${DR_ANMAR_ISAAC_PYTHON}" scripts/dr_anmar_learning_benchmark.py play \
  --task DrAnmar-Handover-Needle-Safe-Bite-T1-v0 \
  --analytic-only \
  --num_envs 1200 \
  --num_frames 1250 \
  --seed 17 \
  --benchmark_formatter schema,json \
  --output_path output/dranmar-learning/t1/analytic-seed17
```

Repeat with seeds `2361` and `4099`. Because the first episode begins with an
empty snapshot cache, these are full-chain handover-to-entry outcomes. Report:

- completed handover, armed entry, and timeout rates;
- premature-contact and retention-loss rates;
- minimum position-error quantiles and achieved dwell;
- fixture node count, peak VRAM/RAM, and throughput; and
- `checkpoint: null` plus `analytic_only: true`.

## Gate 4: puncture-transition continuity

Exercise the nonterminal chain with the analytic controller:

```bash
"${DR_ANMAR_ISAAC_PYTHON}" scripts/dr_anmar_learning_benchmark.py play \
  --task DrAnmar-Handover-Needle-Safe-Bite-Chain-v0 \
  --analytic-only \
  --num_envs 64 \
  --num_frames 1250 \
  --seed 17 \
  --benchmark_formatter schema,json \
  --output_path output/dranmar-learning/t1/chain-seed17
```

Required:

- armed entry does not terminate the chain;
- the bounded post-arm command moves along the sampled inward direction;
- simulator contact onset plus inward speed records
  `authorized_contact_transition`;
- pre-arm contact still terminates as failure; and
- no policy-written puncture or contact flag is accepted.

This gate proves continuity into contact, not puncture mechanics or calibrated
tissue fidelity.

## Gate 5: train only if the analytic baseline misses

Do not train merely because training is available. If the analytic baseline
already meets the contract, keep it. Otherwise train only the bounded
receiver-pose residual:

```bash
DR_ANMAR_SEED=17 \
DR_ANMAR_SUCCESS_THRESHOLD=0.80 \
./dr_anmar_learning.sh train \
  DrAnmar-Handover-Needle-Safe-Bite-T1-v0 \
  1200 \
  1500 \
  output/dranmar-learning/t1/train-seed17
```

Training must preserve:

- analytic pickup, handover, jaws, and post-arm inward transition;
- frozen shared handover features and exploration scale;
- receiver pose-only residual authority;
- zero reward for holding, proximity, contact, height, or phase occupancy; and
- separate snapshot-initialized and full-chain evidence.

Promote a checkpoint only after held-out full-chain play at seeds `17`, `2361`,
and `4099` with the snapshot cache initially empty.

## Gate 6: 2,400 environments (requires explicit approval)

After Gates 1–5 pass at 1,200 environments, first run a short 2,400-environment
probe and inspect measured VRAM headroom. Only then run a 2,400-environment
training or qualification job. No command for this gate is authorized by this
prepared plan alone.
