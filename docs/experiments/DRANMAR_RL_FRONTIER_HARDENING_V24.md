# Dr.Anmar RL Frontier Hardening v24

## Decision

The 74% v23 result is useful development evidence, not a policy to discard and
not a promoted result. v24 keeps its checkpoint, analytic pickup/recovery
primitive, learned joint-transfer adapter, terminal contract, and safety
terminations frozen and active. A separate exact-zero adapter addresses the
structural gaps found by the frontier audit.

No v24 result may be called improved until held-out standard and durability
qualification pass on every declared seed. This contract is simulator
validation only; it is not physics calibration or clinical validation.

## What changed

### Checkpoint semantics are now fail-closed

Every checkpoint launch must supply a versioned policy bundle binding:

- checkpoint SHA-256;
- serving task;
- adaptation mode;
- controller profile name and profile SHA-256;
- behavior-bearing policy and controller fields.

The v23 migration bundle is
`config/policy_bundles/joint-transfer-v23.json`. A mismatched task, checkpoint,
profile, or runtime field fails before Isaac launches. The explicit
`--allow-unbundled-checkpoint` option exists only for unsafe development
diagnostics and is recorded in evidence.

Successful v24 training writes `model_final.policy-bundle.json` beside the
checkpoint. That new bundle, not a set of remembered CLI flags, is the serving
artifact.

### Geometry is canonical without corrupting v23

The v23 profile preserves its yaw-only and unrotated legacy offsets exactly.
The `frontier-hardening-v24` profile rotates every needle-local grasp offset
with the observed `(x, y, z, w)` quaternion. The frozen v23 actor and its
learned adapter retain their original feature tensors. Only the new v24
adapter consumes full canonical SE(3) features.

### Mid-air custody is controlled before loss

The controller derives a bounded custody-quality score from:

- weakest jaw contact;
- left/right contact balance;
- contact trend against the previous control frame;
- needle linear and angular motion.

Below the slow threshold, transport authority is smoothly reduced and the
giver receives a small bounded centering correction. The giver never opens in
mid-air, contact is not fabricated, and the existing three-attempt physical
regrasp/re-lift recovery remains authoritative. Evidence separately records
minimum quality, minimum transport scale, and episodes where the governor was
active.

### The new residual is narrow and exact-zero

`_FrontierHardeningAdapter` sees canonical giver/receiver grasp errors,
presentation error, object twist, custody quality, previous SE(3) actions,
recovery context, and deadline. Its output layer is initialized exactly to
zero. It can correct:

- giver XY during approach/regrasp;
- giver SE(3) during pickup and transport;
- receiver SE(3) during acquisition.

It cannot command either gripper or redefine a terminal. The v23 phase
network, joint adapter, normalization, and all older adapters are frozen.

### Roles, resets, and replay cover the real failure distribution

The v24 task assigns Robot 1 and Robot 2 as giver in an alternating 50/50
population and swaps each environment on subsequent resets. Resting needles
sample the full planar yaw range and a 25 mm XY placement envelope. Roll and
pitch remain zero because a freely tilted resting needle is not a physically
valid table state.

Mass and friction randomization are intentionally disabled until measured
instrument/needle calibration receipts define credible ranges. This is a
calibration boundary, not a claim of robustness.

Receiver-curriculum replay is stratified by giver identity and recovered versus
first-attempt custody. Within each stratum, states that previously produced
failures gain bounded sampling priority; successes remain in the distribution.
Complete simulator and Markov state restoration is unchanged.

### Dense reward cannot pay for a failed partial trajectory

Terminal retained transfer remains `+80`; physical transfer failure remains
`-80`; safety penalties and terminations remain independent. v24 replaces the
one-time phase bonus with:

`F(s, s') = gamma * Phi(s') - Phi(s)`

`Phi` is bounded, uses simulator-owned approach/lift/presentation/retention
state, and is forced to zero on success, failure, or timeout. Approaching,
lifting, or touching can improve credit assignment between states, but a
failed episode cannot retain that accumulated positive credit.

## Efficiency experiment

`config/experiments/dranmar_rl_efficiency_matrix.json` fixes every candidate to
13,824,000 simulator frames per seed:

| Environments | Rollout steps | Iterations | Frames |
| ---: | ---: | ---: | ---: |
| 600 | 64 | 360 | 13,824,000 |
| 1,200 | 64 | 180 | 13,824,000 |
| 2,400 | 64 | 90 | 13,824,000 |

The declared seeds are 17, 2361, and 104729. The matrix tool rejects missing
cells, unbundled source checkpoints, changed frame budgets, and runs that did
not emit a serving bundle. Throughput alone does not win: the provisional
winner is lowest multi-seed median estimated time to the training threshold.

Promotion still requires:

1. held-out `DrAnmar-Handover-Needle-Frontier-Eval-v0`;
2. held-out
   `DrAnmar-Handover-Needle-Frontier-Durability-Eval-v0`, which requires 60
   receiver-only control steps (1.2 seconds at the current 50 Hz policy rate);
3. both giver roles represented;
4. at least 95% completed first outcomes;
5. no safety-terminal regression;
6. standard success at least 74% and durability success at least 65% on every
   declared seed.

## Approved-run template

Do not execute this GPU command until the user approves the 2,400-environment
run:

```bash
DR_ANMAR_TRUST_REQUESTED_NUM_ENVS=1 \
DR_ANMAR_SEED=17 \
DR_ANMAR_INIT_CHECKPOINT=output/v8-joint-2400x90-lr1e5/runs/2026-07-28_03-19-21/model_20.pt \
DR_ANMAR_POLICY_BUNDLE=config/policy_bundles/joint-transfer-v23.json \
DR_ANMAR_POLICY_LEARNING_RATE=1e-5 \
DR_ANMAR_CHECK_SUCCESS=0 \
./dr_anmar_learning.sh train \
  DrAnmar-Handover-Needle-Frontier-Hardening-v0 \
  2400 \
  90 \
  output/frontier-hardening-v24/2400-seed17
```

The source bundle validates before simulator startup. After training, held-out
play must use the generated `model_final.policy-bundle.json`; it must not
reconstruct v24 with adaptation or controller CLI overrides.
