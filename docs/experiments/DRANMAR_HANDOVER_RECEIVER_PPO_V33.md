# Needle Handover Receiver PPO v33

Date: 2026-07-27

Status: promoted development training baseline; not Stage 6 qualified.

## Evidence-led change

The v27 controller remains frozen. A behavior-preserving diagnostic showed that
successful episodes take a median 98 control steps from stable presentation to
receiver contact, while 76 of 489 stable presentations never reached receiver
contact on the matched 600-environment rollout. The largest 2,000-environment
gap remained downstream: 756 episodes lifted without receiver acquisition.

Starting from the qualified v24 checkpoint, v33 performed 50 additional PPO
iterations with 2,000 environments and a reduced learning rate of 0.00005.
Only the receiver approach residual remained learnable. Giver pickup, lift,
fixed-pose presentation, hold, and release stayed under the v27 analytic and
physics-gated controller.

Rewards, success, object state, force limits, action limits, seed, and episode
horizon were unchanged.

## Checkpoint selection

The final checkpoint was selected over iteration 125 even though iteration 125
recorded one additional retained handover on the 600-environment screen:

| Metric | v27 baseline | v33 iteration 125 | v33 final |
|---|---:|---:|---:|
| Retained handovers, 600 | 367 | 378 | 377 |
| Receiver retention losses | 32 | 23 | 21 |
| Hard safety events | 1 | 3 | 2 |

The final checkpoint therefore preserved nearly all of the completion gain with
better retention and safety than iteration 125.

## Deterministic scale result

| Metric | v27, 2,000 | v33 final, 2,000 |
|---|---:|---:|
| Retained handovers | 1,017 (50.85%) | 1,021 (51.05%) |
| Raw physics success term | 1,043 | 1,044 |
| Receiver retention loss | 50 | 45 |
| Stable presentations | 1,429 | 1,429 |
| Receiver contact | 1,137 | 1,130 |
| Pickup attempts exhausted | 160 | 161 |
| Object drop | 4 | 3 |
| Protected-surface force | 14 | 5 |
| Excessive object force | 0 | 0 |
| Total hard safety events | 18 | 8 |

This is a valid but modest success-rate promotion. Its strongest verified gain
is reducing hard safety events by more than half while slightly improving
retained completion and reducing retention loss.

## Remaining bottleneck

The funnel is still dominated by 756 episodes that physically lift but never
acquire with the receiver. Of 2,000 episodes, 1,429 reach stable presentation
and 1,130 reach receiver contact. The next change should target the two measured
causes separately:

1. Preserve giver contact through transport so more lifts reach stable
   presentation.
2. Improve receiver contact geometry or timing only after stable presentation.

The retained-transfer predicate, rewards, and object dynamics must remain
unchanged.
