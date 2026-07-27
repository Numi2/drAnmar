# Needle Handover Contact Centering v55

Date: 2026-07-27

Status: promoted development controller candidate; pending the 2,000-environment
scale gate and not Stage 6 qualified.

## Causal change

The v33 checkpoint, pickup controller, receiver approach, curved-needle
orientation, rewards, success predicate, and physics settings remain unchanged.
The only behavior change doubles the receiver's existing post-contact centering
authority from `0.0025` to `0.005`.

That translation is active only after at least one receiver jaw has physically
contacted the needle. It therefore cannot command pickup, lift, presentation, or
the pre-contact receiver approach. Its purpose is to seat an off-center first
contact into bilateral capture before release.

## Matched deterministic results

| Metric | v33 baseline, 512 | v55, 512 | v33 baseline, 600 | v55, 600 |
|---|---:|---:|---:|---:|
| Retained handovers | 322 | 325 | 377 | 381 |
| Raw physics success term | 328 | 329 | 380 | 384 |
| Receiver retention loss | 29 | 26 | 21 | 18 |
| Pickup attempts exhausted | 17 | 17 | 21 | 21 |
| Stable presentations | 418 | 418 | 489 | 489 |
| Receiver contact | 356 | 356 | 413 | 414 |
| Windowed bilateral capture | 351 | 351 | 399 | 402 |
| Hard safety events | 2 | 3 | 2 | 2 |
| Excessive object force | 0 | 0 | 0 | 0 |

The candidate improved retained completion on both matched populations while
preserving pickup exhaustion and stable presentation. On the 600 gate, total
hard safety events remained two: object drops changed from one to zero and
protected-surface events changed from one to two.

The upper calibration at `0.0075` was rejected after falling to 315 retained
handovers on the matched 512 population, with 19 pickup exhaustions and 31
retention losses.

## Qualification boundary

This is simulator evidence, not clinical validation. The 0.005 controller is a
development candidate because concurrent GPU workloads left insufficient free
VRAM for the matched 2,000-environment scale run. The scale-qualified v33
checkpoint remains the formal baseline until that gate passes.
