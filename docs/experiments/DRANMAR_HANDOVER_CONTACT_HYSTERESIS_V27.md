# Needle Handover Contact-Hysteresis v27

Date: 2026-07-27

Status: promoted development controller baseline; not Stage 6 qualified.

## Structural correction

The physical state machine enters pickup phase 1 only after filtered bilateral
giver contact and declares pickup loss only after three consecutive contact-loss
steps plus loss of object following. The action controller previously abandoned
lift after one noisy contact sample. It now continues the bounded analytic lift
while phase 1 remains latched and hands control to the existing recovery phase
when the physical loss debounce fires.

Rewards, success, object state, force limits, action limits, checkpoint weights,
seed, and episode horizon were unchanged.

## Matched deterministic result

| Metric | v24 baseline, 2,000 | v27, 2,000 |
|---|---:|---:|
| Retained handovers | 939 (46.95%) | 1,017 (50.85%) |
| Giver contact without 10 mm lift | 462 | 175 |
| Successful recovered handovers | 7 | 67 |
| Stable presentations | 1,307 | 1,429 |
| Receiver contact | 1,046 | 1,137 |
| Object drop + excessive force + protected surface | 28 | 18 |
| Pickup attempts exhausted | 12 | 160 |
| Receiver retention loss | 47 | 50 |

The matched 600-environment screen also improved from 349 to 367 retained
handovers, reduced grasp-without-lift failures from 92 to 22, and increased
recovered successes from 2 to 21.

## Decision and next bottleneck

The result is retained because it raises simulator-owned terminal success,
fixes the measured pickup/recovery failure causally, and reduces hard physical
safety events. The exhausted-retry increase is not hidden: retry quality is
still weak, and the largest remaining stage is now downstream receiver
acquisition (755 of 2,000 environments lifted without acquisition).

The next experiment should improve retry selection and receiver acquisition
from this frozen baseline. It must not change rewards, the retained-transfer
success predicate, or object state.
