# Recovery Receiver Adapter v7 and Retry Audit

## Outcome

The zero-initialized needle-local receiver adapter is retained as an isolated
experimental architecture. Its trained weights are rejected. Both
no-contact receiver retry controllers are also rejected and disabled.
Pickup-recovery option v4 model 87 remains the performance champion.

The adapter adds 1,222 parameters and receives canonical needle-local
position/orientation error, receiver contact, and transfer-contract state.
It is active only after physical pickup recovery. The loaded v4 phase network,
nominal path, pickup recovery, success definition, rewards, and safety
terminations remain frozen.

## Full-scale training

The v7 run used the requested 1,200 Isaac Lab environments without fitting
down, completed 90 updates and 6,912,000 frames in 331.4 seconds, and measured
20,857 frames/s. It captured 723 real post-recovery stable presentations.

Update 22 was the earliest learned checkpoint that matched the zero adapter
at the 64-environment screen. At the 256-environment gate it produced
171 retained handovers versus 174 for the matched zero-adapter baseline.
Receiver contacts fell from 187 to 185 and recovered successes from 13 to 10.
The weights therefore do not promote.

## Receiver retry root cause

Telemetry found that the existing receiver retry state never activated:
0/256 retries despite 60 lifted-without-acquisition episodes. The state only
started an attempt after receiver contact, so a complete geometric miss could
only timeout.

Two physics-gated fixes were tested:

| Candidate | Retained / 256 | Receiver contacts | Retry envs | Decision |
| --- | ---: | ---: | ---: | --- |
| Matched zero-adapter baseline | 174 | 187 | 0 | Control |
| 75-step no-contact timeout, 15-step backoff | 138 | 151 | 127 | Rejected |
| 5-step backoff, alternate arc points 0.60/0.70 | 112 | 134 | 140 | Rejected |

Both variants preserved the hard object-drop, excessive-force, and protected
surface gates at zero. They failed because the added acquisition time
destabilized giver custody faster than another deterministic pass could
recover the needle. The behavioral timeout is disabled (`0`) in the retained
source. The state counter remains available for a future learned,
custody-aware retry option.

## Next qualified direction

Do not repeat fixed retry timing or deterministic arc switching. The next
receiver option must jointly condition on:

1. remaining giver bilateral-contact margin and object motion;
2. time remaining before the episode deadline;
3. receiver grasp error in the needle frame;
4. an explicit decision among continue, re-seat, or abort/backoff;
5. held-out full-task improvement before any serving change.

The hashed record is
[`recovery-receiver-option-v7-rejected.json`](evidence/recovery-receiver-option-v7-rejected.json).
