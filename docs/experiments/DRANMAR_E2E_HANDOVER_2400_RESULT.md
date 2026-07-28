# End-to-End Needle Handover: 2,400-Environment Result

Date: 2026-07-27
Baseline commit: `c17d97c6a5e798998321b394ece2fc6642edb53b`
Status: isolated experiment; not stage-qualified

## What was tested

- One learned actor controls both PSM Cartesian delta poses and both jaws.
- Observations are normalized into giver/receiver role order.
- Current and preceding native filtered jaw-contact forces are observed.
- DAgger behavior cloning uses the existing physics-qualified analytic controller
  only as a training-time teacher.
- The original rewards, hard force/drop terminations, and retained-transfer
  success termination are unchanged.
- No analytic action, object attachment, or teleportation exists at inference.

## 2,400-environment training result

- Requested and launched environments: 2,400
- DAgger updates: 1,200
- Deterministic validation frames per environment: 1,000
- Throughput: 55,348 frames/s
- Training and validation wall time: 95.4 s, excluding Isaac startup
- Peak process RAM: 34,499.7 MiB
- Teacher-controlled frames: 2,707,717
- Student-controlled frames: 172,283
- Successful mixed teacher rollouts: 678 / 2,701 (25.10%)
- Imitation loss: 0.15534 initial, 0.00040 final
- Deterministic learned-policy success: 0 / 3,236 completed episodes
- Checkpoint SHA-256:
  `f327361c00de4d50b8f68f4081e3b91842688c71163fd46c1aad7be284c35f00`

## First-outcome diagnosis

The first terminal outcome was measured once for every environment:

- 2,357 / 2,400: no sustained giver bilateral contact
- 42 / 2,400: giver contact but no 10 mm lift
- 1 / 2,400: lifted but did not acquire the receiver
- 0 / 2,400: receiver acquisition or retained handover

Terminal failures:

- 1,934 protected-surface force terminations
- 465 timeouts
- 1 premature giver release
- 0 object drops, pickup-attempt exhaustion, excessive object force, or
  receiver-retention failures

## Conclusion

Gilgamesh can run 2,400 of these environments safely and efficiently. Scaling
from 1,200 to 2,400 approximately doubled measured throughput without increasing
training wall time for the doubled sample count.

The tested direct-action architecture is not yet viable. Average imitation loss
hides small persistent action errors on long approach trajectories. Those errors
accumulate into table contact before grasp, so PPO handover tuning is not
warranted yet.

The next experiment should retain the same anti-reward-hacking physics contract
and change the data/actor structure:

1. Phase-balance teacher and recovery states instead of averaging mostly idle
   channels.
2. Train inactive-arm zero actions and active Cartesian channels with explicit
   per-phase weights.
3. Use longer student-controlled DAgger segments after a full successful-teacher
   warmup.
4. Require a non-zero standalone pickup/lift rate before enabling PPO.
