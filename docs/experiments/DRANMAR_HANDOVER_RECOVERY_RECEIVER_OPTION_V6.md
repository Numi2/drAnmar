# Recovery-Conditioned Receiver Option v6

## Outcome

The recovery-conditioned receiver architecture is retained, but every policy
and controller candidate from this experiment is rejected. The qualified
pickup-recovery option v4 remains the performance champion.

The retained architecture starts receiver training only from
simulator-observed stable presentations reached after a physical pickup loss
and relift. It keeps the pickup-recovery policy active and frozen, trains only
receiver SE(3) rows in phase heads 2 and 3, and routes nominal episodes through
a frozen copy of the loaded v4 phase network.

That frozen reference fixed the structural defect in v5: the v6 model-10
screen reproduced the baseline exactly at 187/256 retained handovers,
163 first-attempt successes, and 24 recovered successes. Later checkpoints
did not improve recovery and were rejected.

## Rejected causal tests

| Candidate | Scale | Baseline | Candidate | Decision |
| --- | ---: | ---: | ---: | --- |
| Recovery preposition height 10 mm | 1,200 | 719 | 718 | Rejected |
| Recovery final approach limit 0.12 | 1,200 | 719 | 715 | Rejected; code removed |
| Absolute-yaw receiver offset | 3,600 | 2,197 | 2,148 | Rejected; code removed |

The absolute-yaw candidate looked positive on seed 17 at 256 and narrowly
positive at 1,200, but seed 2361 regressed by 48 successes. Across all three
seeds it lost 49 retained handovers, 65 stable presentations, 55 receiver
contacts, and 47 bilateral captures while adding 42 exhausted pickup retries.
This is why single-seed controller gains cannot be promoted.

## Root cause and next representation

The receiver currently reconstructs curved-needle position and orientation
from separately calibrated offsets. Rotating the position offset by absolute
robot-root yaw is not equivalent to a needle-local grasp frame. The next
receiver representation must expose one canonical SE(3) grasp pose derived
from the needle asset geometry:

1. position on the needle arc;
2. local tangent and surface-normal orientation;
3. the same transform in the analytic controller, policy task features,
   replay cache, and diagnostics;
4. recovery-only adaptation behind the frozen nominal reference network.

Until that representation exists and passes a matched multiseed gate, no
height, speed, or absolute-yaw receiver candidate should be repeated.

The hashed record is
[`recovery-receiver-option-v6-rejected.json`](evidence/recovery-receiver-option-v6-rejected.json).
