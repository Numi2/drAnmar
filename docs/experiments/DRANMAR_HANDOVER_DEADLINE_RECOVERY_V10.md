# Deadline-Aware Recovery Option v10

## Audit result

The 600-environment development incumbent completed 390 retained handovers
with no hard or protected-surface failures. The remaining 210 episodes were
not dominated by a broken first grasp:

| Maximum physical phase | Episodes |
| --- | ---: |
| Giver contact without 10 mm lift | 26 |
| Lifted without receiver acquisition | 177 |
| Receiver acquired without retained success | 7 |

Of the 177 lifted acquisition failures, 171 had already executed pickup
recovery. Their mean first lift occurred at step 545 versus step 417 for
successful episodes. The receiver normally required about 60 further steps
from stable presentation to first contact. Late recovered presentations
therefore reached a valid downstream task with materially less time remaining.

This explains why training a general receiver residual regressed the incumbent:
it changed healthy first-attempt trajectories to solve a problem concentrated
in the recovered, deadline-constrained cohort.

## Rejected approaches

- A 75-step deterministic no-contact retry reduced retained success from
  390/600 to 345/600 and fired in 373 environments.
- A 150-step retry tied the incumbent at 390/600 and fired once.
- The v9 receiver refinement checkpoints ranged from 378 to 389 retained
  successes on the same 600-environment development population.

Fixed retry timing and general receiver fine-tuning remain disabled.

## v10 architecture

The v10 option is active only after a physical pickup recovery reaches a
qualified stable presentation. The incumbent phase policy, giver motion,
release logic, success predicate, episode horizon, and hard terminations stay
frozen.

The option observes:

- needle-local receiver position and orientation error;
- current and previous native giver jaw-contact forces;
- object linear and angular velocity;
- the actual remaining episode fraction;
- recovery and transfer-contract state.

It makes an explicit differentiable hard choice among continue, re-seat, and
backoff, then adds at most a bounded receiver SE(3) correction. The output layer
is zero initialized and the continue option is the exact initial action, so an
untrained v10 checkpoint must match the incumbent bit-for-bit at policy output.
The three logits begin tied at zero: deterministic `argmax` selects continue,
while the straight-through soft gradient lets PPO cross into either recovery
option without first overcoming a hand-authored logit margin. A diagnostic
9.2-million-frame run with a `+2` continue prior was rejected because the
largest learned bias shift was only `0.002`, so re-seat/backoff never executed.

## Replay correction

Previous recovered-state replay restored the simulator and logical Markov
state but reset the episode clock. That removed the deadline pressure the
option was intended to solve. v10 caches and restores the captured
`episode_length_buf`, so training sees the same remaining horizon as the
physical trajectory that generated the state.

The compact deadline signal reuses the second role-identity channel only in
the isolated v10 task. The frozen incumbent canonicalizes that channel before
its phase network, preserving the 107-value checkpoint shape and exact
incumbent behavior.

## Qualification

Promotion requires:

1. exact-zero equivalence to the incumbent on a paired population;
2. one frozen candidate across at least three pre-registered seeds;
3. identical environment-contract hashes and paired initial-state hashes;
4. at least 0.5 percentage-point aggregate improvement;
5. no more than 1 percentage-point regression on any seed;
6. zero catastrophic failures and protected-surface non-inferiority.

This is simulator evidence only. It is not physics calibration, clinical
validation, or a claim of surgical readiness.
