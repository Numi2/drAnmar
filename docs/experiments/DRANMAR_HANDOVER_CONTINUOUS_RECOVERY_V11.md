# Continuous Deadline Recovery Residual v11

## Decision

The promoted v8 checkpoint remains the policy incumbent. The v10
continue/re-seat/backoff option is rejected after two matched 600-episode
screens produced 358 retained handovers versus the incumbent's 390 and erased
all recovered successes.

v11 preserves the useful parts of v10:

- complete simulator and logical Markov-state replay;
- the original episode clock and remaining-time observation;
- recovery-only activation after a physically qualified presentation;
- frozen pickup, lift, presentation, release, and success behavior;
- zero-initialized, bounded receiver SE(3) authority;
- 128-step rollouts that cover the measured presentation-to-contact latency.

It removes the unsafe abstraction. There is no learned trajectory-level
continue, re-seat, or backoff choice. The incumbent action always remains
active and PPO can add only a bounded continuous receiver correction. At zero
initialization the action is exactly the incumbent action. Exploration is
restricted to the six receiver pose axes only while the recovery gate is
active; neither gripper nor the giver is learnable.

## Why this is the correct learning problem

The residual targets the actual concentrated bottleneck: late presentations
following a physical pickup recovery. It does not perturb the healthy
first-attempt cohort, redefine success, extend the horizon, or pay for partial
progress. Failed timeouts receive the same magnitude terminal cost as retained
success receives terminal value.

This is a conservative residual-RL design: the known controller supplies the
nominal trajectory and learning optimizes a small correction inside a
physics-qualified state region. It avoids the discontinuous switching and
policy-wide relearning that regressed earlier experiments.

## Qualification

The experiment is promotable only if it:

1. reproduces the incumbent exactly with an all-zero adapter;
2. beats 390/600 on development without a hard safety event;
3. improves a frozen candidate over paired fresh populations under identical
   environment and initial-state hashes;
4. passes the existing multi-seed Wilson, non-inferiority, and catastrophic
   failure gates.

Until those gates pass, v11 is an experiment on the protected branch, not a
new serving policy. Results are simulator evidence only, not physics
calibration or clinical validation.
