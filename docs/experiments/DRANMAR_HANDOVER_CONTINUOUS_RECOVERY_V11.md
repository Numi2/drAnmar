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

## Result

The zero adapter exactly reproduced all 390/600 incumbent outcomes with the
same initial-state and environment-contract hashes. At `1e-5`, checkpoint 29
improved retained handovers from 390 to 392 on the development population.
On the paired 2,000-environment scale population it improved 1,130 to 1,134,
entirely through recovered successes (99 to 103), with zero drops or excessive
force. Protected-surface events changed from two to three.

This is directionally correct but below the pre-registered half-percentage
point promotion threshold, so v11 is not promoted. A `3e-5` sweep regressed to
386/600 and was rejected.

The remaining structural bottleneck occurs before v11 can act. Of 708
lifted-without-acquisition failures at scale, only 305 reached stable
presentation. v12 therefore moves replay and bounded receiver preposition
authority earlier, to recovered lifted custody, without changing giver or
gripper authority.
