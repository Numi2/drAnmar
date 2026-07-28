# Two-Stage Recovered Handover v13

## Causal sequence

A recovered handover contains two different control problems:

1. the giver must transport the held needle to a stable, reachable
   presentation;
2. only then should the receiver align, seat its jaws, and acquire the needle.

v12 applied receiver correction during both stages and used a 128-step rollout.
That is the wrong actuator before presentation and too short for terminal
retention credit to reach the lifted-custody action.

## Architecture

v13 restores the explicit giver-presents/receiver-acquires sequence:

- before stable presentation, the frozen incumbent receives at most a bounded
  giver SE(3) residual;
- after stable presentation, the same compact state-conditioned adapter
  supplies at most a bounded receiver SE(3) residual;
- both grippers, giver release, success, timeout, hard terminations, and the
  original episode clock remain analytic and unchanged.

Replay begins from physically recovered lifted custody and restores complete
simulator and logical Markov state. Rollouts increase from 128 to 384 steps so
the optimization window spans transport, stable presentation, receiver
contact, release, and retention. This is temporal credit assignment, not an
episode-horizon extension.

The output layer remains zero initialized. With a zero adapter, the expanded
gate and longer rollout cannot change the incumbent action.

## Qualification

The exact-zero, 600-episode development, 2,000-episode scale, multi-seed,
environment-hash, initial-population-hash, Wilson, and safety gates remain
unchanged. Simulator qualification is not physics calibration or clinical
validation.
