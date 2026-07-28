# Recovered-Transport Receiver Preposition v12

## Root cause

The v11 residual was safe and directionally useful, but its stable-presentation
gate excluded most of the target failure cohort. In the paired
2,000-environment evaluation, 708 episodes lifted the needle without receiver
acquisition and only 305 of those reached stable presentation. The other 403
could not receive a v11 correction.

Successful acquisition normally takes another 50--60 control steps after
stable presentation. Waiting to start receiver correction until that point
wastes the transport interval and is especially costly after a late recovered
pickup.

## v12 architecture

v12 caches complete simulator and logical state at recovered lifted custody,
including the original episode clock. During phase 2 it keeps the promoted
incumbent active and learns only a bounded receiver SE(3) residual. This lets
the receiver preposition concurrently while the giver transports the needle.

The giver trajectory, both grippers, release authority, retained-success
predicate, hard terminations, episode horizon, and contact/force observations
are unchanged. The receiver cannot close early through this residual. The
adapter remains zero initialized, so broadening its activation gate has exactly
zero effect before learning.

The design follows the useful principle from residual and contact-rich RL:
retain the competent nominal controller, give learning only the correction
needed inside a physically qualified state region, and train from replay states
that cover the causal lead-up to the terminal objective.

## Qualification

v12 must first reproduce the incumbent exactly with its zero adapter. A learned
checkpoint is screened on the same 600-episode development population, then
scaled only if it beats 390 without hard failures. Promotion still requires the
frozen multi-seed contract and safety non-inferiority. Simulator results do not
establish physics calibration or clinical validity.

## Result

The zero adapter exactly reproduced 390/600. Checkpoint 5 then regressed to
387/600 with recovered successes falling from 32 to 29 and no hard safety
events. v12 is rejected.

The experiment exposed two errors in its causal design. Receiver correction
cannot make the giver reach a stable presentation, and the 128-step rollout is
shorter than the observed lifted-custody-to-retained-success sequence. v13
assigns the pre-stability correction to the giver, the post-stability
correction to the receiver, and extends rollout credit to 384 steps.
