# Recovered Receiver Deceleration v18

## Measured failure

Pre-reset v17 attribution proves robot 2 crossed the 2 N protected-surface
limit in every v15 safety terminal. Four of five collisions happened after the
needle was lifted but before acquisition. The giver is not the primary
collision source.

## Targeted control

v18 preserves the v15 0.08 contact-qualified recovered transport. Only in a
pickup-recovery episode, and only within 6 mm of the curved-needle receiver
grasp frame, receiver translation is capped at 0.05 instead of 0.10.

This is a final-approach kinetic-energy reduction. It does not move the target,
change needle curvature calibration, alter orientation, delay preposition,
change either jaw command, or modify reward, success, deadline, release, or
hard terminations. First-attempt receiver trajectories are bitwise unchanged.

The screen requires at least 393/600 retained handovers with zero protected
events. Scale requires at least 1,140/2,000 retained handovers, zero drops and
excessive object-force events, and no more than four protected-surface
terminals.

## Result

v18 produced zero protected-surface events but only 390/600 retained
handovers, erasing the recovery gain. It is rejected. v19 narrows deceleration
to the final 4 mm and raises the action cap to 0.075.
