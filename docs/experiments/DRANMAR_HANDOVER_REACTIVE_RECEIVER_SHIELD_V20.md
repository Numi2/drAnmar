# Reactive Recovered-Receiver Shield v20

## Architecture

v17 proved all five scale safety terminals crossed 2 N on the receiver.
v18 and v19 proved unconditional final-approach deceleration removes both
collisions and useful acquisitions.

v20 preserves the fast v15 controller and adds a task-space action projection
inside Isaac Lab. It does not alter the frozen checkpoint or its 107-D learned
observation:

1. only the active receiver in a recovered phase-2 trajectory is eligible;
2. the action term reads PhysX-derived jaw force not attributable to the
   needle;
3. above 0.25 N, it freezes translation and rotation, then commands a 0.5 mm
   positive-Z retreat per control step for three steps; and
4. the existing 2 N hard termination remains unchanged.

The shield is included in the environment runtime-contract hash. Qualification
therefore requires a same-source baseline at 0.06 recovery transport and a
candidate at the v15 0.08 contact-qualified transport ceiling. Reward,
success, deadline, grippers, giver release, and physical force terminals are
unchanged.
