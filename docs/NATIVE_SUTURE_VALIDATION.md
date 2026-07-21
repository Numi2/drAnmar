# Native suture validation gate

The threaded-ring and knot exercises must remain unavailable until all items
below pass in the live Isaac/PhysX runtime. A visual match, expert-controller
phase, trajectory waypoint, or completion score is never evidence of physics.

## Authority boundary

- PhysX owns deformable thread state, gravity, velocity, collision and contact.
- The needle is a rigid body and the ring is rigid collision geometry.
- The thread-to-needle boundary is a native deformable attachment or a native
  FEM kinematic boundary tied to the needle eye.
- Render geometry is read from the PhysX state. It never writes a desired curve
  back into the solver.
- Expert actions only command robot joints and grippers. They cannot move the
  needle or thread directly and cannot declare a grasp, pass, knot or recovery.
- Task completion is derived from physical state after the action has occurred.

## Required qualification

1. A free strand falls under gravity with no authored point writes.
2. The needle attachment follows the needle eye and releases only when that
   physical boundary is explicitly removed.
3. The strand collides with the ring tube while passing freely through its open
   center.
4. The needle cannot pass through the ring tube.
5. Strand self-contact prevents crossings from occupying the same volume.
6. Releasing both instruments leaves the needle and strand under physics.
7. Reset produces the same initial state without hidden latches.
8. No expert state, score, or UI flag changes any physics state.

## Current result

`scripts/dr_anmar_native_suture_probe.py` is intentionally a failing gate.
Gilgamesh confirms a native tetrahedral PhysX strand responds to gravity, but
the deformable-to-kinematic-needle attachment does not yet follow the moving
needle correctly. The threaded-ring room therefore remains disabled. The
former projected knot route and gesture-scored completion path are not used.

