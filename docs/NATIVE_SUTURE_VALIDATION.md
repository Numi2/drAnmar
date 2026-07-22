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

The original hand-authored probe in `scripts/dr_anmar_native_suture_probe.py`
remains a useful recorded failure: its deformable-to-kinematic-cube boundary
did not follow the moving anchor. It is not the promoted implementation.

The next qualification route uses NVIDIA's unmodified SoftMimicGen
`Rope.usd` and `Ring.usd`, pinned and checksum-verified by
`physics_next/softmimicgen.json`. The official release provides a native PhysX
FEM strand and rigid ring, but its PSM task grasps the strand directly: it does
not contain a needle attachment, knot mechanics, or enabled strand
self-collision.

That exact upstream task is available as the separate **PSM strand ring
threading** room. `dr_anmar_suture_native.sh validate-upstream demo_0` resets
the released initial state, replays the released 123 relative-IK actions
unchanged, evaluates NVIDIA's own ring-crossing predicate, and compares the
live robot, ring and every one of the 549 FEM nodes with the recorded states.
NVIDIA's bundled `replay_demos.py --validate_states` is not used as evidence:
at this pinned revision it crashes on the saved singleton environment dimension
and does not compare deformable objects.

The 2026-07-22 Gilgamesh strict replay passed. The live task first satisfied
the native predicate at action 113 and remained successful at action 123. The
terminal strand error was 0.000054 m RMS and 0.000220 m worst-node absolute;
the worst node deviation across the full replay was 0.000680 m. The robot
joint-position maximum error was 0.000138 and the ring pose matched exactly.
There were no shape errors or non-finite values. The complete report
is `physics_next/benchmarks/softmimicgen-threading-replay.json`.

`dr_anmar_suture_native.sh qualify core` separately tests the
missing native rigid-needle attachment. On 2026-07-22 Gilgamesh's
RTX 4090 passed this core boundary twice with the real ORBIT-Surgical rigid
needle. In the installed qualification run, the needle moved 0.082540 m, the
terminal FEM nodes followed with 0.002083 m transform error, the free end fell
0.259716 m, and the probe authored no deformable points. The evidence is stored in
`physics_next/benchmarks/softmimicgen-needle-attachment.json`.

These are two deliberately separate pieces of engineering evidence: NVIDIA's
direct-strand ring task, and Dr.Anmar's needle-to-strand attachment boundary.
They do not combine into a qualified needle-through-ring or knot simulation.
Those rooms remain unavailable because release/reset, needle-ring contact and
strand self-contact have not all passed together. The former projected knot
route and gesture-scored completion path are not used.

`dr_anmar_suture_native.sh qualify promotion` remains the authoritative full
gate and must fail until all eight requirements above have measured evidence.
