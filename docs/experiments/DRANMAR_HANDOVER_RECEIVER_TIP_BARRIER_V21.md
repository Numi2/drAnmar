# Prospective Receiver-Tip Barrier v21

## Geometry

The needle-derived giver and receiver grasp frames are 15.198 mm apart. A
valid receiver acquisition therefore does not require the two PSM tool-tip
frames to cross or coincide.

## Control barrier

During recovered phase-2 receiver approach, v21 computes receiver tool-tip
position in the giver root frame. Inside an 18 mm activation radius, it
projects out only the radial action component that would command separation
below 12 mm. Tangential translation and all orientation commands remain
unchanged.

This is prospective collision avoidance from existing policy state. It adds
no observation, changes no frozen checkpoint weights, and does not wait for a
contact impulse. First-attempt trajectories, outer approach, the curved-needle
grasp target, giver transport, grippers, release, reward, success, deadline,
and 2 N hard termination are unchanged.

Qualification requires at least 393/600 and 1,140/2,000 retained handovers,
zero drops and excessive object-force events, and no more than four
protected-surface terminals.

## Result

v21 produced 393/600, then the 2,000-environment result was bitwise identical
to v15: 1,164 retained handovers, 133 recovered successes, and five
protected-surface terminals. The collision does not cross the 12 mm tool-tip
barrier. v21 is rejected and removed from serving control.

The remaining attribution must separate receiver-jaw contact against the giver
tool from receiver-jaw contact against the support surface or other geometry.
