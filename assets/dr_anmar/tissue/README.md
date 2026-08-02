# Dr.Anmar suturable tissue

This family provides the intact tissue composition, left and right retained
closure surfaces, and explicit tetrahedral layer. The physics envelope and
known blocked behaviors are defined in
`physics_next/tissues/dr-anmar-suturable-tissue-v1.json`.

The penetration and pullout tasks use the per-flap `.tet.usda` assets. Each
flap is one connected GPU FEM volume containing:

- a regular `SimulationMesh` TetMesh with explicit rest-shape attributes;
- a separate surface-matching `CollisionMesh` TetMesh;
- a render mesh bound to the deformable pose;
- a nearly incompressible DrAnmar research material (180 kPa tangent modulus,
  0.47 Poisson ratio, 1050 kg/m3 density, damping and wet friction seeds);
- 24 position iterations, self-collision, 0.5 mm contact offset and 0.1 mm
  rest offset.

At reset the task constrains only the remote four-millimetre outer band of
each flap: 190 nodes per side. The wound edges remain physically free to
indent, stretch, and rebound. The controller remains at 50 Hz while FEM
physics advances at 1 ms. PSM and jaw collisions remain enabled against the
tissue. Needle passage is still owned by the force-gated DrAnmar tract model
because fixed-topology PhysX FEM does not create a persistent puncture hole.

Deformation and contact support do not imply qualified arbitrary puncture,
persistent tracts, thread passage, cutting, healing, or clinical validity.
