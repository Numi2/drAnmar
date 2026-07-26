# DrAnmar Warp Suture Backend

The DrAnmar 4-0 braided suture uses a dedicated NVIDIA Warp dynamics backend
for knotting and other self-contact-heavy tasks. The OpenUSD asset and
`physics_next/sutures/dr-anmar-suture-4-0.json` remain the authoritative
identity, geometry, material, damage, and validation contracts.

## Why this backend exists

The layered PhysX payload remains useful for ordinary OpenUSD inspection and
compatibility, but it is not the promoted microscopic knotting path. The Warp
backend represents the 360 authored segments
with 361 particles and solves:

- a global, nonlinear axial XPBD system;
- discrete bending constraints with a stiffer swage transition;
- exact closest-point contact between non-local centerline segments treated as
  capsules;
- Coulomb tangential friction and persistent moving-contact knot compaction;
- a breakable needle-exit attachment and instrument grasps that can transfer;
- localized breakage using the profile force-elongation and damage envelope.

All 63,903 eligible non-local pairs are evaluated on the GPU for the current
asset. This is intentional: exhaustive contact is affordable at this fixed
resolution and cannot miss a knot crossing because of a broad-phase false
negative. A fixed CSR gather replaces floating-point correction atomics, so the
same inputs produce bit-exact replay on the qualified stack.

`WarpSuture.segment_transforms()` maps the solved centerline back to all 360
authored segment visuals using parallel-transport frames, avoiding arbitrary
roll flips along loops and knots.

Free displacement is capped below half a strand diameter per internal substep.
That substep rule and segment-capsule contact close the gaps that point-particle
collision would leave between 0.5 mm nodes on a 0.25 mm strand.

## NVIDIA alignment

The implementation follows current NVIDIA boundaries:

- PhysX PBD, FEM, and particle features are GPU paths, while the old
  `warp.sim` module has been removed from current Warp.
- Warp kernels use statically typed arrays and native CUDA execution.
- Fixed allocation and pair ordering avoid dynamic GPU buffer growth during a
  step.
- The native-simulator evidence run includes deterministic replay because CUDA floating-point
  atomics are not deterministic by default.

References:

- <https://nvidia.github.io/warp/stable/user_guide/basics.html>
- <https://nvidia.github.io/warp/stable/user_guide/deterministic_execution.html>
- <https://nvidia-omniverse.github.io/PhysX/physx/5.7.0/docs/GPURigidBodies.html>
- <https://nvidia-omniverse.github.io/PhysX/ovphysx/latest/simulation_setup/particles.html>

## Run on the qualified NVIDIA workstation

```bash
DR_ANMAR_STABLE_ISAAC_PYTHON=/path/to/qualified/isaac/python \
DR_ANMAR_ROOT=/path/to/dr-anmar-runtime \
./dr_anmar_suture_warp.sh qualify
```

The committed native-simulator evidence is
`physics_next/benchmarks/dr-anmar-warp-suture-native-simulator-evidence.json`.

## Evidence boundary

Passing the GPU report means the implementation passed its deterministic
engineering scenarios on the recorded NVIDIA stack. It does not establish
clinical validity. Resolution and timestep convergence, physical straight and
knotted tensile distributions, wet friction, instrument-crush localization,
swage pullout, real-video perception, and clinician validation remain
fail-closed gates.
