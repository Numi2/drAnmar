# DrAnmar 4-0 Surgical Suture

DrAnmar Suture is an independently authored, research-grade 4-0 braided
surgical-thread asset. Its dimensions and constitutive seeds are
research-informed, but the model is not a medical device and remains blocked
from clinical use pending physical and clinician validation.

## Promoted implementation

The NVIDIA Warp backend is the only promoted dynamics path for knotting:

- 360 segments represented by 361 particles at the authored 0.5 mm spacing;
- global nonlinear axial XPBD and discrete bending constraints;
- exact non-local segment-capsule self-contact;
- Coulomb tangential friction and persistent knot compaction;
- transferable instrument grasps;
- a breakable DrAnmar Needle swage attachment;
- localized strand failure from the profile force-elongation envelope; and
- deterministic correction gathering without floating-point atomics.

The OpenUSD asset remains authoritative for identity, scale, render geometry,
materials, collision geometry, and its research parameter contract. Its
layered PhysX payload is retained for portable loading and inspection. It is a
compatibility representation, not the promoted microscopic knot solver.

Implementation and qualification details are in
`docs/DR_ANMAR_WARP_SUTURE.md`.

## Asset construction

The authored strand is 180 mm long and 0.25 mm in nominal diameter. It includes
closed braided render geometry, a compact normal/roughness texture, a smooth
3 mm swage transition, and a physical envelope that is not enlarged for visual
convenience.

The mechanical profile includes:

- 1,200–1,400 kg/m³ density bounds;
- a 20–25 N straight-strand failure envelope;
- 10–30% elongation-at-break bounds;
- local knot-strength reduction;
- wet stress relaxation;
- crush and abrasion damage state; and
- load-dependent suture-on-suture friction seeds.

Exact sources and the parameter each source informed are recorded in
`physics_next/sutures/dr-anmar-suture-4-0.json`.

## Needle connection

`assets/dr_anmar/needle/DrAnmarNeedle.usda` contains the independently authored
DrAnmar Needle and swage geometry. The runtime attaches the first suture
particle to the needle's swage-exit target. This constraint is stiffer than the
free strand, follows the needle during manipulation, and releases separately
when the configured pullout force is exceeded.

The connection models a factory-swaged transition, not a thread passed through
an eye. Its current 18 N pullout value is an engineering seed and requires
bench calibration.

## Build and qualify

Author and validate the portable asset:

```bash
./dr_anmar_suture_asset.sh rebuild
./dr_anmar_suture_asset.sh inspect
```

Run the promoted GPU qualification on an NVIDIA CUDA host:

```bash
DR_ANMAR_STABLE_ISAAC_PYTHON=/path/to/isaaclab/python \
./dr_anmar_suture_warp.sh qualify
```

The deterministic validator checks scale, material and source provenance,
layer isolation, render/collision correspondence, constitutive behavior,
runtime damage behavior, room integration, and the committed Warp GPU report.

## Sim-to-real boundary

GPU qualification is engineering evidence, not clinical validation. Promotion
to a calibrated medical model still requires:

- resolution and timestep convergence;
- straight and knotted tensile distributions across manufacturing lots;
- wet relaxation and friction measurements;
- driver crush, abrasion, and grasp-transfer testing;
- needle-swage pullout testing;
- tissue interaction and puncture calibration; and
- clinician validation under an approved protocol.

Do not radially rescale the asset: doing so invalidates its diameter, mass,
contact, stiffness, friction, and failure calibration.
