# Dr.Anmar 4-0 Surgical Suture Asset

The Dr.Anmar 4-0 suture is a separate, research-grade physics asset for
operating-room simulation. It does not modify, replace, reference, or inherit
the current NVIDIA SoftMimicGen thread.

## Research basis

The design is based on primary experimental and computational work, not a
generic rope preset:

- Pagnanelli et al. measured 4-0 polyglactin 910 at 7.6639 GPa over the first
  0–3% strain and showed that its stiffness and strength change in biological
  fluids.
- Horch et al. measured 24.22 N mean straight strength for 4-0 Vicryl and
  quantified progressive loss after repeated needle-holder crush cycles.
- Bariol et al. found 42.4 N and 26.3% extension for undamaged 3-0 Vicryl and
  demonstrated damage after a 45 MPa, one-second laparoscopic instrument grasp.
- Wang et al. found that relaxation is greatest early, and that suture-on-suture
  friction force increases with normal load while remaining effectively
  velocity-independent over their test range.
- Stasiak et al. measured about 59% retained stress for wet braided PGA after
  long-duration relaxation, supporting a two-time-constant wet model.
- Savage et al. showed that repeated bending and abrasion are distinct suture
  fatigue modes rather than cosmetic effects.
- Tera and Åberg demonstrated that knot efficiency depends jointly on knot,
  material, and gauge; it cannot be one universal constant.
- Baek et al. showed that tight knots require full three-dimensional
  deformation, self-contact, and Coulomb friction rather than a loose 1-D
  centerline approximation.
- Karthikeyan et al. measured S-shaped tensile response and inelastic load
  drops in knotted and looped braided Vicryl.

The exact source URLs, identifiers, and the parameter each informed are stored
in the profile JSON alongside the asset.

## What was built

The asset is authored at its real 0.25 mm diameter and 180 mm length. It is a
360-element discrete Cosserat rod: every visible capsule is also the exact
collision body. Breakable D6 joints independently represent axial stiffness,
bending compliance, torsion, damping, limited extension, and overload failure.
There is no enlarged collision proxy and no rendered curve that can diverge
from physics.

The design includes:

- 4-0 coated, braided-polyglactin-equivalent geometry and density;
- physical micrometre-scale radius modulation to approximate braid texture;
- non-adjacent self-contact for loops and knots, with adjacent bodies filtered;
- Coulomb contact friction and a sidecar load-dependent self-friction model;
- a 20–25 N straight failure envelope and reduced knotted strength;
- wet stress relaxation, cyclic hysteresis seed, crush damage, and abrasion
  state in the constitutive contract;
- automatic per-joint overload failure;
- a tapered 3 mm swage transition with greater axial and bending stiffness;
- a replaceable needle interface whose pullout joint fails separately.

The central invention is the representation itself. A conventional isotropic
FEM rope ties axial and bending stiffness together and cannot independently
match a thin braided suture's high tensile stiffness and soft handling. This
asset instead uses a discrete Cosserat rod: axial, flexural, and torsional
responses are independently calibrated at every 0.5 mm element. It also stores
damage locally, so crushing one section or compacting one knot does not
incorrectly weaken or relax the whole strand.

The profile and derived mechanics live in
`physics_next/sutures/dr-anmar-suture-4-0.json`. The committed OpenUSD asset is
`assets/dr_anmar/suture/DrAnmarSuture4_0.usda`.

The entry file is intentionally lightweight. It composes a binary USDC capsule
geometry layer, a visual look-development layer, an engine-neutral physics
layer, and a PhysX-only tuning layer. Geometry contains no material or physics
opinions; the neutral layer contains no PhysX or Newton schemas; and the PhysX
layer owns hybrid sweep/speculative CCD, solver iterations, damping,
friction-combine modes, and radius-scaled contact offsets. This follows the
NVIDIA USD Asset Structure pattern while preserving the original prim paths
used by the runtime.

## DrAnmar Needle

`assets/dr_anmar/needle/DrAnmarNeedle.usda` is the independently authored
DrAnmar Needle, not a renamed or inherited needle mesh. Its 22 mm half-circle
centerline, taper point, swage transition, 2,049-vertex visual mesh, 40-part
compound collision shape, mass, material, and solver settings are generated
from `physics_next/needles/dr-anmar-needle-v1.json`. A fixed factory-swage joint
connects the needle to the replaceable suture interface; the first suture joint
retains the separate pullout failure limit.

Every locally constructed procedure room receives this additional instrument
without replacing the room's existing task object or current thread.

## Sim-to-real boundary

The live workstation deterministically randomizes needle mass, contact
friction, restitution, and surface roughness from bounded ranges on every
scenario reset. The scenario seed exactly replays the sampled domain, and the
sampled values are exposed in workstation state and recordings.

The needle profile also carries a machine-readable gap register and fail-closed
qualification gates for manufacturing metrology, bend and yield, driver slip,
tissue puncture, swage pullout, endoscopic perception, and numerical contact.
The current needle is intentionally rigid: recoverable bending, permanent set,
and needle breakage remain explicit model gaps until physical bend tests can
identify them. Clinical use remains blocked until independent validation under
an approved protocol.

## Build and validate

```bash
./dr_anmar_suture_asset.sh rebuild
./dr_anmar_suture_asset.sh inspect
./dr_anmar_suture_asset.sh physics-probe
```

`rebuild` is platform-independent and deterministically authors the asset,
validates the constitutive contract, and tests the stateful damage runtime.
`inspect` uses the Isaac/Kit OpenUSD runtime to parse and count the authored
prims. `physics-probe` runs the complete 360-body, 360-joint asset under native
PhysX gravity and fails if the strand is absent, static, non-finite, or unstable.

The recommended simulation step is 0.5 ms (16 substeps per 120 Hz rendered
frame). Hybrid linear and angular continuous collision detection is enabled
because a 0.25 mm strand can otherwise tunnel through fast instruments or thin
tissue.

## Integration contract

Reference the asset's default prim, transform it into the operative pose, and
replace the `NeedleInterface` kinematic test role with a fixed attachment to the
rigid needle's swage body. The interface and first segment are connected by the
separately breakable swage joint `J0000`. Do not radially rescale the asset:
changing its scale invalidates the 4-0 diameter, mass, stiffness, friction, and
failure calibration.

Instrument software can record local crush and abrasion events through the
profile's damage curve. `scripts/dr_anmar_suture_runtime.py` turns those events
into live per-joint break force, break torque, drive-force ceiling, and relaxed
axial stiffness. Tight-knot compaction applies the configured 0.50–0.80
efficiency range locally rather than weakening the entire strand. The base
asset supplies the self-contact, flexural mechanics, friction, and breakable
joints consumed by that runtime.

## Validation boundary

The deterministic validator proves scale, density, constitutive targets,
relaxation shape, friction behavior, crush damage, swage structure, breakable
joint coverage, physical/render identity, source provenance, and isolation from
the current thread. It does not prove clinical fidelity.

Before this is treated as a medical model, it still requires physical tests for
straight tension, flexural rigidity, wet relaxation, load-dependent capstan
friction, instrument damage, abrasion fatigue, square/surgeon's knot security,
and needle pullout, followed by clinician validation.
