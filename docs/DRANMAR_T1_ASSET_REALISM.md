# DrAnmar T1 Asset Realism Contract

T1 has two deliberately separate lanes:

- **Training** remains camera-free and RTX-free.
- **Visual qualification** is limited to four environments and uses visual
  overlays attached to the same live physics state.

Both lanes use the same task frames, object transforms, random seed, initial
state and physics authority. A visual overlay may improve shape readability,
UVs and material response. It may not change collision, contact, success,
object scale or task geometry.

## Current source-qualified assets

### Needle-ready tissue v2.1.0

The current tissue is a deterministic, layered tetrahedral asset with two
separate wound flaps. Its wound-side refinement was improved while preserving
the three LOD point and tetrahedron counts:

| LOD | points | tetrahedra | fixture nodes | minimum mean ratio | minimum scaled Jacobian | maximum edge ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| training | 560 | 1,944 | 80 | 0.340085 | 0.121905 | 7.371557 |
| contact | 2,470 | 10,368 | 380 | 0.507860 | 0.235509 | 3.814037 |
| validation | 16,650 | 82,944 | 1,998 | 0.505563 | 0.229784 | 3.829719 |

The authored `anchor_outer` sets are the fixture authority. The visual surface
is one-to-one, same-index geometry for the selected physics LOD; a detached
high-resolution surface and silent LOD substitution are forbidden.

The visual package uses 2K spatially varying color, roughness, normal and
subsurface-weight maps. OpenPBR 1.1 MaterialX is primary and
UsdPreviewSurface is the portable fallback. NVIDIA PhysicalAI SimReady
Materials v0.2.0 contributes an unchanged MIT-0 OpenPBR base and a bounded
skin micro-normal input; the tissue geometry and other textures are
deterministically authored by DrAnmar. Normal maps use quality-99, 4:4:4 JPEG;
other maps remain lossless PNG. Surface and wound roughness never drop below
0.58, coat weight is zero, and the support cassette is visual-only.

These are static source, topology and receipt qualifications. They are not
native deformable-motion, RTX appearance, biomechanical or clinical
qualifications.

### T1 robot, table and compatibility-needle visuals

The `Props/SurgicalScene/T1` package provides appearance-only overlays:

- The PSM overlay preserves articulation, joints, transforms and collision,
  removes dangling material bindings, and applies restrained satin metal and
  matte polymer responses.
- The table overlay preserves the legacy collision while adding UV-authored
  operating-table, pad and drape render geometry.
- The compatibility-needle overlay preserves the active legacy needle and its
  authored task scale of **0.4**.

The package is source-qualified for dependency closure, hashes, materials and
physics-layer parity. Native composition, RTX render review and live
collision-debug parity remain pending.

### Continuous T1 needle candidate

`NeedleT1Compatibility` is an inactive, unwired qualification candidate. It is
a single connected watertight mesh with 12,546 vertices and 25,088 triangles,
face-varying UVs, a 96-capsule continuous collider, a 25 micrometre tip seed,
and mesh-derived mass, centre of mass and inertia. Its satin texture has a
measured roughness range of **0.475–0.600**.

The candidate is authored for runtime spawn scale **1.0**. The active legacy
needle remains at scale **0.4**. Replacing only the USD path while retaining
the legacy scale is forbidden. Promotion must compose each asset with its own
declared scale and prove world-space tip, circle-centre, centreline-radius and
grasp-frame parity under the same task transform.

Runtime and JSON quaternions are `(x, y, z, w)` with identity
`(0, 0, 0, 1)`. OpenUSD quaternion serialization is `(w, x, y, z)`. The
contract names both forms explicitly; tuple copying across that boundary is
forbidden.

This candidate preserves the legacy 40 mm / 1.65 mm envelope for controller
parity, so it is not presented as a clinically representative needle. It
cannot replace the active asset until held-out pickup, single-arm retention,
mid-air transport and two-arm handover parity pass in the pinned native
runtime.

### PSM jaw-contact candidate

The jaw candidate is also inactive and unwired. It replaces the inherited
nonphysical friction ordering with an ordinary Coulomb research seed:
static friction 0.60, dynamic friction 0.45, zero adhesion, zero cohesion,
zero magnetism and zero suction. These are uncalibrated engineering
hypotheses, not a fix proven by success rate.

Activation requires paired held-out tests of the existing analytic pickup,
first-attempt and recovered retention, mid-air slip, two-arm handover,
per-jaw force/contact telemetry and sensitivity across the preregistered
friction envelope. It may not use attachment, force injection or contact-state
overrides.

### Low-complexity collider candidates

Two more inactive overlays separate render detail from bounded collision
geometry without changing root, link, joint, mass, inertia or material
authority:

- The PSM candidate replaces 288,052 enabled legacy collision-mesh triangles
  with 31 link-local primitives, only 48 of which are mesh triangles. Its
  source-level AABB support error is at most 0.20 mm.
- The table candidate replaces 127,622 legacy collision-mesh triangles with
  seven components and 28 mesh triangles. It makes only the pad-supported
  sterile field rigid; the hanging drape remains non-colliding.

Both packages use unit scale, inherit the existing root transform, retain
hash-locked external base and visual-overlay dependencies, and have no runtime
references. Their static approximation reports are not contact, penetration,
torque, load, stability or physics calibration evidence. Activation remains
blocked until native spawn, collision-debug, contact-stability and paired task
noninferiority gates pass.

## Rendering boundary

Reference capture targets RTX Path Tracing; interactive review targets RTX
Real-Time 2.0. The source contract specifies a recorded physical camera,
coaxial surgical lighting plus a soft fill, locked 4,500 K white balance,
manual exposure, ACES, scene-linear half-float EXR masters and transformed
sRGB review frames. Motion blur is disabled and depth of field may not hide
contact.

These are capture requirements, not completed evidence.

## Native promotion still required

The current topology changed from v2.0.0 to v2.1.0. Older Newton, Isaac Lab,
deterministic-replay and 2,400-environment receipts remain historical records;
their asset hashes do not match v2.1.0 and they do not transfer. The historical
contact run also reached 52.6 mm displacement, 8.03 m/s peak speed and
48.4 mm recovery residual, so it is not healthy calibrated-tissue evidence.

Promotion requires fresh, revision-bound evidence for:

- current-asset spawn and dependency closure in the pinned Isaac stack;
- live tissue visual synchronization and collision-debug parity;
- finite, stable and calibrated deformable response;
- native needle pickup, retention, transport and handover parity;
- native material bindings and realistic RTX captures;
- total 2,400-environment memory and throughput, only after explicit approval.

No native Isaac, RTX, Numi, current-topology Newton, puncture, damage,
biomechanical calibration, clinical validation or 2,400-environment result is
claimed by this source-only work. The machine-readable authority is
[`config/dranmar_t1_asset_quality.json`](../config/dranmar_t1_asset_quality.json).
