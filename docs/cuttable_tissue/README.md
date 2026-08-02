# DrAnmar cuttable-tissue development lane

This branch begins with the intact-contact prerequisite for arbitrary scalpel
cutting. It does not yet claim that tissue can be cut.

The CPU reference solver provides a deterministic Total-Lagrangian tetrahedral
coupon with compressible Neo-Hookean stress, one-term Prony relaxation,
pre-tensioned fixed boundary bands, and two-way contact against a finite rounded
scalpel edge. The qualification presses, holds, slides, and retracts that edge.
Fracture is explicitly disabled and fails closed.

Scalpel contact projects blade-attached edge samples into the current
piecewise-linear tissue surface, integrates an analytic rounded-edge contact
strip, and distributes force through triangle barycentric coordinates. An
off-grid sweep must retain contact across a complete mesh cell; node-only
collision is not accepted.

Run the reference qualification with:

```bash
python scripts/qualify_dranmar_cuttable_tissue.py \
  --output physics_next/receipts/cuttable-tissue-intact-contact-reference.json
```

The receipt gates finite state, element inversion, minimum Jacobian, anchor
drift, volume change, contact penetration, force relaxation, elastic recovery,
zero fracture events, and deterministic replay. It is simulator-engineering
evidence only; the parameters are not calibrated human-skin properties.

The next stage must preserve this receipt while replacing the CPU reference
with a GPU backend. `dr_anmar_cuttable_tissue_warp.py` now carries Warp kernels
for the same Neo-Hookean/Prony internal force and projected scalpel contact. Its
retained CPU-device parity receipt validates the kernel equations but leaves
CUDA promotion pending. Run it inside an isolated Warp/Isaac runtime with:

```bash
python scripts/dr_anmar_cuttable_tissue_warp.py --device cuda:0 \
  --output physics_next/receipts/cuttable-tissue-warp-cuda-parity.json
```

Only after CUDA parity and the full dynamic receipt pass should fracture be
enabled using a blade-seeded dynamic front, cohesive fracture work,
adhesion/wear, rate-dependent separation, and persistent two-sided collision
surfaces.

## Cohesive fracture reference

The next constitutive/topology oracle is now available:

```bash
python scripts/qualify_dranmar_cohesive_fracture.py
```

It assigns an eligible cohesive interface to every internal tetrahedral face,
rather than authoring a few permitted cut locations. A finite swept scalpel
edge may seed only interfaces connected to the top entry surface or the
existing cut front. Separation then follows an irreversible bilinear
mixed-mode law with fracture work, Benzeggagh–Kenane mode mixing, viscous rate
strengthening, and an independent compression response after complete failure.

The retained reference receipt qualifies all 1,000 internal interfaces, exact
deterministic replay, off-grid blade-entry coverage, Mode I/II/mixed-mode
energy closure, damage irreversibility, and post-failure compression. This is
not yet a dynamic cut: the intact FEM still fails closed while fracture is
enabled. The next runtime stage must give adjacent tetrahedra discontinuous
degrees of freedom, generate persistent two-sided collision surfaces, preserve
mass, and demonstrate mesh-refinement convergence on CUDA.

The cohesive law also has an independent Warp implementation. Its retained CPU
parity receipt covers intact loading, softening, mixed mode, rate strengthening,
unseeded overload, and post-failure compression:

```bash
python scripts/dr_anmar_cohesive_fracture_warp.py --device cuda:0 \
  --output physics_next/receipts/cuttable-tissue-cohesive-warp-cuda-parity.json
```

CPU parity is not CUDA qualification and does not enable runtime fracture.

The architecture follows the experimentally informed separation of bulk large
deformation, cohesive fracture, rate-dependent dissipation, adhesion/wear, and
tool contact described in [Moreno-Mateos and Steinmann
(2026)](https://doi.org/10.1038/s41524-025-01869-y). Parameters in this profile
remain provisional until matched to measurements from the exact tissue analog;
they must not be presented as human-skin or clinical validation.

## Persistent arbitrary topology

The cut-cell topology reference is qualified with:

```bash
python scripts/qualify_dranmar_persistent_cut_topology.py
```

The 48×32×12 field stores multiple oriented discontinuity patches per cell.
It does not delete voxels or remove tissue volume. Fracture work alone advances
damage; adhesion, wear, viscous dissipation, and Coulomb friction are retained
as separate auditable channels. This prevents a large friction impulse from
being mislabeled as material fracture.

Each fractured patch is clipped against its cell and reconstructed twice with
opposed normals. These paired zero-volume sheets are the persistent wound
surfaces and collision source for the later discontinuous FEM runtime. Multiple
patches in one cell permit curved and intersecting incisions; replaying an
existing incision is topology-idempotent.

The retained reference covers 18,432 field cells, nine arbitrary entry origins,
a curved incision, a repeated incision, and a crossing incision. It produced
72 intersection cells, exactly equal positive/negative wound area, zero removed
volume, complete wound-collision sampling, and deterministic replay. These are
topology/reference results—not yet proof that the dynamic FEM opens and collides
correctly under a robot-held scalpel.
