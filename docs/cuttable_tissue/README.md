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
