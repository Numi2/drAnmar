# DrAnmar laparotomy sponge

DrAnmar ships a self-contained, Apache-2.0 laparotomy-sponge component at the
I4H-compatible catalog subpath:

```text
Props/SurgicalCount/LaparotomySponge
```

The asset is category-level research content. It is not a manufacturer-exact
digital twin, clinically validated, physics-calibrated, or approved for
patient-care use.

## Representations

| Representation | Geometry | Intended work |
| --- | ---: | --- |
| `lap_sponge_unfolded.usda` | 1,027 points, 1,868 triangles, one connected component | Surface-deformable folding, contact, retrieval-loop interaction and synthetic data |
| `lap_sponge_folded_proxy.usda` | 1,026 body points, 2,048 body triangles | Perception, grasping, handover, tray placement and count-task development |

Both representations expose coordinated `dry` and `wet` state variants. The
X-ray-marker region is visible and semantically identified, but radiographic
attenuation is not simulated. The retrieval loop is part of the unfolded
connected surface and is covered by 32 capsule colliders on the folded proxy.

## Runtime selection

| Route | Selected runtime | Evidence |
| --- | --- | --- |
| Folded rigid proxy | Isaac Sim 5.1.0.0, Isaac Lab 2.3.2, CUDA 12.8 | Dry and wet each completed 240 RTX 4090 steps with finite state |
| Unfolded surface | Isaac Sim 6.0.1.0, Isaac Lab 3.0.0 beta 2, Omni Physics 110.1.13, CUDA 12.8 | Runtime authoring and view path delivered; execution record awaits explicit operator activation of the installed Kit licence |

The unfolded route is not disabled while physical measurements are missing.
The runtime helper cooks the triangular simulation mesh, binds one effective
surface material, enables self-collision, and creates an Isaac Lab deformable
view. Isaac Lab 2.3's legacy deformable view is volume-only, so the folded proxy
is the 5.1 route and the current surface path uses the isolated 6.0 lane.

## Provisional parameter calibration

The geometry envelope uses the current McKesson 16-2118181 18 x 18 inch,
100%-cotton, four-ply category specification, supplemented by the Medline
MDS251518LF seven-inch-loop category specification. Those public specifications
do not publish mass, thickness, modulus, damping, or friction, so the following
remain solver baselines rather than measurements.

| Parameter | Dry | Wet |
| --- | ---: | ---: |
| Mass | 0.022 kg | 0.120 kg |
| Effective solver thickness | 0.004 m | 0.004 m |
| Effective density | 26.163188532 kg/m³ | 142.708301081 kg/m³ |
| Young's modulus | 100,000 Pa | 60,000 Pa |
| Poisson's ratio | 0.35 | 0.35 |
| Dynamic friction | 0.65 | 0.55 |
| Elasticity damping | 0.10 | 0.18 |
| Bend damping | 0.10 | 0.18 |

Density is computed from each target mass, the exact connected
0.210219025612 m² solver area, and effective thickness. The explicit bend
override is zero, which selects the runtime's thickness-aware derivation. The
folded proxy additionally binds dry/wet rigid physics materials with static
friction 0.75/0.65, matching dynamic friction 0.65/0.55, and zero restitution.

## Collision coverage

The folded body visual envelope is 140 x 114 x 24.96 mm. Its collider is
138 x 112 x 24.96 mm, centered 0.48 mm above the asset origin. This provides
full thickness coverage and 98.57%/98.25% lateral coverage; the deliberate
1 mm per-side inset avoids corner ghost contacts. The 3.0 mm visual loop uses
32 contiguous 3.2 mm-radius capsules, giving 106.67% radial coverage.

Run the deterministic repository inspection with:

```bash
python scripts/dr_anmar_laparotomy_sponge_validate.py \
  --output /tmp/dr-anmar-laparotomy-sponge-validation.json
```

Run either CUDA representation in its selected Isaac environment with:

```bash
python scripts/dr_anmar_laparotomy_sponge_smoke.py \
  --headless --device cuda:0 --representation rigid --state dry

python scripts/dr_anmar_laparotomy_sponge_smoke.py \
  --headless --device cuda:0 --representation surface --state wet
```

## Runtime integration

```python
from orbit.surgical.assets import (
    SurgicalCountAssets,
    apply_surface_deformable,
    make_rigid_proxy_cfg,
    spawn_unfolded_reference,
)
```

`SurgicalCountAssets` uses the same relative strings accepted by
`BaseI4HAssets` and `i4h-asset-retrieve --sub-path`. Local DrAnmar rooms resolve
the same paths from the `orbit.surgical.assets` extension data root.

The authoritative profile is
`physics_next/surgical-count/dr-anmar-laparotomy-sponge-v1.json`; executed CUDA
evidence is recorded in
`physics_next/benchmarks/dr-anmar-laparotomy-sponge-validation.json`.
