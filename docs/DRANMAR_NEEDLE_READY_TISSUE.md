# DrAnmar Needle-Ready Tissue

The first geometry after physical needle handover is a layered, open-incision
tissue unit—not a rigid ring. Its authority is the `dr-assets` catalog package
at
`source/extensions/orbit.surgical.assets/data/Props/SurgicalTissue/NeedleReadyTissueUnit`.

## What is built

The geometry has two independently deformable wound flaps and three material
layers whose interfaces coincide with tetrahedral faces:

- fascia: deep load-bearing layer;
- bulk: compliant hydrated volume; and
- surface: primary needle-contact layer.

Every vertex carries a stable `(component, u, v, w)` coordinate. The asset also
authors outer attachment bands, wound edges, safe-bite regions, contact regions,
layer element sets, and layer-dependent fiber seeds. These fields are geometry
and curriculum labels; they are not clinical recommendations and they cannot
write task success.

## LOD strategy

| LOD | Points | Tetrahedra | Use |
| --- | ---: | ---: | --- |
| training | 560 | 1,944 | large-batch policy curriculum |
| contact | 2,470 | 10,368 | needle contact and intact deformation |
| validation | 16,650 | 82,944 | convergence and rendered inspection |

The point sets are exactly nested. State and diagnostic fields can therefore
move from training to contact to validation without nearest-surface ambiguity.
All three are generated deterministically from the same versioned contract.

## NVIDIA and open-source leverage

The runtime uses NVIDIA's current stack directly:

- OpenUSD `TetMesh` is the solver-neutral geometry representation;
- Isaac Lab `DeformableObjectCfg` discovers and spawns the canonical TetMesh;
- Newton VBD advances intact tissue;
- Warp supplies GPU kernels; and
- Newton `ModelBuilder.replicate()` batch-copies one pre-colored tissue
  template into isolated worlds.

That last point matters for efficiency. Repeatedly calling `add_soft_mesh()`
for every environment rebuilt surface-edge data against an ever-growing
builder and became the dominant setup cost. Replicating one colored template
is NVIDIA's native batched path and reduced the measured 1,200-instance
replication step to about 1.32 seconds.

Isaac Lab and the original ORBIT-Surgical compatibility foundation are
BSD-3-Clause. Newton, Warp, and the reviewed SoftMimicGen reference are
Apache-2.0. CRESSim-MPM is a BSD-3-Clause future topology-backend candidate.
No external tissue geometry, patient mesh, or texture is copied into the
asset.

## Measured Gilgamesh result

The isolated training-LOD capacity lane ran 2,400 simultaneous tissues on the
RTX 4090:

- 1,344,000 particles and 4,665,600 tetrahedra;
- 21.70 ms median and 21.72 ms p95 physics step;
- zero final inverted tetrahedra;
- 0.0544% final global volume error; and
- about 1.35 GiB measured GPU-memory delta.

This is roughly 110,600 tissue-environment frames/s. It excludes robot
articulations, rigid-soft contact, cameras, observations, PPO, and policy
networks, so it is a geometry/VBD capacity result—not a full-task throughput
claim.

The single-instance contact and validation lanes separately exercise fixture,
retraction, release, rigid-soft contact, deterministic replay, and the Isaac
Lab spawn/reset/step adapter.

## What remains unvalidated

Newton VBD currently qualifies intact deformation and contact. It does not
create a hole, persistent tract, cut, tear, or thread passage. We will not call
needle contact a puncture and will not use a policy-written phase flag as
evidence. T1 does not add a collision barrier: its training task ends when the
safe entry frame is armed, while its continuation task keeps stepping and
allows subsequent contact. Persistent puncture is routed to the pinned
CRESSim-MPM backend candidate rather than being faked inside VBD.

The post-handover progression is:

1. approach a sampled safe-bite entry frame while retaining the needle, with
   premature contact failing and post-arm contact allowed;
2. qualify calibrated pre-puncture indentation and unloading;
3. promote a topology-capable backend only after persistent-tract and
   force-depth gates pass; and
4. couple the already-qualified Warp suture only after tissue topology is
   real.

Material parameters remain research seeds pending synchronized physical
force/trajectory data. The asset is not clinically validated and is not for
patient care.

The machine-readable runtime and reward boundary is
[`config/dranmar_needle_ready_tissue.json`](../config/dranmar_needle_ready_tissue.json).
