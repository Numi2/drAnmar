# Dr.Anmar physics-next

This directory is the versioned contract for Dr.Anmar's native multi-solver
surgical physics. The stable doctor-facing Isaac Sim worker is the PhysX rigid
lane; Dr.Anmar never substitutes a browser or Python mechanics model for a
missing simulator capability.

The stable workstation reports the backend that actually produced its state.
Its first promoted deformable room imports the patient liver surface through
NVIDIA's mesh converter and lets native PhysX cook the simulation tetrahedra.
The explicit OpenUSD `TetMesh` is retained for Isaac Sim 6 / Isaac Lab 3 and
Newton work. Newton VBD and CRESSim-MPM remain isolated until they are connected
as workers with every native capability required by a room.

## Authority ladder

1. `physx_rigid`: robot articulation, rigid objects, contacts and sensors.
2. `physx_fem`: intact volumetric tissue, native contact and stress telemetry.
3. `newton_vbd`: high-throughput two-way deformable comparison and policy work.
4. `cressim_mpm`: topology-changing cutting and puncture research.

OpenUSD remains the common scene and visual-asset layer. A canonical organ has
separate render, collision and simulation representations plus explicit mapping,
attachments, material regions, calibration provenance and an optional vascular
graph.

Run `python scripts/dr_anmar_physics_next.py validate` for the fail-closed
contract check. Use `./dr_anmar_physics_next.sh status` on Gilgamesh to inspect
the isolated runtime without touching the live Dr.Anmar suite.

After installation, `./dr_anmar_physics_next.sh benchmark physx` and
`./dr_anmar_physics_next.sh benchmark newton` execute the same material seed,
attachment pattern, retraction trajectory and telemetry collection on both
backends. The first procedural tissue coupon is deliberately a solver benchmark.
The patient-specific liver has two solver-native OpenUSD representations: a
surface mesh cooked by the live PhysX worker and an explicit `TetMesh` for the
new Omni Physics API.
Newton executes kitlessly and records a paired exact-state replay. The PhysX
command requires the operator to accept NVIDIA's Omniverse Kit EULA explicitly;
the wrapper never accepts that agreement or hides the resulting activation gate.
`./dr_anmar_physics_next.sh compare RESULT_JSON ...` evaluates results against
the versioned contract and records an unmeasured gate as incomplete, never as a
pass.

`./dr_anmar_physics_next.sh patient-liver-smoke` loads the authored patient
liver NPZ into kitless Newton VBD, applies a small geometric fixture/retraction,
and records finite state, timing, volume and inversion evidence. This command is
explicitly a patient-asset solver-integration smoke: its coordinate-selected
fixture bands are not anatomical boundary conditions and cannot promote the
asset. The default 8-iteration canonical coupon is separate from the lighter
2-iteration patient-mesh smoke.

`scripts/dr_anmar_physics_asset_prepare.py` extracts the normalized patient
surface and reports boundary/non-manifold edges before meshing. It never repairs
or tetrahedralizes silently; the extraction manifest and hashes become the
provenance input to the promoted canonical asset.

`scripts/dr_anmar_tetrahedralize.py` consumes that hashed extraction, refuses
surfaces with recorded topology defects, runs fTetWild at an explicit physical
edge length, computes element and render-mapping evidence, and writes NPZ plus
VTK representations. Attachment regions remain deliberately pending anatomical
review instead of being guessed from mesh coordinates.
The wrapper commands `extract-liver` and `tetrahedralize-liver` reproduce this
pipeline; tetrahedralization defaults to one thread so parallel meshing order
does not silently change the candidate artifact.
`author-liver-usd` then writes one right-handed `UsdGeom.TetMesh` plus its
surface visualization. PhysX FEM and Newton VBD consume that same volume,
preventing backend-specific remeshing from corrupting comparisons.
