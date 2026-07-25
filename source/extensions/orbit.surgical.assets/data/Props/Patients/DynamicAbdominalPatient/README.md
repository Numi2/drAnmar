# Dynamic abdominal patient

The dynamic abdominal patient is a layered, scenario-selectable OpenUSD family
for Dr.Anmar operating-room research. The catalog root contains the composed
patient and room entrypoints; `anatomy/`, `fluids/`, `glb/`, and `textures/`
hold their local dependencies.

`asset_manifest.json`, `anatomy_manifest.json`, `mechanics_contract.json`, and
`patient_runtime_contract.json` define identity, component routing, mechanics,
and runtime limits. Only one solver-active deformable is enabled by default.
Surface and volume lanes, remeshing, incisions, and procedure-specific changes
must retain their separate qualification evidence.

This asset is not a validated patient model, medical device, diagnostic
system, treatment planner, or clinical outcome predictor.
