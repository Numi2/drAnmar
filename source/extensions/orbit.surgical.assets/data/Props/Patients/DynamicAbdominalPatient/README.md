# Dynamic abdominal patient

The dynamic abdominal patient is a layered, scenario-selectable OpenUSD family
for Dr.Anmar operating-room research. The catalog root contains the composed
patient and room entrypoints; `anatomy/`, `fluids/`, `glb/`, and `textures/`
hold their local dependencies.

`asset_manifest.json`, `anatomy_manifest.json`, `mechanics_contract.json`, and
`patient_runtime_contract.json` define identity, component routing, mechanics,
and runtime limits. The `open` patient variant composes a centered median
laparotomy with ten explicit TetMesh bodies: bilateral skin, fat, fascia,
abdominal-wall, and peritoneal wound margins. The `intact` variant hides those
mechanics.

The wound margins can be attached to the Dr.Anmar atraumatic exposure tool
through distributed vertex-to-Xform attachments. Material and force values are
provisional engineering parameters, not calibrated clinical thresholds.

This asset is not a validated patient model, medical device, diagnostic
system, treatment planner, or clinical outcome predictor.
