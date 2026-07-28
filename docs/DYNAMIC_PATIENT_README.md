# Dr.Anmar Dynamic Physiological Surgical Patient

A Dr.Anmar-owned interactive abdominal simulation-training patient for Isaac
Sim and Isaac Lab. It combines independently addressable anatomy, mixed
deformable representations, a solver-independent physiology runtime,
evidence-bound patient effects, and a unified procedure scene. OpenUSD, Isaac Lab, and
PhysX provide bounded runtime infrastructure; Dr.Anmar owns the patient,
workflow, state, authority, and evidence contracts.

The patient is available for simulation training, data generation, and
evaluation. It is not patient-specific; real-world correlation and clinical
evidence are not established. Its material, anatomical, physiological, fluid,
damage, and effect values are disclosed engineering parameters.

## Runtime contract

```python
patient.respiration
patient.perfusion
patient.bleeding
patient.vital_signs
patient.tissue_state
patient.organ_motion
patient.damage
patient.contact_effects
patient.patient_authority
patient.incision
patient.event_bus
patient.fluids
```

## Primary assets

- `dranmar_dynamic_abdominal_patient.usda`
- `dranmar_dynamic_abdominal_patient_rigid_proxy.usda`
- `dranmar_dynamic_abdominal_patient_operating_scene.usda`
- `anatomy/dranmar_laparotomy_wound.usda`
- `anatomy/` modular component assets
- `fluids/` blood, bile, urine, and irrigation carriers
- `anatomy_manifest.json`
- `physiology_network.json`
- `mechanics_contract.json`
- `robot_compatibility.json`
- `procedure_scenarios.json`

The extension exports the runtime from:

`orbit.surgical.assets.dynamic_abdominal_patient`

The access-state variant must be selected before deformable views initialize:

```python
spawn_patient("/World/Patient", access_state="open")
routes = apply_laparotomy_wound_deformables("/World/Patient")
capture_paths = capture_laparotomy_wound_edges(
    "/World/Patient",
    "/World/DrAnmarAtraumaticExposureTool",
)
```

The open variant composes five bilateral wound layers: skin, subcutaneous fat,
fascia, abdominal wall, and peritoneum. Each margin is an authored volume
TetMesh. The exposure tool captures six longitudinal cells per pad per layer,
for 60 distributed attachments across both full-thickness margins. The intact
variant hides this mechanics asset.

The complete implementation and limits are documented in
`DYNAMIC_PATIENT_LAPAROTOMY.md`.

## Current evidence boundary

The current revision was hardened by source inspection only. No test, build,
validator, import, simulator, release gate, or native evidence campaign was run.
Existing reports and prior 720-step scene records are historical and do not
qualify this modified source. Native workcell-to-patient providers remain
incomplete, so closure, sealing, division, patency, perfusion recovery, injury,
resuscitation, and clinical outcome are not established.
