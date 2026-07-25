# Dr.Anmar Dynamic Physiological Surgical Patient

A Dr.Anmar-owned modular abdominal research patient for Isaac Sim and Isaac Lab. It combines independently addressable anatomy, mixed deformable representations, a solver-independent physiology runtime, intervention adapters, and a unified procedure scene. OpenUSD, Isaac Lab, and PhysX provide bounded runtime infrastructure; Dr.Anmar owns the patient, workflow, state, robot-adapter, evidence, and qualification contracts.

The patient is research-only. It is not patient-specific, clinically validated, or approved for patient care. Its material, anatomical, physiological, fluid, damage, and intervention values are provisional engineering parameters.

## Runtime contract

```python
patient.respiration
patient.perfusion
patient.bleeding
patient.vital_signs
patient.tissue_state
patient.organ_motion
patient.damage
patient.interventions
patient.incision
patient.robot
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

## Validation

The physiology and contract regression suite has no geometry dependencies:

```bash
pytest -q tests/test_dynamic_abdominal_patient.py
pytest -q tests/test_dynamic_patient_laparotomy_asset.py
pytest -q tests/test_dynamic_patient_laparotomy_incision.py
python examples/end_to_end_procedure.py
```

The complete geometry/source validator uses pinned, explicit dependencies:

```bash
python -m pip install -r scripts/requirements_dynamic_abdominal_patient_validation.txt
python scripts/validate_dranmar_dynamic_abdominal_patient.py
```

The generated report is written to:

`physics_next/benchmarks/dranmar-dynamic-abdominal-patient-validation.json`

Run the native scene only from the repository's Isaac Lab environment:

```bash
./dr_anmar.sh laparotomy
```

For a bounded headless run with an RTX frame:

```bash
./dr_anmar.sh laparotomy 720 /tmp/dranmar-laparotomy.png
```

Successful execution demonstrates that the authored scene ran in that exact
software and hardware environment. It is not constitutive, grasp-force,
incision, clinical, or medical-device validation.
