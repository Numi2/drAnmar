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
patient.robot
patient.event_bus
patient.fluids
```

## Primary assets

- `dranmar_dynamic_abdominal_patient.usda`
- `dranmar_dynamic_abdominal_patient_rigid_proxy.usda`
- `dranmar_dynamic_abdominal_patient_operating_scene.usda`
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
routes = apply_patient_deformables(
    "/World/Patient",
    include=("peritoneum",),
)
```

Any `not_applied` mechanics route is a failed native-runtime gate, not a
successful fallback. The current guarded workstation boundary permits exactly
one explicitly selected solver-active surface component. Omitting `include`
is useful for route inspection, but it requests all component mechanics and is
not an approved workstation configuration.

## Validation

The physiology and contract regression suite has no geometry dependencies:

```bash
pytest -q tests/test_dynamic_abdominal_patient.py
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
./isaaclab.sh -p examples/dynamic_abdominal_patient_scene.py --device cuda:0
```

That run must load the OpenUSD stage, create the explicitly requested
deformable route, step on CUDA, and remain finite before that route can be
promoted. Qualification of one route does not qualify other organs or a
multi-deformable patient.
