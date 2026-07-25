# DrAnmar surgical oncology

DrAnmar's OncoSurgery Cell is a research integration for adaptive liver-tumor
resection and margin assurance. It connects an articulated 22-joint tool, a
registered 3-D tumor field, protected resection bonds, specimen handling,
multimodal sensing, and a contract-gated procedure episode.

It is not a clinical tumor model, a medical device, or a patient-care system.

## What is integrated

The catalog subtree is:

```text
Props/SurgicalOncology/OncoSurgeryCell
```

Its public OpenUSD interfaces are:

- `dranmar_oncosurgery_tool.usda` — standalone articulation;
- `dranmar_oncology_liver.usda` — registered liver and tumor task substrate;
- `dranmar_oncosurgery_workcell.usda` — mapping, resection, and assistant
  stations;
- `dranmar_tumor_resection_tool_payload.usda` — Franka hand replacement; and
- `dranmar_tumor_resection_tool_rigid_proxy.usda` — lower-cost rigid route.

The three public scene entry points use relative payload arcs. Their identity
and variants remain cheap to inspect, while heavy authored content can be
unloaded in a larger stage. This follows NVIDIA's scalable OpenUSD principles
of legibility, modularity, performance, and navigability.

The Python runtime is:

```text
orbit.surgical.assets.oncologic_resection
```

It exposes standalone, rigid-proxy, Franka, liver, specimen-bag, and workcell
Isaac Lab configurations. Isaac imports are lazy, so deterministic task and
safety tests run outside Kit.

## Executable mechanics

### Registered multimodal fusion

RGB/depth, NIR fluorescence, hyperspectral, ultrasound, OCT, and Raman samples
share one registration and timing contract. Fusion requires at least two
modalities and rejects:

- sample age above 250 ms;
- cross-modality skew above 50 ms;
- registration error above 3 mm;
- probability disagreement above 0.25; or
- fused confidence below 0.55.

This is an engineering abstention policy, not a diagnostic threshold. Portable
USD cameras can be attached at the four optical frames; ultrasound, OCT, and
Raman remain explicit host bridges or calibrated proxies.

### Tumor and margin state

`tumor_field.json` defines 3,028 registered volume cells:

- 2,790 healthy-parenchyma cells;
- 206 infiltrative-halo cells;
- 32 tumor-core cells; and
- 220 cells in the initial resection plan.

The episode computes resected volume, healthy tissue loss, removed and residual
modeled tumor volume, minimum margin, and fifth-percentile margin. Clearing
only the initial plan is not automatically called a successful margin: missed
satellites or an inadequate modeled boundary remain failures until corrected.

### Protected pedicles and detachment

`resection_topology.json` contains 96 boundary bonds: 79 parenchymal, five
fibrous, seven vascular, and five bile-duct pedicles. Each bond retains its
authored modality and work threshold. Protected pedicles reject division until
a bounded compression-and-energy seal is confirmed. The specimen detaches only
after all bonds are released and all 12 protected pedicles are sealed.

The thresholds are provisional simulation values. They are deliberately
fail-closed, but they are not validated electrosurgical prescriptions.

### Specimen and final result

The state machine requires bag deployment, capture of a detached specimen,
closure, and six orientation markers. The final report also requires a
registered cavity scan, correction of residual modeled tumor, hemostasis/bile
verification, blood loss at or below 5 ml, bile loss at or below 0.2 ml, no
protected-structure injury, and a modeled minimum margin of at least 10 mm.

## Isaac Lab training boundary

`OncologicResectionEpisode.observation()` exposes 12 normalized task terms:
phase, bond release, pedicle sealing, residual tumor, healthy-tissue loss,
margin, blood, bile, sensor disagreement, confidence, bag closure, and
orientation completeness. `reward()` shapes progress while retaining the final
result as the success authority.

`sample_domain_parameters()` supplies bounded reset-time perturbations for
registration, tissue stiffness, friction, sensor bias/dropout, and fluid-loss
scales. These correspond to Isaac Lab event-manager reset semantics; USD-level
startup randomization and tensor-level per-environment reset randomization must
remain distinct when scene replication is active.

The curriculum starts with a visible solitary lesion, then adds multifocal
pathology, registration noise, protected pedicles, modality dropout,
cross-sensor disagreement, positive-margin correction, and full specimen
handling.

## Dynamic patient integration

The packaged oncology liver is a registered topology and visual substrate. It
is deliberately not cooked as a thin-shell surface deformable: a closed solid
organ needs a volume model. `spawn_oncology_volume_liver()` instead reuses the
Dynamic Abdominal Patient liver's explicit tetrahedral simulation mesh and
applies the current NVIDIA PhysX GPU volume-deformable route. The runtime fails
closed if it gets a rigid, surface, or missing deformable route.

`activate_dynamic_patient_oncology_liver()` provides the corresponding
whole-patient path:

1. deactivate the demo liver;
2. activate exactly one Dynamic Patient liver volume deformable;
3. bind oncology state to the Dynamic Patient liver and tumor;
4. preserve the patient's vessel and gallbladder/bile paths;
5. use the shared blood and bile ledgers; and
6. retain the mapping, resection, and cavity-scan registration frames.

The deformable solver owns continuous nodal motion, collision, and contact.
It does not perform arbitrary live mesh cutting. The registered tumor-cell
field and 96-bond resection graph remain the explicit, deterministic authority
for irreversible removal, protected-pedicle interlocks, and specimen
detachment. This hybrid boundary follows NVIDIA's deformable guidance without
inflating the implementation into an unsupported topology-changing FEM claim.

The 18 kPa Young's-modulus, 0.47 Poisson-ratio, 1,060 kg/m3 density, damping,
and solver-iteration values are research seeds. Native CUDA stability is not
constitutive, tissue, or clinical calibration.

## Validation

Run the deterministic and OpenUSD gate locally:

```bash
/Users/home/.pyenv/shims/pytest -q tests/test_oncologic_resection.py
python3 scripts/validate_dranmar_oncologic_resection.py --require-usdchecker
```

Run both native representations through the Isaac Lab launcher:

```bash
python examples/validate_oncologic_resection_runtime.py \
  --representation standalone --device cuda:0 --headless
python examples/validate_oncologic_resection_runtime.py \
  --representation franka --device cuda:0 --headless
```

Native CUDA execution proves composition, articulation, joint targets,
registered cameras, a live volume-deformable tensor view, finite liver nodal
state, and finite simulator state for that exact runtime. Physical payload,
contact, constitutive tissue response, sensing, sealing, cutting, specimen,
margin, and clinical validity require separate evidence.

The recorded 25 July 2026 RTX 4090 stability lane passed with 274 live
tetrahedral nodes, 6.146729e-8 m maximum nodal displacement,
1.907188e-5 m/s maximum nodal speed, the expected 22-joint/23-body
articulation, the authored 2.5534 kg payload mass, and zero engine errors.
The liver was intentionally offset 0.45 m into a non-contact qualification
lane while the registered surgical target remained recorded in the report.
This isolates native volume-deformable stability from robot contact. It does
not qualify joint convergence, contact, rendered sensors, constitutive
response, the Franka payload, physical performance, or clinical use. The
machine-readable result is
`physics_next/benchmarks/dranmar-oncology-native-qualification.json`.
