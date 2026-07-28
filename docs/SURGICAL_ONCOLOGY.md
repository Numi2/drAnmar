# DrAnmar surgical oncology

DrAnmar's OncoSurgery Cell is a research asset package for adaptive
liver-tumor resection and margin-assurance work. It packages an articulated
22-joint tool, a registered 3-D tumor field, resection-bond and specimen task
proxies, and sensor integration requirements. It does not currently connect
those proxies to authoritative patient outcomes.

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

## Task proxies and evidence ceiling

### Registered multimodal fusion

The private fusion proxy describes RGB/depth, NIR fluorescence, hyperspectral,
ultrasound, OCT, and Raman registration/timing thresholds. Its provisional
algorithm rejects:

- sample age above 250 ms;
- cross-modality skew above 50 ms;
- registration error above 3 mm;
- probability disagreement above 0.25; or
- fused confidence below 0.55.

There is no workcell-owned sensor adapter, raw-sample identity, common
post-physics envelope, or calibrated modality bridge. Public fusion therefore
always abstains. Portable USD camera prims can be authored at four optical
frames, but cameras are not annotators or outcome evidence; ultrasound, OCT,
and Raman are frames/contracts only.

### Tumor and margin state

`tumor_field.json` defines 3,028 registered volume cells:

- 2,790 healthy-parenchyma cells;
- 206 infiltrative-halo cells;
- 32 tumor-core cells; and
- 220 cells in the initial resection plan.

Private task-proxy code can compute resected volume, healthy tissue loss,
removed and residual modeled tumor volume, and grid-distance margin summaries.
The roughly 9-10 mm cell spacing, full-voxel accounting, and uncalibrated field
make these curriculum features, not geometric, pathological, or clinical
margin evidence.

### Protected pedicles and detachment

`resection_topology.json` contains 96 task-proxy boundary bonds: 79
parenchymal, five fibrous, seven vascular, and five bile-duct pedicles. Each
bond retains an authored modality and work threshold. No bond is connected to
same-step tool contact, tissue/cohesive mechanics, vessel or duct response,
thermal dose, or patient loss ledgers. Public seal, release, injury, and
detachment mutation therefore fail closed.

The private thresholds are provisional curriculum values, not validated
electrosurgical prescriptions.

### Specimen and final result

The specimen bag is a rigid open/closed visual proxy without collision,
deformation, attachment, or containment mechanics. The former task state
tracks deployment, capture, closure, and six orientation strings privately,
but no public final patient result is available. Reports expose explicitly
non-authoritative task-proxy diagnostics and always return `success: false`.

## Isaac Lab training boundary

`OncologicResectionEpisode.observation()`, `reward()`, phase progression, and
finalization fail closed until the oncology scene-evidence bridge exists. The
former 12-term observation and dense reward remain private task-proxy
algorithms only.

`sample_domain_parameters()` returns bounded proposals for registration,
tissue stiffness, friction, sensor bias/dropout, and fluid-loss scales. No
source applies them to the scene, mechanics, sensors, or patient ledgers yet.

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
3. return a manifest of required oncology prim paths and frames.

The function does not bind oncology state to the liver/tumor, vessel or duct
mechanics, shared blood/bile ledgers, or scene evidence. The deformable solver
can own continuous nodal motion and collision, but it does not perform
arbitrary live mesh cutting. The tumor-cell field and 96-bond graph remain
non-authoritative task proxies until a workcell integration establishes those
connections.

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

The recorded native CUDA evidence proves composition, articulation discovery,
a one-step finite setpoint sweep, a live volume-deformable tensor view, and
bounded non-contact liver state for that exact runtime. Physical payload,
contact, joint convergence, rendered sensing, constitutive tissue response,
sealing, cutting, specimen containment, margins, and clinical validity require
separate evidence.

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
`physics_next/benchmarks/dranmar-oncology-native-simulator-evidence.json`.
