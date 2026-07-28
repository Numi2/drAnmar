# Dr.Anmar median laparotomy

The open-patient variant now represents a centered median laparotomy as two
full-thickness wound margins. There is no loose central tissue plug. The
camera-facing operative corridor remains open while the real Dr.Anmar
atraumatic exposure tool approaches from above and retracts the margins
laterally.

## Authored mechanics

`anatomy/dranmar_laparotomy_wound.usda` contains five bilateral layers:

- skin
- subcutaneous fat
- fascia
- abdominal wall
- peritoneum

Each of the ten margins is an explicit volume-deformable hierarchy with an
authored TetMesh and bound visual mesh. Its outer longitudinal band is attached
to the patient frame. Its inner band is divided into six capture regions.
The current runtime does not have an exact prim-bound provider for qualifying
those cells, so operational capture authoring is disabled. The camera-only
fixture can author up to 60 distributed bonds, but those bonds are not evidence
of autonomous grasping or task completion.

This uses the relevant NVIDIA mechanics contracts only:

- volume-deformable Xform hierarchy with authored tetrahedral simulation state
- deformable-pose binding for the visible mesh
- vertex-to-Xform attachments for tissue-to-tool coupling
- explicit patient-internal collision filtering

References:

- https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/deformables/omniphysics_deformable_schema.html
- https://docs.omniverse.nvidia.com/kit/docs/omni_physics/109.0/dev_guide/deformables/deformable_bodies.html
- https://nvidia-omniverse.github.io/PhysX/ovphysx/latest/simulation_setup/deformables.html

## Camera-fixture API

Select the open variant before initializing deformables, then opt into the
non-qualifying camera fixture explicitly:

```python
spawn_patient("/World/Patient", access_state="open")
routes = apply_laparotomy_wound_deformables("/World/Patient")
attachments = capture_laparotomy_wound_edges(
    "/World/Patient",
    "/World/DrAnmarAtraumaticExposureTool",
    prepositioned_fixture=True,
)
```

`patient.wound_grasp.observe(...)` and caller-supplied `qualified_cells` now
fail closed. The retained authored-sample grasp model is a private engineering
fixture, not runtime evidence authority. Promotion requires a native provider
that registers the exact pad, tissue, contact-pair, attachment, raw-record,
episode, environment, step, time, and topology identities.

To release the margins without deleting anatomy:

```python
released = release_laparotomy_wound_edges("/World/Patient")
```

The complete camera demonstration is:

```bash
./dr_anmar.sh laparotomy
```

That scene opts into `prepositioned_fixture=True` so it can preserve the
unobstructed camera composition. It demonstrates deformable response and
distributed attachment motion; it does not qualify autonomous wound-edge
grasping.

For a bounded headless execution and PNG capture:

```bash
./dr_anmar.sh laparotomy 720 /tmp/dranmar-laparotomy.png
```

The asset is deterministically regenerated with:

```bash
python scripts/generate_dranmar_laparotomy_wound.py
```

## Incision boundary

The private incision engineering fixture models a median path through skin,
subcutaneous tissue, linea-alba fascia, and peritoneum. The rectus muscles are
separated at the midline and are not recorded as transected. Its provisional
gates cover force, alignment, pose-derived speed, monotonic travel, lateral
offset, depth, and cutting work. Public `patient.incision.advance(...)` is
disabled because those scalar inputs are caller-authored rather than exact
prim-bound SceneEvidence; it cannot release continuity or mutate patient state.

That controller does not mutate TetMesh topology during simulation. The current
physical scene starts from the pre-segmented open variant. Arbitrary fracture,
instrumented puncture, and live remeshing are not claimed.

The anatomical sequence follows the WSES description of a midline incision
through skin, subcutaneous tissue, linea alba, and peritoneum, and the NCBI
technique description that muscle should not be encountered in the avascular
linea-alba plane:

- https://pmc.ncbi.nlm.nih.gov/articles/PMC10373269/
- https://www.ncbi.nlm.nih.gov/books/NBK525961/

## Parameter boundary

Layer materials, damping, friction, attachment density, and incision gates are
provisional research parameters. They have not been fitted to instrumented
human abdominal tissue or to the Dr.Anmar grasper. A successful Isaac run is
execution evidence, not constitutive, injury, clinical, or medical-device
validation.
