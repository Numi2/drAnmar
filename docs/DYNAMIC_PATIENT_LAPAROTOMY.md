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
to the patient frame. Its inner band is divided into six capture regions, which
are attached to the matching exposure-pad collision cells. This produces 60
distributed tool bonds across the full wound depth.

This uses the relevant NVIDIA mechanics contracts only:

- volume-deformable Xform hierarchy with authored tetrahedral simulation state
- deformable-pose binding for the visible mesh
- vertex-to-Xform attachments for tissue-to-tool coupling
- explicit patient-internal collision filtering

References:

- https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/deformables/omniphysics_deformable_schema.html
- https://docs.omniverse.nvidia.com/kit/docs/omni_physics/109.0/dev_guide/deformables/deformable_bodies.html
- https://nvidia-omniverse.github.io/PhysX/ovphysx/latest/simulation_setup/deformables.html

## Runtime API

Select the open variant before initializing deformables:

```python
spawn_patient("/World/Patient", access_state="open")
routes = apply_laparotomy_wound_deformables("/World/Patient")
attachments = capture_laparotomy_wound_edges(
    "/World/Patient",
    "/World/DrAnmarAtraumaticExposureTool",
)
```

To release the margins without deleting anatomy:

```python
released = release_laparotomy_wound_edges("/World/Patient")
```

The complete interactive scene is:

```bash
./dr_anmar.sh laparotomy
```

For a bounded headless execution and PNG capture:

```bash
./dr_anmar.sh laparotomy 720 /tmp/dranmar-laparotomy.png
```

The asset is deterministically regenerated with:

```bash
python scripts/generate_dranmar_laparotomy_wound.py
```

## Incision boundary

`patient.incision` models ordered progression through skin, subcutaneous fat,
fascia, abdominal wall, and peritoneum. It requires blade contact and checks
force, alignment, speed, forward travel, and cutting work before releasing
pre-authored continuity identifiers. It records persistent damage and access
state.

That controller does not mutate TetMesh topology during simulation. The current
physical scene starts from the pre-segmented open variant. Arbitrary fracture,
instrumented puncture, and live remeshing are not claimed.

## Parameter boundary

Layer materials, damping, friction, attachment density, and incision gates are
provisional research parameters. They have not been fitted to instrumented
human abdominal tissue or to the Dr.Anmar grasper. A successful Isaac run is
execution evidence, not constitutive, injury, clinical, or medical-device
validation.
