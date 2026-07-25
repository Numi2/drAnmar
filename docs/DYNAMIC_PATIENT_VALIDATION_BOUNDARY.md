# Dynamic abdominal patient validation boundary

This package is a research simulation platform. A software check or successful
scene run does not establish biological or clinical realism.

| Area | Current evidence | Boundary |
|---|---|---|
| Source integrity | USDA references, JSON, PNG, GLB, explicit TetMesh, Python compile, and portfolio registration are checked automatically | Engineering regression only |
| Physiology software | Presets, finite state, blood conservation, damage, intervention, reset, and scenario behavior are checked automatically | Reduced-order software model |
| Open laparotomy scene | The composed patient, ten wound-margin TetMeshes, the Dr.Anmar exposure tool, and 60 distributed attachments executed for 720 steps in Isaac Sim 5.1 on an RTX 4090 and produced an RTX frame | Observation for that exact run, not a qualification result |
| Incision state | Contact-gated ordered continuity state is implemented across five abdominal layers | No live arbitrary TetMesh topology mutation |
| Material and grasp response | Layer-specific provisional material seeds and distributed pad capture are implemented | Not fitted to abdominal-tissue or grasper bench data |
| Particle fluids | PBD carrier assets and conservation ledgers exist | Not validated CFD or fluid-structure interaction |
| Clinical use | No patient cohort, safety, human-factors, outcome, or regulatory validation | Not started; prohibited for patient care |

The open scene is a median laparotomy: it has left and right full-thickness
wound margins and no removable center tissue plug. The tool approaches from
above, captures the margins along their length, retracts them laterally, and
leaves the central operative corridor visible.

The deterministic blood, bile, urine, irrigation, perfusion, and damage models
are reduced-order software contracts. Numerical thresholds are simulation state
variables, not clinical decision thresholds.

This platform must not be used for diagnosis, treatment, patient-specific
planning, medical-device control, or patient care.
