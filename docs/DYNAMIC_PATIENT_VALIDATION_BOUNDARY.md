# Dynamic abdominal patient validation boundary

This package is an interactive simulation-training platform. A software check or successful
scene run does not establish biological or clinical realism.

| Area | Current evidence | Boundary |
|---|---|---|
| Source integrity | Current revision reviewed at source level only; no compile, import, validator, build, test, or gate was run | Unexecuted current revision |
| Physiology software | Shared blood/fluid ledger and evidence-led contact/effect routes are implemented; authored intervention outcomes are disabled | Bile and generic suction lack source-compartment conservation; low-level patient models remain mutable; current runtime unverified |
| Open laparotomy scene | A prior revision executed for 720 steps in Isaac Sim 5.1 on an RTX 4090 and produced an RTX frame | Historical observation only; not evidence for current source |
| Incision state | A private scalar-sample engineering fixture models ordered continuity state | Public runtime mutation disabled pending an exact prim-bound incision provider; no live arbitrary TetMesh topology mutation |
| Material and grasp response | Layer-specific provisional material seeds and a camera-only distributed attachment fixture are implemented | Public grasp qualification disabled pending an exact provider; not fitted to abdominal-tissue or grasper bench data |
| Particle fluids | PBD carrier assets and conservation ledgers exist | Not validated CFD or fluid-structure interaction |
| Clinical use | No patient cohort, safety, human-factors, outcome, or regulatory validation | Not started; prohibited for patient care |

The open scene is a median laparotomy: it has left and right full-thickness
wound margins and no removable center tissue plug. The tool approaches from
above, captures the margins along their length, retracts them laterally, and
leaves the central operative corridor visible.

The deterministic blood, bile, urine, irrigation, perfusion, and damage models
are reduced-order software contracts. Numerical thresholds are simulation state
variables, not clinical decision thresholds.

Native providers do not yet connect every workcell's mechanics to patient state.
Absent exact evidence, those workcells must abstain rather than write an
intervention outcome.

This platform must not be used for diagnosis, treatment, patient-specific
planning, medical-device control, or patient care.
