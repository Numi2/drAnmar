# Dynamic abdominal patient validation boundary

This package is a research simulation platform. Passing a lower gate never
implies that a higher gate passed.

| Gate | Current evidence | Status |
|---|---|---|
| Source and payload integrity | USDA references, JSON, PNG, GLB, explicit TetMesh, Python compile, portfolio registration | Automated |
| Solver-independent behavior | Physiology presets, finite state, blood conservation, damage, interventions, reset, scenario orchestration | Automated |
| OpenUSD target-runtime parse | The patient composed and entered a target Isaac Sim 5.1 room during manual engineering smoke work | Partial; no durable qualification artifact is committed |
| Native mechanics | One isolated surface route stepped on the target RTX/PhysX stack during manual smoke work; volume and simultaneous multi-component attempts produced CUDA/contact failures | Partial and unqualified; the room is restricted to one explicitly selected surface lane |
| Particle fluids and sensors | PBD emission, conservation reconciliation, and rendered sensor evidence | Not yet executed |
| Physical calibration | Anatomical metrology, constitutive response, puncture, cutting, retraction, closure, flow, and injury measurements | Not started |
| Clinical qualification | Patient-cohort fitting, clinical outcomes, safety, human factors, regulatory review | Not started |

The automated report's `passed` field means only that the checks executed by
that validator passed. It does not mean the complete patient, every deformable
route, or any clinical gate passed. `overall_qualified` remains false.

The native mechanics restriction is deliberate. A USD collision-filter request
did not make simultaneous overlapping deformables safe on the tested Isaac 5.1
runtime. Multi-component deformable activation and every volume route remain
blocked until each has a retained, reproducible native evidence artifact.

The deterministic blood, bile, urine, irrigation, perfusion, and damage models
are reduced-order software contracts. They are not CFD, validated
fluid-structure interaction, biological healing, or patient-specific
physiology. Numerical thresholds are simulation state variables, not clinical
decision thresholds.

Until the remaining gates have evidence, this platform must not be used for
diagnosis, treatment, patient-specific planning, medical-device control, or
patient care.
