# Dynamic abdominal patient validation boundary

This package is a research simulation platform. Passing a lower gate never
implies that a higher gate passed.

| Gate | Current evidence | Status |
|---|---|---|
| Source and payload integrity | USDA references, JSON, PNG, GLB, explicit TetMesh, Python compile, portfolio registration | Automated |
| Solver-independent behavior | Physiology presets, finite state, blood conservation, damage, interventions, reset, scenario orchestration | Automated |
| OpenUSD target-runtime parse | The target Isaac Sim release opens and resolves every composed layer | Not yet executed for this integration |
| Native mechanics | Every requested volume/surface route cooks and steps on the target PhysX/CUDA stack | Not yet executed |
| Particle fluids and sensors | PBD emission, conservation reconciliation, and rendered sensor evidence | Not yet executed |
| Physical calibration | Anatomical metrology, constitutive response, puncture, cutting, retraction, closure, flow, and injury measurements | Not started |
| Clinical qualification | Patient-cohort fitting, clinical outcomes, safety, human factors, regulatory review | Not started |

The deterministic blood, bile, urine, irrigation, perfusion, and damage models
are reduced-order software contracts. They are not CFD, validated
fluid-structure interaction, biological healing, or patient-specific
physiology. Numerical thresholds are simulation state variables, not clinical
decision thresholds.

Until the remaining gates have evidence, this platform must not be used for
diagnosis, treatment, patient-specific planning, medical-device control, or
patient care.
