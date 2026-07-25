# Mechanics

Major solid organs use authored tetrahedral proxies. Membranes and bowel-like
structures use surface or segmented representations. Runtime volume cooking is
not used by the patient path because it was unstable in the tested Isaac Sim
5.1 environment.

Open-abdominal access uses ten explicit volume-deformable bodies: left and
right margins for skin, subcutaneous fat, fascia, abdominal wall, and
peritoneum. The lateral bands are attached to the patient frame. The medial
bands are captured by six cells per exposure pad and layer so the tool moves a
distributed region instead of pulling a single vertex.

The representation follows NVIDIA's current volume-deformable hierarchy:
an Xform body root, an authored `UsdGeom.TetMesh` carrying the volume
simulation state, and a bound visual mesh. Wound-to-tool coupling uses
`OmniPhysicsVtxXformAttachment` relationships. Patient-internal collision
filtering is explicit.

The `patient.incision` controller advances through ordered, pre-authored
continuity identifiers after contact, force, speed, alignment, travel, and work
gates pass. It records procedural and damage state; it does not mutate live
TetMesh topology. Exact puncture, arbitrary fracture, biological healing, and
calibrated abdominal cutting are not claimed.

See `DYNAMIC_PATIENT_LAPAROTOMY.md` for the asset and runtime entry points.
