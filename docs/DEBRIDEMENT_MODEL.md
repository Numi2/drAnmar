# Debridement model

The demo wound bed contains separate rigid debris fragments. At runtime each
fragment can be attached to the deformable wound surface. The
`DebridementReleaseController` integrates contact work:

```text
work increment = normal force × tangential speed × timestep
```

When cumulative work exceeds a fragment-specific provisional threshold, the
temporary attachment is removed. The released fragment can then be moved by
contact or captured by the annular suction field.

This creates a physical sequence of adhered debris, mechanical mobilization,
release, and aspiration. It does not simulate living-tissue excision, bleeding,
necrosis, bacterial load, tissue viability, or clinical debridement efficacy.
