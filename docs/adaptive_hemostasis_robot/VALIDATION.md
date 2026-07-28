# Capability and evidence boundaries

The Adaptive Hemostasis Robot now has a source-level, fail-closed evidence
contract for field clearing, bilateral compression, clip mechanics, patch
mechanics, blood-volume accounting, and provisional pressure challenge
signals.

The reviewed controller accepts only exact, same-revision scene evidence:
vessel observations, bilateral compression contacts, clip forming state plus
measured load/capacity/slip and attachment identity, patch cohesive state plus
attachment identity, raw native record identity, and exact episode,
environment, topology, step, and physics-time provenance. Caller-authored
closure, retention, seal, cure, and outcome values are rejected.

This source review did not execute the generator, validator, tests, simulator,
or any native runtime. No native scene-evidence provider is implemented in the
repository, so no current simulator task completion is verified. Existing
manifests and generated hashes are intentionally marked stale until an
evidence provider exists and a separately authorized release-qualification
run regenerates them.

The shared vessel, contact, clip, and cohesive models remain reduced-order
research mechanics. The clip model does not simulate metal plasticity or
forming geometry; the patch model does not simulate biochemical cure,
healing, or tissue integration; blood is not qualified CFD; and the parameter
sets are not physically calibrated.

No record in this repository qualifies durable hemostasis, patient-specific
response, clinical performance, or patient-care use.
