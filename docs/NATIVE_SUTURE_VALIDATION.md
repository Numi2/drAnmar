# NVIDIA SoftMimicGen strand reference

The separate **PSM strand ring threading** room preserves NVIDIA
SoftMimicGen's released PhysX FEM strand-and-ring task as an upstream reference.
It is not the DrAnmar suture knotting backend.

`physics_next/softmimicgen.json` pins the upstream source, Isaac Lab fork,
assets, and demonstration dataset by revision and checksum.

## Replay validation

```bash
./dr_anmar_suture_native.sh install-upstream
./dr_anmar_suture_native.sh validate-upstream demo_0
```

Validation resets the released task, replays its 123 relative-IK actions,
evaluates NVIDIA's ring-crossing predicate, and compares the robot, ring, and
all 549 FEM nodes with the recorded episode.

The committed Gilgamesh report is
`physics_next/benchmarks/softmimicgen-threading-replay.json`.

## Evidence boundary

This replay validates reproducibility of the pinned upstream direct-strand
threading task. That task grasps the strand directly and does not establish
DrAnmar needle attachment, instrument transfer, knot mechanics, strand
breakage, or clinical fidelity.

Those DrAnmar-specific behaviors are owned by the Warp backend documented in
`docs/DR_ANMAR_WARP_SUTURE.md`.
