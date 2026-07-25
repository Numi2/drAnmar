# Validation

The release gate combines deterministic regeneration, `usdchecker` parsing of all
eight primary OpenUSD assets, GLB/PNG/JSON/container checks, manifest and mirror
hash verification, controller unit tests, and headless CUDA execution in both the
standalone-tool and Franka-mounted representations.

Runtime qualification verifies:

- current Isaac surface-deformable schemas on the vessel wall;
- two fixture, two temporary-compression, two retained-clip, and eight patch-bond
  vertex attachments;
- all eleven tool joints and, in the combined representation, all seven Franka
  arm joints;
- finite articulation state after stepping;
- conserved particle-volume bookkeeping and annular suction capture;
- provisional clip-retention and patch-cure thresholds;
- pressure-challenge residual-flow integration;
- zero captured engine errors.

This is software and simulation-stack qualification only. It does not establish
clinical performance, calibrated vessel damage, clip plasticity, biochemical
coagulation, patient safety, regulatory suitability, or physical-bench validity.
