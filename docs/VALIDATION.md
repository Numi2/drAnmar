# Validation and promotion gates

The development package has one dependency-free validation entry point:

```bash
python3 scripts/validate_dranmar_wound_preparation_robot.py --require-usdchecker
python3 -m unittest -v tests/test_wound_preparation_robot.py
```

The validator checks all nine primary USDA layers with `usdchecker`, verifies
the asset/source mirrors and manifest hashes, parses every JSON contract, checks
all GLB and PNG containers, compiles the Python sources in memory, enforces the
release inventory, and rejects packaged bytecode or unsafe manifest paths.

The controller tests prove fluid conservation across aspiration, spill, and
explicit discard sinks; collection-capacity behavior; invalid-input rejection;
phase-target completeness and bounds; fail-closed phase/frame lookup; and
sequence snapshot history.

On an Isaac Lab CUDA host, run both runtime representations:

```bash
./isaaclab.sh -p examples/validate_wound_preparation_runtime.py \
  --headless --device cuda:0 --representation standalone
./isaaclab.sh -p examples/validate_wound_preparation_runtime.py \
  --headless --device cuda:0 --representation franka
```

The runtime smoke test requires finite articulation state after simulation,
presence of all five tool joints, successful current surface-deformable cooking,
seven debris attachments, and a conserved multi-nozzle PBD particle burst.

Passing these checks qualifies the package structure and static OpenUSD
composition. It does not qualify Isaac Sim execution, PhysX CUDA particles,
surface-deformable cooking, attachment behavior, contact tuning, physical
metrology, tissue selectivity, or clinical use. Those remain separate promotion
gates and must not be inferred from the static result.
