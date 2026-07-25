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
seven authored debris attachments, a conserved multi-nozzle PBD particle burst,
and zero error-level Isaac/PhysX log messages.

## Qualified runtime snapshot

On 2026-07-25, the standalone tool and complete Franka-mounted representation
each passed 120 headless simulation steps on the Gilgamesh RTX 4090 using
Isaac Lab 6.1.16, Isaac Sim 6.0.1.0, and `cuda:0`. Both runs reported finite
joint state, all five tool joints, seven
`OmniPhysicsVtxXformAttachment` debris constraints, 80 emitted PBD particles,
zero fluid-ledger balance error, and zero error-level engine messages. The
Franka run resolved all seven arm joints and all five tool joints in one
articulation.

This snapshot qualifies bounded execution, composition, surface-deformable
cooking, attachment authoring, and PBD particle creation on that exact software
and GPU stack. It does not qualify contact tuning, physical metrology, fluid
calibration, tissue selectivity, clinical efficacy, or patient use. Those remain
separate promotion gates.
