# Validation and promotion gates

The development package has dependency-free validation and controller-test
entry points:

```bash
python3 scripts/validate_dranmar_atraumatic_exposure_robot.py --require-usdchecker
python3 -m unittest -v tests/test_atraumatic_exposure_robot.py
```

The validator checks all six primary USDA layers with `usdchecker`, verifies
the asset/source mirrors and manifest hashes, parses every JSON contract,
checks all GLB and PNG containers, compiles Python sources in memory, enforces
the release inventory, and rejects packaged bytecode and unsafe manifest paths.

The controller tests cover phase completeness and bounds, compliance-based
force estimation, invalid numeric input rejection, nominal force/visibility
control, hard-overload unloading, ROI metrics, and fail-closed frame/phase
lookup.

On an Isaac Lab CUDA host, qualify both pad geometries in both representations:

```bash
for representation in standalone franka; do
  for pad_type in fenestrated microcup; do
    ./isaaclab.sh -p examples/validate_atraumatic_exposure_runtime.py \
      --headless --device cuda:0 \
      --representation "$representation" --pad-type "$pad_type"
  done
done
```

The runtime test requires finite articulation state after 120 simulation steps,
all eight tool joints, the seven Franka arm joints in the combined
representation, current surface-deformable cooking for both tissue flaps, two
outer-band anchors, twelve independently authored pad capture constraints,
finite force/visibility-controller output, and zero error-level Isaac/PhysX
messages.

## Qualification boundary

On 2026-07-25, both pad geometries passed 120 headless steps in the
standalone and complete Franka-mounted representations on Numi's RTX 4090
using Isaac Lab 6.1.16, Isaac Sim 6.0.1.0, and `cuda:0`. All four cases
reported finite articulation state, both cooked deformable flaps, two outer
anchors, twelve `OmniPhysicsVtxXformAttachment` capture constraints, finite
force/visibility-controller output, and zero error-level engine messages.

A passing smoke test qualifies bounded composition and execution on the
reported software/GPU stack. It does not qualify contact tuning, tissue
mechanics, capture pressure, physical metrology, clinical efficacy, regulatory
status, or patient use. Those remain separate promotion gates.
