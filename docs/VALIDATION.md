# Integrity and runtime boundaries

The wound-preparation release has deterministic static gates for Python
compilation, JSON parsing, USDA structure, dependency closure, archive CRCs,
manifest paths, and controller invariants.

```bash
python3 scripts/validate_dranmar_wound_preparation_robot.py
python3 -m unittest -v tests/test_wound_preparation_robot.py
```

The optional Isaac script is a runtime smoke and diagnostic tool:

```bash
./isaaclab.sh -p examples/validate_wound_preparation_runtime.py \
  --headless --enable_cameras --device cuda:0 --representation standalone
```

Its output may establish that a particular composition loaded, joints moved,
schemas were applied, particles were created, and registered RGB/depth sensors
returned data on the recorded stack. It is not qualification of contact,
irrigation recovery, aspiration, debris release, tissue selectivity, physical
calibration, clinical efficacy, or patient use.

Debridement release is promotable only when work is derived from measured
contact force and tangential cartridge velocity. No test may inject a
threshold value and report the resulting release as physical evidence.
