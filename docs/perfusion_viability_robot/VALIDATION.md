# Validation

The installed overlay is checked with:

```bash
python3 scripts/validate_dranmar_perfusion_viability_robot.py
pytest -q tests/test_perfusion_viability_robot.py
```

Native qualification runs both the standalone 12-joint articulation and the
Franka hand-replacement articulation on CUDA:

```bash
./isaaclab.sh -p examples/validate_perfusion_viability_runtime.py \
  --headless --device cuda:0 --representation standalone
./isaaclab.sh -p examples/validate_perfusion_viability_runtime.py \
  --headless --device cuda:0 --representation franka
```

The runtime probe checks OpenUSD composition, articulation initialization,
canonical phase targets, perfused-tissue registration, registered camera
frames, flow conservation, modality-map finiteness, and closed-loop rescan
behavior. Passing it does not calibrate the optical, thermal, Doppler,
ultrasound, flow, tracer, viability, or diagnostic models. The system remains
manufacturer-neutral, research-only, not clinically validated, and not
approved for patient care.
