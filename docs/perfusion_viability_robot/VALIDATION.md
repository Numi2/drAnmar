# Validation

The installed overlay is checked with:

```bash
python3 scripts/validate_dranmar_perfusion_viability_robot.py
pytest -q tests/test_perfusion_viability_robot.py
```

Native-simulator evidence runs both the standalone 12-joint articulation and the
Franka hand-replacement articulation on CUDA:

```bash
./isaaclab.sh -p examples/validate_perfusion_viability_runtime.py \
  --headless --enable_cameras --device cuda:0 --representation standalone
./isaaclab.sh -p examples/validate_perfusion_viability_runtime.py \
  --headless --enable_cameras --device cuda:0 --representation franka
```

The v0.1.1 runtime probe checks OpenUSD composition, articulation
initialization, canonical phase targets, cooked surface-deformable tissue and
two fixture attachments, perfused-tissue registration, nonconstant rendered
RGB frames from all six cameras using one live RTX pipeline at a time, finite
depth, flow conservation, modality-map
finiteness, blind diagnosis of all six modeled faults, force-coupled probe
contact, evidence-based recovery, and closed-loop rescan behavior. The Franka
run additionally drives the authored 2.537 kg payload through neutral, left,
and right arm poses while recording tracking error, applied torque, and TCP
ground clearance.

The serialized camera gate records a 0.2-second acquisition span. It is a
resource-bounded acquisition test, not a claim that those raw frames are
simultaneous. Live fusion must time-stamp each frame, buffer or interpolate to
a common fusion time, and reject packets beyond the 50-millisecond skew limit.

The NVIDIA surgical bench was also run with only
`perfusion_viability_robot` selected. At the shared `(0.04, 0.04)` side
station, the complete featured mechanism and procedure substrate remained
inside the 960×640 primary-endoscope frame. The bench retains one featured
robot system and one active camera renderer at a time.

Passing it does not calibrate the optical, thermal, Doppler, ultrasound, flow,
tracer, viability, diagnostic, interventional, payload, or contact models. The
system remains manufacturer-neutral and available for simulation training;
clinical and real-world evidence are not established,
and not approved for patient care.
