# Validation and qualification

Run the deterministic generator from an isolated environment using `scripts/requirements_adaptive_anastomosis_generation.txt`, then run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
python scripts/validate_dranmar_adaptive_anastomosis_robot.py --require-usdchecker
```

The native runtime matrix uses `examples/validate_adaptive_anastomosis_runtime.py` twice, once for the standalone 14-DOF mechanism and once for the combined 21-DOF Franka assembly. Each case cooks both tissue surfaces, authors current attachment schemas, anchors both distal ends, creates and releases 12 capture attachments, retains 16 independent staples through 32 leg attachments, cures 32 collar-sector attachments, emits conserved PBD leak particles, evaluates patency, performs an 8-second pressure-decay challenge, and advances at least 120 CUDA simulation steps.

Passing this matrix qualifies software integration on the recorded hardware and software stack only. It does not qualify tissue mechanics, staple penetration or plasticity, collar adhesion, patency measurement, leak calibration, clinical thresholds, sterility, safety, or patient use.
