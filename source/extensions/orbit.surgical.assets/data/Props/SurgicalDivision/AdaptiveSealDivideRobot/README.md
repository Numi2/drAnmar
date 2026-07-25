# DrAnmar Adaptive Seal-and-Divide Robot

A DrAnmar-owned, provider-neutral research end effector for NVIDIA Isaac Sim
and Isaac Lab.

## Workflow

`inspect → center → compress → seal left/right → verify → retract guard → divide → release → verify stumps`

## Primary assets

- `dranmar_adaptive_seal_divide_tool_payload.usda` — Franka payload without a nested articulation root.
- `dranmar_adaptive_seal_divide_tool_standalone.usda` — standalone articulated mechanism.
- `dranmar_adaptive_seal_divide_tool_rigid_proxy.usda` — perception/planning proxy.
- `dranmar_seal_divide_vessel_demo.usda` — two-body hollow vessel and bridge-pin scene.
- `dranmar_tissue_seal_band.usda` — physical dual-surface stump bond carrier.
- `dranmar_division_blade_cartridge.usda` — replaceable fresh/spent blade cartridge.

## Important boundary

This package is not clinically validated, is not a medical device, and is not approved for patient care. Energy, compression, seal strength, tissue response, leakage, and cutting parameters are provisional research values.
