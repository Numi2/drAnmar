# DrAnmar Adaptive Hemostasis Robot v0.1.0

DrAnmar-owned OpenUSD research system for robotic field clearing, temporary vascular compression, physical clip retention, hemostatic patch reinforcement, and post-seal leak verification. It integrates with NVIDIA Isaac Lab and Isaac Sim while remaining provider-independent at the asset-contract layer.

## Catalog path

`Props/SurgicalHemostasis/AdaptiveHemostasisRobot/`

## Main assets

- `dranmar_adaptive_hemostasis_tool_payload.usda` — Franka payload without a nested articulation root.
- `dranmar_adaptive_hemostasis_tool_standalone.usda` — standalone articulated tool.
- `dranmar_adaptive_hemostasis_tool_rigid_proxy.usda` — perception/planning proxy.
- `dranmar_hemostatic_clip.usda` — open/formed clip states and two physical vessel-attachment zones.
- `dranmar_hemostatic_patch.usda` — deformable-ready triangular patch surface.
- `dranmar_hemostatic_patch_rigid_proxy.usda` — stable eight-cell bond carrier.
- `dranmar_bleeding_vessel_demo.usda` — curved hollow vessel wall with reduced-order bleeding source.
- `dranmar_blood_droplet.usda` — particle prototype.

## Procedure

`inspect → clear → compress → temporary control check → clip → release compression → patch → pressure challenge → verify → complete or abort`

All geometry, mechanics, flow, retention, bond, pressure, and damage values are provisional research parameters. This asset is not clinically validated and is not approved for patient care.
