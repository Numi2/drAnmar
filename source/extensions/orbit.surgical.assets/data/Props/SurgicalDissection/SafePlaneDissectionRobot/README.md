# DrAnmar SafePlane Dissection Robot

A DrAnmar-owned, provider-neutral NVIDIA Isaac Sim and Isaac Lab research
system for connectivity-aware tissue dissection with protected anatomy.

## Workflow

`inspect → capture → counter-traction → blunt spread → hydrodissect → selectively cut or apply low energy → evacuate → verify topology → release`

## Primary assets

- `dranmar_safeplane_dissection_tool_payload.usda` — Franka payload without a nested articulation root.
- `dranmar_safeplane_dissection_tool_standalone.usda` — standalone articulated mechanism.
- `dranmar_safeplane_dissection_tool_rigid_proxy.usda` — perception/planning proxy.
- `dranmar_safeplane_tissue_demo.usda` — layered tissue, adhesion network, vessel, nerve, duct, and protected organ surface.
- `dissection_topology.json` — bridge graph, thresholds, structure centerlines, and completion contract.

## Boundary

The package is not clinically validated, is not a medical device, and is not approved for patient care. Tissue, fluid, cutting, energy, force, injury, and safety parameters remain provisional research values.
