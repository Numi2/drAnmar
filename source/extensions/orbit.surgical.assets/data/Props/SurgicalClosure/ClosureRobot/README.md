# DrAnmar Approximate-Staple-Seal Closure End Effector

A modular Franka-compatible research end effector that performs one ordered closure sequence:

1. closes bilateral upper clamps onto the left and right tissue edges;
2. creates temporary clamp-to-deformable attachments in the actual overlap regions;
3. translates both approximation carriages toward the centerline;
4. drives a chambered staple against the central anvil;
5. deploys a rigid formed staple with independent left/right tissue-retention bonds;
6. releases the temporary approximation attachments while the staple remains load bearing;
7. moves the adhesive nozzle onto the closed line and deposits fresh physical bead segments;
8. changes each bead from fresh tack to a cured, stronger dual-sided bond.

The payload replaces the standard Panda hand at `panda_link8`. It is not held by the stock fingers. The composite spawner uses the composable Franka USD, deactivates the stock hand and finger prims before the articulation view initializes, references the payload, and joins `Links/Mount` to `panda_link8` through one fixed joint. Arm and tool therefore appear as one Isaac Lab articulation.

## Core assets

- `dranmar_closure_tool_payload.usda`: hand-replacement payload without its own articulation root.
- `dranmar_closure_tool_standalone.usda`: independently spawnable articulated tool.
- `dranmar_closure_tool_rigid_proxy.usda`: one-body perception, planning, and dataset proxy.
- `dranmar_closure_staple.usda`: open/formed staple with two leg-specific attachment volumes.
- `dranmar_closure_adhesive_bead.usda`: fresh/cured adhesive bridge with two side-specific bond volumes.
- `dranmar_closure_tissue_demo.usda`: split triangular surface reference for closure experiments.

## Mechanism inventory

- 2 tissue-approximation prismatic joints;
- 2 tissue-clamp revolute joints;
- 1 staple-driver prismatic joint;
- 1 adhesive-head deployment prismatic joint;
- 1 adhesive-metering prismatic joint;
- 12 total staple positions, including one chambered staple;
- 0.8 mL provisional adhesive-reservoir state.

All moving-link frames are authored beneath their owning links, for example `Links/AdhesiveCarriage/Frames/adhesive_tip`. `interaction_frames.json` also records a rest-pose reference for inspection.

## Isaac Lab integration

```python
from orbit.surgical.assets.closure_robot import (
    ClosureSequenceController,
    make_franka_closure_robot_cfg,
)

robot_cfg = make_franka_closure_robot_cfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    staple_state="loaded",
    adhesive_state="full",
)

closure = ClosureSequenceController(
    tool_path="/World/envs/env_0/Robot/DrAnmarClosureTool",
    left_tissue_path="/World/Tissue/Left",
    right_tissue_path="/World/Tissue/Right",
)
```

The task or policy commands the joint targets. `ClosureSequenceController` owns the discrete physics events: temporary capture, formed-staple attachment, staple inventory, adhesive deposition, cure progression, and provisional overload release.

## Physical closure contract

- Approximation is produced by articulated carriage motion while tissue is attached to the clamp contact volumes.
- The staple is not decorative: the two formed legs create separate deformable attachments and remain after clamp release.
- Adhesive is not decorative: each deposited bead creates independent left/right tissue bonds. Its task-level strength increases with cure fraction and can fail under an imposed resultant load.
- No biological healing, chemical reaction, calibrated penetration, or clinical efficacy is claimed.

Available for simulation training with category-level geometry and disclosed
engineering parameters. Real-world and clinical evidence are not established.
