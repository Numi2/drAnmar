# Franka Integration

`make_franka_safeplane_dissection_robot_cfg()` starts from Isaac Lab's Franka configuration, loads the composable Franka USD, deactivates the Panda hand and finger prims, references the custom payload, and creates a fixed joint from `panda_link8` to `Links/Mount`. When NVIDIA's composable asset collapses the terminal URDF link into the `panda_link7`-to-hand joint, the spawner reconstructs a lightweight fixed `panda_link8` compatibility body before mounting the payload.

The payload preserves the standard Panda-hand mounting rotation of −45 degrees around local Z. Tool joints are appended to the same articulation and grouped into traction, pad-compliance, spreader, hydro, scissors, energy-tip, and valve actuator sets.

The rigid proxy is a geometry-only planning/perception representation. It does
not provide calibrated sensors, measured payload dynamics, collision-safety
qualification, synthetic-data ground truth, or dissection outcome evidence.
