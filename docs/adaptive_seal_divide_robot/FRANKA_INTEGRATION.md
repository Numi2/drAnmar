# Franka Integration

`make_franka_adaptive_seal_divide_robot_cfg()` starts from the Isaac Lab Franka configuration, loads the composable Franka USD, deactivates the Panda hand and finger prims, references the custom payload, and creates a fixed joint from `panda_link8` to `Links/Mount`.

The spawner snapshots the stock Panda hand joint's parent path and local frame
before deactivating the hand subtree. The documented −45 degree local Z
relationship is used only as a compatibility fallback when the stock hand
joint is unavailable. Tool joints are appended to the same articulation and
grouped into centering, jaw, blade, and valve actuator sets.

The rigid proxy is available for perception, collision-aware planning, and synthetic data when the articulated mechanism is not required.
