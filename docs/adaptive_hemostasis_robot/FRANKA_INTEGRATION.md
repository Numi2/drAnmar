# Franka integration

The custom spawner references the composable Isaac Franka asset, snapshots the stock `panda_hand_joint` body target and local frame, safely deactivates the stock hand and finger prims, references the DrAnmar payload, and creates a fixed joint to `Links/Mount`. A uniquely resolved `panda_link8` with −45 degrees around local Z is retained only as a compatibility fallback.
