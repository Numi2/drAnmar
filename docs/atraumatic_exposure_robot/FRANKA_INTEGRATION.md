# Franka integration

`make_franka_exposure_robot_cfg()` references the composable Isaac Franka,
snapshots the stock `panda_hand_joint` body and local frame, deactivates the
Panda hand and finger prims, then fixes `Links/Mount` to that verified frame.
A uniquely resolved `panda_link8` with the standard -45 degree Z frame is used
only as a compatibility fallback for older Franka layouts.

The payload intentionally has no articulation root. The Franka articulation
owns the complete robot and tool. The standalone asset contains its own
articulation root for isolated mechanism development.
