# Franka integration

`make_franka_wound_preparation_robot_cfg()` starts with Isaac Lab's Franka
configuration, selects the composable Franka USD, deactivates the stock Panda
hand and finger prims, references the DrAnmar payload, and fixes the payload
`Mount` link to `panda_link8`.

The payload deliberately omits `PhysicsArticulationRootAPI`; the Franka root
owns the combined articulation. The standalone asset includes its own
articulation root for isolated mechanism development.

The integration preserves the standard −45 degree hand mounting rotation and
adds five tool-joint actuator groups to the host articulation.
