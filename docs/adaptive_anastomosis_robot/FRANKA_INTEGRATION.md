# DrAnmar Franka integration

The combined spawner references the standard NVIDIA Isaac Franka USD, snapshots the stock Panda hand-joint body target and local mounting frame, disables the Panda hand, finger links and finger joints, references the DrAnmar payload, and creates a fixed wrist-to-tool joint using that resolved frame.

This preserves the standard Panda-hand relationship, including the nominal -45 degree local Z rotation, without assuming a particular flattened Franka prim path. The closure mechanism becomes part of the same reduced-coordinate articulation and is controlled through dedicated Isaac Lab actuator groups.

Use `make_franka_adaptive_anastomosis_robot_cfg()` for the combined robot, `make_tool_cfg()` for the standalone mechanism, and `make_rigid_proxy_cfg()` for perception and planning tasks.
