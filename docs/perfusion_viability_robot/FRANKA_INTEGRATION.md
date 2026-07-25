# Franka integration

Use `make_franka_perfusion_viability_robot_cfg()` to load the standard Isaac Lab Franka, deactivate the Panda hand and finger prims, reference the payload, and attach its `Mount` link to `panda_link8`. State variants are selected before physics views initialize.

Dynamic tissue should use the USD RTX camera route for image generation. The host may bridge the ultrasound probe pose into the i4h robotic-ultrasound ray-tracing application.
