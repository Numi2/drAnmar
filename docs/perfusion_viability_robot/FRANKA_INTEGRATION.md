# Franka integration

Use `make_franka_perfusion_viability_robot_cfg()` to load the standard Isaac Lab Franka, deactivate the Panda hand and finger prims, reference the payload, and attach its `Mount` link to `panda_link8`. State variants are selected before physics views initialize.

Dynamic tissue uses the USD RTX camera route for image generation. USD camera optical -Z is explicitly rotated onto the authored tissue-facing +Z sensor axis. The native-simulator evidence program captures nonconstant RGB from all six camera frames and depth from the left camera with one live RTX camera pipeline at a time. Each frame is timestamped; operational fusion must buffer or interpolate to a common time and apply the 50 ms skew gate. The loaded-arm gate then drives the 2.537 kg payload through neutral, left, and right poses. The host may bridge the ultrasound probe pose into the i4h robotic-ultrasound ray-tracing application.

For low-latency operation, prewarm and reuse one camera/render-product pipeline and its output buffers, then bind or schedule the six registered views serially. Do not construct all six pipelines concurrently. The evidence program intentionally destroys each pipeline before creating the next one to prove cleanup and maximum concurrency of one; that destructive lifecycle is a strong resource gate, not the recommended per-frame production loop.
