# DrAnmar skin stapler asset

This directory contains the repository-local DrAnmar skin stapler at the
I4H-compatible catalog path:

```text
Props/SurgicalClosure/SkinStapler
```

Primary runtime files:

- `skin_stapler_rigid_proxy.usda`: loaded or empty rigid prop for the stable
  Isaac Sim 5.1 / Isaac Lab 2.3.2 operating-room lane.
- `skin_stapler_articulated.usda`: trigger and pusher articulation for the
  Isaac Sim 6.0.1 / Isaac Lab 3.0 research lane.
- `skin_staple.usda`: standalone simulated formed staple.
- `physics_profile.json`: provisional mass, inertia, material and collision
  contract.
- `interaction_frames.json`: grasp, trigger, jaw, exit, placement and count
  frames.

The `geometry`, `glb`, and `textures` directories preserve the complete
inspection payload supplied with v0.2.0. `integration_report.json` records the
DrAnmar import boundary and local OpenUSD checks.

This is category-level research content. It is not clinically validated, is
not a medical device, and must not be used for patient care. Simulated staple
deployment is task bookkeeping; tissue penetration, staple formation, wound
healing, sterility, closure strength and clinical quality are not modeled.

License: Apache-2.0. See `LICENSE.txt` and `NOTICE.txt`.
