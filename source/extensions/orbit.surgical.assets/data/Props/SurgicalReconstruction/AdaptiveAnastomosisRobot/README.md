# DrAnmar Adaptive Anastomosis Robot v0.1.0

Dr.Anmar executable simulation-training workcell for end-to-end hollow-tissue
anastomosis tasks. It includes a provider-neutral Franka-compatible end
effector, portable OpenUSD assets, task state, observations, actions, safety
gates, and evaluation signals for NVIDIA Isaac Sim and Isaac Lab.

## Capabilities

- bilateral circumferential tissue capture with 6 independent cells per side;
- independent axial approximation of the two tissue ends;
- lumen-preserving alignment mandrel and expandable centering cage;
- independent edge-eversion rings;
- one-shot 16-position circumferential staple crown;
- individual retained staple bodies with left/right tissue attachment regions;
- circumferential reinforcement collar with independent bilateral bond sectors;
- lumen patency scoring and pressure-decay leak verification;
- direct replacement of the Panda hand at `panda_link8`;
- standalone articulated, Franka payload, and rigid proxy representations.
- explicit deformable distal fixtures, temporary-capture release, retained staples, and bilateral collar-sector attachments.

## Primary assets

```text
dranmar_adaptive_anastomosis_tool_payload.usda
dranmar_adaptive_anastomosis_tool_standalone.usda
dranmar_adaptive_anastomosis_tool_rigid_proxy.usda
dranmar_hollow_tissue_demo.usda
dranmar_anastomosis_staple.usda
dranmar_reinforcement_collar.usda
dranmar_reinforcement_collar_rigid_proxy.usda
dranmar_leak_test_droplet.usda
```

The current package represents staple formation as a discrete open-to-formed event and reinforcement as staged mechanical attachments. It does not claim clinically calibrated penetration, plasticity, tissue damage, adhesive chemistry, healing, patency, or leak thresholds. It is not approved for patient care.
