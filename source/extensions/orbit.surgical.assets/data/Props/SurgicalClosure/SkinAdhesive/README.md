# DrAnmar Topical Skin Adhesive System v0.1.0

Research-only, category-level OpenUSD assets for robot perception, grasping,
handover, uncapping, activation, path following, simulated dispensing, closure
workflow development, inspection, and disposal.

This package is not a medical device, is not clinically validated, and is not
approved for patient care. It does not reproduce a named commercial product.

## Catalog path

```text
Props/SurgicalClosure/SkinAdhesive/
```

## Assets

- `skin_adhesive_applicator_articulated.usda` — body, two squeeze paddles,
  metering piston, revolute and prismatic joints, state variants, collisions,
  materials, semantics, and robot frames.
- `skin_adhesive_applicator_rigid_proxy.usda` — single rigid body for
  perception, handover, placement, and synthetic-data workflows.
- `skin_adhesive_cap.usda` — independently graspable snap cap.
- `skin_adhesive_bead.usda` — fresh/cured deposit-state asset. This is a
  kinematic task representation, not a fluid or polymerization simulation.

## Applicator states

`sealed`, `activated`, and `empty` are discrete USD variants. They coordinate
body mass properties, reservoir visibility, state indicator, wick material, and
tip-contact physics. Select the state while spawning, before physics views are
initialized.

```python
cfg = make_articulated_cfg(state="activated")
```

## Continuous mechanism

- `left_paddle_joint`: revolute, -11 to 0 degrees
- `right_paddle_joint`: revolute, 0 to +11 degrees
- `metering_piston_joint`: prismatic, 0 to 9 mm

The helper maps one normalized squeeze command to both paddles and the piston.
This coupling is a robotics task contract, not a claim about a particular
commercial applicator mechanism.

## Interaction frames

- `body_grasp`
- `paddle_left_contact`
- `paddle_right_contact`
- `tip`
- `dispense_exit`
- `placement_reference`
- `path_tangent_reference`
- `reservoir_center`
- `activation_reference`
- `count_reference`

The cap provides `cap_grasp`, `cap_snap_axis`, and `count_reference`. The bead
provides start, midpoint, endpoint, tangent, and count frames.

## Intended tasks

1. Locate and grasp the applicator body.
2. Remove and place the cap.
3. Activate both squeeze paddles.
4. Position the wick over an approximated wound path.
5. Follow a path while commanding metering-piston travel.
6. Instantiate or update a fresh bead representation.
7. Switch the deposit to cured for later task stages.
8. Place the used applicator and cap into disposal targets.

## Limitations

The package does not model liquid flow, droplet breakup, adhesive wetting,
polymerization, tissue bonding, wound-edge mechanics, bond strength, toxicity,
or clinical outcome. All unmeasured mass, friction, dose, and drive parameters
are enabled provisional engineering seeds documented in `physics_profile.json`.

## License

Apache License 2.0. Original geometry and textures are independently generated.
