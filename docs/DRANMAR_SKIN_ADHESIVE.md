# DrAnmar Topical Skin Adhesive System v0.1.0

DrAnmar ships the Apache-2.0 research asset under:

```text
Props/SurgicalClosure/SkinAdhesive
```

The NVIDIA native surgical bench exposes the package as one selectable
`skin_adhesive_system`. Selecting it loads:

- the four-link articulated applicator in its `activated` state;
- the independently graspable removable cap aligned with the nozzle;
- the `fresh` kinematic adhesive-bead task representation.

The workstation control panel exposes one normalized activation command. The
runtime maps it to the authored mechanism coordinates:

```text
left paddle  = -activation × 11 degrees
right paddle = +activation × 11 degrees
piston       =  activation × 9 mm
```

The live status reports commanded and measured activation, both paddle angles,
piston travel, cap readiness, and bead state. State variants are selected while
the assets spawn, before Isaac creates its physics views.

## One-button pickup and uncapping

Press **Use adhesive** in the workstation panel. The selected instrument closes
its jaws, the authored `body_grasp` frame is aligned inside the tool, and the
applicator follows that instrument while the cap is moved to its bench parking
position. The operator can immediately move the instrument instead of manually
coordinating two PSMs around the applicator.

This is an explicit guided grasp assist, reported as
`guided_kinematic_grasp`. It is a usability feature for the simulated
workstation, not a claim that a physical robot autonomously planned or executed
the pickup. Resetting the scene releases the assist and restores the applicator
and cap to their initial bench poses.

## Python integration

```python
from orbit.surgical.assets.skin_adhesive import (
    DispenseLedger,
    activation_targets,
    make_articulated_cfg,
    make_bead_cfg,
    make_cap_cfg,
    set_activation_target,
)

applicator = make_articulated_cfg(
    prim_path="{ENV_REGEX_NS}/SkinAdhesiveApplicator",
    state="activated",
)
cap = make_cap_cfg(prim_path="{ENV_REGEX_NS}/SkinAdhesiveCap")
bead = make_bead_cfg(
    prim_path="{ENV_REGEX_NS}/SkinAdhesiveBead",
    state="fresh",
)
targets = activation_targets(0.75)
```

`DispenseLedger` records logical bead-placement task events. It deliberately
does not infer dose or clinical adhesive performance.

## Physics and validation boundary

The applicator, cap, joints, collision geometry, interaction frames, mass
properties, and contact materials are active simulation assets. During
repository integration, the bead's two negative generated principal-inertia
components were corrected to their positive magnitudes so the fresh and cured
variants have valid rigid-body inertia tensors.

The adhesive bead remains an explicit kinematic workflow representation. The
package does not simulate liquid flow, droplet breakup, wetting,
polymerization, tissue bonding, wound-edge mechanics, bond strength, toxicity,
or clinical outcome. All unmeasured mass, friction, drive, and dose parameters
remain provisional and the complete system is research-only.
