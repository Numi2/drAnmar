# DrAnmar Topical Skin Adhesive System v0.1.0

DrAnmar ships the Apache-2.0 research asset under:

```text
Props/SurgicalClosure/SkinAdhesive
```

The NVIDIA native surgical bench exposes the package as one selectable
`skin_adhesive_system`. Selecting it loads:

- Instrument 1 with the four-link applicator permanently mounted in place of
  its forceps;
- a real PhysX fixed joint between the PSM tool-tip link and applicator body.

The PSM still moves through the released NVIDIA differential-IK and PhysX
articulation path. Instrument 1's existing jaw-aperture channel is repurposed
as proportional dispense: an open hand releases the dispenser and closing the
thumb-index aperture squeezes it. The runtime maps that command to the authored
mechanism coordinates:

```text
left paddle  = -activation × 11 degrees
right paddle = +activation × 11 degrees
piston       =  activation × 9 mm
```

The live status reports commanded and measured activation, both paddle angles,
piston travel, fixed-mount state, and outlet state. State variants are selected
while the assets spawn, before Isaac creates its physics views.

## Dedicated mounted tool

The adhesive is not picked up by a claw. Selecting the asset replaces
Instrument 1's visible and colliding jaw geometry with
`psm_skin_adhesive.usda`, which composes the normal PSM and the articulated
applicator into one mechanism. The authored
`skin_adhesive_mount_joint` connects the applicator body to
`psm_tool_tip_link`.

There is no per-frame pose write, snap attachment, hidden grasp, or cap
teleport. Moving Instrument 1 moves the mounted dispenser through native IK and
the fixed physical joint. Instrument 2 remains a normal PSM forceps. Press
Space, use the panel slider, or close the tracked left thumb and index finger to
dispense proportionally.

## Python integration

```python
from orbit.surgical.assets.skin_adhesive import (
    activation_targets,
    make_articulated_cfg,
    set_activation_target,
)

applicator = make_articulated_cfg(
    prim_path="{ENV_REGEX_NS}/SkinAdhesiveApplicator",
    state="activated",
)
targets = activation_targets(0.75)
```

The optional cap, bead and logical ledger remain available in the standalone
catalog package, but the mounted DrAnmar room does not spawn or use them.

## Physics and validation boundary

The mounted applicator, fixed joint, paddle and piston joints, collision
geometry, mass properties, and contact materials are active simulation assets.
The room deliberately does not create a visual or kinematic bead. It does not
claim liquid flow, droplet breakup, wetting, polymerization, tissue bonding,
wound-edge mechanics, bond strength, toxicity, or clinical outcome. All
unmeasured mass, friction, drive, and dose parameters remain provisional and
the complete system is research-only.
