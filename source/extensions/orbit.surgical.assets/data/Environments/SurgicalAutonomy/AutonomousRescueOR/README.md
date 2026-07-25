# DrAnmar Autonomous Rescue OR v0.4.0

A simulation-only rescue environment that combines DrAnmar deformable surgical
substrates with contact-owned patient effects. The policy chooses motion and
intervention intent; post-PhysX contact, geometry, retained attachments, and
leak observations determine bleeding, perfusion, repair integrity, damage, and
reward.

## What is executable now

- a native DrAnmar room using NVIDIA i4h dual dVRK PSM articulations;
- an endpoint-anchored Omni Physics volume-deformable rescue vessel;
- jaw-to-vessel contact filtering from the live PhysX scene;
- bilateral force, jaw-separation, and tool-speed measurements;
- reversible compression, force-asymmetry, overload damage, residual-flow,
  blood-loss, and distal-perfusion effects;
- complication detection and rescue-plan selection that cannot author an
  outcome;
- transition-based rescue reward with no reward for waiting in a favorable
  state.

The large OR USD contains four authored station and tool-change layout frames.
Those are composition metadata, not four live Franka articulations. A host may
populate them with compatible robots later. The current DrAnmar workstation
uses the real dual-PSM task it actually instantiates.

The procedure, protocol, benchmark, resource, and tool JSON files describe
intent and scene setup. They are not a safety kernel and they do not grant
success. Scenario parameters may initialize a physical defect or fault before
an episode; policy actions cannot write clinical result fields.

## Executable patient-effect path

Policies request interventions but cannot write bleeding, occlusion, seal,
perfusion, leak, closure, or success values. The environment submits monotonic
post-physics observations containing bilateral contact forces, tool speed,
measured separation, retained attachments, patch contact points, and leaked
particle counts. The DrAnmar runtime derives:

- temporary vessel compression from bilateral contact, symmetry, gap and speed;
- definitive clip control from retained attachment state and contact dwell;
- patch sealing from distributed contact and dwell;
- residual bleeding and conserved blood loss;
- distal-perfusion tradeoffs and overload damage;
- repair approximation, retention and leak;
- complication detection, rescue priority and patient-state reward;
- hemostasis only after a fresh pressure-challenge evidence window.

Runtime modules:

```text
orbit.surgical.assets.autonomous_rescue_or
orbit.surgical.assets.deformable_rescue
```

## Catalog path

```text
Environments/SurgicalAutonomy/AutonomousRescueOR/
```

## Important boundary

The geometry and runtime are implemented for simulation training. Calibration
values are provisional engineering seeds.

This package is research software. It is not clinically validated,
patient-specific, a medical device, or approved for patient care.
