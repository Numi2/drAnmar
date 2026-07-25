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
- bilateral force, jaw-separation, tool-speed, and target-distance measurements;
- reversible compression, force-asymmetry, overload damage, residual-flow,
  blood-loss, and distal-perfusion effects;
- complication detection and rescue-plan selection that cannot author an
  outcome;
- transition-based rescue reward with no reward for waiting in a favorable
  state.

The large OR USD contains four authored station and tool-change layout frames.
Those are composition metadata, not four live Franka articulations. A host may
populate them with compatible robots. The optional
`autonomous_rescue_scene_cfg()` composition does populate those frames with
four Franka/tool articulations; the current DrAnmar workstation room instead
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
particle counts. Hemostasis verification additionally requires measured
upstream pressure; requesting a challenge cannot synthesize one. The DrAnmar
runtime also expires active repair and ventilation effects when their evidence
is not refreshed in the current physics interval. It derives:

- temporary vessel compression from bilateral contact, symmetry, gap, speed,
  and proximity to the authored defect-control frame;
- definitive clip control from retained attachment state and contact dwell;
- patch sealing from distributed contact and dwell;
- residual bleeding and conserved blood loss;
- distal-perfusion tradeoffs, shared-patient MAP coupling, and overload damage;
- repair approximation, retention and leak;
- occlusive-film integrity from eight perimeter bonds, distributed contact,
  measured cavity pressure, leak, and a continuous pressure-hold window;
- crystalloid, blood-product, and vasopressor delivery from mutually
  consistent plunger travel, outlet flow, reservoir mass loss, vascular-line
  attachment, and line pressure;
- ventilation support from airway attachment, valve travel, net circuit flow,
  airway pressure, measured oxygen fraction, and observed chest excursion;
- oxygenation support from airway attachment, valve position, circuit flow and
  leak, airway pressure, delivered oxygen fraction, and measured chest motion;
- complication detection, rescue priority and patient-state reward;
- hemostasis only after a continuous measured pressure-challenge evidence
  window.

Runtime modules:

```text
orbit.surgical.assets.autonomous_rescue_or
orbit.surgical.assets.deformable_rescue
orbit.surgical.assets.resuscitation_effects
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
