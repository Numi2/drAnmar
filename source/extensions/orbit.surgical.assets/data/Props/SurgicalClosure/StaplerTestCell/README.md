# Dr.Anmar Tissue Closure Bench 0.3.0

This research fixture holds the Dr.Anmar articulated skin stapler in a cradle
with a six-DOF proportional-velocity virtual fixture and exposes a bounded virtual
trigger actuator. It removes the need for a robot hand that can grasp a
gun-shaped housing. The fixture now indexes the stapler across seven guided
stations on the Dr.Anmar open-incision suturable-tissue asset.

The cell is intended to measure simulation mechanism behavior:

- commanded versus observed trigger travel;
- measured pre-fire wound-edge approximation;
- synchronized pusher travel;
- measured housing translation and rotation error;
- exactly one logical deployment on each complete threshold crossing;
- rejection of partial strokes;
- rearm behavior; and
- repeatability across cycles;
- seven rigid formed-staple retainers;
- four retained FEM nodes per staple; and
- 6 mm closure-station spacing.

The tissue comes from the real Dr.Anmar open-incision OpenUSD asset. The authoring
tool deterministically separates its left and right connected components into
two PhysX FEM surface bodies. Their outer bands remain attached to the fixture.
At each station, two visible approximation feet drive two wound-edge nodes per
flap toward a 0.8 mm residual gap. Trigger travel does not begin until that
pre-approximation is complete.

After a measured full-threshold crossing, the room creates a rigid formed
staple with collision enabled and retains the same four tissue nodes with
kinematic FEM attachments. Those attachments remain active after the stapler
fixture releases and advances, so prior stations continue to hold the wound
edges together. Reset removes every staple and retained attachment and restores
the original tissue state.

The intended tissue target is 111.9 kPa Young's modulus and 0.461 Poisson ratio.
The interactive PhysX implementation preserves the Young's modulus but bounds
Poisson ratio to 0.40, uses measured damping 0.2307 and at least 32 solver
position iterations. Status telemetry identifies this explicitly as a
`bounded_linear_tangent_for_interactive_physx` stability proxy.

This workflow models simulated tissue deformation and mechanical retention; it
does not model staple-leg puncture, metal plasticity during staple forming,
calibrated pull-out strength, wound healing, sterility, clinical performance,
or patient use. The retained FEM nodes are an explicit approximation of the
formed legs gripping tissue. Trigger torque, pusher force, contact friction and
all other physical values remain provisional until calibrated on an
instrumented physical bench.

The recorded native RTX 4090 simulator run on 2026-07-24 measured an initial
3.7818 mm wound gap and verified that the approximation phase reached 0.8 mm
with the trigger still at 0 degrees. Deployment occurred only after the actual
trigger crossed 24 degrees. The run completed all seven stations, created seven
rigid staple retainers, retained 28 FEM nodes, preserved a 0.8 mm local gap at
every closed station, consumed exactly seven simulated staples and held 6 mm
spacing with zero recorded spacing error. A separate 20.0002-degree partial
stroke produced no deployment or attachment. Reset removed every staple and
attachment, restored the 3.7818 mm gap and returned station 1 with a full
35-staple magazine. See
`tissue_closure_native_simulator_evidence.json` for the recorded evidence.
