# Dr.Anmar Tissue Closure Bench 0.2.0

This research fixture holds the Dr.Anmar articulated skin stapler in a cradle
with a six-DOF proportional-velocity virtual fixture and exposes a bounded virtual
trigger actuator. It removes the need for a robot hand that can grasp a
gun-shaped housing. The fixture now indexes the stapler across seven guided
stations on the Dr.Anmar open-incision suturable-tissue asset.

The cell is intended to measure simulation mechanism behavior:

- commanded versus observed trigger travel;
- synchronized pusher travel;
- measured housing translation and rotation error;
- exactly one logical deployment on each complete threshold crossing;
- rejection of partial strokes;
- rearm behavior; and
- repeatability across cycles;
- seven deterministic formed-staple placement proxies; and
- 6 mm closure-station spacing.

The tissue is the real Dr.Anmar two-flap OpenUSD asset, rotated so each formed
staple crown bridges its open incision. In this room it remains static visual
geometry with collisions disabled: the current backend cannot penetrate or
deform it, and a rigid collision response would push the fixture away rather
than represent closure. Each successful measured threshold crossing records a
visible, non-dynamic formed-staple proxy at the active station, then the fixture
advances after release.

This workflow does not model or validate tissue penetration, tissue
deformation, staple forming, closure strength, wound healing, sterility,
clinical performance, or patient use. Trigger torque, pusher force, friction,
travel and all other physical values remain provisional until calibrated from
a physical instrumented bench.

The recorded native RTX 4090 qualification on 2026-07-24 rejected an
18.7488-degree partial stroke without deployment, completed all seven guided
stations, consumed exactly seven simulated staples, held 6 mm spacing with
zero recorded spacing error, and reported no trigger-limit violation. Across
the seven cycles, peak trigger travel was 26.9193–27.0743 degrees, peak pusher
travel was 8.9209–8.9729 mm, and the largest observed fixture transient was
1.2893 mm / 0.2203 degrees. A separate reset check cleared every placement and
restored station 1 with a full 35-staple magazine. See
`tissue_closure_qualification_report.json` for the recorded evidence.
