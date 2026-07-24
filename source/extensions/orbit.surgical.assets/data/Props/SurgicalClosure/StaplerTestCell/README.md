# Dr.Anmar Stapler Test Cell 0.1.0

This research fixture holds the Dr.Anmar articulated skin stapler in a cradle
with a six-DOF velocity-damped virtual fixture and exposes a bounded virtual
trigger actuator. It removes the need for a robot hand that can grasp a
gun-shaped housing.

The cell is intended to measure simulation mechanism behavior:

- commanded versus observed trigger travel;
- synchronized pusher travel;
- measured housing translation and rotation error;
- exactly one logical deployment on each complete threshold crossing;
- rejection of partial strokes;
- rearm behavior; and
- repeatability across cycles.

The coupon is a non-contact synthetic alignment target, so it cannot block the
authored trigger or pusher travel. The fixture does not model or validate
tissue penetration, staple forming, closure strength, wound healing, sterility,
clinical performance, or patient use. Trigger torque, pusher force, friction,
travel and all other physical values remain provisional until calibrated from
a physical instrumented bench.

The native RTX 4090 qualification on 2026-07-24 measured a 20-degree partial
stroke without deployment, one full deployment at 24.6002 degrees, 8.9742 mm
maximum pusher travel, no trigger-limit violation and maximum fixture error of
0.1207 mm / 0.0685 degrees.
