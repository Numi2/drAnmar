# Protected-Structure Safety Contract

The tissue substrate includes independent vessel, nerve, and duct assets. Each structure has two rigid segments joined by a removable continuity joint and attached to the target tissue at runtime.

The controller calculates local tool clearance against each authored centerline. Minimum provisional clearances differ by modality. Scissors and energy actions are blocked when the nearest structure lies inside the configured safety envelope unless the host task explicitly overrides the interlock.

An override has a physical consequence:

- vessel injury removes continuity and activates blood-loss state;
- nerve injury removes continuity and sets conduction to zero;
- duct injury removes continuity and activates a leak state.

This is a research complication model, not a validated injury predictor.
