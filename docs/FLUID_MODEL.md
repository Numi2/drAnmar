# Irrigation and aspiration model

The runtime creates a PhysX PBD liquid particle system and emits particles from
ten nozzle origins. Every particle has a declared volume. `FluidLedger` debits
the reservoir only for particles actually authored and tracks:

- remaining reservoir volume;
- emitted volume;
- active particle volume;
- aspirated volume;
- spilled volume;
- discarded volume;
- balance error.

The emitter uses a deterministic seed by default to add bounded per-particle
directional spread around each authored jet direction. The suction controller
applies a converging field within the annular capture volume, adds a tangential
component to guide particles toward the throat, and removes particles that cross
the throat radius only while the collection canister has volume available.
Captured particle volume is credited to the collection canister. Explicit
simulation culling must be transferred through `FluidLedger.discard()` so the
balance remains auditable.

This provides deterministic volume accounting and visible liquid interaction.
It is not a Navier–Stokes solver, pressure-flow calibration, aerosol model, or
clinical irrigation-dose model.
