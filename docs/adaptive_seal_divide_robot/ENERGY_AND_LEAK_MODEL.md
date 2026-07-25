# Energy, Interlock, and Leak Model

The runtime helper uses a lumped thermal model per seal zone:

`C dT/dt = P_absorbed - h (T - T_ambient)`

Compression affects absorbed power. A temperature-dependent dose integrates into a bounded seal-maturity state. Impedance is updated as a research proxy for heating and desiccation.

The blade interlock requires:

1. compression inside the configured force window;
2. both seal zones above the maturity threshold;
3. no overtemperature or impedance fault;
4. predicted flow from both future stumps below the configured limit;
5. the ceramic guard fully retracted.

The leak estimate is a reduced-order orifice-flow model driven by seal maturity, residual gap, pressure, and damage. It is not CFD and is not evidence of clinical seal integrity.

The thermal state is lumped per zone. Spatial thermal spread, tissue
histology, smoke chemistry, electrical current paths, generator waveforms, and
collateral injury are not modeled or calibrated.
