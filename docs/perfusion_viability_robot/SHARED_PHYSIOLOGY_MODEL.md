# Shared physiology model

All modalities consume one vascular state defined by `perfusion_network.json`. The model solves nodal pressures and edge flows from arterial and venous boundary pressures, edge resistance, obstruction multipliers, regional compression, and leak sinks. The same flow state drives tracer advection, thermal response, oxygenation, Doppler velocity, ultrasound patency, and viability fusion.

The network is a reduced-order simulation contract. It is not CFD, pharmacokinetics, or patient-specific physiology.
