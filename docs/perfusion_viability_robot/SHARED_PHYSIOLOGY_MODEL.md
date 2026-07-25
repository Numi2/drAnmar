# Shared physiology model

All synthetic modalities consume one vascular state defined by `perfusion_network.json`. The model solves nodal pressures and edge flows from arterial and venous boundary pressures, edge resistance, obstruction multipliers, regional compression, and leak sinks. The same flow state drives tracer advection, thermal response, oxygenation, Doppler velocity, ultrasound patency, and viability fusion.

Corrective actions change a continuous recovery fraction rather than replacing the scenario label. Flow parameters blend monotonically toward the recovered state and every solve retains the mass-conservation check.

The network is a reduced-order simulation contract. It is not CFD, pharmacokinetics, or patient-specific physiology.
