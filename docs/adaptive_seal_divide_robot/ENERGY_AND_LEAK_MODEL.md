# Conditioning, interlock, and stump observations

The controller consumes registered temperature, impedance, voltage/current
energy, bilateral contact, cohesive-interface, and shared-vessel observations.
It derives a bounded conditioning fraction as a provisional control proxy.
That fraction is deliberately not named or treated as seal maturity.

The blade interlock requires same-interval evidence for:

1. registered tissue centering;
2. bilateral compression within force, balance, area, and slip policy;
3. conditioning without overtemperature or impedance fault;
4. intact shared cohesive response and the exact live seal attachments;
5. observed stump leak below the limit at sufficient upstream pressure;
6. acceptable vessel-wall damage;
7. a retracted guard;
8. consistent blade joint/tip position and registered vessel contact; and
9. bridge topology no further released than authorized blade progress.

Stump flow is read from exact shared `VesselObservation` state. It is not
predicted from an authored seal fraction or residual-gap scalar.

The source does not implement spatial current density, tissue electrical
transport, field-dependent heating, water loss, steam, smoke chemistry,
histology, collateral thermal spread, biochemical fusion, or calibrated
generator waveforms. The conditioning proxy and thresholds are provisional
engineering values.
