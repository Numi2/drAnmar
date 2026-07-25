# Force-aware exposure control

Pad normal force is estimated from the authored compliance-axis displacement
and velocity. The supplied controller has two loops:

1. an outer ROI-visibility loop increases lateral separation and lift while
   exposure is below target;
2. independent force limits unload either side before continuing exposure.

A bilateral asymmetry correction prevents one pad from carrying a much
larger load than the other. Hard overload commands immediate unloading and
allows the capture controller to release the affected pad.

The numerical thresholds are provisional research seeds, not tissue-specific
safety limits. Calibration requires instrumented physical specimens and the
selected target procedure.
