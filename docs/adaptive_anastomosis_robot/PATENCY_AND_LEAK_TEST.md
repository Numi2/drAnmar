# DrAnmar patency and leak verification

The mandrel and centering cage provide a reference for minimum lumen radius, centerline offset and axis alignment. `LumenPatencyController` evaluates user-supplied radial samples and reports minimum radius, mean radius, area fraction, offset, axis error and pass state.

`PressureDecayLeakController` is a reduced-order, dimensionally consistent chamber model. Pump inflow and orifice outflow change pressure through an effective chamber compliance. Effective leak area depends on residual edge gap, retained staple fraction and reinforcement bond fraction. It tracks instantaneous leak flow, pressure, integrated leak volume, observation time and pass state. `LeakTestLedger` conserves test medium across the reservoir, isolated chamber, active leaked medium, collection, spill, and discard buckets.

The leak model is a research benchmark. It is not a clinical leak test, does not reproduce full fluid-structure interaction, and must be calibrated against the selected tissue, test medium, pressure protocol and instrumentation.
