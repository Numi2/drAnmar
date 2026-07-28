# DrAnmar patency and leak verification

The mandrel and centering cage provide a reference for minimum lumen radius,
centerline offset, and axis alignment. `LumenPatencyController` consumes exact
scene-bound lumen node sets and reports geometry thresholds only;
`flow_connected_patency_verified` remains false.

`PressureDecayLeakController` consumes the envelope's exact pressure and
measured leak-flow records. It tracks pressure, integrated leak volume,
observation time, and a provisional simulator seal-test threshold.
`LeakTestLedger` conserves test medium across reservoir, chamber, active leak,
collection, spill, and discard buckets, but the ledger is not a native
fluid-structure provider.

The leak model is a research benchmark. It is not a clinical leak test, does not reproduce full fluid-structure interaction, and must be calibrated against the selected tissue, test medium, pressure protocol and instrumentation.
