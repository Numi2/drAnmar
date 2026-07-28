# Protected-Structure Safety Contract

The tissue substrate includes independent vessel, nerve, and duct assets. Each structure has two rigid segments joined by a removable continuity joint and attached to the target tissue at runtime.

The private task proxy calculates caller-supplied local tool coordinates against
undeformed authored centerlines. Minimum provisional clearances differ by
modality, but this geometry is not synchronized with live structure
deformation, tool contact, or a physics step. The proxy compares distance to the
centerline directly and does not subtract the authored structure radius, so its
thresholds are not structure-surface clearances. Public protected-structure
action authorization fails closed; there is no host override that can establish
a safety result.

The legacy private proxy can illustrate three synthetic failure labels:

- vessel: deactivate a rigid continuity joint and increment a scalar blood-loss
  proxy;
- nerve: deactivate a rigid continuity joint and set a scalar conduction proxy
  to zero;
- duct: deactivate a rigid continuity joint and increment a scalar leak proxy.

Those mutations are not exposed as public injury outcomes. The rigid halves and
visual variants are not vessel flow, duct transport, nerve conduction, tissue
failure, or patient physiology. No same-step evidence bridge currently connects
tool contact to shared vessel/duct/nerve mechanics or a patient blood/bile
ledger. This is an uncalibrated task representation, not an injury predictor.
