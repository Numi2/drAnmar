# Capability and evidence boundaries

The Atraumatic Exposure Robot is a source-available Dr.Anmar research workcell
with articulated bilateral capture and a force-limited control contract.

This revision was reviewed and hardened at source level only. No build, test,
simulator, validator, or gate was run for it. Prior runtime artifacts apply only
to their exact prior source revisions and cannot qualify this modified source.

The new adapter can reject incomplete or mismatched evidence and bind all 12
cell contact pairs, exact tissue mesh bodies, attachment prims, ROI visibility
counts, raw native record IDs, and the post-physics clock. It locks the full
workcell/source/calibration/visibility/ROI registration, rejects envelope and
per-source raw-record replay across intervals, rejects fractional cell
identities, and canonicalizes reversed contact-vector
orientation. Controller state is phase-synchronized; release evidence is
preflighted before attachment removal; hold control re-synchronizes from measured
post-physics joint positions; any release latches until a four-stage explicit
recapture is confirmed from a new topology revision. The handshake requires an
adjacent no-attachment preflight, attachment authoring before the evidence
clock advances, and consecutive confirmation intervals meeting configurable
force, area, interval-count, dwell, and maximum-interval gates while episode,
environment, source registration, and topology lineage remain fixed. Each cell carries an
exact attachment-reaction vector as well as exact contact evidence; the
conservative transmitted-load proxy adds their magnitudes and may intentionally
double count overlapping load paths.

The fenestrated and microcup variants alter visible geometry only. They share
the same box capture-cell colliders, material behavior, and attachment contract,
so this revision cannot support comparative pad-physics claims.

There is not yet a native Isaac provider that populates that contract.
Therefore this source does not establish pad contact
force, retained tissue capture, exposure efficacy, tissue safety, physical
calibration, clinical performance, or patient use.

Promotion to runtime evidence requires repeated revision-bound runs from that
provider, including raw contact and visibility records, attachment topology
transitions, convergence sweeps, and failure cases. Promotion to physical
realism requires held-out instrumented retraction, slip, pullout, trauma, and
visibility correlation across representative tissue states.
