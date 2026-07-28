# Force-aware exposure control

Production mechanics evidence contains, for each of six registered cells per
pad, the exact post-physics contact vector and the exact reaction vector on the
cell from its registered attachment. The release proxy is the sum of the
contact-resultant norm and attachment-reaction norm. Those signals can overlap,
so this deliberately may double count; it is a conservative gate that prevents
a co-directed reaction from being hidden by a maximum. ROI visibility is an
integer visible/total sample count bound to the registered camera and ROI. The
supplied controller has two loops:

1. an outer ROI-visibility loop increases lateral separation and lift while
   exposure is below target;
2. independent force limits unload either side before continuing exposure.

A bilateral asymmetry correction prevents one pad from carrying a much larger
measured load than the other. Hard overload commands immediate unloading and
releases scene-confirmed attachments on the affected pad. Every commanded
phase synchronizes the controller's internal carriage and lift state before
closed-loop control. Every hold interval then resynchronizes from exact
post-physics carriage and lift positions before integrating. This prevents
lag on those four integrated axes from becoming hidden controller state. Pitch
and compliance remain fixed commanded targets during hold rather than
closed-loop states; all eight tool joints are verified only during the capture
handshake. Intervals longer than 1/30 s fail closed instead of being
extrapolated.

Any scene-confirmed attachment loss, contact below the configured force/area
retention floors, slip, or overload release latches safe relief. Hold cannot
silently resume against fewer or zero attachments; an explicit, fully
successful recapture is required and partial recapture is rolled back.

Capture is a four-stage handshake:

1. command the capture pose;
2. consume the immediately adjacent, no-attachment post-physics preflight with
   the same episode, environment, source registration, and topology lineage,
   within the configured maximum evidence interval;
3. author all 12 exact attachments before the evidence clock advances; and
4. consume consecutive post-physics confirmation intervals under one new
   topology, each within that maximum interval, until both the configured
   interval count and dwell time are met.

Every cell must meet configurable nonzero normal-force and contact-area
floors throughout. Retraction remains prohibited until that confirmation
sequence completes.

`estimate_pad_force_n()` remains available only as a task-design proxy.
Caller-authored forces and compression-derived estimates are rejected by the
capture and sequence controllers. A native bridge must still supply contact,
attachment, visibility, raw-record, and monotonic clock evidence.

The numerical thresholds are provisional research seeds, not tissue-specific
safety limits. Calibration requires instrumented physical specimens and the
selected target procedure.
