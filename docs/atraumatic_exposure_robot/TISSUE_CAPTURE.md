# Tissue and distributed capture

The included benchmark uses two portable triangular flap meshes over a
central ROI. Runtime code cooks each flap through the current surface-
deformable route and attaches one local outer-edge vertex cluster per flap to
a fixture proxy.

Each contact pad contains six small capture volumes. The controller ranks
tissue vertices by capture-cell overlap and creates one verified vertex
attachment per cell. A cell with fewer than four genuinely overlapping
vertices fails capture; it does not substitute non-overlapping nearest
vertices. All 12 exact rigid targets are registered before mutation; overlap is
then checked while each attachment is authored. If any cell fails, every
attempted attachment path is removed with a verified scene-removal
postcondition before the controller reports failure.

Actor-root tissue paths are resolved once to the exact `SimulationMesh` body
used by both the attachment and evidence contract. Any later local loss of
capture changes the attachment topology and latches safe relief until an
explicit full recapture is confirmed from a fresh topology revision. Merely
authoring attachment prims never clears the latch.

After capture, local or whole-pad release requires exact evidence for every
registered cell: its contact pair, attachment prim, contact force vector,
attachment-reaction vector on the cell, native record identity, and
post-physics interval. Contact-resultant and attachment-reaction magnitudes are
added as a conservative transmitted-load proxy; that may double count an
overlapping load path, but cannot hide a co-directed reaction. The
overlap-ranked vertex attachment and capture cells are simulation contracts.
Attachment existence is not proof of retained load, and the model does not
reproduce calibrated suction pressure, pullout, tissue injury, ischemia, or a
clinical retractor design.
