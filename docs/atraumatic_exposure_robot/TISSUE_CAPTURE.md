# Tissue and distributed capture

The included benchmark uses two portable triangular flap meshes over a
central ROI. Runtime code cooks each flap through the current surface-
deformable route and anchors the outer bands to the fixture.

Each contact pad contains six small capture volumes. The controller ranks
tissue vertices by capture-cell overlap and creates one verified vertex
attachment per cell. If a portable benchmark pose has fewer than four
overlapping vertices, the nearest four are selected deterministically.
This distributes traction spatially and allows local loss of capture without
making the whole pad detach at once.

The nearest-vertex fallback and capture cells are simulation contracts. They
do not claim to reproduce a
specific suction pressure, tissue injury threshold, ischemia response, or
clinical retractor design.
