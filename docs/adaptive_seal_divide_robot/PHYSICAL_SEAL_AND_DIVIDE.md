# Physical seal-and-division contract

## Intact tissue representation

The demo vessel contains two watertight deformable halves connected by eight
temporary bridge-pin attachments and held by distal fixture attachments. This
is a controlled topology surrogate: it does not represent continuous cutting,
wall-layer fracture, or a real intact lumen across the future cut plane.

## Temporary jaw compression

The compression controller creates four deformable-to-jaw attachments. Its
production update accepts only the exact registered bilateral post-physics
contact samples from `SealDivideSceneEvidence`. Caller-reported force is not
admissible. Hard overload releases the temporary attachments; attachment
existence alone does not establish safe or adequate compression.

## Mechanical seal-band qualification

Each stump band has exact registered upper/lower attachment identities and a
shared cohesive-interface response. A band is mechanically qualified only
when the same evidence interval establishes:

- provisional temperature/impedance/electrical conditioning without fault;
- measured compression, balance, contact area, and slip within policy;
- a live exact attachment set;
- cohesive integrity below damage limits and without failure;
- shared-vessel leak below the provisional limit at sufficient pressure; and
- wall damage below the provisional limit.

This is not seal “maturity,” biochemical cure, tissue fusion, healing, burst
strength, or clinical seal efficacy.

## Evidence-authorized division

The blade controller reconciles registered joint position, blade-tip position,
guard state, vessel contact, and live bridge topology. Forward motion can
release bridge attachments only while the same evidence interval satisfies
the mechanical interlock. Observed bridge loss ahead of previously authorized
blade progress is a failure, not successful division.

Complete bridge release is only a simulator topology event. It does not prove
cut quality, sealed stumps, thermal margin, or patient safety.
