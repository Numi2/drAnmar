# Protected-Contact Body Attribution v22

## Question

The v15 recovery frontier reaches 1,164 retained handovers in 2,000 exact
seed-2361 environments, but five episodes cross the unchanged 2 N
protected-contact limit. Existing evidence identifies the receiver jaw and
phase, but the safety signal is only:

`total jaw force - needle-filtered force`

It cannot distinguish the giver instrument from the support table.

## Diagnostic contract

v22 records the responsible jaw's exact native non-object force vector plus the
pre-reset poses of both PSMs' insertion, roll, pitch, yaw, and jaw bodies. It
also records needle position, responsible-jaw height above the known table
surface, and distances to the reporter's other jaw and every counterpart tool
body. All tensors are captured inside the unchanged termination before Isaac
Lab automatically resets a terminal environment.

The unchanged v15 controller, checkpoint, actions, rewards, success definition,
episode deadline, and 2 N termination remain authoritative. The added sensors
cannot write task state or change policy input.

Two GPU force-matrix approaches were rejected before scale. Body paths failed
because several PSM bodies own multiple colliders. Explicit collider paths
initialized, but PhysX reported complex articulation and table filters as
unsupported. A native PhysX contact-report subscription was then evaluated in
the exact 2,000-environment GPU-direct replay. It preserved the 1,164/2,000
outcome and all five protected-force terminals but emitted no Python contact
events, so it is rejected rather than treated as negative-contact evidence.

The replacement is backend-independent simulator state already present in the
GPU tensors. Body-origin proximity is a diagnostic geometric clue, not a claim
of exact collider identity.

## Runtime decision

Replay the exact seed-2361 2,000-environment frontier. No controller change is
allowed until the five protected terminals have a dominant geometric category:

1. counterpart arm or wrist;
2. counterpart jaws;
3. support table; or
4. own-jaw contact; or
5. still ambiguous.

The subsequent correction must target only the measured category and retain at
least 1,140/2,000 successes with at most four protected-contact terminals.

These are simulator diagnostics, not physics calibration or clinical
validation.
