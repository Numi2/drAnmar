# Protected-Contact Body Attribution v22

## Question

The v15 recovery frontier reaches 1,164 retained handovers in 2,000 exact
seed-2361 environments, but five episodes cross the unchanged 2 N
protected-contact limit. Existing evidence identifies the receiver jaw and
phase, but the safety signal is only:

`total jaw force - needle-filtered force`

It cannot distinguish the giver instrument from the support table.

## Diagnostic contract

v22 subscribes to the native PhysX contact-report event stream during an
end-to-end play benchmark. On every control step it keeps the strongest
non-object contact pair for each jaw and environment. When the unchanged
protected-force termination fires, the evidence records the exact
`collider0`/`collider1` pair from that same step before discarding the buffer.

The unchanged v15 controller, checkpoint, actions, rewards, success definition,
episode deadline, and 2 N termination remain authoritative. The added sensors
cannot write task state or change policy input.

Two GPU force-matrix approaches were rejected before scale. Body paths failed
because several PSM bodies own multiple colliders. Explicit collider paths
initialized, but PhysX reported complex articulation and table filters as
unsupported. Neither result is accepted as evidence. The event stream is the
supported native path that exposes the collider identities directly.

## Runtime decision

First validate native contact events on a small population. Then replay the exact
seed-2361 2,000-environment frontier. No controller change is allowed until the
five protected terminals have a dominant body category:

1. counterpart arm or wrist;
2. counterpart jaws;
3. support table; or
4. still unattributed.

The subsequent correction must target only the measured category and retain at
least 1,140/2,000 successes with at most four protected-contact terminals.

These are simulator diagnostics, not physics calibration or clinical
validation.
