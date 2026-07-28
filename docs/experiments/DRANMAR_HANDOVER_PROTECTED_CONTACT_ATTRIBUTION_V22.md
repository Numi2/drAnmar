# Protected-Contact Body Attribution v22

## Question

The v15 recovery frontier reaches 1,164 retained handovers in 2,000 exact
seed-2361 environments, but five episodes cross the unchanged 2 N
protected-contact limit. Existing evidence identifies the receiver jaw and
phase, but the safety signal is only:

`total jaw force - needle-filtered force`

It cannot distinguish the giver instrument from the support table.

## Diagnostic contract

v22 adds four end-to-end-only PhysX contact views, one for each jaw. Each view
preserves one force column for every authored collision shape of the
counterpart PSM and one column for the table collision shape. Forces are
captured by the termination function before Isaac Lab automatically resets a
terminal environment.

The unchanged v15 controller, checkpoint, actions, rewards, success definition,
episode deadline, and 2 N termination remain authoritative. The added sensors
cannot write task state or change policy input.

The live Isaac/PhysX build requires every filter expression to resolve one
collision shape per environment. The initial body-path probe correctly failed
because several PSM bodies own two collision shapes and the table body path
resolved two entries. v22 therefore uses 17 explicit, versioned collider
columns. This is sufficient to separate counterpart jaws, counterpart arm or
wrist, and the support table.

## Runtime decision

First validate that all 17 partner columns resolve. Then replay the exact
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
