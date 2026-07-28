# Force-Projected Recovery Transport v16

## Why binary contact was insufficient

The v15 phase and bilateral-contact gate recovered the full 0.08 performance:
393/600 and 1,164/2,000 retained handovers. It reduced protected-surface
terminals from six to five, but the paired safety limit permits at most four.

Both jaws touching does not prove that a recovered grasp is well seated. The
first-lift diagnostics support a continuous custody measure: successful
episodes have a 10th-percentile weaker-jaw force of 0.025 N, while
lifted-without-acquisition episodes are at 0.0166 N.

## Safety projection

After lifted recovery custody, v16 maps the weaker giver jaw's live force to
the allowed lateral authority:

- at or below the existing 0.01 N lift-contact threshold: 0.06 authority;
- at or above 0.025 N on the weaker jaw: 0.08 authority;
- between them: smoothstep interpolation.

Before lifted custody the authority remains exactly 0.06. This preserves a
continuous action field, avoids a force-threshold trajectory switch, and
concentrates faster transport on physically well-seated grasps.

Rewards, success, timeout, receiver control, grippers, release, and safety
terminations remain unchanged. Qualification still requires at least 393/600,
at least 1,140/2,000, zero drops and excessive object-force events, and no more
than four protected-surface terminals at scale.

## Result

v16 reached 395/600, but the scale result regressed to 1,141/2,000 with 169
pickup attempts exhausted and six protected-surface terminals. Instantaneous
force projection is rejected. The controller returns to the v15 phase and
bilateral-contact gate while v17 adds collision attribution to the evidence.
