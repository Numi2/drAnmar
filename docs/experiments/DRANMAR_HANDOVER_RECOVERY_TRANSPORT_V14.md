# Recovery-Transport Controller v14

## Scale result and final disposition

The recovered-lift bottleneck did not require a new option selector or more
learned giver authority. It required the existing recovery transport to reach
the already-qualified presentation pose before the unchanged episode deadline.

v14 tested increasing only the recovered-pickup lateral carry limit from 0.06
to 0.07.
First-attempt transport, grasp, lift, receiver acquisition, gripper commands,
release, reward, success, timeout, and hard terminations are unchanged.

## Paired 2,000-environment result

The incumbent and candidate used seed 2361, the same initial-state population
hash, the same environment-contract hash, and the complete first terminal
outcome from every environment.

| Metric | Incumbent | v14 |
| --- | ---: | ---: |
| Retained handovers | 1,130 | 1,148 |
| Recovered retained handovers | 99 | 117 |
| Lifted without receiver acquisition | 708 | 691 |
| Stable presentations | 1,451 | 1,471 |
| Receiver contacts | 1,203 | 1,222 |
| Protected-surface terminals | 2 | 4 |
| Drops | 0 | 0 |
| Excessive object-force terminals | 0 | 0 |

The retained-success gain was 18/2,000, or 0.9 percentage points. The
protected-surface increase is exactly 0.1 percentage points, equal to the
predeclared non-inferiority limit.

A fresh default-contract replay then produced 388/600 against the 390/600
incumbent development result. v14 is therefore rejected for cross-population
inconsistency; the 2,000-environment gain remains useful causal evidence but is
not a promotion.

## Rejected nearby variants

The 0.08 limit produced 1,164/2,000 retained handovers but six
protected-surface terminals, exceeding the safety non-inferiority bound. It is
not promoted. Extending the global lateral ramp from 10 mm to 15 mm reduced
the development result to 370/600 and is also rejected.

## Structural correction

The deadline-recovery training task declared controller overrides but the
training branch configured only the adapter and never applied those overrides.
v14 explicitly applies and records the deadline controller configuration.
All recovery curricula now declare the same 0.07 carry limit and the qualified
25 mm receiver preposition used by serving.

v15 keeps 0.06 authority during relift and contact loss, and enables 0.08 only
after lifted custody with live bilateral giver contact. It must beat both
development and scale incumbents before the preregistered multi-seed promotion
gate. These are simulator results, not physics calibration or clinical
validation.
