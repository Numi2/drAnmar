# Needle Handover Recovery Latency v34

## Result

The recovery-latency challenger improved retained handover success on every
matched 2,000-environment seed without changing the checkpoint, reward,
success definition, episode horizon, first-attempt controller, or physics
terminations.

| Metric | Prior architecture | Recovery latency v34 | Change |
| --- | ---: | ---: | ---: |
| Retained handovers | 3,289 / 6,000 | 3,393 / 6,000 | +104 |
| Success rate | 54.82% | 56.55% | +1.73 points |
| Recovered successes | 226 | 329 | +103 |
| Stable presentations | 4,205 | 4,306 | +101 |
| Receiver contacts | 3,469 | 3,580 | +111 |
| Windowed bilateral capture | 3,324 | 3,425 | +101 |
| Excessive object force | 0 | 0 | unchanged |
| Protected-surface force | 13 | 13 | unchanged |
| Object drops | 2 | 4 | +2 |

Per-seed retained-success gains were `+31`, `+44`, and `+29` for seeds 17,
2361, and 4099. The candidate's 95% Wilson interval is 55.29% to 57.80%.

## Causal change

Only episodes already marked as pickup recovery change behavior:

- recovery carry lateral authority increases from `0.06` to `0.10`;
- receiver preposition height decreases from 25 mm to 15 mm after a recovery.

This reduces the time between relift and receiver acquisition. Reset-aligned
pickup, normal transport, receiver capture, release, retention, the 1,000-frame
horizon, and all hard termination thresholds remain unchanged.

## Qualification boundary

This is the new performance champion, but the versioned zero-hard-failure
promotion gate retains the prior runtime defaults because the candidate still
contains hard events. The aggregate protected-surface count did not increase
and excessive force remained zero; object drops increased from 2 to 4 in 6,000
episodes. The controls remain explicit runtime overrides until the safety gate
is resolved.

The formal hashed result is
[`recovery-latency-v34-multiseed-promotion.json`](evidence/recovery-latency-v34-multiseed-promotion.json).

## Rejected nearby changes

- Recovery carry `0.04`: reduced contact loss but caused late timeouts.
- Recovery carry `0.08`: nearly flat at 2,000 environments.
- Recovery relift authority `0.02`: destabilized custody and regressed to
  313/512.
- Contact-adaptive carry: safe but below fixed `0.10`.
- Curvature-remapped recovery grasp: zero recovered successes.
- Recovery grasp height `+0.5` mm: zero recovered successes and more
  protected-surface contacts.
