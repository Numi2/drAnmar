# Needle Handover Pickup-Recovery Option v4

## Result

The recovery option improved the unchanged end-to-end handover on matched
512- and 1,200-environment populations. It learns only bounded giver XY
corrections after a simulator-observed pickup slip; nominal pickup, the
receiver policy, the full-task success definition, and all physics
terminations remain unchanged.

| Metric | Frozen v33 | Recovery option v4 | Change |
| --- | ---: | ---: | ---: |
| Retained handovers at 512 | 352 | 356 | +4 |
| Recovered successes at 512 | 35 | 39 | +4 |
| Retained handovers at 1,200 | 703 | 719 | +16 |
| Success rate at 1,200 | 58.58% | 59.92% | +1.33 points |
| Recovered successes at 1,200 | 51 | 66 | +15 |
| Exhausted pickup retries at 1,200 | 113 | 81 | -32 |
| Stable presentations at 1,200 | 905 | 921 | +16 |
| Receiver contacts at 1,200 | 756 | 771 | +15 |
| Windowed bilateral capture at 1,200 | 711 | 724 | +13 |

The matched three-seed, 3,600-episode gate reproduced the gain on every seed:

| Metric | Frozen v33 | Recovery option v4 | Change |
| --- | ---: | ---: | ---: |
| Retained handovers | 2,163 | 2,197 | +34 |
| Success rate | 60.08% | 61.03% | +0.94 points |
| Recovered successes | 210 | 243 | +33 |
| Exhausted pickup retries | 281 | 211 | -70 |
| Stable presentations | 2,733 | 2,800 | +67 |
| Receiver contacts | 2,301 | 2,344 | +43 |
| Windowed bilateral capture | 2,193 | 2,230 | +37 |

Per-seed retained-success changes were `+16`, `+10`, and `+8`.

## Structural change

The previous pickup-recovery learner had to wait for a complete receiver
handover even though it could control only giver XY after a slip. The v4
curriculum terminates its training option at a physically held, lifted, stable
presentation. Promotion is still judged only on the unchanged complete
handover.

The curriculum replays simulator-observed slip states across environments,
uses a 98% replay and 2% fresh-state schedule, and applies the qualified v34
recovery controller during training. A 90-update run at `1e-5` processed
2,949,120 frames, restored 3,014 physical recovery states, and reached a
67.69% tail option-success rate.

## Qualification boundary

Checkpoint `e73720b8430f8ef8d57d28d4013d5384257e4dc85d62aa206fb3ae4ccd698c80`
is the reproducible multiseed performance champion. It is not the default
promoted policy: the 3,600-episode candidate had six protected-surface
terminations versus five for v33, so the unchanged absolute zero-hard-failure
gate retains the prior default. Drops and excessive object force were zero for
both policies.

The hashed qualification record is
[`pickup-recovery-option-v4-multiseed-1200.json`](evidence/pickup-recovery-option-v4-multiseed-1200.json).
