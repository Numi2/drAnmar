# Recovery Receiver Shaft Guard v23

## Root cause

The exact v22 replay preserved the v15 frontier and captured all five
protected-force terminals before reset. Their forces were lateral, with jaw
origins 30–59 mm above the table. Offline collision-mesh reconstruction showed
that four post-lift receiver paths crossed the giver's main insertion shaft or
distal roll geometry 28–45 mm behind the needle. One pre-lift episode was a
direct distal-jaw collision.

The previously rejected tip-to-tip barrier did not cover this geometry.

## Control contract

v23 models the giver's insertion shaft as a capsule beginning 25 mm behind its
tool tip and ending at its fixed remote center. During recovery phase 2 only,
the controller measures the receiver tip and proximal jaw point against that
shaft. If a command would cross the 15 mm clearance inside the 18 mm activation
band, it removes only the inward component. Tangential motion, orientation,
gripper control, and the distal needle-acquisition zone remain unchanged.

The checkpoint, rewards, success definition, episode deadline, and unchanged
2 N hard termination remain authoritative.

## Exact results

| Metric | v15 | v23 |
| --- | ---: | ---: |
| Retained handovers, 2,000 | 1,164 | 1,166 |
| Recovered retained handovers | 133 | 135 |
| Pickup attempts exhausted | 144 | 138 |
| Protected-force terminals | 5 | 3 |
| Drops | 0 | 0 |
| Excessive object-force terminals | 0 | 0 |

The guard activated in 230 environments for 623 control steps. It passes the
predeclared scale gates of at least 1,140 retained handovers and at most four
protected-force terminals. Evidence is frozen at
`docs/experiments/evidence/recovery-shaft-guard-v23-seed2361-2000.json` with
SHA-256
`387b38b78c6c65fe23d0029a96873125f016dec52481290e455712cec3dcd0d0`.

This is an Isaac Lab simulator result. It is not physics calibration, clinical
validation, or authorization for patient use.
