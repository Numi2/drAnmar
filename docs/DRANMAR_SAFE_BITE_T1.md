# DrAnmar T1: Retained Needle Safe-Bite Approach

T1 begins only after simulator-derived receiver-only needle custody. The
receiver keeps both jaws physically engaged and moves the curved needle to a
sampled entry frame over the deformable Needle-Ready Tissue Unit.

## Practical success contract

Each episode samples either tissue flap, a bite distance 5–9 mm from the wound,
a 5–8 mm stand-off, and a 35–55 degree inward entry direction. Success requires:

- the full physical handover prerequisite or a reset snapshot captured from
  that same successful sequence;
- current bilateral receiver contact;
- needle-tip error no greater than 5 mm;
- tip-tangent error no greater than 25 degrees;
- needle-plane error no greater than 30 degrees;
- needle speed no greater than 50 mm/s and angular speed no greater than
  3 rad/s;
- conservative needle and receiver-tool clearance from tissue; and
- five consecutive 50 Hz control steps meeting every condition.

This is deliberately a reachable pre-contact envelope, not a clinically
prescribed bite or an unrealistic perfect-pose gate.

Receiver-tool clearance is not inferred from the tool-tip frame alone. The
runtime samples the tip-to-link segment for both physical jaws and uses the
minimum conservative tissue clearance across those samples.

The tissue is not allowed to free-fall. At every reset, the runtime restores
its default nodal state and kinematically pins exactly the 80 nodes in the
authored `anchor_outer` set. The wound edge and sampled safe-bite region stay
fully dynamic. The position-derived runtime selector was checked against the
versioned USD semantic set and matched all 80 nodes exactly.

## Efficient learning without reward hacking

The deterministic controller completes pickup, handover, and guarded transport.
The learned policy can change only the receiver's six pose axes after handover,
with a 0.003 action residual. It cannot command either jaw, change the giver
sequence, write contact state, or write task success.

Training may restore four of every five eligible resets from per-environment
snapshots captured after real handover success. A rotating per-environment
quota—not random chance—forces at least one complete handover in every five
episodes. Promotion disables snapshot restore and reports full-chain success
separately.

Positive dense credit is the clipped decrease in normalized position and
orientation error from the preceding step. A stationary policy earns exactly
zero; absolute proximity, holding contact, height, and phase occupancy earn
nothing. Terminal retained entry readiness earns positive credit, while
retention loss and premature tissue contact are independent terminal failures.

## Puncture transition

T1 does not mechanically disable puncture. Before entry readiness is armed,
needle or tool contact with tissue is a failure. After arming, contact is
allowed and an inward simulator-contact onset is recorded for the successor
curriculum. The training environment terminates at the armed state for sample
efficiency; the chain environment does not. In the chain environment, the
analytic controller continues along the sampled inward needle direction at a
bounded 0.03 action until physical contact begins, then records the transition
for the puncture backend.

Newton VBD owns T1 intact tissue and two-way contact. Persistent puncture,
tract, and thread passage are routed to the pinned CRESSim-MPM backend and
remain unqualified until native execution and synchronized physical
force-depth evidence pass. No policy-written puncture flag is accepted.

The machine-readable authority is
[`config/dranmar_safe_bite_t1.json`](../config/dranmar_safe_bite_t1.json).
