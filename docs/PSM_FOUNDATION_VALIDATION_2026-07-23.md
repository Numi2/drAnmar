# PSM foundation validation — 2026-07-23

This report records what was actually exercised on Gilgamesh's RTX 4090 against the pinned Isaac for Healthcare
v0.7 checkout. It separates control/data compatibility from task quality.

## Qualified

- A native single-PSM absolute-IK reach rollout produced a 150-frame HDF5 episode with seven policy actions,
  eight physical joint states, RGB, and a maximum joint-action round-trip error below `4e-9`.
- NVIDIA's unmodified dataset component converted that episode to LeRobot. The Dr.Anmar GR00T PSM loader
  accepted the seven-action, eight-state, single-camera dataset and materialized a transformed sample.
- The browser workstation ran in an isolated room on port 2496. A real keyboard drive produced a 90-step
  recording with `actions (90,7)`, `cartesian_actions (90,7)`, `resolved_joint_targets (90,7)`,
  `obs/joint_pos (90,8)`, and synchronized `obs/room (90,240,360,3)`.
- NVIDIA's dual-PSM reach state machine completed successfully. Its finalized HDF5 contains
  `actions (150,14)`, `resolved_joint_targets (150,14)`, and `native_ee_pose_w (150,14)`;
  the maximum per-arm round-trip error was below `2e-9`.
- The local handover MDP started and stepped on GPU with 14D relative-IK actions, 96D finite policy observations,
  13 reward terms, and 5 termination terms. Its phase state remained correctly at approach under zero action.
- HDF5 finalization is idempotent and validates both logical grippers independently.
- The live Dr.Anmar room on port 2396 was not restarted or modified during qualification.

Validation artifacts are under:

```text
<validation-root>/psm-foundation/
```

## Correct rejection

The workstation's short manual nudge did not complete needle lift. Its HDF5 records `success=false`, and NVIDIA's
converter excluded it from expert training data. The recording remains useful for control diagnostics, but it
was not relabeled or promoted.

## Not yet qualified

- NVIDIA's current scripted needle-lift grasp was physically successful in only 3/5 attempts in one run and 1/5
  in another. A table-clearance variant failed 0/5 and was discarded rather than shipped.
- The handover MDP is implemented, but no native rollout has yet physically completed
  approach → giver grasp → present → dual grasp → stable receiver ownership.
- No GR00T checkpoint has been trained from the reach diagnostic dataset. Training must begin with accepted,
  successful task demonstrations and end with held-out physical rollouts.

## Next acceptance gates

1. Make the native thin-needle grasp repeatable without attachment, teleport, or success relabeling.
2. Record a set of accepted needle-lift demonstrations through the workstation and convert them unchanged.
3. Complete and repeat the five-phase physical handover, then accept bimanual demonstrations for policy use.
