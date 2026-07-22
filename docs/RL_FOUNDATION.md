# Dr.Anmar RL foundation

Dr.Anmar uses one canonical control contract for doctor operation, expert demonstrations, replay, and reinforcement learning:

- registered non-play `*-IK-Rel-v0` environments;
- six normalized Cartesian relative-pose actions per instrument;
- one normalized binary gripper action per instrument;
- action bounds of `[-1, 1]`;
- physical increments of 10 mm translation and 0.05 rad rotation at full-scale input.

Joint-position variants remain upstream compatibility environments. The Dr.Anmar training entry points do not select them.

## Needle lift MDP

The policy observation is native Isaac Lab state: robot joint position and velocity, measured end-effector pose, full needle pose, needle linear and angular velocity, commanded pose, per-jaw filtered contact force, and the previous normalized action.

Rewards cover tool approach, bilateral physical grasp, lift, coarse and fine target position, target orientation, stable object motion, bounded object-contact force, unintended jaw contact, PSM remote-center-link motion, action rate, and joint velocity. Completion requires bilateral contact, position and orientation tolerance, and bounded linear and angular velocity. Excessive contact and object dropping terminate the episode.

## Needle handover MDP

The handover policy observes both robots, both tool poses, the needle pose in both robot frames, needle velocity, four per-jaw contact forces, the receiver target, the physical procedure phase, and the previous normalized action.

The phase state is monotonic and contact-grounded:

1. approach;
2. giver bilateral grasp;
3. lifted presentation near the receiver;
4. simultaneous giver and receiver bilateral contact;
5. giver release with receiver-only stable ownership at the commanded recovery pose.

Closing a jaw near the needle cannot advance the phase. Dr.Anmar reads filtered PhysX contact tensors and never attaches, teleports, or otherwise scripts object ownership.

## Runtime validation on Gilgamesh

On 2026-07-22, the isolated RTX 4090 validation workspace loaded and stepped both canonical environments on CUDA:

- needle lift: 52 policy observations and 7 normalized actions;
- needle handover: 96 policy observations and 14 normalized actions;
- all robot joint positions and velocities remained finite;
- one complete RSL-RL PPO iteration ran for each environment with 8 parallel environments and 192 total simulation transitions.

This proves environment construction, tensor shapes, native simulation stepping, reward/termination evaluation, and optimizer integration. It does not prove policy convergence or clinical validity.

## Calibration and qualification still required

- Measure real dVRK jaw/contact data and calibrate the current research force envelopes.
- Qualify needle pose conventions and orientation tolerances against surgeon-reviewed grasps.
- Demonstrate every handover phase under manual teleoperation and verify that invalid phase ordering never succeeds.
- Run multi-seed learning curves, ablations, and held-out initial-pose evaluation for lift and handover.
- Validate RCM and protected-surface thresholds against the intended trocar, anatomy, and procedure setup.
- Establish sim-to-real observation, latency, actuator, force, and safety-envelope correspondence before any physical-robot claim.
- Migrate or discard legacy demonstrations recorded before the normalized relative-IK action contract; do not mix numerical action conventions in one dataset.
