# Bimanual webcam teleoperation

Dr.Anmar's webcam controller is a simulation-only master-pose input for the two native NVIDIA PSM instruments.

## Operator behavior

- Physical left hand controls Instrument 1; physical right hand controls Instrument 2.
- Hand translation maps to relative depth, lateral motion, and vertical motion.
- Palm-frame orientation maps to bounded roll, pitch, and yaw.
- Only thumb and index spacing controls proportional jaw aperture.
- Explicit per-arm and global Engage/Freeze buttons are the motion clutch. No other finger pose is a command.
- A new engagement captures a new hand anchor, so freezing and recentering cannot move the robot.

Calibration records numeric neutral palm scale and normalized closed/open thumb–index spacing per browser camera in
`localStorage`. The webcam stream and MediaPipe landmarks stay inside the browser.

## Runtime contract

The browser sends:

```text
POST /api/teleop/hands
{
  sequence,
  hands: [{
    arm,
    tracked,
    motion_engaged,
    translation_offset_m: [depth, lateral, vertical],
    rotation_vector_rad: [roll, pitch, yaw],
    aperture_normalized,
    confidence
  }]
}
```

The endpoint accepts one complete frame at a time, validates unique available arms, finite values, confidence and
aperture in `0..1`, translation inside ±0.12 m, rotation inside ±0.8 rad, the active operator lease, and strictly
increasing sequence numbers. A rejected frame does not alter robot state.

Each arm keeps a cumulative master-pose target and the displacement already consumed by the simulator. Every
simulation step divides the remaining displacement by the six real scales read from that arm's active NVIDIA
`DifferentialInverseKinematicsAction`, clips the normalized command, and advances the consumed displacement by the
amount actually requested. All joint limits, articulation control, contact, and PhysX behavior remain native.

The gripper remains one-dimensional. `-1` is the exact configured closed endpoint, `+1` is the exact open endpoint,
and intermediate values linearly interpolate the two symmetric jaw targets. Recorded policy `actions` remain binary
for compatibility; `cartesian_actions`, `resolved_joint_targets`, and webcam aperture telemetry preserve the
proportional command.

After 250 ms without a valid frame, that arm freezes, pending displacement is discarded, and the last aperture is
held. The arm must produce a tracked frozen frame before movement can resume. Manual input disables webcam authority
globally until the operator explicitly enables it again.

## Pinned local assets

Install the exact MediaPipe Tasks Vision runtime and Hand Landmarker model:

```bash
python3 scripts/install_hand_control_assets.py
```

The installer verifies the pinned package and model SHA-256 hashes before atomically placing them under
`~/.local/share/dr-anmar/assets/hand-control/mediapipe-tasks-vision-0.10.35`. The workstation serves only an explicit
allow-list of required files; operation never depends on a live CDN.

For remote use, terminate HTTPS on the tailnet and keep ports 2360/2361 private. For example, on Gilgamesh:

```bash
tailscale serve --bg http://127.0.0.1:2360
```

The Doctor Studio iframe grants `camera; microphone`, and both hub and workstation return a same-origin camera
Permissions Policy.

## Verification

Run the deterministic contract suites:

```bash
python3 -m unittest tests/test_hand_teleop.py -v
node --test tests/hand_control.test.mjs
```

Real-webcam acceptance must still be performed from the secured Mac browser: calibrate both hands, verify independent
XYZ and wrist rotation, check 0/50/100% jaw positions, coordinate both arms, freeze/recenter, leave and re-enter the
frame, then take over with keyboard control.
