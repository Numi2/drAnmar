# Bimanual webcam teleoperation

Dr.Anmar's webcam controller is a simulation-only master-pose input for the two native NVIDIA PSM instruments.

## Operator behavior

- Physical left hand controls Instrument 1; physical right hand controls Instrument 2.
- Hand translation maps to endoscope-forward depth plus the live operative
  view's right and up axes. The worker rotates that camera-space target into
  each PSM's native IK root frame.
- Palm-frame orientation maps through the same live camera frame to bounded
  tool rotation.
- Only thumb and index spacing controls proportional jaw aperture.
- Pointing the index finger down with the other fingers tucked is the natural
  motion clutch. Holding the point steadily for 180 ms engages; relaxing or
  lifting the point freezes immediately. The tucked fingers distinguish a
  deliberate point from an open hand and never affect jaw aperture.
- A new engagement captures the current hand pose and robot target, then ramps
  gain in smoothly, so freezing and recentering cannot jump.
- Precision is progressive: small movements receive microsurgical gain while
  larger deliberate movements expand into the full bounded 120 mm
  camera-relative workspace without a mode change.
- The index fingertip is the vertical intent point. Moving it toward the
  preview's bottom **TABLE REACH** line progressively lowers the instrument;
  touching the line requests the full safe downward endpoint so native PhysX
  contact can settle the tool onto the table.
- The second hand must first appear relaxed and then deliberately point down
  before it is admitted. Visibility alone never activates Instrument 2.
- The webcam launcher lives on the operative camera rather than inside the
  keyboard-control drawer. Its window floats independently above the room,
  can be dragged by its title bar, resized from the lower corner, and restores
  its last position and size. `Fit` returns it to a safe visible default.

Calibration records a robust median from 24 stable samples for neutral palm scale and normalized closed/open
thumb–index spacing per browser camera in `localStorage`. Median absolute deviation rejects a moving hand rather than
persisting a noisy calibration. The webcam stream and MediaPipe landmarks stay inside the browser.

The browser runs inference on decoded camera frames with `requestVideoFrameCallback` where supported. Physical
left/right identity combines MediaPipe handedness with palm-position continuity so one flickering handedness label
cannot exchange the two instruments. A palm geometry score, field-of-view margin, landmark confidence, velocity, and
scale-change plausibility form the transmitted signal-quality value. Frames below the quality gate do not move the
robot or update jaw aperture.

Each arm uses time-aware One Euro filtering: it suppresses stationary landmark jitter while automatically reducing
lag during deliberate motion. A bounded 25 ms velocity prediction compensates part of camera/inference latency,
small translation and rotation deadbands remove resting chatter, and the interface exposes vision rate, inference
time, round-trip command time, per-arm quality, point-down clutch state, table reach, and the active safety state.

Depth uses the ratio between image-space palm geometry and corresponding
camera-aligned world-landmark geometry. This compensates for palm
foreshortening before the calibrated relative-depth logarithm is evaluated, so
wrist rotation does not masquerade as pushing the hand toward the camera.

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
    translation_offset_m: [camera_forward, camera_right, camera_up],
    rotation_vector_rad: [camera_forward, camera_right, camera_up],
    aperture_normalized,
    confidence
  }]
}
```

The endpoint accepts one complete frame at a time, validates unique available arms, finite values, confidence and
aperture in `0..1`, translation inside ±0.12 m, rotation inside ±0.8 rad, the active operator lease, and strictly
increasing sequence numbers. A rejected frame does not alter robot state.

The workstation resolves the camera forward/right/up basis against each PSM
root orientation. Changing or aiming the operative camera freezes webcam
motion and increments the control-frame revision; engagement then captures a
fresh anchor in the new view.

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
globally until the operator explicitly enables it again. The server independently rejects tracked frames below 60%
quality, and simulator-rate commands use bounded acceleration for starts and direction changes plus exact residual
settling to avoid chatter or cumulative-pose overshoot.

## Pinned local assets

Install the exact MediaPipe Tasks Vision runtime and Hand Landmarker model:

```bash
python3 scripts/install_hand_control_assets.py
```

The installer verifies the pinned package and model SHA-256 hashes before atomically placing them under
`~/.local/share/dr-anmar/assets/hand-control/mediapipe-tasks-vision-0.10.35`. The workstation serves only an explicit
allow-list of required files; operation never depends on a live CDN.

### Mac-to-Gilgamesh camera access

The browser's secure-context rule cannot be disabled by Dr.Anmar. The included launcher satisfies it without
requiring HTTPS administration by forwarding the remote hub to the Mac's browser-trusted loopback origin:

```bash
./dr_anmar_webcam.sh start
```

This opens `http://127.0.0.1:12360/`. The page is local from the browser's perspective, while all workstation traffic
travels through the authenticated SSH connection over Tailscale. The remote hub and worker remain private. Useful
commands are:

```bash
./dr_anmar_webcam.sh status
./dr_anmar_webcam.sh stop
```

The SSH destination defaults to the current Gilgamesh deployment and can be changed without editing the script:

```bash
DR_ANMAR_SSH_TARGET=user@host ./dr_anmar_webcam.sh start
```

Tailnet administrators may instead publish the hub using Tailscale Serve:

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

The JavaScript suite covers bimanual identity stability, robust calibration, point-down clutch discrimination,
bottom-line table reach, long-range bounds, palm-frame degeneracy, adaptive filtering, bounded prediction, quality
rejection, depth, rotation, proportional aperture, and frame synchronization.
The Python suite covers validation without mutation, quality holds, acceleration conditioning, cumulative target
resampling, watchdog/reacquisition, native IK scales, manual takeover, proportional gripper endpoints, and recording
compatibility.

Real-webcam acceptance must still be performed from the secured Mac browser: calibrate both hands, verify deliberate
point-down engagement and immediate relaxed-pose freeze, move the fingertip to the **TABLE REACH** line and confirm
native table contact, verify independent XYZ and wrist rotation, check 0/50/100% jaw positions, coordinate both arms,
freeze/recenter, leave and re-enter the frame, then take over with keyboard control.
