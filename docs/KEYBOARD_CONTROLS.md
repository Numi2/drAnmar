# Dr.Anmar keyboard surgical control

Dr.Anmar's operating room is designed so a doctor can control the complete simulated workstation without
moving a hand to the mouse. Movement keys are **hold-to-move**: releasing the key stops that command.
`Esc` always stops motion and returns supervision to manual control.

This interface is for simulation, education, and preclinical robotics research. The smart actions below are
bounded input conveniences, not autonomous clinical actions.

## Learn these first

| Key | Action |
| --- | --- |
| `W` / `S` | Move toward / away from the patient |
| `A` / `D` | Move left / right |
| `R` / `F` | Move up / down |
| `Q` / `E` | Roll left / right |
| Arrow keys | Pitch and yaw |
| `Space` | Toggle the gripper |
| `Enter` | Context action: approach, grasp, then lift |
| `L` | Start or rerun the live simulation expert |
| `I` | Pause or resume the active expert for inspection |
| `Esc` | Stop immediately and take manual control |
| `?` | Open or close the complete keyboard map |

Hold `Option` for a temporary precision clutch. Hold `Shift` for a temporary fast clutch. The selected base
speed remains unchanged when the clutch is released.

## One-key surgical combinations

Each combination supports both interaction styles: tap for a short bounded precision nudge, or hold for
continuous motion that stops on release. Translational and rotational vectors are normalized so a combined
command does not gain accidental diagonal speed.

| Key | Combined movement | Intended use |
| --- | --- | --- |
| `Z` | Left translation + left yaw | Orbit around a target while keeping it framed |
| `X` | Right translation + right yaw | Orbit around a target in the opposite direction |
| `V` | Surface-guided advance, then clockwise roll | Approach the nearest sampled tissue point, lock the entry vector at puncture, and rotate after measured entry |
| `B` | Reverse the locked entry vector + counter-roll | Withdraw through the same entry axis and continue to a visible clear gap |
| `N` | Lift + retract | Clear a grasped object away from the working surface |
| `K` | Lower + approach | Return toward the working surface on a diagonal |

## Repeated-Enter workflow

`Enter` changes its bounded action from current simulator state:

1. **Approach:** when the target is not aligned, a short precision pulse follows the two largest target-offset axes.
2. **Grasp:** inside the capture radius, the jaws close on the target.
3. **Lift:** after a simulator-confirmed grasp, a short lift-and-retract pulse clears the object.
4. **Recover:** if the jaws are closed without a confirmed grasp, the jaws reopen so the doctor can retry.

The button beside the live view states what the next `Enter` press will do before the user presses it.

Far from the target or tissue, semantic keyboard actions receive a simulator-rate-aware travel boost. That
boost switches off before fine alignment: target approach returns to normal scale inside 50 mm, and needle
driving returns to precision scale inside 20 mm. The tissue-entry direction is fixed at puncture so changes in
the sampled curved surface cannot make the tip wander during insertion or withdrawal.

## Complete map

| Area | Keys |
| --- | --- |
| Instrument | `1`, `2` |
| Camera sensors | `3` stereo left, `4` stereo right, `5` wrist 1, `6` wrist 2, `C` cycle |
| Camera views | `7` operative, `8` close, `9` overview, `Shift+C` cycle |
| Base speed | `,` precision, `.` normal, `/` fast |
| Gripper | `O` open, `P` close, `Space` toggle |
| Supervision | `M` manual, `G` guided, `L` start expert, `I` pause/resume expert, `Esc` stop and take control |
| Expert path | `H` show/hide |
| Recording | `Y` start, `U` stop and save, `J` replay, `Delete` reset scene |
| Annotations | `Shift+1` approach, `Shift+2` grasp, `Shift+3` manipulate, `Shift+4` recovery |
| Events | `Shift+5` task event, `Shift+6` safety event |

Keyboard commands are ignored while the user is typing into a text field. Losing browser focus or hiding the
page clears held commands and sends a stop command. The live interface audits its own buttons and displays
`all controls mapped` only when every visible button declares a keyboard equivalent.

Starting the expert with `L` opens a synchronized demonstration automatically. `I` pauses only the expert
phase clock so the current camera, state and telemetry can be inspected. `Esc` preserves the current simulator
state, records a takeover intervention, stops expert authority and returns movement to the doctor. A completed
run is not automatically an approved reference; see
[`EXECUTABLE_EXPERT_GUIDANCE.md`](EXECUTABLE_EXPERT_GUIDANCE.md).
