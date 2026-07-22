# Dr.Anmar surgical control

Dr.Anmar lets a doctor control one or two simulated surgical robots with the keyboard, an Xbox-style controller,
or explicit voice commands. These inputs all reach the same Isaac Lab action boundary; they do not replace native
contacts, constraints, tissue mechanics, or simulation stepping.

This interface is for simulation, education, and preclinical robotics research. Movement is hold-to-move unless a
command is explicitly described as a bounded pulse. `Esc` or `Backspace` always stops both robots and returns control
to the doctor.

## Two-hand keyboard control

Each hand permanently owns one robot, so both instruments can move at the same time.

| Left hand · Instrument 1 | Action | Right hand · Instrument 2 | Action |
| --- | --- | --- | --- |
| `W` / `S` | Up / down | `I` / `K` | Up / down |
| `A` / `D` | Left / right | `J` / `L` | Left / right |
| `Q` / `E` | Toward / away from patient | `U` / `O` | Toward / away from patient |
| Left `Shift` + movement | Pitch, yaw, or roll wrist | Right `Shift` + movement | Pitch, yaw, or roll wrist |
| `Space` | Toggle left gripper | `Enter` | Toggle right gripper |
| `1` / `2` / `3` | Fine / normal / fast | `8` / `9` / `0` | Fine / normal / fast |

Releasing a movement key removes that axis immediately. Losing browser focus, hiding the page, or disconnecting an
active controller sends a stop. A stopped gamepad cannot resume until its sticks and triggers return to neutral.

## One-key surgical combinations

These are control conveniences only. They send normalized multi-axis commands and never create a grasp, puncture,
attachment, contact, or successful task state outside the simulator.

| Key | Combined movement | Intended use |
| --- | --- | --- |
| `Z` / `X` | Translate and yaw left / right | Orbit a target while keeping it framed |
| `V` / `B` | Advance + roll / reverse + counter-roll | Drive or withdraw a curved needle |
| `N` | Lift + retract | Clear a grasped object from the work surface |
| `F` | Lower + approach | Return toward the work surface |
| `F12` | Context-aware bounded action | Align, grasp, lift, or recover from current simulator state |

The smart action states its next operation before activation. It uses short simulator-rate-aware pulses and returns to
normal precision close to the target. Isaac Lab remains authoritative for whether contact or grasp actually occurs.

## Xbox-style controller

Connect one standard-mapping controller to the computer running the browser, then press any controller button while
the operating room has focus. The browser streams the controller commands to Isaac Lab on Gilgamesh; the controller
does not need to be paired with the remote server itself.

The default layout is permanently bimanual. The left side of the controller owns Instrument 1 and the right side owns
Instrument 2, so neither robot needs to be selected and both can move simultaneously.

| Control | Action |
| --- | --- |
| Left / right stick | Move the left / right instrument in the camera plane |
| Hold `X` + both sticks | Each stick controls its instrument's depth and wrist roll |
| Hold `Y` + both sticks | Each stick controls its instrument's wrist pitch and yaw |
| `LB` / `LT` | Close / open the left gripper |
| `RB` / `RT` | Close / open the right gripper |
| Hold `L3` / `R3` | Temporary precision speed for the left / right instrument |
| `A` | Smart context action for the instrument moved most recently |
| `B` | Emergency stop both robots |
| D-pad up / down | Increase / decrease both instruments' base speed |
| D-pad left / right | Next camera sensor / next camera angle |
| Hold `View` | Camera layer: left stick pans, right stick orbits, triggers zoom |
| Camera layer `LB` / `RB` | Next sensor / next angle |
| Camera layer `Y` or `R3` | Reset the adjustable camera |

Hold `Menu` for the session layer. `Menu+A` starts or stops recording, `Menu+X` starts or pauses the live expert,
`Menu+Y` toggles manual/guided control, `Menu+LB/RB` changes sensor/angle, `Menu+L3` toggles the reference path,
`Menu+R3` takes control, and `Menu+D-pad up/down` replays or resets the scene. `B` remains an emergency stop in every
layer.

The UI samples the standard browser Gamepad API on animation frames, uses a radial deadzone and progressive response
curve for precise center motion and full edge speed, and sends coalesced simulator commands at about 30 Hz. The two
translation/rotation vectors are normalized independently, so diagonal movement cannot exceed the configured speed.
Visible stick/mode feedback is always present. Supported controllers also provide short haptic cues for connection,
mode changes, gripper commands, native grasp contact, and emergency stop; haptics never replace visual feedback.

## Voice and typed commands

Voice is push-to-talk, never always listening. Hold the microphone button or the backtick key, speak one command,
then release. Browsers that do not expose speech recognition can run the same command through the adjacent text box.
Microphone use requires the browser's permission.

Examples:

| Say or type | Result |
| --- | --- |
| `left robot up` | Short upward pulse on Instrument 1 |
| `right robot toward` | Short approach pulse on Instrument 2 |
| `left robot up and toward` | Normalized two-axis pulse on Instrument 1 |
| `close left gripper` | Explicitly close Instrument 1 jaws |
| `open right gripper` | Explicitly open Instrument 2 jaws |
| `left robot precision speed` | Set Instrument 1 to fine speed |
| `camera overhead` | Select overhead view |
| `camera wrist one` | Select Instrument 1 wrist camera |
| `smart assist` | Run the context action for the selected instrument |
| `stop` | Stop both robots and return to manual control |

Unrecognized speech causes no robot action. Spoken movement is always a bounded pulse followed by an explicit stop;
continuous teleoperation belongs on the keyboard or controller.

## Cameras and session controls

| Area | Keys |
| --- | --- |
| Camera sensors | `4` stereo left, `5` stereo right, `6` wrist 1, `7` wrist 2, `C` cycle |
| Camera views | `F1` operative, `F2` close, `F3` wide, `F4` overhead, `F5` left angle, `F6` right angle, `F7` opposite |
| Adjustable camera | `F8` toggle, `Home` reset, drag orbit, `Shift`-drag pan, wheel zoom |
| Robot selection for pointer/voice | `[` Instrument 1, `]` Instrument 2 |
| Supervision | `M` manual, `G` guided, `F9` start expert, `F10` pause/resume, `Esc` stop and take control |
| Expert path | `H` show/hide |
| Recording | `Y` start, `T` stop and save, `R` replay, `Delete` reset scene |
| Annotations | `Option+1` approach, `Option+2` grasp, `Option+3` manipulation, `Option+4` recovery |
| Events | `Option+5` task complete, `Option+6` safety review |
| Help | `?` complete in-app map |

Keyboard robot commands are ignored while the doctor is typing. The live interface audits every visible button and
reports full coverage only when each has a keyboard equivalent. A completed expert run is not automatically an
approved research reference; see [Executable expert guidance](EXECUTABLE_EXPERT_GUIDANCE.md).
