# Executable expert guidance

Dr.Anmar provides one live simulation expert controller across supported local procedure rooms. It is an executable
teaching aid: the controller sends actions to the active Isaac environment while the OpenUSD room, operative
camera, grippers, procedure mechanics, phase annotations and safety telemetry continue updating. It is not a
prerecorded video and it is not a validated clinician policy.

## The eight-phase teaching loop

| Phase | What the doctor should inspect |
| --- | --- |
| **Rest** | Confirm the neutral pose, anatomy and operative camera. |
| **Approach** | Watch the open instrument move above the first target. |
| **Align** | Inspect depth, angle and working-axis alignment before contact. |
| **Contact** | Enter the interaction zone slowly while preserving the research safety envelope. |
| **Grasp** | Close deliberately and verify that the intended object or tissue follows. |
| **Manipulate** | Execute the room-specific path, handoff or tissue interaction. |
| **Verify** | Hold still while task, force, tissue and camera evidence are inspected. |
| **Recover** | Withdraw to a stable pose without undoing the completed work. |

The phase rail is always visible beside the operative view. The active phase is also shown in the outer
operating-room header, making the sequence readable in a demonstration or screen recording.

## Controls

| Action | Button | Keyboard |
| --- | --- | --- |
| Start or run again | **Watch expert** | `L` |
| Pause for inspection / resume | **Pause expert** / **Resume expert** | `I` |
| Take manual control at the current state | **Take control** | `Esc` |

Pause freezes the expert phase clock while the simulator remains available for inspection. Takeover preserves
the current simulator state and recording, records the intervention, and transfers authority to the doctor.

## Completion is not qualification

`8/8 phases` means the controller traversed the complete teaching state machine and saved the synchronized run.
It does **not** mean that the clinical task succeeded or that the trajectory is suitable for training. Dr.Anmar
keeps these decisions separate:

- `status=completed` records complete phase traversal;
- `clean_reference_eligible=true` requires a warning-free controller run;
- `behavior_cloning_reference_candidate=true` is set only for a clean uninterrupted simulation-expert run;
- `reference_review_status=pending_clinician_review` still requires human review before reference promotion;
- bounded target or waypoint timeouts keep the saved run available for debugging but prevent automatic
  Behavior Cloning reference qualification.

The user interface therefore distinguishes a completed teaching run from an approved expert reference.

## Live examples

### Dual-instrument needle handover

<p align="center">
  <img src="screenshots/expert-guidance-needle-handover.gif" width="960" alt="Live dual-instrument needle-handover simulation expert with phase and control feedback">
</p>

The capture shows both instruments entering the shared workspace and the live phase rail advancing through
manipulation and recovery. The run saved 687 robot-state frames plus 64 camera frames. Large approach, align and
contact residuals correctly prevented it from becoming a Behavior Cloning reference.

## Research data contract

Each run synchronizes the available members of this contract:

- robot actions, joint state, joint torque and tool/body poses;
- operative camera observations and timing;
- active procedure phase and event sequence;
- gripper, task object, native outcome and intervention state;
- needle, thread, tissue, cutting, vascular, ultrasound or recovery mechanics when active;
- controller status, completed phases, warnings and reference-qualification state.

The GIFs are documentation captures only. Training and evaluation consume the checksummed NPZ/JSON
demonstration pair, never pixels extracted from the GIF.

## Remaining work

- Replace generic convergence targets with room-specific expert acquisition and manipulation plans.
- Achieve clean reference qualification for each supported local room before clinician review.
- Exercise the active room catalog end to end, then repeat across the anatomy presets and challenge scenarios.
- Validate phase semantics, pause/takeover usability, task success, forces and teaching value with clinicians.
- Keep all generated references simulation-only until independent review and the relevant validation gates pass.
