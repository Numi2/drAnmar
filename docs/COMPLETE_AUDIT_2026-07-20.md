# Dr.Anmar complete engineering audit — 2026-07-20

This review covers runtime ownership, OpenUSD startup, procedure composition, controls, recording,
research evidence, browser safety, curriculum integrity, deployment, performance and public-release
hygiene. The software remains simulation-only and is not clinically validated.

## Outcome

All source-level defects found during this pass were corrected. Static release gates now cover the
entire keyboard surface plus cross-file curriculum/task/procedure integrity. Live Isaac/OpenUSD,
biomechanical and clinician-facing claims remain explicit validation gates rather than inferred facts.

## Critical defects corrected

1. **Training competed with the live Isaac worker.** The training route described the workstation as
   paused but did not stop it. Training now stops the interactive worker before allocating the GPU.
2. **The wrong room returned after a GPU job.** Only the Isaac task name was remembered. Full procedure,
   anatomy scene, anatomy title, organ asset and OpenUSD environment context are now captured and restored.
3. **GPU activities could collide.** Training, NVIDIA healthcare workflows, Failure Lab matrices, room
   switches, replay, reset, scenario changes, autonomy changes and handoff mutations now use exclusive
   lifecycle gates.
4. **Hub restarts could orphan GPU jobs.** Managed jobs are terminated during a clean shutdown. On startup,
   stale training, healthcare and matrix manifests are reconciled and marked interrupted; a matching
   orphan process group is stopped before the workstation can resume.
5. **Every restart rebuilt all OpenUSD assets.** Suite startup now performs a lightweight seven-scene
   manifest/file preflight. Geometry sanitization and composition run only when required or when
   `DR_ANMAR_REBUILD_OPENUSD=1` is explicitly set.
6. **Stale PID files could identify unrelated processes.** Suite and workstation PID checks now verify the
   live command identity before treating a process as Dr.Anmar.

## High-impact corrections

- The browser now learns the configured worker port from hub status instead of assuming port 2361.
- Task-only launches no longer guess the first procedure when multiple rooms share one Isaac task.
- The workstation encodes the primary endoscope plus only camera streams with active viewers. Recording
  still captures the configured raw multimodal observations, avoiding needless JPEG CPU load.
- Demonstrations are written through temporary files and atomically promoted. Each manifest records a
  SHA-256 of the final data file, nominal and observed state rates, and a bounded automatic-save limit.
- A save failure no longer crashes the active simulator; it becomes an explicit workstation coaching/error
  state. Active recording is also saved defensively during shutdown.
- Hub and worker reject cross-site browser mutations and emit no-store, no-sniff, no-referrer and same-site
  resource headers. This reduces browser-origin attacks but does not replace authentication or TLS.
- User-authored study and dataset titles, plus displayed server errors, are HTML-escaped before insertion.
- Public-release scanning uses bounded parallel reads of tracked and candidate public files rather than
  recursively hydrating ignored/cloud-evicted runtime assets, so a release audit cannot freeze on local data.

## Curriculum and procedure integrity

- All 19 procedure rooms now have at least one doctor-facing curriculum lesson.
- Needle pickup, handover and liver retraction now open their exact intended room rather than relying on an
  ambiguous shared-task fallback.
- Needle passing/regrasping and a complete anatomy-handling course were added for incision, gallbladder
  repositioning, bladder relocation and patient-shape variation.
- `audit_project_consistency.py` fails CI for duplicate identifiers, unknown tasks, orphan rooms, mismatched
  lesson/room tasks, missing steps or malformed waypoints.
- `audit_keyboard_controls.py` remains a CI gate for every visible workstation control.

## Verification completed in this pass

- Python source compilation for the changed modules.
- Shell syntax for suite, workstation, training and primary launchers.
- Public snapshot credential/private-path/runtime-data scan.
- Keyboard coverage: 51 of 51 visible controls mapped.
- Cross-file integrity: 8 courses, 29 lessons and 19 procedure rooms, with every reference valid.
- JavaScript syntax extraction/check and Git whitespace checks are release gates for the final source state.

## Remaining validation gates

These are not unimplemented source defects; they require the Linux/RTX runtime, specialist hardware or
domain evidence:

- Run every OpenUSD room and procedure state machine on Gilgamesh after the host is reachable, including
  collision, grasp, cutting, suturing, shunt, ultrasound, anatomy variation and camera checks.
- Measure isolated 4090 render/control rate, CPU use, GPU memory and long-recording save latency.
- Validate thread tension, knot security, tissue deformation, incision topology, ultrasound acoustics and
  organ material parameters against accepted bench or biomechanical references.
- Conduct clinician usability and construct-validity studies; telemetry coaching remains an engineering
  research proxy and must not be represented as a clinical score.
- Configure the new optional application access token and HTTPS before use outside a private trusted network.
  The single-operator lease now prevents two browser sessions from mutating the same live workstation.
- Validate optional RTI DDS, Clarius, haptic, XR, eye tracking and physical hardware only after their licensed
  runtimes and explicit safety boundaries are available.

The detailed evidence backlog remains in `docs/VALIDATION_BACKLOG.md`. Runtime availability is now derived
from the single native solver-capability contract in `physics_next/manifest.json` rather than a parallel list
of room-specific gates.

## Post-audit evidence — 2026-07-21

- The visible keyboard surface now audits 54/54 mappings, including `L` expert start and `I` pause/resume.
- Interrupted suturing, needle handover and ultrasound-guided access each completed the eight-phase controller
  and saved synchronized trajectories, but bounded convergence warnings correctly prevented reference
  qualification. The original audit's clinician-review gate therefore remains open.
- Three live UI GIFs and the exact completion-versus-qualification boundary are documented in
  [`EXECUTABLE_EXPERT_GUIDANCE.md`](EXECUTABLE_EXPERT_GUIDANCE.md).
