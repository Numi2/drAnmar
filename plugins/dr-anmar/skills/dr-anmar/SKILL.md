---
name: dr-anmar
description: Use when the user wants Codex to operate, inspect, train, evaluate, teleoperate, configure, or extend surgical robots through the local Dr.Anmar simulation and research studio.
---

# Dr.Anmar

Treat Codex as the research collaborator and Dr.Anmar as the user-owned,
simulation-only surgical robotics studio. Work from natural-language intent;
do not force the user into a fixed surgical-task schema or invent a second
workflow engine over the repository's live capabilities.

## Start from live truth

1. Locate the active DrAnmar repository and inspect its branch, revision,
   worktree status, submodule state, and configured `DR_ANMAR_ROOT` before
   changing or launching anything.
2. Read `docs/OWNERSHIP.md`, then load only the owner document needed for the
   request. Use `docs/ARCHITECTURE.md` for runtime work and
   `docs/EVIDENCE_LEVELS.md` for claims or qualification.
3. Run `./dr_anmar.sh doctor` when repository/runtime readiness matters. Run
   `./dr_anmar_suite.sh status` before starting, stopping, or replacing a live
   room.
4. Inspect the owning launcher's usage and its live implementation before
   acting. The main surfaces are `dr_anmar.sh`, `dr_anmar_suite.sh`,
   `dr_anmar_learning.sh`, `dr_anmar_webcam.sh`, and the corresponding code
   under `scripts/`, `source/extensions/`, and `source/standalone/`.
5. Recheck remote endpoints, installed Isaac stack, GPU state, processes, and
   release paths at use time. Retained host facts are orientation, not current
   readiness evidence.

## Runtime ownership

- Dr.Anmar owns Doctor Studio, room and procedure contracts, controls, operator
  leases, expert guidance, recording, evaluation, lifecycle, and evidence.
- The hub owns identity, one-operator mutation access, lifecycle coordination,
  provenance, and the browser surface. The active worker owns one Isaac Lab
  environment, bounded controls, sensors, recordings, and simulator state.
- Isaac Sim, Isaac Lab, and NVIDIA PhysX own articulation, rigid/deformable
  mechanics, contacts, sensors, and solver execution. OpenUSD owns composed
  scene and asset representation. Provider integrations remain provider-owned
  behind Dr.Anmar contracts.
- Learned policies and expert controllers express bounded motion or intent.
  They must not directly write success, injury, bleeding control, repair, or
  patient stability. Environment-owned post-physics effects determine those
  outcomes from contact, force, geometry, flow, dwell, and attachment evidence.
- Only one GPU worker is active at a time. Interactive rooms, training jobs,
  provider workflows, and mutation-heavy evaluations must respect the live
  lifecycle and resource-exclusion rules rather than accumulating Isaac
  processes.
- Repository source is immutable application state. Mutable assets,
  demonstrations, logs, processes, checkpoints, and training artifacts belong
  under `DR_ANMAR_ROOT` unless the live configuration explicitly says otherwise.

Codex on the user's Apple machine may act as the control and review plane while
the authoritative Isaac/PhysX simulation runs on a configured CUDA host. Use a
private loopback tunnel for webcam/browser control, keep services loopback-only
by default, and never treat tunnel reachability as proof that the worker or
physics stream is healthy.

## Operating patterns

- Doctor Studio lifecycle: inspect `./dr_anmar_suite.sh` and use its
  `status`, `start`, `stop`, `restart`, or `logs` action as appropriate.
- Repository and simulator readiness: inspect `./dr_anmar.sh` and begin with
  `doctor`, `catalog`, or a bounded `smoke` when the requested evidence needs it.
- Learning: inspect `./dr_anmar_learning.sh` first. Select the smallest relevant
  validate, probe, train, replay, or held-out evaluation path, preserve source
  and checkpoint provenance, and compare fresh physical outcomes against the
  frozen incumbent before promotion.
- Private webcam control: inspect `./dr_anmar_webcam.sh`; verify both the tunnel
  and hub/worker health. Starting a tunnel or opening a browser does not arm a
  physical robot.
- New capability: change the lowest owning layer—room/product contract,
  environment/task, physics-owned effect, learning workflow, asset composition,
  or evidence gate. Reuse existing launch and lifecycle surfaces instead of
  expanding a static command catalog.

## Evidence and safety contract

Keep these levels separate: product capability, repository verification,
native-simulator evidence, real-world evidence, and clinical evidence. Never
promote one level into another by wording alone.

For every consequential run, report the exact parent and asset revisions,
dirty state, runtime and provider versions, GPU/driver, room/task, seeds,
policy/checkpoint hashes, arguments, artifact paths, simulator failures,
physical/contact outcomes, safety terminals, and comparison gate actually
produced. Reward, offline loss, rendered appearance, phase completion, or test
success alone is not surgical competence.

Dr.Anmar is simulation and research software. Do not claim clinical validity,
medical-device readiness, patient-specific fidelity, or physical-robot control
without the separate evidence and integration that establish that exact claim.
Do not replace a champion policy without its fresh, isolated, physics-owned
promotion gate.
