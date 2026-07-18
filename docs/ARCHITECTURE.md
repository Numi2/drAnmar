# Dr.Anmar architecture

## Purpose and safety boundary

Dr.Anmar is a simulation-only interface around ORBIT-Surgical. The browser submits bounded actions to a
single Isaac Lab worker, receives simulated endoscopic frames, and stores demonstrations locally. No
component contains a physical-robot driver or a clinical workflow integration.

## Runtime components

1. `dr_anmar_suite.sh` manages the hub and the one active GPU worker.
2. `scripts/dr_anmar_hub.py` serves Doctor Studio, curriculum data, anatomy inventory, and bounded
   training requests.
3. `scripts/dr_anmar_workstation.py` owns the active Isaac Lab environment and exposes simulated tool
   movement, camera frames, reset, and demonstration recording.
4. `scripts/dr_anmar_anatomy_viewer.py` provides a low-overhead static preview for an installed official
   anatomy scene without keeping an Isaac worker active.
5. `web/doctor_studio.html` presents the doctor-facing learning and operating-room interface.

Only one GPU worker is active at a time. Switching a lesson or anatomy room replaces that worker rather
than accumulating Isaac Sim processes.

## Storage model

The repository is immutable application source. Mutable data lives under `DR_ANMAR_ROOT`, which defaults
to `~/.local/share/dr-anmar`:

```text
assets/sufia_bc/       extracted optional anatomy scenes
downloads/sufia_bc/    resumable release archives
demos/                 recorded demonstrations and manifests
logs/                  hub, worker, smoke, and training logs
run/                   process IDs and asset-install status
state/                 local curriculum progress
training/              bounded training jobs and metadata
isaac_portable/        isolated Isaac caches and settings
```

These paths are ignored if a developer deliberately points `DR_ANMAR_ROOT` inside the repository.

## Network model

The hub defaults to port 2360 and the worker to port 2361. Both bind to `0.0.0.0` so a trusted device can
operate the interface. The current research build has no authentication or TLS; deployment must provide
a private network boundary or an authenticated reverse proxy.

## Compatibility baseline

This derivative ports the upstream Isaac Sim 4.1-era environment code to the installed Isaac Sim 5.1 /
Isaac Lab 2.3.2 APIs. The upstream environment namespace and asset layout remain intact to preserve
workflow and checkpoint compatibility where the underlying API permits it.
