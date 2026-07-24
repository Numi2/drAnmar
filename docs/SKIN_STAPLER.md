# DrAnmar skin stapler

DrAnmar ships the user-authored skin stapler as a repository-local,
Apache-2.0 asset at the I4H-compatible catalog path:

```text
Props/SurgicalClosure/SkinStapler
```

The complete v0.2.0 payload is preserved: 36 source geometry files, 14 GLB
inspection exports, 11 textures, three OpenUSD runtime layers, interaction
frames and a physics profile.

## Runtime representations

| Representation | Intended use | Current status |
| --- | --- | --- |
| `skin_stapler_rigid_proxy.usda` | Stable rigid perception, grasping, handover and positioning work in Isaac Sim 5.1 / Isaac Lab 2.3.2 | Available as an optional loaded prop in the main operating room; native CUDA qualification pending |
| `skin_stapler_articulated.usda` | Trigger and pusher control in Isaac Sim 6.0.1 / Isaac Lab 3.0 | Imported with helpers; native CUDA qualification pending |
| `skin_staple.usda` | Explicit simulated staple spawning and closure-task bookkeeping | Imported with helpers; tissue interaction is not modeled |

The source layers declared `UsdPrimvarReader_float2.inputs:varname` as a
`token`. DrAnmar corrected all 11 declarations to the shader's required
`string` type without changing geometry or physics. The integrated layers pass
`usdcat` parsing and current `usdchecker` validation for their default, loaded
and empty states. Exact source and integrated hashes are recorded in
`integration_report.json` beside the asset.

## Operating-room use

Open the main operating-room setup and enable **Dr.Anmar skin stapler**. The
room spawns the loaded rigid representation on its broad side at a separate
table landing with its authored mass, inertia, compound collision, physics
materials, semantic labels and contact sensors.

The prop is optional so the established needle, pad and scissors bench remains
unchanged by default. The UI-selected asset set is preserved across worker
restarts in the same way as the other configurable bench props.

## Python integration

```python
from orbit.surgical.assets.skin_stapler import (
    SurgicalClosureAssets,
    make_articulated_skin_stapler_cfg,
    make_rigid_skin_stapler_cfg,
)
```

`SurgicalClosureAssets` exposes relative paths compatible with the NVIDIA I4H
asset-helper convention. The helper package also includes loaded/empty state
selection, semantic labels, synchronized trigger/pusher targets, deterministic
simulated deployment bookkeeping, standalone staple composition and
closure-line scoring utilities.

## Validation boundary

The imported asset is category-level research content. Physical parameters,
including mass allocation, inertia, friction, restitution, mechanism travel
and drive gains, remain provisional until measured and calibrated.

The model does not establish tissue penetration, staple formation, closure
strength, wound healing, sterility or clinical quality. It is not clinically
validated, is not a medical device and must not be used for patient care.
