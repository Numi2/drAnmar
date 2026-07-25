# DrAnmar Atraumatic Surgical Exposure Robot v0.1.0

DrAnmar-owned OpenUSD research assets for bilateral soft-tissue capture,
force-limited retraction, maintained surgical exposure, and ROI visibility
benchmarking, integrated with NVIDIA Isaac Lab and Isaac Sim.

## Catalog path

```text
Props/SurgicalExposure/AtraumaticExposureRobot/
```

## Primary assets

- `dranmar_atraumatic_exposure_tool_payload.usda`: hand-replacement payload for `panda_link8`.
- `dranmar_atraumatic_exposure_tool_standalone.usda`: standalone articulation.
- `dranmar_atraumatic_exposure_tool_rigid_proxy.usda`: perception/planning proxy.
- `dranmar_fenestrated_retraction_pad.usda`: replaceable geometric-trapping pad.
- `dranmar_microcup_retraction_pad.usda`: replaceable distributed low-vacuum proxy pad.
- `dranmar_exposure_tissue_demo.usda`: two deformable flaps over an ROI target.

## Mechanism

Each side has an independent lateral carriage, vertical lift, pad-pitch axis,
and 6 mm compliant force-sensing axis. Each pad exposes six independent
capture cells. Tissue capture is created at runtime from overlap-prioritized,
explicitly verified deformable vertex attachments. Overload logic can release individual cells before
releasing a complete pad.

## Validation

Static validation, controller tests, and the four-case headless CUDA matrix
are documented in `docs/atraumatic_exposure_robot/VALIDATION.md`. The runtime validator qualifies each
pad geometry in standalone and complete Franka-mounted representations.

## Research boundary

All dimensions, friction values, tissue mechanics, capture strengths, force
thresholds, vacuum behavior, and controller gains are provisional engineering
seeds. The package does not claim calibrated tissue trauma, safe surgical
force limits, clinical effectiveness, sterility, regulatory approval, or
suitability for patient care.
