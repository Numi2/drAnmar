# Dr.Anmar Needle System 0.3

Dr.Anmar ships the new surgical-closure package under the repository-local
I4H-compatible catalog:

```text
Props/SurgicalClosure/Needle
Props/SurgicalClosure/NeedleThread
```

## Selectable Bench Setup assets

| Selection | Runtime representation | Intended work |
| --- | --- | --- |
| Dr.Anmar needle v0.3 | `dranmar_needle.usda` | Needle pickup, pose, handover and swage-frame research |
| Dr.Anmar needle + thread · coiled | `dranmar_needle_thread.usda` | Dynamic coiled thread, regrasp and attached-strand experiments |
| Dr.Anmar needle + thread · extended | `dranmar_needle_thread_extended.usda` | Pull-through, tension and extended-layout experiments |
| Dr.Anmar needle + thread · rigid proxy | `dranmar_needle_thread_rigid_proxy.usda` | Stable perception, handover, positioning and dataset generation |

The standalone needle and rigid proxy are tensor-managed rigid assets. The
coiled and extended assemblies are raw maximal-coordinate OpenUSD physics:
one rigid needle, 120 independently moving thread segments, D6 constraints and
an explicit swage joint. They are intentionally not represented as one
reduced-coordinate articulation.

All four options are disabled by default and can be selected independently in
the main Bench Setup dialog.

## Coordinate and physical contract

- SI units, Z-up.
- Half-circle round-body taper-point needle.
- Explicit swage recess.
- 180 mm violet braided 4-0 thread.
- Authored tip, grasp, swage, thread-grasp, tail, handover and closure frames.
- Apache-2.0.

The geometry, needle inertia and collision are authored in the imported
package. Thread stiffness, damping, friction, break forces, swage pullout and
other physical parameters remain provisional until measured and calibrated.

## Validation boundary

These are category-level research assets. They are not a manufacturer digital
twin, are not clinically validated and must not be used for patient care. A
successful simulator load establishes asset composition and mechanism
execution only; it does not establish penetration, tissue response, knot
security, closure strength, sterility or clinical quality.
