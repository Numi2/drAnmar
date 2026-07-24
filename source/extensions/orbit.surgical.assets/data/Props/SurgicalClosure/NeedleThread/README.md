# DrAnmar Needle System 0.3.0

Research-only category-level surgical closure assets for OpenUSD, NVIDIA Isaac Sim and Isaac Lab.
They are not clinically validated, not a manufacturer digital twin and not approved for patient care.

## Assets

- `Props/SurgicalClosure/Needle/dranmar_needle.usda`
  - watertight half-circle round-body taper-point needle;
  - explicit swage recess;
  - compound capsule collision;
  - explicit mesh-derived mass, center of mass and principal inertia;
  - robot grasp, tip, swage and count frames.

- `Props/SurgicalClosure/NeedleThread/dranmar_needle_thread.usda`
  - rigid needle plus 180 mm violet braided 4-0 thread;
  - 120 independently moving rigid thread segments;
  - D6 axial, bend and twist constraints;
  - explicit swage joint;
  - coiled initial configuration.

- `Props/SurgicalClosure/NeedleThread/dranmar_needle_thread_extended.usda`
  - the same mechanism in an extended initial configuration.

- `Props/SurgicalClosure/NeedleThread/dranmar_needle_thread_rigid_proxy.usda`
  - single rigid body for perception, handover, positioning and dataset generation.

## Coordinate convention

- SI units; Z-up.
- The needle lies primarily in the XY plane.
- The tip is at the lower end of the half-circle.
- The swage is at the upper end.
- Frame quaternions are WXYZ and authored using flat USDA syntax.

## Integration

The repository overlay installs the catalog assets under the existing
`orbit.surgical.assets` data tree and provides `needle_thread.py` with:

- `make_needle_cfg`
- `make_needle_thread_rigid_proxy_cfg`
- `spawn_segmented_needle_thread`
- `frame_path`

The segmented assembly is raw USD maximal-coordinate physics. It is not exposed as
an Isaac Lab reduced-coordinate articulation because its thread uses general D6 joints.

## Provisional values

Dimensions and physics parameters are category-level engineering seeds. The needle mass
and inertia are derived from the actual generated solid mesh. Thread segment mass is based
on a 0.25 mm diameter, 180 mm long, 1300 kg/m³ braided-polymer proxy with a solver floor
of 1e-7 kg per segment. Joint stiffness, damping, friction, swage pullout and break values
remain provisional and should be calibrated by the user.
