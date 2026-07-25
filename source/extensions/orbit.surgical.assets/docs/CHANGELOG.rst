Changelog
---------

0.6.0 (2026-07-25)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the wound-preparation, atraumatic-exposure, adaptive-hemostasis,
  adaptive-anastomosis, adaptive seal-and-divide and SafePlane-dissection
  systems to the main Bench Setup as optional robot stations.
* Each station composes the real standalone articulation and its authored
  procedure substrate while validating the associated Franka payload and
  rigid planning proxy.
* Added an exclusive featured-robot station contract so the shared bench loads
  one large system at a time without changing the default NVIDIA handover
  composition.

0.5.0 (2026-07-24)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the DrAnmar Topical Skin Adhesive System 0.1.0 under
  ``Props/SurgicalClosure/SkinAdhesive``.
* Added articulated applicator, rigid proxy, removable cap, fresh/cured bead,
  interaction-frame, state-selection and coordinated-activation helpers.
* Added the complete system as an optional main operating-room bench asset.

0.4.0 (2026-07-24)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the DrAnmar Needle System 0.3.0 under
  ``Props/SurgicalClosure/Needle`` and
  ``Props/SurgicalClosure/NeedleThread``.
* Added standalone, coiled, extended and rigid-proxy Bench Setup routes.
* Added a dedicated six-DOF fixture-held articulated stapler test cell with bounded
  trigger control, pusher telemetry, rearm tracking and deterministic
  deployment evidence.

0.3.0 (2026-07-24)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the DrAnmar skin stapler under
  ``Props/SurgicalClosure/SkinStapler`` with rigid, articulated and standalone
  staple representations.
* Added I4H-compatible paths, Isaac Lab 2.3.2 and 3.0 configuration factories,
  state selection, semantics, trigger/pusher control, simulated deployment and
  closure-task helpers.
* Added the rigid loaded stapler as an optional prop in the main operating-room
  bench.

0.2.0 (2026-07-24)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the Apache-2.0 DrAnmar laparotomy-sponge component under
  ``Props/SurgicalCount/LaparotomySponge``.
* Added dry/wet folded-rigid and unfolded surface-deformable runtime helpers,
  I4H-compatible catalog paths, collision coverage, semantic labels and
  provisional physics presets.

0.1.1 (2024-08-24)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^

* The end effector frame for the dVRK PSM is updated to align with a Z-up orientation.

0.1.0 (2024-05-16)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Initial release of the extension.

* The following robot configurations are added:

  * The dVRK Patient Side Manipulator (PSM)
  * The dVRK Endoscopic Camera Manipulator (ECM)
  * The Smart Tissue Autonomous Robot (STAR)
