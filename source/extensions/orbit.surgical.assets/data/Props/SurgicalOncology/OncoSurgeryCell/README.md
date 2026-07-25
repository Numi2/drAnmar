# DrAnmar OncoSurgery Cell

Manufacturer-neutral research assets and executable mechanics for multimodal
tumor mapping, margin-constrained resection, protected-pedicle management,
specimen containment and orientation, cavity verification, and corrective
resection in NVIDIA Isaac Lab and PhysX.

## Runtime entry points

- `dranmar_oncosurgery_tool.usda` is the lightweight payload-backed standalone
  interface.
- `dranmar_tumor_resection_tool_payload.usda` replaces the Franka hand at
  `panda_link8`.
- `dranmar_tumor_resection_tool_rigid_proxy.usda` is the lower-cost
  perception/contact route.
- `dranmar_oncology_liver.usda` is the registered task-state liver interface.
- `dranmar_oncosurgery_workcell.usda` composes the three-station research cell.
- `orbit.surgical.assets.oncologic_resection` provides Isaac Lab factories,
  multimodal fusion, tumor and margin accounting, resection-bond interlocks,
  specimen state, episode observations, reward, domain randomization, and a
  fail-closed Dynamic Patient GPU volume-deformable liver route.

The interface layers follow NVIDIA's reference-payload asset structure: public
variants and identity remain cheap to inspect, while the authored geometry and
physics can be unloaded in larger compositions. All composition arcs are
relative so the catalog subtree is relocatable.

## Safety and realism boundary

Protected vascular and bile-duct bonds cannot be divided until a simulated
compression-and-energy seal is confirmed. Sensor results abstain when
registration, timing, modality count, confidence, or cross-modality agreement
fails. Final success requires zero residual modeled tumor, a 10 mm modeled
margin, intact protected structures, bounded blood and bile loss, detached and
closed specimen containment, complete orientation markers, a cavity scan, and
a hemostasis/bile check.

The 3-D tumor field and discrete bonds are research task models. The demo liver
is not incorrectly treated as a thin-shell deformable. Native tissue mode uses
the Dynamic Abdominal Patient liver's explicit tetrahedral PhysX GPU volume
deformable for continuous nodal motion and collision. PhysX does not mutate
that mesh topology at runtime, so the registered bond graph remains the
irreversible resection and detachment authority. Whole-patient use activates
exactly one liver and preserves the shared blood/bile ledgers.

The material values are uncalibrated research seeds. This package is not
clinically validated, is not a medical device, and is not approved for patient
care.
