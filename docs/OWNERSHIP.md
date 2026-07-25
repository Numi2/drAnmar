# Dr.Anmar ownership and provenance

Dr.Anmar is the product: it owns the clinician-facing workflow, procedure
catalog, room composition, controls, safety boundaries, expert guidance,
recording, evaluation, and research evidence contracts.

The simulator and provider runtimes are implementation dependencies. They are
credited here so that Dr.Anmar can take product ownership without claiming
ownership of upstream code, assets, solvers, or sensor models.

## Dr.Anmar owns

- Doctor Studio, the learning loop, curriculum presentation, and procedure UX.
- Procedure definitions, room bindings, launch/readiness checks, and recovery
  behavior.
- Native PSM interaction contracts, gripper/contact interpretation, safety
  stops, operator leases, and worker lifecycle coordination.
- The eight-phase expert controller, phase/event annotations, coaching,
  intervention handling, and reference qualification.
- Demonstration manifests, replay inspection, multimodal capture, dataset
  cards, policy-evaluation cards, challenge matrices, and provenance hashes.
- OpenUSD composition, asset sanitization, room presentation, and the boundary
  between visual anatomy and native physical state.
- The capability and evidence contracts that determine what Dr.Anmar may call
  ready, research-qualified, or clinically unvalidated.

## Dr.Anmar's local simulation substrate

The `source/extensions/orbit.surgical.*` packages are an ORBIT-Surgical-derived
foundation for dVRK PSM/ECM and STAR assets, reach, lift, and handover task
families, Isaac Lab registration, and standard learning workflows. The current
checkout is a port and modification of that foundation, not an untouched
upstream checkout. Dr.Anmar-specific contact, grasp, reset, reward,
termination, and data-contract changes are part of the local product surface.

Isaac Sim, Isaac Lab, and PhysX execute the articulation, rigid-body, contact,
sensor, and solver state. ORBIT-derived configuration describes those entities;
it does not replace the underlying NVIDIA simulation engine.

## External providers

These integrations remain provider-owned behind Dr.Anmar contracts:

| Provider | Provider-owned responsibility | Dr.Anmar-owned responsibility |
| --- | --- | --- |
| Isaac Sim / Isaac Lab / PhysX | Runtime, solver, articulation, contact, and sensor execution | Room contract, controls, lifecycle, evidence, and readiness presentation |
| ORBIT-Surgical-derived substrate | Base robot assets and task implementations | Product rooms, procedure semantics, gates, and Dr.Anmar modifications |
| NVIDIA Isaac for Healthcare | Official healthcare workflows, sensor physics, hardware bridges, and Arena state machines | Guarded discovery, launch, study context, logs, and provenance |
| NVIDIA SoftMimicGen | Released strand/ring deformable task and replay state | Room integration, ORBIT PSM composition, qualification boundary, and evidence packaging |
| SonoGym | L4 patient assets, ultrasound generation, task stepping, rewards, and safety constraints | Browser bridge, room catalog, study flow, and source/asset pinning |

## Product-language rule

Product-facing copy should lead with Dr.Anmar and describe external systems as
technical foundations or providers. Technical documentation must retain the
upstream names, licenses, citations, revisions, and ownership boundaries.

Dr.Anmar must not claim native tissue, ultrasound, clinical validation, or
physical-robot control unless the corresponding provider and evidence gates
have actually passed.

Product availability and evidence strength are different dimensions. Use the
repository-wide [capability and evidence language](EVIDENCE_LEVELS.md): lead
with the available Dr.Anmar training capability, then state repository,
native-simulator, real-world, and clinical evidence separately.
