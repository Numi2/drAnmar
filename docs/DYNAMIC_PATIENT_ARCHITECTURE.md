# Architecture

The patient is split into four coupled but independently replaceable layers:
anatomy, mechanics, physiology, and evidence-bound effects. Anatomy owns names,
transforms, geometry, and semantics. Mechanics owns solver representations,
attachments, and topology. Physiology owns continuous state and a shared blood/
fluid ledger. Blood and irrigation have partial conservation; bile and generic
suction remain output counters without source-compartment conservation. Effects
may cross into patient state only from an
owning mechanics subsystem after exact post-physics scene evidence is accepted.
The former caller-authored intervention registry and robot adapter are disabled.

The physiology runtime never assumes a particular mesh topology. Organ geometry can therefore be upgraded without rewriting cardiovascular, respiratory, renal, biliary, coagulation, or vital-sign models.

This architecture is source-hardened but incomplete: not every workcell has a
native provider that supplies the required evidence, and direct low-level model
objects remain publicly mutable research implementation details rather than an
evidence-owned capability boundary. That mutability blocks a B source grade.
