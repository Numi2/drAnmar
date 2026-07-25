# Architecture

The patient is split into four coupled but independently replaceable layers: anatomy, mechanics, physiology, and interventions. Anatomy owns names, transforms, geometry, and semantics. Mechanics owns solver representations and attachments. Physiology owns continuous state. Interventions translate robot actions into changes in mechanics and physiology.

The physiology runtime never assumes a particular mesh topology. Organ geometry can therefore be upgraded without rewriting cardiovascular, respiratory, renal, biliary, coagulation, or vital-sign models.
