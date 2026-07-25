# Hydrodissection and Energy Models

The hydrodissection helper creates a PhysX PBD particle system, emits quantized seven-port bursts from the authored nozzle frame, and conserves reservoir, active, aspirated, absorbed, and spilled volume in a ledger. Bridge weakening uses a distance-weighted delivered-volume proxy.

The low-energy probe uses a lumped thermal model with force-dependent absorption, heat loss, temperature, delivered energy, smoke generation, and overtemperature state. It does not reproduce electrosurgical current paths, collagen transformation, steam, charring, or tissue-specific thermal spread.

The annular suction controller accelerates fluid particles toward the authored suction center and transfers captured particle volume into the collection ledger.
