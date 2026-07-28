# Hydrodissection and Energy Models

The hydrodissection helper creates a PhysX PBD particle system and emits
quantized seven-port bursts from the authored nozzle frame. Its ledger accounts
for commanded emission and particles removed by the scene-space suction pass.
It does not measure jet pressure, tissue contact, deposited volume, absorption,
or patient fluid balance. Caller-authored absorption and spillage are
fail-closed, and the old distance-weighted bridge weakening calculation is a
private task proxy only.

The former low-energy calculation is a private lumped task proxy with
caller-supplied force, time, and power. Public energy/dose mutation fails closed
until scene evidence provides contact, electrical delivery, impedance,
temperature, and shared tissue-damage state. The proxy does not reproduce
electrosurgical current paths, collagen transformation, steam, charring, or
tissue-specific thermal spread.

The annular suction controller accelerates authored fluid particles toward the
suction center and transfers geometrically captured particle volume into the
collection inventory. This is particle bookkeeping, not evidence of
hydrodissection, tissue absorption, clinical irrigation, or patient fluid loss.
