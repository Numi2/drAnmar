# Mechanism

The end effector mounts to `panda_link8` and replaces the stock Panda hand.

## Concentric work head

The central spindle carries the debridement cartridge. Ten irrigation nozzles
surround the spindle and converge toward the work axis. Twelve suction slots
form an outer annulus, reducing the need to reorient the arm between fluid
delivery and recovery.

## Moving links

- `ContactGuard`: spring-driven 8 mm prismatic compression range.
- `DebridementCarriage`: 20 mm tool extension range.
- `DebridementRotor`: continuous revolute spindle.
- `IrrigationValve`: 6 mm metering-spool travel.
- `SuctionValve`: 0–85 degree valve opening.

The guard establishes standoff and provides a contact frame before the brush or
curette reaches the wound bed. The default cartridge is the soft brush. Separate
ring-curette and microtextured-pad assets use the same cartridge frame.
