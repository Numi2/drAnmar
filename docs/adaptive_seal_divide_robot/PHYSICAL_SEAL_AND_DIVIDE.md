# Physical Seal and Division Contract

## Intact tissue representation

The demo vessel contains two watertight deformable halves connected by 8 temporary bridge pins. Each pin is attached physically to both halves. Two explicit kinematic distal fixtures keep the ends grounded. The initial visual bridge conceals the submillimetre seam.

Surface self-collision is disabled by default for portable GPU deformable
cooking. It can be enabled explicitly for a qualified solver configuration.

## Temporary jaw compression

The runtime compression controller creates four verified deformable
attachments: upper and lower jaw seal contacts for each future stump. Force is
reported by the caller and checked against the provisional soft and hard
limits. These attachments are temporary and are released only after both
retained seal bands exist.

## Seal retention

A seal operation deploys one `DrAnmarTissueSealBand` on each future stump. Each band has independent upper and lower bond volumes. The vessel is compressed when those attachments are created, so the band preserves the collapsed wall configuration after jaw release.

Seal-band state progresses from `fresh` to `mature`. Excess load removes the attachments and changes the state to `failed`.

## Division

The blade does not silently replace the vessel mesh. Blade progress releases bridge-pin attachments in a defined order. At complete progress, the two deformable halves are mechanically independent while their seal bands remain attached.

This is a controlled topology-surrogate strategy. It does not claim continuous fracture, histological thermal fusion, or clinically validated stump strength.
