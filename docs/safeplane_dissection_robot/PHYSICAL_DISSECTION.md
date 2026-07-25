# Physical Safe-Plane Dissection Contract

## Tissue connectivity

The demonstration substrate contains two independent triangular tissue surfaces connected by 28 discrete adhesion bridges. Each bridge consists of an upper rigid anchor attached to the superficial flap, a lower rigid anchor attached to the target bed, and one removable fixed continuity joint between the anchors. Each joint meets at the authored mid-plane rather than constraining the separated anchor origins together.

The target bed is attached to two explicit kinematic fixtures. Surface
self-collision is disabled by default for portable GPU deformable cooking and
can be enabled explicitly only on a qualified solver configuration.

Releasing a bridge removes its continuity joint. The anchor halves remain attached to their respective tissue layers, so separation is produced by the physics solver and traction mechanism rather than by directly rewriting the tissue transform.

## Four release mechanisms

- blunt spreading accumulates mechanical contact work;
- hydrodissection accumulates local delivered fluid volume and weakens mechanical thresholds;
- guarded scissors release the bridge nearest the cut volume after guard and safety interlocks pass;
- low-energy dissection accumulates a provisional local thermal-energy dose.

All bridge thresholds are category-level engineering seeds and are not biomechanical or clinical claims.
