# Physical Safe-Plane Dissection Contract

## Tissue connectivity

The demonstration substrate contains two independent triangular tissue surfaces connected by 28 discrete adhesion bridges. Each bridge consists of an upper rigid anchor attached to the superficial flap, a lower rigid anchor attached to the target bed, and one removable fixed continuity joint between the anchors. Each joint meets at the authored mid-plane rather than constraining the separated anchor origins together.

The target bed is attached to two explicit kinematic fixtures. Surface
self-collision is disabled by default for portable GPU deformable cooking and
can be enabled explicitly only on a qualified solver configuration.

The authored representation permits a continuity joint to be removed while its
anchor halves remain attached to their respective tissue layers. That is an
asset-level topology mechanism, not evidence that a bridge failed under tool
contact. Public bridge release currently fails closed because no exact-step
scene-evidence adapter derives cohesive damage or disconnect state from the
physics solver. The former caller-supplied release logic is retained only as a
private task proxy.

## Proposed release modalities

- blunt spreading requires contact-derived force, relative motion, and integrated
  work;
- hydrodissection requires particle/tissue contact, deposited volume, and fluid
  mass balance;
- guarded scissors require live guard/blade transforms, contact/crossing
  evidence, and a topology event;
- low-energy dissection requires measured electrical/thermal delivery and shared
  tissue damage state.

None of those evidence bridges exists in this package today. The static
distance-weighted work, volume, and energy thresholds remain private,
non-authoritative task proxies. All bridge thresholds are category-level
engineering seeds and are not biomechanical or clinical claims.
