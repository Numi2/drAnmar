# Blood-flow and leak model

The package separates particle transport from the reduced-order source model. The source computes an orifice-flow estimate from pressure, effective defect area, density, and a discharge coefficient. Temporary compression, clip occlusion, and patch sealing reduce the effective area multiplicatively.

PhysX PBD particles represent emitted blood in the field. `HemorrhageLedger` conserves reservoir, active, suctioned, spilled, and discarded volumes. `AnnularSuctionController` accelerates particles toward the authored suction center and transfers captured particle volume into the collection ledger.

The verification controller integrates residual flow over a pressure-challenge observation window. It is a research benchmark and not proof of clinical hemostasis.
