# Blood-flow and leak model

The shared reduced-order vessel derives inlet/outlet/leak/storage flows, lumen
area, damage, and a continuity residual from pressure boundaries and contact
loads. It is a quasi-steady unidirectional model, not blood CFD.

PhysX PBD particles represent emitted blood in the field. `HemorrhageLedger` conserves reservoir, active, suctioned, spilled, and discarded volumes. `AnnularSuctionController` accelerates particles toward the authored suction center and transfers captured particle volume into the collection ledger.

The verification controller integrates exact envelope-bound residual flow and
pressure over a challenge window. It is a provisional simulator benchmark and
not proof of retained repair or clinical hemostasis.
