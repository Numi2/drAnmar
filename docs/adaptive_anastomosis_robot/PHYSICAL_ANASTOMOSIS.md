# DrAnmar physical anastomosis contract

## Tissue capture

Each tissue side is connected to six separate capture-cell colliders through current `OmniPhysicsVtxXformAttachment` schemas. Two explicit kinematic distal fixtures prevent unconstrained rigid drift of the cooked surfaces. Carriage motion therefore loads the tissue through the articulated mechanism. The tissue is not moved by directly overwriting its transform.

Surface self-collision is opt-in. The authored geometric wall is 2.4 mm thick and the provisional surface thickness is also 2.4 mm; enabling self-collision without retuning those values creates a full inner/outer-wall contact set rather than a qualified lumen-collapse model.

The portable solver mesh uses 16 axial by 32 circumferential segments per inner and outer wall. This keeps current CUDA surface-contact capacity bounded while retaining a closed, watertight hollow-wall topology; higher-resolution visual or calibrated solver meshes can be substituted through the same prim contract.

## Edge apposition and eversion

The lumen mandrel is inserted before final approximation. Eversion rings then move toward the central seam. Their rounded contact surfaces are intended to interact with the outer wall and rim region while the capture collars maintain distributed support.

## Staple retention

Sixteen formed staple bodies are spawned around the seam. Each staple owns two independent attachment volumes: one for the left tissue and one for the right. The 12 temporary capture constraints can then be released while the 32 staple-leg attachments remain the load-bearing bridge. Pullout is represented by removing a staple's tissue attachments when the caller reports load above a provisional threshold.

Continuous metal plasticity, penetration, puncture damage, wall crushing, ischemia and cut-through require calibrated solver and specimen data and are not claimed by this release.

## Reinforcement collar

The reinforcement collar is supplied as a connected triangular surface and as a stable rigid bond carrier. The rigid carrier contains 16 independent left and 16 independent right bond cells. Bond strength rises from a provisional 0.18 N sector tack value to 2.2 N over 45 seconds. This models the mechanical result of reinforcement but not biochemical adhesion or healing.
