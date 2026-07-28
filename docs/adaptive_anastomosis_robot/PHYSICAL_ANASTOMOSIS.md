# DrAnmar physical anastomosis contract

## Tissue capture

Each tissue side is connected to six separate capture-cell colliders through current `OmniPhysicsVtxXformAttachment` schemas. Two explicit kinematic distal fixtures prevent unconstrained rigid drift of the cooked surfaces. Carriage motion therefore loads the tissue through the articulated mechanism. The tissue is not moved by directly overwriting its transform.

Surface self-collision is opt-in. The authored geometric wall is 2.4 mm thick and the provisional surface thickness is also 2.4 mm; enabling self-collision without retuning those values creates a full inner/outer-wall contact set rather than a qualified lumen-collapse model.

The portable solver mesh uses 16 axial by 32 circumferential segments per inner and outer wall. This keeps current CUDA surface-contact capacity bounded while retaining a closed, watertight hollow-wall topology; higher-resolution visual or calibrated solver meshes can be substituted through the same prim contract.

## Edge apposition and eversion

The lumen mandrel is inserted before final approximation. Eversion rings then move toward the central seam. Their rounded contact surfaces are intended to interact with the outer wall and rim region while the capture collars maintain distributed support.

## Staple retention

Sixteen formed staple bodies are spawned around the seam. Each staple owns two
independent attachment volumes: one for the left tissue and one for the right.
Source qualification now requires the exact shared rod result, exact bilateral
attachment set, intact topology, and observed bilateral reactions. Retention
remains false because no calibrated proof-load or pullout criterion exists.
Caller-authored pullout loads are not accepted.

Continuous metal plasticity, penetration, puncture damage, wall crushing, ischemia and cut-through require calibrated solver and specimen data and are not claimed by this release.

## Reinforcement collar

The reinforcement collar is supplied as a connected triangular surface and as
a rigid bond carrier with 16 independent left and 16 independent right cells.
A sector qualifies only from the exact latest shared cohesive-interface
response, exact bilateral attachments, nonfailed topology, positive contact
area, and positive bilateral reactions. This is provisional source mechanics,
not calibrated bonding, biochemical adhesion, cure, or healing.
