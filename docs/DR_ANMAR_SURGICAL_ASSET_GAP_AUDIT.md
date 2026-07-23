# DrAnmar Surgical Asset Gap Audit

Date: 2026-07-23

## Decision

The highest-impact missing foundational asset is **DrAnmar Suturable Tissue**:
a deformable, open-incision, solver-neutral tissue volume designed for needle
puncture, thread passage, stitch capture, wound approximation, pullout, local
damage, and sim-to-real calibration.

This is more important than adding another rigid instrument. A needle, suture,
driver, scissors, clip applier, or autonomous policy cannot demonstrate
surgical value without tissue that reacts, fails, and records evidence.

## What is currently available

The pinned NVIDIA Isaac for Healthcare v0.7.0 catalog provides:

- dVRK PSM/ECM, STAR, KUKA, Franka, Kinova, MIRA, SO-ARM and other robots;
- a curved suture needle, suture pad, scissors, table, tray and peg-board props;
- renderable anatomy, an abdominal phantom, ultrasound fixtures and hospital
  environments;
- Lightwheel R&D-only visual or task assets including trocars, drainage tubes,
  tweezers, puncture devices and deformable cloth.

The robotic-surgery workflow exposes reach, needle lift, handover and related
rigid-object tasks. The SO-ARM starter demonstrates rigid scissors
pick-and-place.

Direct inspection of NVIDIA's distributed `SuturePad` USD found one static
triangle collision mesh and no deformable-body API, tetrahedral simulation
mesh, puncture state, tear model, stitch constraint, wound state, or pullout
contract. The distributed scissors USD is one rigid mesh rather than an
articulated cutting mechanism.

## Important missing foundations

Scores are a DrAnmar product-prioritization tool, not clinical evidence.

| Candidate | Ubiquity | New interaction coverage | Platform gap | Sim-to-real leverage | Reuse | Weighted score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Suturable incised tissue | 30/30 | 25/25 | 20/20 | 15/15 | 10/10 | **100** |
| Articulated scissors with cutting | 25 | 20 | 18 | 11 | 9 | 83 |
| Vessel, clip and hemostasis system | 24 | 22 | 20 | 12 | 8 | 86 |
| Suction-irrigation with conserved fluid | 22 | 20 | 20 | 12 | 8 | 82 |
| Atraumatic gauze with absorption | 24 | 17 | 19 | 10 | 9 | 79 |
| Trocar and compliant abdominal wall | 20 | 17 | 14 | 12 | 8 | 71 |

The tissue asset wins because it unlocks all of the following with one
calibratable substrate:

- needle driving and path quality;
- tissue manipulation and counter-traction;
- thread passage and local friction;
- wound-edge approximation and eversion;
- knot-tension consequences;
- bite-depth and spacing studies;
- cheese-wiring, edge tearing, tract enlargement and pullout;
- force-feedback, perception and autonomous-policy benchmarks.

## State-of-the-art implementation contract

DrAnmar Suturable Tissue uses two disconnected, watertight tetrahedral flaps
with a real open incision rather than a painted line. A versioned procedural
generator authors:

- a stable surface mesh for native PhysX tetrahedral cooking;
- an explicit OpenUSD `TetMesh` for Isaac Lab 3, Newton and future backends;
- surface, bulk, fascia and wound-edge material regions, including an explicit
  layer ID on every tetrahedron;
- opposed outer attachment bands;
- deterministic topology and content hashes;
- a decomposed needle-force model: pre-puncture compression, cutting, tissue
  sweep and shaft friction;
- bite-dependent suture holding, cyclic damage and pullout;
- deterministic parameter sampling across tissue mechanics, wetness, contact,
  puncture, holding strength and appearance;
- a ten-item sim-to-real gap register and fail-closed qualification
  requirements.

The stable PhysX lane owns intact deformation, grasping and wound-edge
approximation. It does **not** claim arbitrary puncture or cutting. Needle
tracts, thread passage and topology change remain blocked until the MPM backend
passes the versioned puncture benchmark. This boundary prevents a visually
convincing but physically false demo from being promoted as medical fidelity.

## Research anchors

- [NVIDIA Isaac for Healthcare v0.7.0 catalog](https://github.com/isaac-for-healthcare/i4h-asset-catalog/blob/v0.7.0/catalog.md)
- [NVIDIA Isaac for Healthcare platform](https://developer.nvidia.com/isaac/healthcare)
- [Isaac Lab volume deformable architecture](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/tutorials/01_assets/run_deformable_object.html)
- [Needle-tissue force decomposition](https://pmc.ncbi.nlm.nih.gov/articles/PMC3966136/)
- [MIS suture-needle puncture experiments](https://doi.org/10.1016/j.bsbt.2016.05.001)
- [Needle-cutting fracture mechanics](https://pure.psu.edu/en/publications/fracture-mechanics-model-of-needle-cutting-tissue/)
- [Bite size and tissue thickness in suture pullout](https://pubmed.ncbi.nlm.nih.gov/2530644/)

## Validation boundary

The committed asset is an engineering research platform. Its geometry,
topology, deterministic mechanics functions and runtime integration can be
verified automatically. Biomechanical parameters remain broad research seeds.
Physical specimens, instrumented force/trajectory tests, held-out real video,
solver convergence, and independent clinical review are required before any
medical-fidelity or clinical-use claim.
