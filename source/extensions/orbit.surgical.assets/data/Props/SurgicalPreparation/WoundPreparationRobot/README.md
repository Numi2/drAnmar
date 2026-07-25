# DrAnmar Wound Preparation End Effector

Version: 0.1.0

A Franka-compatible, manufacturer-neutral wound-preparation end effector for
robotic inspection, irrigation, aspiration, controlled debridement, debris
removal, and fluid-accounting research.

## Core design

The tool uses a single concentric work head:

- ten inward-converging irrigation microjets;
- a twelve-slot annular suction crown;
- an extendable rotary debridement cartridge;
- a compliant force-sensing guard ring;
- stereo RGB, depth and fluorescence sensor frames;
- separate irrigation and collection inventories;
- direct replacement of the Panda hand at `panda_link8`.

The shared TCP allows irrigation, suction, debridement and inspection without a
tool exchange or change in robot kinematic reference.

## Catalog path

```text
Props/SurgicalPreparation/WoundPreparationRobot/
```

## Main assets

```text
dranmar_wound_preparation_tool_payload.usda
dranmar_wound_preparation_tool_standalone.usda
dranmar_wound_preparation_tool_rigid_proxy.usda
dranmar_irrigation_droplet.usda
dranmar_debridement_fragment.usda
dranmar_wound_bed_demo.usda
dranmar_debridement_brush_cartridge.usda
dranmar_debridement_curette_cartridge.usda
dranmar_debridement_pad_cartridge.usda
```

## Runtime scope

The supplied integration helper supports:

- standalone Isaac Lab articulation configuration;
- combined Franka + payload configuration;
- current PhysX surface-deformable cooking for the wound bed;
- PBD fluid particle creation and multi-nozzle emission;
- conservative fluid-volume bookkeeping;
- annular suction forces and particle capture;
- temporary debris-to-wound attachments;
- cumulative-work debris release;
- canonical wound-preparation phase targets.

The fluid implementation is a particle-scale task model, not CFD. The
debridement implementation releases discrete debris fragments and does not
claim tissue viability classification, biological cutting, bacterial reduction,
or clinical efficacy.

## Validation

Run the dependency-free package validator and controller tests before promotion:

```bash
python3 scripts/validate_dranmar_wound_preparation_robot.py --require-usdchecker
python3 -m unittest -v tests/test_wound_preparation_robot.py
```

The validator checks OpenUSD parsing, mirrored asset integrity, manifest hashes,
JSON contracts, GLB/PNG containers, Python syntax, release inventory, and
non-portable build-artifact exclusion. Bounded Isaac Sim/PhysX CUDA smoke
execution is recorded in `docs/VALIDATION.md`; physical calibration remains a
separate promotion gate.

## Deterministic regeneration

Install the pinned dependency families and run the generator from any Python
3.10-or-newer environment:

```bash
python3 -m pip install -r scripts/requirements_wound_preparation_generation.txt
python3 scripts/generate_dranmar_wound_preparation_robot.py
```

The generator cleans only its owned output paths, mirrors the catalog into the
extension data tree, writes fixed-timestamp ZIP members, and excludes bytecode
and workstation metadata from manifests and archives.

## Evidence boundary

This simulation-training workcell is available for task execution and
evaluation. Real-world and clinical evidence are not established. All
unmeasured mechanical, fluid, tissue, and contact values remain disclosed
engineering parameters.
