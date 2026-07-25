# Dr.Anmar asset catalog

Dr.Anmar uses one catalog contract for its bundled OpenUSD assets and the
external NVIDIA Isaac for Healthcare asset provider. The contract is designed
to keep simulation assets portable, reviewable, and reproducible without
loading Isaac Sim merely to inspect the tree.

## What is authoritative

- `config/dranmar_asset_catalog.json` pins the i4h Git release, full commit,
  content version, content hash, supported download bundles, local provider
  roots, generated release artifacts, and minimum metadata policy.
- `config/dranmar_asset_catalog.lock.json` is the checked-in,
  content-addressed identity of every dependency-complete local asset unit and
  every product-facing portfolio entry.
- `catalog.md` is generated from that lock and is the reviewable human index.
- `scripts/dr_anmar_asset_registry.py` resolves provider-relative paths,
  inventories every local asset family, verifies JSON and textual USDA
  dependencies, validates the full product portfolio, and verifies the
  canonical lock and generated index by default.
- `scripts/dr_anmar_i4h_adapter.py` derives its Dr.Anmar capability entries
  from the complete 19-entry `physics_next/dr-anmar-assets.json` portfolio,
  including the readiness of every declared artifact closure.
- Each modern asset family lives under a stable category path such as
  `Props/SurgicalClosure/SkinStapler` or
  `Props/Patients/DynamicAbdominalPatient`.
- The existing per-asset manifests, physics profiles, interaction frames,
  qualification reports, licenses, and notices remain the detailed source of
  truth. The registry does not replace physics-specific qualification.

The catalog structure follows the useful parts of the upstream
[`i4h-asset-catalog`](https://github.com/isaac-for-healthcare/i4h-asset-catalog):
stable relative paths, an explicit provider release and content hash, lazy
subpath retrieval, deterministic folder hashing, and simulator-independent
tests.

The current structural gate discovers 27 local asset units and 127 direct USD
entrypoints. These counts are generated from the tree; they are not a
hand-maintained allow-list. The smaller runtime
`DrAnmarSurgicalRobotAssets` interface exposes the eight procedure-specific
standalone/payload/proxy robot families, including oncologic resection.

## Commands

Run the fast structural and runtime-reference gate:

```bash
python3 scripts/dr_anmar_asset_registry.py verify
```

Inspect every bundled asset unit:

```bash
python3 scripts/dr_anmar_asset_registry.py inventory
```

Resolve a path exactly as the hub and workstation do:

```bash
python3 scripts/dr_anmar_asset_registry.py resolve \
  dr_anmar \
  Props/SurgicalClosure/SkinStapler/skin_stapler_rigid_proxy.usda \
  --require
```

Update the canonical release artifacts only after the asset tree has passed
its applicable structural, native-physics, visual, and physical qualification:

```bash
python3 scripts/dr_anmar_asset_registry.py lock
python3 scripts/dr_anmar_asset_registry.py catalog
python3 scripts/dr_anmar_asset_registry.py verify
```

The gate fails when the asset tree, portfolio, lock, or generated catalog
drifts. `verify --skip-release-artifacts` is available only for development
while preparing a coordinated lock update. The lock includes one deterministic
SHA-256 per asset family plus an overall self-digest. It is a release integrity
artifact, not evidence that an asset is clinically validated or that its
Isaac/PhysX behavior passed a GPU qualification.

## Adding or importing an asset

1. Place it under a stable category path. Do not make a loose root-level asset.
2. Keep all USD-relative dependencies inside the repository and use relative
   references. Absolute workstation paths fail the catalog gate.
3. Add the applicable license or preserve the provider notice.
4. Add a manifest, physics profile, mechanics contract, or qualification
   report at the asset-family root.
5. Add a README describing entrypoints, representations, supported simulator
   lanes, state variants, known limits, and the non-clinical boundary.
6. Register room-visible entrypoints with a known provider:
   `dr_anmar`, `dr_anmar_repository`, or `nvidia_i4h`.
7. Run the structural gate, asset-specific tests, and the relevant native
   Isaac/PhysX qualification before updating both canonical release artifacts.

External i4h payloads are downloaded separately and retain their own license
terms. In particular, research-only or non-commercial assets must never be
silently promoted into a commercial Dr.Anmar release.

Each partial NVIDIA bundle download receives a v2 installation receipt with a
SHA-256 over every requested file or directory closure. Verify local bytes
against that receipt with:

```bash
python3 scripts/dr_anmar_i4h_receipt.py verify
```
