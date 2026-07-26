# Dr.Anmar asset catalog

Dr.Anmar uses one catalog contract for its bundled OpenUSD and multimodal
simulation assets plus the external NVIDIA Isaac for Healthcare asset provider.
The contract is designed to keep assets portable, reviewable, and reproducible
without loading Isaac Sim merely to inspect the tree.

## What is authoritative

- `config/dranmar_asset_catalog.json` pins the i4h Git release, full commit,
  content version, content hash, supported download bundles, local provider
  roots, generated release artifacts, and minimum metadata policy.
- `config/dranmar_asset_catalog.lock.json` is the checked-in,
  content-addressed identity of every dependency-complete local asset unit and
  every product-facing portfolio entry.
- `catalog.md` is generated from that lock and is the reviewable human index.
- `physics_next/benchmarks/dranmar-portfolio-evidence-index.json` binds every
  portfolio claim to content-addressed declared artifacts and the parent or
  asset-submodule revision that contains them. A native artifact's own tested
  revision remains the execution authority.
- `scripts/dr_anmar_asset_registry.py` resolves provider-relative paths,
  inventories every local asset family, verifies JSON and textual USDA
  dependencies, validates the full product portfolio, and verifies the
  canonical lock and generated index by default.
- `scripts/dr_anmar_multimodal_assets.py` validates content-addressed image,
  video, action, statistics, model, evidence, timing, geometry-preservation,
  and authority contracts. Generative frames remain observation-only.
- `scripts/dr_anmar_i4h_adapter.py` derives its Dr.Anmar capability entries
  from the complete 21-entry `physics_next/dr-anmar-assets.json` portfolio,
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

The current structural gate discovers 29 local asset units and 139 direct USD
or registered data entrypoints. These counts are generated from the tree; they
are not a hand-maintained allow-list. The smaller runtime
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

Validate multimodal bundles and the deterministic action fixture:

```bash
python3 scripts/generate_dranmar_multimodal_fixture.py --check
python3 scripts/dr_anmar_multimodal_assets.py
python3 scripts/generate_dranmar_evidence_index.py --check
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
2. Use a direct USD stage or registered `asset_bundle.json` data entrypoint.
3. Keep all USD-relative dependencies inside the repository and use relative
   references. Absolute workstation paths fail the catalog gate.
4. Add the applicable license or preserve the provider notice.
5. Add a manifest, physics profile, mechanics contract, or qualification
   report at the asset-family root.
6. Add a README describing entrypoints, representations, supported simulator
   lanes, state variants, known limits, and the non-clinical boundary.
7. For media/data/model bundles, record full hashes, byte sizes, immutable
   revisions, dimension and clock semantics, preprocessing, safe loading, and
   evidence/authority boundaries.
8. Register room-visible entrypoints with a known provider:
   `dr_anmar`, `dr_anmar_repository`, or `nvidia_i4h`.
9. Run the structural gate, asset-specific tests, and the relevant native
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

Repository-owned source assets are additionally bounded by
`quality.growth_budgets` in `config/dranmar_asset_catalog.json`. The catalog
gate rejects an asset unit above 64 MiB or aggregate catalog source above
512 MiB. Raising either limit is a deliberate, reviewable policy change.
Generated previews, checkpoints, recordings, datasets, and caches do not
belong in those budgets because they remain external runtime data.
