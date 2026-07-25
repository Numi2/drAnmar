# NVIDIA Isaac for Healthcare asset catalog v0.7

Dr.Anmar consumes the NVIDIA catalog as a pinned upstream provider. It does not copy the catalog into this
repository, rename physical assets, or replace their authored OpenUSD physics.

## Immutable provider

- Repository: `https://github.com/isaac-for-healthcare/i4h-asset-catalog`
- Release: `v0.7.0`
- Source commit: `b0b7ad39f26490d58d12407cfa74b3c9ad861769`
- Asset version: `0.7.0`
- Content address: `724f82e`
- Active source link: `${DR_ANMAR_ROOT}/vendor/i4h-asset-catalog-current`
- Local content root: `${I4H_ASSET_DOWNLOAD_DIR}/724f82e`

The upstream helper retrieves public assets from NVIDIA's production object store. No asset payload is
committed to Dr.Anmar.

## DrAnmar-authored catalog extensions

DrAnmar also ships its own catalog content under the same relative-addressing
convention. Licensing is repository- and asset-specific; local content must
not be assumed to inherit the Apache-2.0 license of the upstream catalog
helper.

```text
Props/Patients/...
Props/SurgicalAssessment/...
Props/SurgicalClosure/...
Props/SurgicalCount/...
Props/SurgicalDissection/...
Props/SurgicalDivision/...
Props/SurgicalExposure/...
Props/SurgicalHemostasis/...
Props/SurgicalOncology/...
Props/SurgicalPreparation/...
Props/SurgicalReconstruction/...
assets/dr_anmar/...
```

These local extensions are not represented as NVIDIA-authored catalog content.
The repository registry discovers complete asset directories and validates
their manifests, licensing evidence, JSON, and relative OpenUSD dependencies.
The `DrAnmarSurgicalRobotAssets` compatibility interface exposes eight
procedure-specific standalone robot paths in the form expected by
`BaseI4HAssets`. DrAnmar's capability payload is generated from all 19
authoritative portfolio entries and reports provider, path, representation,
qualification state, and complete declared-artifact readiness.

The surgical-oncology extension adds payload-backed standalone and
Franka-mounted resection-tool routes, a rigid proxy, registered liver and
three-station workcell, multimodal sensor frames, tumor/margin state, protected
pedicle interlocks, specimen containment, a fail-closed Dynamic Patient GPU
volume-liver route, and an Isaac Lab episode contract.
It is documented in [`SURGICAL_ONCOLOGY.md`](SURGICAL_ONCOLOGY.md).

The complete local catalog contract and validation commands are documented in
[`ASSET_CATALOG.md`](ASSET_CATALOG.md).

The sponge package keeps both dry/wet representations runnable. The topical
skin-adhesive package adds an articulated applicator, removable rigid cap,
fresh/cured deposit state, and coordinated activation helper. The needle
system adds independently selectable standalone, coiled, extended and rigid
proxy routes. The stapler package adds loaded/empty state handling, stable and
articulated runtime routes, semantics, trigger/pusher control, simulated
deployment bookkeeping, an optional loaded rigid prop and a dedicated fixed
test-cell room. Provisional physics and nonclinical provenance are recorded in
[`LAPAROTOMY_SPONGE.md`](LAPAROTOMY_SPONGE.md) and
[`SKIN_STAPLER.md`](SKIN_STAPLER.md), the adhesive package in
[`DRANMAR_SKIN_ADHESIVE.md`](DRANMAR_SKIN_ADHESIVE.md), and the needle package
documented
in [`DR_ANMAR_NEEDLE_SYSTEM_0_3.md`](DR_ANMAR_NEEDLE_SYSTEM_0_3.md).

## Useful Dr.Anmar bundles

| Bundle | Upstream content | Approximate size |
| --- | --- | ---: |
| `surgical-core` | dVRK PSM/ECM, STAR, suture needle/SDF, suture pad, table and instruments | 82 MiB |
| `surgical-anatomy` | Renderable organ assembly and its materials | 939 MiB |
| `ultrasound` | Abdominal phantom, ultrasound fixtures, Vention table and Franka | 834 MiB |
| `medical-robots` | KUKA LBR14/LBR7Med and Kinova KIMA | 88 MiB |
| `rheo` | Lightwheel scene, deformable cloth, tube, trocars, trays and equipment | 318 MiB |

Install and retrieve a bundle with:

```bash
./scripts/install_i4h_asset_catalog.sh
./scripts/fetch_i4h_assets.sh surgical-core
```

The helper writes an installation record to
`${DR_ANMAR_ROOT}/run/i4h_asset_catalog.json`. Dr.Anmar's capability API reports the source pin, asset content
address, declared canonical paths, local readiness and the upstream asset root used by NVIDIA Arena.

## Runtime boundary

NVIDIA's I4H v0.7 Agentic surgical environments currently reference the catalog's v0.5 content prefix in
their own `arena/assets/constants.py`. Dr.Anmar preserves that exact source contract. Installing the v0.7
catalog does not mutate an Arena environment, change an asset URL, or imply that the v0.7 copy is
physics-equivalent.

The v0.7 provider is used when composing a new Dr.Anmar research room from current NVIDIA assets. Promotion
still requires native scene loading, contact/constraint evidence and a recorded qualification result.

## Gilgamesh installation evidence

On 2026-07-23, the source and helper were installed on Gilgamesh without restarting the existing room:

- source commit and exact tag: `b0b7ad39f26490d58d12407cfa74b3c9ad861769`, `v0.7.0`;
- helper-managed content address: `0.7.0/724f82e`;
- surgical core: 9 files, 85,337,862 bytes;
- live Dr.Anmar room on port 2396 remained reachable;
- all nine surgical-core objects have the same object size and S3 ETag at NVIDIA's Arena v0.5 content prefix
  and the catalog v0.7 content prefix.

The equality check covers the dVRK PSM/ECM, STAR, needle, SDF needle, suture pad, table, scissors and tray.
It supports reusing the locally cached copies for these assets. It does not authorize replacing an upstream
Arena URL or imply that other v0.7 assets are unchanged.

| Evidence | SHA-256 |
| --- | --- |
| `logs/i4h-asset-catalog-v0.7.0-install.log` | `59a7da099c6c39a2a24b1cd0d8bd66ebb964400f0bb5389f98e0d355bc313e9a` |
| `logs/i4h-asset-catalog-v0.7.0-surgical-core.log` | `abab52cbb783a87f50bf20dee37e70e8302fa9199bdda2f259f021feec19f5ca` |
| `run/i4h_asset_catalog.json` | `c7e5ee64c313dd32015e22420bed2a27cae877df759e7cffc446eca5b886142f` |

## Licensing

The catalog helper source is Apache-2.0. Downloaded assets can carry their own terms and must be reviewed
before redistribution. NVIDIA explicitly identifies the Lightwheel SimReady assets as non-commercial,
research-and-development-only content. Dr.Anmar records this restriction and never classifies those assets as
commercial-safe.
