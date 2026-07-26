# Multimodal and generative simulation assets

Dr.Anmar treats a multimodal episode as a dependency-complete simulation asset,
not a loose image, video, NumPy array, or checkpoint. The contract is enforced
without importing Isaac Sim:

- `config/schemas/dranmar_multimodal_asset_bundle.schema.json` publishes the
  portable bundle shape.
- `config/schemas/dranmar_action_contract.schema.json` publishes dimension,
  unit, frame, timing, bound, interpolation, and stop semantics.
- `config/schemas/dranmar_timestamped_action_stream.schema.json` publishes the
  timestamped stream shape.
- `scripts/dr_anmar_multimodal_assets.py` verifies local hashes and sizes,
  immutable external revisions, authority boundaries, media geometry,
  action-stream timing and bounds, safe model handling, and evidence status.
- `scripts/generate_dranmar_multimodal_fixture.py` deterministically regenerates
  the dual-PSM transport and stale-stop fixture.

The first bundle is
`assets/dr_anmar/multimodal/cosmos_h_dreams_knot_tying_v1`. It binds the audited
Cosmos-H-Dreams example assets at one Git commit and the public model artifacts
at one Hugging Face revision. External bytes are not copied into Dr.Anmar.
Every referenced component has a byte size, full SHA-256, license identifier,
and immutable content URI.

## Authority boundary

Generated pixels can be displayed as a non-authoritative observation or used
for perceptual robustness studies. They cannot:

- create contact, attachment, puncture, cut, seal, bleeding, perfusion, vital,
  complication, success, or clinical labels;
- replace OpenUSD/PhysX state;
- command a simulated or physical robot;
- qualify a trajectory as a clinician demonstration; or
- establish native-simulator, real-world, or clinical evidence.

Only synchronized native simulation state, sensor packets, and action streams
on a declared clock can supply those functions. Remote control additionally
requires the existing authenticated TLS and operator-lease path.

## Media contract

The audited source is 640×480 while the referenced model checkpoint targets
512×288. A direct resize would deform anatomy and instruments. Dr.Anmar
therefore requires `letterbox` or `center_crop`; the canonical bundle specifies
a 384×288 content image with 64-pixel left and right padding. The upstream
media has no encoded color metadata, so the preview conversion records its
explicit but unverified sRGB assignment instead of implying measured color
fidelity.

Every produced preview must record target size, frame timestamps, output frame
rate, color space, generation seed, model revision, input component hashes, and
output hash. The source video and source trajectory remain rejected as a pair:
they have different implied durations and no shared clock or frame-to-action
mapping.

## Action behavior

The Dr.Anmar fixture uses the native two-PSM, 14-value policy contract:
six bounded NVIDIA joint-position inputs plus one binary gripper sign per PSM.
It begins and ends at neutral, uses a 20 Hz simulation clock, names every
dimension, declares frames and units, and constrains continuous per-sample
motion. Continuous axes interpolate linearly; grippers use sample-and-hold.

Sampling fails closed:

- non-finite time → neutral stop;
- input older than 250 ms → neutral stop;
- request outside the trajectory → neutral stop;
- invalid dimensions, bounds, or discrete values → bundle rejection.

The fixture is deterministic engineering data for ingestion, resampling, and
safety tests. It is not a recorded procedure and is not training-reference
eligible.

## Safe model handling

The upstream checkpoint and CR1 embedding use Torch pickle ZIP containers.
They are recorded as external, quarantined artifacts with runtime loading
disabled. Dr.Anmar does not call `torch.load` on them. Any future integration
must first provide a non-executable tensor container, verify the immutable
hash, pin the loader and model revision, isolate the generative runtime, and
retain the observation-only boundary.

## Promotion gates

Repository verification is available for the contract and fixture. Promotion
beyond it requires new evidence rather than a metadata change:

1. capture camera, robot action, resolved target, state, contact, mechanics,
   and patient-effect streams on one simulation clock;
2. record the exact Dr.Anmar, asset, Isaac, PhysX, driver, GPU, configuration,
   and seed revisions;
3. validate action/frame alignment, dropped frames, latency, stale behavior,
   task mechanics, complications, and measured outcomes;
4. compare the claimed physical behavior with instrumented bench data; and
5. obtain qualified clinician review before promoting any reference episode.

Clinical validation remains false.
