# Data governance boundary

Dr.Anmar is simulation and preclinical research software. Its default workstation records simulated robot,
anatomy, sensor, and browser-control telemetry. It does not request patient identifiers, patient images,
operator video, voice, or measured eye gaze.

The pointer channel is labelled `pointer_attention_proxy`; it must not be represented as eye tracking.
External eye-tracker and XR gaze ingestion fails closed unless the deployment deliberately sets all of:

- `DR_ANMAR_ENABLE_EXTERNAL_OPERATOR_SENSORS=1`
- a non-empty `DR_ANMAR_STUDY_ID`
- a non-empty `DR_ANMAR_CONSENT_PROTOCOL`

Before enabling those values, the research owner must define consent, lawful basis, allowed operators,
pseudonymization, retention/deletion, export controls, incident handling, and institutional approval. The study
protocol—not this repository—owns those decisions. Each resulting demonstration manifest records the configured
study and consent-protocol identifiers so an ungoverned dataset cannot be mistaken for approved study evidence.

Access to a shared deployment should use `DR_ANMAR_ACCESS_TOKEN` and HTTPS with
`DR_ANMAR_COOKIE_SECURE=1`. The application lease prevents simultaneous robot commands; it is not a substitute
for institutional identity, authorization, audit logging, or a managed research-data repository.

## Simulation-expert trajectories

Executable expert runs contain generated robot actions and simulated sensor/mechanics data; they are not human
expert recordings. Each manifest records controller status, phase completion, interventions, degraded reasons,
physics authority and calibration state. A clean uninterrupted run may be marked
`behavior_cloning_reference_candidate=true`, but remains `pending_clinician_review` until a qualified reviewer
accepts it under the study protocol. A warning-laden run is retained for debugging and Failure Lab analysis and
must not be silently mixed into an approved reference dataset.

Documentation GIFs are visual explanations only. They are not dataset inputs and do not replace the checksummed
NPZ/JSON trajectory pair. See [`EXECUTABLE_EXPERT_GUIDANCE.md`](EXECUTABLE_EXPERT_GUIDANCE.md) for the phase,
qualification and capture contract.

## Demonstration integrity

New captures use `dr.anmar.demonstration.v3`. The manifest retains the exact
robot body and joint ordering; coordinate, velocity, and effort units for every
joint; a shared monotonic clock contract; camera intrinsics; world-frame camera
extrinsics aligned to every vision sample; array shapes, dtypes, units,
coordinate frames, and authority; and the NPZ SHA-256.

Validate a retained capture before replay, training, or analysis:

```bash
python3 scripts/dr_anmar_telemetry.py /path/to/demonstration.json
```

The validator rejects hash drift, missing arrays, frame-count mismatches,
non-monotonic control or vision clocks, invalid depth, non-unit world
quaternions, inconsistent joint ordering/units, and invalid camera
intrinsics. This establishes dataset integrity, not sensor calibration or
ground-truth accuracy; those native and physical gates remain separate.

## Generative observations

Generated frames are non-authoritative observations. They cannot supply robot
commands, physics state, patient effects, complication labels, task success, or
clinical evidence. A generative asset must bind immutable source and model
revisions, component hashes, media preprocessing, frame timestamps, action
dimensions, units, coordinate frames, sample clock, pairing status, and output
provenance. Unpaired media and action streams fail closed for replay and
training.

Torch pickle checkpoints and embeddings remain quarantined and runtime-disabled
until converted to a non-executable tensor container and separately approved.
See [`MULTIMODAL_GENERATIVE_ASSETS.md`](MULTIMODAL_GENERATIVE_ASSETS.md).
