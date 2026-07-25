# Multimodal sensor model

RGB cameras provide scene context. ICG-like fluorescence is generated from graph-transport tracer history. Laser speckle reads normalized regional flow. Thermal output uses a perfusion heat-transfer proxy. Surface oxygenation uses an oxygen-delivery and consumption proxy. Doppler projects solved edge velocity onto the probe beam. Ultrasound can use the supplied synthetic B-mode generator or bridge to the NVIDIA Isaac for Healthcare robotic-ultrasound application using the authored probe pose.

The estimator receives registered observable maps and temporal ICG metrics only. Scenario labels and latent flow fields are excluded from inference and may be supplied only as evaluation annotations. Failed modalities are removed and remaining weights are renormalized. Registration error, timestamp skew, insufficient modality coverage, or low diagnostic confidence produces an explicit abstention.

Contrast and coupling gel use conservative ledgers. Empty consumables disable dependent outputs; `ready`, `degraded`, and `fault` operating states alter measurement validity and confidence rather than only changing visuals.
