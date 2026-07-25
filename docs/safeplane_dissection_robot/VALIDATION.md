# Validation and qualification

The release gate combines deterministic generation, strict OpenUSD parsing,
hashed manifests, archive CRC/checksum validation, CPU controller tests, and
headless CUDA qualification in standalone and Franka-mounted representations.

The CUDA matrix must prove 17 tool joints (24 with the Franka arm), two cooked
surface deformables, two target-bed fixture attachments, eight traction
attachments, 56 bridge-anchor attachments, six protected-structure
attachments, all 28 continuity joints released through guarded modalities,
protected vessel/nerve/duct continuity retained, conserved PBD fluid emission
and suction capture, finite articulation state for 120 steps, and zero engine
errors.

This is simulation qualification. It is not evidence that an anatomical plane
is clinically safe, nor is it physical calibration of traction, pressure,
energy delivery, thermal spread, cutting, or protected-structure avoidance.
