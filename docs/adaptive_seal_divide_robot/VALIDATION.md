# Validation and qualification

The release gate has three layers:

1. deterministic generation, Python compilation, JSON parsing, PNG and GLB
   structural checks, archive CRC checks, checksums, and strict `usdchecker`;
2. CPU unit tests for energy maturity and faults, leak monotonicity, force
   envelopes, blade interlocks, division ordering, and phase contracts;
3. headless CUDA qualification in both standalone-tool and Franka-mounted
   representations.

The CUDA matrix must prove two cooked vessel surface deformables, two distal
fixture attachments, sixteen bridge attachments, four temporary jaw
compression attachments, four retained seal-band attachments, successful
interlocked division, release of all bridge and compression attachments, exact
joint counts, finite state for 120 simulation steps, and zero engine errors.

A `qualification_report.json` in the catalog records the exact runtime,
machine, GPU, driver, commands, and results. This is simulation qualification,
not physical calibration or clinical validation.
