# Dr.Anmar capability and evidence language

Dr.Anmar describes what the product can do separately from how much evidence
exists for each implementation layer. A training workcell does not stop being
a training workcell because its physical parameters have not been correlated
to an instrumented bench.

## Evidence ladder

| Layer | What it establishes | What it does not establish |
| --- | --- | --- |
| Product capability | The asset or workcell is integrated and available for its stated simulation-training workflow | Numerical fidelity |
| Repository verification | Paths, OpenUSD/JSON structure, asset closure, task contracts, controller invariants, and tests pass | Native engine behavior |
| Native-simulator evidence | A recorded revision ran on a named Isaac/PhysX stack and hardware configuration | Real-world behavior |
| Real-world evidence | Instrumented hardware, material, sensor, or wet/dry-bench measurements support a specific correlation claim | Clinical effectiveness |
| Clinical evidence | A defined clinical study and review support a specific clinical claim | Any claim outside that study |

## Language rules

- Lead product surfaces with the available Dr.Anmar capability.
- Call automated checks **repository verification**, not real-world validation.
- Call Isaac/PhysX runs **native-simulator evidence** and record the exact
  revision, stack, hardware, scenario, and measurements.
- Use **real-world evidence not established** when no instrumented correlation
  exists. Do not turn absence of that evidence into a claim that the simulation
  workcell is unavailable.
- Reduced-order models are implementation details. They may drive training
  state and rewards, but they are not physical or clinical evidence.
- Archived smoke observations are not qualification artifacts and do not
  belong in the release catalog.
- `clinical_validation: false` remains the machine-readable clinical boundary
  until actual clinical evidence exists.

The unqualified word **validated** must never be used to blur these layers.
