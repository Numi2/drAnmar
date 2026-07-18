# Contributing to Dr.Anmar

Thank you for helping improve open surgical-robotics research and education.

## Before starting

- Use an issue to describe substantial behavior or architecture changes before implementation.
- Keep all workflows simulation-only. Physical robot output, patient data, diagnosis, treatment advice,
  and claims of clinical validation are outside this project's scope.
- Do not commit anatomy downloads, demonstrations, checkpoints, logs, credentials, or machine-specific
  paths.
- Preserve ORBIT-Surgical attribution and the BSD 3-Clause notices in derived files.

## Development setup

Follow the setup in `README.md`, copy `.env.example` to `.env`, and keep mutable data under
`DR_ANMAR_ROOT`. Use synthetic objects and non-patient data only.

## Required checks

Run the checks that do not require Isaac Sim:

```bash
python3 scripts/check_public_release.py
python3 -m compileall -q scripts source
bash -n dr_anmar.sh dr_anmar_suite.sh dr_anmar_train.sh dr_anmar_workstation.sh orbitsurgical.sh
```

For simulator changes, also run the smallest relevant smoke task on a compatible NVIDIA Linux host and
include the task ID, Isaac Sim version, Isaac Lab version, GPU, and result in the pull request.

## Pull requests

Keep changes focused. Explain the doctor-facing outcome, the safety boundary, how the result was
validated, and whether compatibility or stored data changed. Screenshots are useful for interface
changes, but they do not replace functional evidence.

By contributing, you agree that your contribution is licensed under this repository's BSD 3-Clause
License.
