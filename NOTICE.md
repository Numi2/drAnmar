# Notices and provenance

Dr.Anmar is a derivative work based on
[ORBIT-Surgical](https://github.com/orbit-surgical/orbit-surgical), originally copyright 2024 The
ORBIT-Surgical Project Developers and distributed under the BSD 3-Clause License. The imported baseline
for this snapshot is upstream commit `6e47534` (`fix visualization markers`). The upstream paper and
repository should be cited in research that uses this work.

Dr.Anmar additions include the Doctor Studio browser interface, curriculum and catalog, portable service
launchers, anatomy-scene integration, demonstration workflow, and compatibility changes for Isaac Sim
5.1 / Isaac Lab 2.3.2. These additions are copyright 2026 Dr.Anmar Project Developers.

The repository includes source and the robot/prop assets already present in the ORBIT-Surgical source
tree. It does not include NVIDIA Isaac Sim, Isaac Lab, model checkpoints, user demonstrations, or the
optional multi-gigabyte anatomy archives. Those components and downloaded assets retain their own
licenses, notices, and usage terms. Review them before use or redistribution.

The optional orthopedic-ultrasound provider installs the MIT-licensed
[SonoGym](https://github.com/SonoGym/SonoGym) source at pinned commit
`e67be58334d1a5274f0913af36f56e4b0b7ffe5a`. Its public CT-derived patient assets and ultrasound models are
downloaded separately from the upstream Hugging Face dataset under CC BY 4.0 and are not redistributed by
Dr.Anmar. Preserve upstream attribution and review the dataset terms before redistribution or downstream use.

The multimodal catalog references, but does not redistribute, Apache-2.0
[Cosmos-H-Dreams](https://github.com/isaac-for-healthcare/Cosmos-H-Dreams)
example assets at a pinned Git revision. Referenced checkpoint and embedding
artifacts remain external under the NVIDIA Open Model License. Their full
content hashes, revisions, and loading restrictions are recorded in the
Dr.Anmar asset bundle.

The names ORBIT-Surgical, SonoGym, Cosmos, NVIDIA, Isaac Sim, Isaac Lab, dVRK,
and other marks belong to their respective owners. Their appearance does not
imply endorsement of Dr.Anmar.
