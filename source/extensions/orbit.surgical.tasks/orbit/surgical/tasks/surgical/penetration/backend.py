# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""DrAnmar-owned tissue-entry backend interface.

Policies and MDP terms depend on this interface, never on CRESSim symbols.
The pinned CRESSim provider is the first qualified numerical implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from .cressim import CouplingWrench, CressimMpmAdapter, NeedlePose


@dataclass(frozen=True)
class TissueEntryBackendMetadata:
    provider: str
    revision: str
    library_sha256: str
    integration_step_s: float
    clinical_validation: bool = False


class DrAnmarTissueEntryBackend(Protocol):
    """Stable ABI between DrAnmar policy code and tissue mechanics."""

    @property
    def num_scenes(self) -> int: ...

    @property
    def metadata(self) -> TissueEntryBackendMetadata: ...

    def step(
        self,
        tip_poses: Sequence[NeedlePose],
        arc_poses: Sequence[NeedlePose],
        punctured: Sequence[bool],
        *,
        dt_s: float = 0.02,
    ) -> tuple[CouplingWrench, ...]: ...

    def close(self) -> None: ...


class DrAnmarCressimTissueEntryBackend(CressimMpmAdapter):
    """Private pinned MPM provider for the public DrAnmar backend ABI."""

    @property
    def metadata(self) -> TissueEntryBackendMetadata:
        from .cressim import CRESSIM_REVISION

        return TissueEntryBackendMetadata(
            provider="cressim_mpm",
            revision=CRESSIM_REVISION,
            library_sha256=self.library_sha256,
            integration_step_s=self.integration_step_s,
        )


def create_tissue_entry_backend(
    num_scenes: int,
    *,
    integration_step_s: float = 0.002,
    library_path: str | Path | None = None,
) -> DrAnmarTissueEntryBackend:
    """Create the locked backend or fail before the first policy step."""

    return DrAnmarCressimTissueEntryBackend(
        num_scenes,
        integration_step_s=integration_step_s,
        library_path=library_path,
    )
