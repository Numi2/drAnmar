#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Physics-backend authority and provenance for Dr.Anmar.

The module deliberately has no Isaac, Warp, or CUDA imports.  It can therefore
describe and validate every backend from the doctor-facing stable process while
experimental runtimes are installed and exercised in isolation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "physics_next/manifest.json"
SUPPORTED_BACKENDS = {
    "physx_rigid",
    "physx_fem",
    "newton_vbd",
    "cressim_mpm",
}

# One solver-capability contract replaces room-specific readiness gates.  A
# procedure may add teaching, recording, or scoring around the simulator, but
# its physical outcome is runnable only when the active NVIDIA backend owns all
# capabilities listed here.
FIDELITY_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "nvidia_i4h_reference": ("rigid_body", "articulation", "contact"),
    "native_object_physics": ("rigid_body", "articulation", "contact"),
    "anatomy_context": ("rigid_body", "articulation"),
    "interactive_recovery_scenarios": ("rigid_body", "articulation", "contact"),
    "native_volume_deformable": ("volume_deformable", "two_way_robot_tissue"),
    "native_suturing": (
        "volume_deformable",
        "deformable_strand",
        "native_attachment",
        "strand_self_contact",
        "tissue_puncture",
    ),
    "native_suturing_drylab": (
        "deformable_strand",
        "native_attachment",
        "strand_self_contact",
    ),
    "native_topology_change": ("topology_change", "two_way_robot_tissue"),
    "native_flexible_shunt": ("flexible_body", "two_way_robot_tissue"),
    "native_fluid_structure": ("volume_deformable", "fluid_structure"),
    "native_vascular_flow": ("volume_deformable", "vascular_flow"),
    "native_hemostasis": ("fluid_dynamics", "coagulation"),
    "nvidia_medical_ultrasound": ("medical_ultrasound",),
}

CAPABILITY_LABELS = {
    "volume_deformable": "native volumetric deformable tissue",
    "two_way_robot_tissue": "two-way robot-tissue coupling",
    "deformable_strand": "native deformable thread",
    "native_attachment": "native needle/thread attachment",
    "strand_self_contact": "thread self-contact",
    "tissue_puncture": "native puncture and tract mechanics",
    "topology_change": "native topology-changing tissue",
    "flexible_body": "native flexible tube mechanics",
    "fluid_structure": "native fluid-structure coupling",
    "vascular_flow": "native vascular flow",
    "fluid_dynamics": "native fluid dynamics",
    "coagulation": "native hemostasis mechanics",
    "medical_ultrasound": "physics-based medical ultrasound",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


@dataclass(frozen=True)
class PhysicsAuthority:
    manifest_path: Path
    manifest: dict[str, Any]

    @classmethod
    def load(cls, path: Path | None = None) -> "PhysicsAuthority":
        selected = Path(
            path or os.environ.get("DR_ANMAR_PHYSICS_MANIFEST", DEFAULT_MANIFEST)
        ).expanduser().resolve()
        payload = json.loads(selected.read_text(encoding="utf-8"))
        authority = cls(selected, payload)
        authority.validate()
        return authority

    def validate(self) -> None:
        if self.manifest.get("schema") != "dr.anmar.physics-authority.v2":
            raise ValueError("Unsupported Dr.Anmar physics-authority schema")
        backends = self.manifest.get("backends")
        if not isinstance(backends, dict) or set(backends) != SUPPORTED_BACKENDS:
            raise ValueError("Physics manifest must define the complete backend set")
        if self.manifest.get("stable_default") not in SUPPORTED_BACKENDS:
            raise ValueError("stable_default must reference a known backend")
        for backend_id, backend in backends.items():
            if backend.get("id") != backend_id:
                raise ValueError(f"Backend id mismatch for {backend_id}")
            if backend.get("clinical_validation") is not False:
                raise ValueError(f"Backend {backend_id} must remain explicitly non-clinical")
            if not backend.get("authority_scope"):
                raise ValueError(f"Backend {backend_id} needs an authority_scope")
            capabilities = backend.get("capabilities")
            if not isinstance(capabilities, list) or not capabilities:
                raise ValueError(f"Backend {backend_id} needs explicit native capabilities")
        for relative_path in self.manifest.get("required_contracts", []):
            target = (self.manifest_path.parent / relative_path).resolve()
            if not target.is_file():
                raise ValueError(f"Missing physics contract: {target}")
            contract = json.loads(target.read_text(encoding="utf-8"))
            if contract.get("clinical_validation") is True:
                raise ValueError(f"Research physics contract claims clinical validation: {target}")
        routing = self.manifest.get("routing")
        if not isinstance(routing, dict) or not routing:
            raise ValueError("Physics manifest needs procedure routing")
        for procedure, candidates in routing.items():
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(f"Physics route {procedure} has no candidates")
            unknown = set(candidates) - SUPPORTED_BACKENDS
            if unknown:
                raise ValueError(f"Physics route {procedure} references unknown backends: {sorted(unknown)}")

    @property
    def requested_backend(self) -> str:
        requested = os.environ.get("DR_ANMAR_PHYSICS_BACKEND", "stable").strip()
        if requested == "stable":
            return str(self.manifest["stable_default"])
        if requested not in SUPPORTED_BACKENDS:
            return str(self.manifest["stable_default"])
        return requested

    def runtime_payload(
        self,
        *,
        native_deformable_count: int = 0,
        runtime_family: str = "isaac-sim-5.1-stable",
        effective_backend: str | None = None,
    ) -> dict[str, Any]:
        requested = self.requested_backend
        # The stable workstation is native PhysX rigid-body simulation.  It is
        # never relabelled as FEM/VBD/MPM merely because a visual organ or a
        # Python proxy happens to exist in the stage.
        effective = effective_backend or str(self.manifest["stable_default"])
        if effective not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unknown effective native backend: {effective}")
        experimental_requested = requested in {"newton_vbd", "cressim_mpm"}
        experimental_permitted = os.environ.get("DR_ANMAR_ENABLE_EXPERIMENTAL_PHYSICS", "0") == "1"
        data_root = Path(
            os.environ.get("DR_ANMAR_ROOT", Path.home() / ".local/share/dr-anmar")
        ).expanduser()
        next_root = Path(
            os.environ.get("DR_ANMAR_PHYSICS_NEXT_ROOT", data_root / "physics-next")
        ).expanduser()
        return {
            "schema": self.manifest["schema"],
            "runtime_family": runtime_family,
            "requested_backend": requested,
            "effective_backend": effective,
            "experimental_backend_requested": experimental_requested,
            "experimental_backend_permitted": experimental_permitted,
            "experimental_backend_active": effective in {"newton_vbd", "cressim_mpm"},
            "native_deformable_count": int(native_deformable_count),
            "fallback_explicit": False,
            "fallback_reason": None,
            "clinical_validation": False,
            "calibration_status": "research_defaults_unvalidated",
            "manifest_sha256": _sha256(self.manifest_path),
            "backend": self.manifest["backends"][effective],
            "next_runtime": {
                "isaac_sim": "6.0.1.0",
                "isaac_lab": "3.0.0-beta2",
                "newton": "VBD experimental integration",
                "isolated": True,
                "ready_marker": (next_root / "READY").is_file(),
                "module_probe_current_process": {
                    "isaaclab": _module_available("isaaclab"),
                    "isaaclab_newton": _module_available("isaaclab_newton"),
                    "warp": _module_available("warp"),
                },
            },
        }

    def procedure_readiness(
        self,
        procedure: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return one fail-closed native-physics decision for a procedure."""

        runtime = runtime or self.runtime_payload()
        fidelity = str(procedure.get("fidelity", ""))
        requirements = FIDELITY_REQUIREMENTS.get(fidelity)
        if requirements is None:
            requirements = ("unclassified_native_physics",)
        available = set(runtime.get("backend", {}).get("capabilities", []))
        missing = [capability for capability in requirements if capability not in available]
        ready = not missing
        backend = str(runtime.get("effective_backend", "unknown"))
        if ready:
            reason = f"Native {backend} owns this room's physical state."
        else:
            readable = ", ".join(CAPABILITY_LABELS.get(item, item.replace("_", " ")) for item in missing)
            reason = (
                f"Unavailable until the active Isaac Lab worker provides {readable}. "
                "Dr.Anmar will not substitute projected motion or proxy mechanics."
            )
        return {
            "schema": "dr.anmar.native-physics-readiness.v1",
            "ready": ready,
            "fidelity": fidelity,
            "backend": backend,
            "required_capabilities": list(requirements),
            "available_capabilities": sorted(available),
            "missing_capabilities": missing,
            "reason": reason,
        }


def load_physics_authority(path: Path | None = None) -> PhysicsAuthority:
    """Load the repository contract, failing closed on malformed configuration."""

    return PhysicsAuthority.load(path)
