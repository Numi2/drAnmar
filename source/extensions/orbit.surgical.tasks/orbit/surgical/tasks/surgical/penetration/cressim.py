# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Single-process ctypes adapter for the pinned CRESSim-MPM C API."""

from __future__ import annotations

import ctypes
import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


CRESSIM_REVISION = "09aa5009b8580351f516b6df7660e87821fc5eb6"
DEFAULT_LIBRARY = Path(
    "/home/numi/dr_anmar/physics-next/CRESSim-MPM/build-dranmar/bin/libcrmpm_c_api.so"
)


class CrVec3f3(ctypes.Structure):
    _fields_ = (("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float))


class CrVec4f(ctypes.Structure):
    _fields_ = (
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
        ("w", ctypes.c_float),
    )


class CrQuat(ctypes.Structure):
    _fields_ = (
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
        ("w", ctypes.c_float),
    )


class CrBounds3(ctypes.Structure):
    _fields_ = (("center", CrVec3f3), ("extents", CrVec3f3))


class CrParticleData(ctypes.Structure):
    _fields_ = (
        ("num_particles", ctypes.c_uint),
        ("position_mass", ctypes.c_void_p),
        ("velocity", ctypes.c_void_p),
    )


@dataclass(frozen=True)
class NeedlePose:
    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class CouplingWrench:
    force_n: tuple[float, float, float]
    torque_nm: tuple[float, float, float]


@dataclass
class _SceneHandles:
    scene: int
    tissue_geometry: int
    tissue: int
    tip_geometry: int
    tip_shape: int
    arc_geometry: int
    arc_shape: int
    particle_mass_kg: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CressimMpmAdapter:
    """Own all CRESSim scenes behind the library's non-thread-safe API."""

    _owner_lock = threading.Lock()
    _owner_exists = False

    def __init__(
        self,
        num_scenes: int,
        *,
        library_path: str | Path | None = None,
        integration_step_s: float = 0.002,
        solver_type: int = 1,
        tip_radius_m: float = 0.00026,
    ) -> None:
        if not 1 <= num_scenes <= 12:
            raise ValueError("entry policy supports 1..12 CRESSim scenes")
        with self._owner_lock:
            if self.__class__._owner_exists:
                raise RuntimeError("CRESSim is process-owned; create one adapter only")
            self.__class__._owner_exists = True
        try:
            configured = library_path or os.environ.get("DR_ANMAR_CRESSIM_LIBRARY")
            self.library_path = Path(configured or DEFAULT_LIBRARY).expanduser().resolve()
            if not self.library_path.is_file():
                raise FileNotFoundError(
                    f"pinned CRESSim shared library not found: {self.library_path}"
                )
            self.library_sha256 = _sha256(self.library_path)
            self.integration_step_s = float(integration_step_s)
            if tip_radius_m <= 0.0:
                raise ValueError("tip radius must be positive")
            self.tip_radius_m = float(tip_radius_m)
            if solver_type not in (0, 1):
                raise ValueError("entry policy supports CPU or GPU MLS-MPM only")
            self.solver_type = solver_type
            # CRESSim MLS-MPM discards grid nodes below 1e-4 numerical mass.
            # SI-scaled 1.2 mm tissue particles are below that threshold, so
            # mass and modulus are scaled together to preserve acceleration.
            # Returned coupling momentum is divided by the same factor.
            self.mass_scale = 1024.0
            self._lock = threading.RLock()
            self._lib = ctypes.CDLL(str(self.library_path))
            self._bind()
            # Roughly 12k particles per 70x45x6 mm coupon at 1.2 mm spacing.
            particle_capacity = num_scenes * 20_000
            self._lib.CrInitializeEngine(
                particle_capacity,
                num_scenes * 2,
                num_scenes * 3,
                0,
                1,
                num_scenes,
            )
            if self._lib.CrGetInitializationStatus() == 0:
                raise RuntimeError("CRESSim engine initialization failed")
            self._scenes = [self._create_scene(index) for index in range(num_scenes)]
            self._closed = False
        except Exception:
            with self._owner_lock:
                self.__class__._owner_exists = False
            raise

    @property
    def num_scenes(self) -> int:
        return len(self._scenes)

    def _bind(self) -> None:
        handle = ctypes.c_void_p
        lib = self._lib
        lib.CrInitializeEngine.argtypes = [
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.CrGetInitializationStatus.restype = ctypes.c_int
        lib.CrCreateScene.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            CrVec3f3,
            CrBounds3,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.CrCreateScene.restype = handle
        lib.CrCreateGeometry.argtypes = [ctypes.c_int, CrVec4f]
        lib.CrCreateGeometry.restype = handle
        lib.CrCreateParticleObject.argtypes = [
            ctypes.c_float,
            ctypes.c_float,
            handle,
            ctypes.POINTER(CrVec3f3),
            ctypes.POINTER(CrQuat),
            ctypes.POINTER(CrVec3f3),
            ctypes.c_int,
            ctypes.POINTER(CrVec4f),
            ctypes.POINTER(ctypes.c_uint),
        ]
        lib.CrCreateParticleObject.restype = handle
        lib.CrAddParticleObjectToScene.argtypes = [handle, handle, ctypes.POINTER(ctypes.c_uint)]
        lib.CrCreateShape.argtypes = [
            handle,
            CrVec3f3,
            CrQuat,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            CrVec3f3,
            CrVec4f,
            CrVec4f,
            CrVec4f,
        ]
        lib.CrCreateShape.restype = handle
        lib.CrAddShapeToScene.argtypes = [handle, handle]
        lib.CrSetShapePose.argtypes = [handle, ctypes.POINTER(CrVec3f3), ctypes.POINTER(CrQuat)]
        lib.CrSetShapeVelocity.argtypes = [handle, ctypes.POINTER(CrVec3f3), ctypes.POINTER(CrVec3f3)]
        lib.CrGetShapeCouplingForceTorque.argtypes = [
            handle,
            ctypes.POINTER(CrVec4f),
            ctypes.POINTER(CrVec4f),
        ]
        lib.CrGetParticleData.argtypes = [handle]
        lib.CrGetParticleData.restype = CrParticleData
        lib.CrAdvanceAll.argtypes = [ctypes.c_float]
        lib.CrFetchResultsAll.argtypes = []
        for name in ("CrReleaseScene", "CrReleaseShape", "CrReleaseGeometry", "CrReleaseParticleObject"):
            getattr(lib, name).argtypes = [handle]

    def _create_scene(self, index: int) -> _SceneHandles:
        # GPU MLS-MPM, zero gravity: the fixed coupon lane does not yet expose
        # per-particle kinematic attachment through the public C API.
        scene = self._lib.CrCreateScene(
            self.solver_type,
            20_000,
            CrVec3f3(0.0, 0.0, 0.0),
            CrBounds3(CrVec3f3(0.0, 0.0, 0.0), CrVec3f3(0.05, 0.04, 0.02)),
            0.0015,
            self.integration_step_s,
            12,
            0,
        )
        if not scene:
            raise RuntimeError(f"CRESSim scene {index} creation failed")
        tissue_geometry = self._lib.CrCreateGeometry(
            0, CrVec4f(0.035, 0.0225, 0.003, 0.0)
        )
        # Canonical v1 coupon.  The upper curriculum range is admitted only
        # after the 2 ms versus 1 ms convergence gate; 2 kPa is stable at the
        # initial MLS grid resolution and represents very soft tissue.
        youngs_modulus = 2_000.0
        poisson = 0.47
        lame_lambda = youngs_modulus * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        lame_mu = youngs_modulus / (2.0 * (1.0 + poisson))
        position = CrVec3f3(0.0, 0.0, 0.0)
        rotation = CrQuat(0.0, 0.0, 0.0, 1.0)
        inv_scale = CrVec3f3(1.0, 1.0, 1.0)
        material = CrVec4f(
            lame_lambda * self.mass_scale,
            lame_mu * self.mass_scale,
            0.0,
            0.0,
        )
        particle_count = ctypes.c_uint()
        spacing = 0.0012
        particle_mass = 1050.0 * spacing**3 * self.mass_scale
        tissue = self._lib.CrCreateParticleObject(
            spacing,
            particle_mass,
            tissue_geometry,
            ctypes.byref(position),
            ctypes.byref(rotation),
            ctypes.byref(inv_scale),
            0,
            ctypes.byref(material),
            ctypes.byref(particle_count),
        )
        if not tissue or particle_count.value == 0:
            raise RuntimeError(f"CRESSim scene {index} tissue creation failed")
        offset = ctypes.c_uint()
        self._lib.CrAddParticleObjectToScene(tissue, scene, ctypes.byref(offset))

        # A tiny analytical arc is used as the pre-puncture tip proxy.  CRESSim
        # projects particles out of closed sphere SDFs during P2G without
        # recording the equal-and-opposite momentum, which makes a sphere
        # unsuitable for a force-gated puncture.  Arc contacts deliberately
        # stay on the grid-contact path that records coupling momentum.
        tip_geometry = self._lib.CrCreateGeometry(
            8, CrVec4f(-1.5707963, 1.5707963, 0.0, 0.0)
        )
        arc_geometry = self._lib.CrCreateGeometry(8, CrVec4f(-1.5707963, 1.5707963, 0.0, 0.0))
        inactive = CrVec3f3(1000.0 + index, 1000.0, 1000.0)
        dynamic_shape = 2
        common = (
            rotation,
            dynamic_shape,
            # The MPM node grid is 1.5 mm.  The physical 0.52 mm needle body
            # remains encoded by the sphere/arc scale and SDF fattening, while
            # this nodal contact band must span one grid cell to avoid the tool
            # passing between nodes without producing a coupling impulse.
            0.0015,
            0.00026,
            1.0,
            0.30,
            CrVec4f(0.0, 0.0, 0.0, 0.0),
            CrVec4f(0.0, 0.0, 0.0, 0.0),
            CrVec4f(0.0, 0.0, 0.0, 0.0),
        )
        tip_shape = self._lib.CrCreateShape(
            tip_geometry, inactive, common[0], common[1], common[2], common[3],
            common[4], common[5],
            CrVec3f3(
                1.0 / self.tip_radius_m,
                1.0 / self.tip_radius_m,
                1.0 / self.tip_radius_m,
            ),
            common[6], common[7], common[8]
        )
        radius = 0.0070028175
        arc_shape = self._lib.CrCreateShape(
            arc_geometry, inactive, common[0], common[1], common[2], common[3],
            common[4], common[5], CrVec3f3(1.0 / radius, 1.0 / radius, 1.0 / radius),
            common[6], common[7], common[8]
        )
        self._lib.CrAddShapeToScene(tip_shape, scene)
        self._lib.CrAddShapeToScene(arc_shape, scene)
        return _SceneHandles(
            scene=scene,
            tissue_geometry=tissue_geometry,
            tissue=tissue,
            tip_geometry=tip_geometry,
            tip_shape=tip_shape,
            arc_geometry=arc_geometry,
            arc_shape=arc_shape,
            particle_mass_kg=1050.0 * spacing**3,
        )

    def _tissue_momentum(
        self, handles: _SceneHandles
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return physical linear and angular momentum about the world origin."""
        data = self._lib.CrGetParticleData(handles.scene)
        count = int(data.num_particles)
        position_buffer = (ctypes.c_float * (count * 4)).from_address(data.position_mass)
        velocity_buffer = (ctypes.c_float * (count * 3)).from_address(data.velocity)
        positions = np.ctypeslib.as_array(position_buffer).reshape(count, 4)[:, :3]
        velocities = np.ctypeslib.as_array(velocity_buffer).reshape(count, 3)
        particle_momentum = velocities * handles.particle_mass_kg
        return particle_momentum.sum(axis=0), np.cross(positions, particle_momentum).sum(axis=0)

    @staticmethod
    def _set_pose(lib: ctypes.CDLL, shape: int, pose: NeedlePose) -> None:
        position = CrVec3f3(*pose.position)
        rotation = CrQuat(*pose.quaternion_xyzw)
        linear = CrVec3f3(*pose.linear_velocity)
        angular = CrVec3f3(*pose.angular_velocity)
        lib.CrSetShapePose(shape, ctypes.byref(position), ctypes.byref(rotation))
        lib.CrSetShapeVelocity(shape, ctypes.byref(linear), ctypes.byref(angular))

    def step(
        self,
        tip_poses: Sequence[NeedlePose],
        arc_poses: Sequence[NeedlePose],
        punctured: Sequence[bool],
        *,
        dt_s: float = 0.02,
    ) -> tuple[CouplingWrench, ...]:
        if len(tip_poses) != self.num_scenes or len(arc_poses) != self.num_scenes:
            raise ValueError("pose count must match CRESSim scene count")
        if len(punctured) != self.num_scenes:
            raise ValueError("puncture-state count must match CRESSim scene count")
        if dt_s <= 0.0:
            raise ValueError("CRESSim advance duration must be positive")
        inactive_pose = NeedlePose((1000.0, 1000.0, 1000.0), (0.0, 0.0, 0.0, 1.0))
        with self._lock:
            momentum_before = [self._tissue_momentum(handles) for handles in self._scenes]
            for handles, tip_pose, arc_pose, is_punctured in zip(
                self._scenes, tip_poses, arc_poses, punctured, strict=True
            ):
                self._set_pose(self._lib, handles.tip_shape, inactive_pose if is_punctured else tip_pose)
                self._set_pose(self._lib, handles.arc_shape, arc_pose if is_punctured else inactive_pose)
            self._lib.CrAdvanceAll(float(dt_s))
            self._lib.CrFetchResultsAll()
            result: list[CouplingWrench] = []
            for scene_index, (handles, is_punctured) in enumerate(
                zip(self._scenes, punctured, strict=True)
            ):
                force = CrVec4f()
                torque = CrVec4f()
                active_shape = handles.arc_shape if is_punctured else handles.tip_shape
                self._lib.CrGetShapeCouplingForceTorque(
                    active_shape, ctypes.byref(force), ctypes.byref(torque)
                )
                # This pinned CRESSim revision records zero dynamic-shape
                # coupling for the kinematic contact model even though the MPM
                # particles receive momentum.  With zero gravity and no other
                # external loads, physical momentum balance is the raw contact
                # wrench and preserves CRESSim as the interaction authority.
                linear_before, angular_before = momentum_before[scene_index]
                linear_after, angular_after = self._tissue_momentum(handles)
                tool_force = -(linear_after - linear_before) / float(dt_s)
                tool_torque_world = -(angular_after - angular_before) / float(dt_s)
                active_pose = arc_poses[scene_index] if is_punctured else tip_poses[scene_index]
                tool_origin = np.asarray(active_pose.position, dtype=np.float64)
                tool_torque = tool_torque_world - np.cross(tool_origin, tool_force)
                result.append(
                    CouplingWrench(
                        tuple(float(value) for value in tool_force),
                        tuple(float(value) for value in tool_torque),
                    )
                )
            return tuple(result)

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        with self._lock:
            for handles in reversed(self._scenes):
                self._lib.CrReleaseShape(handles.arc_shape)
                self._lib.CrReleaseShape(handles.tip_shape)
                self._lib.CrReleaseParticleObject(handles.tissue)
                self._lib.CrReleaseGeometry(handles.arc_geometry)
                self._lib.CrReleaseGeometry(handles.tip_geometry)
                self._lib.CrReleaseGeometry(handles.tissue_geometry)
                self._lib.CrReleaseScene(handles.scene)
            self._lib.CrReleaseEngine()
            self._closed = True
        with self._owner_lock:
            self.__class__._owner_exists = False

    def __enter__(self) -> "CressimMpmAdapter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
