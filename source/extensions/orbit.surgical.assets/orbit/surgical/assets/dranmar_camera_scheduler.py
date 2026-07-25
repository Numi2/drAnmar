# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Bounded camera scheduling for DrAnmar robot-mounted sensors.

Only one RTX camera pipeline is live at a time. Repeated acquisition from the
same camera reuses that pipeline; switching views closes the previous sensor
before opening the next. Every capture carries both simulation and monotonic
wall timestamps so downstream fusion can reject excessive skew.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Iterable, Mapping, Protocol


class CameraAdapter(Protocol):
    """Runtime adapter used by :class:`SerializedCameraScheduler`."""

    def open(self, camera_path: str) -> Any: ...

    def capture(self, sensor: Any) -> Mapping[str, Any]: ...

    def close(self, sensor: Any) -> None: ...


@dataclass(frozen=True)
class CameraFramePacket:
    camera_name: str
    camera_path: str
    sequence: int
    simulation_timestamp_s: float
    wall_timestamp_s: float
    data: Mapping[str, Any]


class SerializedCameraScheduler:
    """Serialize robot-camera acquisition with a hard one-pipeline bound."""

    def __init__(
        self,
        camera_paths: Mapping[str, str],
        adapter: CameraAdapter,
        *,
        simulation_clock: Callable[[], float],
        wall_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not camera_paths:
            raise ValueError("camera_paths must not be empty")
        normalized = {str(name): str(path) for name, path in camera_paths.items()}
        if any(not name or not path for name, path in normalized.items()):
            raise ValueError("camera names and paths must be non-empty")
        self._camera_paths = normalized
        self._adapter = adapter
        self._simulation_clock = simulation_clock
        self._wall_clock = wall_clock
        self._active_name: str | None = None
        self._active_sensor: Any = None
        self._sequence = 0
        self._closed = False
        self._switch_count = 0

    @property
    def active_camera(self) -> str | None:
        return self._active_name

    @property
    def switch_count(self) -> int:
        return self._switch_count

    @property
    def maximum_concurrent_pipelines(self) -> int:
        return 1

    def _activate(self, camera_name: str) -> Any:
        if self._closed:
            raise RuntimeError("camera scheduler is closed")
        if camera_name not in self._camera_paths:
            raise KeyError(
                f"Unknown camera {camera_name!r}; expected one of "
                f"{sorted(self._camera_paths)}"
            )
        if self._active_name == camera_name:
            return self._active_sensor
        if self._active_sensor is not None:
            self._adapter.close(self._active_sensor)
            self._active_sensor = None
            self._active_name = None
        sensor = self._adapter.open(self._camera_paths[camera_name])
        if sensor is None:
            raise RuntimeError(f"Camera adapter did not open {camera_name}")
        self._active_sensor = sensor
        self._active_name = camera_name
        self._switch_count += 1
        return sensor

    def capture(self, camera_name: str) -> CameraFramePacket:
        sensor = self._activate(camera_name)
        data = self._adapter.capture(sensor)
        if not isinstance(data, Mapping) or not data:
            raise RuntimeError(f"Camera {camera_name} returned no frame data")
        simulation_timestamp = float(self._simulation_clock())
        wall_timestamp = float(self._wall_clock())
        if (
            not math.isfinite(simulation_timestamp)
            or not math.isfinite(wall_timestamp)
        ):
            raise RuntimeError("camera timestamps must be finite")
        self._sequence += 1
        return CameraFramePacket(
            camera_name=camera_name,
            camera_path=self._camera_paths[camera_name],
            sequence=self._sequence,
            simulation_timestamp_s=simulation_timestamp,
            wall_timestamp_s=wall_timestamp,
            data=data,
        )

    def capture_cycle(
        self,
        camera_names: Iterable[str] | None = None,
        *,
        maximum_fusion_skew_s: float | None = None,
    ) -> dict[str, Any]:
        names = tuple(camera_names or self._camera_paths)
        if not names:
            raise ValueError("camera cycle must not be empty")
        if maximum_fusion_skew_s is not None:
            maximum_fusion_skew_s = float(maximum_fusion_skew_s)
            if (
                not math.isfinite(maximum_fusion_skew_s)
                or maximum_fusion_skew_s < 0.0
            ):
                raise ValueError("maximum_fusion_skew_s must be non-negative")
        packets = [self.capture(name) for name in names]
        timestamps = [packet.simulation_timestamp_s for packet in packets]
        skew = max(timestamps) - min(timestamps)
        within_gate = (
            None
            if maximum_fusion_skew_s is None
            else skew <= maximum_fusion_skew_s
        )
        return {
            "frames": {packet.camera_name: packet for packet in packets},
            "acquisition_order": [packet.camera_name for packet in packets],
            "simulation_skew_s": skew,
            "maximum_fusion_skew_s": maximum_fusion_skew_s,
            "within_fusion_skew_gate": within_gate,
            "maximum_concurrent_pipelines": 1,
            "fusion_policy": (
                "buffer_or_interpolate_to_common_timestamp_and_reject_"
                "cycles_outside_the_configured_skew_gate"
            ),
        }

    def close(self) -> None:
        if self._active_sensor is not None:
            self._adapter.close(self._active_sensor)
        self._active_sensor = None
        self._active_name = None
        self._closed = True

    def __enter__(self) -> "SerializedCameraScheduler":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


class IsaacLabCameraAdapter:
    """Lazy Isaac Lab adapter for RGB and distance-to-camera acquisition."""

    def __init__(
        self,
        simulation_context: Any,
        *,
        width: int = 160,
        height: int = 120,
        warmup_render_steps: int = 4,
    ) -> None:
        if width < 32 or height < 32:
            raise ValueError("camera dimensions must be at least 32 pixels")
        if warmup_render_steps < 1:
            raise ValueError("warmup_render_steps must be positive")
        self._sim = simulation_context
        self._width = int(width)
        self._height = int(height)
        self._warmup_render_steps = int(warmup_render_steps)

    def open(self, camera_path: str) -> Any:
        import omni.usd
        import isaaclab.sim as sim_utils
        from isaaclab.sensors.camera import Camera, CameraCfg
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        camera_prim = UsdGeom.Camera.Get(stage, camera_path)
        if not camera_prim:
            raise RuntimeError(f"USD camera is missing: {camera_path}")
        clipping = camera_prim.GetClippingRangeAttr().Get()
        sensor = Camera(
            CameraCfg(
                prim_path=camera_path,
                update_period=0.0,
                height=self._height,
                width=self._width,
                data_types=["rgb", "distance_to_camera"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=float(camera_prim.GetFocalLengthAttr().Get()),
                    focus_distance=float(
                        camera_prim.GetFocusDistanceAttr().Get() or 0.4
                    ),
                    horizontal_aperture=float(
                        camera_prim.GetHorizontalApertureAttr().Get()
                    ),
                    clipping_range=(
                        float(clipping[0]),
                        float(clipping[1]),
                    ),
                ),
            )
        )
        if not sensor.is_initialized:
            sensor._initialize_callback(None)
        for _ in range(self._warmup_render_steps):
            self._sim.step(render=True)
            sensor.update(self._sim.get_physics_dt(), force_recompute=True)
        return sensor

    def capture(self, sensor: Any) -> Mapping[str, Any]:
        self._sim.step(render=True)
        sensor.update(self._sim.get_physics_dt(), force_recompute=True)
        return {
            "rgb": sensor.data.output["rgb"][0].clone(),
            "depth": sensor.data.output["distance_to_camera"][0].clone(),
        }

    def close(self, sensor: Any) -> None:
        import gc

        try:
            sensor.__del__()
        finally:
            for attribute in ("_renderer", "_render_data"):
                if hasattr(sensor, attribute):
                    setattr(sensor, attribute, None)
            gc.collect()
