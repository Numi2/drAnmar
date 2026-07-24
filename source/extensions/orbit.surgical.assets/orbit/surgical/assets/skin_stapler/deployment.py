"""Logical magazine and standalone-staple deployment helpers.

These utilities support simulation task development only. They do not model
staple forming, tissue penetration, closure strength, wound healing, sterility,
or safe use on patients.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

from .control import FIRE_THRESHOLD_DEG, REARM_THRESHOLD_DEG
from .paths import staple_usd
from .semantics import apply_semantic_labels
from .state import StaplerState


@dataclass
class StapleMagazine:
    """Discrete simulated magazine state.

    The magazine count is task state. It is intentionally separate from the USD
    ``state`` variant because a deployed-staple workflow may update the count at
    runtime while the integrating application decides when to respawn or switch
    the source asset to the ``empty`` variant.
    """

    capacity: int = 35
    remaining: int = 35

    def __post_init__(self) -> None:
        self.capacity = int(self.capacity)
        self.remaining = int(self.remaining)
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if not 0 <= self.remaining <= self.capacity:
            raise ValueError("remaining must be between zero and capacity")

    @property
    def state(self) -> StaplerState:
        return StaplerState.LOADED if self.remaining > 0 else StaplerState.EMPTY

    @property
    def deployed(self) -> int:
        return self.capacity - self.remaining

    def deploy_one(self) -> int:
        """Consume one simulated staple and return the remaining count."""

        if self.remaining <= 0:
            raise RuntimeError("Cannot deploy from an empty simulated magazine")
        self.remaining -= 1
        return self.remaining

    def reset(self, *, remaining: int | None = None) -> None:
        """Reset the magazine to full capacity or an explicit remaining count."""

        next_remaining = self.capacity if remaining is None else int(remaining)
        if not 0 <= next_remaining <= self.capacity:
            raise ValueError("remaining must be between zero and capacity")
        self.remaining = next_remaining


@dataclass(frozen=True)
class StapleDeploymentEvent:
    """A discrete simulated deployment request emitted on a trigger edge."""

    sequence_index: int
    remaining: int
    state: StaplerState
    trigger_position_rad: float

    @property
    def trigger_position_deg(self) -> float:
        return math.degrees(self.trigger_position_rad)

    @property
    def deployed_index(self) -> int:
        """Compatibility alias for deterministic deployed-staple numbering."""

        return self.sequence_index


# Backward-compatible short name used by early task scripts.
DeploymentEvent = StapleDeploymentEvent


@dataclass
class TriggerEdgeDeploymentController:
    """Emit one deployment event per complete trigger press.

    ``update`` accepts Isaac articulation joint positions in radians. Thresholds
    remain human-readable in degrees and are converted internally. The
    controller is deliberately separate from the USD articulation: it makes
    rigid staple spawning explicit and deterministic while mechanism and tissue
    interaction parameters remain provisional.
    """

    magazine: StapleMagazine
    fire_threshold_deg: float = FIRE_THRESHOLD_DEG
    rearm_threshold_deg: float = REARM_THRESHOLD_DEG
    _armed: bool = True
    _sequence: int = 0

    def __post_init__(self) -> None:
        self.fire_threshold_deg = float(self.fire_threshold_deg)
        self.rearm_threshold_deg = float(self.rearm_threshold_deg)
        if not math.isfinite(self.fire_threshold_deg) or not math.isfinite(self.rearm_threshold_deg):
            raise ValueError("trigger thresholds must be finite")
        if not 0.0 <= self.rearm_threshold_deg < self.fire_threshold_deg:
            raise ValueError("rearm threshold must be non-negative and below fire threshold")

    @classmethod
    def from_degrees(
        cls,
        magazine: StapleMagazine,
        *,
        fire_threshold_deg: float = FIRE_THRESHOLD_DEG,
        rearm_threshold_deg: float = REARM_THRESHOLD_DEG,
    ) -> "TriggerEdgeDeploymentController":
        """Explicit degree-based constructor for human-authored task configs."""

        return cls(
            magazine=magazine,
            fire_threshold_deg=fire_threshold_deg,
            rearm_threshold_deg=rearm_threshold_deg,
        )

    @property
    def fire_threshold_rad(self) -> float:
        return math.radians(self.fire_threshold_deg)

    @property
    def rearm_threshold_rad(self) -> float:
        return math.radians(self.rearm_threshold_deg)

    @property
    def next_sequence_index(self) -> int:
        return self._sequence

    @property
    def armed(self) -> bool:
        return self._armed

    def reset(self, *, reset_magazine: bool = False, remaining: int | None = None) -> None:
        """Rearm the controller and restart deterministic event numbering.

        Set ``reset_magazine=True`` to reset the associated magazine as well.
        When ``remaining`` is supplied, the magazine is reset even if
        ``reset_magazine`` is false.
        """

        self._armed = True
        self._sequence = 0
        if reset_magazine or remaining is not None:
            self.magazine.reset(remaining=remaining)

    def update(self, trigger_position_rad: float) -> StapleDeploymentEvent | None:
        """Update from an Isaac articulation joint position in radians."""

        position = float(trigger_position_rad)
        if not math.isfinite(position):
            raise ValueError("trigger_position_rad must be finite")

        if position <= self.rearm_threshold_rad:
            self._armed = True
            return None

        if not self._armed or position < self.fire_threshold_rad:
            return None

        # An empty magazine consumes the press edge but does not emit an event.
        if self.magazine.remaining <= 0:
            self._armed = False
            return None

        self._armed = False
        remaining = self.magazine.deploy_one()
        event = StapleDeploymentEvent(
            sequence_index=self._sequence,
            remaining=remaining,
            state=self.magazine.state,
            trigger_position_rad=position,
        )
        self._sequence += 1
        return event

    def update_degrees(self, trigger_position_deg: float) -> StapleDeploymentEvent | None:
        """Human-facing degree wrapper around :meth:`update`."""

        position_deg = float(trigger_position_deg)
        if not math.isfinite(position_deg):
            raise ValueError("trigger_position_deg must be finite")
        return self.update(math.radians(position_deg))

    # Explicit alias for code that names the Isaac input unit at the call site.
    update_radians = update


def deployed_staple_path(base_path: str, index: int) -> str:
    """Return a deterministic prim path for a deployed staple instance."""

    index = int(index)
    if index < 0:
        raise ValueError("index must be non-negative")
    base = str(base_path).rstrip("/")
    if not base:
        raise ValueError("base_path must not be empty")
    return f"{base}/SkinStaple_{index:03d}"


def _validated_vec(values: Sequence[float], length: int, name: str) -> tuple[float, ...]:
    if len(values) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def add_staple_reference(
    stage: Any,
    prim_path: str,
    *,
    usd_path: str | Path | None = None,
    translation_m: Sequence[float] = (0.0, 0.0, 0.0),
    orientation_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
) -> Any:
    """Add and pose a standalone staple reference on an existing USD stage.

    Pose opinions are authored before an Isaac rigid-body view is expected to
    discover the prim. ``orientation_wxyz`` follows OpenUSD quaternion order.
    """

    if stage is None:
        raise ValueError("stage must not be None")
    if not str(prim_path).startswith("/"):
        raise ValueError("prim_path must be an absolute USD path")

    translation = _validated_vec(translation_m, 3, "translation_m")
    orientation = _validated_vec(orientation_wxyz, 4, "orientation_wxyz")
    norm = math.sqrt(sum(component * component for component in orientation))
    if norm <= 1e-12:
        raise ValueError("orientation_wxyz must have non-zero length")
    orientation = tuple(component / norm for component in orientation)

    try:
        from pxr import Gf, UsdGeom  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("add_staple_reference requires the target OpenUSD runtime") from exc

    source = Path(usd_path) if usd_path is not None else staple_usd()
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    prim = stage.DefinePrim(str(prim_path), "Xform")
    prim.GetReferences().AddReference(str(source))

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*translation))
    w, x, y, z = orientation
    xformable.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    return prim


def spawn_staple_at_exit(
    stage: Any,
    stapler_prim_path: str,
    output_prim_path: str,
    *,
    usd_path: str | Path | None = None,
    exit_frame_relative_path: str = "Links/Housing/Frames/staple_exit",
    forward_offset_m: float = 0.00075,
    apply_semantics: bool = True,
) -> Any:
    """Reference a formed staple at the stapler exit frame's world pose.

    Call this after receiving a :class:`StapleDeploymentEvent`. The output is an
    ordinary dynamic rigid USD reference. The integrating task may then assign
    velocity, capture it with a gripper, or connect a separate tissue model.
    """

    offset_m = float(forward_offset_m)
    if not math.isfinite(offset_m):
        raise ValueError("forward_offset_m must be finite")

    try:
        from pxr import Gf, Usd, UsdGeom  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("spawn_staple_at_exit requires the target OpenUSD runtime") from exc

    frame_path = f"{str(stapler_prim_path).rstrip('/')}/{str(exit_frame_relative_path).strip('/')}"
    frame_prim = stage.GetPrimAtPath(frame_path)
    if not frame_prim or not frame_prim.IsValid():
        raise ValueError(f"Staple exit frame not found: {frame_path}")

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_matrix = cache.GetLocalToWorldTransform(frame_prim)
    world_position = world_matrix.Transform(Gf.Vec3d(offset_m, 0.0, 0.0))
    rotation = world_matrix.ExtractRotationQuat()
    imaginary = rotation.GetImaginary()
    orientation_wxyz = (
        float(rotation.GetReal()),
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
    )

    prim = add_staple_reference(
        stage,
        output_prim_path,
        usd_path=usd_path,
        translation_m=(float(world_position[0]), float(world_position[1]), float(world_position[2])),
        orientation_wxyz=orientation_wxyz,
    )
    if apply_semantics:
        apply_semantic_labels(prim, ("skin_staple", "closure", "disposal"), instance_name="class")
    return prim


def spawn_formed_staple_at_placement(
    stage: Any,
    stapler_prim_path: str,
    output_prim_path: str,
    *,
    usd_path: str | Path | None = None,
    apply_semantics: bool = True,
) -> Any:
    """Spawn the final formed-staple proxy at the authored closure reference.

    This bypasses the transient pre-form path and is the preferred helper for
    closure-task scoring. It remains a rigid research proxy and does not model
    tissue penetration or staple-forming mechanics.
    """

    return spawn_staple_at_exit(
        stage,
        stapler_prim_path,
        output_prim_path,
        usd_path=usd_path,
        exit_frame_relative_path="Links/Housing/Frames/formed_staple_reference",
        forward_offset_m=0.0,
        apply_semantics=apply_semantics,
    )
