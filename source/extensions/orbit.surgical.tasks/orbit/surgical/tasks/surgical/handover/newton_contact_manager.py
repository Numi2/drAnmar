# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded T1 contact routing for the coupled MJWarp/VBD Newton manager.

Newton's default soft-contact candidate set contains every world-compatible
particle/shape pair.  That is intentionally general, but it is not viable for
thousands of replicated surgical environments.  This manager installs a
versioned, fail-closed pair list containing only boundary particles and the
needle, distal PSM tools, and table.  Contact storage is independently bounded.

The receipt is vertex-contact telemetry.  Generic needle contact is never
called puncture.  A separate tip-local particle receipt lets the task require
contact near the sampled entry region before recording a transition.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import warp as wp
from isaaclab.utils.configclass import configclass
from isaaclab_contrib.deformable.coupled_mjwarp_vbd_manager import (
    NewtonCoupledMJWarpVBDManager,
)
from isaaclab_contrib.deformable.newton_manager_cfg import (
    CoupledMJWarpVBDSolverCfg,
)
from isaaclab_newton.physics.newton_manager import NewtonManager
from newton import CollisionPipeline, Contacts, Model, ShapeFlags, State
from newton.solvers import SolverBase, SolverMuJoCo, SolverVBD


_CONTACT_CLASSES = 4
NEEDLE_CONTACT_CLASS = 0
ROBOT_1_CONTACT_CLASS = 1
ROBOT_2_CONTACT_CLASS = 2
OTHER_CONTACT_CLASS = 3

_SOFT_PAIR_BYTES = 8
_SOFT_REPLAY_BYTES = 4
_SOFT_CONTACT_AND_VBD_BYTES = 160
_RIGID_SENSOR_CONTACT_BYTES = 256


@configclass
class DrAnmarCoupledMJWarpVBDSolverCfg(CoupledMJWarpVBDSolverCfg):
    """T1 solver limits carried with the normal Isaac Lab solver config."""

    class_type: type[NewtonManager] | str = (
        "orbit.surgical.tasks.surgical.handover.newton_contact_manager:"
        "DrAnmarCoupledMJWarpVBDManager"
    )
    maximum_environment_count: int = 2400
    soft_contacts_per_environment: int = 256
    rigid_sensor_contacts_per_environment: int = 64
    maximum_soft_candidate_pairs: int = 15_000_000
    maximum_contact_pipeline_memory_bytes: int = 536_870_912
    expected_surface_particles_per_environment: int = 400
    expected_soft_shapes_per_environment: int = 14
    soft_contact_margin_m: float = 0.002
    needle_tip_offset_m: tuple[float, float, float] = (
        0.02004,
        -0.019154,
        0.0,
    )
    needle_tip_contact_radius_m: float = 0.003
    soft_shape_label_fragments: tuple[str, ...] = (
        "/object",
        "needle",
        "psm_tool_",
        "/table",
    )


def estimate_contact_pipeline_memory_bytes(
    *,
    soft_candidate_pairs: int,
    soft_contact_capacity: int,
    rigid_sensor_contact_capacity: int,
) -> int:
    """Return a conservative contact-only allocation estimate."""

    return (
        int(soft_candidate_pairs) * (_SOFT_PAIR_BYTES + _SOFT_REPLAY_BYTES)
        + int(soft_contact_capacity) * _SOFT_CONTACT_AND_VBD_BYTES
        + int(rigid_sensor_contact_capacity) * _RIGID_SENSOR_CONTACT_BYTES
    )


class _DrAnmarBoundedSolverVBD(SolverVBD):
    """Stop upstream VBD from allocating shape_count * particle_count state."""

    def __init__(
        self,
        model: Model,
        *,
        dranmar_soft_contact_max: int,
        **kwargs: Any,
    ):
        self._dranmar_soft_contact_max = int(dranmar_soft_contact_max)
        if self._dranmar_soft_contact_max <= 0:
            raise ValueError("T1 VBD soft-contact capacity must be positive")
        self._dranmar_initializing_contact_state = True
        super().__init__(model, **kwargs)
        self._dranmar_initializing_contact_state = False
        retained_capacities = {
            int(self.body_particle_contact_penalty_k.shape[0]),
            int(self.body_particle_contact_material_ke.shape[0]),
            int(self.body_particle_contact_material_kd.shape[0]),
            int(self.body_particle_contact_material_mu.shape[0]),
        }
        if retained_capacities != {self._dranmar_soft_contact_max}:
            raise RuntimeError("T1 VBD did not retain its bounded soft-contact state")

    def _init_body_particle_contact_state(
        self,
        soft_contact_max: int,
    ) -> None:
        requested = int(soft_contact_max)
        if self._dranmar_initializing_contact_state:
            requested = self._dranmar_soft_contact_max
        elif requested > self._dranmar_soft_contact_max:
            raise RuntimeError(
                "T1 VBD soft-contact state requested capacity "
                f"{requested}, above its bounded limit "
                f"{self._dranmar_soft_contact_max}"
            )
        else:
            # Do not allow an internal refresh to shrink the arrays that must
            # remain capture-safe for the qualified contact capacity.
            requested = self._dranmar_soft_contact_max
        super()._init_body_particle_contact_state(requested)


@wp.kernel
def _advance_contact_receipt_generation(
    receipt_generation: wp.array(dtype=wp.int32),
):
    wp.atomic_add(receipt_generation, 0, 1)


@wp.kernel
def _reset_environment_contact_receipt(
    receipt_generation: wp.array(dtype=wp.int32),
    environment_generation: wp.array(dtype=wp.int32),
    candidate_seen: wp.array(dtype=wp.int32),
    penetration_seen: wp.array(dtype=wp.int32),
    maximum_penetration: wp.array(dtype=wp.float32),
):
    receipt_index = wp.tid()
    environment_index = receipt_index // _CONTACT_CLASSES
    contact_class = receipt_index - environment_index * _CONTACT_CLASSES
    if contact_class == 0:
        environment_generation[environment_index] = receipt_generation[0]
    candidate_seen[receipt_index] = 0
    penetration_seen[receipt_index] = 0
    maximum_penetration[receipt_index] = 0.0


@wp.kernel
def _reset_tip_particle_contact_receipt(
    tip_particle_penetration_seen: wp.array(dtype=wp.int32),
    tip_particle_maximum_penetration: wp.array(dtype=wp.float32),
):
    particle_index = wp.tid()
    tip_particle_penetration_seen[particle_index] = 0
    tip_particle_maximum_penetration[particle_index] = 0.0


@wp.kernel
def _invalidate_environment_contact_receipt(
    environment_indices: wp.array(dtype=wp.int64),
    environment_generation: wp.array(dtype=wp.int32),
    candidate_seen: wp.array(dtype=wp.int32),
    penetration_seen: wp.array(dtype=wp.int32),
    maximum_penetration: wp.array(dtype=wp.float32),
):
    item_index = wp.tid()
    environment_index = environment_indices[item_index]
    environment_generation[environment_index] = -1
    for contact_class in range(_CONTACT_CLASSES):
        receipt_index = environment_index * _CONTACT_CLASSES + contact_class
        candidate_seen[receipt_index] = 0
        penetration_seen[receipt_index] = 0
        maximum_penetration[receipt_index] = 0.0


@wp.kernel
def _invalidate_masked_contact_receipt(
    world_mask: wp.array(dtype=wp.bool),
    environment_generation: wp.array(dtype=wp.int32),
    candidate_seen: wp.array(dtype=wp.int32),
    penetration_seen: wp.array(dtype=wp.int32),
    maximum_penetration: wp.array(dtype=wp.float32),
):
    environment_index = wp.tid()
    if not world_mask[environment_index]:
        return
    environment_generation[environment_index] = -1
    for contact_class in range(_CONTACT_CLASSES):
        receipt_index = environment_index * _CONTACT_CLASSES + contact_class
        candidate_seen[receipt_index] = 0
        penetration_seen[receipt_index] = 0
        maximum_penetration[receipt_index] = 0.0


@wp.kernel
def _reset_overflow_receipt(
    overflow_seen: wp.array(dtype=wp.int32),
):
    overflow_seen[0] = 0


@wp.kernel
def _accumulate_soft_contact_receipt(
    contact_count: wp.array(dtype=wp.int32),
    contact_capacity: int,
    contact_particle: wp.array(dtype=wp.int32),
    contact_shape: wp.array(dtype=wp.int32),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=wp.float32),
    particle_world: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=wp.int32),
    shape_margin: wp.array(dtype=wp.float32),
    shape_class: wp.array(dtype=wp.int32),
    environment_count: int,
    needle_tip_offset: wp.vec3,
    needle_tip_contact_radius: float,
    candidate_seen: wp.array(dtype=wp.int32),
    penetration_seen: wp.array(dtype=wp.int32),
    maximum_penetration: wp.array(dtype=wp.float32),
    tip_particle_penetration_seen: wp.array(dtype=wp.int32),
    tip_particle_maximum_penetration: wp.array(dtype=wp.float32),
    overflow_seen: wp.array(dtype=wp.int32),
):
    """Accumulate actual penetration, not Newton's broad soft candidates."""

    contact_index = wp.tid()
    if contact_index == 0 and contact_count[0] > contact_capacity:
        overflow_seen[0] = 1
    if contact_index >= contact_count[0]:
        return
    particle_index = contact_particle[contact_index]
    shape_index = contact_shape[contact_index]
    if particle_index < 0 or shape_index < 0:
        return
    environment_index = particle_world[particle_index]
    if environment_index < 0 or environment_index >= environment_count:
        return
    contact_class = shape_class[shape_index]
    receipt_index = environment_index * _CONTACT_CLASSES + contact_class
    wp.atomic_max(candidate_seen, receipt_index, 1)

    body_index = shape_body[shape_index]
    body_surface_position = contact_body_pos[contact_index]
    world_surface_position = body_surface_position
    if body_index >= 0:
        world_surface_position = wp.transform_point(
            body_q[body_index],
            body_surface_position,
        )
    signed_distance = wp.dot(
        contact_normal[contact_index],
        particle_q[particle_index] - world_surface_position,
    )
    rigid_margin = 0.0
    if shape_margin.shape[0] > 0:
        rigid_margin = shape_margin[shape_index]
    penetration = -(signed_distance - particle_radius[particle_index] - rigid_margin)
    if penetration <= 0.0:
        return
    wp.atomic_max(penetration_seen, receipt_index, 1)
    wp.atomic_max(
        maximum_penetration,
        receipt_index,
        penetration,
    )
    if (
        contact_class == NEEDLE_CONTACT_CLASS
        and body_index >= 0
        and wp.length(body_surface_position - needle_tip_offset) <= needle_tip_contact_radius
    ):
        wp.atomic_max(
            tip_particle_penetration_seen,
            particle_index,
            1,
        )
        wp.atomic_max(
            tip_particle_maximum_penetration,
            particle_index,
            penetration,
        )


def _shape_contact_class(label: str) -> int:
    normalized = label.casefold()
    if "/object" in normalized or "needle" in normalized:
        return NEEDLE_CONTACT_CLASS
    if "robot_1" in normalized or "robot1" in normalized:
        return ROBOT_1_CONTACT_CLASS
    if "robot_2" in normalized or "robot2" in normalized:
        return ROBOT_2_CONTACT_CLASS
    return OTHER_CONTACT_CLASS


def _approved_soft_shape_indices(
    model: Model,
    fragments: tuple[str, ...],
) -> np.ndarray:
    normalized_fragments = tuple(fragment.casefold() for fragment in fragments if fragment)
    if not normalized_fragments:
        raise RuntimeError("T1 soft-contact shape allow-list is empty")
    collide_particles = (
        np.asarray(model.shape_flags.numpy(), dtype=np.int32) & int(ShapeFlags.COLLIDE_PARTICLES)
    ) != 0
    approved = np.asarray(
        [
            index
            for index, label in enumerate(model.shape_label)
            if (
                bool(collide_particles[index])
                and any(fragment in str(label).casefold() for fragment in normalized_fragments)
            )
        ],
        dtype=np.int32,
    )
    if approved.size == 0:
        raise RuntimeError("T1 soft-contact shape allow-list matched no Newton shapes")
    approved_labels = [str(model.shape_label[int(index)]).casefold() for index in approved]
    required_labels = {
        "needle": any("/object" in label or "needle" in label for label in approved_labels),
        "robot_1_tool": any(
            "psm_tool_" in label and ("robot_1" in label or "robot1" in label)
            for label in approved_labels
        ),
        "robot_2_tool": any(
            "psm_tool_" in label and ("robot_2" in label or "robot2" in label)
            for label in approved_labels
        ),
        "table": any("/table" in label for label in approved_labels),
    }
    missing_labels = [name for name, available in required_labels.items() if not available]
    if missing_labels:
        raise RuntimeError(
            f"T1 soft-contact allow-list did not resolve required shapes: {missing_labels}"
        )
    return approved


def _surface_particle_indices(model: Model) -> np.ndarray:
    if model.tri_indices is None or int(model.tri_count) <= 0:
        raise RuntimeError("T1 requires volume-boundary triangles for bounded soft contact")
    surface = np.unique(np.asarray(model.tri_indices.numpy(), dtype=np.int32).reshape(-1))
    if surface.size == 0:
        raise RuntimeError("T1 resolved zero boundary particles")
    return surface


def _build_bounded_soft_pairs(
    model: Model,
    solver_cfg: DrAnmarCoupledMJWarpVBDSolverCfg,
) -> tuple[np.ndarray, dict[str, int]]:
    world_count = int(model.world_count)
    if world_count <= 0:
        raise RuntimeError("T1 requires at least one Newton world")
    if world_count > int(solver_cfg.maximum_environment_count):
        raise RuntimeError(
            "T1 requested "
            f"{world_count} environments, above the qualified limit "
            f"{solver_cfg.maximum_environment_count}"
        )

    surface = _surface_particle_indices(model)
    particle_world = np.asarray(
        model.particle_world.numpy(),
        dtype=np.int32,
    )
    shape_world = np.asarray(model.shape_world.numpy(), dtype=np.int32)
    surface_world = particle_world[surface]
    if np.any(surface_world < 0):
        raise RuntimeError("T1 does not permit unassigned/global deformable particles")
    surface_counts = np.bincount(
        surface_world,
        minlength=world_count,
    )
    expected_surface = int(solver_cfg.expected_surface_particles_per_environment)
    if np.any(surface_counts != expected_surface):
        received = sorted(set(map(int, surface_counts.tolist())))
        raise RuntimeError(
            "T1 boundary-particle count drift: "
            f"expected {expected_surface} per environment, got {received}"
        )
    expected_grouped_surface_world = np.repeat(
        np.arange(world_count, dtype=np.int32),
        expected_surface,
    )
    if not np.array_equal(
        surface_world,
        expected_grouped_surface_world,
    ):
        raise RuntimeError("T1 boundary particles are not contiguous by world")
    surface_by_world = surface.reshape(
        world_count,
        expected_surface,
    )

    approved_shapes = _approved_soft_shape_indices(
        model,
        tuple(solver_cfg.soft_shape_label_fragments),
    )
    approved_world = shape_world[approved_shapes]
    global_shape_count = int(np.count_nonzero(approved_world < 0))
    local_shape_counts = np.bincount(
        approved_world[approved_world >= 0],
        minlength=world_count,
    )
    expected_shapes = int(solver_cfg.expected_soft_shapes_per_environment)
    effective_shape_counts = local_shape_counts + global_shape_count
    if np.any(effective_shape_counts != expected_shapes):
        received = sorted(set(map(int, effective_shape_counts.tolist())))
        raise RuntimeError(
            "T1 approved soft-shape count drift: "
            f"expected {expected_shapes} per environment, got {received}"
        )

    pair_count = world_count * expected_surface * expected_shapes
    if pair_count > int(solver_cfg.maximum_soft_candidate_pairs):
        raise RuntimeError(
            "T1 bounded soft-contact preflight rejected "
            f"{pair_count:,} candidate pairs; limit is "
            f"{solver_cfg.maximum_soft_candidate_pairs:,}"
        )

    soft_capacity = world_count * int(solver_cfg.soft_contacts_per_environment)
    rigid_sensor_capacity = world_count * int(solver_cfg.rigid_sensor_contacts_per_environment)
    estimated_bytes = estimate_contact_pipeline_memory_bytes(
        soft_candidate_pairs=pair_count,
        soft_contact_capacity=soft_capacity,
        rigid_sensor_contact_capacity=rigid_sensor_capacity,
    )
    if estimated_bytes > int(solver_cfg.maximum_contact_pipeline_memory_bytes):
        raise RuntimeError(
            "T1 contact-memory preflight rejected "
            f"{estimated_bytes:,} estimated bytes; limit is "
            f"{solver_cfg.maximum_contact_pipeline_memory_bytes:,}"
        )

    global_shapes = approved_shapes[approved_world < 0]
    local_shapes = approved_shapes[approved_world >= 0]
    local_world = approved_world[approved_world >= 0]
    local_shapes_per_world = expected_shapes - global_shape_count
    if local_shapes_per_world:
        local_order = np.argsort(local_world, kind="stable")
        grouped_local_world = local_world[local_order]
        expected_grouped_local_world = np.repeat(
            np.arange(world_count, dtype=np.int32),
            local_shapes_per_world,
        )
        if not np.array_equal(
            grouped_local_world,
            expected_grouped_local_world,
        ):
            raise RuntimeError("T1 could not group approved shapes by world")
        local_shapes_by_world = local_shapes[local_order].reshape(
            world_count,
            local_shapes_per_world,
        )
    else:
        local_shapes_by_world = np.empty(
            (world_count, 0),
            dtype=np.int32,
        )
    global_shapes_by_world = np.broadcast_to(
        global_shapes,
        (world_count, global_shape_count),
    )
    shapes_by_world = np.concatenate(
        (local_shapes_by_world, global_shapes_by_world),
        axis=1,
    )
    if shapes_by_world.shape != (world_count, expected_shapes):
        raise RuntimeError("T1 soft-pair builder produced an invalid shape grouping")

    # Fill the per-world Cartesian product directly.  Scanning every surface
    # particle once per approved shape would be O(worlds^2) at 2,400 worlds.
    # These broadcast assignments are linear in the 13.44M qualified pairs.
    pair_grid = np.empty(
        (
            world_count,
            expected_surface,
            expected_shapes,
            2,
        ),
        dtype=np.int32,
    )
    pair_grid[..., 0] = surface_by_world[:, :, None]
    pair_grid[..., 1] = shapes_by_world[:, None, :]
    pairs = pair_grid.reshape(pair_count, 2)
    return pairs, {
        "environment_count": world_count,
        "surface_particles_per_environment": expected_surface,
        "approved_soft_shapes_per_environment": expected_shapes,
        "soft_candidate_pair_count": pair_count,
        "soft_contact_capacity": soft_capacity,
        "rigid_sensor_contact_capacity": rigid_sensor_capacity,
        "estimated_contact_pipeline_bytes": estimated_bytes,
    }


def _build_bounded_collision_pipeline(
    model: Model,
    solver_cfg: DrAnmarCoupledMJWarpVBDSolverCfg,
    pairs: np.ndarray,
) -> CollisionPipeline:
    """Construct without ever materializing Newton's all-particle pair list."""

    device = model.device
    empty_rigid_pairs = wp.array(
        np.empty((0, 2), dtype=np.int32),
        dtype=wp.vec2i,
        device=device,
    )
    original_particle_count = int(model.particle_count)
    try:
        # Pinned Newton's own coupled manager temporarily sets particle_count=0
        # while stepping MuJoCo.  Here it prevents CollisionPipeline.__init__
        # from allocating the default all-particle pair list.
        model.particle_count = 0
        pipeline = CollisionPipeline(
            model,
            reduce_contacts=False,
            broad_phase="explicit",
            shape_pairs_filtered=empty_rigid_pairs,
            rigid_contact_max=0,
            max_triangle_pairs=1,
            soft_contact_max=(
                int(model.world_count) * int(solver_cfg.soft_contacts_per_environment)
            ),
            soft_contact_margin=float(solver_cfg.soft_contact_margin_m),
            enable_rigid_soft_full_surface_contact=False,
            verify_buffers=False,
        )
    finally:
        model.particle_count = original_particle_count

    required_layout = (
        "soft_rigid_contact_pairs",
        "_soft_rigid_contact_pair_count",
        "soft_edge_rigid_pairs",
        "soft_face_rigid_pairs",
    )
    missing = [name for name in required_layout if not hasattr(pipeline, name)]
    if missing:
        raise RuntimeError(
            "Pinned Newton CollisionPipeline layout changed; missing "
            f"{missing}. Refusing an unbounded fallback."
        )
    pipeline.soft_rigid_contact_pairs = wp.array(
        pairs,
        dtype=wp.vec2i,
        device=device,
    )
    pipeline._soft_rigid_contact_pair_count = int(pairs.shape[0])
    if pipeline.soft_rigid_contact_pair_count != int(pairs.shape[0]):
        raise RuntimeError("T1 failed to install its bounded soft-pair list")
    expected_soft_capacity = int(model.world_count) * int(solver_cfg.soft_contacts_per_environment)
    if int(pipeline.soft_contact_max) != expected_soft_capacity:
        raise RuntimeError("T1 collision pipeline changed soft-contact capacity")
    if len(pipeline.soft_edge_rigid_pairs) or len(pipeline.soft_face_rigid_pairs):
        raise RuntimeError("T1 collision pipeline unexpectedly enabled full-surface pairs")
    return pipeline


class DrAnmarCoupledMJWarpVBDManager(NewtonCoupledMJWarpVBDManager):
    """Coupled manager with bounded collision work and fresh T1 receipts."""

    _dr_anmar_solver_cfg: ClassVar[DrAnmarCoupledMJWarpVBDSolverCfg | None] = None
    _dr_anmar_environment_count: ClassVar[int] = 0
    _dr_anmar_particles_per_environment: ClassVar[int] = 0
    _dr_anmar_shape_class: ClassVar[Any | None] = None
    _dr_anmar_receipt_generation: ClassVar[Any | None] = None
    _dr_anmar_environment_generation: ClassVar[Any | None] = None
    _dr_anmar_candidate_seen: ClassVar[Any | None] = None
    _dr_anmar_penetration_seen: ClassVar[Any | None] = None
    _dr_anmar_maximum_penetration: ClassVar[Any | None] = None
    _dr_anmar_tip_particle_penetration_seen: ClassVar[Any | None] = None
    _dr_anmar_tip_particle_maximum_penetration: ClassVar[Any | None] = None
    _dr_anmar_overflow_seen: ClassVar[Any | None] = None
    _dr_anmar_rigid_sensor_contacts: ClassVar[Contacts | None] = None
    _dr_anmar_pipeline_preflight: ClassVar[dict[str, int] | None] = None
    _dr_anmar_receipt_torch_views: ClassVar[dict[str, Any] | None] = None

    @classmethod
    def _build_solver(
        cls,
        model: Model,
        solver_cfg: DrAnmarCoupledMJWarpVBDSolverCfg,
    ) -> None:
        if not isinstance(
            solver_cfg,
            DrAnmarCoupledMJWarpVBDSolverCfg,
        ):
            raise TypeError(
                "DrAnmarCoupledMJWarpVBDManager requires its bounded solver configuration"
            )
        if str(solver_cfg.coupling_mode) != "two_way":
            raise ValueError("T1 contact receipts require two-way MJWarp/VBD coupling")
        environment_count = int(model.world_count)
        if environment_count <= 0:
            raise RuntimeError("T1 requires a positive Newton world count")
        if environment_count > int(solver_cfg.maximum_environment_count):
            raise RuntimeError(
                "T1 requested "
                f"{environment_count} environments, above the qualified limit "
                f"{solver_cfg.maximum_environment_count}"
            )
        if int(model.particle_count) <= 0:
            raise RuntimeError("T1 requires deformable tissue particles")
        if int(model.particle_count) % environment_count:
            raise RuntimeError("T1 particle count is not divisible by environment count")
        particles_per_environment = int(model.particle_count) // environment_count
        particle_world = np.asarray(
            model.particle_world.numpy(),
            dtype=np.int32,
        )
        expected_particle_world = np.repeat(
            np.arange(environment_count, dtype=np.int32),
            particles_per_environment,
        )
        if not np.array_equal(particle_world, expected_particle_world):
            raise RuntimeError(
                "T1 requires contiguous, equal per-world particle storage for "
                "zero-copy per-environment contact receipts"
            )

        # Construct the coupled solvers explicitly.  Pinned Newton's ordinary
        # SolverVBD constructor allocates shape_count * particle_count
        # body-particle state before the bounded CollisionPipeline exists.  At
        # 2,400 worlds that global cross-product is the dominant structural
        # allocation, so the T1 subclass replaces it with the versioned soft
        # contact capacity from the outset.
        soft_contact_capacity = environment_count * int(solver_cfg.soft_contacts_per_environment)
        if soft_contact_capacity <= 0:
            raise RuntimeError("T1 soft-contact capacity must be positive")
        cls._coupling_mode = str(solver_cfg.coupling_mode)
        cls._rigid_solver = SolverMuJoCo(
            model,
            **cls._filter_solver_kwargs(
                SolverMuJoCo,
                solver_cfg.rigid_solver_cfg,
            ),
        )
        cls._soft_solver = _DrAnmarBoundedSolverVBD(
            model,
            dranmar_soft_contact_max=soft_contact_capacity,
            **cls._filter_solver_kwargs(
                SolverVBD,
                solver_cfg.soft_solver_cfg,
            ),
        )
        # The canonical slot is intentionally a no-op facade.  Coupled
        # stepping, lifecycle notifications, reset, and sensing are routed to
        # the two real sub-solvers by this manager and its pinned parent.
        NewtonManager._solver = SolverBase(model)
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = True

        cls._dr_anmar_solver_cfg = solver_cfg
        cls._dr_anmar_receipt_torch_views = None
        shape_classes = [_shape_contact_class(str(label)) for label in model.shape_label]
        device = model.device
        cls._dr_anmar_environment_count = environment_count
        cls._dr_anmar_particles_per_environment = particles_per_environment
        cls._dr_anmar_shape_class = wp.array(
            shape_classes,
            dtype=wp.int32,
            device=device,
        )
        receipt_count = environment_count * _CONTACT_CLASSES
        cls._dr_anmar_receipt_generation = wp.zeros(
            1,
            dtype=wp.int32,
            device=device,
        )
        cls._dr_anmar_environment_generation = wp.full(
            environment_count,
            -1,
            dtype=wp.int32,
            device=device,
        )
        cls._dr_anmar_candidate_seen = wp.zeros(
            receipt_count,
            dtype=wp.int32,
            device=device,
        )
        cls._dr_anmar_penetration_seen = wp.zeros(
            receipt_count,
            dtype=wp.int32,
            device=device,
        )
        cls._dr_anmar_maximum_penetration = wp.zeros(
            receipt_count,
            dtype=wp.float32,
            device=device,
        )
        cls._dr_anmar_tip_particle_penetration_seen = wp.zeros(
            int(model.particle_count),
            dtype=wp.int32,
            device=device,
        )
        cls._dr_anmar_tip_particle_maximum_penetration = wp.zeros(
            int(model.particle_count),
            dtype=wp.float32,
            device=device,
        )
        cls._dr_anmar_overflow_seen = wp.zeros(
            1,
            dtype=wp.int32,
            device=device,
        )

    @classmethod
    def _initialize_contacts(cls) -> None:
        solver_cfg = cls._dr_anmar_solver_cfg
        if solver_cfg is None:
            raise RuntimeError("T1 solver limits were not initialized")
        if int(cls._collision_decimation) != 0:
            raise RuntimeError(
                "T1 coupled manager requires collision_decimation=0 because "
                "it collides inside every coupled substep"
            )
        pairs, report = _build_bounded_soft_pairs(cls._model, solver_cfg)
        pipeline = _build_bounded_collision_pipeline(
            cls._model,
            solver_cfg,
            pairs,
        )
        soft_capacity = int(report["soft_contact_capacity"])
        NewtonManager._collision_pipeline = pipeline
        NewtonManager._contacts = Contacts(
            0,
            soft_capacity,
            soft_contact_tids_size=int(report["soft_candidate_pair_count"]),
            requires_grad=bool(cls._model.requires_grad),
            device=cls._model.device,
        )

        configured_rigid_capacity = int(report["rigid_sensor_contact_capacity"])
        actual_rigid_capacity = int(cls._rigid_solver.get_max_contact_count())
        if actual_rigid_capacity > configured_rigid_capacity:
            raise RuntimeError(
                "T1 MuJoCo contact capacity drift: solver requires "
                f"{actual_rigid_capacity}, configured bounded sensor buffer is "
                f"{configured_rigid_capacity}"
            )
        requested_attributes = set(cls._model.get_requested_contact_attributes())
        requested_attributes.add("force")
        cls._dr_anmar_rigid_sensor_contacts = Contacts(
            configured_rigid_capacity,
            0,
            device=cls._model.device,
            requested_attributes=requested_attributes,
        )
        cls._dr_anmar_pipeline_preflight = report

    @classmethod
    def _reset_dranmar_contact_receipt(cls) -> None:
        if cls._dr_anmar_candidate_seen is None:
            return
        wp.launch(
            _advance_contact_receipt_generation,
            dim=1,
            inputs=[cls._dr_anmar_receipt_generation],
        )
        wp.launch(
            _reset_environment_contact_receipt,
            dim=cls._dr_anmar_environment_count * _CONTACT_CLASSES,
            inputs=[
                cls._dr_anmar_receipt_generation,
                cls._dr_anmar_environment_generation,
                cls._dr_anmar_candidate_seen,
                cls._dr_anmar_penetration_seen,
                cls._dr_anmar_maximum_penetration,
            ],
        )
        wp.launch(
            _reset_tip_particle_contact_receipt,
            dim=(cls._dr_anmar_environment_count * cls._dr_anmar_particles_per_environment),
            inputs=[
                cls._dr_anmar_tip_particle_penetration_seen,
                cls._dr_anmar_tip_particle_maximum_penetration,
            ],
        )
        wp.launch(
            _reset_overflow_receipt,
            dim=1,
            inputs=[cls._dr_anmar_overflow_seen],
        )

    @classmethod
    def invalidate_dranmar_contact_receipt(cls, env_ids) -> None:
        """Invalidate selected worlds immediately after an Isaac Lab reset."""

        if (
            cls._dr_anmar_environment_generation is None
            or env_ids is None
            or int(env_ids.numel()) == 0
        ):
            return
        contiguous = env_ids.contiguous()
        wp.launch(
            _invalidate_environment_contact_receipt,
            dim=int(contiguous.numel()),
            inputs=[
                wp.from_torch(contiguous, dtype=wp.int64),
                cls._dr_anmar_environment_generation,
                cls._dr_anmar_candidate_seen,
                cls._dr_anmar_penetration_seen,
                cls._dr_anmar_maximum_penetration,
            ],
        )

    @classmethod
    def _observe_soft_contacts(
        cls,
        contacts: Contacts | None,
        state: State,
    ) -> None:
        if (
            contacts is None
            or cls._dr_anmar_shape_class is None
            or cls._dr_anmar_candidate_seen is None
        ):
            return
        capacity = int(contacts.soft_contact_particle.shape[0])
        if capacity <= 0:
            return
        solver_cfg = cls._dr_anmar_solver_cfg
        if solver_cfg is None:
            raise RuntimeError("T1 contact observer lost its solver config")
        tip = tuple(map(float, solver_cfg.needle_tip_offset_m))
        model = cls._model
        wp.launch(
            _accumulate_soft_contact_receipt,
            dim=capacity,
            inputs=[
                contacts.soft_contact_count,
                capacity,
                contacts.soft_contact_particle,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_normal,
                state.particle_q,
                model.particle_radius,
                model.particle_world,
                state.body_q,
                model.shape_body,
                model.shape_margin,
                cls._dr_anmar_shape_class,
                cls._dr_anmar_environment_count,
                wp.vec3(tip[0], tip[1], tip[2]),
                float(solver_cfg.needle_tip_contact_radius_m),
                cls._dr_anmar_candidate_seen,
                cls._dr_anmar_penetration_seen,
                cls._dr_anmar_maximum_penetration,
                cls._dr_anmar_tip_particle_penetration_seen,
                cls._dr_anmar_tip_particle_maximum_penetration,
                cls._dr_anmar_overflow_seen,
            ],
        )

    @classmethod
    def _apply_reactions(
        cls,
        state: State,
        state_prev: State,
        dt: float,
    ) -> None:
        # The parent calls this immediately after its per-substep collide().
        cls._observe_soft_contacts(cls._contacts, state)
        super()._apply_reactions(state, state_prev, dt)

    @classmethod
    def _simulate_full(cls) -> None:
        """Run the coupled loop without the base manager's redundant collide."""

        cls._reset_dranmar_contact_receipt()
        physics_dt = cls._solver_dt * cls._num_substeps
        for _ in range(cls._decimation):
            if cls._adapter is not None:
                cls._adapter.step(cls._state_0, cls._control, physics_dt)
            for callback in cls._post_actuator_callbacks:
                callback()
            cls._run_solver_substeps(cls._contacts)
        for callback in cls._post_step_callbacks:
            callback()
        cls._update_sensors(cls._contacts)

    @classmethod
    def _simulate_physics_only(cls) -> None:
        """Run coupled substeps without the redundant top-level collide."""

        cls._reset_dranmar_contact_receipt()
        if hasattr(cls._soft_solver, "rebuild_bvh"):
            cls._soft_solver.rebuild_bvh(cls._state_0)
        cls._run_solver_substeps(cls._contacts)
        for callback in cls._post_step_callbacks:
            callback()
        cls._update_sensors(cls._contacts)

    @classmethod
    def _reset_solver_internals(cls, world_mask) -> None:
        """Clear both real coupled solvers and invalidate reset receipts."""

        if world_mask is None:
            return
        cls._rigid_solver.reset(
            cls._state_0,
            world_mask=world_mask,
            flags=0,
        )
        cls._soft_solver.reset(
            cls._state_0,
            world_mask=world_mask,
            flags=0,
        )
        if cls._dr_anmar_environment_generation is not None:
            wp.launch(
                _invalidate_masked_contact_receipt,
                dim=cls._dr_anmar_environment_count,
                inputs=[
                    world_mask,
                    cls._dr_anmar_environment_generation,
                    cls._dr_anmar_candidate_seen,
                    cls._dr_anmar_penetration_seen,
                    cls._dr_anmar_maximum_penetration,
                ],
            )

    @classmethod
    def _update_sensors(cls, contacts) -> None:
        """Route rigid sensors through a separate, explicitly sized buffer."""

        if cls._newton_frame_transform_sensors:
            for sensor in cls._newton_frame_transform_sensors:
                sensor.update(cls._state_0)
        if cls._newton_imu_sensors:
            for sensor in cls._newton_imu_sensors:
                sensor.update(cls._state_0)
        if cls._report_contacts:
            sensor_contacts = cls._dr_anmar_rigid_sensor_contacts
            if sensor_contacts is None:
                raise RuntimeError("T1 rigid sensor contact buffer was not initialized")
            cls._rigid_solver.update_contacts(
                sensor_contacts,
                cls._state_0,
            )
            for sensor in cls._newton_contact_sensors.values():
                sensor.update(cls._state_0, sensor_contacts)

    @classmethod
    def _solver_specific_clear(cls) -> None:
        super()._solver_specific_clear()
        cls._dr_anmar_solver_cfg = None
        cls._dr_anmar_environment_count = 0
        cls._dr_anmar_particles_per_environment = 0
        cls._dr_anmar_shape_class = None
        cls._dr_anmar_receipt_generation = None
        cls._dr_anmar_environment_generation = None
        cls._dr_anmar_candidate_seen = None
        cls._dr_anmar_penetration_seen = None
        cls._dr_anmar_maximum_penetration = None
        cls._dr_anmar_tip_particle_penetration_seen = None
        cls._dr_anmar_tip_particle_maximum_penetration = None
        cls._dr_anmar_overflow_seen = None
        cls._dr_anmar_rigid_sensor_contacts = None
        cls._dr_anmar_pipeline_preflight = None
        cls._dr_anmar_receipt_torch_views = None
        cls._rigid_solver = None
        cls._soft_solver = None
        cls._coupling_mode = None

    @classmethod
    def get_dranmar_soft_contact_receipt(
        cls,
    ) -> dict[str, Any] | None:
        """Return zero-copy Torch views of the latest complete tick."""

        if cls._dr_anmar_candidate_seen is None:
            return None
        if cls._dr_anmar_receipt_torch_views is not None:
            return cls._dr_anmar_receipt_torch_views
        environment_shape = (
            cls._dr_anmar_environment_count,
            _CONTACT_CLASSES,
        )
        particle_shape = (
            cls._dr_anmar_environment_count,
            cls._dr_anmar_particles_per_environment,
        )
        receipt = {
            "generation": wp.to_torch(cls._dr_anmar_receipt_generation),
            "environment_generation": wp.to_torch(cls._dr_anmar_environment_generation),
            "candidate_seen": wp.to_torch(cls._dr_anmar_candidate_seen).reshape(environment_shape),
            "penetration_seen": wp.to_torch(cls._dr_anmar_penetration_seen).reshape(
                environment_shape
            ),
            "maximum_penetration_m": wp.to_torch(cls._dr_anmar_maximum_penetration).reshape(
                environment_shape
            ),
            "needle_tip_particle_penetration_seen": wp.to_torch(
                cls._dr_anmar_tip_particle_penetration_seen
            ).reshape(particle_shape),
            "needle_tip_particle_maximum_penetration_m": wp.to_torch(
                cls._dr_anmar_tip_particle_maximum_penetration
            ).reshape(particle_shape),
            "overflow_seen": wp.to_torch(cls._dr_anmar_overflow_seen),
            "contact_classes": {
                "needle": NEEDLE_CONTACT_CLASS,
                "robot_1": ROBOT_1_CONTACT_CLASS,
                "robot_2": ROBOT_2_CONTACT_CLASS,
                "other": OTHER_CONTACT_CLASS,
            },
            "pipeline_preflight": cls._dr_anmar_pipeline_preflight,
            "authority": ("bounded_newton_vbd_vertex_penetration_accumulated_across_substeps"),
            "generic_needle_contact_is_puncture": False,
            "force_calibrated": False,
            "full_surface_contact": False,
        }
        cls._dr_anmar_receipt_torch_views = receipt
        return receipt


__all__ = [
    "DrAnmarCoupledMJWarpVBDSolverCfg",
    "DrAnmarCoupledMJWarpVBDManager",
    "NEEDLE_CONTACT_CLASS",
    "OTHER_CONTACT_CLASS",
    "ROBOT_1_CONTACT_CLASS",
    "ROBOT_2_CONTACT_CLASS",
    "estimate_contact_pipeline_memory_bytes",
]
