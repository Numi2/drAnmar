# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""T1: retain the handed-over needle and approach deformable tissue safely."""

from __future__ import annotations

import json
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab_contrib.deformable import (
    NewtonModelCfg,
    VBDSolverCfg,
)
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.renderers import (
    IsaacRtxRendererCfg,
    IsaacRtxRendererGlobalSettingsCfg,
)
from isaaclab.sensors import CameraCfg
from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR
from orbit.surgical.assets.needle_ready_tissue import (
    load_needle_ready_tissue_geometry_contract,
    make_needle_ready_tissue_cfg,
)

from ... import mdp
from ...newton_contact_manager import (
    DrAnmarCoupledMJWarpVBDSolverCfg,
)
from . import e2e_ik_rel_env_cfg


def _load_repository_contract() -> dict:
    for parent in Path(__file__).resolve().parents:
        path = parent / "config/dranmar_safe_bite_t1.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError("config/dranmar_safe_bite_t1.json not found")


def terminal_safe_bite_failure(env):
    """Penalize premature tissue contact once; handover failures stay separate."""

    return mdp.safe_bite_premature_contact(env).float()


def _coupled_physics_cfg(
    contract: dict,
    *,
    continuation: bool,
) -> tuple[NewtonCfg, float]:
    """Build either the throughput or contact-transition solver profile."""

    scene_contract = contract["scene"]
    mjwarp = scene_contract["mjwarp"]
    vbd = scene_contract["vbd"]
    contact_pipeline = scene_contract["contact_pipeline"]
    needle_frame = contract["needle_frame"]
    tissue_lod = str(scene_contract["tissue_lod"])
    profile = scene_contract["continuation_solver"] if continuation else {}
    physics = NewtonCfg(
        solver_cfg=DrAnmarCoupledMJWarpVBDSolverCfg(
            model_cfg=NewtonModelCfg(
                soft_contact_ke=float(profile.get("soft_contact_ke", 2.5e3)),
                soft_contact_kd=float(profile.get("soft_contact_kd", 5.0e-2)),
                soft_contact_mu=float(profile.get("soft_contact_mu", 0.3)),
            ),
            rigid_solver_cfg=MJWarpSolverCfg(
                njmax=int(mjwarp["constraint_capacity_per_environment"]),
                nconmax=int(mjwarp["contact_capacity_per_environment"]),
                iterations=int(mjwarp["iterations"]),
                ls_iterations=int(mjwarp["line_search_iterations"]),
                cone=str(mjwarp["contact_cone"]),
                impratio=1,
                integrator=str(mjwarp["integrator"]),
            ),
            soft_solver_cfg=VBDSolverCfg(
                iterations=int(profile.get("vbd_iterations", vbd["iterations"])),
                integrate_with_external_rigid_solver=True,
                particle_enable_self_contact=bool(vbd["self_contact"]),
                particle_collision_detection_interval=-1,
            ),
            coupling_mode=str(vbd["coupling_mode"]),
            maximum_environment_count=int(contact_pipeline["maximum_environment_count"]),
            soft_contacts_per_environment=int(contact_pipeline["soft_contacts_per_environment"]),
            rigid_sensor_contacts_per_environment=int(
                contact_pipeline["rigid_sensor_contacts_per_environment"]
            ),
            maximum_soft_candidate_pairs=int(contact_pipeline["maximum_soft_candidate_pairs"]),
            maximum_contact_pipeline_memory_bytes=int(
                contact_pipeline["maximum_contact_pipeline_memory_bytes"]
            ),
            expected_surface_particles_per_environment=int(
                contact_pipeline["expected_surface_particles_by_lod"][tissue_lod]
            ),
            expected_soft_shapes_per_environment=int(
                contact_pipeline["expected_approved_soft_shapes_per_environment"]
            ),
            soft_contact_margin_m=float(contact_pipeline["soft_contact_margin_m"]),
            needle_tip_offset_m=tuple(map(float, needle_frame["tip_offset_in_needle_root_m"])),
            needle_tip_contact_radius_m=float(needle_frame["tip_contact_region_radius_m"]),
            soft_shape_label_fragments=tuple(
                map(
                    str,
                    contact_pipeline["approved_soft_shape_label_fragments"],
                )
            ),
        ),
        num_substeps=int(
            profile.get(
                "newton_substeps",
                scene_contract["newton_substeps"],
            )
        ),
        collision_decimation=0,
        use_cuda_graph=True,
    )
    return physics, float(profile.get("physics_dt_s", scene_contract["physics_dt_s"]))


@configclass
class NeedleHandoverSafeBiteT1EnvCfg(e2e_ik_rel_env_cfg.NeedleHandoverEndToEndEnvCfg):
    """Efficient successor task with full-chain physical promotion support."""

    def __post_init__(self):
        super().__post_init__()
        contract = _load_repository_contract()
        geometry = load_needle_ready_tissue_geometry_contract()
        contract["tissue_geometry"] = geometry["geometry"]
        contract["tissue_lods"] = geometry["lods"]
        contract["tissue_semantics"] = geometry["semantics"]
        self.dr_anmar_safe_bite_contract = contract

        scene_contract = contract["scene"]
        tissue_position = tuple(map(float, scene_contract["tissue_position_in_environment_m"]))
        self.scene.tissue = make_needle_ready_tissue_cfg(
            lod=str(scene_contract["tissue_lod"]),
            prim_path="{ENV_REGEX_NS}/NeedleReadyTissue",
            position=tissue_position,
        )
        self.scene.num_envs = int(contract["launch_profiles"]["training_2400"]["environment_count"])

        self.sim.physics, self.sim.dt = _coupled_physics_cfg(
            contract,
            continuation=False,
        )
        self.decimation = int(scene_contract["control_decimation"])
        self.sim.render_interval = self.decimation
        self.episode_length_s = 25.0

        self.events.safe_bite_snapshot_reset = EventTerm(
            func=mdp.reset_safe_bite_from_handover_cache,
            mode="reset",
        )
        self.events.tissue_outer_fixture_reset = EventTerm(
            func=mdp.reset_tissue_outer_fixture,
            mode="reset",
        )
        self.observations.policy.safe_bite = ObsTerm(
            func=mdp.safe_bite_observation,
            clip=(-5.0, 5.0),
        )

        rewards = contract["rewards"]
        # Handover is a prerequisite in this environment. Its sticky success
        # bit must not emit 80 points on every successor step.
        self.rewards.phase_progress.weight = float(rewards["handover_phase_reward"])
        self.rewards.success.weight = 0.0
        self.rewards.terminal_transfer_failure.weight = float(rewards["terminal_failure"])
        self.rewards.safe_bite_progress = RewTerm(
            func=mdp.safe_bite_approach_progress,
            weight=float(rewards["approach_progress_weight"]),
        )
        self.rewards.safe_bite_success = RewTerm(
            func=mdp.safe_bite_entry_armed,
            weight=float(rewards["terminal_armed_entry"]),
        )
        self.rewards.safe_bite_failure = RewTerm(
            func=terminal_safe_bite_failure,
            weight=float(rewards["terminal_failure"]),
        )
        self.rewards.authorized_contact_transition = RewTerm(
            func=mdp.safe_bite_authorized_contact_transition,
            weight=0.0,
        )
        self.rewards.safe_bite_success_rate = RewTerm(
            func=mdp.sticky_success_rate,
            params={"success_fn": mdp.safe_bite_entry_armed},
            weight=0.0,
        )

        self.terminations.success = DoneTerm(func=mdp.safe_bite_entry_armed)
        self.terminations.premature_tissue_contact = DoneTerm(func=mdp.safe_bite_premature_contact)


@configclass
class NeedleHandoverSafeBiteT1EnvCfg_PLAY(NeedleHandoverSafeBiteT1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


def _configure_visual_qualification_scene(cfg, *, lod: str) -> None:
    """Add render-only assets without changing task frames or physics."""

    scene_contract = cfg.dr_anmar_safe_bite_contract["scene"]
    visual_root = (
        Path(ORBITSURGICAL_ASSETS_DATA_DIR)
        / "Props"
        / "SurgicalScene"
        / "T1"
    )
    cfg.scene.robot_1.spawn = cfg.scene.robot_1.spawn.replace(
        usd_path=str(visual_root / "psm_visual_v1.usda"),
    )
    cfg.scene.robot_2.spawn = cfg.scene.robot_2.spawn.replace(
        usd_path=str(visual_root / "psm_visual_v1.usda"),
    )
    cfg.scene.table.spawn = cfg.scene.table.spawn.replace(
        usd_path=str(visual_root / "table_visual_v1.usda"),
    )
    cfg.scene.object.spawn = cfg.scene.object.spawn.replace(
        usd_path=str(visual_root / "legacy_needle_visual_v1.usda"),
    )
    tissue_position = tuple(map(float, scene_contract["tissue_position_in_environment_m"]))
    cfg.scene.tissue = make_needle_ready_tissue_cfg(
        lod=lod,
        prim_path="{ENV_REGEX_NS}/NeedleReadyTissue",
        position=tissue_position,
        visual_quality=True,
    )
    cfg.scene.light = AssetBaseCfg(
        prim_path="/World/T1QualificationDome",
        spawn=sim_utils.DomeLightCfg(
            color=(0.18, 0.18, 0.18),
            intensity=350.0,
            enable_color_temperature=True,
            color_temperature=4500.0,
            visible_in_primary_ray=False,
        ),
    )
    cfg.scene.surgical_key_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/T1QualificationKey",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.12, 0.04, 0.30),
        ),
        spawn=sim_utils.DiskLightCfg(
            radius=0.075,
            intensity=2200.0,
            normalize=True,
            enable_color_temperature=True,
            color_temperature=4500.0,
        ),
    )
    cfg.scene.surgical_fill_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/T1QualificationFill",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.02, -0.18, 0.20),
        ),
        spawn=sim_utils.SphereLightCfg(
            radius=0.10,
            intensity=650.0,
            normalize=True,
            enable_color_temperature=True,
            color_temperature=4500.0,
        ),
    )
    cfg.scene.qualification_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/T1QualificationCamera",
        update_period=1.0 / 30.0,
        height=1080,
        width=1920,
        data_types=["rgb", "rgb_hdr", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.0,
            focus_distance=0.25,
            f_stop=4.0,
            horizontal_aperture=20.955,
            clipping_range=(0.02, 2.0),
        ),
        renderer_cfg=IsaacRtxRendererCfg(
            global_settings=IsaacRtxRendererGlobalSettingsCfg(
                antialiasing_mode="DLAA",
                enable_reflections=True,
                enable_global_illumination=True,
                enable_direct_lighting=True,
                enable_shadows=True,
                enable_ambient_occlusion=True,
                enable_dl_denoiser=True,
                samples_per_pixel=4,
                carb_settings={
                    "/omni/replicator/captureMotionBlur": False,
                    "/rtx/post/histogram/enabled": False,
                    "/rtx/post/lensFlares/enabled": False,
                    "/rtx/post/motionblur/enabled": False,
                    "/rtx/post/tonemap/cameraShutter": 1.0 / 120.0,
                    "/rtx/post/tonemap/enabled": True,
                    "/rtx/post/tonemap/fNumber": 4.0,
                    "/rtx/post/tonemap/filmIso": 400.0,
                    "/rtx/post/tonemap/op": 4,
                    "/rtx/post/tvNoise/enabled": False,
                    ("/rtx/raytracing/subsurface/transmission/bsdfSampleCount"): 4,
                    ("/rtx/raytracing/subsurface/transmission/denoiser/enabled"): True,
                    ("/rtx/raytracing/subsurface/transmission/enabled"): True,
                    ("/rtx/raytracing/subsurface/transmission/perBsdfScatteringSampleCount"): 4,
                    "/rtx/raytracing/subsurface/maxSamplePerFrame": 16,
                },
            ),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.04, -0.10, 0.18),
            rot=(
                -0.8605267898,
                -0.2672708466,
                0.1286286612,
                0.4141432194,
            ),
            convention="ros",
        ),
    )
    cfg.scene.num_envs = 4
    cfg.scene.env_spacing = 2.5
    cfg.observations.policy.enable_corruption = False
    cfg.viewer.eye = (-0.04, -0.10, 0.18)
    cfg.viewer.lookat = (-0.15, 0.06, 0.025)


@configclass
class NeedleHandoverSafeBiteT1VisualEnvCfg(NeedleHandoverSafeBiteT1EnvCfg):
    """Small, frame-identical precontact visual-qualification lane."""

    def __post_init__(self):
        super().__post_init__()
        _configure_visual_qualification_scene(
            self,
            lod=str(self.dr_anmar_safe_bite_contract["scene"]["tissue_lod"]),
        )


@configclass
class NeedleHandoverSafeBiteChainEnvCfg(NeedleHandoverSafeBiteT1EnvCfg):
    """Keep stepping after entry readiness so authorized contact is possible."""

    def __post_init__(self):
        super().__post_init__()
        scene_contract = self.dr_anmar_safe_bite_contract["scene"]
        continuation_lod = str(scene_contract["continuation_tissue_lod"])
        scene_contract["tissue_lod"] = continuation_lod
        tissue_position = tuple(map(float, scene_contract["tissue_position_in_environment_m"]))
        self.scene.tissue = make_needle_ready_tissue_cfg(
            lod=continuation_lod,
            prim_path="{ENV_REGEX_NS}/NeedleReadyTissue",
            position=tissue_position,
        )
        self.sim.physics, self.sim.dt = _coupled_physics_cfg(
            self.dr_anmar_safe_bite_contract,
            continuation=True,
        )
        self.scene.num_envs = int(
            self.dr_anmar_safe_bite_contract["launch_profiles"]["contact_qualification_256"][
                "environment_count"
            ]
        )
        self.terminations.success = None
        self.rewards.safe_bite_success.weight = 0.0


@configclass
class NeedleHandoverSafeBiteChainVisualEnvCfg(NeedleHandoverSafeBiteChainEnvCfg):
    """Small contact-LOD visual lane with identical coupled physics."""

    def __post_init__(self):
        super().__post_init__()
        _configure_visual_qualification_scene(
            self,
            lod=str(self.dr_anmar_safe_bite_contract["scene"]["tissue_lod"]),
        )
