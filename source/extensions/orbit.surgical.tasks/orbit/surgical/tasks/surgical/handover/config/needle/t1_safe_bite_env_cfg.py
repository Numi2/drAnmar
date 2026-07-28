# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""T1: retain the handed-over needle and approach deformable tissue safely."""

from __future__ import annotations

import json
from pathlib import Path

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab_contrib.deformable import (
    CoupledMJWarpVBDSolverCfg,
    NewtonModelCfg,
    VBDSolverCfg,
)
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from orbit.surgical.assets.needle_ready_tissue import (
    load_needle_ready_tissue_geometry_contract,
    make_needle_ready_tissue_cfg,
)

from ... import mdp
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


@configclass
class NeedleHandoverSafeBiteT1EnvCfg(
    e2e_ik_rel_env_cfg.NeedleHandoverEndToEndEnvCfg
):
    """Efficient successor task with full-chain physical promotion support."""

    def __post_init__(self):
        super().__post_init__()
        contract = _load_repository_contract()
        geometry = load_needle_ready_tissue_geometry_contract()
        contract["tissue_geometry"] = geometry["geometry"]
        self.dr_anmar_safe_bite_contract = contract

        scene_contract = contract["scene"]
        tissue_position = tuple(
            map(float, scene_contract["tissue_position_in_environment_m"])
        )
        self.scene.tissue = make_needle_ready_tissue_cfg(
            lod=str(scene_contract["tissue_lod"]),
            prim_path="{ENV_REGEX_NS}/NeedleReadyTissue",
            position=tissue_position,
        )

        mjwarp = scene_contract["mjwarp"]
        vbd = scene_contract["vbd"]
        self.sim.physics = NewtonCfg(
            solver_cfg=CoupledMJWarpVBDSolverCfg(
                model_cfg=NewtonModelCfg(
                    soft_contact_ke=2.5e3,
                    soft_contact_kd=5.0e-2,
                    soft_contact_mu=0.3,
                ),
                rigid_solver_cfg=MJWarpSolverCfg(
                    njmax=int(
                        mjwarp["constraint_capacity_per_environment"]
                    ),
                    nconmax=int(
                        mjwarp["contact_capacity_per_environment"]
                    ),
                    iterations=int(mjwarp["iterations"]),
                    ls_iterations=int(mjwarp["line_search_iterations"]),
                    cone=str(mjwarp["contact_cone"]),
                    impratio=1,
                    integrator=str(mjwarp["integrator"]),
                ),
                soft_solver_cfg=VBDSolverCfg(
                    iterations=int(vbd["iterations"]),
                    integrate_with_external_rigid_solver=True,
                    particle_enable_self_contact=bool(vbd["self_contact"]),
                    particle_collision_detection_interval=-1,
                ),
                coupling_mode=str(vbd["coupling_mode"]),
            ),
            num_substeps=int(scene_contract["newton_substeps"]),
            use_cuda_graph=True,
        )
        self.sim.dt = float(scene_contract["physics_dt_s"])
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
        self.rewards.phase_progress.weight = float(
            rewards["handover_phase_reward"]
        )
        self.rewards.success.weight = 0.0
        self.rewards.terminal_transfer_failure.weight = float(
            rewards["terminal_failure"]
        )
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

        self.terminations.success = DoneTerm(
            func=mdp.safe_bite_entry_armed
        )
        self.terminations.premature_tissue_contact = DoneTerm(
            func=mdp.safe_bite_premature_contact
        )


@configclass
class NeedleHandoverSafeBiteT1EnvCfg_PLAY(
    NeedleHandoverSafeBiteT1EnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class NeedleHandoverSafeBiteChainEnvCfg(
    NeedleHandoverSafeBiteT1EnvCfg
):
    """Keep stepping after entry readiness so authorized contact is possible."""

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success = None
        self.rewards.safe_bite_success.weight = 0.0
