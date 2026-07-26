# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# Copyright (c) 2026, Dr.Anmar Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for various robotic environments."""

import os
from collections.abc import Callable

import gymnasium as gym
import toml
from isaaclab_tasks.utils import import_packages

# Conveniences to other module directories via relative paths
ORBITSURGICAL_TASKS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
"""Path to the extension source directory."""

ORBITSURGICAL_TASKS_METADATA = toml.load(os.path.join(ORBITSURGICAL_TASKS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = ORBITSURGICAL_TASKS_METADATA["package"]["version"]

##
# Register Gym environments.
##

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)


def _dranmar_env_entry_point(entry_point: Callable, task_id: str) -> Callable:
    """Build a named config factory for stable DrAnmar task registration."""

    def load_dranmar_env_cfg():
        return entry_point()

    load_dranmar_env_cfg.__name__ = f"load_{task_id.lower().replace('-', '_')}_cfg"
    load_dranmar_env_cfg.__qualname__ = load_dranmar_env_cfg.__name__
    return load_dranmar_env_cfg


def _register_dranmar_learning_path() -> tuple[str, ...]:
    """Expose stable Dr.Anmar task IDs without breaking legacy recordings."""

    registered: list[str] = []
    for task_id, spec in tuple(gym.registry.items()):
        if not task_id.startswith("Isaac-"):
            continue
        env_cfg = (spec.kwargs or {}).get("env_cfg_entry_point")
        module_name = getattr(env_cfg, "__module__", "")
        if not module_name.startswith("orbit.surgical.tasks."):
            continue
        dranmar_id = f"DrAnmar-{task_id.removeprefix('Isaac-')}"
        if dranmar_id not in gym.registry:
            kwargs = dict(spec.kwargs or {})
            kwargs["env_cfg_entry_point"] = _dranmar_env_entry_point(env_cfg, dranmar_id)
            gym.register(
                id=dranmar_id,
                entry_point=spec.entry_point,
                kwargs=kwargs,
                max_episode_steps=spec.max_episode_steps,
                disable_env_checker=True,
            )
        registered.append(dranmar_id)
    return tuple(sorted(registered))


DRANMAR_LEARNING_TASK_IDS = _register_dranmar_learning_path()
"""Gym task IDs that make up the versioned Dr.Anmar Learning Path."""
