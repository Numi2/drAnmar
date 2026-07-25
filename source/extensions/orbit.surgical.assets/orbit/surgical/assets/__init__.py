# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing asset and sensor configurations."""

import os
import toml

# Conveniences to other module directories via relative paths
ORBITSURGICAL_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
"""Path to the extension source directory."""

ORBITSURGICAL_ASSETS_DATA_DIR = os.path.join(ORBITSURGICAL_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""

ORBITSURGICAL_ASSETS_METADATA = toml.load(os.path.join(ORBITSURGICAL_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = ORBITSURGICAL_ASSETS_METADATA["package"]["version"]


##
# Configuration for different assets.
##

from .ecm import *
from .closure_robot import *
from .laparotomy_sponge import *
from .dranmar_asset_catalog import *
from .dranmar_camera_scheduler import *
from .needle_thread import *
from .psm import *
from .skin_adhesive import *
from .skin_stapler import *
from .star import *
from .wound_preparation_robot import *
from .atraumatic_exposure_robot import *
from .adaptive_hemostasis_robot import *
from .adaptive_anastomosis_robot import *
from .adaptive_seal_divide_robot import *
from .safeplane_dissection_robot import *
from .perfusion_viability_robot import *
from .dynamic_abdominal_patient import *
from .oncologic_resection import *
from .deformable_rescue import *
from .resuscitation_effects import *
from .autonomous_rescue_or import *
from .autonomous_rescue_scene import *
