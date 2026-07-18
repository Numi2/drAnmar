#!/usr/bin/env bash
set -euo pipefail

# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

: "${IsaacLab_PATH:?Set IsaacLab_PATH to your Isaac Lab checkout before running this installer.}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"${IsaacLab_PATH}/isaaclab.sh" -p -m pip install -e "${ROOT}/source/extensions/orbit.surgical.ext"
"${IsaacLab_PATH}/isaaclab.sh" -p -m pip install -e "${ROOT}/source/extensions/orbit.surgical.assets"
"${IsaacLab_PATH}/isaaclab.sh" -p -m pip install -e "${ROOT}/source/extensions/orbit.surgical.tasks"
