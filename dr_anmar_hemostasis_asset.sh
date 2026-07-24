#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${REPOSITORY_ROOT}/physics_next/vessels/dr-anmar-hemostasis-v1.json"
ASSET_DIRECTORY="${REPOSITORY_ROOT}/assets/dr_anmar/hemostasis"
VESSEL="${ASSET_DIRECTORY}/DrAnmarVessel.usda"
TETMESH="${ASSET_DIRECTORY}/DrAnmarVessel.tet.usda"
CLIP="${ASSET_DIRECTORY}/DrAnmarVascularClip.usda"
ASSET_REPORT="${ASSET_DIRECTORY}/DrAnmarHemostasis.report.json"

case "${1:-author}" in
    author)
        python3 "${REPOSITORY_ROOT}/scripts/dr_anmar_hemostasis_author.py" \
            --profile "${PROFILE}" \
            --vessel-output "${VESSEL}" \
            --tet-output "${TETMESH}" \
            --clip-output "${CLIP}" \
            --report "${ASSET_REPORT}"
        ;;
    rebuild)
        "$0" author
        ;;
    inspect)
        python3 - <<'PY' "${ASSET_REPORT}"
import json
import sys
for path in sys.argv[1:]:
    payload = json.load(open(path, encoding="utf-8"))
    print(json.dumps(payload, indent=2))
PY
        ;;
    *)
        echo "Usage: $0 {author|rebuild|inspect}" >&2
        exit 2
        ;;
esac
