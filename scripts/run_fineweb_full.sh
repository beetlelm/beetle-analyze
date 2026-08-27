#!/usr/bin/env bash
# run_fineweb_full.sh — full analysis pipeline over Beetle-FineWeb models.
#
# Run scripts/sync_fineweb_manifest.py first to refresh
# analyze/fineweb_model_manifest.json from the sibling BeetleLM repo.

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "${DIR}/../analyze/fineweb_model_manifest.json" ]]; then
    echo "[fineweb] refreshing manifest from ../beetlelm"
    python3 "${DIR}/sync_fineweb_manifest.py" --beetlelm ../beetlelm
fi

bash "${DIR}/run_beetle_analyze.sh" \
    --family fineweb \
    --langs "deu nld zho eng" \
    --cache-mode rotate \
    "$@"
