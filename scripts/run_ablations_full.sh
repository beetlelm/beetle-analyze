#!/usr/bin/env bash
# run_ablations_full.sh — analyse the 24-repo minimal ablation sweep on nld-eng
# (8 variants × 3 seeds). Aggregates across seeds in figures/paper_figures.

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "${DIR}/../analyze/ablation_manifest.json" ]]; then
    echo "[ablations] refreshing manifest"
    python3 "${DIR}/sync_ablation_manifest.py" --beetlelm ../beetlelm
fi

bash "${DIR}/run_beetle_analyze.sh" \
    --family ablations \
    --langs "nld eng" \
    --cache-mode rotate \
    "$@"
