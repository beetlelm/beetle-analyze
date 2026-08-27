#!/usr/bin/env bash
# run_humanscale_full.sh — full analysis pipeline over the 20 HumanScale models.
#
# Thin wrapper around scripts/run_beetle_analyze.sh. Invokes every data-
# producing stage, then figures + paper_figures last.

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${DIR}/run_beetle_analyze.sh" \
    --family humanscale \
    --langs "deu nld zho eng" \
    --cache-mode rotate \
    "$@"
