#!/usr/bin/env bash
# run_paper_figures.sh — regenerate the ACL main-body figures from existing CSVs.
# Fast: does NOT re-download any model.

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${DIR}/run_beetle_analyze.sh" \
    --family all \
    --langs "deu nld zho eng" \
    --stages "paper_figures figures" \
    --cache-mode keep \
    "$@"
