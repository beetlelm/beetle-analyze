#!/usr/bin/env bash
# smoke_test.sh — end-to-end sanity check on one model.
#
# Usage:
#     bash scripts/smoke_test.sh [--repo <hf_repo>] [--lang <iso>]
#
# Runs --ppl + --paper-figures on a single HumanScale monolingual model,
# then asserts the expected outputs exist. Exits non-zero on failure.

set -euo pipefail
REPO="Beetle-HumanScale/beetle-monolingual-humanscale-nld"
LANG="nld"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo) REPO="$2"; shift 2;;
        --lang) LANG="$2"; shift 2;;
        -h|--help)
            sed -n '2,10p' "$0"
            exit 0;;
        *) echo "Unknown flag: $1" >&2; exit 2;;
    esac
done

echo "[smoke] repo=${REPO}  lang=${LANG}"

# 1) Model registry assertion.
python3 - <<PY
from models import MODEL_GROUPS, get_all_models
from analyze.model_naming import parse_repo
all_m = get_all_models()
assert "${REPO}" in all_m, "repo not in MODEL_GROUPS: ${REPO}"
meta = parse_repo("${REPO}")
assert meta.family in {"humanscale","fineweb","beetlelm_legacy"}, meta
print(f"[smoke] registry OK — {len(all_m)} models registered; "
      f"family={meta.family}, condition={meta.condition}")
PY

# 2) PPL + paper_figures dry-run.
python3 analyze/run.py --ppl --paper-figures \
    --family humanscale --langs "${LANG}" \
    --cache-mode keep

# 3) Output assertions.
CSV_GLOB="results/ppl/ppl_${LANG}_*.csv"
PDF_GLOB="results/figures/main/*.pdf"
PNG_GLOB="results/figures/main/*.png"

shopt -s nullglob
csvs=(${CSV_GLOB})
pdfs=(${PDF_GLOB})
pngs=(${PNG_GLOB})

if [[ ${#csvs[@]} -eq 0 ]]; then
    echo "[smoke] FAIL: no PPL CSV matching ${CSV_GLOB}" >&2
    exit 1
fi
if [[ ${#pdfs[@]} -eq 0 ]] || [[ ${#pngs[@]} -eq 0 ]]; then
    echo "[smoke] WARNING: no figures under results/figures/main/"
fi

# 4) Vector-PDF sanity.
for p in "${pdfs[@]}"; do
    if ! file "$p" | grep -q "PDF document"; then
        echo "[smoke] FAIL: $p is not a valid PDF" >&2
        exit 1
    fi
done

echo "[smoke] OK — PPL: ${#csvs[@]} CSV(s), figures: ${#pdfs[@]} PDF(s) / ${#pngs[@]} PNG(s)"
