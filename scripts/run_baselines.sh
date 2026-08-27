#!/usr/bin/env bash
# run_baselines.sh
# -----------------
# Compute MECO alignment + BLiMP significance for every baseline model
# registered in models.py: B-GPT, bilingual LLMs (CroissantLLM, MAP-Neo,
# Rinna), the BLOOM size ladder (560M -> 7B), and EuroLLM (1.7B, 9B).
#
# Beetle checkpoints are included too — they are the reference group for
# the Fisher z-test, mixed-effects model, and Wilcoxon robustness tests,
# so the cross-group comparison is produced in the SAME run.
#
# Outputs
# -------
#   results/meco_baselines/alignment_per_model.csv
#   results/meco_baselines/fisher_ztest.csv
#   results/meco_baselines/mixedlm_summary.csv
#   results/meco_baselines/wilcoxon_paired.csv
#   results/meco_baselines/friedman_by_language.csv
#   results/meco_baselines/claim.txt
#   results/figures/10_meco_correlation_bar.png ... 17_meco_eurollm_vs_beetle.png
#   results/blimp_significance/{blimp_eng,blimp_nl,zhoblimp}_{paired,friedman,wilcoxon_pairwise,mixedlm_summary}.*
#   results/figures/06_blimp_effects_*.png ... 09_blimp_phenomenon_by_model_*.png
#
# Usage
# -----
#   bash script/run_baselines.sh                     # default: EN + NL
#   bash script/run_baselines.sh --langs en nl es pl # custom languages
#   bash script/run_baselines.sh --mock              # synthetic data smoke test
#   bash script/run_baselines.sh --large             # enable BLOOM-176B/GLM-130B
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

# -- argument parsing ------------------------------------------------------
LANGS=(en nl)
EXTRA_FLAGS=()
MOCK=""
LARGE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --langs)
            shift
            LANGS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                LANGS+=("$1"); shift
            done
            ;;
        --mock)   MOCK="--mock";   shift ;;
        --large)  LARGE="--large"; shift ;;
        --skip-blimp) SKIP_BLIMP=1; shift ;;
        *)        EXTRA_FLAGS+=("$1"); shift ;;
    esac
done

OUT_MECO="results/meco_baselines"
OUT_BLIMP="results/blimp_significance"
mkdir -p "$OUT_MECO" "$OUT_BLIMP" "results/figures"

export PYTHONPATH="${ROOT}:${ROOT}/analyze${PYTHONPATH:+:${PYTHONPATH}}"
PY="${PYTHON:-python3}"

# -- 1. MECO alignment over every baseline group --------------------------
echo "[run_baselines] MECO alignment — languages: ${LANGS[*]}"
"$PY" -m analyze.run_meco_comparison \
    --languages "${LANGS[@]}" \
    --include-bgpt \
    --include-bilingual-llm \
    --include-bloom-ladder \
    --include-eurollm \
    --include-multilingual-llm \
    $LARGE $MOCK \
    --output "$OUT_MECO" \
    ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}

echo "[run_baselines] MECO results:"
ls -1 "$OUT_MECO" 2>/dev/null | sed 's/^/    /'

# -- 2. BLiMP / BLiMP-NL / ZhoBLiMP significance --------------------------
if [[ -z "${SKIP_BLIMP:-}" ]]; then
    echo "[run_baselines] BLiMP / BLiMP-NL / ZhoBLiMP significance tests"
    "$PY" -m analyze.run_blimp_significance \
        --tasks blimp_eng blimp_nl zhoblimp \
        --monolingual-baseline goldfish \
        --output "$OUT_BLIMP" || \
        echo "[run_baselines] BLiMP step skipped (missing per-phenomenon CSVs)."
fi

echo
echo "[run_baselines] Done. Key artefacts:"
echo "    $OUT_MECO/claim.txt   (auto-filled paper claim paragraph)"
echo "    results/figures/10_meco_correlation_bar.png ... 17_meco_eurollm_vs_beetle.png"
echo "    results/figures/06_blimp_effects_*.png ... 09_blimp_phenomenon_by_model_*.png"
