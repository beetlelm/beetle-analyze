#!/usr/bin/env bash
# run_beetle.sh
# --------------
# Beetle-specific analyses:
#
# 1. MECO alignment for every registered Beetle model (no external
#    baselines). Produces per-(model, language) Spearman rho, OLS beta,
#    lexical-control beta, and the publication figures restricted to the
#    Beetle group.
#
# 2. Within-family significance selection for the B1 / B2 / B3 ablation
#    sweep produced by beetlelm/script/ablation_minimal_nld_eng.sh —
#    paired permutation test on MECO alignment, Cohen's d_z, Holm-Bonferroni
#    correction WITHIN family.
#
# 3. BLiMP / BLiMP-NL / ZhoBLiMP significance testing for Beetle models
#    (same CSVs as the baseline script; the filter on "Beetle only" is
#    applied by blimp_significance.analyse_task when the bilingual CSV
#    already contains only Beetle rows, or can be narrowed via --task-csv
#    overrides).
#
# Expected manifest (default location) from the ablation sweep:
#   ../beetlelm/logs_ablation_nld_eng/variants_manifest.csv
#
# Outputs
# -------
#   results/meco_beetle/alignment_per_model.csv
#   results/meco_beetle/fisher_ztest.csv               (ref=small_bilingual)
#   results/meco_beetle/mixedlm_summary.csv
#   results/meco_beetle/claim.txt
#   results/ablation_significance_B1.json
#   results/ablation_significance_B2.json
#   results/ablation_significance_B3.json
#   results/blimp_significance/*
#
# Usage
# -----
#   bash script/run_beetle.sh
#   bash script/run_beetle.sh --manifest /path/to/variants_manifest.csv
#   bash script/run_beetle.sh --mock
#   bash script/run_beetle.sh --langs en nl
#   bash script/run_beetle.sh --skip-ablation     # only MECO + BLiMP
#   bash script/run_beetle.sh --skip-blimp        # only MECO + ablation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

# -- argument parsing ------------------------------------------------------
LANGS=(en nl)
MANIFEST_DEFAULT="${ROOT}/../beetlelm/logs_ablation_nld_eng/variants_manifest.csv"
MANIFEST="$MANIFEST_DEFAULT"
MOCK=""
SKIP_ABLATION=""
SKIP_BLIMP=""
EXTRA_FLAGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --langs)
            shift; LANGS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do LANGS+=("$1"); shift; done
            ;;
        --manifest)     MANIFEST="$2"; shift 2 ;;
        --mock)         MOCK="--mock"; shift ;;
        --skip-ablation) SKIP_ABLATION=1; shift ;;
        --skip-blimp)   SKIP_BLIMP=1; shift ;;
        *)              EXTRA_FLAGS+=("$1"); shift ;;
    esac
done

OUT_MECO="results/meco_beetle"
OUT_BLIMP="results/blimp_significance"
OUT_ABL="results"
mkdir -p "$OUT_MECO" "$OUT_BLIMP" "results/figures"

export PYTHONPATH="${ROOT}:${ROOT}/analyze${PYTHONPATH:+:${PYTHONPATH}}"
PY="${PYTHON:-python3}"

# -- 1. MECO alignment over Beetle models only ----------------------------
echo "[run_beetle] MECO alignment — Beetle models, languages: ${LANGS[*]}"
"$PY" -m analyze.run_meco_comparison \
    --languages "${LANGS[@]}" \
    $MOCK \
    --output "$OUT_MECO" \
    ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}

echo "[run_beetle] Beetle MECO results:"
ls -1 "$OUT_MECO" 2>/dev/null | sed 's/^/    /'

# -- 2. B1/B2/B3 within-family significance selection ---------------------
if [[ -z "${SKIP_ABLATION}" ]]; then
    if [[ ! -f "$MANIFEST" ]]; then
        echo "[run_beetle] No manifest at $MANIFEST"
        echo "            Run beetlelm/script/ablation_minimal_nld_eng.sh first,"
        echo "            or pass --manifest /path/to/variants_manifest.csv."
        if [[ -n "$MOCK" ]]; then
            echo "[run_beetle] --mock: synthesising a manifest with mock metrics."
            export BEETLE_MOCK_METRICS=1
            MANIFEST="$(mktemp -t variants_manifest_XXXXXX.csv)"
            {
                echo "variant,seed,config_path,log_path,hf_push_suffix"
                for v in B1 B1b B2 B2_ewc B2_lamol B2_lr B3 B3_full; do
                    for s in 13 42 97; do
                        echo "$v,$s,cfg/$v.yaml,log/${v}_seed${s}.log,seed${s}"
                    done
                done
            } > "$MANIFEST"
        fi
    fi
    if [[ -f "$MANIFEST" ]]; then
        echo "[run_beetle] Within-family selection on $MANIFEST"
        "$PY" -m analyze.significance_ablation \
            --variants-file "$MANIFEST" \
            --all-families \
            --output-dir "$OUT_ABL" \
            ${MOCK:+--mock-metrics}
    fi
fi

# -- 3. BLiMP / BLiMP-NL / ZhoBLiMP on Beetle models ----------------------
if [[ -z "${SKIP_BLIMP}" ]]; then
    echo "[run_beetle] BLiMP / BLiMP-NL / ZhoBLiMP significance tests"
    "$PY" -m analyze.run_blimp_significance \
        --tasks blimp_eng blimp_nl zhoblimp \
        --monolingual-baseline goldfish \
        --output "$OUT_BLIMP" || \
        echo "[run_beetle] BLiMP step skipped (missing per-phenomenon CSVs)."
fi

echo
echo "[run_beetle] Done. Key artefacts:"
echo "    $OUT_MECO/claim.txt"
[[ -z "$SKIP_ABLATION" ]] && \
    echo "    results/ablation_significance_{B1,B2,B3}.json   (winners per family)"
echo "    results/figures/10_meco_*.png ... 17_meco_*.png"
echo "    results/figures/06_blimp_*.png ... 09_blimp_*.png"
