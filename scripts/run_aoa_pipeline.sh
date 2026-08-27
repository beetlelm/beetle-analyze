#!/bin/bash
# =============================================================================
# run_aoa_pipeline.sh — Full Chang & Bergen (2022) AoA pipeline for BeetleLM
#
# Drives the two training regimes (human-scale BabyBabel and FineWeb) over
# the priority languages (deu, nld, zho by default), then the full AoA
# language set when given LANGS=all.
#
# Prereqs:
#   * beetle-data repo is a sibling of beetle-analyze and held-out Parquet
#     shards have been produced via either:
#       - scripts/launch_held_out_babybabel.sh   (humanscale regime)
#       - scripts/launch_held_out.sh             (fineweb regime)
#   * HF_TOKEN is exported if the model repos are gated.
#
# Usage:
#   bash scripts/run_aoa_pipeline.sh                              # deu/nld/zho, both regimes
#   REGIMES=humanscale LANGS="deu" bash scripts/run_aoa_pipeline.sh
#   MODELS="Beetle-HumanScale/beetlelm_nld_L1-eng_L2_balanced" \\
#       LANGS=deu REGIMES=humanscale bash scripts/run_aoa_pipeline.sh
#
# Env vars:
#   LANGS                 — space-sep ISO-3 codes, or "all" (default: "deu nld zho")
#   REGIMES               — "humanscale", "fineweb", or both (default: "humanscale fineweb")
#   MODELS                — explicit model repo(s) (default: auto-discover)
#   OUTPUT_DIR            — pipeline output base (default: results/aoa)
#   HELD_OUT_BB_DIR       — BabyBabel held-out base (default: ../beetle-data/pipeline_output/held_out_babybabel)
#   HELD_OUT_FW_DIR       — FineWeb held-out base (default: ../beetle-data/pipeline_output/held_out)
#   MAX_CHECKPOINTS       — log-spaced checkpoint count (default: 200)
#   N_CONTEXTS            — contexts per word (default: 512)
#   DEBUG                 — "1" → single model per language
# =============================================================================

set -euo pipefail

LANGS="${LANGS:-deu nld zho}"
REGIMES="${REGIMES:-humanscale fineweb}"
MODELS="${MODELS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-results/aoa}"
HELD_OUT_BB_DIR="${HELD_OUT_BB_DIR:-../beetle-data/pipeline_output/held_out_babybabel}"
HELD_OUT_FW_DIR="${HELD_OUT_FW_DIR:-../beetle-data/pipeline_output/held_out}"
MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-200}"
N_CONTEXTS="${N_CONTEXTS:-512}"
DEBUG="${DEBUG:-0}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

extra_flags=()
if [ "$DEBUG" = "1" ]; then
    extra_flags+=(--debug)
fi
if [ -n "$MODELS" ]; then
    # shellcheck disable=SC2206
    model_args=(--model $MODELS)
else
    model_args=()
fi

for regime in $REGIMES; do
    echo "============================================================"
    echo "AoA pipeline: regime=$regime | langs=$LANGS"
    echo "============================================================"

    case "$regime" in
        humanscale) held_out_base="$HELD_OUT_BB_DIR" ;;
        fineweb)    held_out_base="$HELD_OUT_FW_DIR" ;;
        *)
            echo "Unknown regime: $regime" >&2
            exit 1
            ;;
    esac

    python3 -m aoa.run_learning_curves \
        --regime "$regime" \
        --langs $LANGS \
        --output-dir "$OUTPUT_DIR" \
        --held-out-base "$held_out_base" \
        --n-contexts "$N_CONTEXTS" \
        --max-checkpoints "$MAX_CHECKPOINTS" \
        "${model_args[@]}" "${extra_flags[@]}"

    # Per-language Part-2 analyses (predictors → regression → child → rankings).
    for lang in $LANGS; do
        if [ "$lang" = "all" ]; then
            continue
        fi
        eval_base="$OUTPUT_DIR/eval_data"
        # Pick the one eval JSONL produced for this regime+lang.
        eval_jsonl=$(ls "$eval_base"/"${lang}"_*.jsonl 2>/dev/null | head -n1 || true)
        if [ -z "$eval_jsonl" ]; then
            echo "No eval JSONL for $lang; skipping Part-2 analyses"
            continue
        fi

        predictors_csv="$OUTPUT_DIR/predictors/${lang}_predictors.csv"
        regression_json="$OUTPUT_DIR/regression/${lang}_${regime}_regression.json"
        rankings_json="$OUTPUT_DIR/rankings/${lang}_${regime}_rankings.json"
        child_json="$OUTPUT_DIR/human_vs_lm/${lang}_${regime}.json"

        echo ">>> Predictors for $lang"
        python3 -m aoa.predictors \
            --lang "$lang" \
            --eval-jsonl "$eval_jsonl" \
            --held-out-dir "$held_out_base/$lang" \
            --output-dir "$OUTPUT_DIR/predictors"

        # Find the latest word_metrics.csv for this regime/lang
        wm_csv=$(find "$OUTPUT_DIR/analysis/$regime" -path "*/${lang}/word_metrics.csv" 2>/dev/null | head -n1 || true)
        if [ -z "$wm_csv" ]; then
            echo "No word_metrics.csv found for $lang under regime $regime; skipping regression/rankings"
            continue
        fi

        echo ">>> Regression for $lang ($regime)"
        python3 -m aoa.regress_aoa \
            --predictors "$predictors_csv" \
            --word-metrics "$wm_csv" \
            --aoa-col aoa_paper \
            --output "$regression_json" || echo "regression failed for $lang"

        echo ">>> Rankings / final-perf robustness for $lang ($regime)"
        python3 -m aoa.compare_rankings \
            --word-metrics "$wm_csv" \
            --predictors "$predictors_csv" \
            --aoa-col aoa_paper \
            --output "$rankings_json" || echo "rankings failed for $lang"

        echo ">>> CHILDES comparison for $lang"
        python3 -m aoa.child_comparison \
            --lang "$lang" \
            --predictors "$predictors_csv" \
            --lm-regression "$regression_json" \
            --output "$child_json" || echo "child comparison failed for $lang"
    done

done

echo "AoA pipeline complete. Outputs in $OUTPUT_DIR/"
