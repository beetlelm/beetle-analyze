#!/usr/bin/env bash
# =============================================================================
# scripts/run_beetlelm_aoa.sh
#
# End-to-end 8xA100 final-checkpoint Age-of-Acquisition evaluation.
#
# Pipeline
# --------
#   Phase A (rank 0 only):  generate every (lang, tokenizer) eval jsonl that
#                           the subsequent workers will consume. Done serially
#                           so the Hub downloads happen once.
#   Phase B (8 workers   ): each GPU runs `evals/eval_aoa.py` with a different
#                           rank; models are round-robin striped across ranks.
#   Phase C (rank 0 only):  consume the aggregated CSV and emit the final
#                           LaTeX table.
#
# Outputs
# -------
#   ${OUTPUT_DIR}/aoa_results.csv           — one row per (model, lang)
#   ${OUTPUT_DIR}/beetlelm_aoa_table.tex    — LaTeX wide table
#   ${OUTPUT_DIR}/logs/aoa_*_rank{0..7}.log — per-rank logs
#
# Usage
# -----
#   bash scripts/run_beetlelm_aoa.sh                          # defaults
#   OUTPUT_DIR=results/evals/aoa WORLD_SIZE=8 \
#       bash scripts/run_beetlelm_aoa.sh
#   REGIME=fineweb BATCH_SIZE=16 bash scripts/run_beetlelm_aoa.sh
#   DRY_RUN=1 bash scripts/run_beetlelm_aoa.sh                # plan only
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_DIR="${OUTPUT_DIR:-results/evals/aoa}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-results/aoa/eval_data}"
REGIME="${REGIME:-humanscale}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WORLD_SIZE="${WORLD_SIZE:-8}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON="${PYTHON:-python3}"

MODELS_SPEC=(${MODELS_SPEC:-beetle-humanscale beetle-fineweb baselines})

RESULTS_CSV="${OUTPUT_DIR}/aoa_results.csv"
LATEX_OUT="${OUTPUT_DIR}/beetlelm_aoa_table.tex"

mkdir -p "${OUTPUT_DIR}" "${EVAL_DATA_DIR}"

echo "============================================================"
echo "  Beetle AoA (final checkpoint)"
echo "============================================================"
echo "  output       : ${OUTPUT_DIR}"
echo "  eval data    : ${EVAL_DATA_DIR}"
echo "  regime       : ${REGIME}"
echo "  world size   : ${WORLD_SIZE}"
echo "  batch size   : ${BATCH_SIZE}"
echo "  models       : ${MODELS_SPEC[*]}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Phase A — rank 0 eval-data preparation
# ---------------------------------------------------------------------------

echo "[1/3] Preparing eval data on rank 0 …"
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "      DRY_RUN=1 — skipping."
else
    ${PYTHON} evals/eval_aoa.py \
        --models "${MODELS_SPEC[@]}" \
        --regime "${REGIME}" \
        --output "${OUTPUT_DIR}" \
        --eval-data-dir "${EVAL_DATA_DIR}" \
        --batch-size "${BATCH_SIZE}" \
        --rank 0 --world-size "${WORLD_SIZE}" \
        --prep-only
fi

# ---------------------------------------------------------------------------
# Phase B — 8 workers in parallel (one per GPU)
# ---------------------------------------------------------------------------

echo "[2/3] Dispatching ${WORLD_SIZE} worker(s) …"
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "      DRY_RUN=1 — skipping workers."
else
    pids=()
    for (( i=0; i<WORLD_SIZE; i++ )); do
        CUDA_VISIBLE_DEVICES="$i" \
        ${PYTHON} evals/eval_aoa.py \
            --models "${MODELS_SPEC[@]}" \
            --regime "${REGIME}" \
            --output "${OUTPUT_DIR}" \
            --eval-data-dir "${EVAL_DATA_DIR}" \
            --batch-size "${BATCH_SIZE}" \
            --rank "$i" --world-size "${WORLD_SIZE}" \
            --skip-prep \
            > "${OUTPUT_DIR}/logs/worker_rank${i}.out" 2>&1 &
        pids+=($!)
        echo "      launched rank $i (pid ${pids[$i]}) on CUDA_VISIBLE_DEVICES=$i"
    done
    mkdir -p "${OUTPUT_DIR}/logs"

    # Wait for every worker; abort on any failure.
    fail=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "      WORKER FAILED (pid=$pid)" >&2
            fail=1
        fi
    done
    if [[ "$fail" -ne 0 ]]; then
        echo "ERROR: at least one worker failed — see ${OUTPUT_DIR}/logs/worker_rank*.out" >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Phase C — emit LaTeX table
# ---------------------------------------------------------------------------

echo "[3/3] Writing LaTeX table → ${LATEX_OUT}"
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "      DRY_RUN=1 — skipping."
    exit 0
fi

if [[ ! -f "${RESULTS_CSV}" ]]; then
    echo "      No ${RESULTS_CSV} produced — nothing to tabulate." >&2
    exit 0
fi

${PYTHON} scripts/make_beetlelm_aoa_table.py \
    --input "${RESULTS_CSV}" \
    --output "${LATEX_OUT}"

echo ""
echo "Done."
echo "  Results CSV : ${RESULTS_CSV}"
echo "  LaTeX table : ${LATEX_OUT}"
