#!/usr/bin/env bash
# evals/run_evals.sh — Run all three benchmark evaluations across 8×A100 GPUs.
#
# Each evaluation stage fans out across N_GPUS parallel processes.
# Models are sharded by index: rank i scores models[i::N_GPUS].
# All ranks run concurrently; the script waits for all before moving on.
#
# Ledger stages (rank 0 only):
#   Stage 0 — backfill: imports any existing CSVs into the ledger so re-runs
#             can skip already-completed evals.
#   Stage 4 — merge:  consolidates per-rank JSONL shards into ledger.json.
#   Stage 5 — report: prints a completion table per benchmark × model family.
#
# Usage:
#   bash evals/run_evals.sh                        # all stages, 8 GPUs
#   N_GPUS=4 bash evals/run_evals.sh               # 4 GPUs
#   STAGES="blimp flores" bash evals/run_evals.sh  # skip meco
set -euo pipefail
cd "$(dirname "$0")/.."

# ── Config ────────────────────────────────────────────────────────────────────
N_GPUS="${N_GPUS:-8}"
STAGES="${STAGES:-blimp meco flores}"   # space-separated subset to run
OUT_ROOT="results/evals"
LOG_DIR="${OUT_ROOT}/logs"
# Model weights are shared across ranks (read-only after first download).
# Dataset tokenisation cache is per-rank to avoid concurrent write conflicts.
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${TMPDIR:-/tmp}/beetle_hf_cache}"
HF_HOME_SHARED="${HF_CACHE_ROOT}/shared"

mkdir -p "$LOG_DIR" "$HF_CACHE_ROOT" "$HF_HOME_SHARED"

echo "[$(date -u +%FT%TZ)] N_GPUS=${N_GPUS}  STAGES=${STAGES}"

# ── GPU utilisation monitor ────────────────────────────────────────────────────
nvidia-smi dmon -s u -d 10 > "${LOG_DIR}/gpu_util.log" 2>&1 &
NVMON_PID=$!
echo "[$(date -u +%FT%TZ)] GPU monitor PID=${NVMON_PID} → ${LOG_DIR}/gpu_util.log"

# Ensure the monitor is killed when the script exits for any reason
trap 'kill $NVMON_PID 2>/dev/null || true' EXIT

# ── Stage 0: backfill ledger from any existing CSVs ───────────────────────────
# Idempotent — marks already-present results as done so subsequent stages skip.
echo ""
echo "[$(date -u +%FT%TZ)] === Stage 0: ledger backfill ==="
HF_HOME="${HF_HOME_SHARED}" \
python -m analyze.eval_ledger backfill --output "${OUT_ROOT}" \
    2>&1 | tee "${LOG_DIR}/ledger_backfill.log" || true

# ── Helper: fan out a command across N_GPUS ranks ─────────────────────────────
# Usage: fan_out <log_prefix> <script_name> [extra args...]
# Appends --rank $RANK --world-size $N_GPUS to each invocation.
# Shared HF_HOME (model weights cache) to avoid re-downloading across ranks.
# Per-rank HF_DATASETS_CACHE to avoid concurrent tokenisation write conflicts.
fan_out() {
    local log_prefix="$1"; shift
    local -a pids=()
    for RANK in $(seq 0 $((N_GPUS - 1))); do
        CUDA_VISIBLE_DEVICES="$RANK" \
        HF_HOME="${HF_HOME_SHARED}" \
        HF_DATASETS_CACHE="${HF_CACHE_ROOT}/rank_${RANK}/datasets" \
        python evals/"$@" \
            --rank "$RANK" --world-size "$N_GPUS" \
            >> "${LOG_DIR}/${log_prefix}_rank${RANK}.log" 2>&1 &
        pids+=($!)
    done
    echo "  [$(date -u +%FT%TZ)] Launched ${#pids[@]} ranks — waiting …"
    local failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || { echo "  rank process $pid failed"; failed=$((failed + 1)); }
    done
    if [[ $failed -gt 0 ]]; then
        echo "ERROR: $failed rank(s) failed for ${log_prefix}" >&2
        exit 1
    fi
    echo "  [$(date -u +%FT%TZ)] All ranks done for ${log_prefix}"
}

# ── Stage 1: BLiMP ────────────────────────────────────────────────────────────
if [[ "$STAGES" == *blimp* ]]; then
    echo ""
    echo "[$(date -u +%FT%TZ)] === Stage 1: BLiMP / BLiMP-NL / ZhoBLIMP / MultiBLiMP ==="
    fan_out "blimp" eval_blimp.py \
        --models beetle-humanscale beetle-fineweb baselines \
        --output "${OUT_ROOT}/blimp"
fi

# ── Stage 2: MECO ─────────────────────────────────────────────────────────────
if [[ "$STAGES" == *meco* ]]; then
    echo ""
    echo "[$(date -u +%FT%TZ)] === Stage 2: MECO eye-tracking surprisal ==="
    fan_out "meco" eval_meco.py \
        --models beetle-humanscale beetle-fineweb baselines \
        --stimuli data/meco_l2_stims.tsv \
        --output "${OUT_ROOT}/meco"
fi

# ── Stage 3: FLORES PPL ───────────────────────────────────────────────────────
if [[ "$STAGES" == *flores* ]]; then
    echo ""
    echo "[$(date -u +%FT%TZ)] === Stage 3: FLORES-200 perplexity ==="
    fan_out "flores" eval_flores_ppl.py \
        --models beetle-humanscale beetle-fineweb baselines \
        --baseline-langs eng nld deu zho \
        --output "${OUT_ROOT}/ppl"
fi

# ── Stage 4: merge per-rank ledger shards ─────────────────────────────────────
echo ""
echo "[$(date -u +%FT%TZ)] === Stage 4: merge ledger shards ==="
HF_HOME="${HF_HOME_SHARED}" \
python -m analyze.eval_ledger merge --output "${OUT_ROOT}" \
    2>&1 | tee "${LOG_DIR}/ledger_merge.log" || true

# ── Stage 5: completion report ────────────────────────────────────────────────
echo ""
echo "[$(date -u +%FT%TZ)] === Stage 5: completion report ==="
HF_HOME="${HF_HOME_SHARED}" \
python -m analyze.eval_ledger report --output "${OUT_ROOT}" \
    2>&1 | tee "${LOG_DIR}/ledger_report.log" || true

echo ""
echo "[$(date -u +%FT%TZ)] === All evaluations completed successfully ==="
echo "Results under ${OUT_ROOT}/"
