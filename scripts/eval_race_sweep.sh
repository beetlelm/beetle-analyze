#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# eval_race_sweep.sh
#
# Sweep evals/eval_race_student.py over a checkpoint list of one Beetle
# repo, sharded across N_GPUS ranks. For each checkpoint, all 5 student
# personas are scored and a single row (acc_1..acc_5, corr, penalty, M)
# is appended to results/evals/race/race_<slug>.csv. After all ranks
# finish, plots/make_race_sweep.py renders the M-vs-step curve.
#
# Default repo + checkpoints come from the source notebook
# (Beetle-FineWeb2-24B late-binding b5 deu-eng). Override with REPO= and
# CHECKPOINTS=.
#
# Usage
# -----
#   bash scripts/eval_race_sweep.sh
#   N_GPUS=8 bash scripts/eval_race_sweep.sh
#   REPO=Beetle-FineWeb/some-other-model \
#     CHECKPOINTS="main step-512 step-2048" \
#     bash scripts/eval_race_sweep.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."

OUT="${OUT:-results}"
N_GPUS="${N_GPUS:-8}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${TMPDIR:-/tmp}/beetle_hf_cache}"
SKIP_PREFETCH="${SKIP_PREFETCH:-0}"
RANK_LAUNCH_DELAY="${RANK_LAUNCH_DELAY:-5}"

REPO="${REPO:-Beetle-FineWeb2-24B/beetle-bilingual-l2-80-late-b5-fineweb-deu-eng}"
MAX_QUESTIONS="${MAX_QUESTIONS:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"

# Default checkpoint list = the source notebook's 17 step-N entries.
DEFAULT_CHECKPOINTS=(
    "step-73241" "step-73242" "step-73243" "step-73245"
    "step-73247" "step-73249" "step-73257" "step-73273"
    "step-73305" "step-73369" "step-73497" "step-73753"
    "step-74265" "step-75289" "step-77337" "step-81433"
    "step-89625"
)
if [[ -n "${CHECKPOINTS:-}" ]]; then
    # Allow space-separated string from env.
    read -r -a CKPTS <<< "$CHECKPOINTS"
else
    CKPTS=("${DEFAULT_CHECKPOINTS[@]}")
fi

export HF_HOME="$HF_CACHE_ROOT"
export TRANSFORMERS_CACHE="$HF_CACHE_ROOT"
export HF_HUB_CACHE="$HF_CACHE_ROOT"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "[$(date -u +%FT%TZ)] === RACE student-simulation sweep ==="
echo "[$(date -u +%FT%TZ)] REPO=${REPO}"
echo "[$(date -u +%FT%TZ)] N_GPUS=${N_GPUS}  CHECKPOINTS=${#CKPTS[@]}  MAX_Q=${MAX_QUESTIONS}"

mkdir -p "$HF_CACHE_ROOT" "$OUT/logs" "$OUT/ledger" "$OUT/evals/race"

# ─── Stage 0.5 — serial pre-fetch of all referenced revisions ───────────────
if [[ "$SKIP_PREFETCH" != "1" ]]; then
    echo "[$(date -u +%FT%TZ)] Stage 0.5 — pre-fetch ${#CKPTS[@]} revisions (serial)"
    for rev in "${CKPTS[@]}"; do
        echo "  [$(date -u +%FT%TZ)] fetching $REPO @ $rev"
        python - <<PY >> "$OUT/logs/prefetch_race.log" 2>&1
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${REPO}",
    revision="${rev}",
    allow_patterns=["*.json", "*.txt", "*.model", "tokenizer.*", "*.safetensors", "*.bin"],
    ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx", "*.mlmodel"],
    max_workers=2,
)
print("OK ${REPO}@${rev}")
PY
        if [[ $? -ne 0 ]]; then
            echo "    WARNING: prefetch of $REPO@$rev failed; see $OUT/logs/prefetch_race.log" >&2
        fi
    done
    echo "[$(date -u +%FT%TZ)] Stage 0.5 — prefetch complete"
else
    echo "[$(date -u +%FT%TZ)] Stage 0.5 — SKIP prefetch (SKIP_PREFETCH=1)"
fi

_stagger() {
    local rank="$1"
    if (( rank > 0 && RANK_LAUNCH_DELAY > 0 )); then
        sleep "$RANK_LAUNCH_DELAY"
    fi
}

# ─── Stage 1 — per-rank sweep over the checkpoint slice ─────────────────────
echo "[$(date -u +%FT%TZ)] Stage 1 — RACE sweep (${N_GPUS} ranks)"
declare -a stage1_pids=()
for RANK in $(seq 0 $((N_GPUS-1))); do
    _stagger "$RANK"
    CUDA_VISIBLE_DEVICES="$RANK" \
    setsid python evals/eval_race_student.py \
        --repo "$REPO" \
        --revisions "${CKPTS[@]}" \
        --max-questions "$MAX_QUESTIONS" \
        --batch-size "$BATCH_SIZE" \
        --rank "$RANK" --world-size "$N_GPUS" \
        --output-root "$OUT" \
        </dev/null > "$OUT/logs/race_rank${RANK}.log" 2>&1 &
    stage1_pids+=($!)
done
for pid in "${stage1_pids[@]}"; do wait "$pid" || true; done

# ─── Stage 5 — merge ledger + plot ──────────────────────────────────────────
echo "[$(date -u +%FT%TZ)] Stage 5 — ledger merge + plot"
python -m analyze.eval_ledger merge --output "$OUT" \
    > "$OUT/logs/ledger_merge_race.log" 2>&1 || true
REPO="$REPO" python plots/make_race_sweep.py \
    > "$OUT/logs/make_race_sweep.log" 2>&1 || \
    echo "    (non-fatal) plot failed; see $OUT/logs/make_race_sweep.log"

echo "[$(date -u +%FT%TZ)] === RACE sweep done ==="
echo "  CSV:  $OUT/evals/race/race_$(echo "$REPO" | tr '/' '__').csv"
echo "  Plot: plots/race_sweep_$(echo "$REPO" | tr '/' '__').pdf"
