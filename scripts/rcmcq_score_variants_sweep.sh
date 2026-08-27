#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# rcmcq_score_variants_sweep.sh
#
# Sweep evals/eval_rcmcq_variants.py across a list of (repo, revision) pairs,
# sharded across N_GPUS ranks. Computes mean / sum / PMI-mean / PMI-sum
# accuracies for each (prompt, cefr) cell, then aggregates with
# analyze/rcmcq_rescore.py.
#
# Defaults target the FineWeb-2B b2/b3 nld-eng final checkpoints (the only
# two that are above HumanScale on the baseline RC-MCQ run).
#
# Usage
# -----
#   bash scripts/rcmcq_score_variants_sweep.sh
#   N_GPUS=8 bash scripts/rcmcq_score_variants_sweep.sh
#   PAIRS="Beetle-FineWeb-2B/...:step-5862 Beetle-HumanScale/...:main" \
#     bash scripts/rcmcq_score_variants_sweep.sh
#
# Each PAIR is "<repo>:<revision>". Use ":main" for the default branch.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate

OUT="${OUT:-results/evals_paper/03_rcmcq_variants}"
N_GPUS="${N_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PROMPTS="${PROMPTS:-default cefr_persona stratified gradient}"
CEFR_LEVELS="${CEFR_LEVELS:-A2 B1 B2 C1 C2}"
DATASET="${DATASET:-education/Cambridge Multiple-Choice Questions Reading Dataset.jsonl}"
LOG_DIR="${LOG_DIR:-/root/beetle_pipeline/logs/manual_stage18/rcmcq_variants_sweep}"

DEFAULT_PAIRS=(
    "Beetle-FineWeb-2B/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-2b-nld-eng:step-5862"
    "Beetle-FineWeb-2B/beetle-bilingual-l2-50-simultaneous-b2-fineweb-2b-nld-eng:step-5862"
    "Beetle-FineWeb-2B/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-2b-nld-eng:main"
    "Beetle-FineWeb-2B/beetle-bilingual-l2-50-simultaneous-b2-fineweb-2b-nld-eng:main"
    "Beetle-HumanScale/beetle-bilingual-l2-50-sequential-33-67-b3-humanscale-nld-eng:main"
    "Beetle-HumanScale/beetle-monolingual-humanscale-eng:main"
)
read -ra PAIRS <<< "${PAIRS:-${DEFAULT_PAIRS[*]}}"

mkdir -p "$LOG_DIR" "$OUT"
echo "[$(date -u +%FT%TZ)] starting variants sweep: ${#PAIRS[@]} pairs across $N_GPUS GPU(s)" \
    | tee -a "$LOG_DIR/dispatch.log"

pids=()
for ((rank=0; rank<N_GPUS; rank++)); do
    rank_pairs=()
    for ((j=rank; j<${#PAIRS[@]}; j+=N_GPUS)); do
        rank_pairs+=("${PAIRS[$j]}")
    done
    [[ ${#rank_pairs[@]} -eq 0 ]] && continue
    log="$LOG_DIR/rank${rank}.log"
    (
        for pair in "${rank_pairs[@]}"; do
            repo="${pair%:*}"; rev="${pair##*:}"
            echo "[$(date -u +%FT%TZ)] rank=$rank repo=$repo rev=$rev" | tee -a "$log"
            CUDA_VISIBLE_DEVICES=$rank RANK=0 WORLD_SIZE=1 \
                python -m evals.eval_rcmcq_variants \
                    --models "$repo" \
                    --revision "$rev" \
                    --dataset "$DATASET" \
                    --cefr-levels $CEFR_LEVELS \
                    --prompts $PROMPTS \
                    --output-root "$OUT" \
                    --batch-size "$BATCH_SIZE" \
                    >>"$log" 2>&1 || \
                    echo "[$(date -u +%FT%TZ)]   FAILED: $repo @ $rev" | tee -a "$log"
        done
    ) &
    pids+=($!)
    sleep 5
done

for pid in "${pids[@]}"; do wait "$pid" || true; done
echo "[$(date -u +%FT%TZ)] all ranks done; aggregating" | tee -a "$LOG_DIR/dispatch.log"

python -m analyze.rcmcq_rescore \
    --root "$OUT" \
    --merge-rcmcq results/evals_paper/03_rcmcq \
    | tee -a "$LOG_DIR/dispatch.log"

echo "[$(date -u +%FT%TZ)] DONE" | tee -a "$LOG_DIR/dispatch.log"
