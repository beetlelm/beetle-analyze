#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# race_zeroshot_for_rc.sh
#
# Run evals/eval_race_student.py across the same Beetle (repo, revision)
# pairs we use for RC-MCQ variants. RACE is the canonical English RC-MCQ
# benchmark; if Beetle is at chance there as well as on Cambridge, the
# at-chance result is general (a scale-boundary claim). If RACE is above
# chance for some checkpoint, that's the strongest candidate to start a
# RACE→Cambridge SFT transfer experiment from.
#
# Usage
# -----
#   bash scripts/race_zeroshot_for_rc.sh
#   N_GPUS=8 bash scripts/race_zeroshot_for_rc.sh
#   MAX_QUESTIONS=200 bash scripts/race_zeroshot_for_rc.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate

OUT="${OUT:-results/evals_paper/03b_race_zeroshot}"
N_GPUS="${N_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_QUESTIONS="${MAX_QUESTIONS:-256}"
LOG_DIR="${LOG_DIR:-/root/beetle_pipeline/logs/manual_stage18/race_zeroshot}"

DEFAULT_PAIRS=(
    "Beetle-FineWeb-2B/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-2b-nld-eng:step-5862"
    "Beetle-FineWeb-2B/beetle-bilingual-l2-50-simultaneous-b2-fineweb-2b-nld-eng:step-5862"
    "Beetle-HumanScale/beetle-monolingual-humanscale-eng:main"
)
read -ra PAIRS <<< "${PAIRS:-${DEFAULT_PAIRS[*]}}"

mkdir -p "$LOG_DIR" "$OUT"
echo "[$(date -u +%FT%TZ)] race-zeroshot: ${#PAIRS[@]} pairs across $N_GPUS GPU(s); max_q=$MAX_QUESTIONS" \
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
            CUDA_VISIBLE_DEVICES=$rank \
                python -m evals.eval_race_student \
                    --repo "$repo" \
                    --revisions "$rev" \
                    --output-root "$OUT" \
                    --max-questions "$MAX_QUESTIONS" \
                    --batch-size "$BATCH_SIZE" \
                    >>"$log" 2>&1 || \
                    echo "[$(date -u +%FT%TZ)]   FAILED: $repo @ $rev" | tee -a "$log"
        done
    ) &
    pids+=($!)
    sleep 5
done

for pid in "${pids[@]}"; do wait "$pid" || true; done
echo "[$(date -u +%FT%TZ)] DONE" | tee -a "$LOG_DIR/dispatch.log"
