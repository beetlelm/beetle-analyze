#!/usr/bin/env bash
# Run BLiSS on the FineWeb-2B b1..b5 + monolingual-humanscale-eng so the L2-fit
# headline figure has all the same models as the §4 sweep. HumanScale b1-b5
# nld-eng BLiSS already exists in results/evals_checkpoints/bliss/bliss_results.csv.
set -uo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

OUT="${OUT:-results/evals_paper/04e_bliss_pedagogical}"
LOG_DIR="${LOG_DIR:-/root/beetle_pipeline/logs/manual_pedagogical/stage4e_bliss}"
mkdir -p "$OUT" "$LOG_DIR"

PAIRS=(
    "Beetle-FineWeb-2B/beetle-bilingual-balanced-b1-fineweb-2b-nld-eng:main"
    "Beetle-FineWeb-2B/beetle-bilingual-l2-50-simultaneous-b2-fineweb-2b-nld-eng:main"
    "Beetle-FineWeb-2B/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-2b-nld-eng:main"
    "Beetle-FineWeb-2B/beetle-bilingual-l2-50-classroom-20-b4-fineweb-2b-nld-eng:main"
    "Beetle-FineWeb-2B/beetle-bilingual-l2-80-late-b5-fineweb-2b-nld-eng:main"
    "Beetle-HumanScale/beetle-monolingual-humanscale-eng:main"
)

ts() { date -u +%FT%TZ; }
log() { echo "[$(ts)] $*"; }

log "BLiSS pedagogical grid: ${#PAIRS[@]} pairs across 8 GPUs"

pids=()
N_GPUS=8
for ((rank=0; rank<N_GPUS; rank++)); do
    rank_pairs=()
    for ((j=rank; j<${#PAIRS[@]}; j+=N_GPUS)); do
        rank_pairs+=("${PAIRS[$j]}")
    done
    [[ ${#rank_pairs[@]} -eq 0 ]] && continue
    lf="$LOG_DIR/rank${rank}.log"
    (
        for pair in "${rank_pairs[@]}"; do
            repo="${pair%:*}"; rev="${pair##*:}"
            echo "[$(ts)] rank=$rank $repo @ $rev" >> "$lf"
            CUDA_VISIBLE_DEVICES=$rank \
                python -m evals.eval_bliss \
                    --model "$repo" \
                    --revision "$rev" \
                    --split train \
                    --batch-size 32 \
                    --output "$OUT" \
                    --langs en nl \
                    >> "$lf" 2>&1 || \
                echo "[$(ts)]   FAILED: $repo @ $rev" >> "$lf"
        done
    ) &
    pids+=($!)
    sleep 3
done
for pid in "${pids[@]}"; do wait "$pid" || true; done
log "BLiSS pedagogical grid done"
