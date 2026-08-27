#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# pedagogical_utility_sweep.sh
#
# Master orchestrator for the pedagogical-utility study. Runs the
# investigation in 4 stages, each gated by a STAGE flag so individual
# stages can be re-run without redoing the rest.
#
#   STAGE_VARIANTS=1   variant rescore over chosen base ckpts (cheap; ~15 min/ckpt)
#   STAGE_RACE=1       RACE zero-shot over the same ckpts (cheap; ~few min)
#   STAGE_SFT=1        LoRA SFT on RACE-train for each base ckpt (~30-60 min/ckpt)
#   STAGE_EVAL_SFT=1   re-run RC-MCQ variants with adapter attached
#
# Defaults: only STAGE_VARIANTS+STAGE_RACE on (no SFT). Set the env vars
# explicitly when you want to spend the GPU budget on training.
#
# Usage
# -----
#   bash scripts/pedagogical_utility_sweep.sh
#   STAGE_SFT=1 STAGE_EVAL_SFT=1 N_GPUS=8 bash scripts/pedagogical_utility_sweep.sh
#
# The orchestrator never touches the `beetle` tmux session and never starts
# concurrent GPU training (it serializes SFT runs across GPUs by default).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."

STAGE_VARIANTS="${STAGE_VARIANTS:-1}"
STAGE_RACE="${STAGE_RACE:-1}"
STAGE_SFT="${STAGE_SFT:-0}"
STAGE_EVAL_SFT="${STAGE_EVAL_SFT:-0}"

# Same default pair list across stages so SFT lands on what the variant
# rescore identifies as the strongest base ckpts.
DEFAULT_PAIRS=(
    "Beetle-FineWeb-2B/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-2b-nld-eng:step-5862"
    "Beetle-FineWeb-2B/beetle-bilingual-l2-50-simultaneous-b2-fineweb-2b-nld-eng:step-5862"
)
read -ra PAIRS <<< "${PAIRS:-${DEFAULT_PAIRS[*]}}"

N_GPUS="${N_GPUS:-1}"
LOG_DIR="${LOG_DIR:-/root/beetle_pipeline/logs/manual_stage18/pedagogical_utility}"
mkdir -p "$LOG_DIR"
ORCH_LOG="$LOG_DIR/orchestrator.log"

ts() { date -u +%FT%TZ; }
log() { echo "[$(ts)] $*" | tee -a "$ORCH_LOG"; }

log "starting pedagogical_utility_sweep"
log "  PAIRS=${PAIRS[*]}"
log "  N_GPUS=$N_GPUS"
log "  STAGE_VARIANTS=$STAGE_VARIANTS  STAGE_RACE=$STAGE_RACE  STAGE_SFT=$STAGE_SFT  STAGE_EVAL_SFT=$STAGE_EVAL_SFT"

# ---- Stage 1: variant rescore -------------------------------------------------
if [[ "$STAGE_VARIANTS" == "1" ]]; then
    log "=== stage 1: rcmcq score-variants ==="
    PAIRS="${PAIRS[*]}" N_GPUS="$N_GPUS" \
        bash scripts/rcmcq_score_variants_sweep.sh
fi

# ---- Stage 2: RACE zero-shot --------------------------------------------------
if [[ "$STAGE_RACE" == "1" ]]; then
    log "=== stage 2: race zero-shot ==="
    PAIRS="${PAIRS[*]}" N_GPUS="$N_GPUS" \
        bash scripts/race_zeroshot_for_rc.sh
fi

# ---- Stage 3: RACE→Cambridge SFT ---------------------------------------------
if [[ "$STAGE_SFT" == "1" ]]; then
    log "=== stage 3: SFT on RACE for each base ==="
    # Serialize SFT across GPUs to avoid colliding with the running pipeline /
    # the meco_significance CPU-bound job. One pair → one GPU at a time.
    gpu=0
    for pair in "${PAIRS[@]}"; do
        repo="${pair%:*}"; rev="${pair##*:}"
        log "  SFT $repo @ $rev (gpu=$gpu)"
        CUDA_VISIBLE_DEVICES="$gpu" \
            BASE_MODEL="$repo" REVISION="$rev" \
            DATASET="${SFT_DATASET:-race_combined}" \
            EPOCHS="${SFT_EPOCHS:-1}" \
            MAX_EXAMPLES="${SFT_MAX_EXAMPLES:-20000}" \
            bash scripts/sft_race_rc.sh
        gpu=$(( (gpu + 1) % N_GPUS ))
    done
fi

# ---- Stage 4: eval SFT'd adapters --------------------------------------------
if [[ "$STAGE_EVAL_SFT" == "1" ]]; then
    log "=== stage 4: eval SFT adapters on Cambridge RC-MCQ ==="
    gpu=0
    for pair in "${PAIRS[@]}"; do
        repo="${pair%:*}"; rev="${pair##*:}"
        slug=$(echo "${repo}_${rev}_${SFT_DATASET:-race_combined}" \
            | tr '/' '_' | tr ' ' '_' | sed 's/[^a-zA-Z0-9_.-]/_/g')
        adapter="results/sft_rc/${slug}"
        if [[ ! -d "$adapter" ]]; then
            log "  skip $repo @ $rev (no adapter at $adapter)"
            continue
        fi
        log "  eval $repo @ $rev with adapter $adapter (gpu=$gpu)"
        CUDA_VISIBLE_DEVICES="$gpu" \
            ADAPTER="$adapter" \
            BASE_MODEL="$repo" \
            REVISION="$rev" \
            bash scripts/eval_sft_on_rcmcq.sh
        gpu=$(( (gpu + 1) % N_GPUS ))
    done
fi

log "DONE"
echo
echo "Aggregated results:"
echo "  results/evals_paper/03_rcmcq_variants/_per_model_means.csv"
echo "  results/evals_paper/03_rcmcq_variants/_aggregated_long.csv"
echo "  results/evals_paper/03_rcmcq_variants_sft/   (if STAGE_EVAL_SFT=1)"
echo "  results/evals_paper/03b_race_zeroshot/       (if STAGE_RACE=1)"
echo
echo "Plots (run after sweep):"
echo "  python -m plots.make_rcmcq_score_variants"
echo "  python -m plots.make_pedagogical_utility_summary"
