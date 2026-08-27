#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# eval_sft_on_rcmcq.sh
#
# Stack a LoRA adapter on top of a Beetle base model and re-run the
# variant RC-MCQ eval (mean / sum / PMI). Outputs land under a sibling
# directory of the base-model results so they don't collide.
#
# Usage
# -----
#   ADAPTER=results/sft_rc/<slug> \
#       BASE_MODEL=Beetle-FineWeb-2B/...nld-eng \
#       REVISION=step-5862 \
#       bash scripts/eval_sft_on_rcmcq.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate

ADAPTER="${ADAPTER:?Set ADAPTER=<path-or-hf-repo-of-LoRA-adapter>}"
BASE_MODEL="${BASE_MODEL:-Beetle-FineWeb-2B/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-2b-nld-eng}"
REVISION="${REVISION:-step-5862}"
OUT="${OUT:-results/evals_paper/03_rcmcq_variants_sft}"
DATASET="${DATASET:-education/Cambridge Multiple-Choice Questions Reading Dataset.jsonl}"
PROMPTS="${PROMPTS:-default cefr_persona stratified gradient}"
CEFR_LEVELS="${CEFR_LEVELS:-A2 B1 B2 C1 C2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LOG_DIR="${LOG_DIR:-/root/beetle_pipeline/logs/manual_stage18/eval_sft_on_rcmcq}"

mkdir -p "$LOG_DIR" "$OUT"
adapter_tag=$(basename "$ADAPTER")
output_tag="rev=${REVISION}__sft=${adapter_tag}"
LOG="$LOG_DIR/${BASE_MODEL//\//_}__${output_tag}.log"

echo "[$(date -u +%FT%TZ)] eval_sft_on_rcmcq" | tee -a "$LOG"
echo "  base=$BASE_MODEL @ $REVISION  adapter=$ADAPTER  out_tag=$output_tag" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python -m evals.eval_rcmcq_variants \
    --models "$BASE_MODEL" \
    --revision "$REVISION" \
    --dataset "$DATASET" \
    --cefr-levels $CEFR_LEVELS \
    --prompts $PROMPTS \
    --output-root "$OUT" \
    --batch-size "$BATCH_SIZE" \
    --adapter "$ADAPTER" \
    --output-tag "$output_tag" \
    >>"$LOG" 2>&1

python -m analyze.rcmcq_rescore \
    --root "$OUT" \
    --merge-rcmcq results/evals_paper/03_rcmcq \
    | tee -a "$LOG"

echo "[$(date -u +%FT%TZ)] DONE" | tee -a "$LOG"
