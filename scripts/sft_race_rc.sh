#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# sft_race_rc.sh
#
# LoRA SFT a Beetle base model on RACE (or BoolQ) for RC-MCQ transfer.
# Single-GPU; pin via CUDA_VISIBLE_DEVICES if other ranks are busy.
#
# Usage
# -----
#   bash scripts/sft_race_rc.sh
#   BASE_MODEL=Beetle-FineWeb-2B/...nld-eng REVISION=step-5862 \
#       DATASET=race_combined MAX_EXAMPLES=20000 EPOCHS=1 \
#       bash scripts/sft_race_rc.sh
#   # smoke (no GPU spend):
#   BASE_MODEL=gpt2 DATASET=boolq MAX_STEPS=30 \
#       bash scripts/sft_race_rc.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate

BASE_MODEL="${BASE_MODEL:-Beetle-FineWeb-2B/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-2b-nld-eng}"
REVISION="${REVISION:-step-5862}"
DATASET="${DATASET:-race_combined}"
MAX_EXAMPLES="${MAX_EXAMPLES:-20000}"
MAX_STEPS="${MAX_STEPS:--1}"
EPOCHS="${EPOCHS:-1.0}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
PERSONA_DROPOUT="${PERSONA_DROPOUT:-0.25}"
LORA_R="${LORA_R:-16}"

slug=$(echo "${BASE_MODEL}_${REVISION}_${DATASET}" \
    | tr '/' '_' | tr ' ' '_' | sed 's/[^a-zA-Z0-9_.-]/_/g')
OUTPUT_DIR="${OUTPUT_DIR:-results/sft_rc/${slug}}"
LOG_DIR="${LOG_DIR:-/root/beetle_pipeline/logs/manual_stage18/sft_rc}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
LOG="$LOG_DIR/${slug}.log"

echo "[$(date -u +%FT%TZ)] sft_race_rc starting" | tee -a "$LOG"
echo "  base=$BASE_MODEL @ $REVISION  dataset=$DATASET  out=$OUTPUT_DIR" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python -m finetune.sft_rc \
    --base-model "$BASE_MODEL" \
    --revision "$REVISION" \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --max-steps "$MAX_STEPS" \
    --max-examples "$MAX_EXAMPLES" \
    --learning-rate "$LEARNING_RATE" \
    --per-device-batch-size "$PER_DEVICE_BATCH_SIZE" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --persona-dropout "$PERSONA_DROPOUT" \
    --lora-r "$LORA_R" \
    >>"$LOG" 2>&1

echo "[$(date -u +%FT%TZ)] DONE → $OUTPUT_DIR" | tee -a "$LOG"
echo "Adapter saved to: $OUTPUT_DIR"
echo "To eval on Cambridge RC-MCQ:"
echo "  ADAPTER=$OUTPUT_DIR BASE_MODEL=$BASE_MODEL REVISION=$REVISION \\"
echo "      bash scripts/eval_sft_on_rcmcq.sh"
