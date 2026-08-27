#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# sft_learner_pipeline.sh
#
# Instruction-tune a Beetle base model on its L1-matched lilygoulder/
# learner_tuning_<LANG> corpus using LoRA. Outputs a per-model adapter
# directory under $OUT/sft/<slug>.
#
# Usage
#   bash scripts/sft_learner_pipeline.sh <pair> [model_repo]
#
# Args
#   <pair>       Beetle language pair: deu-eng | zho-eng | nld-eng
#                (also accepts fra-eng, spa-eng, ara-eng, tur-eng, pol-eng
#                where the L1 has a matching learner_tuning corpus.)
#   model_repo   Optional explicit HF repo of the base model. If omitted,
#                the script picks the canonical Beetle-HumanScale b2 default
#                slug for that pair.
#
# Coverage
#   deu-eng → lilygoulder/learner_tuning_GER
#   zho-eng → lilygoulder/learner_tuning_CHI
#   fra-eng → lilygoulder/learner_tuning_FRE
#   spa-eng → lilygoulder/learner_tuning_SPA
#   ara-eng → lilygoulder/learner_tuning_ARA
#   tur-eng → lilygoulder/learner_tuning_TUR
#   pol-eng → lilygoulder/learner_tuning_POL
#   nld-eng → NO matching learner corpus; script will refuse with a hint.
#
# Env-overridable hyperparameters (defaults match finetune/sft_learner.py):
#   EPOCHS, LEARNING_RATE, PER_DEVICE_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
#   LORA_R, LORA_ALPHA, MAX_SEQ_LENGTH, MAX_STEPS (use for smoke tests),
#   PUSH_TO_HUB (HF repo id), OUT (default: results), REVISION (default: main).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."

PAIR="${1:-}"
MODEL_REPO="${2:-}"

if [[ -z "$PAIR" ]]; then
    echo "Usage: $0 <pair> [model_repo]" >&2
    echo "  e.g.  $0 deu-eng" >&2
    exit 2
fi

declare -A L1_TO_LEARNER=(
    [deu]=GER
    [zho]=CHI
    [fra]=FRE
    [spa]=SPA
    [ara]=ARA
    [tur]=TUR
    [pol]=POL
)

L1="${PAIR%-*}"
L2="${PAIR#*-}"
LEARNER_CODE="${L1_TO_LEARNER[$L1]:-}"

if [[ -z "$LEARNER_CODE" ]]; then
    echo "ERROR: no lilygoulder/learner_tuning corpus for L1='$L1' (pair='$PAIR')." >&2
    echo "  Supported L1s: ${!L1_TO_LEARNER[*]}" >&2
    echo "  For nld-eng you can either skip SFT, use English-only SFT data, or" >&2
    echo "  pick a typologically related L1 manually via the standalone CLI:" >&2
    echo "    python -m finetune.sft_learner --base-model <repo> --dataset GER ..." >&2
    exit 3
fi

if [[ -z "$MODEL_REPO" ]]; then
    # Canonical Beetle-HumanScale b2 default for this pair
    MODEL_REPO="Beetle-HumanScale/beetle-bilingual-l2-50-simultaneous-b2-humanscale-${L1}-${L2}"
fi

OUT="${OUT:-results}"
REVISION="${REVISION:-main}"
SLUG="$(echo "$MODEL_REPO" | tr '/' '_')__sft_${LEARNER_CODE}"
ADAPTER_DIR="$OUT/sft/$SLUG"

EPOCHS="${EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
MAX_STEPS="${MAX_STEPS:--1}"
PUSH_TO_HUB="${PUSH_TO_HUB:-}"

extra=()
[[ -n "$PUSH_TO_HUB" ]] && extra+=(--push-to-hub "$PUSH_TO_HUB")

echo "[sft] base=$MODEL_REPO dataset=lilygoulder/learner_tuning_${LEARNER_CODE}"
echo "[sft] adapter_dir=$ADAPTER_DIR"

mkdir -p "$ADAPTER_DIR"

python -m finetune.sft_learner \
    --base-model "$MODEL_REPO" \
    --revision "$REVISION" \
    --dataset "$LEARNER_CODE" \
    --output-dir "$ADAPTER_DIR" \
    --epochs "$EPOCHS" \
    --learning-rate "$LEARNING_RATE" \
    --per-device-batch-size "$PER_DEVICE_BATCH_SIZE" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --max-steps "$MAX_STEPS" \
    "${extra[@]}"

echo "[sft] done. adapter at $ADAPTER_DIR"
echo
echo "Next: evaluate the SFT'd model on UniversalCEFR with"
echo "  ADAPTER='$ADAPTER_DIR' MODELS='$MODEL_REPO' MODE=generation \\"
echo "    bash scripts/eval_universalcefr.sh"
