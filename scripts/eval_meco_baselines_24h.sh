#!/usr/bin/env bash
# eval_meco_baselines_24h.sh — single-GPU MECO sweep over a fixed baseline cohort.
#
# Cohort (18 models, all evaluated on the main/final checkpoint):
#   - utter-project/EuroLLM-1.7B
#   - utter-project/EuroLLM-9B
#   - google/gemma-2-2b
#   - google/gemma-2-9b
#   - Qwen/Qwen3-4B-Base
#   - Qwen/Qwen3-8B-Base
#   - meta-llama/Llama-3.1-8B
#   - swiss-ai/Apertus-8B-2509
#   - llm-jp/llm-jp-3-{150m,440m,980m,1.8b,3.7b,13b}
#   - Qwen/Qwen-14B            (trust_remote_code)
#   - baichuan-inc/Baichuan-7B (trust_remote_code)
#   - 01-ai/Yi-6B              (trust_remote_code)
#   - THUDM/chatglm-6b         (non-CausalLM head; will likely SKIP cleanly)
#
# Pinned to a single GPU via CUDA_VISIBLE_DEVICES; defaults to "1" so this
# can run alongside the GPU-0 BLiMP sweep without contention.

set -euo pipefail

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES must be set (single GPU)" >&2
    exit 1
fi
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    echo "ERROR: this script runs on a SINGLE GPU; CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' has multiple" >&2
    exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python}"
WALLTIME_SECONDS="${WALLTIME_SECONDS:-82800}"  # 23h
LOG_DIR="$ROOT/logs/meco_baselines"
mkdir -p "$LOG_DIR"
SWEEP_LOG="$LOG_DIR/sweep_$(date -u +%Y%m%dT%H%M%SZ).log"

# HF cache pinned to scratch (home is at 100%).
SCRATCH="${SCRATCH:-${TMPDIR:-/tmp}/beetle}"
export HF_HOME="${HF_HOME:-$SCRATCH/huggingface_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$SCRATCH/hf_datasets_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export TMPDIR="${TMPDIR:-$SCRATCH/tmp}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-0}"
# xet's adaptive-concurrency download workers leaked threads on llm-jp-3-3.7b
# (process hung after download completed, ~166 sleeping threads). Force the
# legacy HTTP downloader so each model load exits cleanly.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-0}"
mkdir -p "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR"

# Cohort. Smallest first so the autopush has artefacts within minutes.
MODELS=(
    "llm-jp/llm-jp-3-150m"
    "llm-jp/llm-jp-3-440m"
    "llm-jp/llm-jp-3-980m"
    "utter-project/EuroLLM-1.7B"
    "llm-jp/llm-jp-3-1.8b"
    "google/gemma-2-2b"
    "llm-jp/llm-jp-3-3.7b"
    "Qwen/Qwen3-4B-Base"
    "01-ai/Yi-6B"
    "baichuan-inc/Baichuan-7B"
    "THUDM/chatglm-6b"
    "meta-llama/Llama-3.1-8B"
    "Qwen/Qwen3-8B-Base"
    "swiss-ai/Apertus-8B-2509"
    "google/gemma-2-9b"
    "utter-project/EuroLLM-9B"
    "llm-jp/llm-jp-3-13b"
    "Qwen/Qwen-14B"
)

echo "[$(date -u +%FT%TZ)] meco-evals sweep start  GPU=$CUDA_VISIBLE_DEVICES  walltime=${WALLTIME_SECONDS}s  cohort=${#MODELS[@]}" \
    | tee -a "$SWEEP_LOG"
echo "  HF_HOME=$HF_HOME" | tee -a "$SWEEP_LOG"

T_START=$(date +%s)
deadline=$(( T_START + WALLTIME_SECONDS ))

for repo in "${MODELS[@]}"; do
    now=$(date +%s)
    if (( now >= deadline )); then
        echo "[$(date -u +%FT%TZ)] walltime reached — remaining models skipped" | tee -a "$SWEEP_LOG"
        break
    fi
    remaining=$(( deadline - now ))
    echo "[$(date -u +%FT%TZ)] === $repo  (remaining=${remaining}s) ===" | tee -a "$SWEEP_LOG"

    # Per-model timeout: cap at 90 min so a hung load can't burn the whole budget.
    timeout "$(( remaining < 5400 ? remaining : 5400 ))" \
    "$PY" -m evals.eval_per_model \
        --models "$repo" \
        --revision main \
        --rank 0 --world-size 1 \
        --output-root "$ROOT/results" \
        --skip-blimp --skip-bliss --skip-universalcefr \
        2>&1 | tee -a "$SWEEP_LOG" || {
            ec=$?
            echo "[$(date -u +%FT%TZ)] $repo exited rc=$ec — continuing" | tee -a "$SWEEP_LOG"
        }
done

echo "[$(date -u +%FT%TZ)] meco-evals sweep done in $(( $(date +%s) - T_START ))s" | tee -a "$SWEEP_LOG"
