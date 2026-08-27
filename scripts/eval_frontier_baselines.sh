#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# eval_frontier_baselines.sh
#
# Evaluate the frontier public-LLM cohort on the paper's P0 linguistic-skill
# suite (MECO + BLiMP family + BLiSS), at each model's single `main` revision.
#
# Cohort (defined in evals/baseline_models.py::FRONTIER_FAMILIES):
#   Gemma 2      — google/gemma-2-{2b,9b,27b}
#   Qwen 3       — Qwen/Qwen3-{4B,8B}-Base
#   Llama 3.1    — meta-llama/Llama-3.1-8B
#   Apertus      — swiss-ai/Apertus-{8B,70B}-2509
#   EuroLLM      — utter-project/EuroLLM-{1.7B,9B}
#
# Benchmarks (3, run per-model in turn so only one model is VRAM-resident
# at any time):
#   BLiMP suite  — blimp_eng  blimp_nl  zhoblimp  multiblimp
#   MECO L2      — per-word surprisal on MECO stimuli
#   BLiSS        — RP@0 / RP@tau / NGS / CPS  (gated dataset; needs HF_TOKEN)
#
# FLORES PPL is intentionally NOT part of this script — run analyze/ppl_eval.py
# directly if you need it.
#
# Execution: each rank runs evals/eval_per_model.py over its sharded subset
# of models. For each model, the orchestrator loads the HF weights once,
# runs BLiMP + BLiSS back-to-back, unloads, then loads minicons for MECO,
# unloads, and proceeds to the next model. Net: at most 2 in-VRAM loads
# per model (down from 3-4 in the previous "stage by benchmark" layout).
#
# *** Download policy (important) ***
# All ranks share a single HF cache (HF_HOME=$HF_CACHE_ROOT). Before any
# parallel compute begins, Stage 0.5 serially snapshot_downloads each model
# (one at a time, one progress bar at a time). This prevents the "RAM/disk
# explosion" that occurs when N_GPUS ranks simultaneously pull tens of GB of
# weights from the HF CDN into N_GPUS duplicate caches.
#
# Apertus-70B (~140 GB in bf16) is skipped by default — it needs multi-GPU
# sharded loading and a very large disk. Opt in explicitly with
#   SKIP_APERTUS_70B=0 bash scripts/eval_frontier_baselines.sh
# and run on a node with ≥2×A100-80G free per eval process.
#
# Usage:
#   bash scripts/eval_frontier_baselines.sh
#   N_GPUS=4 bash scripts/eval_frontier_baselines.sh
#   HF_CACHE_ROOT=/scratch/$USER/hf bash scripts/eval_frontier_baselines.sh
#   SKIP_APERTUS_70B=0 bash scripts/eval_frontier_baselines.sh    # opt-in
#   SKIP_PREFETCH=1 bash scripts/eval_frontier_baselines.sh       # cache warm already
#   RANK_LAUNCH_DELAY=20 bash scripts/eval_frontier_baselines.sh  # seconds between ranks
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."

# Adopt the shared safety contract (see scripts/_sweep_safety.sh):
# RAM-aware concurrency cap, MALLOC_ARENA_MAX, canonical HF cache,
# setsid (SSH-SIGHUP immunity), trap-protected peer termination.
# shellcheck disable=SC1091
source scripts/_sweep_safety.sh
sweep_safety_init

OUT="${OUT:-results}"
N_GPUS="${N_GPUS:-8}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${BEETLE_HF_CACHE_ROOT}}"
BLIMP_TASKS=(blimp_eng blimp_nl zhoblimp multiblimp)
REVISION="main"
BATCH_SIZE_BLIMP="${BATCH_SIZE_BLIMP:-32}"
BATCH_SIZE_BLISS="${BATCH_SIZE_BLISS:-16}"
# Default ON — Apertus-70B needs >1 GPU and ~140 GB cache. Opt-in only.
SKIP_APERTUS_70B="${SKIP_APERTUS_70B:-1}"
SKIP_PREFETCH="${SKIP_PREFETCH:-0}"
# Small stagger between rank launches so the first model load per rank
# doesn't all hit the same HF CDN / filesystem at once even if the cache
# is already warm.
RANK_LAUNCH_DELAY="${RANK_LAUNCH_DELAY:-10}"

# *** Single shared HF cache for ALL ranks ***
# HF Hub uses advisory file locks for concurrent downloads; a per-rank
# cache just duplicates every model N_GPUS times on disk and multiplies
# download bandwidth/time by N_GPUS. Share the cache.
export HF_HOME="$HF_CACHE_ROOT"
export TRANSFORMERS_CACHE="$HF_CACHE_ROOT"
export HF_HUB_CACHE="$HF_CACHE_ROOT"
# Cap tokenizers threads so 8 ranks don't each spawn N_CPU tokenizer threads.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# Limit concurrent connections per rank so prefetch can saturate link
# without starving the eval processes of file handles.
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"

echo "[$(date -u +%FT%TZ)] === Frontier baselines eval (MECO + BLiMP + BLiSS + FLORES) ==="
echo "[$(date -u +%FT%TZ)] N_GPUS=${N_GPUS}  HF_CACHE_ROOT=${HF_CACHE_ROOT}  OUT=${OUT}"

mkdir -p "$HF_CACHE_ROOT" "$OUT/logs" "$OUT/ledger" \
         "$OUT/evals/blimp" "$OUT/evals/meco" "$OUT/evals/bliss" "$OUT/evals/ppl"

# ─── Resolve the cohort once and print it ────────────────────────────────────
MODEL_LIST_RAW="$(python - <<PY
from evals.baseline_models import resolve_models
skip70b = ${SKIP_APERTUS_70B} == 1
pairs = resolve_models(["frontier"])
if skip70b:
    pairs = [(r, f) for (r, f) in pairs if "70B" not in r]
for repo, family in pairs:
    print(f"{repo}\t{family}")
PY
)"

if [[ -z "$MODEL_LIST_RAW" ]]; then
    echo "ERROR: frontier cohort resolved to 0 models" >&2
    exit 1
fi

echo "  Frontier cohort ($(echo "$MODEL_LIST_RAW" | wc -l | tr -d ' ') models):"
echo "$MODEL_LIST_RAW" | sed 's/^/    /'

REPOS=()
while IFS=$'\t' read -r repo _family; do
    REPOS+=("$repo")
done <<< "$MODEL_LIST_RAW"

# ─── Stage 0 — ledger backfill (idempotent) ──────────────────────────────────
echo "[$(date -u +%FT%TZ)] Stage 0 — backfill eval ledger"
python -m analyze.eval_ledger backfill --output "$OUT" \
    > "$OUT/logs/ledger_backfill_frontier.log" 2>&1 || true

# ─── Stage 0.5 — serial pre-fetch of model weights ───────────────────────────
# ONE download at a time into the shared HF cache. Without this, N_GPUS
# parallel ranks each try to download the same set of multi-GB checkpoints
# simultaneously, duplicating RAM/disk usage N_GPUS times and frequently
# OOM-killing the node.
#
# Set SKIP_PREFETCH=1 to skip (e.g. cache is already warm on a shared FS).
if [[ "$SKIP_PREFETCH" != "1" ]]; then
    echo "[$(date -u +%FT%TZ)] Stage 0.5 — serial model pre-fetch (HF_HOME=$HF_HOME)"
    # Fail loud on first failed prefetch: a missing model here will OOM
    # Stage 1A under N_GPUS parallel ranks all retrying the same download.
    # `|| { echo ...; exit 1; }` overrides set -e's silent exit and surfaces
    # the offending repo in the parent shell.
    for repo in "${REPOS[@]}"; do
        echo "  [$(date -u +%FT%TZ)] fetching $repo"
        if ! python - >> "$OUT/logs/prefetch_frontier.log" 2>&1 <<PY
from huggingface_hub import snapshot_download
# Text-model weights only — skip the large image/video/audio shards that
# some multimodal repos ship alongside bf16 checkpoints.
snapshot_download(
    repo_id="${repo}",
    revision="${REVISION}",
    allow_patterns=[
        "*.json", "*.txt", "*.model", "tokenizer.*",
        "*.safetensors", "*.bin",
    ],
    ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx", "*.mlmodel"],
    max_workers=2,
)
print("OK ${repo}")
PY
        then
            echo "    ERROR: prefetch of $repo failed; see $OUT/logs/prefetch_frontier.log" >&2
            echo "    Aborting before Stage 1A to avoid N_GPUS-way concurrent re-download." >&2
            exit 1
        fi
    done
    echo "[$(date -u +%FT%TZ)] Stage 0.5 — prefetch complete"
    # Report disk usage so the user sees what landed
    du -sh "$HF_CACHE_ROOT" 2>/dev/null | sed 's/^/  HF cache size: /' || true
else
    echo "[$(date -u +%FT%TZ)] Stage 0.5 — SKIP prefetch (SKIP_PREFETCH=1)"
fi

# Helper: stagger rank launches so the first HF hit per rank is spread out
# even on a warm cache (reading a 50GB safetensors file from cold page cache
# still causes I/O pressure if 8 ranks do it simultaneously).
_stagger() {
    local rank="$1"
    if (( rank > 0 && RANK_LAUNCH_DELAY > 0 )); then
        sleep "$RANK_LAUNCH_DELAY"
    fi
}

# Note on HF cache: all ranks now share $HF_HOME (exported at the top of this
# script). Stage 0.5 has already populated it serially, so the parallel
# launches below should be cache-hits with no concurrent HF CDN traffic.

# ─── Stage 1A — Per-model orchestrator (small/medium/large, rank-parallel) ─────
#
# Each rank runs evals/eval_per_model.py over its sharded subset of the
# non-xlarge frontier models. Per-repo batch sizes come from MODEL_SIZE_HINT
# in evals/baseline_models.py via --auto-batch-size (default ON), so a
# 27 B model gets batch=8 while a 2 B model gets batch=32.
#
# xlarge models (>=30 B params, e.g. Apertus-70B if SKIP_APERTUS_70B=0) are
# skipped here via --skip-xlarge and run alone in Stage 1B with all GPUs
# visible to a single process via device_map="auto" — running them in
# parallel across N_GPUS ranks would multiply VRAM by N_GPUS and OOM the
# node. Without this split the launcher reliably runs out of GPU RAM on
# Gemma-2-27B + Apertus-70B cohorts even on 8×A100-80G.
echo "[$(date -u +%FT%TZ)] Stage 1A — per-model parallel via launch_ranks (skip xlarge)"
# launch_ranks supplies setsid + per-rank logging + trap-protected peer
# termination + RAM-aware concurrency cap. Inline RANK_LAUNCH_DELAY stagger
# inside the bash -c body since the helper exports RANK/WORLD_SIZE.
export REVISION OUT RANK_LAUNCH_DELAY
export BLIMP_TASKS_STR="${BLIMP_TASKS[*]}"
SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-${OUT}/logs/sweep_frontier}" \
launch_ranks "stage1a_frontier" \
    bash -c '
        if (( RANK > 0 && RANK_LAUNCH_DELAY > 0 )); then sleep "$((RANK * RANK_LAUNCH_DELAY))"; fi
        # shellcheck disable=SC2086
        IFS=" " read -r -a _BLIMP_TASKS <<< "$BLIMP_TASKS_STR"
        python evals/eval_per_model.py \
            --models frontier \
            --revision "$REVISION" \
            --output-root "$OUT" \
            --blimp-benchmarks "${_BLIMP_TASKS[@]}" \
            --skip-xlarge \
            --rank "$RANK" --world-size "$WORLD_SIZE"
    '

# ─── Stage 1B — xlarge models, single process, all GPUs, device_map=auto ───────
#
# For Apertus-70B (or any future xlarge baseline). One Python process
# spans all N_GPUS via device_map="auto"; auto-batch-size yields batch=4.
# This stage is a no-op if the cohort contains no xlarge repos.
ALL_GPUS="$(seq -s, 0 $((N_GPUS-1)))"
echo "[$(date -u +%FT%TZ)] Stage 1B — xlarge (single process, CUDA_VISIBLE_DEVICES=${ALL_GPUS})"
CUDA_VISIBLE_DEVICES="$ALL_GPUS" \
setsid python evals/eval_per_model.py \
    --models frontier \
    --revision "$REVISION" \
    --output-root "$OUT" \
    --blimp-benchmarks "${BLIMP_TASKS[@]}" \
    --xlarge-only \
    --rank 0 --world-size 1 \
    </dev/null > "$OUT/logs/per_model_frontier_xlarge.log" 2>&1 || \
    echo "    (non-fatal) Stage 1B exited non-zero; see $OUT/logs/per_model_frontier_xlarge.log"

# Note: BLiSS will auto-skip per-model if HF_TOKEN is unset (the orchestrator
# logs a warning and continues). FLORES-PPL is intentionally NOT run here —
# this script's documented scope is the P0 linguistic-skill suite (BLiMP +
# MECO + BLiSS). To compute PPL separately, run analyze/ppl_eval.py directly.

# ─── Stage 5 — ledger merge + regenerate tables/figures (rank 0) ────────────
echo "[$(date -u +%FT%TZ)] Stage 5 — ledger merge + regenerate tables/figures"

python -m analyze.eval_ledger merge --output "$OUT" \
    > "$OUT/logs/ledger_merge_frontier.log" 2>&1 || true

# Tables (all six + new frontier table)
python plots/make_tables.py \
    > "$OUT/logs/make_tables_frontier.log" 2>&1 || true

# Main figures
for py in \
    plots/make_plots.py \
    plots/make_cross_eval_plots.py \
    plots/make_learning_dynamics.py \
    plots/make_bliss_vs_meco.py \
    plots/make_fig6a.py \
    plots/make_fig7_extras.py \
    plots/make_fig7_qualitative.py \
    plots/make_table13_meco_langrows.py
do
    name="$(basename "$py" .py)"
    echo "  Regenerating via $py"
    python "$py" > "$OUT/logs/${name}_frontier.log" 2>&1 || \
        echo "    (non-fatal) $py exited non-zero; see $OUT/logs/${name}_frontier.log"
done

# Ledger report
python -m analyze.eval_ledger report --output "$OUT" \
    | tee "$OUT/logs/ledger_report_frontier.log"

echo "[$(date -u +%FT%TZ)] === Done: frontier baselines ==="
