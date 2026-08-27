#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# eval_llmjp_baselines.sh
#
# Evaluate the LLM-jp v3 cohort (Aizawa et al., 2024) on the Japanese/English
# linguistic-skill suite at each model's `main` revision. Mirrors
# scripts/eval_frontier_baselines.sh.
#
# Evals run by this script:
#   - English BLiMP (nyu-mll/blimp)
#   - MultiBLiMP filtered to {eng, jpn} (jumelet/multiblimp)
#   - JBLiMP (polm-stability/jblimp), scored with SLOR
#   - BLiSS filtered to l1_language=Japanese
#   - xCOMPs Japanese (fpadovani/xcomps-dataset, comps_ja)
# MECO and UniversalCEFR are NOT run for this cohort.
#
# Cohort (defined in evals/baseline_models.py::BASELINE_MODELS["llmjp"]):
#   LLM-jp v3 — llm-jp/llm-jp-3-{150m,440m,980m,1.8b,3.7b,13b,172b-beta1}
#
# Trained on the LLM-jp Corpus v3 (1.7 T tokens: 950 B en, 592 B ja, 114 B
# code, 0.8 B ko, 0.3 B zh).
#
# Two-phase execution (same Phase A / Phase B split as eval_frontier_baselines.sh
# — see that script's header for the rationale):
#   Phase A : small / medium / large parallel across ${N_GPUS} ranks.
#   Phase B : 172B-beta1 (xlarge, ~340 GB bf16) loaded once with
#             device_map="auto" across all N_GPUS. Skipped by default
#             via SKIP_LLMJP_172B=1.
#
# Usage
# -----
#   bash scripts/eval_llmjp_baselines.sh
#   N_GPUS=8 bash scripts/eval_llmjp_baselines.sh
#   SKIP_LLMJP_172B=0 bash scripts/eval_llmjp_baselines.sh   # opt-in (≥340 GB)
#   SKIP_PREFETCH=1 bash scripts/eval_llmjp_baselines.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."

OUT="${OUT:-results}"
N_GPUS="${N_GPUS:-8}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${TMPDIR:-/tmp}/beetle_hf_cache}"
BLIMP_TASKS=(blimp_eng multiblimp)
BLIMP_LANGS=(eng jpn)        # multiblimp restricted to {eng, jpn}; blimp_eng kept (iso=eng)
BLISS_LANGS=(jpn)
XCOMPS_LANGS=(ja)
REVISION="main"
SKIP_LLMJP_172B="${SKIP_LLMJP_172B:-1}"      # 172 B-beta1 is opt-in only
SKIP_PREFETCH="${SKIP_PREFETCH:-0}"
RANK_LAUNCH_DELAY="${RANK_LAUNCH_DELAY:-10}"

export HF_HOME="$HF_CACHE_ROOT"
export TRANSFORMERS_CACHE="$HF_CACHE_ROOT"
export HF_HUB_CACHE="$HF_CACHE_ROOT"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"

echo "[$(date -u +%FT%TZ)] === LLM-jp baselines eval (BLiMP[en] + MultiBLiMP[en+ja] + JBLiMP + BLiSS[ja] + xCOMPs[ja]) ==="
echo "[$(date -u +%FT%TZ)] N_GPUS=${N_GPUS}  HF_CACHE_ROOT=${HF_CACHE_ROOT}  OUT=${OUT}"

mkdir -p "$HF_CACHE_ROOT" "$OUT/logs" "$OUT/ledger" \
         "$OUT/evals/blimp" "$OUT/evals/bliss" \
         "$OUT/evals/jblimp" "$OUT/evals/xcomps"

# ─── Resolve the cohort once and print it ────────────────────────────────────
MODEL_LIST_RAW="$(python - <<PY
from evals.baseline_models import BASELINE_MODELS
skip172b = ${SKIP_LLMJP_172B} == 1
for repo in BASELINE_MODELS["llmjp"]:
    if skip172b and "172b" in repo:
        continue
    print(f"{repo}\tllmjp")
PY
)"

if [[ -z "$MODEL_LIST_RAW" ]]; then
    echo "ERROR: llmjp cohort resolved to 0 models" >&2
    exit 1
fi

echo "  LLM-jp cohort ($(echo "$MODEL_LIST_RAW" | wc -l | tr -d ' ') models):"
echo "$MODEL_LIST_RAW" | sed 's/^/    /'

REPOS=()
while IFS=$'\t' read -r repo _family; do
    REPOS+=("$repo")
done <<< "$MODEL_LIST_RAW"

# ─── Stage 0 — ledger backfill ───────────────────────────────────────────────
echo "[$(date -u +%FT%TZ)] Stage 0 — backfill eval ledger"
python -m analyze.eval_ledger backfill --output "$OUT" \
    > "$OUT/logs/ledger_backfill_llmjp.log" 2>&1 || true

# ─── Stage 0.5 — serial pre-fetch ────────────────────────────────────────────
if [[ "$SKIP_PREFETCH" != "1" ]]; then
    echo "[$(date -u +%FT%TZ)] Stage 0.5 — serial model pre-fetch (HF_HOME=$HF_HOME)"
    for repo in "${REPOS[@]}"; do
        echo "  [$(date -u +%FT%TZ)] fetching $repo"
        python - <<PY >> "$OUT/logs/prefetch_llmjp.log" 2>&1
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${repo}",
    revision="${REVISION}",
    allow_patterns=["*.json", "*.txt", "*.model", "tokenizer.*", "*.safetensors", "*.bin"],
    ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx", "*.mlmodel"],
    max_workers=2,
)
print("OK ${repo}")
PY
        if [[ $? -ne 0 ]]; then
            echo "    WARNING: prefetch of $repo failed; see $OUT/logs/prefetch_llmjp.log" >&2
        fi
    done
    echo "[$(date -u +%FT%TZ)] Stage 0.5 — prefetch complete"
    du -sh "$HF_CACHE_ROOT" 2>/dev/null | sed 's/^/  HF cache size: /' || true
else
    echo "[$(date -u +%FT%TZ)] Stage 0.5 — SKIP prefetch (SKIP_PREFETCH=1)"
fi

_stagger() {
    local rank="$1"
    if (( rank > 0 && RANK_LAUNCH_DELAY > 0 )); then
        sleep "$RANK_LAUNCH_DELAY"
    fi
}

# ─── Stage 1A — non-xlarge models, rank-parallel ─────────────────────────────
echo "[$(date -u +%FT%TZ)] Stage 1A — per-model parallel (${N_GPUS} ranks; skip xlarge)"
declare -a stage1_pids=()
for RANK in $(seq 0 $((N_GPUS-1))); do
    _stagger "$RANK"
    CUDA_VISIBLE_DEVICES="$RANK" \
    setsid python evals/eval_per_model.py \
        --models llmjp \
        --revision "$REVISION" \
        --output-root "$OUT" \
        --blimp-benchmarks "${BLIMP_TASKS[@]}" \
        --blimp-langs "${BLIMP_LANGS[@]}" \
        --bliss-langs "${BLISS_LANGS[@]}" \
        --run-jblimp \
        --run-xcomps --xcomps-langs "${XCOMPS_LANGS[@]}" \
        --skip-meco \
        --skip-universalcefr \
        --skip-xlarge \
        --rank "$RANK" --world-size "$N_GPUS" \
        </dev/null > "$OUT/logs/per_model_llmjp_rank${RANK}.log" 2>&1 &
    stage1_pids+=($!)
done
for pid in "${stage1_pids[@]}"; do wait "$pid" || true; done

# ─── Stage 1B — xlarge (172B-beta1), single process, all GPUs ───────────────
if [[ "$SKIP_LLMJP_172B" != "1" ]]; then
    ALL_GPUS="$(seq -s, 0 $((N_GPUS-1)))"
    echo "[$(date -u +%FT%TZ)] Stage 1B — xlarge (CUDA_VISIBLE_DEVICES=${ALL_GPUS})"
    CUDA_VISIBLE_DEVICES="$ALL_GPUS" \
    setsid python evals/eval_per_model.py \
        --models llmjp \
        --revision "$REVISION" \
        --output-root "$OUT" \
        --blimp-benchmarks "${BLIMP_TASKS[@]}" \
        --blimp-langs "${BLIMP_LANGS[@]}" \
        --bliss-langs "${BLISS_LANGS[@]}" \
        --run-jblimp \
        --run-xcomps --xcomps-langs "${XCOMPS_LANGS[@]}" \
        --skip-meco \
        --skip-universalcefr \
        --xlarge-only \
        --rank 0 --world-size 1 \
        </dev/null > "$OUT/logs/per_model_llmjp_xlarge.log" 2>&1 || \
        echo "    (non-fatal) Stage 1B exited non-zero; see $OUT/logs/per_model_llmjp_xlarge.log"
else
    echo "[$(date -u +%FT%TZ)] Stage 1B — SKIP llm-jp-3-172b-beta1 (SKIP_LLMJP_172B=1)"
fi

# ─── Stage 5 — ledger merge ──────────────────────────────────────────────────
echo "[$(date -u +%FT%TZ)] Stage 5 — ledger merge"
python -m analyze.eval_ledger merge --output "$OUT" \
    > "$OUT/logs/ledger_merge_llmjp.log" 2>&1 || true
python -m analyze.eval_ledger report --output "$OUT" \
    | tee "$OUT/logs/ledger_report_llmjp.log"

echo "[$(date -u +%FT%TZ)] === Done: LLM-jp baselines ==="
