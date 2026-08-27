#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# eval_chinese_english_baselines.sh
#
# Evaluate the Chinese–English bilingual cohort on the Chinese/English
# linguistic-skill suite at each model's `main` revision. Mirrors
# scripts/eval_frontier_baselines.sh.
#
# Evals run by this script:
#   - English BLiMP (nyu-mll/blimp)
#   - ZhoBLiMP (Junrui1202/zhoblimp)
#   - MultiBLiMP filtered to {eng, zho} (jumelet/multiblimp)
#   - MECO (existing data/meco_l2_stims.tsv L2 reading stimuli)
#   - BLiSS filtered to l1_language=Chinese
#   - xCOMPs Chinese (fpadovani/xcomps-dataset, comps_zh)
# UniversalCEFR is NOT run for this cohort.
#
# Cohort (defined in evals/baseline_models.py::BASELINE_MODELS["chinese_english"]):
#   QWEN 14B          — Qwen/Qwen-14B
#   BAICHUAN 7B       — baichuan-inc/Baichuan-7B
#   YI 6B             — 01-ai/Yi-6B
#   CHATGLM 6B        — THUDM/chatglm-6b   (non-CausalLM head; will SKIP if
#                                            AutoModelForCausalLM cannot load)
#
# Qwen-14B and Baichuan-7B require trust_remote_code=True, which is already
# set in evals/model_loader.py. ChatGLM-6B uses a custom architecture and may
# fail to load via AutoModelForCausalLM; the loader logs SKIP and the
# launcher continues with the rest of the cohort.
#
# All four models fit in a single A100-80G at bf16 (Qwen-14B is the largest
# at ~28 GB), so there is no Phase B / xlarge pass.
#
# Usage
# -----
#   bash scripts/eval_chinese_english_baselines.sh
#   N_GPUS=8 bash scripts/eval_chinese_english_baselines.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."

OUT="${OUT:-results}"
N_GPUS="${N_GPUS:-8}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${TMPDIR:-/tmp}/beetle_hf_cache}"
BLIMP_TASKS=(blimp_eng zhoblimp multiblimp)
BLIMP_LANGS=(eng zho)        # multiblimp restricted to {eng, zho}; blimp_eng + zhoblimp kept
BLISS_LANGS=(zho)
XCOMPS_LANGS=(zh)
REVISION="main"
SKIP_PREFETCH="${SKIP_PREFETCH:-0}"
RANK_LAUNCH_DELAY="${RANK_LAUNCH_DELAY:-10}"

export HF_HOME="$HF_CACHE_ROOT"
export TRANSFORMERS_CACHE="$HF_CACHE_ROOT"
export HF_HUB_CACHE="$HF_CACHE_ROOT"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"

echo "[$(date -u +%FT%TZ)] === Chinese–English baselines eval (BLiMP[en] + ZhoBLiMP + MultiBLiMP[en+zh] + MECO + BLiSS[zh] + xCOMPs[zh]) ==="
echo "[$(date -u +%FT%TZ)] N_GPUS=${N_GPUS}  HF_CACHE_ROOT=${HF_CACHE_ROOT}  OUT=${OUT}"

mkdir -p "$HF_CACHE_ROOT" "$OUT/logs" "$OUT/ledger" \
         "$OUT/evals/blimp" "$OUT/evals/meco" "$OUT/evals/bliss" \
         "$OUT/evals/xcomps"

MODEL_LIST_RAW="$(python - <<PY
from evals.baseline_models import BASELINE_MODELS
for repo in BASELINE_MODELS["chinese_english"]:
    print(f"{repo}\tchinese_english")
PY
)"

if [[ -z "$MODEL_LIST_RAW" ]]; then
    echo "ERROR: chinese_english cohort resolved to 0 models" >&2
    exit 1
fi

echo "  Chinese–English cohort ($(echo "$MODEL_LIST_RAW" | wc -l | tr -d ' ') models):"
echo "$MODEL_LIST_RAW" | sed 's/^/    /'

REPOS=()
while IFS=$'\t' read -r repo _family; do
    REPOS+=("$repo")
done <<< "$MODEL_LIST_RAW"

echo "[$(date -u +%FT%TZ)] Stage 0 — backfill eval ledger"
python -m analyze.eval_ledger backfill --output "$OUT" \
    > "$OUT/logs/ledger_backfill_chinese_english.log" 2>&1 || true

if [[ "$SKIP_PREFETCH" != "1" ]]; then
    echo "[$(date -u +%FT%TZ)] Stage 0.5 — serial model pre-fetch (HF_HOME=$HF_HOME)"
    for repo in "${REPOS[@]}"; do
        echo "  [$(date -u +%FT%TZ)] fetching $repo"
        python - <<PY >> "$OUT/logs/prefetch_chinese_english.log" 2>&1
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${repo}",
    revision="${REVISION}",
    allow_patterns=["*.json", "*.txt", "*.model", "tokenizer.*", "*.safetensors", "*.bin", "*.py"],
    ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx", "*.mlmodel"],
    max_workers=2,
)
print("OK ${repo}")
PY
        if [[ $? -ne 0 ]]; then
            echo "    WARNING: prefetch of $repo failed; see $OUT/logs/prefetch_chinese_english.log" >&2
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

# ─── Stage 1 — per-model parallel (no xlarge in this cohort) ────────────────
echo "[$(date -u +%FT%TZ)] Stage 1 — per-model parallel (${N_GPUS} ranks)"
declare -a stage1_pids=()
for RANK in $(seq 0 $((N_GPUS-1))); do
    _stagger "$RANK"
    CUDA_VISIBLE_DEVICES="$RANK" \
    setsid python evals/eval_per_model.py \
        --models chinese_english \
        --revision "$REVISION" \
        --output-root "$OUT" \
        --blimp-benchmarks "${BLIMP_TASKS[@]}" \
        --blimp-langs "${BLIMP_LANGS[@]}" \
        --bliss-langs "${BLISS_LANGS[@]}" \
        --run-xcomps --xcomps-langs "${XCOMPS_LANGS[@]}" \
        --skip-universalcefr \
        --rank "$RANK" --world-size "$N_GPUS" \
        </dev/null > "$OUT/logs/per_model_chinese_english_rank${RANK}.log" 2>&1 &
    stage1_pids+=($!)
done
for pid in "${stage1_pids[@]}"; do wait "$pid" || true; done

echo "[$(date -u +%FT%TZ)] Stage 5 — ledger merge"
python -m analyze.eval_ledger merge --output "$OUT" \
    > "$OUT/logs/ledger_merge_chinese_english.log" 2>&1 || true
python -m analyze.eval_ledger report --output "$OUT" \
    | tee "$OUT/logs/ledger_report_chinese_english.log"

echo "[$(date -u +%FT%TZ)] === Done: Chinese–English baselines ==="
