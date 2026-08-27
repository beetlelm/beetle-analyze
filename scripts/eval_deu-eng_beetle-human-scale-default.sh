#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# eval_deu-eng_beetle-human-scale-default.sh
#
# Evaluate German-English bilingual models in the Beetle-HumanScale HF org
# (https://huggingface.co/Beetle-HumanScale) under the paper's default regime.
#
# Scope:
#   * Org filter   : Beetle-HumanScale only.
#   * Language set : {deu, eng} — strict subset.
#   * BLiMP suites : blimp_eng, multiblimp (MultiBLiMP covers German).
#
# See scripts/eval_nld-eng_beetle-human-scale-default.sh for the full
# pipeline structure (priority tiers, per-model ordering, ledger backfill).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."

# Adopt the shared safety contract (see scripts/_sweep_safety.sh):
# RAM-aware concurrency cap, MALLOC_ARENA_MAX, canonical HF cache,
# setsid (SSH-SIGHUP immunity), trap-protected peer termination.
# shellcheck disable=SC1091
source scripts/_sweep_safety.sh
sweep_safety_init

LANG_PAIR="deu-eng"
LANGS=(deu eng)
ORGS=(Beetle-HumanScale)
FAMILY="humanscale"
CONDITIONS=(mono B1 B2 B3 B4 B5)
BLIMP_TASKS=(blimp_eng multiblimp)
OUT=results

PRIORITY="${PRIORITY:-P0}"
ORDER="${ORDER:-per-model}"
N_GPUS="${N_GPUS:-8}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${BEETLE_HF_CACHE_ROOT}}"
# Checkpoint iteration scope: all | latest | both (default both).
# - all    : every step-N branch iterated; only *_trajectory figures emitted.
# - latest : only highest step-N (or main); only *_final figures emitted.
# - both   : every step-N iterated AND both trajectory + final figures.
CKPT_SCOPE="${CKPT_SCOPE:-both}"

BLIMP_LATEST_FLAG=""
if [[ "$CKPT_SCOPE" == "latest" ]]; then
    BLIMP_LATEST_FLAG="--latest_only"
fi

echo "[$(date -u +%FT%TZ)] === Eval ${LANG_PAIR} (Beetle-HumanScale) ==="
echo "[$(date -u +%FT%TZ)] PRIORITY=${PRIORITY}  ORDER=${ORDER}  CKPT_SCOPE=${CKPT_SCOPE}"
echo "[$(date -u +%FT%TZ)] N_GPUS=${N_GPUS}  HF_CACHE_ROOT=${HF_CACHE_ROOT}"
echo "[$(date -u +%FT%TZ)] ORGS=${ORGS[*]}  BLIMP_TASKS=${BLIMP_TASKS[*]}"

mkdir -p "$HF_CACHE_ROOT" "$OUT/logs" "$OUT/ledger"

echo "[$(date -u +%FT%TZ)] Stage 0 — backfill eval ledger"
python -m analyze.eval_ledger backfill --output "$OUT" \
    > "$OUT/logs/ledger_backfill_${LANG_PAIR}.log" 2>&1 || true

echo "[$(date -u +%FT%TZ)] Stage 1 — analyze.run ${PRIORITY} via launch_ranks"
SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-${OUT}/logs/sweep_${LANG_PAIR}}" \
launch_ranks "stage1_${LANG_PAIR}" \
    python -m analyze.run \
        --priority "$PRIORITY" \
        --order "$ORDER" \
        --ckpt-scope "$CKPT_SCOPE" \
        --family "$FAMILY" \
        --orgs "${ORGS[@]}" \
        --condition "${CONDITIONS[@]}" \
        --langs "${LANGS[@]}" \
        --cache-mode keep \
        --output-root "$OUT"

echo "[$(date -u +%FT%TZ)] Stage 2 — BLiMP suite (${N_GPUS} ranks)"
MODEL_LIST="$(python - <<PY
from models import MODEL_GROUPS
from analyze.model_naming import parse_repo

allowed_langs = {"deu", "eng"}
allowed_orgs  = {"Beetle-HumanScale"}
out = []
for repo in MODEL_GROUPS["humanscale"]:
    try:
        meta = parse_repo(repo)
    except Exception:
        continue
    if repo.split("/", 1)[0] not in allowed_orgs:
        continue
    repo_langs = {meta.l1} | ({meta.l2} if meta.l2 else set())
    repo_langs.discard(None)
    if repo_langs and repo_langs <= allowed_langs:
        out.append(repo)
print(",".join(out))
PY
)"

if [[ -z "$MODEL_LIST" ]]; then
    echo "ERROR: no Beetle-HumanScale ${LANG_PAIR} models resolved" >&2
    exit 1
fi

echo "  MODEL_LIST ($(echo "$MODEL_LIST" | tr ',' '\n' | wc -l) models):"
echo "$MODEL_LIST" | tr ',' '\n' | sed 's/^/    /'

echo "[$(date -u +%FT%TZ)] Stage 2 — BLiMP suite via launch_ranks"
export MODEL_LIST
export ORGS="${ORGS[*]}"
export BLIMP_TASKS="${BLIMP_TASKS[*]}"
SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-${OUT}/logs/sweep_${LANG_PAIR}}" \
launch_ranks "stage2_${LANG_PAIR}" \
    bash -c "python legacy/eval_model.py \
        --rank \"\$RANK\" --world_size \"\$WORLD_SIZE\" \
        --output_dir \"$(pwd)\" \
        --no_push \
        $BLIMP_LATEST_FLAG"

echo "[$(date -u +%FT%TZ)] Stage 3 — ledger merge + figures + significance"
python -m analyze.eval_ledger merge --output "$OUT" \
    > "$OUT/logs/ledger_merge_${LANG_PAIR}.log" 2>&1 || true
python -m analyze.run \
    --figures --paper-figures \
    --ckpt-scope "$CKPT_SCOPE" \
    --family "$FAMILY" --orgs "${ORGS[@]}" --langs "${LANGS[@]}" \
    --output-root "$OUT" --rank 0 --world-size 1 \
    > "$OUT/logs/figures_${LANG_PAIR}.log" 2>&1 || true
python -m analyze.run_blimp_significance \
    --tasks "${BLIMP_TASKS[@]}" \
    --output "$OUT/blimp_significance/${LANG_PAIR}" \
    > "$OUT/logs/blimp_significance_${LANG_PAIR}.log" 2>&1 || true
python -m analyze.eval_ledger report --output "$OUT" \
    | tee "$OUT/logs/ledger_report_${LANG_PAIR}.log"

echo "[$(date -u +%FT%TZ)] === Done: ${LANG_PAIR} (Beetle-HumanScale) ==="
