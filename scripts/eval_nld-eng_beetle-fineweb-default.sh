#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# eval_nld-eng_beetle-fineweb-default.sh
#
# Evaluate Dutch-English bilingual models in the Beetle-FineWeb HF orgs
# (https://huggingface.co/Beetle-FineWeb and
#  https://huggingface.co/Beetle-FineWeb-Bilingual) under the 24B-token
# FineWeb regime.
#
# Scope:
#   * Org filter   : Beetle-FineWeb + Beetle-FineWeb-Bilingual.
#                    BeetleLM/* legacy excluded.
#   * Language set : {nld, eng} — strict subset.
#   * BLiMP suites : blimp_eng, blimp_nl, multiblimp.
#
# GPU budget (8×A100): ≈ 90 min/model × N_FW_MODELS ≈ 6–9 h wallclock for
# the Dutch-English slice; use PRIORITY=P0 first, then P1 for baselines.
#
# See scripts/eval_nld-eng_beetle-human-scale-default.sh for the full
# pipeline structure.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."

# Adopt the shared safety contract (see scripts/_sweep_safety.sh):
# RAM-aware concurrency cap, MALLOC_ARENA_MAX, canonical HF cache,
# setsid (SSH-SIGHUP immunity), trap-protected peer termination.
# shellcheck disable=SC1091
source scripts/_sweep_safety.sh
sweep_safety_init

LANG_PAIR="nld-eng"
LANGS=(nld eng)
ORGS=(Beetle-FineWeb Beetle-FineWeb-Bilingual)
FAMILY="fineweb"
CONDITIONS=(mono B1 B2 B3 B4 B5)
BLIMP_TASKS=(blimp_eng blimp_nl multiblimp)
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

echo "[$(date -u +%FT%TZ)] === Eval ${LANG_PAIR} (Beetle-FineWeb) ==="
echo "[$(date -u +%FT%TZ)] PRIORITY=${PRIORITY}  ORDER=${ORDER}  CKPT_SCOPE=${CKPT_SCOPE}"
echo "[$(date -u +%FT%TZ)] N_GPUS=${N_GPUS}  HF_CACHE_ROOT=${HF_CACHE_ROOT}"
echo "[$(date -u +%FT%TZ)] ORGS=${ORGS[*]}  BLIMP_TASKS=${BLIMP_TASKS[*]}"

mkdir -p "$HF_CACHE_ROOT" "$OUT/logs" "$OUT/ledger"

echo "[$(date -u +%FT%TZ)] Stage 0 — backfill eval ledger"
python -m analyze.eval_ledger backfill --output "$OUT" \
    > "$OUT/logs/ledger_backfill_${LANG_PAIR}_fw.log" 2>&1 || true

echo "[$(date -u +%FT%TZ)] Stage 1 — analyze.run ${PRIORITY} via launch_ranks"
SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-${OUT}/logs/sweep_${LANG_PAIR}_fw}" \
launch_ranks "stage1_${LANG_PAIR}_fw" \
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

allowed_langs = {"nld", "eng"}
allowed_orgs  = {"Beetle-FineWeb", "Beetle-FineWeb-Bilingual"}
out = []
for key in ("fineweb", "fineweb_legacy", "fineweb_bidirectional", "fineweb_planned"):
    for repo in MODEL_GROUPS.get(key, []):
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
# Dedup preserving order.
seen = set()
uniq = []
for r in out:
    if r not in seen:
        seen.add(r); uniq.append(r)
print(",".join(uniq))
PY
)"

if [[ -z "$MODEL_LIST" ]]; then
    echo "ERROR: no Beetle-FineWeb ${LANG_PAIR} models resolved" >&2
    exit 1
fi

echo "  MODEL_LIST ($(echo "$MODEL_LIST" | tr ',' '\n' | wc -l) models):"
echo "$MODEL_LIST" | tr ',' '\n' | sed 's/^/    /'

echo "[$(date -u +%FT%TZ)] Stage 2 — BLiMP suite via launch_ranks"
export MODEL_LIST
export ORGS="${ORGS[*]}"
export BLIMP_TASKS="${BLIMP_TASKS[*]}"
SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-${OUT}/logs/sweep_${LANG_PAIR}_fw}" \
launch_ranks "stage2_${LANG_PAIR}_fw" \
    bash -c "python legacy/eval_model.py \
        --rank \"\$RANK\" --world_size \"\$WORLD_SIZE\" \
        --output_dir \"$(pwd)\" \
        --no_push \
        $BLIMP_LATEST_FLAG"

echo "[$(date -u +%FT%TZ)] Stage 3 — ledger merge + figures + significance"
python -m analyze.eval_ledger merge --output "$OUT" \
    > "$OUT/logs/ledger_merge_${LANG_PAIR}_fw.log" 2>&1 || true
python -m analyze.run \
    --figures --paper-figures \
    --ckpt-scope "$CKPT_SCOPE" \
    --family "$FAMILY" --orgs "${ORGS[@]}" --langs "${LANGS[@]}" \
    --output-root "$OUT" --rank 0 --world-size 1 \
    > "$OUT/logs/figures_${LANG_PAIR}_fw.log" 2>&1 || true
python -m analyze.run_blimp_significance \
    --tasks "${BLIMP_TASKS[@]}" \
    --output "$OUT/blimp_significance/${LANG_PAIR}_fineweb" \
    > "$OUT/logs/blimp_significance_${LANG_PAIR}_fw.log" 2>&1 || true
python -m analyze.eval_ledger report --output "$OUT" \
    | tee "$OUT/logs/ledger_report_${LANG_PAIR}_fw.log"

echo "[$(date -u +%FT%TZ)] === Done: ${LANG_PAIR} (Beetle-FineWeb) ==="
