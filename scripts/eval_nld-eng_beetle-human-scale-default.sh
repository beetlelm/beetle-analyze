#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# eval_nld-eng_beetle-human-scale-default.sh
#
# Evaluate Dutch-English bilingual models in the Beetle-HumanScale HF org
# (https://huggingface.co/Beetle-HumanScale) under the paper's default regime:
#   mono (nld / eng baselines) + B1 (balanced) + B2 (simultaneous) +
#   B3 (sequential 33-67) + B4 (classroom) + B5 (late-onset).
#
# Scope:
#   * Org filter   : Beetle-HumanScale only (BeetleLM/* legacy excluded).
#   * Language set : {nld, eng} — strict subset, so fas-nld etc. are excluded.
#   * BLiMP suites : blimp_eng, blimp_nl, multiblimp.
#
# Execution order: per-model (each model finishes every P0 stage before the
# next begins), so interrupted jobs leave whole models complete rather than
# "PPL done for everyone, BLiMP done for no-one".
#
# Priority tiers:
#   P0 (default) : figure-critical core evals (ppl, convergence, meco,
#                  meco-checkpoints) + the BLiMP suite.
#   P1           : P0 + Beetle-vs-LLM baselines.
#   P2           : P1 + interpretability (representations, cognates, cli).
#
# 8×A100 fan-out: Stages 1 and 2 launch N_GPUS parallel processes with
# per-rank HF caches. Stage 3 (ledger merge, figures, BLiMP significance,
# ledger report) runs once on rank 0.
#
# Usage:
#   bash scripts/eval_nld-eng_beetle-human-scale-default.sh
#   PRIORITY=P1 bash scripts/...
#   N_GPUS=4 bash scripts/...
#   HF_CACHE_ROOT=/scratch/$USER/hf bash scripts/...
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."

# Adopt the shared safety contract: RAM-aware concurrency cap, MALLOC_ARENA_MAX,
# canonical HF cache (never tmpfs), setsid (SSH-SIGHUP immunity), and
# trap-protected peer termination on first child failure.
# shellcheck disable=SC1091
source scripts/_sweep_safety.sh
sweep_safety_init

LANG_PAIR="nld-eng"
LANGS=(nld eng)
ORGS=(Beetle-HumanScale)
FAMILY="humanscale"
CONDITIONS=(mono B1 B2 B3 B4 B5)
BLIMP_TASKS=(blimp_eng blimp_nl multiblimp)
OUT=results

PRIORITY="${PRIORITY:-P0}"         # P0 | P1 | P2 | all
ORDER="${ORDER:-per-model}"        # per-model | per-eval
N_GPUS="${N_GPUS:-8}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${BEETLE_HF_CACHE_ROOT}}"
# Checkpoint iteration scope for data stages AND figure variants.
#   all    — every step-N branch iterated; only *_trajectory figures emitted
#   latest — only highest step-N (or main); only *_final figures emitted
#   both   — (default) every step-N iterated AND both trajectory + final figures
CKPT_SCOPE="${CKPT_SCOPE:-both}"

# Stage-2 (BLiMP via legacy/eval_model.py) reads --latest_only only when
# CKPT_SCOPE=latest; otherwise every checkpoint is scored.
BLIMP_LATEST_FLAG=""
if [[ "$CKPT_SCOPE" == "latest" ]]; then
    BLIMP_LATEST_FLAG="--latest_only"
fi

echo "[$(date -u +%FT%TZ)] === Eval ${LANG_PAIR} (Beetle-HumanScale) ==="
echo "[$(date -u +%FT%TZ)] PRIORITY=${PRIORITY}  ORDER=${ORDER}  CKPT_SCOPE=${CKPT_SCOPE}"
echo "[$(date -u +%FT%TZ)] N_GPUS=${N_GPUS}  HF_CACHE_ROOT=${HF_CACHE_ROOT}"
echo "[$(date -u +%FT%TZ)] ORGS=${ORGS[*]}  BLIMP_TASKS=${BLIMP_TASKS[*]}"

mkdir -p "$HF_CACHE_ROOT" "$OUT/logs" "$OUT/ledger"

# ─── Stage 0 — backfill ledger from any existing CSVs (fast, idempotent) ────
echo "[$(date -u +%FT%TZ)] Stage 0 — backfill eval ledger from existing CSVs"
python -m analyze.eval_ledger backfill --output "$OUT" \
    > "$OUT/logs/ledger_backfill_${LANG_PAIR}.log" 2>&1 || true

# ─── Stage 1 — analyze.run core pipeline (N_GPUS rank fan-out) ──────────────
# Hand-rolled rank fan-outs were the failure mode that crashed two GPU
# nodes (no host-RAM cap, no peer trap, no SIGHUP detachment). Delegated
# to the shared safety library; per-rank logs land under
# $SWEEP_LOG_DIR/stage1_${LANG_PAIR}/rank_*.log.
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

# ─── Stage 2 — BLiMP suite on all checkpoints (N_GPUS rank fan-out) ─────────
# Resolve the org-scoped model list once; exported as MODEL_LIST below.
MODEL_LIST="$(python - <<PY
from models import MODEL_GROUPS
from analyze.model_naming import parse_repo

allowed_langs = {"nld", "eng"}
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
# legacy/eval_model.py needs --rank / --world_size as flags (it does not
# read them from env), so wrap with bash -c that interpolates the RANK
# and WORLD_SIZE that launch_ranks exports per-child.
SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-${OUT}/logs/sweep_${LANG_PAIR}}" \
launch_ranks "stage2_${LANG_PAIR}" \
    bash -c "python legacy/eval_model.py \
        --rank \"\$RANK\" --world_size \"\$WORLD_SIZE\" \
        --output_dir \"$(pwd)\" \
        --no_push \
        $BLIMP_LATEST_FLAG"

# ─── Stage 3 — ledger merge + figures + BLiMP significance (rank 0 only) ────
echo "[$(date -u +%FT%TZ)] Stage 3 — ledger merge + figures + significance"

python -m analyze.eval_ledger merge --output "$OUT" \
    > "$OUT/logs/ledger_merge_${LANG_PAIR}.log" 2>&1 || true

python -m analyze.run \
    --figures --paper-figures \
    --ckpt-scope "$CKPT_SCOPE" \
    --family "$FAMILY" \
    --orgs "${ORGS[@]}" \
    --langs "${LANGS[@]}" \
    --output-root "$OUT" \
    --rank 0 --world-size 1 \
    > "$OUT/logs/figures_${LANG_PAIR}.log" 2>&1 || true

python -m analyze.run_blimp_significance \
    --tasks "${BLIMP_TASKS[@]}" \
    --output "$OUT/blimp_significance/${LANG_PAIR}" \
    > "$OUT/logs/blimp_significance_${LANG_PAIR}.log" 2>&1 || true

python -m analyze.eval_ledger report --output "$OUT" \
    | tee "$OUT/logs/ledger_report_${LANG_PAIR}.log"

echo "[$(date -u +%FT%TZ)] === Done: ${LANG_PAIR} (Beetle-HumanScale) ==="
