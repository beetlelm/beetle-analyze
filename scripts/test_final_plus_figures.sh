#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# test_final_plus_figures.sh — Narrow smoke test: final-checkpoint eval of
# one model, then render its figures. Runs two phases only:
#
#   (1) analyze.run --all --ckpt-scope latest --family $POOL
#        → every stage (PPL, convergence, MECO, AoA, forgetting, embeddings,
#          cognates, CLI) evaluated ONLY at the highest step-N branch
#   (2) analyze.run --figures --paper-figures --ckpt-scope latest
#        → final-snapshot figures rendered from the CSVs scenario 1 produced
#
# This is intentionally simpler than test_eval_library.sh. No BLiMP, no
# baselines, no trajectory figures, no all-checkpoints sweep.
#
# The eval library emits tqdm progress bars at every hot loop:
#   - analyze/ppl_eval.py      → one bar per iso_code over (repo, revision)
#   - analyze/ppl_utils.py     → inner per-sentence bar under each checkpoint
#   - analyze/convergence.py   → one bar over the rank's repo list
#   - analyze/meco_surprisal.py→ one bar over models + one over step-branches
#   - aoa/run_learning_curves  → one bar over (lang, model) work items
#   - aoa/prepare_eval_data    → streaming bar while pulling held-out sents
#   - aoa/get_beetle_surprisals→ already bars per checkpoint and per batch
#
# Usage
# -----
#   bash scripts/test_final_plus_figures.sh              # 8 GPUs (default)
#   N_GPUS=1 bash scripts/test_final_plus_figures.sh     # single GPU
#   DRY_RUN=1 bash scripts/test_final_plus_figures.sh    # preview commands
#
#   # Smoke-test held-out streaming faster (50k sentences instead of 2M):
#   # Default is 20k (fast). Raise for full-fidelity runs:
#   BEETLE_HELDOUT_MAX_SENTS=200000 bash scripts/test_final_plus_figures.sh
#
#   # Swap the target model:
#   REPO=Beetle-HumanScale/beetle-monolingual-humanscale-nld \
#       LANGS="nld" bash scripts/test_final_plus_figures.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."

# ─── Target ─────────────────────────────────────────────────────────────────
REPO="${REPO:-Beetle-HumanScale/beetle-bilingual-balanced-b1-humanscale-nld-eng}"
LANGS_STR="${LANGS:-nld eng}"
read -ra LANGS <<<"$LANGS_STR"
ORGS=(Beetle-HumanScale)

# ─── Runtime ────────────────────────────────────────────────────────────────
N_GPUS="${N_GPUS:-8}"
OUT_ROOT="${OUT_ROOT:-results-final}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${TMPDIR:-/tmp}/beetle_hf_cache_final}"
DRY_RUN="${DRY_RUN:-0}"
# CPU_ONLY=1 forces CUDA_VISIBLE_DEVICES="" so Torch falls back to CPU.
# Useful for smoke-testing the pipeline wiring on a laptop.
CPU_ONLY="${CPU_ONLY:-0}"

mkdir -p "$HF_CACHE_ROOT" "$OUT_ROOT/logs" "$OUT_ROOT/ledger"

# ─── One-repo pool ──────────────────────────────────────────────────────────
POOL="$(mktemp -t beetle_pool.XXXXXX)"
echo "$REPO" > "$POOL"
trap 'rm -f "$POOL"' EXIT

log() { echo "[$(date -u +%FT%TZ)] $*"; }
hr()  { echo "──────────────────────────────────────────────────────────────"; }

preflight_check() {
    log "Pre-flight: verifying $REPO is reachable"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '  DRY_RUN hf model-info %s\n' "$REPO"
        return
    fi
    python -c "
from huggingface_hub import HfApi
try:
    HfApi().model_info('$REPO')
    print('  OK $REPO')
except Exception as e:
    print(f'  FAIL $REPO: {e}')
    raise
"
}

log "test_final_plus_figures.sh  N_GPUS=$N_GPUS  DRY_RUN=$DRY_RUN  CPU_ONLY=$CPU_ONLY"
log "  target      : $REPO"
log "  langs/orgs  : ${LANGS[*]} / ${ORGS[*]}"
log "  output root : $OUT_ROOT"
log "  hf cache    : $HF_CACHE_ROOT"
[[ -n "${BEETLE_HELDOUT_MAX_SENTS:-}" ]] && \
    log "  heldout cap : $BEETLE_HELDOUT_MAX_SENTS sentences"
preflight_check
hr

# ════════════════════════════════════════════════════════════════════════════
# Phase 1 — Final-checkpoint eval, all stages
# ════════════════════════════════════════════════════════════════════════════
log "Phase 1 ── FINAL CHECKPOINT  (--ckpt-scope latest, --all --resume)"
log "  → $OUT_ROOT"
log "  per-rank logs tail:  tail -F $OUT_ROOT/logs/s1_analyze_rank0.log"

declare -a pids=()
for RANK in $(seq 0 $((N_GPUS-1))); do
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '  DRY_RUN [rank %d] python -m analyze.run --all --resume --ckpt-scope latest --family %s --orgs %s --langs %s --cache-mode keep --output-root %s --rank %d --world-size %d\n' \
               "$RANK" "$POOL" "${ORGS[*]}" "${LANGS[*]}" "$OUT_ROOT" "$RANK" "$N_GPUS"
        continue
    fi
    if [[ "$CPU_ONLY" == "1" ]]; then CUDA_DEV=""; else CUDA_DEV="$RANK"; fi
    CUDA_VISIBLE_DEVICES="$CUDA_DEV" \
    HF_HOME="${HF_CACHE_ROOT}/rank_${RANK}" \
    RANK="$RANK" WORLD_SIZE="$N_GPUS" \
    PYTHONUNBUFFERED=1 \
    python -m analyze.run \
        --all --resume \
        --ckpt-scope latest \
        --family "$POOL" \
        --orgs "${ORGS[@]}" \
        --langs "${LANGS[@]}" \
        --cache-mode keep \
        --output-root "$OUT_ROOT" \
        --rank "$RANK" --world-size "$N_GPUS" \
        > "$OUT_ROOT/logs/s1_analyze_rank${RANK}.log" 2>&1 &
    pids+=($!)
done
if [[ ${#pids[@]} -gt 0 ]]; then for pid in "${pids[@]}"; do wait "$pid"; done; fi
log "Phase 1 done"
hr

# ════════════════════════════════════════════════════════════════════════════
# Phase 2 — Final-snapshot figures
# --ckpt-scope latest emits only *_final figures (matching scenario-1 data).
# ════════════════════════════════════════════════════════════════════════════
log "Phase 2 ── FIGURES  (--figures --paper-figures, ckpt-scope=latest)"
log "  → $OUT_ROOT/figures"

if [[ "$DRY_RUN" == "1" ]]; then
    printf '  DRY_RUN python -m analyze.run --figures --paper-figures --ckpt-scope latest --family %s --orgs %s --langs %s --output-root %s --rank 0 --world-size 1\n' \
           "$POOL" "${ORGS[*]}" "${LANGS[*]}" "$OUT_ROOT"
else
    PYTHONUNBUFFERED=1 \
    python -m analyze.run \
        --figures --paper-figures \
        --ckpt-scope latest \
        --family "$POOL" \
        --orgs "${ORGS[@]}" \
        --langs "${LANGS[@]}" \
        --output-root "$OUT_ROOT" \
        --rank 0 --world-size 1 \
        2>&1 | tee "$OUT_ROOT/logs/s2_figures.log"
fi
log "Phase 2 done"
hr

# ─── Ledger report (best-effort) ────────────────────────────────────────────
if [[ "$DRY_RUN" != "1" ]]; then
    log "Ledger report (from $OUT_ROOT)"
    python -m analyze.eval_ledger report --output "$OUT_ROOT" \
        | tee "$OUT_ROOT/logs/ledger_report.log" || true
fi

log "test_final_plus_figures.sh — done."
log "Figures at: $OUT_ROOT/figures  |  Logs at: $OUT_ROOT/logs"
