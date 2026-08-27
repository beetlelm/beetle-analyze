#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# test_main_fast.sh — Maximally-parallel single-model eval at revision="main".
#
# Target: Beetle-HumanScale/beetle-bilingual-balanced-b1-humanscale-nld-eng
# Policy: all stages pinned to revision="main" via --ckpt-scope latest
#         (the hardcoded default; no trajectory iteration).
#
# Phase 1 — all 8 ranks run PPL + MECO + Convergence + AoA.
#   Within-stage sharding keeps every GPU busy:
#     * PPL score_sentences: stride-shards sentences across ranks
#     * MECO score_checkpoint: stride-shards stimuli across ranks
#     * AoA extract_surprisals: stride-shards eval sequences across ranks
#   Rank 0 reassembles per-rank shards into canonical CSVs / .npy arrays
#   after per-stage barriers.
#
# Phase 2 — GPU 0 only (but forced CPU-only for figures). Runs the rank-0-only
#   aggregators + figures + Ficarra post-processing:
#     --forgetting --embeddings --cognates --cli --aoa-ficarra
#     --figures --paper-figures
#
# 8×A100 optimisations (vs. previous version):
#   * --batch-size raised to $A100_BATCH_SIZE (default 64) — A100 80GB headroom
#   * TORCH_DTYPE=bfloat16 exported — ~1.6× forward-pass throughput on A100
#   * TOKENIZERS_PARALLELISM=false globally — suppress per-fork warning flood
#   * PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True — tighter cache
#   * Rank-0 HF pre-warm before parallel fan-out — saves 30–60 s cold start
#   * Unified Phase 1 stage list so the model is loaded once per rank
#   * Phase 2 runs on CPU (CUDA_VISIBLE_DEVICES="") — figure rendering is
#     matplotlib/pandas-only; frees GPU 0 and avoids CUDA init overhead
#   * wait -n early-abort when any rank fails, surfacing failures fast
#   * Ficarra CPU post-processing kicks in only in Phase 2, so Phase 1 GPU
#     time is unchanged — Ficarra reuses the surprisals emitted by --aoa
#
# Expected wall time on 8×A100 with warm HF cache: ~6–9 min.
#
# Usage
# -----
#   bash scripts/test_main_fast.sh                       # default: 8 GPUs
#   N_GPUS=4 bash scripts/test_main_fast.sh              # run on 4 GPUs
#   OUT_ROOT=/tmp/foo bash scripts/test_main_fast.sh     # custom output root
#   BEETLE_HELDOUT_MAX_SENTS=50000 bash scripts/...      # larger held-out
#   DRY_RUN=1 bash scripts/test_main_fast.sh             # preview commands
#   REPO=Beetle-HumanScale/other-model \
#       LANGS="nld eng" bash scripts/test_main_fast.sh   # swap target
#   FICARRA_FULL=1 bash scripts/test_main_fast.sh        # ϵ sweep
#   FICARRA_HEATMAP=1 bash scripts/test_main_fast.sh     # + Appendix-J heatmap
#   FICARRA_ALL_SIGS=1 bash scripts/test_main_fast.sh    # + σ_I / σ_R
#   SKIP_FICARRA=1 bash scripts/test_main_fast.sh        # disable Ficarra
#   A100_BATCH_SIZE=32 bash scripts/test_main_fast.sh    # smaller GPU card
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
OUT_ROOT="${OUT_ROOT:-results-main-fast}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${TMPDIR:-/tmp}/beetle_hf_cache_main_fast}"
DRY_RUN="${DRY_RUN:-0}"
A100_BATCH_SIZE="${A100_BATCH_SIZE:-64}"
SKIP_FICARRA="${SKIP_FICARRA:-0}"
FICARRA_FULL="${FICARRA_FULL:-0}"
FICARRA_HEATMAP="${FICARRA_HEATMAP:-0}"
FICARRA_ALL_SIGS="${FICARRA_ALL_SIGS:-0}"

# Ficarra's σ+/σ- split (first-half vs second-half contexts per word) benefits
# from more contexts per word for stable means. Raise the default heldout cap
# modestly (still <1 min prep on a single CPU).
: "${BEETLE_HELDOUT_MAX_SENTS:=40000}"
export BEETLE_HELDOUT_MAX_SENTS

# ─── Global perf env ────────────────────────────────────────────────────────
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ─── One-repo pool ──────────────────────────────────────────────────────────
POOL="$(mktemp -t beetle_pool.XXXXXX)"
echo "$REPO" > "$POOL"
trap 'rm -f "$POOL"' EXIT

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/ledger" "$HF_CACHE_ROOT"

log() { echo "[$(date -u +%FT%TZ)] $*"; }
hr()  { echo "──────────────────────────────────────────────────────────────"; }

preflight() {
    log "Pre-flight: $REPO"
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
    print(f'  FAIL $REPO: {e}'); raise
"
}

# Rank-0 HF pre-warm: resolve main revision + tokenizer ONCE before the fan-out
# so all parallel ranks can load from a warm cache instead of cold-downloading
# against the same Hub file on first access.
hf_prewarm() {
    log "HF pre-warm: $REPO @ main into $HF_CACHE_ROOT/rank_0"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '  DRY_RUN hf-prewarm %s main\n' "$REPO"
        return
    fi
    HF_HOME="$HF_CACHE_ROOT/rank_0" python -c "
from huggingface_hub import snapshot_download
try:
    snapshot_download('$REPO', revision='main',
                      allow_patterns=['*.json', '*.txt', 'tokenizer*', '*.safetensors'],
                      max_workers=8)
    print('  OK prewarm')
except Exception as e:
    print(f'  WARN prewarm: {e}')
" || true
}

log "test_main_fast.sh  N_GPUS=$N_GPUS  OUT_ROOT=$OUT_ROOT  DRY_RUN=$DRY_RUN"
log "  target       : $REPO"
log "  langs / orgs : ${LANGS[*]} / ${ORGS[*]}"
log "  hf cache     : $HF_CACHE_ROOT"
log "  heldout cap  : $BEETLE_HELDOUT_MAX_SENTS sentences"
log "  batch size   : $A100_BATCH_SIZE"
log "  torch dtype  : $TORCH_DTYPE"
log "  ficarra      : skip=$SKIP_FICARRA  full=$FICARRA_FULL  heatmap=$FICARRA_HEATMAP  all_sigs=$FICARRA_ALL_SIGS"
preflight
hf_prewarm
hr

# ════════════════════════════════════════════════════════════════════════════
# Phase 1 — all ranks run every model-running stage with within-stage sharding
# ════════════════════════════════════════════════════════════════════════════
log "Phase 1 — PPL + MECO + Convergence + AoA  (world_size=$N_GPUS, main-only)"
log "  follow rank 0:  tail -F $OUT_ROOT/logs/phase1_rank0.log"

declare -a pids=()
for RANK in $(seq 0 $((N_GPUS-1))); do
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '  DRY_RUN [rank %d] python -m analyze.run --ppl --meco --convergence --aoa --resume --ckpt-scope latest --family %s --orgs %s --langs %s --cache-mode keep --output-root %s --batch-size %d --rank %d --world-size %d\n' \
               "$RANK" "$POOL" "${ORGS[*]}" "${LANGS[*]}" "$OUT_ROOT" "$A100_BATCH_SIZE" "$RANK" "$N_GPUS"
        continue
    fi
    CUDA_VISIBLE_DEVICES="$RANK" \
      HF_HOME="$HF_CACHE_ROOT/rank_$RANK" \
      RANK="$RANK" WORLD_SIZE="$N_GPUS" \
      PYTHONUNBUFFERED=1 \
      python -m analyze.run \
          --ppl --meco --convergence --aoa \
          --resume --ckpt-scope latest \
          --family "$POOL" --orgs "${ORGS[@]}" --langs "${LANGS[@]}" \
          --cache-mode keep --output-root "$OUT_ROOT" \
          --batch-size "$A100_BATCH_SIZE" \
          --rank "$RANK" --world-size "$N_GPUS" \
          > "$OUT_ROOT/logs/phase1_rank${RANK}.log" 2>&1 &
    pids+=($!)
done
# Early-abort: if any rank fails, kill the rest instead of waiting serially.
phase1_fail=0
if [[ ${#pids[@]} -gt 0 ]]; then
    remaining=${#pids[@]}
    while (( remaining > 0 )); do
        if wait -n; then
            :
        else
            phase1_fail=1
            log "Phase 1: a rank exited non-zero; cancelling siblings."
            for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
            for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
            break
        fi
        remaining=$((remaining - 1))
    done
fi
if (( phase1_fail != 0 )); then
    log "Phase 1 FAILED — see $OUT_ROOT/logs/phase1_rank*.log"
    exit 1
fi
log "Phase 1 done"
hr

# ════════════════════════════════════════════════════════════════════════════
# Phase 2 — rank-0-only aggregators + figures + Ficarra (CPU)
#
# Figure rendering is matplotlib/pandas-only; no CUDA needed. Ficarra
# post-processing (signatures + ϵ-convergence AoA) is also pure CPU.
# Running on CPU frees GPU 0 and avoids ~3 s of CUDA init per invocation.
# ════════════════════════════════════════════════════════════════════════════
FICARRA_ARGS=()
if [[ "$SKIP_FICARRA" != "1" ]]; then
    FICARRA_ARGS+=(--aoa-ficarra)
    if [[ "$FICARRA_FULL" == "1" ]]; then
        FICARRA_ARGS+=(--ficarra-full-sweep)
    fi
    if [[ "$FICARRA_HEATMAP" == "1" ]]; then
        FICARRA_ARGS+=(--ficarra-heatmap)
    fi
    if [[ "$FICARRA_ALL_SIGS" == "1" ]]; then
        FICARRA_ARGS+=(--ficarra-all-signatures)
    fi
fi

log "Phase 2 — aggregators + figures + Ficarra (CPU; rank 0 only)"
if [[ "$DRY_RUN" == "1" ]]; then
    printf '  DRY_RUN python -m analyze.run --forgetting --embeddings --cognates --cli %s --figures --paper-figures --ckpt-scope latest --family %s --orgs %s --langs %s --output-root %s --rank 0 --world-size 1\n' \
           "${FICARRA_ARGS[*]-}" "$POOL" "${ORGS[*]}" "${LANGS[*]}" "$OUT_ROOT"
else
    CUDA_VISIBLE_DEVICES="" \
      HF_HOME="$HF_CACHE_ROOT/rank_0" \
      PYTHONUNBUFFERED=1 \
      python -m analyze.run \
          --forgetting --embeddings --cognates --cli \
          "${FICARRA_ARGS[@]}" \
          --figures --paper-figures \
          --ckpt-scope latest --cache-mode keep \
          --family "$POOL" --orgs "${ORGS[@]}" --langs "${LANGS[@]}" \
          --output-root "$OUT_ROOT" --rank 0 --world-size 1 \
          2>&1 | tee "$OUT_ROOT/logs/phase2.log"
fi
log "Phase 2 done"
hr

# ─── Ledger report ──────────────────────────────────────────────────────────
if [[ "$DRY_RUN" != "1" ]]; then
    log "Ledger report (from $OUT_ROOT)"
    python -m analyze.eval_ledger report --output "$OUT_ROOT" \
        | tee "$OUT_ROOT/logs/ledger_report.log" || true
fi

log "test_main_fast.sh — done."
log "Figures at:  $OUT_ROOT/figures/     Logs at: $OUT_ROOT/logs/"
if [[ "$SKIP_FICARRA" != "1" ]]; then
    log "Ficarra outputs at: $OUT_ROOT/aoa/ficarra/ and $OUT_ROOT/figures/ficarra/"
fi
