#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# test_eval_library.sh — End-to-end smoke test of the beetle-analyze eval
# library on a single model plus one external baseline.
#
# Exercises four scenarios in sequence against the same repo
#   Beetle-HumanScale/beetle-bilingual-balanced-b1-humanscale-nld-eng:
#
#   (1) FINAL-CHECKPOINT EVAL     analyze.run --all  --ckpt-scope latest
#   (2) ALL-CHECKPOINTS EVAL      analyze.run --all  --ckpt-scope all
#   (3) BASELINE MECO + BLIMP     analyze.run --meco --meco-models <EuroLLM>
#                                 legacy/eval_model.py with MODEL_LIST=<EuroLLM>
#   (4) FIGURE GENERATION         analyze.run --figures --paper-figures --ckpt-scope both
#
# Each scenario writes into its own output root so logs/CSVs don't mix:
#
#   results-test/latest/     ← scenario 1
#   results-test/all/        ← scenario 2
#   results-test/baseline/   ← scenario 3
#   results-test/all/        ← scenario 4 renders figures from scenario-2 data
#
# Scenarios are gated by SCENARIOS env var (default: "1 2 3 4") so a targeted
# re-run can skip expensive stages while debugging. Set N_GPUS=1 for single-GPU
# boxes.
#
# Usage
# -----
#   bash scripts/test_eval_library.sh                      # all four scenarios
#   SCENARIOS="1 4" bash scripts/test_eval_library.sh      # only final + figures
#   N_GPUS=1        bash scripts/test_eval_library.sh      # single GPU
#   DRY_RUN=1       bash scripts/test_eval_library.sh      # print, don't exec
#
# Speed knobs (optional)
# ----------------------
#   BEETLE_HELDOUT_MAX_SENTS=50000   # cap AoA held-out stream to 50k sentences
#                                    # instead of the default 2,000,000. Useful
#                                    # for quick smoke-tests; set lower still
#                                    # for end-to-end wiring checks.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."

# ─── Target model ───────────────────────────────────────────────────────────
REPO="${REPO:-Beetle-HumanScale/beetle-bilingual-balanced-b1-humanscale-nld-eng}"
BASELINE_REPO="${BASELINE_REPO:-utter-project/EuroLLM-1.7B}"

# ─── Scope ──────────────────────────────────────────────────────────────────
LANGS=(nld eng)
ORGS=(Beetle-HumanScale)
BLIMP_TASKS=(blimp_eng blimp_nl multiblimp)

# ─── Runtime config ─────────────────────────────────────────────────────────
N_GPUS="${N_GPUS:-8}"
SCENARIOS="${SCENARIOS:-1 2 3 4}"
OUT_ROOT="${OUT_ROOT:-results-test}"
OUT_LATEST="$OUT_ROOT/latest"
OUT_ALL="$OUT_ROOT/all"
OUT_BASELINE="$OUT_ROOT/baseline"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${TMPDIR:-/tmp}/beetle_hf_cache_test}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$HF_CACHE_ROOT" \
         "$OUT_LATEST/logs"   "$OUT_LATEST/ledger" \
         "$OUT_ALL/logs"      "$OUT_ALL/ledger" \
         "$OUT_BASELINE/logs"

# ─── One-repo pool (reused for scenarios 1, 2, 4) ───────────────────────────
POOL="$(mktemp -t beetle_pool.XXXXXX)"
echo "$REPO" > "$POOL"
trap 'rm -f "$POOL"' EXIT

log()  { echo "[$(date -u +%FT%TZ)] $*"; }
hr()   { echo "──────────────────────────────────────────────────────────────"; }
have() { [[ " $SCENARIOS " == *" $1 "* ]]; }

# Dry-run / exec wrapper. Prints the command when DRY_RUN=1, else runs it.
run() {
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '  DRY_RUN %s\n' "$*"
    else
        eval "$*"
    fi
}

# Pre-flight — confirm the model repos are reachable so we fail fast.
preflight_check() {
    log "Pre-flight: verifying target repos are reachable"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '  DRY_RUN hf model-info %s\n' "$REPO"
        printf '  DRY_RUN hf model-info %s\n' "$BASELINE_REPO"
        return
    fi
    python -c "
from huggingface_hub import HfApi
api = HfApi()
for r in ['$REPO', '$BASELINE_REPO']:
    try:
        api.model_info(r)
        print(f'  OK  {r}')
    except Exception as e:
        print(f'  FAIL {r}: {e}')
        raise
"
}

log "test_eval_library.sh — SCENARIOS='$SCENARIOS'  N_GPUS=$N_GPUS  DRY_RUN=$DRY_RUN"
log "  target       : $REPO"
log "  baseline     : $BASELINE_REPO"
log "  langs / orgs : ${LANGS[*]} / ${ORGS[*]}"
log "  blimp tasks  : ${BLIMP_TASKS[*]}"
log "  output root  : $OUT_ROOT"
log "  hf cache     : $HF_CACHE_ROOT"
preflight_check
hr

# ════════════════════════════════════════════════════════════════════════════
# Scenario 1 — Final-checkpoint eval on the target model, all stages
# ════════════════════════════════════════════════════════════════════════════
if have 1; then
    log "Scenario 1 ── FINAL CHECKPOINT  (--ckpt-scope latest)"
    log "  → $OUT_LATEST"
    declare -a s1_pids=()
    for RANK in $(seq 0 $((N_GPUS-1))); do
        if [[ "$DRY_RUN" == "1" ]]; then
            printf '  DRY_RUN [rank %d] python -m analyze.run --all --resume --ckpt-scope latest --family %s --orgs %s --langs %s --cache-mode keep --output-root %s --rank %d --world-size %d\n' \
                   "$RANK" "$POOL" "${ORGS[*]}" "${LANGS[*]}" "$OUT_LATEST" "$RANK" "$N_GPUS"
            continue
        fi
        CUDA_VISIBLE_DEVICES="$RANK" \
        HF_HOME="${HF_CACHE_ROOT}/rank_${RANK}" \
        RANK="$RANK" WORLD_SIZE="$N_GPUS" \
        python -m analyze.run \
            --all --resume \
            --ckpt-scope latest \
            --family "$POOL" \
            --orgs "${ORGS[@]}" \
            --langs "${LANGS[@]}" \
            --cache-mode keep \
            --output-root "$OUT_LATEST" \
            --rank "$RANK" --world-size "$N_GPUS" \
            > "$OUT_LATEST/logs/s1_analyze_rank${RANK}.log" 2>&1 &
        s1_pids+=($!)
    done
    if [[ ${#s1_pids[@]} -gt 0 ]]; then for pid in "${s1_pids[@]}"; do wait "$pid"; done; fi
    log "Scenario 1 done"
    hr
fi

# ════════════════════════════════════════════════════════════════════════════
# Scenario 2 — All-checkpoints eval on the target model, all stages
# ════════════════════════════════════════════════════════════════════════════
if have 2; then
    log "Scenario 2 ── ALL CHECKPOINTS  (--ckpt-scope all)"
    log "  → $OUT_ALL"
    declare -a s2_pids=()
    for RANK in $(seq 0 $((N_GPUS-1))); do
        if [[ "$DRY_RUN" == "1" ]]; then
            printf '  DRY_RUN [rank %d] python -m analyze.run --all --resume --ckpt-scope all --family %s --orgs %s --langs %s --cache-mode keep --output-root %s --rank %d --world-size %d\n' \
                   "$RANK" "$POOL" "${ORGS[*]}" "${LANGS[*]}" "$OUT_ALL" "$RANK" "$N_GPUS"
            continue
        fi
        CUDA_VISIBLE_DEVICES="$RANK" \
        HF_HOME="${HF_CACHE_ROOT}/rank_${RANK}" \
        RANK="$RANK" WORLD_SIZE="$N_GPUS" \
        python -m analyze.run \
            --all --resume \
            --ckpt-scope all \
            --family "$POOL" \
            --orgs "${ORGS[@]}" \
            --langs "${LANGS[@]}" \
            --cache-mode keep \
            --output-root "$OUT_ALL" \
            --rank "$RANK" --world-size "$N_GPUS" \
            > "$OUT_ALL/logs/s2_analyze_rank${RANK}.log" 2>&1 &
        s2_pids+=($!)
    done
    if [[ ${#s2_pids[@]} -gt 0 ]]; then for pid in "${s2_pids[@]}"; do wait "$pid"; done; fi
    log "Scenario 2 done"
    hr
fi

# ════════════════════════════════════════════════════════════════════════════
# Scenario 3 — MECO + BLiMP for the external baseline (EuroLLM-1.7B)
# MECO: --meco-models bypasses _resolve_repos (family/org filters).
# BLiMP: legacy/eval_model.py reads MODEL_LIST env to drive evaluation.
# ════════════════════════════════════════════════════════════════════════════
if have 3; then
    log "Scenario 3 ── BASELINE MECO + BLIMP  ($BASELINE_REPO)"
    log "  → $OUT_BASELINE"

    # 3a) MECO — latest checkpoint is enough for a baseline LLM.
    log "  3a) MECO on $BASELINE_REPO"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '  DRY_RUN python -m analyze.run --meco --meco-only --meco-models %s --ckpt-scope latest --langs %s --cache-mode keep --output-root %s --rank 0 --world-size 1\n' \
               "$BASELINE_REPO" "${LANGS[*]}" "$OUT_BASELINE"
    else
        CUDA_VISIBLE_DEVICES=0 \
        HF_HOME="${HF_CACHE_ROOT}/rank_0" \
        python -m analyze.run \
            --meco --meco-only \
            --meco-models "$BASELINE_REPO" \
            --ckpt-scope latest \
            --langs "${LANGS[@]}" \
            --cache-mode keep \
            --output-root "$OUT_BASELINE" \
            --rank 0 --world-size 1 \
            > "$OUT_BASELINE/logs/s3a_meco_baseline.log" 2>&1
    fi

    # 3b) BLiMP — ORGS is intentionally NOT set (MODEL_LIST already selects
    # the model; org filtering would otherwise drop utter-project/*).
    log "  3b) BLiMP on $BASELINE_REPO"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '  DRY_RUN MODEL_LIST=%s BLIMP_TASKS="%s" python legacy/eval_model.py --rank 0 --world_size 1 --output_dir %s --no_push --latest_only\n' \
               "$BASELINE_REPO" "${BLIMP_TASKS[*]}" "$OUT_BASELINE"
    else
        CUDA_VISIBLE_DEVICES=0 \
        HF_HOME="${HF_CACHE_ROOT}/rank_0" \
        MODEL_LIST="$BASELINE_REPO" \
        BLIMP_TASKS="${BLIMP_TASKS[*]}" \
        python legacy/eval_model.py \
            --rank 0 --world_size 1 \
            --output_dir "$OUT_BASELINE" \
            --no_push \
            --latest_only \
            > "$OUT_BASELINE/logs/s3b_blimp_baseline.log" 2>&1
    fi
    log "Scenario 3 done"
    hr
fi

# ════════════════════════════════════════════════════════════════════════════
# Scenario 4 — Figure generation for the target model
# Trajectory figures need scenario-2 data; final-snapshot figures need
# scenario-1 OR scenario-2. --ckpt-scope both emits both variants.
# ════════════════════════════════════════════════════════════════════════════
if have 4; then
    log "Scenario 4 ── FIGURES  (--figures --paper-figures, ckpt-scope=both)"
    log "  → $OUT_ALL (trajectory + final)"

    if [[ ! -d "$OUT_ALL" ]] && ! have 2; then
        log "  WARNING: $OUT_ALL not found and scenario 2 is disabled — "
        log "  trajectory figures will be empty. Enable SCENARIOS='2 4'."
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
        printf '  DRY_RUN python -m analyze.run --figures --paper-figures --ckpt-scope both --family %s --orgs %s --langs %s --output-root %s --rank 0 --world-size 1\n' \
               "$POOL" "${ORGS[*]}" "${LANGS[*]}" "$OUT_ALL"
    else
        python -m analyze.run \
            --figures --paper-figures \
            --ckpt-scope both \
            --family "$POOL" \
            --orgs "${ORGS[@]}" \
            --langs "${LANGS[@]}" \
            --output-root "$OUT_ALL" \
            --rank 0 --world-size 1 \
            > "$OUT_ALL/logs/s4_figures.log" 2>&1
    fi
    log "Scenario 4 done"
    hr
fi

# ─── Final ledger report for the trajectory run (best-effort) ───────────────
if [[ "$DRY_RUN" != "1" ]] && have 2; then
    log "Ledger report (from $OUT_ALL)"
    python -m analyze.eval_ledger report --output "$OUT_ALL" \
        | tee "$OUT_ALL/logs/ledger_report.log" || true
fi

log "test_eval_library.sh — done."
