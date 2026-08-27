#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_slt_sweep.sh — 4-day SLT geometry sweep on 8×A100.
#
# Stages (in order):
#   D1  setup + smoke test on pico-decoder-tiny
#   D1  --llc       on 50 cells (8 models × 6 ckpts + 2 falsification)
#   D2  --fisher    on 50 cells
#   D2  --susceptibilities on 50 cells (heaviest stage)
#   D3  --bridge --per-lang-llc
#   D3  --beta-sweep on 4 anchor cells
#   D4  --slt-aggregate --slt-plot --paper-figures
#   D4  reruns of any non-converged cells
#
# Usage:
#   bash scripts/run_slt_sweep.sh                       # full sweep
#   STAGE=smoke bash scripts/run_slt_sweep.sh           # day-1 morning
#   STAGE=llc   bash scripts/run_slt_sweep.sh
#   STAGE=fisher | susceptibilities | bridge | aggregate
#   N_GPUS=4 bash scripts/run_slt_sweep.sh
#
# Anchor cells (one per regime) are pinned below.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."

# Reuse the safety library if present (RAM caps, peer trap, SIGHUP-immune).
if [[ -f scripts/_sweep_safety.sh ]]; then
    # shellcheck disable=SC1091
    source scripts/_sweep_safety.sh
    sweep_safety_init
fi

OUT="${OUT:-results}"
N_GPUS="${N_GPUS:-8}"
STAGE="${STAGE:-all}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${BEETLE_HF_CACHE_ROOT:-${HOME}/.cache/huggingface}}"

# Anchor cells for β-sweep + falsification. Update with the pinned repo
# strings once models.py has been consulted on day 1.
ANCHOR_MONO="${ANCHOR_MONO:-pico-lm/pico-decoder-tiny}"
ANCHOR_BI="${ANCHOR_BI:-Beetle-nld-eng-Variants/balanced}"
ANCHOR_MULTI="${ANCHOR_MULTI:-Beetle-FineWeb-100M/main}"
ANCHOR_PICO="${ANCHOR_PICO:-pico-lm/pico-decoder-small}"

A2A_REPOS="${A2A_REPOS:-Beetle-nld-eng-Variants/simultaneous}"
A2B_REPOS="${A2B_REPOS:-Beetle-FineWeb-100M/main}"

mkdir -p "$OUT/logs/slt" "$HF_CACHE_ROOT"

run_ranks() {
    # $1 = stage label, $2.. = python args
    local label="$1"; shift
    if declare -F launch_ranks >/dev/null 2>&1; then
        SWEEP_LOG_DIR="$OUT/logs/slt/${label}" launch_ranks "${label}" python -m analyze.run "$@"
    else
        echo "[$(date -u +%FT%TZ)] WARN: launch_ranks not available — running rank 0 only"
        RANK=0 WORLD_SIZE=1 python -m analyze.run "$@"
    fi
}

stage_smoke() {
    echo "[$(date -u +%FT%TZ)] === smoke test ==="
    python -m slt.run_llc \
        --model pico-lm/pico-decoder-tiny --revision main \
        --num-chains 1 --num-draws 20 --batch-size 4 \
        --output-root "$OUT" --smoke
}

stage_llc() {
    echo "[$(date -u +%FT%TZ)] === --llc + --falsification ==="
    run_ranks "llc" \
        --llc --all-checkpoints \
        --output-root "$OUT" --cache-mode keep
    run_ranks "falsification" \
        --falsification \
        --falsification-a2a-repos $A2A_REPOS \
        --falsification-a2b-repos $A2B_REPOS \
        --output-root "$OUT" --cache-mode keep
}

stage_fisher() {
    echo "[$(date -u +%FT%TZ)] === --fisher ==="
    run_ranks "fisher" \
        --fisher --all-checkpoints \
        --output-root "$OUT" --cache-mode keep
}

stage_sus() {
    echo "[$(date -u +%FT%TZ)] === --susceptibilities ==="
    run_ranks "susceptibilities" \
        --susceptibilities --all-checkpoints \
        --output-root "$OUT" --cache-mode keep
}

stage_bridge() {
    echo "[$(date -u +%FT%TZ)] === --bridge --per-lang-llc ==="
    RANK=0 WORLD_SIZE=1 python -m analyze.run \
        --bridge --per-lang-llc \
        --output-root "$OUT"
}

stage_beta() {
    echo "[$(date -u +%FT%TZ)] === --beta-sweep on anchors ==="
    run_ranks "beta_sweep" \
        --beta-sweep \
        --beta-sweep-anchors "$ANCHOR_MONO" "$ANCHOR_BI" "$ANCHOR_MULTI" "$ANCHOR_PICO" \
        --output-root "$OUT" --cache-mode keep
}

stage_aggregate() {
    echo "[$(date -u +%FT%TZ)] === --slt-aggregate --slt-plot ==="
    RANK=0 WORLD_SIZE=1 python -m analyze.run \
        --slt-aggregate --slt-plot \
        --output-root "$OUT"
    (cd slt/paper && make pdf || echo "  paper compile skipped (no latex)")
}

case "$STAGE" in
    smoke)            stage_smoke ;;
    llc)              stage_llc ;;
    fisher)           stage_fisher ;;
    susceptibilities) stage_sus ;;
    bridge)           stage_bridge ;;
    beta)             stage_beta ;;
    aggregate)        stage_aggregate ;;
    all)
        stage_smoke
        stage_llc
        stage_fisher
        stage_sus
        stage_bridge
        stage_beta
        stage_aggregate
        ;;
    *)
        echo "Unknown STAGE='$STAGE'. Use one of: smoke llc fisher susceptibilities bridge beta aggregate all" >&2
        exit 1
        ;;
esac

echo "[$(date -u +%FT%TZ)] === run_slt_sweep.sh ($STAGE) complete ==="
