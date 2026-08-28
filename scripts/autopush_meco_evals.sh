#!/usr/bin/env bash
# autopush_meco_evals.sh — commit + push every hour to the meco-evals branch.
#
# Targeted paths only (results/meco/, results/evals/meco/, results/ledger/,
# logs/meco_baselines/, scripts/eval_meco_baselines_24h.sh,
# scripts/autopush_meco_evals.sh, scripts/run_emnlp_meco_24h.sh).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BRANCH="${BRANCH:-meco-evals}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-3600}"
HARD_WALLTIME_SECONDS="${HARD_WALLTIME_SECONDS:-86400}"
LOG_DIR="$ROOT/logs/meco_baselines"
mkdir -p "$LOG_DIR"
PUSH_LOG="$LOG_DIR/autopush_$(date -u +%Y%m%dT%H%M%SZ).log"

PATHS=(
    "results/meco"
    "results/evals/meco"
    "results/ledger"
    "results/figures/main"
    "logs/meco_baselines"
    "analyze/meco_vs_baselines.py"
    "scripts/fit_meco_baselines_evals.py"
    "scripts/eval_meco_baselines_24h.sh"
    "scripts/autopush_meco_evals.sh"
    "scripts/run_emnlp_meco_24h.sh"
)

echo "[$(date -u +%FT%TZ)] autopush start  branch=$BRANCH interval=${INTERVAL_SECONDS}s walltime=${HARD_WALLTIME_SECONDS}s" \
    | tee -a "$PUSH_LOG"

# Worktree is already on $BRANCH; if for some reason it's not, switch.
if [[ "$(git rev-parse --abbrev-ref HEAD)" != "$BRANCH" ]]; then
    git switch "$BRANCH" 2>&1 | tee -a "$PUSH_LOG"
fi

t0=$(date +%s)
iter=0

push_once() {
    local label="$1"
    iter=$((iter + 1))
    echo "[$(date -u +%FT%TZ)] autopush iter=$iter ($label)" | tee -a "$PUSH_LOG"

    # Refit any new surprisal CSVs into ΔlogLik, then regen all figures so
    # every push reflects the latest GPU-1 progress.
    PY=${PY:-python}
    if [[ -x "$PY" ]]; then
        "$PY" scripts/fit_meco_baselines_evals.py >>"$PUSH_LOG" 2>&1 || true
        "$PY" -m analyze.meco_vs_baselines        >>"$PUSH_LOG" 2>&1 || true
    fi

    for p in "${PATHS[@]}"; do
        if [[ -e "$p" ]]; then
            git add -- "$p" 2>>"$PUSH_LOG" || true
        fi
    done

    if git diff --cached --quiet; then
        echo "  no staged changes — nothing to push" | tee -a "$PUSH_LOG"
        return 0
    fi

    local ts
    ts="$(date -u +%FT%TZ)"
    if ! git -c commit.gpgsign=false commit \
            -m "meco-evals autopush: ${label} @ ${ts}" 2>>"$PUSH_LOG"; then
        echo "  commit failed" | tee -a "$PUSH_LOG"
        return 0
    fi

    if git push -u origin "$BRANCH" 2>>"$PUSH_LOG"; then
        echo "  push ok" | tee -a "$PUSH_LOG"
    else
        echo "  push failed (will retry next iter)" | tee -a "$PUSH_LOG"
    fi
}

push_once "initial"

while true; do
    elapsed=$(( $(date +%s) - t0 ))
    if (( elapsed >= HARD_WALLTIME_SECONDS )); then
        echo "[$(date -u +%FT%TZ)] autopush hit walltime — final push then exit" | tee -a "$PUSH_LOG"
        push_once "final-walltime"
        exit 0
    fi
    sleep "$INTERVAL_SECONDS"
    push_once "hourly"
done
