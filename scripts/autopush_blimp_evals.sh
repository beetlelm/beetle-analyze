#!/usr/bin/env bash
# autopush_blimp_evals.sh — commit + push every hour to the blimp-evals branch.
#
# Targeted paths only — we don't want to sweep up unrelated working-tree changes.
#   - results/figures/main/fig_blimp_l1l2_*.{pdf,png}
#   - results/figures/appendix/fig_blimp_l1l2_grid.{pdf,png}
#   - results/blimp/*_results.csv
#   - results/evals/blimp/blimp_results.csv
#   - logs/blimp_scale_curriculum/*.log
#   - analyze/blimp_scale_curriculum.py
#   - analyze/blimp_eval_oneshot.py
#   - scripts/eval_blimp_scale_curriculum.sh
#   - scripts/autopush_blimp_evals.sh
#   - scripts/run_emnlp_blimp_24h.sh
#
# Loops until the eval process exits OR the script's hard walltime expires.
# Exits 0 either way; the tmux session can stop normally.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BRANCH="${BRANCH:-blimp-evals}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-3600}"        # 1h
HARD_WALLTIME_SECONDS="${HARD_WALLTIME_SECONDS:-86400}"  # 24h
LOG_DIR="$ROOT/logs/blimp_scale_curriculum"
mkdir -p "$LOG_DIR"
PUSH_LOG="$LOG_DIR/autopush_$(date -u +%Y%m%dT%H%M%SZ).log"

PATHS=(
    "results/figures/main"
    "results/figures/appendix"
    "results/blimp"
    "results/evals/blimp/blimp_results.csv"
    "results/meco/meco_dutch_trajectory_summary.csv"
    "logs/blimp_scale_curriculum"
    "analyze/blimp_scale_curriculum.py"
    "analyze/blimp_eval_oneshot.py"
    "analyze/blimp_checkpoint_trajectory.py"
    "analyze/blimp_per_model_grids.py"
    "analyze/meco_dutch_grid.py"
    "scripts/eval_blimp_scale_curriculum.sh"
    "scripts/autopush_blimp_evals.sh"
    "scripts/run_emnlp_blimp_24h.sh"
)

echo "[$(date -u +%FT%TZ)] autopush start  branch=$BRANCH interval=${INTERVAL_SECONDS}s walltime=${HARD_WALLTIME_SECONDS}s" \
    | tee -a "$PUSH_LOG"

t0=$(date +%s)
iter=0

# Make sure the local branch exists and tracks origin/blimp-evals if upstream is set.
if ! git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
    git switch -c "$BRANCH" 2>&1 | tee -a "$PUSH_LOG"
else
    git switch "$BRANCH" 2>&1 | tee -a "$PUSH_LOG"
fi

push_once() {
    local label="$1"
    iter=$((iter + 1))
    echo "[$(date -u +%FT%TZ)] autopush iter=$iter ($label)" | tee -a "$PUSH_LOG"

    # Regenerate the publishable figure set BEFORE staging so the latest
    # CSV state is reflected in every push.
    PY=${PY:-/local/scratch/sas245/venvs/beetle-emnlp/bin/python}
    if [[ -x "$PY" ]]; then
        "$PY" -m analyze.blimp_scale_curriculum     >>"$PUSH_LOG" 2>&1 || true
        "$PY" -m analyze.blimp_checkpoint_trajectory >>"$PUSH_LOG" 2>&1 || true
        "$PY" -m analyze.blimp_per_model_grids       >>"$PUSH_LOG" 2>&1 || true
        "$PY" -m analyze.meco_dutch_grid             >>"$PUSH_LOG" 2>&1 || true
    fi

    # Stage only the listed paths, ignore missing ones.
    for p in "${PATHS[@]}"; do
        if [[ -e "$p" ]]; then
            git add -- "$p" 2>>"$PUSH_LOG" || true
        fi
    done

    # Anything actually staged?
    if git diff --cached --quiet; then
        echo "  no staged changes — nothing to push" | tee -a "$PUSH_LOG"
        return 0
    fi

    local ts
    ts="$(date -u +%FT%TZ)"
    if ! git -c commit.gpgsign=false commit \
            -m "blimp-evals autopush: ${label} @ ${ts}" 2>>"$PUSH_LOG"; then
        echo "  commit failed" | tee -a "$PUSH_LOG"
        return 0
    fi

    # Push (set upstream on first push). Don't fail the loop on transient errors.
    if git push -u origin "$BRANCH" 2>>"$PUSH_LOG"; then
        echo "  push ok" | tee -a "$PUSH_LOG"
    else
        echo "  push failed (will retry next iter)" | tee -a "$PUSH_LOG"
    fi
}

# Initial push so the branch exists on origin even before any eval finishes.
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
