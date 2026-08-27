#!/usr/bin/env bash
# p1_autopush_meco_deltalogl.sh — periodic commit + push of P1 ΔlogLik
# trajectory cache while the fit job is running.
#
# Targeted paths only (mirrors scripts/autopush_blimp_evals.sh pattern):
#   - results/meco/canonical_deltalogl_nld_eng_traj.csv
#   - results/figures/appendix/fig_meco_dutch_grid_nld_eng_deltalogl.{pdf,png}
#   - logs/p1_meco_deltalogl/*.log
#   - analyze/meco_dutch_grid.py
#   - scripts/p1_meco_deltalogl_nld_eng_traj.py
#   - scripts/p1_autopush_meco_deltalogl.sh
#   - scripts/p1_run_tmux.sh
#
# Loops until INTERVAL_SECONDS × MAX_ITERS elapses or the FIT_PID stops.
# Always pushes a final commit on exit.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BRANCH="${BRANCH:-meco-deltalogl-evals}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-900}"            # 15 min
HARD_WALLTIME_SECONDS="${HARD_WALLTIME_SECONDS:-86400}"  # 24h cap
FIT_PID="${FIT_PID:-}"                                  # optional — exit when this pid is gone
LOG_DIR="$ROOT/logs/p1_meco_deltalogl"
mkdir -p "$LOG_DIR"
PUSH_LOG="$LOG_DIR/autopush_$(date -u +%Y%m%dT%H%M%SZ).log"

PATHS=(
    "results/meco/canonical_deltalogl_nld_eng_traj.csv"
    "results/figures/appendix/fig_meco_dutch_grid_nld_eng_deltalogl.pdf"
    "results/figures/appendix/fig_meco_dutch_grid_nld_eng_deltalogl.png"
    "logs/p1_meco_deltalogl"
    "analyze/meco_dutch_grid.py"
    "scripts/p1_meco_deltalogl_nld_eng_traj.py"
    "scripts/p1_autopush_meco_deltalogl.sh"
    "scripts/p1_run_tmux.sh"
)

echo "[$(date -u +%FT%TZ)] autopush start branch=$BRANCH interval=${INTERVAL_SECONDS}s walltime=${HARD_WALLTIME_SECONDS}s fit_pid=${FIT_PID}" \
    | tee -a "$PUSH_LOG"

# Ensure the local branch exists. We do NOT switch the working tree to a
# different branch — the user may already be on a feature branch. We commit
# from whatever branch we're on. If you want a dedicated branch, switch
# manually before launching.
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "[$(date -u +%FT%TZ)] committing from current branch: $CURRENT_BRANCH" \
    | tee -a "$PUSH_LOG"

t0=$(date +%s)
iter=0

commit_and_push() {
    local label="$1"
    local any_changes=0
    for p in "${PATHS[@]}"; do
        if [ -e "$p" ]; then
            if ! git diff --quiet -- "$p" 2>/dev/null || \
               git ls-files --others --exclude-standard -- "$p" | grep -q .; then
                git add -- "$p"
                any_changes=1
            fi
        fi
    done

    if [ "$any_changes" -eq 0 ] || git diff --cached --quiet; then
        echo "[$(date -u +%FT%TZ)] [$label] no staged changes — skipping commit" \
            | tee -a "$PUSH_LOG"
        return 0
    fi

    local msg="p1: meco ΔlogLik nld-eng trajectory progress ($label)"
    git commit -m "$msg" 2>&1 | tee -a "$PUSH_LOG" || {
        echo "[$(date -u +%FT%TZ)] [$label] commit failed; continuing" \
            | tee -a "$PUSH_LOG"
        return 0
    }
    if git push origin "$CURRENT_BRANCH" 2>&1 | tee -a "$PUSH_LOG"; then
        echo "[$(date -u +%FT%TZ)] [$label] pushed to origin/$CURRENT_BRANCH" \
            | tee -a "$PUSH_LOG"
    else
        echo "[$(date -u +%FT%TZ)] [$label] push failed; will retry on next iter" \
            | tee -a "$PUSH_LOG"
    fi
}

trap 'commit_and_push "final-on-exit"' EXIT

while :; do
    iter=$((iter + 1))
    sleep "$INTERVAL_SECONDS"
    commit_and_push "iter-$iter"

    now=$(date +%s)
    elapsed=$((now - t0))
    if [ -n "$FIT_PID" ] && ! kill -0 "$FIT_PID" 2>/dev/null; then
        echo "[$(date -u +%FT%TZ)] fit pid=$FIT_PID has exited; final flush + stop" \
            | tee -a "$PUSH_LOG"
        break
    fi
    if [ "$elapsed" -ge "$HARD_WALLTIME_SECONDS" ]; then
        echo "[$(date -u +%FT%TZ)] hard walltime reached; stop" \
            | tee -a "$PUSH_LOG"
        break
    fi
done

echo "[$(date -u +%FT%TZ)] autopush exit (iter=$iter, elapsed=${elapsed}s)" \
    | tee -a "$PUSH_LOG"
