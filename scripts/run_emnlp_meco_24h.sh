#!/usr/bin/env bash
# run_emnlp_meco_24h.sh — orchestrator for the 24h MECO-baselines tmux session.
#
# Layout (single tmux session "emnlp-meco" running inside this worktree):
#   window 0 "eval"     — runs scripts/eval_meco_baselines_24h.sh on GPU 1
#   window 1 "autopush" — hourly commit/push to branch "meco-evals"

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SESSION="${SESSION:-emnlp-meco}"
GPU="${GPU:-1}"
WALLTIME_SECONDS="${WALLTIME_SECONDS:-82800}"
PUSH_WALLTIME_SECONDS="${PUSH_WALLTIME_SECONDS:-86400}"

if ! command -v tmux >/dev/null 2>&1; then
    echo "ERROR: tmux not installed" >&2; exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' already exists. Kill it first: tmux kill-session -t $SESSION" >&2
    exit 1
fi

mkdir -p "$ROOT/logs/meco_baselines"

tmux new-session -d -s "$SESSION" -n eval -x 220 -y 50
tmux send-keys -t "$SESSION:eval" "cd $ROOT" C-m
tmux send-keys -t "$SESSION:eval" "export CUDA_VISIBLE_DEVICES=$GPU" C-m
tmux send-keys -t "$SESSION:eval" "export WALLTIME_SECONDS=$WALLTIME_SECONDS" C-m
tmux send-keys -t "$SESSION:eval" "bash scripts/eval_meco_baselines_24h.sh" C-m

tmux new-window -t "$SESSION" -n autopush
tmux send-keys -t "$SESSION:autopush" "cd $ROOT" C-m
tmux send-keys -t "$SESSION:autopush" "export BRANCH=meco-evals" C-m
tmux send-keys -t "$SESSION:autopush" "export INTERVAL_SECONDS=3600" C-m
tmux send-keys -t "$SESSION:autopush" "export HARD_WALLTIME_SECONDS=$PUSH_WALLTIME_SECONDS" C-m
tmux send-keys -t "$SESSION:autopush" "bash scripts/autopush_meco_evals.sh" C-m

echo
echo "tmux session '$SESSION' launched (worktree: $ROOT)."
echo "  attach:   tmux attach -t $SESSION"
echo "  windows:  eval (window 0)   autopush (window 1)"
echo "  GPU:      CUDA_VISIBLE_DEVICES=$GPU"
echo "  branch:   meco-evals (auto-push every 1h, hard walltime 24h)"
echo "  logs:     logs/meco_baselines/"
