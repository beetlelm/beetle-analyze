#!/usr/bin/env bash
# run_emnlp_blimp_24h.sh — orchestrator for the 24h BLiMP-evals tmux session.
#
# Layout (single tmux session "emnlp-blimp"):
#   window 0 "eval"     — runs scripts/eval_blimp_scale_curriculum.sh on GPU 0
#   window 1 "autopush" — hourly commit/push to branch "blimp-evals"
#
# Both windows tail their own logs in logs/blimp_scale_curriculum/.
# After the eval window's process exits the autopush window keeps running
# until its hard walltime so the very last batch always lands on origin.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SESSION="${SESSION:-emnlp-blimp}"
GPU="${GPU:-0}"
WALLTIME_SECONDS="${WALLTIME_SECONDS:-82800}"      # 23h for eval
PUSH_WALLTIME_SECONDS="${PUSH_WALLTIME_SECONDS:-86400}"  # 24h for autopush

if ! command -v tmux >/dev/null 2>&1; then
    echo "ERROR: tmux not installed" >&2; exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' already exists. Kill it first: tmux kill-session -t $SESSION" >&2
    exit 1
fi

mkdir -p "$ROOT/logs/blimp_scale_curriculum"

# Window 0: eval. Single-GPU pin via CUDA_VISIBLE_DEVICES.
tmux new-session -d -s "$SESSION" -n eval -x 220 -y 50
tmux send-keys -t "$SESSION:eval" "cd $ROOT" C-m
tmux send-keys -t "$SESSION:eval" "export CUDA_VISIBLE_DEVICES=$GPU" C-m
tmux send-keys -t "$SESSION:eval" "export WALLTIME_SECONDS=$WALLTIME_SECONDS" C-m
tmux send-keys -t "$SESSION:eval" "bash scripts/eval_blimp_scale_curriculum.sh" C-m

# Window 1: autopush. Hourly to branch blimp-evals.
tmux new-window -t "$SESSION" -n autopush
tmux send-keys -t "$SESSION:autopush" "cd $ROOT" C-m
tmux send-keys -t "$SESSION:autopush" "export BRANCH=blimp-evals" C-m
tmux send-keys -t "$SESSION:autopush" "export INTERVAL_SECONDS=3600" C-m
tmux send-keys -t "$SESSION:autopush" "export HARD_WALLTIME_SECONDS=$PUSH_WALLTIME_SECONDS" C-m
tmux send-keys -t "$SESSION:autopush" "bash scripts/autopush_blimp_evals.sh" C-m

echo
echo "tmux session '$SESSION' launched."
echo "  attach:   tmux attach -t $SESSION"
echo "  windows:  eval (window 0)   autopush (window 1)"
echo "  GPU:      CUDA_VISIBLE_DEVICES=$GPU"
echo "  branch:   blimp-evals (auto-push every 1h, hard walltime 24h)"
echo "  logs:     logs/blimp_scale_curriculum/"
