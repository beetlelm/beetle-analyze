#!/usr/bin/env bash
# scripts/launch_evals_safe.sh — Launch scripts/eval_frontier_baselines.sh
# in a way that survives SSH disconnects, terminal closes, and Ctrl+C in
# the parent shell.
#
# Prefers tmux (so you can reattach and watch progress); falls back to nohup.
#
# Usage:
#   bash scripts/launch_evals_safe.sh
#   N_GPUS=4 bash scripts/launch_evals_safe.sh
#   HF_CACHE_ROOT=/scratch/$USER/hf bash scripts/launch_evals_safe.sh
#   SKIP_APERTUS_70B=0 bash scripts/launch_evals_safe.sh
#   SKIP_PREFETCH=1 bash scripts/launch_evals_safe.sh
#
# All env vars used by eval_frontier_baselines.sh are forwarded.
set -euo pipefail
cd "$(dirname "$0")/.."

SESSION="beetle-evals-$(date +%Y%m%d-%H%M%S)"
WRAPPER_LOG="results/logs/wrapper_${SESSION}.log"
mkdir -p "$(dirname "$WRAPPER_LOG")"

# Forward the relevant env vars explicitly so they survive the tmux/nohup hop.
ENV_PREFIX=""
for v in N_GPUS OUT HF_CACHE_ROOT BATCH_SIZE_BLIMP BATCH_SIZE_BLISS \
         SKIP_APERTUS_70B SKIP_PREFETCH RANK_LAUNCH_DELAY \
         HF_TOKEN HUGGING_FACE_HUB_TOKEN; do
    if [[ -n "${!v:-}" ]]; then
        ENV_PREFIX+="${v}=$(printf %q "${!v}") "
    fi
done

CMD="${ENV_PREFIX}bash scripts/eval_frontier_baselines.sh"

if command -v tmux >/dev/null 2>&1; then
    tmux new-session -d -s "$SESSION" \
        "$CMD 2>&1 | tee \"$WRAPPER_LOG\""
    echo "Launched in tmux session: $SESSION"
    echo "  Reattach:  tmux attach -t $SESSION"
    echo "  Detach:    Ctrl-b d"
    echo "  Tail log:  tail -f $WRAPPER_LOG"
    echo "  Per-rank:  tail -f results/logs/blimp_frontier_rank0.log"
else
    nohup bash -c "$CMD" > "$WRAPPER_LOG" 2>&1 &
    PID=$!
    disown
    echo "Launched under nohup (PID $PID)."
    echo "  Tail log:  tail -f $WRAPPER_LOG"
    echo "  Per-rank:  tail -f results/logs/blimp_frontier_rank0.log"
    echo "  Check:     ps -p $PID"
fi
