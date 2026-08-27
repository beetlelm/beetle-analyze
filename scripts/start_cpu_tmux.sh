#!/usr/bin/env bash
# scripts/start_cpu_tmux.sh — Launch the CPU overnight pipeline in a
# detached tmux session named "beetle-cpu". Idempotent: re-running while
# the session is alive just attaches to it.
#
# Usage:
#   scripts/start_cpu_tmux.sh             # start (or reattach) the session
#   tmux attach -t beetle-cpu             # attach explicitly
#   tmux kill-session -t beetle-cpu       # stop everything
#
# The runner script writes a log to logs/cpu_overnight/run_<timestamp>.log
# regardless of the tmux state, so you can `tail -f logs/cpu_overnight/...`
# from any other terminal too.

set -euo pipefail

cd "$(dirname "$0")/.."

SESSION="${BEETLE_TMUX_SESSION:-beetle-cpu}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not installed. Install with 'brew install tmux' on macOS."
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[tmux] '$SESSION' already running — attaching."
  exec tmux attach -t "$SESSION"
fi

mkdir -p logs/cpu_overnight
echo "[tmux] launching '$SESSION' → bash scripts/run_cpu_overnight.sh"

# -d = detached; the trailing 'exec bash' keeps the pane open after the
# script exits so you can inspect output / regenerate plots interactively.
tmux new-session -d -s "$SESSION" \
  "bash scripts/run_cpu_overnight.sh \"\$@\" ; echo ; echo '[tmux] runner exited — pane left open for inspection' ; exec bash"

echo "[tmux] session detached. Attach with:  tmux attach -t $SESSION"
echo "[tmux] live log:  ls -t logs/cpu_overnight/ | head -1 | xargs -I {} tail -f logs/cpu_overnight/{}"
