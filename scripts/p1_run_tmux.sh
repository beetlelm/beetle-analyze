#!/usr/bin/env bash
# p1_run_tmux.sh — launch the P1 pipeline (MECO ΔlogLik fits → re-render
# fig_meco_dutch_grid_nld_eng_deltalogl) inside a detached tmux session
# alongside the autopush sidekick.
#
# Layout:
#   session: beetle_p1
#     window 0 "fit"     — runs scripts/p1_meco_deltalogl_nld_eng_traj.py
#     window 1 "autopush"— runs scripts/p1_autopush_meco_deltalogl.sh
#     window 2 "monitor" — tail -f of the fit log
#
# After the fit finishes, the fit window automatically renders the figure
# via analyze.meco_dutch_grid (which now reads the ΔlogLik cache).
#
# Usage:
#   bash scripts/p1_run_tmux.sh                # default: LANGS=du WORKERS=8
#   LANGS="ALL" WORKERS=12 bash scripts/p1_run_tmux.sh
#   tmux attach -t beetle_p1                   # attach
#   tmux kill-session -t beetle_p1             # kill
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SESSION="${SESSION:-beetle_p1}"
LANGS="${LANGS:-du}"
WORKERS="${WORKERS:-8}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-900}"
PYTHON="${PYTHON:-python3}"

LOG_DIR="$ROOT/logs/p1_meco_deltalogl"
RUN_DIR="$ROOT/.p1_run"
mkdir -p "$LOG_DIR" "$RUN_DIR"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
FIT_LOG="$LOG_DIR/fit_${TS}.log"

# Locate tmux: PATH first, then common Homebrew/Linuxbrew locations.
TMUX_BIN="${TMUX_BIN:-}"
if [ -z "$TMUX_BIN" ]; then
    if command -v tmux >/dev/null 2>&1; then
        TMUX_BIN="$(command -v tmux)"
    elif [ -x /opt/homebrew/bin/tmux ]; then
        TMUX_BIN=/opt/homebrew/bin/tmux
    elif [ -x /usr/local/bin/tmux ]; then
        TMUX_BIN=/usr/local/bin/tmux
    else
        echo "tmux not found — install via 'brew install tmux' or set TMUX_BIN" >&2
        exit 2
    fi
fi

if "$TMUX_BIN" has-session -t "$SESSION" 2>/dev/null; then
    echo "session '$SESSION' already exists — attach with: $TMUX_BIN attach -t $SESSION"
    echo "or kill it first: $TMUX_BIN kill-session -t $SESSION"
    exit 1
fi

echo "[p1] starting tmux session '$SESSION' via $TMUX_BIN"
echo "[p1] LANGS=$LANGS WORKERS=$WORKERS"
echo "[p1] fit log: $FIT_LOG"

# Materialise the per-window commands as small scripts to avoid heredoc/quoting
# pitfalls when handing complex multi-line commands to tmux new-session.
FIT_SCRIPT="$RUN_DIR/fit_${TS}.sh"
cat > "$FIT_SCRIPT" <<EOF
#!/usr/bin/env bash
set -uo pipefail
cd "$ROOT"
echo "[p1-fit] starting at \$(date -u +%FT%TZ)"
PYTHONUNBUFFERED=1 LANGS="$LANGS" WORKERS="$WORKERS" "$PYTHON" -u scripts/p1_meco_deltalogl_nld_eng_traj.py 2>&1 | tee "$FIT_LOG"
echo "[p1-fit] fits done — re-rendering figures via analyze.meco_dutch_grid"
PYTHONUNBUFFERED=1 "$PYTHON" -u -m analyze.meco_dutch_grid 2>&1 | tee -a "$FIT_LOG"
echo "[p1-fit] complete at \$(date -u +%FT%TZ); leave window open for inspection"
exec bash
EOF
chmod +x "$FIT_SCRIPT"

"$TMUX_BIN" new-session -d -s "$SESSION" -n "fit" "$FIT_SCRIPT"

# Capture the bash subshell pid for the fit window — autopush will watch
# this and exit cleanly once the fit window finishes its work. (The fit
# script itself flushes after every fit so partial progress is on disk.)
sleep 1
FIT_PID="$("$TMUX_BIN" list-panes -t "$SESSION":0 -F "#{pane_pid}" | head -n1)"
echo "[p1] fit window pane pid=$FIT_PID"

AUTOPUSH_SCRIPT="$RUN_DIR/autopush_${TS}.sh"
cat > "$AUTOPUSH_SCRIPT" <<EOF
#!/usr/bin/env bash
set -uo pipefail
cd "$ROOT"
echo "[p1-autopush] starting at \$(date -u +%FT%TZ)"
INTERVAL_SECONDS="$INTERVAL_SECONDS" FIT_PID="$FIT_PID" \\
    bash scripts/p1_autopush_meco_deltalogl.sh
echo "[p1-autopush] complete at \$(date -u +%FT%TZ); leave window open for inspection"
exec bash
EOF
chmod +x "$AUTOPUSH_SCRIPT"

"$TMUX_BIN" new-window -t "$SESSION" -n "autopush" "$AUTOPUSH_SCRIPT"

MONITOR_SCRIPT="$RUN_DIR/monitor_${TS}.sh"
cat > "$MONITOR_SCRIPT" <<EOF
#!/usr/bin/env bash
sleep 2
tail -F "$FIT_LOG"
EOF
chmod +x "$MONITOR_SCRIPT"

"$TMUX_BIN" new-window -t "$SESSION" -n "monitor" "$MONITOR_SCRIPT"
"$TMUX_BIN" select-window -t "$SESSION":0

cat <<EOF

[p1] tmux session '$SESSION' launched.
     attach: $TMUX_BIN attach -t $SESSION
     detach: Ctrl-b d
     kill:   $TMUX_BIN kill-session -t $SESSION

Windows: fit (0) · autopush (1) · monitor (2)
Logs:    $FIT_LOG
Cache:   results/meco/canonical_deltalogl_nld_eng_traj.csv
Figure:  results/figures/appendix/fig_meco_dutch_grid_nld_eng_deltalogl.{pdf,png}
EOF
