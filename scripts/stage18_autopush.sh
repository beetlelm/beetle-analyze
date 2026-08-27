#!/usr/bin/env bash
# Periodic git commit+push of stage 18 results to the dedicated branch.
# Runs as a background sidecar to the stage 18 launcher.
# Stops cleanly when /root/beetle_pipeline/state/stage18.done exists or this
# process gets SIGTERM.

set -u -o pipefail

REPO="/root/beetle-analyze"
BRANCH="stage18-results-2026-05-04"
INTERVAL="${STAGE18_PUSH_INTERVAL:-600}"   # 10 min
DONE_FLAG="/root/beetle_pipeline/state/stage18.done"
LOG="/root/beetle_pipeline/logs/stage18_autopush.log"

mkdir -p "$(dirname "$LOG")"
cd "$REPO"

log() { echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }

trap 'log "received signal; final push"; do_push final; exit 0' TERM INT

do_push() {
    local tag="${1:-tick}"
    git add results/ plots/ 2>>"$LOG" || true
    if git diff --cached --quiet; then
        log "$tag: nothing to commit"
        return 0
    fi
    git commit -m "stage18 $tag: $(date -u +%FT%TZ)" >>"$LOG" 2>&1 || {
        log "$tag: commit failed"; return 1; }
    git push origin "$BRANCH" >>"$LOG" 2>&1 || {
        log "$tag: push failed (will retry next tick)"; return 1; }
    log "$tag: pushed"
}

log "autopush start (branch=$BRANCH interval=${INTERVAL}s)"
git checkout "$BRANCH" >>"$LOG" 2>&1

while true; do
    do_push tick
    if [[ -f "$DONE_FLAG" ]]; then
        log "DONE flag seen; final push and exit"
        do_push final
        exit 0
    fi
    sleep "$INTERVAL"
done
