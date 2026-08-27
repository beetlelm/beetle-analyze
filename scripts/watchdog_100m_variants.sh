#!/usr/bin/env bash
# watchdog_100m_variants.sh
#
# CPU-only sentinel watcher for the nld-eng B3 100M sweep.
#
# Every POLL_SECS, parse the per-variant training logs and regenerate the
# learning-dynamics figure from text only — no torch import, no CUDA — so
# updates can flow to the 100m-variants branch while training holds all 8
# A100s. Commits + pushes when:
#   (a) a new step-N entry appears in any variant's log, OR
#   (b) a new <variant>.done sentinel appears.
#
# GPU-safety contract:
#   * never imports torch
#   * never calls nvidia-smi
#   * never opens any CUDA device
# This is enforced by setting CUDA_VISIBLE_DEVICES="" before any python call.
#
# Env:
#   POLL_SECS    default 300 (5 min)
#   BRANCH       default 100m-variants
#   REMOTE       default origin
set -uo pipefail

REPO="/root/beetle-analyze"
LOG_DIR="/root/beetlelm/logs/nld_eng_b3_100m_safe"
RESULT_DIR="${REPO}/results/100m-variants"
FIG_DIR="${RESULT_DIR}/figs"

POLL_SECS="${POLL_SECS:-300}"
BRANCH="${BRANCH:-100m-variants}"
REMOTE="${REMOTE:-origin}"

export CUDA_VISIBLE_DEVICES=""        # belt-and-braces GPU isolation
export PATH="/opt/conda/bin:${PATH}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

cd "${REPO}"

# Ensure the branch exists locally; never destructively reset.
if ! git rev-parse --verify "${BRANCH}" >/dev/null 2>&1; then
    git switch -c "${BRANCH}" 2>/dev/null || git checkout -b "${BRANCH}"
fi
git switch "${BRANCH}" 2>/dev/null || git checkout "${BRANCH}"

mkdir -p "${RESULT_DIR}/figs" "${LOG_DIR}"

# Hash-of-inputs signature; change ⇒ regen + commit + push.
last_signature=""
sig() {
    local items=()
    shopt -s nullglob
    for f in "${LOG_DIR}"/*.log "${LOG_DIR}"/*.done; do
        [[ -f "$f" ]] && items+=("$(stat -c '%n %s %Y' "$f" 2>/dev/null)")
    done
    shopt -u nullglob
    printf '%s\n' "${items[@]}" | sha256sum | awk '{print $1}'
}

regen_and_push() {
    local stamp="$1"
    echo "[$(date -u +%FT%TZ)] regen → commit → push (signature changed: ${stamp:0:12})"

    if ! /opt/conda/bin/python3 -m analyze.learning_dynamics_100m >/tmp/lhd_100m.log 2>&1; then
        echo "  WARN: figure regen failed; see /tmp/lhd_100m.log"
        cat /tmp/lhd_100m.log
        return 0
    fi

    git add -A "${RESULT_DIR}/" "${REPO}/analyze/learning_dynamics_100m.py" 2>/dev/null
    if git diff --cached --quiet; then
        echo "  no diff to commit"
        return 0
    fi

    # Snapshot of completion progress for the commit message.
    local done_count
    done_count=$(ls "${LOG_DIR}"/*.done 2>/dev/null | wc -l | tr -d ' ')
    local sentinels
    sentinels=$(ls "${LOG_DIR}"/*.done 2>/dev/null | xargs -r -n1 basename 2>/dev/null | sed 's/\.done$//' | tr '\n' ',' | sed 's/,$//')

    git commit -m "auto: 100m-variants figures (${done_count} variants done: ${sentinels:-none})" \
        >/dev/null
    if git push "${REMOTE}" "${BRANCH}" 2>&1 | tail -3; then
        echo "  pushed → ${REMOTE}/${BRANCH}"
    else
        echo "  WARN: push failed; will retry on next tick"
    fi
}

echo "[$(date -u +%FT%TZ)] watchdog start: poll=${POLL_SECS}s branch=${BRANCH}"
echo "[$(date -u +%FT%TZ)] watching ${LOG_DIR}/*.log + *.done"

while true; do
    cur_signature="$(sig)"
    if [[ "${cur_signature}" != "${last_signature}" ]]; then
        regen_and_push "${cur_signature}"
        last_signature="${cur_signature}"
    fi
    sleep "${POLL_SECS}"
done
