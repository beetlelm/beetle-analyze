#!/usr/bin/env bash
# scripts/cpu_review_finalize.sh
# Stages and pushes ONLY the cpu_review_runner outputs to origin/emnlp-interp.
# Concurrent GPU runs may have pushed since we last fetched, so we cherry-pick
# our staged commit onto the remote tip rather than rebasing the whole tree
# (the working tree contains pre-existing typechange entries we mustn't touch).
#
# Stages:
#   results/meco/significance/         (bootstrap CIs, paired perm tests, etc.)
#   results/cache/bpt_bpb_ratios.csv   (W9 cross-tokeniser cache)
#   logs/cpu_review/                   (runner stdout/stderr)
#
# Refuses to stage anything else.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
cd "${REPO_ROOT}"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "[finalize] CWD=${REPO_ROOT}"
echo "[finalize] HEAD=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"

PATHS=(
  "results/meco/significance/"
  "results/cache/bpt_bpb_ratios.csv"
  "logs/cpu_review/"
)

# Stage only the allow-listed paths.
for p in "${PATHS[@]}"; do
  if [[ -e "${p}" ]]; then
    git add -- "${p}"
  else
    echo "[finalize] WARN: ${p} not found"
  fi
done

# Show what is about to be committed.
echo "[finalize] staged diff summary:"
git diff --cached --stat | tail -30 || true

if git diff --cached --quiet; then
  echo "[finalize] nothing staged — exiting"
  exit 0
fi

MSG="emnlp-interp: cpu/review-runner W6+W9+DC results @ $(ts)"
git commit -m "$(cat <<EOF
${MSG}

Outputs from scripts/cpu_review_runner.sh:
  - results/meco/significance/  (paired bootstrap CIs + permutation tests + Holm-Bonferroni / BH-FDR)
  - results/cache/bpt_bpb_ratios.csv  (cross-tokeniser BPT->BPB ratios for BLOOM ladder + Beetle B1/B3/B5)
  - logs/cpu_review/            (run logs)

Reviewer-point coverage:
  W6  significance bundle now in-paper-ready
  W9  empirical tokeniser-fair conversion factors for the headline gap
  DC  Table 12 duplicate-row probe + App. K BLiSS join scope diagnostics

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"

NEW_SHA=$(git rev-parse HEAD)
echo "[finalize] commit=${NEW_SHA}"

# Try a fast-forward push first; if rejected, rebase our single commit onto
# the remote tip via cherry-pick (working-tree typechanges block ordinary
# rebase, so we detach + cherry-pick instead).
push_attempt() {
  git push origin emnlp-interp 2>&1 | tail -8
}

if push_attempt | grep -q -E "rejected|non-fast-forward"; then
  echo "[finalize] non-ff — rebasing single commit onto remote tip"
  git fetch origin emnlp-interp
  git -c advice.detachedHead=false checkout origin/emnlp-interp
  git checkout -B emnlp-interp
  git cherry-pick "${NEW_SHA}"
  push_attempt
fi

echo "[finalize] DONE @ $(ts)"
git log --oneline -3
