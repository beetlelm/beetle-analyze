#!/usr/bin/env bash
# Helper: stage all pedagogical-sweep results + plots, commit, and push to the
# requested branch. Used twice in the EMNLP workflow:
#   1. After full §4.x sweep completes → push to emnlp-pareto-multiblimp
#   2. After we curate the headline figures → push to emnlp-figs
#
# Usage:
#   bash scripts/pedagogical_push_branch.sh emnlp-pareto-multiblimp "pedagogical sweep §4 results"
set -uo pipefail
cd "$(dirname "$0")/.."

BRANCH="${1:?branch name required}"
MSG="${2:-pedagogical sweep update}"

ts() { date -u +%FT%TZ; }
log() { echo "[$(ts)] $*"; }

log "switching to branch $BRANCH"
git fetch origin "$BRANCH" 2>&1 | tail -3 || true
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    git checkout "$BRANCH" 2>&1 | tail -3
else
    git checkout -t "origin/$BRANCH" 2>&1 | tail -3 || \
        git checkout -b "$BRANCH" 2>&1 | tail -3
fi
git pull --rebase origin "$BRANCH" 2>&1 | tail -3 || true

log "staging results + plots + new evals/scripts"
git add results/evals_paper/04a_cloze \
        results/evals_paper/04b_interlanguage \
        results/evals_paper/04c_evp_trajectory \
        results/evals_paper/04d_cefr_ppl 2>&1 | tail -3 || true
git add plots/panelA_cefr_ppl_gradient.pdf \
        plots/panelB_cloze.pdf \
        plots/panelC_interlanguage_forest.pdf \
        plots/panelD_evp_trajectory.pdf 2>&1 | tail -3 || true
git add evals/eval_cloze.py \
        evals/eval_cefr_perplexity.py \
        evals/eval_interlanguage.py \
        evals/eval_evp_trajectory.py \
        scripts/pedagogical_master_sweep.sh \
        scripts/pedagogical_push_branch.sh \
        plots/make_pedagogical_panels.py \
        data/evp/cefrj-vocabulary-profile-1.5.csv \
        data/evp/octanove-vocabulary-profile-c1c2-1.0.csv 2>&1 | tail -3 || true

if git diff --cached --quiet; then
    log "nothing to commit"
    exit 0
fi

git commit -m "$MSG"
git push origin "$BRANCH"
log "pushed to $BRANCH"
