#!/usr/bin/env bash
# =============================================================================
# scripts/run_units_trajectory.sh
#
# Developmental interpretability of syntactic units (analyze.units).
#
# For each Beetle repo, localize the units carrying subject-verb / anaphor /
# determiner-noun agreement at every training checkpoint and trace their
# emergence, cross-phenomenon overlap ("same units"), mask persistence, and
# causal ablation effect over training.
#
# Wraps the localization + ablation methodology of Kryvosheieva et al. (2026,
# arXiv:2512.03676) — github.com/elliepreed/syntax-units and the
# syntax-interpretability replication — as a checkpoint-aware library.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#   bash scripts/run_units_trajectory.sh                       # default repos
#   REPOS="org/model-a org/model-b" bash scripts/run_units_trajectory.sh
#   DRY_RUN=1  bash scripts/run_units_trajectory.sh            # preview only
#   TOP_FRAC=0.02 METHOD=welch bash scripts/run_units_trajectory.sh
#   NO_ABLATION=1 bash scripts/run_units_trajectory.sh         # skip causal pass
# =============================================================================
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-results/units}"
TOP_FRAC="${TOP_FRAC:-0.01}"
METHOD="${METHOD:-ttest}"
SITE="${SITE:-resid}"
MAX_PAIRS="${MAX_PAIRS:-150}"
CKPT_SCOPE="${CKPT_SCOPE:-all}"
CATEGORIES="${CATEGORIES:-subject_verb_agreement anaphor_agreement determiner_noun_agreement}"

# Default: the five Beetle-FineWeb-2B NLD-ENG bilingual curricula (b1..b5),
# the same set pre-registered in analyze/xling/registry.py.
REPOS="${REPOS:-\
Beetle-FineWeb-2B/beetle-bilingual-balanced-b1-fineweb-2b-nld-eng \
Beetle-FineWeb-2B/beetle-bilingual-l2-50-simultaneous-b2-fineweb-2b-nld-eng \
Beetle-FineWeb-2B/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-2b-nld-eng \
Beetle-FineWeb-2B/beetle-bilingual-l2-50-classroom-20-b4-fineweb-2b-nld-eng \
Beetle-FineWeb-2B/beetle-bilingual-l2-80-late-b5-fineweb-2b-nld-eng}"

ABLATION_FLAG="--with-ablation"
[[ "${NO_ABLATION:-0}" == "1" ]] && ABLATION_FLAG=""

mkdir -p "$OUTPUT_DIR"

for repo in $REPOS; do
    tag="$(echo "$repo" | tr '/' '__')"
    out="$OUTPUT_DIR/trajectory_${tag}.csv"
    cmd=(python -m analyze.units.trajectory
         --repo "$repo"
         --ckpt-scope "$CKPT_SCOPE" --log-spaced
         --categories $CATEGORIES
         --max-pairs "$MAX_PAIRS"
         --top-frac "$TOP_FRAC" --method "$METHOD" --site "$SITE"
         $ABLATION_FLAG
         --out "$out")
    echo "+ ${cmd[*]}"
    [[ "${DRY_RUN:-0}" == "1" ]] || "${cmd[@]}"
done

echo "[run_units_trajectory] done → $OUTPUT_DIR"
