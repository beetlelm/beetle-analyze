#!/usr/bin/env bash
# scripts/run_finalize_cpu.sh — finish the CPU run that died after stage 7.
#
# The original run_cpu_overnight.sh pass stopped writing to its log after
# stage 7 (likely tmux session went away during a Mac sleep / network
# blip). Three things still need to happen:
#
#   A. Top up b5 in §A / §D / §H — first pass had a wrong b5 repo name
#      and silently skipped it; CSVs only contain b1 + b3.
#   B. Run §K learner-corpus on b1 + b3 + b5 — never executed.
#   C. Render figI1 / figI2 / figI3 / figI4 / figI5 / figI10 / figM1.
#      figI7 and figI9 already rendered in the GPU pass.
#
# Idempotent: every stage de-duplicates by ``model_id × layer × stratum``
# (or appends-only for learner_corpus), so re-running this script does
# not corrupt data.

set -uo pipefail
cd "$(dirname "$0")/.."

export BEETLE_FORCE_CPU=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(sysctl -n hw.physicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)}"
export BEETLE_REPO_ROOT="$(pwd)"
export BEETLE_PUSH_BRANCH="${BEETLE_PUSH_BRANCH:-emnlp-interp}"

LOG_DIR="logs/cpu_overnight"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/finalize_$(date +%Y%m%dT%H%M%S).log"
echo "[finalize] log → $RUN_LOG"

push() {
  python - <<PY 2>&1 | tee -a "$RUN_LOG"
from analyze.results_push import push
push("$1")
PY
}

stage() {
  local label="$1"; shift
  echo "" | tee -a "$RUN_LOG"
  echo "==== [$(date -u +%H:%M:%S)] $label ====" | tee -a "$RUN_LOG"
  if "$@" 2>&1 | tee -a "$RUN_LOG"; then
    push "finalize/$label"
  else
    echo "[fail] $label — pushing partial state" | tee -a "$RUN_LOG"
    push "finalize/$label-failed"
  fi
}

REPOS=(
  "Beetle-FineWeb/beetle-bilingual-balanced-b1-fineweb-nld-eng"
  "Beetle-FineWeb/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-nld-eng"
  "Beetle-FineWeb/beetle-bilingual-balanced-b5-fineweb-nld-eng"
)
CURRICULA=(b1 b3 b5)
B5="${REPOS[2]}"

# A. Top up b5 in §A / §D / §H (b1/b3 already on disk; the dedupe step
#    in cka_curriculum/etc keeps existing rows untouched).
stage cka-b5 \
  python -m analyze.xling.cka_curriculum --n-pairs 64 --n-boot 200 \
    --repos "$B5" --curricula b5

stage logit-lens-b5 \
  python -m analyze.lens.logit_lens_lang \
    --repos "$B5" --curricula b5 --n-sentences 16

stage meco-layerwise-b5 \
  python -m analyze.meco_layerwise --pilot \
    --repos "$B5" --curricula b5

# B. §K learner-corpus on all three curricula.
for i in "${!REPOS[@]}"; do
  stage "learner-corpus-${CURRICULA[$i]}" \
    python -m analyze.learner_corpus \
      --repo "${REPOS[$i]}" --curriculum "${CURRICULA[$i]}"
done

# C. Render every figI* whose CSV is now on disk plus figM1.
for fig in figI1_geometry figI2_patching figI3_aoa figI4_meco_layerwise \
           figI5_logit_lens figI10_learner_corpus figM1_schematic; do
  stage "fig-$fig" python -m plots.make_$fig
done

echo "[finalize] done @ $(date -u +%H:%M:%SZ)" | tee -a "$RUN_LOG"
push "finalize/final"
