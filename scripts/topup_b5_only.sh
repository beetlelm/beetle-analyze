#!/usr/bin/env bash
# scripts/topup_b5_only.sh — append b5-only rows to the §A / §D / §H CSVs
# after the main run_cpu_overnight.sh pass finishes.
#
# Why this exists
# ---------------
# The first beetle-cpu pass shipped with a wrong b5 repo name baked into
# the runner array (l2-80-late-b5-... rather than balanced-b5-...). The
# bug was only fixed mid-run, so b1 and b3 landed for every stage but b5
# silently skipped. This top-up script re-runs the three stages that
# tried (and failed) to load b5 — they are idempotent, so b1/b3 rows in
# the CSVs are de-duplicated by ``model_id × layer × stratum`` and only
# the new b5 rows are appended.

set -uo pipefail
cd "$(dirname "$0")/.."

export BEETLE_FORCE_CPU=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(sysctl -n hw.physicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)}"
export BEETLE_REPO_ROOT="$(pwd)"
export BEETLE_PUSH_BRANCH="${BEETLE_PUSH_BRANCH:-emnlp-interp}"

LOG_DIR="logs/cpu_overnight"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/topup_b5_$(date +%Y%m%dT%H%M%S).log"
echo "[topup-b5] log → $RUN_LOG"

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
    push "topup/$label"
  else
    echo "[fail] $label — pushing partial state" | tee -a "$RUN_LOG"
    push "topup/$label-failed"
  fi
}

B5="Beetle-FineWeb/beetle-bilingual-balanced-b5-fineweb-nld-eng"

stage cka-b5 \
  python -m analyze.xling.cka_curriculum --n-pairs 64 --n-boot 200 \
    --repos "$B5" --curricula b5

stage logit-lens-b5 \
  python -m analyze.lens.logit_lens_lang \
    --repos "$B5" --curricula b5 --n-sentences 16

stage meco-layerwise-b5 \
  python -m analyze.meco_layerwise --pilot \
    --repos "$B5" --curricula b5

# Re-render figures that depend on these CSVs.
for fig in figI1_geometry figI4_meco_layerwise figI5_logit_lens; do
  stage "fig-$fig" python -m plots.make_$fig
done

echo "[topup-b5] done @ $(date -u +%H:%M:%SZ)" | tee -a "$RUN_LOG"
push "topup/final"
