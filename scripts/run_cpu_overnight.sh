#!/usr/bin/env bash
# scripts/run_cpu_overnight.sh — Long-running CPU pipeline for Mac tmux.
#
# Sequence of stages a Mac M-series CPU can complete in ~3–4 hours,
# producing CSVs + figures and **pushing every stage's outputs to
# origin/emnlp-interp** so a tmux disconnect, Mac sleep, or sudden battery
# death never costs more than one stage.
#
# Stages (rough wall-time on M2/M3):
#
#   1.  §A cognate-strata partition         ~10 sec
#   2.  §A non-parallel stimuli pool         ~5 sec
#   3.  Byte-normalize cache (BPT→BPB)       ~5 min  (+download first run)
#   4.  §A geometry CKA  b1/b3/b5            30–60 min
#   5.  §I external baseline (XLM-R, mBERT)  20–40 min
#   6.  §D logit-lens language commit        30–60 min
#   7.  §H MECO layerwise pilot              30–45 min
#   8.  §K learner-corpus surprisal          10–15 min
#   9.  Render figI1 / figI4 / figI5 / figI10 / figM1   ~5 min
#
# Each stage commits + pushes via analyze.results_push.push(...). Set
# BEETLE_PUSH_DISABLED=1 to keep the run local.
#
# Usage:
#   scripts/run_cpu_overnight.sh                # all stages, sequential
#   scripts/run_cpu_overnight.sh --start-from 5 # resume after a crash
#   BEETLE_PUSH_DISABLED=1 scripts/run_cpu_overnight.sh   # offline run
#   scripts/start_cpu_tmux.sh                   # detach into a tmux pane

set -uo pipefail

cd "$(dirname "$0")/.."

export BEETLE_FORCE_CPU=1
export TOKENIZERS_PARALLELISM=false
# Mac: sysctl hw.physicalcpu; Linux: nproc.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(sysctl -n hw.physicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export BEETLE_REPO_ROOT="$(pwd)"
export BEETLE_PUSH_BRANCH="${BEETLE_PUSH_BRANCH:-emnlp-interp}"

START_FROM=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --start-from) START_FROM="$2"; shift 2 ;;
    -h|--help)
      grep -E '^# ' "$0" | head -50
      exit 0 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

LOG_DIR="logs/cpu_overnight"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run_$(date +%Y%m%dT%H%M%S).log"
echo "[run] log → $RUN_LOG"
echo "[run] OMP_NUM_THREADS=$OMP_NUM_THREADS branch=$BEETLE_PUSH_BRANCH"

push() {
  python - <<PY 2>&1 | tee -a "$RUN_LOG"
import sys
try:
    from analyze.results_push import push
    push("$1")
except Exception as e:
    print(f"[push] error: {e}", file=sys.stderr)
PY
}

stage() {
  local idx="$1"; local label="$2"; shift 2
  if (( idx < START_FROM )); then
    echo "[skip] stage $idx ($label) — before --start-from=$START_FROM" | tee -a "$RUN_LOG"
    return 0
  fi
  echo "" | tee -a "$RUN_LOG"
  echo "==== [$(date -u +%H:%M:%S)] stage $idx — $label ====" | tee -a "$RUN_LOG"
  local t0
  t0=$(date +%s)
  if "$@" 2>&1 | tee -a "$RUN_LOG"; then
    local elapsed=$(( $(date +%s) - t0 ))
    echo "[done] stage $idx in ${elapsed}s" | tee -a "$RUN_LOG"
    push "cpu/${idx}-${label}"
  else
    echo "[fail] stage $idx — pushing partial state and continuing" | tee -a "$RUN_LOG"
    push "cpu/${idx}-${label}-failed"
  fi
}

REPOS=(
  "Beetle-FineWeb/beetle-bilingual-balanced-b1-fineweb-nld-eng"
  "Beetle-FineWeb/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-nld-eng"
  "Beetle-FineWeb/beetle-bilingual-balanced-b5-fineweb-nld-eng"
)
CURRICULA=(b1 b3 b5)
EXT_REPOS=(
  "FacebookAI/xlm-roberta-base"
  "google-bert/bert-base-multilingual-cased"
)

# ── 1. cognate / non-parallel partition (no model needed) ────────────────
stage 1 cognate-filter \
  python -m analyze.xling.cognate_filter --l1 nld --l2 eng

stage 2 non-parallel-stimuli \
  python -m analyze.xling.non_parallel_stimuli --l1 nld --l2 eng --n-pairs 1000

# ── 3. tokenizer-only byte-normalization cache ──────────────────────────
stage 3 byte-normalize \
  python -m analyze.byte_normalize \
    --repos "${REPOS[@]}" "${EXT_REPOS[@]}" \
    --langs nld eng

# ── 4. §A geometry on b1/b3/b5 ──────────────────────────────────────────
# Uses ``python -m analyze.xling.cka_curriculum`` rather than the
# ``beetle-cka`` console script so the runner does not depend on the
# package having been ``pip install -e .``-ed in the active venv.
stage 4 cka-curriculum \
  python -m analyze.xling.cka_curriculum --n-pairs 64 --n-boot 200 \
    --repos "${REPOS[@]}" --curricula "${CURRICULA[@]}"

# ── 5. §I external baseline ─────────────────────────────────────────────
stage 5 external-baseline \
  python -m analyze.xling.external_baseline --n-pairs 64 --n-boot 200

# ── 6. §D logit lens ────────────────────────────────────────────────────
stage 6 logit-lens \
  python -m analyze.lens.logit_lens_lang \
    --repos "${REPOS[@]}" --curricula "${CURRICULA[@]}" \
    --n-sentences 16

# ── 7. §H MECO layerwise pilot (logit-lens variant; FLORES smoke fallback) ─
stage 7 meco-layerwise-pilot \
  python -m analyze.meco_layerwise --pilot \
    --repos "${REPOS[@]}" --curricula "${CURRICULA[@]}"

# ── 8. §K learner-corpus surprisal — per repo ───────────────────────────
for i in "${!REPOS[@]}"; do
  stage "8-${CURRICULA[$i]}" "learner-corpus-${CURRICULA[$i]}" \
    python -m analyze.learner_corpus \
      --repo "${REPOS[$i]}" --curriculum "${CURRICULA[$i]}"
done

# ── 9. Render figures from the CSVs we just produced ────────────────────
for fig in figI1_geometry figI5_logit_lens figI4_meco_layerwise \
           figI10_learner_corpus figM1_schematic; do
  stage "9-$fig" "fig-$fig" python -m plots.make_$fig
done

echo "" | tee -a "$RUN_LOG"
echo "[run] all stages complete @ $(date -u +%H:%M:%SZ)" | tee -a "$RUN_LOG"
push "cpu/final"
