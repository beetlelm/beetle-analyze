#!/usr/bin/env bash
# scripts/run_interp_pipeline.sh — Tiered EMNLP-interp orchestrator.
#
# Three iteration buckets per the user spec:
#
#   --tier smoke   : ~5h on a Colab A100 (or 30 min on CPU with mocked sizes).
#                    Runs §A geometry + §I external baseline + §D logit lens
#                    + §H pilot + figI1/figI4/figI5. The 5-hour first
#                    paper-ready iteration.
#
#   --tier expand  : ~10h on Flamingo (or 8×A100). Adds §E patching, §G AoA
#                    (with frequency residualisation), §F b5 attrition,
#                    §K learner corpus, §J dev trajectory (if Phase 0 OK).
#                    The 10-hour second iteration.
#
#   --tier full    : 2-week run. Adds §B SAE training (M+2), §C brain
#                    encoding (gated), §A↔§B chaining, deu-eng + zho-eng
#                    replications, qualitative example curation. Drives
#                    every figI*.
#
# All stages are idempotent — re-running picks up where it left off via
# the eval ledger and per-stage CSVs.
#
# Environment
# -----------
#   BEETLE_FORCE_CPU=1   force CPU even if CUDA is visible (smoke runs)
#   N_PAIRS=512          override sentence-pair budget for §A / §J
#   OUT_ROOT=results/evals_paper

set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 --tier {smoke|expand|full} [--cpu] [--out-root DIR]

Tiers:
  smoke  ~5h: §A + §I + §D + §H pilot + figI1/figI4/figI5
  expand ~10h: smoke + §E + §G + §F + §K + §J
  full   2 weeks: expand + §B SAE + §C brain + chaining + replications

Examples:
  ./scripts/run_interp_pipeline.sh --tier smoke --cpu
  BEETLE_FORCE_CPU=1 N_PAIRS=32 ./scripts/run_interp_pipeline.sh --tier smoke
  ./scripts/run_interp_pipeline.sh --tier full
EOF
}

TIER=""
CPU=0
OUT_ROOT="${OUT_ROOT:-results/evals_paper}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier)     TIER="$2"; shift 2 ;;
    --cpu)      CPU=1; shift ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    -h|--help)  usage; exit 0 ;;
    *) echo "unknown arg: $1"; usage; exit 1 ;;
  esac
done
[[ -z "$TIER" ]] && { usage; exit 1; }

if [[ $CPU -eq 1 ]]; then
  export BEETLE_FORCE_CPU=1
fi

N_PAIRS="${N_PAIRS:-512}"
[[ $TIER == "smoke" && $CPU -eq 1 ]] && N_PAIRS="${N_PAIRS:-32}"

cd "$(dirname "$0")/.."
mkdir -p "$OUT_ROOT" plots

REPOS_BEETLE=(
  "Beetle-FineWeb/beetle-bilingual-b1-nld-eng"
  "Beetle-FineWeb/beetle-bilingual-b3-nld-eng"
  "Beetle-FineWeb/beetle-bilingual-b5-nld-eng"
)
CURRICULA=(b1 b3 b5)

run() {
  local label="$1"; shift
  echo "[interp] >>> $label"
  if "$@"; then
    echo "[interp]     OK  $label"
  else
    echo "[interp]     FAIL  $label  (continuing)"
  fi
}

# ── Tier: smoke ─────────────────────────────────────────────────────────
smoke_phase() {
  run "§A CKA (parallel)" python -m analyze.xling.cka_curriculum \
    --n-pairs "$N_PAIRS" --n-boot 200 \
    --repos "${REPOS_BEETLE[@]}" --curricula "${CURRICULA[@]}"

  run "§I external baseline" python -m analyze.xling.external_baseline \
    --n-pairs "$N_PAIRS" --n-boot 200

  run "§D logit lens" python -m analyze.lens.logit_lens_lang \
    --repos "${REPOS_BEETLE[@]}" --curricula "${CURRICULA[@]}" \
    --n-sentences 32

  run "§H pilot (layerwise MECO)" python -m analyze.meco_layerwise --pilot \
    --repos "${REPOS_BEETLE[@]}" --curricula "${CURRICULA[@]}"

  run "figI1" python -m plots.make_figI1_geometry
  run "figI4" python -m plots.make_figI4_meco_layerwise
  run "figI5" python -m plots.make_figI5_logit_lens
  run "figM1 schematic" python -m plots.make_figM1_schematic
}

# ── Tier: expand ────────────────────────────────────────────────────────
expand_phase() {
  smoke_phase

  run "§A cognate split"  python -m analyze.xling.cka_curriculum --mode cognate-split
  run "§A non-parallel"   python -m analyze.xling.cka_curriculum --mode non-parallel

  for i in "${!REPOS_BEETLE[@]}"; do
    run "§E patching ${CURRICULA[$i]}" python -m analyze.causal.cognate_patch \
      --repo "${REPOS_BEETLE[$i]}" --curriculum "${CURRICULA[$i]}"
    run "§K learner corpus ${CURRICULA[$i]}" python -m analyze.learner_corpus \
      --repo "${REPOS_BEETLE[$i]}" --curriculum "${CURRICULA[$i]}"
  done

  run "§F b5 attrition" python -m analyze.forgetting_pkg.l1_attrition

  if [[ -f "aoa/word_list_for_traj.csv" ]]; then
    for i in "${!REPOS_BEETLE[@]}"; do
      run "§G AoA ${CURRICULA[$i]}" python -m aoa.word_learning_traj \
        --repo "${REPOS_BEETLE[$i]}" --curriculum "${CURRICULA[$i]}" \
        --word-list-csv aoa/word_list_for_traj.csv
    done
  else
    echo "[interp]     skip §G — aoa/word_list_for_traj.csv not present"
  fi

  run "§J dev trajectory" python -m analyze.xling.dev_trajectory \
    --repo "${REPOS_BEETLE[1]}" --curriculum b3 --n-pairs "$N_PAIRS"

  for fig in figI2_patching figI3_aoa figI7_attrition figI9_dev_trajectory \
             figI10_learner_corpus; do
    run "$fig" python -m plots.make_${fig}
  done
}

# ── Tier: full ──────────────────────────────────────────────────────────
full_phase() {
  expand_phase

  for i in "${!REPOS_BEETLE[@]}"; do
    run "§B SAE train ${CURRICULA[$i]}" python -m analyze.sae.train_curriculum_sae \
      --repo "${REPOS_BEETLE[$i]}" --curriculum "${CURRICULA[$i]}" --layer 6
    run "§B label ${CURRICULA[$i]}" python -m analyze.sae.label_l1_transfer \
      --repo "${REPOS_BEETLE[$i]}" --curriculum "${CURRICULA[$i]}" \
      --sae-dir "${OUT_ROOT}/06_sae/${CURRICULA[$i]}__nld-eng__layer=6"
  done

  if [[ -f "data/bliss_false_friend.csv" ]]; then
    for i in "${!REPOS_BEETLE[@]}"; do
      run "§B ablation ${CURRICULA[$i]}" python -m analyze.sae.causal_ablation_bliss \
        --repo "${REPOS_BEETLE[$i]}" --curriculum "${CURRICULA[$i]}" \
        --sae-dir "${OUT_ROOT}/06_sae/${CURRICULA[$i]}__nld-eng__layer=6" \
        --feature-table "${OUT_ROOT}/06_sae/${CURRICULA[$i]}__nld-eng__layer=6/feature_labels.csv" \
        --bliss-csv data/bliss_false_friend.csv
    done
    run "§A↔§B chain" python -m analyze.sae.chaining_analysis \
      --repo "${REPOS_BEETLE[1]}" --curriculum b3 \
      --sae-dir "${OUT_ROOT}/06_sae/b3__nld-eng__layer=6" \
      --feature-table "${OUT_ROOT}/06_sae/b3__nld-eng__layer=6/feature_labels.csv" \
      --bliss-csv data/bliss_false_friend.csv
  fi

  if [[ -f "data/zuco_l2_responses.csv" ]]; then
    run "§C brain encoder" python -m analyze.brain.encoder_curriculum \
      --brain-csv data/zuco_l2_responses.csv
  fi

  for fig in figI6_sae_causal figIA1_examples; do
    run "$fig" python -m plots.make_${fig}
  done
}

case "$TIER" in
  smoke)  smoke_phase ;;
  expand) expand_phase ;;
  full)   full_phase ;;
  *) echo "unknown tier: $TIER"; usage; exit 1 ;;
esac

echo "[interp] tier=$TIER complete."
