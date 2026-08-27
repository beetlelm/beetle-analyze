#!/usr/bin/env bash
# GPU1 extras (round 2): meco + extra trajectory/attrition variants + figures.
set -u
set -o pipefail

REPO=/local/scratch/sas245/beetle-analyze
VENV=/local/scratch/sas245/venvs/beetle-emnlp
LOG_DIR="$REPO/logs/emnlp_interp_local/gpu1"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=1
export BEETLE_REPO_ROOT="$REPO"
export BEETLE_RESULTS_ROOT="$REPO/results/evals_paper"
export BEETLE_PUSH_BRANCH=emnlp-interp
export PATH="$VENV/bin:$PATH"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/local/scratch/sas245/huggingface_cache"
export HF_HUB_CACHE="/local/scratch/sas245/huggingface_cache/hub"
export HF_DATASETS_CACHE="/local/scratch/sas245/huggingface_cache/datasets"

cd "$REPO"
PY="$VENV/bin/python"

push_stage() {
  local stage="$1"
  flock "$REPO/.push.lock" -c "
    cd '$REPO'
    $PY -c \"from analyze.results_push import push; push('$stage')\"
  " 2>&1 | tee -a "$LOG_DIR/push.log"
}

run_stage() {
  local name="$1"; shift
  local logf="$LOG_DIR/${name}.log"
  echo "[$(date -Is)] [GPU1] STAGE $name START"
  echo "[$(date -Is)] STAGE $name START" >> "$logf"
  "$@" >> "$logf" 2>&1
  local rc=$?
  echo "[$(date -Is)] STAGE $name END rc=$rc" >> "$logf"
  echo "[$(date -Is)] [GPU1] STAGE $name END rc=$rc"
  return $rc
}

declare -A REPOS=(
  [b1]="Beetle-FineWeb/beetle-bilingual-balanced-b1-fineweb-nld-eng"
  [b3]="Beetle-FineWeb/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-nld-eng"
  [b5]="Beetle-FineWeb/beetle-bilingual-balanced-b5-fineweb-nld-eng"
)

echo "[GPU1 extras2 START] $(date -Is)"

# §H meco_layerwise (real run, no --pilot, all 3 curricula)
run_stage meco_layerwise_full2 "$PY" -m analyze.meco_layerwise \
  && push_stage gpu1-meco-layerwise-full

# Extra dev_trajectory: b1 (control) and b5 (alt)
DEV_DIR="$BEETLE_RESULTS_ROOT/14_dev_trajectory"
mkdir -p "$DEV_DIR"
run_stage dev_trajectory_b1 "$PY" -m analyze.xling.dev_trajectory \
  --repo "${REPOS[b1]}" --curriculum b1 --n-pairs 200 --max-revisions 14 \
  --out "$DEV_DIR/per_layer_cka_by_step_b1.csv" \
  && push_stage gpu1-dev-trajectory-b1

run_stage dev_trajectory_b5 "$PY" -m analyze.xling.dev_trajectory \
  --repo "${REPOS[b5]}" --curriculum b5 --n-pairs 200 --max-revisions 14 \
  --out "$DEV_DIR/per_layer_cka_by_step_b5.csv" \
  && push_stage gpu1-dev-trajectory-b5

# Extra l1_attrition: b3 and b1 traces
ATT_DIR="$BEETLE_RESULTS_ROOT/10_attrition"
run_stage l1_attrition_b3 "$PY" -m analyze.forgetting_pkg.l1_attrition \
  --repo "${REPOS[b3]}" --curriculum b3 --n-pairs 200 --max-revisions 14 \
  --out "$ATT_DIR/b3_l1_metrics_trace.csv" \
  && push_stage gpu1-l1-attrition-b3

run_stage l1_attrition_b1 "$PY" -m analyze.forgetting_pkg.l1_attrition \
  --repo "${REPOS[b1]}" --curriculum b1 --n-pairs 200 --max-revisions 14 \
  --out "$ATT_DIR/b1_l1_metrics_trace.csv" \
  && push_stage gpu1-l1-attrition-b1

# Render figures from the new CSVs
for fig in figI2_patching figI4_meco_layerwise figI5_logit_lens \
           figI10_learner_corpus; do
  run_stage "fig_${fig}" "$PY" -m "plots.make_${fig}"
done
push_stage gpu1-extras2-figures

echo "[GPU1 extras2 END] $(date -Is)"
