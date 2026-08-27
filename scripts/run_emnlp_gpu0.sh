#!/usr/bin/env bash
# GPU0 pipeline — single-model GPU evals (CKA, baselines, patching, lens, meco).
# Uses physical GPU 0 only.
set -u
set -o pipefail

REPO=/local/scratch/sas245/beetle-analyze
VENV=/local/scratch/sas245/venvs/beetle-emnlp
LOG_DIR="$REPO/logs/emnlp_interp_local/gpu0"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=0
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
  echo "[$(date -Is)] [GPU0] STAGE $name START"
  echo "[$(date -Is)] STAGE $name START" >> "$logf"
  "$@" >> "$logf" 2>&1
  local rc=$?
  echo "[$(date -Is)] STAGE $name END rc=$rc" >> "$logf"
  echo "[$(date -Is)] [GPU0] STAGE $name END rc=$rc"
  return $rc
}

declare -A REPOS=(
  [b1]="Beetle-FineWeb/beetle-bilingual-balanced-b1-fineweb-nld-eng"
  [b3]="Beetle-FineWeb/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-nld-eng"
  [b5]="Beetle-FineWeb/beetle-bilingual-balanced-b5-fineweb-nld-eng"
)

echo "[GPU0 pipeline START] $(date -Is)"

# §A CKA parallel (full)
run_stage cka_parallel "$VENV/bin/beetle-cka" --n-pairs 1000 --n-boot 1000 --mode parallel \
  && push_stage gpu0-cka-parallel

# §A cognate-split confound (reduced n_boot for time budget)
run_stage cka_cognate_split "$VENV/bin/beetle-cka" --n-pairs 1000 --n-boot 200 --mode cognate-split \
  && push_stage gpu0-cka-cognate-split

# §A non-parallel confound (reduced n_boot for time budget)
run_stage cka_non_parallel "$VENV/bin/beetle-cka" --n-pairs 1000 --n-boot 200 --mode non-parallel \
  && push_stage gpu0-cka-non-parallel

# §I external baseline (reduced n_boot)
# (keeps statistical signal but cuts wall time)

# §I external baseline (XLM-R + mBERT) — reduced n_boot for time budget
run_stage external_baseline "$VENV/bin/beetle-baseline" --n-pairs 1000 --n-boot 200 \
  && push_stage gpu0-external-baseline

# §E/§K/§D/§H moved to GPU1 (run_extras_gpu1.sh) to parallelise.
# Render figures (CPU, but here for convenience after CKA confounds + baseline).
for fig in figI1_geometry figM1_schematic; do
  run_stage "fig_${fig}" "$PY" -m "plots.make_${fig}"
done
push_stage gpu0-figures

echo "[GPU0 pipeline END] $(date -Is)"
