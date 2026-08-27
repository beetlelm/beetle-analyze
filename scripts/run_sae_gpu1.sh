#!/usr/bin/env bash
# GPU1 SAE training (3 curricula × ~10M tokens). Runs after main GPU1 pipeline finished.
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

echo "[GPU1 SAE pipeline START] $(date -Is)"

for cur in b1 b3 b5; do
  repo="${REPOS[$cur]}"
  run_stage "sae2_${cur}" "$PY" -m analyze.sae.train_curriculum_sae \
    --repo "$repo" --curriculum "$cur" --layer 6 \
    --n-tokens 10000000 --n-features 16384 --top-k 64 --batch-size 8192
  push_stage "gpu1-sae2-${cur}"
done

# Re-render SAE figure now that we have data
run_stage fig_figI6_sae_causal2 "$PY" -m plots.make_figI6_sae_causal
push_stage gpu1-sae2-figures

echo "[GPU1 SAE pipeline END] $(date -Is)"
