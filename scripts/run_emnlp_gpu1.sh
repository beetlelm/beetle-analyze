#!/usr/bin/env bash
# GPU1 pipeline — multi-checkpoint trajectory + heavy SAE training.
# Uses physical GPU 1 only.
set -u
set -o pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
VENV="${VENV:-$HOME/.venvs/beetle-analyze}"
LOG_DIR="$REPO/logs/emnlp_interp_local/gpu1"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=1
export BEETLE_REPO_ROOT="$REPO"
export BEETLE_RESULTS_ROOT="$REPO/results/evals_paper"
export BEETLE_PUSH_BRANCH=emnlp-interp
export PATH="$VENV/bin:$PATH"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

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

B3_REPO="Beetle-FineWeb/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-nld-eng"
B5_REPO="Beetle-FineWeb/beetle-bilingual-balanced-b5-fineweb-nld-eng"
B1_REPO="Beetle-FineWeb/beetle-bilingual-balanced-b1-fineweb-nld-eng"

echo "[GPU1 pipeline START] $(date -Is)"

# §J b3 dev trajectory — 14 ckpts × CKA across the L1→L2 transition
run_stage dev_trajectory_b3 "$PY" -m analyze.xling.dev_trajectory \
  --repo "$B3_REPO" --curriculum b3 --n-pairs 200 --max-revisions 14 \
  && push_stage gpu1-dev-trajectory-b3

# §F b5 L1-attrition retrace — 14 ckpts × Dutch FLORES BPB
run_stage l1_attrition_b5 "$PY" -m analyze.forgetting_pkg.l1_attrition \
  --repo "$B5_REPO" --curriculum b5 --n-pairs 200 --max-revisions 14 \
  && push_stage gpu1-l1-attrition-b5

# §B SAE training b1/b3/b5 (default ~500M tokens; run in sequence)
for cur in b1 b3 b5; do
  case "$cur" in
    b1) repo="$B1_REPO" ;;
    b3) repo="$B3_REPO" ;;
    b5) repo="$B5_REPO" ;;
  esac
  run_stage "sae_${cur}" "$PY" -m analyze.sae.train_curriculum_sae \
    --repo "$repo" --curriculum "$cur" --layer 6 --n-tokens 10000000 --n-features 16384
  push_stage "gpu1-sae-${cur}"
done

# Render trajectory + attrition + sae figures
for fig in figI7_attrition figI9_dev_trajectory figI6_sae_causal; do
  run_stage "fig_${fig}" "$PY" -m "plots.make_${fig}"
done
push_stage gpu1-figures

echo "[GPU1 pipeline END] $(date -Is)"
