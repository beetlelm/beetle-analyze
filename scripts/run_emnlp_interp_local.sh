#!/usr/bin/env bash
# Local runner mirroring notebooks/colab_emnlp_interp.ipynb.
# Pins to GPU0+GPU1 only, keeps continuous push between stages.
set -u
set -o pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
VENV="${VENV:-$HOME/.venvs/beetle-analyze}"
LOG_DIR="$REPO/logs/emnlp_interp_local"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=0,1
export BEETLE_REPO_ROOT="$REPO"
export BEETLE_RESULTS_ROOT="$REPO/results/evals_paper"
export BEETLE_PUSH_BRANCH=emnlp-interp
export PATH="$VENV/bin:$PATH"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

cd "$REPO"

PY="$VENV/bin/python"

push_stage() {
  local stage="$1"
  "$PY" -c "from analyze.results_push import push; push('$stage')" 2>&1 \
    | tee -a "$LOG_DIR/push.log"
}

run_stage() {
  local name="$1"; shift
  local logf="$LOG_DIR/${name}.log"
  echo "=========================================="
  echo "[$(date -Is)] STAGE $name START"
  echo "=========================================="
  echo "[$(date -Is)] STAGE $name START" >> "$logf"
  "$@" 2>&1 | tee -a "$logf"
  local rc=${PIPESTATUS[0]}
  echo "[$(date -Is)] STAGE $name END rc=$rc" | tee -a "$logf"
  return $rc
}

echo "[$(date -Is)] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'count', torch.cuda.device_count())"

# Stage 1 — §A geometry CKA
run_stage stage1_cka "$VENV/bin/beetle-cka" --n-pairs 1000 --n-boot 1000 \
  && push_stage stage-1-geometry

# Stage 2 — §I external baseline (XLM-R + mBERT)
run_stage stage2_baseline "$VENV/bin/beetle-baseline" --n-pairs 1000 --n-boot 1000 \
  && push_stage stage-2-external-baseline

# Stage 3 — §E activation patching, all 3 curricula
declare -A BEETLE_REPOS=(
  [b1]="Beetle-FineWeb/beetle-bilingual-balanced-b1-fineweb-nld-eng"
  [b3]="Beetle-FineWeb/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-nld-eng"
  [b5]="Beetle-FineWeb/beetle-bilingual-l2-80-late-b5-fineweb-nld-eng"
)
for cur in b1 b3 b5; do
  repo="${BEETLE_REPOS[$cur]}"
  run_stage "stage3_patching_${cur}" "$PY" -m analyze.causal.cognate_patch \
    --repo "$repo" --curriculum "$cur"
done
push_stage stage-3-cognate-patching

# Stage 4 — §H MECO layerwise pilot
run_stage stage4_meco "$PY" -m analyze.meco_layerwise --pilot \
  && push_stage stage-4-meco-layerwise-pilot

# Stage 5 — §D logit-lens
run_stage stage5_logit_lens "$PY" -m analyze.lens.logit_lens_lang --n-sentences 64 \
  && push_stage stage-5-logit-lens

# Stage 6 — §K learner corpus, all 3 curricula
for cur in b1 b3 b5; do
  repo="${BEETLE_REPOS[$cur]}"
  run_stage "stage6_learner_${cur}" "$PY" -m analyze.learner_corpus \
    --repo "$repo" --curriculum "$cur"
done
push_stage stage-6-learner-corpus

# Stage 7 — render figures
for fig in figI1_geometry figI2_patching figI4_meco_layerwise \
           figI5_logit_lens figI10_learner_corpus figM1_schematic; do
  run_stage "stage7_${fig}" "$PY" -m "plots.make_${fig}"
done
push_stage stage-7-figures

# Bonus — flores BPB plot, best-effort
run_stage bonus_flores "$PY" -m plots.make_F10_flores_beetle_vs_bgpt || true
push_stage bonus-flores-bpb

echo "[$(date -Is)] ALL DONE"
