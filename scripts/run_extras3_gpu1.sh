#!/usr/bin/env bash
# GPU1 extras (round 3): probe alignment + cognate surprisal + SAE feature extraction.
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

echo "[GPU1 extras3 START] $(date -Is)"

# Probe alignment (§A, figI1 cosine panel) — runs across all 3 curricula
run_stage probe_alignment "$PY" -m analyze.xling.probe_alignment \
  --n-words 2000 \
  && push_stage gpu1-probe-alignment

# Cognate surprisal (cognate / false-friend / non-cognate)
run_stage cognate_surprisal "$PY" -m analyze.cognate_surprisal \
  --repos "${REPOS[b1]}" "${REPOS[b3]}" "${REPOS[b5]}" \
  --overwrite \
  && push_stage gpu1-cognate-surprisal

# SAE feature extraction for the 3 trained SAEs (uses the sae.pt we wrote earlier)
SAE_ROOT="$BEETLE_RESULTS_ROOT/06_sae"
for cur in b1 b3 b5; do
  case "$cur" in
    b1) sae_dir="$SAE_ROOT/b1__nld-eng__layer=6/sae/Beetle-FineWeb__beetle-bilingual-balanced-b1-fineweb-nld-eng/main/layer6/n16384_k64/lr0.0003" ;;
    b3) sae_dir="$SAE_ROOT/b3__nld-eng__layer=6/sae/Beetle-FineWeb__beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-nld-eng/main/layer6/n16384_k64/lr0.0003" ;;
    b5) sae_dir="$SAE_ROOT/b5__nld-eng__layer=6/sae/Beetle-FineWeb__beetle-bilingual-balanced-b5-fineweb-nld-eng/main/layer6/n16384_k64/lr0.0003" ;;
  esac
  run_stage "sae_extract_${cur}" "$PY" -m analyze.sae.extract_features \
    --repo "${REPOS[$cur]}" --layer 6 --sae-dir "$sae_dir" \
    --output-dir "$SAE_ROOT/${cur}__nld-eng__layer=6/features"
  push_stage "gpu1-sae-extract-${cur}"
done

# Re-render figI1 (now with probe-cosine column)
run_stage fig_figI1_geometry "$PY" -m plots.make_figI1_geometry \
  && push_stage gpu1-figI1-geometry

echo "[GPU1 extras3 END] $(date -Is)"
