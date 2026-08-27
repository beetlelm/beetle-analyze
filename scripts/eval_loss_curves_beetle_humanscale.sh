#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# eval_loss_curves_beetle_humanscale.sh
#
# Compute FLORES + held-out NLL trajectories for every published
# Beetle-HumanScale repo (mono + B1..B5 across deu/eng/nld/zho).
#
# Usage:
#   bash scripts/eval_loss_curves_beetle_humanscale.sh
#   N_GPUS=8 RANK=0 WORLD_SIZE=8 bash scripts/...   # one rank per GPU
#   LANG_FILTER=nld bash scripts/...                # only nld-touching repos
#   FLORES_ONLY=1 bash scripts/...                  # skip held-out
#   HELDOUT_ONLY=1 bash scripts/...                 # skip FLORES
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."

# Self-launcher: fan out across visible GPUs under the shared safety
# contract (RAM-aware cap, setsid, trap-protected peer termination).
if [[ -z "${RANK:-}" ]]; then
    # shellcheck disable=SC1091
    source scripts/_sweep_safety.sh
    sweep_safety_init
    SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-logs/loss_curves}"
    export OVERWRITE LANG_FILTER HF_TOKEN SAMPLE_SIZE CKPT_SCOPE HELD_OUT_DIR BEETLE_DATA FLORES_ONLY HELDOUT_ONLY
    echo "[loss-curves/humanscale] dispatching via launch_ranks"
    launch_ranks "loss_curves_humanscale" bash "$0"
    exit $?
fi

source scripts/_loss_curves_lib.sh

# Pull HumanScale repo list from models.py at runtime.
mapfile -t REPOS < <(python - <<'PY'
from models import HUMAN_SCALE_MODELS
for r in HUMAN_SCALE_MODELS:
    print(r)
PY
)

if [[ -n "${LANG_FILTER:-}" ]]; then
  mapfile -t REPOS < <(filter_repos_by_lang "$LANG_FILTER" "${REPOS[@]}")
fi

mapfile -t MY_REPOS < <(shard_repos "${REPOS[@]}")

echo "[loss-curves/humanscale] rank=${RANK}/${WORLD_SIZE} repos=${#MY_REPOS[@]}"

for repo in "${MY_REPOS[@]}"; do
  langs=$(langs_for_repo "$repo")
  if [[ -z "$langs" ]]; then
    echo "[loss-curves/humanscale] SKIP $repo (no eval languages parsed)"
    continue
  fi

  if [[ "${HELDOUT_ONLY:-0}" != "1" ]]; then
    run_flores_traj "$repo" $langs || echo "[loss-curves/humanscale] FLORES failed for $repo"
  fi
  if [[ "${FLORES_ONLY:-0}" != "1" ]]; then
    run_heldout_traj "$repo" $langs || echo "[loss-curves/humanscale] held-out failed for $repo"
  fi
done

echo "[loss-curves/humanscale] done."
