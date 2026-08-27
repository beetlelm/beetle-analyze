#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# eval_loss_curves_bgpt.sh
#
# Compute FLORES NLL trajectories for the 16 Catherine Arnett B-GPT
# baselines. B-GPT publishes 47 numeric-step branches (0..128000) which are
# now picked up by analyze.convergence.list_checkpoints (numeric-branch
# regex extension).
#
# Held-out is OFF by default (B-GPT did not release training corpora). To
# score against a proxy held-out (e.g. decontaminated FineWeb in beetle-data),
# pass HELDOUT_PROXY=1 — the helper resolves it via beetle-data fallback.
#
# Usage:
#   bash scripts/eval_loss_curves_bgpt.sh
#   N_GPUS=8 RANK=0 WORLD_SIZE=8 bash scripts/...
#   LANG_FILTER=pl bash scripts/...
#   HELDOUT_PROXY=1 bash scripts/...
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."

# Self-launcher: when invoked without RANK, fan out across visible GPUs
# under the shared safety contract (RAM-aware cap, setsid, trap-protected
# peer termination). Re-execs this same script per rank.
if [[ -z "${RANK:-}" ]]; then
    # shellcheck disable=SC1091
    source scripts/_sweep_safety.sh
    sweep_safety_init
    SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-logs/loss_curves}"
    export OVERWRITE LANG_FILTER HF_TOKEN SAMPLE_SIZE CKPT_SCOPE HELD_OUT_DIR BEETLE_DATA HELDOUT_PROXY
    echo "[loss-curves/bgpt] dispatching via launch_ranks"
    launch_ranks "loss_curves_bgpt" bash "$0"
    exit $?
fi

source scripts/_loss_curves_lib.sh

mapfile -t REPOS < <(python - <<'PY'
from models import BGPT_BASELINES
for r in BGPT_BASELINES:
    print(r)
PY
)

if [[ -n "${LANG_FILTER:-}" ]]; then
  mapfile -t REPOS < <(filter_repos_by_lang "$LANG_FILTER" "${REPOS[@]}")
fi

mapfile -t MY_REPOS < <(shard_repos "${REPOS[@]}")
echo "[loss-curves/bgpt] rank=${RANK}/${WORLD_SIZE} repos=${#MY_REPOS[@]}"

for repo in "${MY_REPOS[@]}"; do
  langs=$(langs_for_repo "$repo")
  if [[ -z "$langs" ]]; then
    echo "[loss-curves/bgpt] SKIP $repo (no eval languages parsed)"
    continue
  fi

  run_flores_traj "$repo" $langs || echo "[loss-curves/bgpt] FLORES failed for $repo"

  if [[ "${HELDOUT_PROXY:-0}" == "1" ]]; then
    run_heldout_traj "$repo" $langs || echo "[loss-curves/bgpt] held-out (proxy) failed for $repo"
  fi
done

echo "[loss-curves/bgpt] done."
