#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# eval_loss_curves_beetle_fineweb.sh
#
# Compute FLORES + held-out NLL trajectories for Beetle-FineWeb models.
# Pulls the active FineWeb roster from models.FINEWEB_MODELS plus any
# planned manifest entries (scripts/sync_fineweb_manifest.py).
#
# Held-out resolution looks at $BEETLE_DATA/pipeline_output/
# (held_out_babybabel/<lang>/*.parquet, then decontaminated/<iso2>/*.parquet).
# Defaults BEETLE_DATA to ../beetle-data (sibling to this repo).
#
# Usage:
#   bash scripts/eval_loss_curves_beetle_fineweb.sh
#   BEETLE_DATA=/abs/path/beetle-data bash scripts/...
#   LANG_FILTER=nld bash scripts/...
#   FLORES_ONLY=1 bash scripts/...
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
    echo "[loss-curves/fineweb] dispatching via launch_ranks"
    launch_ranks "loss_curves_fineweb" bash "$0"
    exit $?
fi

source scripts/_loss_curves_lib.sh

mapfile -t REPOS < <(python - <<'PY'
import models as M
roster = list(getattr(M, "FINEWEB_MODELS", []))
roster += list(getattr(M, "FINEWEB_MODELS_PLANNED", []))
seen = set(); out = []
for r in roster:
    if r and r not in seen:
        seen.add(r); out.append(r)
for r in out:
    print(r)
PY
)

if [[ ${#REPOS[@]} -eq 0 ]]; then
  echo "[loss-curves/fineweb] No FineWeb repos resolved — check models.FINEWEB_MODELS or sync the manifest."
  exit 0
fi

if [[ -n "${LANG_FILTER:-}" ]]; then
  mapfile -t REPOS < <(filter_repos_by_lang "$LANG_FILTER" "${REPOS[@]}")
fi

mapfile -t MY_REPOS < <(shard_repos "${REPOS[@]}")
echo "[loss-curves/fineweb] rank=${RANK}/${WORLD_SIZE} repos=${#MY_REPOS[@]}  beetle_data=${BEETLE_DATA}"

for repo in "${MY_REPOS[@]}"; do
  langs=$(langs_for_repo "$repo")
  if [[ -z "$langs" ]]; then
    echo "[loss-curves/fineweb] SKIP $repo (no eval languages parsed)"
    continue
  fi

  if [[ "${HELDOUT_ONLY:-0}" != "1" ]]; then
    run_flores_traj "$repo" $langs || echo "[loss-curves/fineweb] FLORES failed for $repo"
  fi
  if [[ "${FLORES_ONLY:-0}" != "1" ]]; then
    run_heldout_traj "$repo" $langs || echo "[loss-curves/fineweb] held-out failed for $repo"
  fi
done

echo "[loss-curves/fineweb] done."
