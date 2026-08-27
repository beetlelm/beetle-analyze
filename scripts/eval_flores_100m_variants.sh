#!/usr/bin/env bash
# eval_flores_100m_variants.sh — dispatch the FLORES re-eval for the 8-variant
# nld-eng B3 100M cohort. Wraps the python driver and uses the project's
# multi-rank launcher (_sweep_safety) for GPU sharding.
#
# Usage:
#   bash scripts/eval_flores_100m_variants.sh                 # all 8 variants, full trajectory
#   N_GPUS=4 bash scripts/eval_flores_100m_variants.sh        # 4-way sharding
#   OVERWRITE=1 bash scripts/eval_flores_100m_variants.sh     # bust caches
#   FINAL_ONLY=1 bash scripts/eval_flores_100m_variants.sh    # only step-9096
#   VARIANTS="pre-pretrain safe" bash scripts/eval_flores_100m_variants.sh
#
# Safety: this is eval-only — it does NOT acquire a training lock. Caller
# must verify no training is in flight on the same GPUs.
set -uo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${RANK:-}" ]]; then
    # shellcheck disable=SC1091
    source scripts/_sweep_safety.sh
    sweep_safety_init
    SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-logs/flores_100m_variants}"
    export OVERWRITE FINAL_ONLY VARIANTS HF_TOKEN
    echo "[flores-100m] dispatching via launch_ranks"
    launch_ranks "flores_100m_variants" bash "$0"
    exit $?
fi

# Per-rank execution.
PY_ARGS=()
[[ "${OVERWRITE:-0}" == "1" ]]   && PY_ARGS+=("--overwrite")
[[ "${FINAL_ONLY:-0}" == "1" ]]  && PY_ARGS+=("--final-only")
if [[ -n "${VARIANTS:-}" ]]; then
    # shellcheck disable=SC2206
    V=( ${VARIANTS} )
    PY_ARGS+=("--variants" "${V[@]}")
fi

echo "[flores-100m] rank ${RANK}/${WORLD_SIZE} → python args: ${PY_ARGS[*]}"
exec python scripts/eval_flores_100m_variants.py "${PY_ARGS[@]}"
