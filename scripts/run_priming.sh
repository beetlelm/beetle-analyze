#!/usr/bin/env bash
#
# scripts/run_priming.sh — Drive Arnett (2025) priming replication on Beetle.
#
# Stage A (§3.1 loss curves) and Stage B (§4 structural priming) for every
# Beetle bilingual eng-X / X-eng model at every checkpoint, fanned out across
# all visible GPUs. Results are written to results/evals/priming/. After all
# ranks finish, a comparison job joins our Beetle numbers with Arnett's
# published BGPT results from arnett-2025-priming-raw/.
#
# Usage:
#   scripts/run_priming.sh                     # all eng-X/X-eng Beetle bilinguals
#   scripts/run_priming.sh --max-checkpoints 12  # coarse trajectory sweep
#   scripts/run_priming.sh --revisions main      # final-checkpoint smoke test
#   scripts/run_priming.sh --skip-priming        # just §3.1 loss curves
#
# Environment:
#   N_GPUS    number of ranks to launch (default: nvidia-smi -L | wc -l)
#   MODELS    --models argument (default: "beetle-humanscale beetle-fineweb")

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODELS=${MODELS:-"beetle-humanscale beetle-fineweb"}
N_GPUS=${N_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}
N_GPUS=${N_GPUS:-1}

OUTPUT_DIR="results/evals/priming"
mkdir -p "$OUTPUT_DIR/logs"

echo "[run_priming] N_GPUS=${N_GPUS} MODELS='${MODELS}'"
echo "[run_priming] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[run_priming] Forwarding extra args: $*"

PIDS=()
for ((rank=0; rank<N_GPUS; rank++)); do
    log="${OUTPUT_DIR}/logs/rank${rank}.log"
    echo "[run_priming] launching rank ${rank}/${N_GPUS} → ${log}"
    CUDA_VISIBLE_DEVICES=${rank} \
        python evals/eval_priming.py \
            --models ${MODELS} \
            --output "${OUTPUT_DIR}" \
            --rank "${rank}" --world-size "${N_GPUS}" \
            "$@" \
            > "${log}" 2>&1 &
    PIDS+=($!)
done

# Wait on all ranks; report any non-zero exits.
fail=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        echo "[run_priming] rank pid=${pid} failed"
        fail=1
    fi
done

if [[ "${fail}" -ne 0 ]]; then
    echo "[run_priming] one or more ranks failed; see ${OUTPUT_DIR}/logs/."
    exit 1
fi

echo "[run_priming] all ranks done; running comparison vs Arnett BGPT."
python evals/compare_priming_to_bgpt.py \
    --beetle-dir "${OUTPUT_DIR}" \
    --arnett-dir arnett-2025-priming-raw \
    --output     "${OUTPUT_DIR}/comparison"

echo "[run_priming] done. See ${OUTPUT_DIR}/comparison/summary.md"
