#!/usr/bin/env bash
# =============================================================================
# run_alta_blimp.sh — Evaluate RA-ALTA models on BLiMP across N GPUs.
#
# Each GPU process handles a slice of the 19 models (models[rank::world_size]).
# Results are written to a single shared CSV (process-safe via flock).
# BLiMP pairs are downloaded once by rank-0 and cached as a .pkl file —
# all other ranks wait for the cache before starting evaluation.
#
# Usage:
#   bash run_alta_blimp.sh                   # uses all visible GPUs
#   bash run_alta_blimp.sh --gpus 4          # use 4 GPUs (cuda:0..3)
#   bash run_alta_blimp.sh --gpus 1          # single GPU (debug)
#   bash run_alta_blimp.sh --batch_size 16   # smaller batch if OOM
#   bash run_alta_blimp.sh --no_resume       # re-evaluate everything
#   bash run_alta_blimp.sh --hf_token TOKEN  # for gated models
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
N_GPUS=$(python3 -c "import torch; print(max(1, torch.cuda.device_count()))" 2>/dev/null || echo 1)
BATCH_SIZE=32
HF_TOKEN="${HF_TOKEN:-}"
EXTRA_FLAGS=""

OUTPUT_DIR="$(pwd)/alta_blimp_results"
SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_alta_blimp.py"

# ── Parse flags ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)        shift; N_GPUS="$1" ;;
    --gpus=*)      N_GPUS="${1#*=}" ;;
    --batch_size)  shift; BATCH_SIZE="$1" ;;
    --batch_size=*)BATCH_SIZE="${1#*=}" ;;
    --hf_token)    shift; HF_TOKEN="$1" ;;
    --hf_token=*)  HF_TOKEN="${1#*=}" ;;
    --no_resume)   EXTRA_FLAGS="$EXTRA_FLAGS --no_resume" ;;
    --output_dir)  shift; OUTPUT_DIR="$1" ;;
    --output_dir=*)OUTPUT_DIR="${1#*=}" ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
  shift
done

mkdir -p "${OUTPUT_DIR}"

OUTPUT_CSV="${OUTPUT_DIR}/blimp_results_alta.csv"
PKL_CACHE="${OUTPUT_DIR}/.blimp_cache.pkl"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "================================================================"
echo "  RA-ALTA × BLiMP Evaluation"
echo "  GPUs       : ${N_GPUS}"
echo "  Batch size : ${BATCH_SIZE}"
echo "  Output CSV : ${OUTPUT_CSV}"
echo "  Logs       : ${LOG_DIR}/"
echo "================================================================"
echo ""

# ── Build common python args ──────────────────────────────────────────────────
COMMON_ARGS=(
  --output_csv  "${OUTPUT_CSV}"
  --batch_size  "${BATCH_SIZE}"
  --pkl_cache   "${PKL_CACHE}"
  --world_size  "${N_GPUS}"
)
[[ -n "${HF_TOKEN}"    ]] && COMMON_ARGS+=(--hf_token "${HF_TOKEN}")
[[ -n "${EXTRA_FLAGS}" ]] && COMMON_ARGS+=(${EXTRA_FLAGS})

# ── Launch one process per GPU ────────────────────────────────────────────────
PIDS=()

for (( rank=0; rank<N_GPUS; rank++ )); do
  log="${LOG_DIR}/rank${rank}.log"
  echo "  Launching rank=${rank} (CUDA_VISIBLE_DEVICES=${rank}) → ${log}"

  CUDA_VISIBLE_DEVICES="${rank}" \
    python3 "${SCRIPT}" \
      --rank "${rank}" \
      "${COMMON_ARGS[@]}" \
    > "${log}" 2>&1 &

  PIDS+=($!)

  # Stagger rank-0 first so it can download+cache BLiMP before others start.
  if (( rank == 0 )); then
    echo "  Rank 0 downloading BLiMP cache — waiting 10s before launching remaining ranks …"
    sleep 10
  else
    sleep 2
  fi
done

echo ""
echo "All ${#PIDS[@]} process(es) running."
echo ""
echo "  Live logs    : tail -f ${LOG_DIR}/rank*.log"
echo "  Results so far: wc -l ${OUTPUT_CSV}"
echo ""

# ── Wait for all processes ────────────────────────────────────────────────────
FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    echo "WARNING: PID ${pid} exited non-zero." >&2
    FAIL=1
  fi
done

echo ""
if [[ "${FAIL}" -eq 0 ]]; then
  echo "All evaluations completed successfully."
  echo "Results: ${OUTPUT_CSV}"
  echo ""
  # Quick summary: mean accuracy per model
  python3 - <<'EOF' "${OUTPUT_CSV}"
import csv, sys, collections

path = sys.argv[1]
totals = collections.defaultdict(list)
try:
    with open(path) as f:
        for row in csv.DictReader(f):
            totals[row["model"]].append(float(row["accuracy"]))
except FileNotFoundError:
    print("(no results file found)")
    sys.exit(0)

print(f"{'Model':<45}  {'Phenomena':>9}  {'Mean Acc':>9}")
print("-" * 68)
for model, accs in sorted(totals.items()):
    print(f"{model:<45}  {len(accs):>9}  {sum(accs)/len(accs):>9.4f}")
EOF
else
  echo "Some processes failed — check logs in ${LOG_DIR}/"
  exit 1
fi