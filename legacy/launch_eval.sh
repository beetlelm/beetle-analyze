#!/usr/bin/env bash
# =============================================================================
# launch_all.sh — Launch BeetleLM evaluation sweep across 8 A100s.
#
# Each GPU process runs eval_model.py on its model slice (ALL_MODELS[rank::8]).
# Models are evaluated one at a time — only one model ever in RAM per GPU.
# After every model, results are committed and pushed to:
#     git@github.com:suchirsalhan/beetle-analyze.git
#
# Usage:
#   cd /path/to/beetle-analyze
#   bash eval/launch_all.sh                  # local, 8 GPUs
#   bash eval/launch_all.sh --mode slurm     # SLURM
#   bash eval/launch_all.sh --no_push        # skip git push (debug)
#   bash eval/launch_all.sh --world_size 4   # use 4 GPUs only
#   bash eval/launch_all.sh --latest_only    # evaluate only the highest step-N ckpt
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MODE="${MODE:-local}"
N_GPUS=8
BATCH_SIZE=64
NO_PUSH_FLAG=""
LATEST_ONLY_FLAG=""

# The repo root is wherever this script is called FROM (beetle-analyze/).
# results/ will be created inside it.
REPO_ROOT="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${REPO_ROOT}/results/logs"
HF_TOKEN="${HF_TOKEN:-}"

# ── Parse flags ───────────────────────────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    --mode=*)       MODE="${arg#*=}"        ;;
    --mode)         shift; MODE="$1"        ;;
    --no_push)      NO_PUSH_FLAG="--no_push" ;;
    --latest_only)  LATEST_ONLY_FLAG="--latest_only" ;;
    --world_size=*) N_GPUS="${arg#*=}"      ;;
    --world_size)   shift; N_GPUS="$1"      ;;
  esac
done

mkdir -p "${REPO_ROOT}/results" "${LOG_DIR}"

# ── Git sanity check ──────────────────────────────────────────────────────────
echo "Checking git remote …"
if ! git -C "${REPO_ROOT}" remote get-url origin &>/dev/null; then
  echo "ERROR: no git remote 'origin' found in ${REPO_ROOT}."
  echo "       Run: git remote add origin git@github.com:suchirsalhan/beetle-analyze.git"
  exit 1
fi
REMOTE_URL="$(git -C "${REPO_ROOT}" remote get-url origin)"
echo "  Remote : ${REMOTE_URL}"

# Pull latest so we start clean and avoid immediate push conflicts
if [[ -z "${NO_PUSH_FLAG}" ]]; then
  echo "  Pulling latest from origin/main …"
  git -C "${REPO_ROOT}" pull --rebase origin main 2>/dev/null || true
fi
echo ""

# ── Shared HF dataset cache ───────────────────────────────────────────────────
# All 8 processes read from one directory — prevents concurrent writes to
# ~/.cache/huggingface which causes cache file corruption under high concurrency.
HF_DATASETS_CACHE="${REPO_ROOT}/results/.hf_cache"
export HF_DATASETS_CACHE
mkdir -p "${HF_DATASETS_CACHE}"

# ── Pre-fetch all_configs datasets and save to .pkl ────────────────────────
# Instead of relying on the HF datasets cache (unreliable under 8 concurrent
# processes), we download each multi-config dataset ONCE here and save pairs
# as a plain pickle file.  Workers load the .pkl directly — zero HTTP
# requests during evaluation, no cache misses, no 429s.
PKL_DIR="${REPO_ROOT}/results/.pkl_cache"
mkdir -p "${PKL_DIR}"

echo "Pre-fetching all_configs datasets → ${PKL_DIR} ..."
PKL_DIR="${PKL_DIR}" python3 "${SCRIPT_DIR}/prefetch_datasets.py"
echo ""

echo "================================================================"
echo "  BeetleLM evaluation — model-at-a-time, ${N_GPUS} GPUs"
echo "  Mode       : ${MODE}"
echo "  Repo root  : ${REPO_ROOT}"
echo "  Results    : ${REPO_ROOT}/results/"
[[ -n "${LATEST_ONLY_FLAG}" ]] && \
echo "  Checkpoints: latest step-N only"
echo "================================================================"
echo ""

# =============================================================================
# LOCAL MODE — one background process per GPU
# =============================================================================
if [[ "${MODE}" == "local" ]]; then

  PIDS=()

  for (( rank=0; rank<N_GPUS; rank++ )); do
    log="${LOG_DIR}/rank${rank}.log"

    # Build command as an array (no word-splitting issues)
    CMD=(
      python3 "${SCRIPT_DIR}/eval_model.py"
        --rank        "${rank}"
        --world_size  "${N_GPUS}"
        --output_dir  "${REPO_ROOT}"
        --batch_size  "${BATCH_SIZE}"
        --resume
    )
    [[ -n "${NO_PUSH_FLAG}"     ]] && CMD+=(--no_push)
    [[ -n "${LATEST_ONLY_FLAG}" ]] && CMD+=(--latest_only)
    [[ -n "${HF_TOKEN}"         ]] && CMD+=(--hf_token "${HF_TOKEN}")

    echo "  Launching rank=${rank} → ${log}"

    # CUDA_VISIBLE_DEVICES restricts this process to one physical GPU.
    # CUDA renumbers it as cuda:0 inside the process, so eval_model.py
    # always uses cuda:0 — no out-of-range GPU index errors.
    CUDA_VISIBLE_DEVICES="${rank}" \
    HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
      "${CMD[@]}" > "${log}" 2>&1 &

    PIDS+=($!)

    # Stagger: 4s between each rank launch so they don't all hit the HF
    # Hub for the same first model at exactly the same millisecond.
    (( rank < N_GPUS - 1 )) && sleep 4
  done

  echo ""
  echo "All ${#PIDS[@]} processes running."
  echo ""
  echo "  Monitor live:   tail -f ${LOG_DIR}/rank*.log"
  echo "  Count results:  watch -n 10 'wc -l ${REPO_ROOT}/results/*.csv 2>/dev/null'"
  echo ""

  FAIL=0
  for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
      echo "WARNING: PID ${pid} exited non-zero"
      FAIL=1
    fi
  done

  if [[ "${FAIL}" -eq 0 ]]; then
    echo "All evaluations completed successfully."
  else
    echo "Some processes failed. Check logs in ${LOG_DIR}/"
    exit 1
  fi

# =============================================================================
# SLURM MODE — single job, 8 tasks
# =============================================================================
elif [[ "${MODE}" == "slurm" ]]; then

  jobscript="${LOG_DIR}/job_beetlelm.sh"

  cat > "${jobscript}" << SLURM
#!/usr/bin/env bash
#SBATCH --job-name=beetlelm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${N_GPUS}
#SBATCH --gres=gpu:${N_GPUS}
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=${LOG_DIR}/slurm_%j_%t.log
#SBATCH --error=${LOG_DIR}/slurm_%j_%t.err

module load cuda/12.1
source activate beetlelm    # ← update to match your environment

export HF_TOKEN="${HF_TOKEN}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"

srun --ntasks=${N_GPUS} --ntasks-per-node=${N_GPUS} bash -c "
  rank=\${SLURM_LOCALID}
  sleep \$((rank * 4))
  CUDA_VISIBLE_DEVICES=\${rank} \\
  HF_DATASETS_CACHE='${HF_DATASETS_CACHE}' \\
  python3 '${SCRIPT_DIR}/eval_model.py' \\
    --rank \${rank} \\
    --world_size ${N_GPUS} \\
    --output_dir '${REPO_ROOT}' \\
    --batch_size ${BATCH_SIZE} \\
    --resume \\
    ${NO_PUSH_FLAG} \\
    ${LATEST_ONLY_FLAG} \\
    \$([ -n '${HF_TOKEN}' ] && echo '--hf_token ${HF_TOKEN}')
"
SLURM

  job_id=$(sbatch --parsable "${jobscript}")
  echo "Submitted → job ${job_id}"
  echo "Monitor:  squeue -u \$USER"
  echo "Logs:     ${LOG_DIR}/slurm_${job_id}_*.log"

else
  echo "Unknown MODE='${MODE}'. Use: local | slurm"
  exit 1
fi
