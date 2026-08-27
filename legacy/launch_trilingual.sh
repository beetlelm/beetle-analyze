#!/usr/bin/env bash
set -euo pipefail

# ════════════════════════════════════════════════════════════════════════════
# run_tri_seq_eval.sh — evaluate trilingual-sequential BeetleLM models
#
# ── CHECKPOINT MODE TOGGLE ───────────────────────────────────────────────────
#
#   Set EVAL_MODE to one of:
#
#     all            Evaluate every available checkpoint branch for each model.
#
#     final_only     Evaluate only the single checkpoint with the highest step
#                    number (i.e. the fully-trained final checkpoint).
#                    Passes --final_only to eval_model.py.
#
#     specific_steps Evaluate only the step checkpoints listed in
#                    SPECIFIC_STEPS below.
#                    Passes --steps <SPECIFIC_STEPS> to eval_model.py.
#
#     final_and_steps  Union of the above two: the listed step checkpoints
#                    PLUS the final (highest) checkpoint.
#                    Passes --final_only --steps <SPECIFIC_STEPS>.
#
EVAL_MODE="final_and_steps"   # all | final_only | specific_steps | final_and_steps

# Comma-separated step numbers used when EVAL_MODE is "specific_steps"
# or "final_and_steps".  Edit as needed.
SPECIFIC_STEPS="10000,20000,30000"

# ════════════════════════════════════════════════════════════════════════════

# ── Config ────────────────────────────────────────────────────────────────
N_GPUS=8
BATCH_SIZE=64
REPO_ROOT="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${REPO_ROOT}/results/logs_tri_seq"
HF_TOKEN="${HF_TOKEN:-}"
mkdir -p "${LOG_DIR}" "${REPO_ROOT}/results"

# ── Define ONLY TRILINGUAL_SEQUENTIAL_MODELS ──────────────────────────────
MODELS=(
  # EWC + MAML combined
  "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l5_maml_h25"
  "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l5_maml_h50"
  "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l5_maml_h100"
  "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l20_maml_h25"
  "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l20_maml_h50"
  "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l20_maml_h75"
  "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l20_maml_h100"
  "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l50_maml_h25"
  "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l50_maml_h50"
  "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l50_maml_h75"
  "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l50_maml_h100"
  "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l150_maml_h25"
  "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l150_maml_h50"
  "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l150_maml_h75"
  "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l150_maml_h100"
  # EWC only
  "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l5"
  "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l20"
  "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l50"
  "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_ewc_l150"
  # MAML only
  "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_maml_h25"
  "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_maml_h50"
  "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_maml_h75"
  "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_maml_h100"
  # Baseline
  "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_sequential_baseline"
)

# ── Export model list as a comma-separated string ─────────────────────────
MODEL_LIST=$(IFS=,; echo "${MODELS[*]}")
export MODEL_LIST

# ── Translate EVAL_MODE into eval_model.py flags ─────────────────────────
# These are the only flags that change checkpoint-selection behaviour.
# All other flags (--rank, --world_size, etc.) are appended below.
case "${EVAL_MODE}" in
  all)
    CKPT_FLAGS=""
    CKPT_DESC="all available checkpoints"
    ;;
  final_only)
    CKPT_FLAGS="--final_only"
    CKPT_DESC="final (highest step) checkpoint only"
    ;;
  specific_steps)
    CKPT_FLAGS="--steps ${SPECIFIC_STEPS}"
    CKPT_DESC="specific steps: ${SPECIFIC_STEPS}"
    ;;
  final_and_steps)
    CKPT_FLAGS="--final_only --steps ${SPECIFIC_STEPS}"
    CKPT_DESC="final checkpoint + steps: ${SPECIFIC_STEPS}"
    ;;
  *)
    echo "ERROR: unknown EVAL_MODE '${EVAL_MODE}'." >&2
    echo "       Valid values: all | final_only | specific_steps | final_and_steps" >&2
    exit 1
    ;;
esac

# ── Shared HF dataset cache ───────────────────────────────────────────────
# Must be exported so every rank writes to the same place and avoids
# redundant downloads. _pin_hf_cache() in eval_model.py also sets this via
# --output_dir, but explicit export here guarantees it for the prefetch step.
HF_DATASETS_CACHE="${REPO_ROOT}/results/.hf_cache"
export HF_DATASETS_CACHE
mkdir -p "${HF_DATASETS_CACHE}"

# ── Pre-fetch all_configs datasets → .pkl ─────────────────────────────────
# CRITICAL: blimp_eng, blimp_nl, and zhoblimp all use config_mode=all_configs.
# eval_model.py reads these from .pkl files (one per benchmark). Without the
# pkl files, every rank falls back to a simultaneous live HF download, hits
# rate limits, fails silently, and leaves all_data empty for those benchmarks
# → applicable() returns no tasks → no evaluation, no CSV writes.
#
# This single prefetch runs once before any GPU work and writes:
#   results/.pkl_cache/blimp_eng.pkl
#   results/.pkl_cache/blimp_nl.pkl
#   results/.pkl_cache/zhoblimp.pkl
PKL_DIR="${REPO_ROOT}/results/.pkl_cache"
export PKL_DIR
mkdir -p "${PKL_DIR}"

echo "Pre-fetching all_configs datasets → ${PKL_DIR} …"
PKL_DIR="${PKL_DIR}" \
HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
  python3 "${SCRIPT_DIR}/prefetch_datasets.py"
echo ""

echo "================================================================"
echo "  BeetleLM TRILINGUAL SEQUENTIAL evaluation"
echo "  GPUs        : ${N_GPUS}"
echo "  Models      : ${#MODELS[@]}"
echo "  Checkpoint  : ${CKPT_DESC}"
echo "  Repo root   : ${REPO_ROOT}"
echo "  Results     : ${REPO_ROOT}/results/"
echo "  Logs        : ${LOG_DIR}/"
echo "  Benchmarks  : blimp_eng (eng) · blimp_nl (nld) · zhoblimp (zho)"
echo "                xcomps/zho · xnli/en · xnli/zh"
echo "  Note        : Only benchmarks whose relevant_groups include 'tri'"
echo "                are evaluated.  multiblimp and xnli/fr/de/bg are"
echo "                skipped automatically for these models."
echo "================================================================"
echo ""

# ── Launch one process per GPU ────────────────────────────────────────────
PIDS=()
for (( rank=0; rank<N_GPUS; rank++ )); do
  log="${LOG_DIR}/rank${rank}.log"

  CUDA_VISIBLE_DEVICES="${rank}" \
  MODEL_LIST="${MODEL_LIST}" \
  HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
  PKL_DIR="${PKL_DIR}" \
    python3 "${SCRIPT_DIR}/eval_model.py" \
      --rank        "${rank}" \
      --world_size  "${N_GPUS}" \
      --output_dir  "${REPO_ROOT}" \
      --batch_size  "${BATCH_SIZE}" \
      --resume \
      ${CKPT_FLAGS} \
      ${HF_TOKEN:+--hf_token "${HF_TOKEN}"} \
      > "${log}" 2>&1 &

  PIDS+=($!)
  sleep 3   # stagger startup to avoid simultaneous HF metadata fetches
done

echo "Launched ${#PIDS[@]} jobs  [EVAL_MODE=${EVAL_MODE}  flags: ${CKPT_FLAGS:-<none>}]"
echo "  Monitor live:  tail -f ${LOG_DIR}/rank*.log"
echo "  Count results: watch -n 10 'wc -l ${REPO_ROOT}/results/*.csv 2>/dev/null'"
echo ""

FAIL=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || FAIL=1
done

if [[ "${FAIL}" -eq 0 ]]; then
  echo "All trilingual sequential evaluations completed successfully."
else
  echo "Some processes failed — check logs in ${LOG_DIR}/"
  exit 1
fi
