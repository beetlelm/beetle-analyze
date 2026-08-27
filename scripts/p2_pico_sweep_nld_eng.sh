#!/usr/bin/env bash
# p2_pico_sweep_nld_eng.sh
#
# Run pico-analyze weight-only metrics (gini/hoyer/per/condition_number/norm)
# on every nld-eng Beetle checkpoint trajectory we have not yet covered, so that
# pico signatures can be aligned with downstream MECO/BLiMP/BLiSS/priming.
#
# Existing coverage (skip): FineWeb-100M × 5 curricula (already on
# emnlp-beetle-comparative branch).
#
# To do: FineWeb-2B, FineWeb (24B), HumanScale × {B1, B2, B3, B4, B5}.
#
# Output: writes JSON metric files into the pico-analyze worktree at
# ../pico-analyze-emnlp/analysis_results/<run>_pico_analysis/step_<N>/.
# Logs per repo go to ../pico-analyze-emnlp/sweep_logs/.
set -uo pipefail

PICO_ROOT="/Users/suchirsalhan/Documents/PHD/PHD/pico-analyze-emnlp"
PICO_VENV="/Users/suchirsalhan/Documents/PHD/PHD/pico-analyze/.venv"
PYTHON="${PICO_VENV}/bin/python"
ANALYZER="${PICO_ROOT}/scripts/analyze_beetle_branches.py"
PLOTTER="${PICO_ROOT}/scripts/plot_emnlp.py"
LOG_DIR="${PICO_ROOT}/sweep_logs"
SUMMARY="${LOG_DIR}/p2_nld_eng_summary.tsv"
mkdir -p "${LOG_DIR}"

PER_REPO_TIMEOUT="${PER_REPO_TIMEOUT:-1800}"   # 30 min
MAX_STEPS_FW2B="${MAX_STEPS_FW2B:-20}"
MAX_STEPS_FW="${MAX_STEPS_FW:-24}"             # FineWeb (24B) — most ckpts
MAX_STEPS_HS="${MAX_STEPS_HS:-20}"

# nld-eng repos we want, per org. Listed explicitly so we don't accidentally
# pull in deu-eng / eng-deu / zho-eng siblings.
declare -a TARGETS=(
  # Beetle-FineWeb-2B nld-eng
  "Beetle-FineWeb-2B/beetle-bilingual-balanced-b1-fineweb-2b-nld-eng|${MAX_STEPS_FW2B}"
  "Beetle-FineWeb-2B/beetle-bilingual-l2-50-simultaneous-b2-fineweb-2b-nld-eng|${MAX_STEPS_FW2B}"
  "Beetle-FineWeb-2B/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-2b-nld-eng|${MAX_STEPS_FW2B}"
  "Beetle-FineWeb-2B/beetle-bilingual-l2-50-classroom-20-b4-fineweb-2b-nld-eng|${MAX_STEPS_FW2B}"
  "Beetle-FineWeb-2B/beetle-bilingual-l2-80-late-b5-fineweb-2b-nld-eng|${MAX_STEPS_FW2B}"

  # Beetle-FineWeb (24B) nld-eng
  "Beetle-FineWeb/beetle-bilingual-balanced-b1-fineweb-nld-eng|${MAX_STEPS_FW}"
  "Beetle-FineWeb/beetle-bilingual-l2-50-simultaneous-b2-fineweb-nld-eng|${MAX_STEPS_FW}"
  "Beetle-FineWeb/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-nld-eng|${MAX_STEPS_FW}"
  "Beetle-FineWeb/beetle-bilingual-balanced-b4-fineweb-nld-eng|${MAX_STEPS_FW}"
  "Beetle-FineWeb/beetle-bilingual-balanced-b5-fineweb-nld-eng|${MAX_STEPS_FW}"

  # Beetle-HumanScale nld-eng — core 5 curricula only (skip variants for now)
  "Beetle-HumanScale/beetle-bilingual-balanced-b1-humanscale-nld-eng|${MAX_STEPS_HS}"
  "Beetle-HumanScale/beetle-bilingual-l2-50-simultaneous-b2-humanscale-nld-eng|${MAX_STEPS_HS}"
  "Beetle-HumanScale/beetle-bilingual-l2-50-sequential-33-67-b3-humanscale-nld-eng|${MAX_STEPS_HS}"
  "Beetle-HumanScale/beetle-bilingual-l2-50-classroom-20-b4-humanscale-nld-eng|${MAX_STEPS_HS}"
  "Beetle-HumanScale/beetle-bilingual-l2-80-late-b5-humanscale-nld-eng|${MAX_STEPS_HS}"
)

cd "${PICO_ROOT}"

echo -e "repo_id\tstatus\tseconds" > "${SUMMARY}"
START_TS=$(date +%s)

log() { echo "[p2-sweep $(date '+%H:%M:%S')] $*"; }

run_one() {
  local repo_id="$1"
  local max_steps="$2"
  local label
  label="$(echo "${repo_id}" | tr / _)"
  local rep_log="${LOG_DIR}/p2_${label}.log"
  log "→ ${repo_id} (max_steps=${max_steps})  log=${rep_log}"

  local t0; t0=$(date +%s)
  PYTHONPATH="${PICO_ROOT}" "${PYTHON}" "${ANALYZER}" \
      --repo_id "${repo_id}" \
      --max_steps "${max_steps}" \
      --output_root "${PICO_ROOT}/analysis_results" \
      >>"${rep_log}" 2>&1
  local rc=$?
  local t1; t1=$(date +%s)

  if [[ ${rc} -eq 0 ]]; then
    log "  analyzer ok ($((t1 - t0))s) → plotting"
    local run_name
    run_name="$(basename "${repo_id}")"
    local out_dir="${PICO_ROOT}/analysis_results/${run_name}_pico_analysis"
    if [[ -d "${out_dir}" ]]; then
      "${PYTHON}" "${PLOTTER}" "${out_dir}" >>"${rep_log}" 2>&1 || \
        log "  plotter warning (${repo_id})"
    fi
    echo -e "${repo_id}\tOK\t$((t1 - t0))" >> "${SUMMARY}"
  else
    log "  analyzer FAILED rc=${rc} (see ${rep_log})"
    echo -e "${repo_id}\tFAIL_${rc}\t$((t1 - t0))" >> "${SUMMARY}"
  fi
}

log "queue size: ${#TARGETS[@]}"
for entry in "${TARGETS[@]}"; do
  IFS='|' read -r repo max_steps <<< "${entry}"
  run_one "${repo}" "${max_steps}"
done

END_TS=$(date +%s)
log "DONE total elapsed: $((END_TS - START_TS))s"
log "summary: ${SUMMARY}"
