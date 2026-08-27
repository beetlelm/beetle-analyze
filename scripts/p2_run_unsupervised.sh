#!/usr/bin/env bash
# p2_run_unsupervised.sh
#
# Drives the full nld-eng pico sweep then re-runs the higher-order
# pico↔downstream analysis. Designed to run unsupervised under nohup.
#
# Stages
#   1) p2_pico_sweep_nld_eng.sh   — fills FW-2B / FW-24B / HumanScale × 5 curricula
#   2) p2_higher_order_analysis.py — rebuilds wide CSV, predictiveness, figures
#
# Logs all stages to logs/p2_run/.
set -uo pipefail

REPO="/Users/suchirsalhan/Documents/PHD/PHD/beetle-analyze"
cd "${REPO}"
LOG_DIR="${REPO}/logs/p2_run"
mkdir -p "${LOG_DIR}"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(stamp)] STAGE 1: pico sweep" | tee -a "${LOG_DIR}/driver.log"
bash "${REPO}/scripts/p2_pico_sweep_nld_eng.sh" \
    >>"${LOG_DIR}/sweep_stdout.log" 2>>"${LOG_DIR}/sweep_stderr.log"
sweep_rc=$?
echo "[$(stamp)] STAGE 1 done rc=${sweep_rc}" | tee -a "${LOG_DIR}/driver.log"

echo "[$(stamp)] STAGE 2: higher-order analysis" | tee -a "${LOG_DIR}/driver.log"
/usr/bin/python3 -u "${REPO}/scripts/p2_higher_order_analysis.py" \
    >>"${LOG_DIR}/analysis.log" 2>&1
analysis_rc=$?
echo "[$(stamp)] STAGE 2 done rc=${analysis_rc}" | tee -a "${LOG_DIR}/driver.log"

echo "[$(stamp)] all done sweep=${sweep_rc} analysis=${analysis_rc}" | tee -a "${LOG_DIR}/driver.log"
exit $(( sweep_rc != 0 ? sweep_rc : analysis_rc ))
