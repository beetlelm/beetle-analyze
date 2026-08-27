#!/usr/bin/env bash
# run_beetle_analyze.sh — unified orchestrator for the beetle-analyze pipeline.
#
# Drives analyze/run.py across multiple stages, with responsible HF cache
# management and per-stage logging. Figures/paper_figures always run last.
#
# Example:
#     bash scripts/run_beetle_analyze.sh \
#         --family humanscale \
#         --langs "deu nld zho eng" \
#         --stages "ppl forgetting embeddings convergence meco aoa \
#                   cognates cli learning_dynamics representations \
#                   dynamics novel figures paper_figures" \
#         --cache-mode rotate

set -euo pipefail

# ─── defaults ───────────────────────────────────────────────────────────────
FAMILY="all"
LANGS="deu nld zho eng"
CONDITIONS=""
VARIANTS=""
STAGES="ppl forgetting embeddings convergence meco aoa cognates cli \
        learning_dynamics representations dynamics novel figures paper_figures"
CACHE_MODE="rotate"
CACHE_BUDGET_GB=200
OUTPUT_ROOT="results"
HF_TOKEN="${HF_TOKEN:-}"
COMPARE=""

# ─── arg parser ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --family)          FAMILY="$2"; shift 2;;
        --langs)           LANGS="$2"; shift 2;;
        --conditions)      CONDITIONS="$2"; shift 2;;
        --variants)        VARIANTS="$2"; shift 2;;
        --stages)          STAGES="$2"; shift 2;;
        --cache-mode)      CACHE_MODE="$2"; shift 2;;
        --cache-budget-gb) CACHE_BUDGET_GB="$2"; shift 2;;
        --output-root)     OUTPUT_ROOT="$2"; shift 2;;
        --hf-token)        HF_TOKEN="$2"; shift 2;;
        --compare)         COMPARE="$2"; shift 2;;
        -h|--help)
            sed -n '2,17p' "$0"
            exit 0;;
        *)
            echo "Unknown flag: $1" >&2; exit 2;;
    esac
done

mkdir -p "${OUTPUT_ROOT}/logs"
export BEETLE_HF_CACHE="${BEETLE_HF_CACHE:-${OUTPUT_ROOT}/.hf_cache}"
export BEETLE_CACHE_BUDGET_GB="${CACHE_BUDGET_GB}"
mkdir -p "${BEETLE_HF_CACHE}"

echo "────────────────────────────────────────────────────────────────"
echo "beetle-analyze orchestrator"
echo "  family       : ${FAMILY}"
echo "  langs        : ${LANGS}"
echo "  conditions   : ${CONDITIONS:-all}"
echo "  variants     : ${VARIANTS:-all}"
echo "  stages       : ${STAGES}"
echo "  cache mode   : ${CACHE_MODE} (budget ${CACHE_BUDGET_GB} GB)"
echo "  output root  : ${OUTPUT_ROOT}"
echo "  hf cache     : ${BEETLE_HF_CACHE}"
echo "────────────────────────────────────────────────────────────────"

# Build the common flag list fed to analyze/run.py.
COMMON=(--family "${FAMILY}" --langs ${LANGS} \
        --cache-mode "${CACHE_MODE}" --output-root "${OUTPUT_ROOT}")
[[ -n "${CONDITIONS}" ]] && COMMON+=(--condition ${CONDITIONS})
[[ -n "${VARIANTS}"   ]] && COMMON+=(--variant ${VARIANTS})
[[ -n "${COMPARE}"    ]] && COMMON+=(--compare "${COMPARE}")
[[ -n "${HF_TOKEN}"   ]] && COMMON+=(--hf_token "${HF_TOKEN}")

# Figures and paper_figures always execute LAST.
STAGES_ORDERED=""
FIGURE_STAGES=""
for s in ${STAGES}; do
    case "${s}" in
        figures|paper_figures|paper-figures)
            FIGURE_STAGES+=" ${s}";;
        *)
            STAGES_ORDERED+=" ${s}";;
    esac
done
STAGES_ORDERED+="${FIGURE_STAGES}"

# Map stage names → run.py flags.
stage_flag() {
    case "$1" in
        ppl)                echo --ppl;;
        forgetting)         echo --forgetting;;
        embeddings)         echo --embeddings;;
        convergence)        echo --convergence;;
        meco)               echo --meco;;
        meco_sweep)         echo --meco-sweep;;
        aoa)                echo --aoa;;
        cognates)           echo --cognates;;
        cli)                echo --cli;;
        representations)    echo --representations;;
        dynamics)           echo --dynamics;;
        novel)              echo --novel;;
        learning_dynamics)  echo --learning-dynamics;;
        baselines)          echo --baselines;;
        figures)            echo --figures;;
        paper_figures|paper-figures) echo --paper-figures;;
        *) echo ""; return 1;;
    esac
}

for stage in ${STAGES_ORDERED}; do
    flag="$(stage_flag "${stage}" || true)"
    if [[ -z "${flag}" ]]; then
        echo "Skipping unknown stage '${stage}'" >&2
        continue
    fi
    log_file="${OUTPUT_ROOT}/logs/${stage}_$(date +%F).log"
    echo "────── stage: ${stage} ──────"
    echo "log: ${log_file}"
    if ! python3 analyze/run.py "${flag}" "${COMMON[@]}" 2>&1 | tee -a "${log_file}"; then
        echo "[orchestrator] stage ${stage} failed — aborting." >&2
        exit 1
    fi
    # Summarise HF cache after each data-producing stage.
    if command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli scan-cache 2>/dev/null | tail -n 2 >> "${log_file}" || true
    fi
done

echo "────────────────────────────────────────────────────────────────"
echo "orchestrator done — outputs under ${OUTPUT_ROOT}/"
