#!/bin/bash
# =============================================================================
# BeetleLM MECO Fit Sweep
#
# Runs all bilingual conditions (B1–B5) with EWC/MAML/LR sweep variants
# for the MECO-relevant language pairs (nld→eng, deu→eng, zho→eng).
#
# After training, surprisal is computed against MECO stimuli and fit
# to reading time data using mixed-effects models.
#
# Usage:
#   ./run_meco_sweep.sh <NODE_ID> [NUM_NODES]
#
#   NODE_ID   = 1-based node index
#   NUM_NODES = total nodes (default: 4)
#
# Prerequisites:
#   1. Generate configs:  python configs/generate_conditions.py
#   2. Ensure MECO data:  ../beetlelm-rts-private/meco_l2/
# =============================================================================

set -euo pipefail

CONFIG_DIR="configs/conditions/bilingual"
LOG_DIR="logs_meco_sweep"
GPUS_PER_NODE=8
PARALLEL_JOBS=2
GPUS_PER_JOB=$(( GPUS_PER_NODE / PARALLEL_JOBS ))

NODE_ID=${1:?ERROR: provide NODE_ID}
NUM_NODES=${2:-4}

mkdir -p "$LOG_DIR"

# Only MECO-relevant pairs
MECO_PAIRS=("nld_L1-eng_L2" "deu_L1-eng_L2" "zho_L1-eng_L2")

GPU_SLOTS=()
for (( slot=0; slot<PARALLEL_JOBS; slot++ )); do
    start=$(( slot * GPUS_PER_JOB ))
    end=$(( start + GPUS_PER_JOB - 1 ))
    GPU_SLOTS+=( "$(seq -s, "$start" "$end")" )
done

# Discover MECO-relevant configs across all conditions
ALL_CONFIGS=()
for condition_dir in "$CONFIG_DIR"/B*; do
    if [[ ! -d "$condition_dir" ]]; then continue; fi
    for pair in "${MECO_PAIRS[@]}"; do
        while IFS= read -r -d '' f; do
            ALL_CONFIGS+=("$f")
        done < <(find "$condition_dir" -type f -name "*${pair}*" -print0 | sort -z)
    done
done

TOTAL=${#ALL_CONFIGS[@]}
echo "MECO sweep: $TOTAL configs across ${NUM_NODES} nodes"

# Assign to this node
NODE_CONFIGS=()
for (( i=0; i<TOTAL; i++ )); do
    if (( i % NUM_NODES == NODE_ID - 1 )); then
        NODE_CONFIGS+=("${ALL_CONFIGS[$i]}")
    fi
done

echo "Node $NODE_ID: ${#NODE_CONFIGS[@]} configs"

run_config() {
    local cfg="$1"
    local slot="$2"
    local gpus="${GPU_SLOTS[$slot]}"
    local run_name log_file

    run_name=$(basename "$cfg" .yaml)
    log_file="${LOG_DIR}/${run_name}_node${NODE_ID}.log"

    echo "[N${NODE_ID} slot${slot}] START ${run_name}"

    # Training
    CUDA_VISIBLE_DEVICES="$gpus" python script/train.py \
        --config_path "$cfg" >> "$log_file" 2>&1

    if [[ $? -eq 0 ]]; then
        echo "[N${NODE_ID} slot${slot}] TRAINED ${run_name}"
    else
        echo "[N${NODE_ID} slot${slot}] TRAIN_FAIL ${run_name} -> ${log_file}"
        return 1
    fi
}

echo "=== Node $NODE_ID: MECO sweep starting (${#NODE_CONFIGS[@]} configs) ==="
idx=0
total_cfgs=${#NODE_CONFIGS[@]}

while (( idx < total_cfgs )); do
    pids=()
    for (( slot=0; slot<PARALLEL_JOBS; slot++ )); do
        if (( idx < total_cfgs )); then
            run_config "${NODE_CONFIGS[$idx]}" "$slot" &
            pids+=($!)
            (( idx++ ))
        fi
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || true
    done
done

echo "=== Node $NODE_ID: MECO sweep complete ==="
echo ""
echo "Next steps:"
echo "  1. Compute surprisal:  python script/compute_meco_surprisal.py --all"
echo "  2. Fit MECO data:      Rscript ../beetlelm-rts-private/stats/meco_l2.Rmd"
echo ""
echo "For a more comprehensive sweep with all EWC/MAML/LR combinations:"
echo "  ./script/run_meco_full_sweep.sh generate"
echo "  ./script/run_meco_full_sweep.sh train <NODE_ID> [NUM_NODES]"
