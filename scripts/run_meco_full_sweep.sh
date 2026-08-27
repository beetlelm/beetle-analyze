#!/bin/bash
# =============================================================================
# BeetleLM Full MECO Sweep Pipeline
#
# End-to-end: generate configs → train all conditions → compute surprisal
#
# This script runs ALL bilingual (B1–B5) and trilingual (T1–T4) conditions
# with combinatorial EWC × MAML × LR sweeps for MECO language pairs.
#
# Usage:
#   # Phase 1: Generate all sweep configs (run once, locally)
#   ./script/run_meco_full_sweep.sh generate
#
#   # Phase 2: Train (run on each GPU node)
#   ./script/run_meco_full_sweep.sh train <NODE_ID> [NUM_NODES] [--bilingual|--trilingual|--all]
#
#   # Phase 3: Compute surprisal (run after training completes)
#   ./script/run_meco_full_sweep.sh eval [--conditions B1 B2 ...]
#
#   # All-in-one on a single node (generate + train + eval)
#   ./script/run_meco_full_sweep.sh all <NODE_ID> [NUM_NODES]
#
# Prerequisites:
#   - Python environment with beetlelm installed
#   - MECO stimuli at ../beetlelm-rts-private/stimuli/meco_l2_stims.tsv
#
# Parameter Grids (configured below):
#   EWC lambda:       0, 10, 20, 30, 40
#   MAML ratio:       0, 0.2, 0.3, 0.4
#   LR boost:         1.0, 1.5, 2.0, 2.5, 3.0
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG_DIR="configs/conditions"
RUNS_DIR="./runs"
LOG_DIR="logs_meco_full_sweep"
STIMULI="../beetlelm-rts-private/stimuli/meco_l2_stims.tsv"
RESULTS_DIR="../beetlelm-rts-private/stats"

GPUS_PER_NODE=8
PARALLEL_JOBS=2
GPUS_PER_JOB=$(( GPUS_PER_NODE / PARALLEL_JOBS ))

# ── Sweep grids ─────────────────────────────────────────────────────────────
EWC_GRID="0 10 20 30 40"
MAML_GRID="0 0.2 0.3 0.4"
LR_GRID="1.0 1.5 2.0 2.5 3.0"

# ── Helper functions ────────────────────────────────────────────────────────

build_gpu_slots() {
    GPU_SLOTS=()
    for (( slot=0; slot<PARALLEL_JOBS; slot++ )); do
        start=$(( slot * GPUS_PER_JOB ))
        end=$(( start + GPUS_PER_JOB - 1 ))
        GPU_SLOTS+=( "$(seq -s, "$start" "$end")" )
    done
}

run_config() {
    local cfg="$1"
    local slot="$2"
    local node_id="$3"
    local gpus="${GPU_SLOTS[$slot]}"
    local run_name log_file

    run_name=$(basename "$cfg" .yaml)
    log_file="${LOG_DIR}/${run_name}_node${node_id}.log"

    echo "[N${node_id} slot${slot} GPUs:${gpus}] START ${run_name}"
    CUDA_VISIBLE_DEVICES="$gpus" python script/train.py \
        --config_path "$cfg" >> "$log_file" 2>&1

    if [[ $? -eq 0 ]]; then
        echo "[N${node_id} slot${slot}] DONE  ${run_name}"
    else
        echo "[N${node_id} slot${slot}] FAIL  ${run_name} -> ${log_file}"
        return 1
    fi
}

schedule_configs() {
    local node_id="$1"
    local num_nodes="$2"
    shift 2
    local configs=("$@")

    local total=${#configs[@]}
    echo "Node $node_id: scheduling ${total} configs across ${num_nodes} nodes"

    # Assign configs to this node (interleaved split)
    local node_configs=()
    for (( i=0; i<total; i++ )); do
        if (( i % num_nodes == node_id - 1 )); then
            node_configs+=("${configs[$i]}")
        fi
    done

    echo "Node $node_id assigned ${#node_configs[@]} configs"

    local idx=0
    local n_cfgs=${#node_configs[@]}

    while (( idx < n_cfgs )); do
        local pids=()
        for (( slot=0; slot<PARALLEL_JOBS; slot++ )); do
            if (( idx < n_cfgs )); then
                run_config "${node_configs[$idx]}" "$slot" "$node_id" &
                pids+=($!)
                (( idx++ ))
            fi
        done
        for pid in "${pids[@]}"; do
            wait "$pid" || true
        done
    done
}

# ── Phase 1: Generate configs ──────────────────────────────────────────────

cmd_generate() {
    echo "=== Generating MECO sweep configs ==="
    python script/beetle_experiment.py sweep \
        --meco \
        --conditions B1 B2 B3 B4 B5 T1 T2 T3 T4 \
        --ewc $EWC_GRID \
        --maml $MAML_GRID \
        --lr-boost $LR_GRID \
        --output "$CONFIG_DIR"
    echo ""
    echo "=== Config generation complete ==="
    echo ""

    # Count configs
    local n_bi n_tri
    n_bi=$(find "$CONFIG_DIR/bilingual" -name "*.yaml" 2>/dev/null | wc -l | tr -d ' ')
    n_tri=$(find "$CONFIG_DIR/trilingual" -name "*.yaml" 2>/dev/null | wc -l | tr -d ' ')
    echo "Bilingual configs:  $n_bi"
    echo "Trilingual configs: $n_tri"
    echo "Total:              $(( n_bi + n_tri ))"
}

# ── Phase 2: Train ─────────────────────────────────────────────────────────

cmd_train() {
    local node_id=${1:?ERROR: provide NODE_ID}
    local num_nodes=${2:-4}
    local scope=${3:---all}

    mkdir -p "$LOG_DIR"
    build_gpu_slots

    # Collect configs based on scope
    local all_configs=()
    case "$scope" in
        --bilingual)
            while IFS= read -r -d '' f; do
                all_configs+=("$f")
            done < <(find "$CONFIG_DIR/bilingual" -name "*.yaml" -print0 | sort -z)
            ;;
        --trilingual)
            while IFS= read -r -d '' f; do
                all_configs+=("$f")
            done < <(find "$CONFIG_DIR/trilingual" -name "*.yaml" -print0 | sort -z)
            ;;
        --all|*)
            while IFS= read -r -d '' f; do
                all_configs+=("$f")
            done < <(find "$CONFIG_DIR" -name "*.yaml" -print0 | sort -z)
            ;;
    esac

    echo "=== MECO sweep training ==="
    echo "Scope: $scope"
    echo "Total configs: ${#all_configs[@]}"
    echo "Node: $node_id / $num_nodes"
    echo "GPU slots: ${GPU_SLOTS[*]}"
    echo ""

    schedule_configs "$node_id" "$num_nodes" "${all_configs[@]}"

    echo ""
    echo "=== Node $node_id: training complete ==="
}

# ── Phase 3: Evaluate ──────────────────────────────────────────────────────

cmd_eval() {
    local conditions="${@:-B1 B2 B3 B4 B5 T1 T2 T3 T4}"

    echo "=== Computing MECO surprisal ==="

    if [[ ! -f "$STIMULI" ]]; then
        echo "ERROR: MECO stimuli not found at: $STIMULI"
        echo "Expected: ../beetlelm-rts-private/stimuli/meco_l2_stims.tsv"
        exit 1
    fi

    mkdir -p "$RESULTS_DIR"

    python script/compute_meco_surprisal.py \
        --runs-dir "$RUNS_DIR" \
        --stimuli "$STIMULI" \
        --output "$RESULTS_DIR/results_conditions.tsv" \
        --conditions $conditions

    echo ""
    echo "=== Surprisal computation complete ==="
    echo "Results: $RESULTS_DIR/results_conditions.tsv"
    echo ""
    echo "Next: Rscript ../beetlelm-rts-private/stats/meco_l2.Rmd"
}

# ── All-in-one ──────────────────────────────────────────────────────────────

cmd_all() {
    local node_id=${1:?ERROR: provide NODE_ID}
    local num_nodes=${2:-4}

    cmd_generate
    echo ""
    cmd_train "$node_id" "$num_nodes" "--all"
    echo ""
    cmd_eval
}

# ── Main ────────────────────────────────────────────────────────────────────

COMMAND=${1:-help}
shift || true

case "$COMMAND" in
    generate)
        cmd_generate
        ;;
    train)
        cmd_train "$@"
        ;;
    eval)
        cmd_eval "$@"
        ;;
    all)
        cmd_all "$@"
        ;;
    help|--help|-h)
        echo "Usage: $0 {generate|train|eval|all} [args]"
        echo ""
        echo "Commands:"
        echo "  generate              Generate all sweep YAML configs"
        echo "  train NODE NUM_NODES  Train configs on this node"
        echo "  eval [CONDITIONS]     Compute MECO surprisal"
        echo "  all NODE [NUM_NODES]  Full pipeline (generate + train + eval)"
        echo ""
        echo "Examples:"
        echo "  $0 generate"
        echo "  $0 train 1 4 --bilingual"
        echo "  $0 train 2 4 --trilingual"
        echo "  $0 eval B1 B2 B3"
        echo "  $0 all 1 1"
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Run '$0 help' for usage"
        exit 1
        ;;
esac
