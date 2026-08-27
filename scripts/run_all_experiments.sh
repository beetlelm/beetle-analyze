#!/bin/bash
# =============================================================================
# run_all_experiments.sh — Complete Beetle-Analyze evaluation pipeline.
#
# Runs all evaluation and analysis stages:
#   1. Benchmark evaluation (8-GPU parallel)
#   2. PPL evaluation on FLORES
#   3. Catastrophic forgetting analysis
#   4. Embedding drift analysis
#   5. Convergence / checkpoint trajectory analysis
#   6. MECO eye-tracking surprisal evaluation
#   7. MECO checkpoint trajectory analysis
#   8. Generate all figures
#
# Usage:
#   # Full pipeline
#   bash scripts/run_all_experiments.sh
#
#   # Specific phase
#   bash scripts/run_all_experiments.sh --phase benchmarks
#   bash scripts/run_all_experiments.sh --phase analysis
#   bash scripts/run_all_experiments.sh --phase meco
#   bash scripts/run_all_experiments.sh --phase figures
#
#   # Specific languages
#   bash scripts/run_all_experiments.sh --langs deu nld zho
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

PHASE="all"
LANGS=""
WORLD_SIZE=8

while [[ $# -gt 0 ]]; do
    case $1 in
        --phase) PHASE="$2"; shift 2 ;;
        --langs) shift; LANGS=""; while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do LANGS="$LANGS $1"; shift; done ;;
        --world-size) WORLD_SIZE="$2"; shift 2 ;;
        *) shift ;;
    esac
done

LANG_ARGS=""
if [[ -n "$LANGS" ]]; then
    LANG_ARGS="--langs $LANGS"
fi

echo "============================================================"
echo "Beetle-Analyze: Run All Experiments"
echo "  Phase:      $PHASE"
echo "  Languages:  ${LANGS:-all}"
echo "  World size: $WORLD_SIZE"
echo "============================================================"

# ── Phase 1: Benchmark Evaluation ────────────────────────────────────────────
if [[ "$PHASE" == "all" || "$PHASE" == "benchmarks" ]]; then
    echo ""
    echo "── Phase 1: Benchmark evaluation (${WORLD_SIZE}-GPU parallel) ──"
    bash launch_eval.sh --world_size "$WORLD_SIZE" --latest_only
fi

# ── Phase 2: Analysis Pipeline ───────────────────────────────────────────────
if [[ "$PHASE" == "all" || "$PHASE" == "analysis" ]]; then
    echo ""
    echo "── Phase 2: PPL evaluation ──"
    python analyze/run.py --ppl $LANG_ARGS

    echo ""
    echo "── Phase 3: Catastrophic forgetting ──"
    python analyze/run.py --forgetting $LANG_ARGS

    echo ""
    echo "── Phase 4: Embedding drift ──"
    python analyze/run.py --embeddings $LANG_ARGS

    echo ""
    echo "── Phase 5: Convergence analysis ──"
    python analyze/run.py --convergence $LANG_ARGS
fi

# ── Phase 3: MECO Evaluation ────────────────────────────────────────────────
if [[ "$PHASE" == "all" || "$PHASE" == "meco" ]]; then
    echo ""
    echo "── Phase 6: MECO surprisal evaluation ──"
    python analyze/run.py --meco

    echo ""
    echo "── Phase 7: MECO checkpoint trajectories ──"
    python analyze/run.py --meco --meco-checkpoints
fi

# ── Phase 4: Generate Figures ────────────────────────────────────────────────
if [[ "$PHASE" == "all" || "$PHASE" == "figures" ]]; then
    echo ""
    echo "── Phase 8: Generating all figures ──"
    python analyze/run.py --figures $LANG_ARGS
fi

echo ""
echo "============================================================"
echo "Beetle-Analyze: All experiments complete."
echo "  Results: results/"
echo "  Figures: results/figures/"
echo "============================================================"
