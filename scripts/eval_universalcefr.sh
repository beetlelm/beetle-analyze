#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# eval_universalcefr.sh
#
# Run the UniversalCEFR proficiency-level classification eval on a cohort of
# models via evals/eval_per_model.py. Skips BLiMP/BLiSS/MECO so the model is
# loaded once and only UniversalCEFR runs.
#
# Usage
#   bash scripts/eval_universalcefr.sh [MODELS] [LANGS]
#
# Args (env-overridable):
#   MODELS      Model spec(s) for resolve_models. Default: "frontier".
#               Examples: "beetle-humanscale" "Beetle-HumanScale/...-deu-eng"
#   LANGS       Space-separated ISO-639-1 codes for default-corpus selection.
#               Default: "en de nl" (matches Beetle's L1/L2 coverage).
#   MODE        surprisal | generation. Default: surprisal (use generation
#               for SFT'd / instruction-tuned variants).
#   TASK        read | write. Default: read.
#   LIMIT       Cap samples per corpus (smoke testing). Default: empty (all).
#   N_GPUS      Number of ranks. Default: 1.
#   OUT         Output root. Default: results.
#   REVISION    HF branch / revision. Default: main.
#   ADAPTER     Optional path or HF id of a LoRA adapter to attach. Default: none.
#               (Only honoured when MODELS resolves to a single repo via
#               evals/eval_universalcefr.py invoked directly.)
#
# Examples
#   # Smoke test on one Beetle model, one corpus, surprisal mode:
#   MODELS="Beetle-HumanScale/beetle-bilingual-...-deu-eng" \
#   LIMIT=20 bash scripts/eval_universalcefr.sh
#
#   # Full eval, all default en/de/nl corpora, surprisal mode, 4 ranks:
#   N_GPUS=4 MODELS=beetle-humanscale bash scripts/eval_universalcefr.sh
#
#   # Generation mode (for SFT'd Beetle variants):
#   MODE=generation bash scripts/eval_universalcefr.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."

MODELS="${MODELS:-frontier}"
LANGS="${LANGS:-en de nl}"
MODE="${MODE:-surprisal}"
TASK="${TASK:-read}"
LIMIT="${LIMIT:-}"
N_GPUS="${N_GPUS:-1}"
OUT="${OUT:-results}"
REVISION="${REVISION:-main}"
ADAPTER="${ADAPTER:-}"

extra=()
[[ -n "$LIMIT" ]] && extra+=(--universalcefr-limit "$LIMIT")

# If a single concrete repo + adapter is requested, use the standalone CLI so
# the LoRA adapter can be attached. Otherwise fan out via eval_per_model.
if [[ -n "$ADAPTER" ]]; then
    if [[ "$N_GPUS" -ne 1 || "$MODELS" =~ ' ' ]]; then
        echo "ERROR: ADAPTER mode supports only a single repo + N_GPUS=1." >&2
        exit 2
    fi
    echo "[universalcefr] standalone mode (adapter=$ADAPTER) for $MODELS"
    python -m evals.eval_universalcefr \
        --model "$MODELS" --revision "$REVISION" \
        --adapter "$ADAPTER" \
        --languages $LANGS \
        --mode "$MODE" --task "$TASK" \
        --output "$OUT/evals/universalcefr" \
        ${LIMIT:+--limit "$LIMIT"}
    exit 0
fi

echo "[universalcefr] models=$MODELS langs='$LANGS' mode=$MODE task=$TASK n_gpus=$N_GPUS"

pids=()
for ((rank=0; rank<N_GPUS; rank++)); do
    CUDA_VISIBLE_DEVICES="$rank" python evals/eval_per_model.py \
        --models $MODELS \
        --revision "$REVISION" \
        --rank "$rank" --world-size "$N_GPUS" \
        --output-root "$OUT" \
        --skip-blimp --skip-bliss --skip-meco \
        --universalcefr-mode "$MODE" \
        --universalcefr-task "$TASK" \
        --universalcefr-languages $LANGS \
        "${extra[@]}" &
    pids+=($!)
done

# Wait for all ranks; propagate first non-zero exit.
fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=$?
done

# Merge ledger shards once everyone is done
python -m analyze.eval_ledger merge --output "$OUT"
python -m analyze.eval_ledger report --output "$OUT" || true

exit $fail
