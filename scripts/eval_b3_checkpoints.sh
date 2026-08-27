#!/usr/bin/env bash
# =============================================================================
# scripts/eval_b3_checkpoints.sh
#
# Re-evaluate the b3_manipulations regimes (nld-eng B3 2B FineWeb) across a
# selected sweep of intermediate checkpoints, on MECO surprisal and BLiSS
# (ALTACambridge/BLiSS). The downstream regression of MECO surprisal onto
# human reading times (ΔlogLik) is then run by
#   python -m analyze.meco_canonical --backend statsmodels
# which reads every per-(model, branch) surprisal CSV produced here.
#
# Output layout
#   results/evals_b3_checkpoints/
#     meco/ surprisals_<repo>__step-<N>.csv        (per ckpt)
#     bliss/ bliss_overall_<repo>__step-<N>.csv    (overall BLiSS metrics)
#            bliss_perlang_<repo>__step-<N>.csv    (per-language BLiSS metrics)
#     logs/  rank<i>.log
#
# Default: 8 GPUs, 6 regimes × 9 checkpoints = 54 evals.
# arch-bgpt is excluded (custom architecture lacks model_type in config.json).
#
# Usage
#   bash scripts/eval_b3_checkpoints.sh                    # full sweep
#   N_GPUS=4 bash scripts/eval_b3_checkpoints.sh
#   STAGES="meco" bash scripts/eval_b3_checkpoints.sh      # only MECO
#   STAGES="bliss" bash scripts/eval_b3_checkpoints.sh
#   DRY_RUN=1 bash scripts/eval_b3_checkpoints.sh
#   REGIMES="tilt sas" bash scripts/eval_b3_checkpoints.sh
#   CKPTS="step-2048 step-5862" bash scripts/eval_b3_checkpoints.sh
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

N_GPUS="${N_GPUS:-8}"
STAGES="${STAGES:-meco bliss blimp}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BLIMP_BATCH_SIZE="${BLIMP_BATCH_SIZE:-64}"
OUT_ROOT="${OUT_ROOT:-results/evals_b3_checkpoints}"
LOG_DIR="${OUT_ROOT}/logs"
DRY_RUN="${DRY_RUN:-0}"
PY="${PYTHON:-python3}"

# Regimes (manipulation slugs). arch-bgpt skipped — custom arch.
REGIMES_DEFAULT="pre-pretrain alibi tilt layer-growth sas arch-moe"
REGIMES="${REGIMES:-$REGIMES_DEFAULT}"

# Checkpoints to evaluate. Selected to span:
#   - init (step-0)
#   - early L1 (step-512)
#   - mid L1 (step-2048)
#   - L2-onset window (step-3814 = first L2 step; step-3878, step-4070)
#   - post-L2 trajectory (step-4326, step-4838)
#   - final (step-5862)
CKPTS_DEFAULT="step-0 step-512 step-2048 step-3814 step-3878 step-4070 step-4326 step-4838 step-5862"
CKPTS="${CKPTS:-$CKPTS_DEFAULT}"

HF_PREFIX="Beetle-nld-eng-Variants/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-2b-nld-eng"

mkdir -p "$OUT_ROOT"/{meco,bliss,blimp,logs}

# Activate venv with minicons + datasets + transformers.
if [[ -f /root/beetle-analyze/venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source /root/beetle-analyze/venv/bin/activate
fi

ts() { date -u +%FT%TZ; }
log() { echo "[$(ts)] $*"; }

# ------ Build plan ------
PLAN_FILE="$OUT_ROOT/plan.tsv"
: > "$PLAN_FILE"
n=0
for regime in $REGIMES; do
    repo="${HF_PREFIX}-${regime}"
    for ckpt in $CKPTS; do
        printf '%s\t%s\t%s\n' "$regime" "$repo" "$ckpt" >> "$PLAN_FILE"
        n=$((n + 1))
    done
done
log "plan: $n (regime, ckpt) pairs across $N_GPUS GPUs"
if [[ "$DRY_RUN" == "1" ]]; then
    column -t -s$'\t' "$PLAN_FILE"
    exit 0
fi

# ------ Per-rank worker ------
# Reads $PLAN_FILE; processes rows where (line_idx % N_GPUS) == RANK.
run_rank() {
    local rank=$1
    local rank_log="$LOG_DIR/rank${rank}.log"
    local idx=0
    while IFS=$'\t' read -r regime repo ckpt; do
        if (( idx % N_GPUS == rank )); then
            (
                export CUDA_VISIBLE_DEVICES="$rank"
                export TOKENIZERS_PARALLELISM=false
                local tag="${regime}@${ckpt}"
                echo "[$(ts)] rank${rank} START ${tag}" >> "$rank_log"

                # ---------- MECO surprisal ----------
                if [[ " $STAGES " == *" meco "* ]]; then
                    out_csv="$OUT_ROOT/meco/surprisals_${repo//\//_}__${ckpt}.csv"
                    if [[ -s "$out_csv" ]]; then
                        echo "[$(ts)] rank${rank} SKIP meco ${tag} (have $out_csv)" >> "$rank_log"
                    else
                        $PY evals/eval_meco.py \
                            --models "$repo" \
                            --revision "$ckpt" \
                            --output "$OUT_ROOT/meco" \
                            >> "$rank_log" 2>&1
                        echo "[$(ts)] rank${rank} DONE meco ${tag} rc=$?" >> "$rank_log"
                    fi
                fi

                # ---------- BLiMP (English + Dutch) ----------
                # eval_blimp.py appends rows to results/<output>/blimp_results.csv
                # keyed on (model, branch, benchmark, lang); the eval ledger
                # short-circuits already-completed (repo,benchmark,lang,revision).
                if [[ " $STAGES " == *" blimp "* ]]; then
                    blimp_marker="$OUT_ROOT/blimp/.done_${repo//\//_}__${ckpt}"
                    if [[ -f "$blimp_marker" ]]; then
                        echo "[$(ts)] rank${rank} SKIP blimp ${tag} (have $blimp_marker)" >> "$rank_log"
                    else
                        $PY evals/eval_blimp.py \
                            --models "$repo" \
                            --revision "$ckpt" \
                            --benchmarks blimp_eng blimp_nl \
                            --langs eng nld \
                            --output "$OUT_ROOT/blimp" \
                            --batch-size "$BLIMP_BATCH_SIZE" \
                            --rank 0 --world-size 1 \
                            >> "$rank_log" 2>&1
                        rc=$?
                        echo "[$(ts)] rank${rank} DONE blimp ${tag} rc=$rc" >> "$rank_log"
                        [[ $rc -eq 0 ]] && touch "$blimp_marker"
                    fi
                fi

                # ---------- BLiSS ----------
                if [[ " $STAGES " == *" bliss "* ]]; then
                    overall_csv="$OUT_ROOT/bliss/bliss_overall_${repo//\//_}__${ckpt}.csv"
                    if [[ -s "$overall_csv" ]]; then
                        echo "[$(ts)] rank${rank} SKIP bliss ${tag} (have $overall_csv)" >> "$rank_log"
                    else
                        $PY evals/eval_bliss.py \
                            --model "$repo" \
                            --revision "$ckpt" \
                            --split train \
                            --batch-size "$BATCH_SIZE" \
                            --langs en nl \
                            --output "$OUT_ROOT/bliss" \
                            >> "$rank_log" 2>&1
                        # eval_bliss writes results/evals/bliss/bliss_*.csv with --output;
                        # rename to namespace by ckpt so reruns don't clobber.
                        for src in "$OUT_ROOT/bliss/bliss_overall_${repo//\//_}.csv" \
                                   "$OUT_ROOT/bliss/bliss_perlang_${repo//\//_}.csv"; do
                            [[ -f "$src" ]] || continue
                            dst="${src%.csv}__${ckpt}.csv"
                            mv -f "$src" "$dst"
                        done
                        echo "[$(ts)] rank${rank} DONE bliss ${tag} rc=$?" >> "$rank_log"
                    fi
                fi

                echo "[$(ts)] rank${rank} END ${tag}" >> "$rank_log"
            )
        fi
        idx=$((idx + 1))
    done < "$PLAN_FILE"
    echo "[$(ts)] rank${rank} ALL DONE" >> "$rank_log"
}

# ------ Fan out ------
log "launching $N_GPUS ranks; logs in $LOG_DIR"
pids=()
for ((r = 0; r < N_GPUS; r++)); do
    run_rank "$r" &
    pids+=("$!")
done
rc=0
for p in "${pids[@]}"; do
    wait "$p" || rc=$?
done

log "all ranks finished rc=$rc"
log "outputs:"
log "  meco surprisal CSVs : $OUT_ROOT/meco/"
log "  bliss CSVs          : $OUT_ROOT/bliss/"
log "  logs                : $LOG_DIR/"
log "next step:"
log "  python -m analyze.meco_canonical --backend statsmodels \\"
log "    --cache-path $OUT_ROOT/meco_canonical_deltalogl.csv"
exit $rc
