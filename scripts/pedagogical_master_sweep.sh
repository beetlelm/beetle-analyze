#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# pedagogical_master_sweep.sh
#
# Runs §4.1 Cloze + §4.2 Interlanguage + §4.3 EVP trajectory + §4.4 CEFR PPL
# on the b1–b5 × FineWeb-2B + HumanScale grid. Each stage is shard-parallel
# across N_GPUS via _sweep_safety.sh's launch_ranks contract. Stages run
# sequentially so we never overlap GPU jobs. Logs all four stages under
# /root/beetle_pipeline/logs/manual_pedagogical/.
#
# Stage gates (env vars; 0 to skip):
#   STAGE_44=1       CEFR PPL gradient   ~10 min for 11 models
#   STAGE_41=1       Cloze (closed+cam)  ~2 hours
#   STAGE_42=1       Interlanguage       ~half day
#   STAGE_43=1       EVP trajectory      ~half day across ~14 ckpts × 5 repos
#   STAGE_PLOTS=1    Plot generation     after each eval that's been run
#
# Pinned model lists are computed from the canonical b1..b5 sweep.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate
source scripts/_sweep_safety.sh
sweep_safety_init

LOG_ROOT="${LOG_ROOT:-/root/beetle_pipeline/logs/manual_pedagogical}"
mkdir -p "$LOG_ROOT"
ORCH_LOG="$LOG_ROOT/orchestrator.log"

ts() { date -u +%FT%TZ; }
log() { echo "[$(ts)] $*" | tee -a "$ORCH_LOG"; }

# ─── Stage gates ────────────────────────────────────────────────────────────
STAGE_44="${STAGE_44:-1}"
STAGE_41="${STAGE_41:-1}"
STAGE_42="${STAGE_42:-1}"
STAGE_43="${STAGE_43:-1}"
STAGE_PLOTS="${STAGE_PLOTS:-1}"

N_GPUS="${N_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"

OUT_ROOT="${OUT_ROOT:-results/evals_paper}"
OUT_44="$OUT_ROOT/04d_cefr_ppl"
OUT_41="$OUT_ROOT/04a_cloze"
OUT_42="$OUT_ROOT/04b_interlanguage"
OUT_43="$OUT_ROOT/04c_evp_trajectory"
mkdir -p "$OUT_44" "$OUT_41" "$OUT_42" "$OUT_43"

# ─── Canonical model lists ──────────────────────────────────────────────────
# FineWeb-2B nld-eng, b1..b5 (the §4.x grid).  Use main + step-5862 (the
# canonical late-step used in the existing Stage 18 eval).  Adding step-5862
# alongside main doubles model loads but lets us see the trajectory between
# the late-training and final-checkpoint behaviour.
FW2B_NLD_ENG=(
    "Beetle-FineWeb-2B/beetle-bilingual-balanced-b1-fineweb-2b-nld-eng:main"
    "Beetle-FineWeb-2B/beetle-bilingual-l2-50-simultaneous-b2-fineweb-2b-nld-eng:main"
    "Beetle-FineWeb-2B/beetle-bilingual-l2-50-sequential-33-67-b3-fineweb-2b-nld-eng:main"
    "Beetle-FineWeb-2B/beetle-bilingual-l2-50-classroom-20-b4-fineweb-2b-nld-eng:main"
    "Beetle-FineWeb-2B/beetle-bilingual-l2-80-late-b5-fineweb-2b-nld-eng:main"
)
HS_NLD_ENG=(
    "Beetle-HumanScale/beetle-bilingual-balanced-b1-humanscale-nld-eng:main"
    "Beetle-HumanScale/beetle-bilingual-l2-50-simultaneous-b2-humanscale-nld-eng:main"
    "Beetle-HumanScale/beetle-bilingual-l2-50-sequential-33-67-b3-humanscale-nld-eng:main"
    "Beetle-HumanScale/beetle-bilingual-l2-50-classroom-20-b4-humanscale-nld-eng:main"
    "Beetle-HumanScale/beetle-bilingual-l2-80-late-b5-humanscale-nld-eng:main"
)
MONO=(
    "Beetle-HumanScale/beetle-monolingual-humanscale-eng:main"
    "Beetle-HumanScale/beetle-monolingual-humanscale-nld:main"
)
# §4.2 needs a few non-Dutch L1 anchors so the matched-bilingual claim has a
# foil. Pull HumanScale zho-eng + deu-eng b1..b5 where they exist.
HS_ZHO_ENG=(
    "Beetle-HumanScale/beetle-bilingual-balanced-b1-humanscale-zho-eng:main"
    "Beetle-HumanScale/beetle-bilingual-l2-50-simultaneous-b2-humanscale-zho-eng:main"
    "Beetle-HumanScale/beetle-bilingual-l2-50-sequential-33-67-b3-humanscale-zho-eng:main"
    "Beetle-HumanScale/beetle-bilingual-l2-50-classroom-20-b4-humanscale-zho-eng:main"
    "Beetle-HumanScale/beetle-bilingual-l2-80-late-b5-humanscale-zho-eng:main"
)

CORE_GRID=("${FW2B_NLD_ENG[@]}" "${HS_NLD_ENG[@]}" "${MONO[@]}")
INTERLANG_GRID=("${FW2B_NLD_ENG[@]}" "${HS_NLD_ENG[@]}" "${HS_ZHO_ENG[@]}" "${MONO[@]}")

log "starting pedagogical_master_sweep"
log "  STAGE_44=$STAGE_44 STAGE_41=$STAGE_41 STAGE_42=$STAGE_42 STAGE_43=$STAGE_43 STAGE_PLOTS=$STAGE_PLOTS"
log "  N_GPUS=$N_GPUS  CORE_GRID=${#CORE_GRID[@]} models  INTERLANG_GRID=${#INTERLANG_GRID[@]}"

run_shard() {
    # run_shard <stage_name> <out_dir> <python_module> <pair_array...>
    local stage="$1"; shift
    local out_dir="$1"; shift
    local module="$1"; shift
    local pairs=("$@")
    local logdir="$LOG_ROOT/$stage"
    mkdir -p "$logdir"
    log "$stage: ${#pairs[@]} pairs across $N_GPUS ranks → $logdir"
    local pids=()
    for r in $(seq 0 $((N_GPUS - 1))); do
        local lf="$logdir/rank_${r}.log"
        CUDA_VISIBLE_DEVICES="$r" RANK="$r" WORLD_SIZE="$N_GPUS" \
        setsid python -m "$module" \
            --models "${pairs[@]}" \
            --output-root "$out_dir" \
            --batch-size "$BATCH_SIZE" \
            > "$lf" 2>&1 &
        pids+=($!)
    done
    local rc=0
    for pid in "${pids[@]}"; do
        wait "$pid" || rc=$?
    done
    log "$stage: ranks done (rc=$rc)"
    return 0
}

# ─── Stage 4.4 — CEFR perplexity gradient ───────────────────────────────────
if [[ "$STAGE_44" == "1" ]]; then
    log "=== Stage §4.4: CEFR perplexity gradient ==="
    run_shard "stage44_cefr_ppl" "$OUT_44" \
        "evals.eval_cefr_perplexity" "${CORE_GRID[@]}"
fi

# ─── Stage 4.1 — Cloze (closed + Cambridge derivative) ──────────────────────
if [[ "$STAGE_41" == "1" ]]; then
    log "=== Stage §4.1: Cloze ==="
    run_shard "stage41_cloze" "$OUT_41" \
        "evals.eval_cloze" "${CORE_GRID[@]}"
fi

# ─── Stage 4.2 — Interlanguage error scoring ────────────────────────────────
if [[ "$STAGE_42" == "1" ]]; then
    log "=== Stage §4.2: Interlanguage ==="
    run_shard "stage42_interlanguage" "$OUT_42" \
        "evals.eval_interlanguage" "${INTERLANG_GRID[@]}"
fi

# ─── Stage 4.3 — EVP-style trajectory ────────────────────────────────────────
# This is repo-sharded (not pair-sharded) because per-repo we iterate over
# ~14 step-N branches.
if [[ "$STAGE_43" == "1" ]]; then
    log "=== Stage §4.3: EVP trajectory ==="
    EVP_REPOS=()
    for entry in "${FW2B_NLD_ENG[@]}" "${HS_NLD_ENG[@]}"; do
        EVP_REPOS+=("${entry%:*}")
    done
    log "  trajectory repos: ${#EVP_REPOS[@]}"
    logdir="$LOG_ROOT/stage43_evp"
    mkdir -p "$logdir"
    pids=()
    for r in $(seq 0 $((N_GPUS - 1))); do
        lf="$logdir/rank_${r}.log"
        CUDA_VISIBLE_DEVICES="$r" RANK="$r" WORLD_SIZE="$N_GPUS" \
        setsid python -m evals.eval_evp_trajectory \
            --repos "${EVP_REPOS[@]}" \
            --output-root "$OUT_43" \
            --batch-size 64 \
            --max-checkpoints 14 \
            --per-level 200 \
            > "$lf" 2>&1 &
        pids+=($!)
    done
    rc=0
    for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
    log "stage43 ranks done (rc=$rc)"
fi

# ─── Stage PLOTS — render figures ───────────────────────────────────────────
if [[ "$STAGE_PLOTS" == "1" ]]; then
    log "=== Stage PLOTS ==="
    python -m plots.make_pedagogical_panels \
        --cefr-ppl-root  "$OUT_44" \
        --cloze-root     "$OUT_41" \
        --interlang-root "$OUT_42" \
        --evp-root       "$OUT_43" \
        --output-dir     plots \
        > "$LOG_ROOT/plots.log" 2>&1 || \
        log "plots failed; see $LOG_ROOT/plots.log"
fi

log "DONE pedagogical_master_sweep"
